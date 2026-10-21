# -*- coding: utf-8 -*-
"""
统一训练器 (trainer.py)
========================
所有铰链世界实验共享的训练循环骨架。
各实验脚本只需配置 model + criterion + dataset，调用 trainer.run()。

用法示例：
    from trainer import HingeTrainer, common_args
    parser = common_args()
    # 添加实验特有参数...
    args = parser.parse_args()
    trainer = HingeTrainer(args, model, criterion, train_ds, val_ds)
    trainer.run()
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from hinge_data import pos_rmse
from physics_prior import OBJ_DIMS

POS_IDX = [1, 2, 12, 13]


def common_args():
    """返回所有实验共享的 argparse 参数。"""
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_prototypes", type=int, default=32)
    p.add_argument("--w_state", type=float, default=1.0)
    p.add_argument("--w_yao", type=float, default=0.1)
    p.add_argument("--w_proto", type=float, default=0.05)
    p.add_argument("--w_commit", type=float, default=0.01)
    p.add_argument("--w_usage", type=float, default=0.02)
    p.add_argument("--w_roll", type=float, default=0.0)
    p.add_argument("--ss_max", type=float, default=0.5)
    p.add_argument("--ss_warmup", type=int, default=20,
                   help="scheduled sampling 线性升温 epoch 数")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    p.add_argument("--heldout_path", type=str,
                   default="experiments/trajectories_hinge_heldout.npz")
    return p


class HingeTrainer:
    """统一训练器：epoch 循环 + 验证 + 保存 + 日志。"""

    def __init__(self, args, model, criterion, train_ds, val_ds,
                 compute_loss_fn=None, validate_fn=None, log_tag=""):
        """
        compute_loss_fn(model, batch, h, c, fine, fine_next, feat_w, device)
            -> (loss, parts_dict)
            如果为 None，使用默认的 criterion(model, h, c, fine, a0, db, yao, fine_next, fw)

        validate_fn(model, val_ds, device, K, max_windows)
            -> metrics_dict
            如果为 None，使用默认的 validate()
        """
        self.args = args
        self.model = model
        self.criterion = criterion
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.compute_loss = compute_loss_fn or self._default_loss
        self.validate_fn = validate_fn or self._default_validate
        self.log_tag = log_tag or getattr(args, 'arm', '') or getattr(args, 'variant', '')

        self.device = torch.device(args.device)
        self.K = getattr(args, 'future_steps', 5)

        self.feat_w = torch.from_numpy(
            __import__("layout_h", fromlist=["feat_weights"]).feat_weights(OBJ_DIMS)
        ).to(self.device)

        self.use_amp = (self.device.type == "cuda") and not args.no_amp
        self.best = float("inf")
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    def _default_loss(self, model, batch, h, c, fine, fine_next, feat_w, device):
        xb = batch["x"].to(device)
        ab = batch["a0"].to(device)
        db = batch["d_norm"].to(device)
        yao = batch["yao"].to(device).float()
        loss, parts, _ = self.criterion(model, h, c, fine, ab, db, yao,
                                         fine_next, feat_w)
        return loss, parts

    @torch.no_grad()
    def _default_validate(self, model, ds, device, K, max_windows=20000):
        model.eval()
        idx_all = np.arange(len(ds))
        if max_windows and len(idx_all) > max_windows:
            rng = np.random.RandomState(0)
            idx_all = np.sort(rng.choice(idx_all, size=max_windows, replace=False))
        M = len(idx_all)
        bs = 2048
        pred_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
        prior_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
        pred_roll = [np.zeros((M, OBJ_DIMS), dtype=np.float32) for _ in range(K)]
        yao_hits = 0; yao_n = 0

        for j in range(0, M, bs):
            idx = idx_all[j:j + bs]
            b = ds.gather(idx)
            sl = slice(j, j + len(idx))
            xb = b["x"].to(device)
            ab = b["a0"].to(device)
            h, c = model.encode_sequence(xb)
            fine = model.evidence(c)
            delta = model.predict_state(h, ab).float().cpu().numpy()
            prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab).float().cpu().numpy()
            pred_one[sl] = (prior_n + delta) * (ds.std_obj + 1e-8) + ds.mean_obj
            prior_one[sl] = prior_n * (ds.std_obj + 1e-8) + ds.mean_obj

            yao_log = model.predict_yao(h)
            yp = (torch.sigmoid(yao_log.float()) > 0.5).cpu().numpy()
            yt = b["yao"].numpy()
            yao_hits += (yp == yt).mean() * len(idx)
            yao_n += len(idx)

            afb = b["af"].to(device); xf = b["x_fut"].to(device)
            abs_preds, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=0.0)
            for k in range(K):
                pred_roll[k][sl] = (abs_preds[k].float().cpu().numpy()
                                    * (ds.std_obj + 1e-8) + ds.mean_obj)

        frm_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L]
        tgt = frm_last * (ds.std_obj + 1e-8) + ds.mean_obj
        s_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L - 1] \
            * (ds.std_obj + 1e-8) + ds.mean_obj
        rollout_tgt = [
            ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L + k]
            * (ds.std_obj + 1e-8) + ds.mean_obj
            for k in range(K)
        ]
        metrics = {
            "one_step_model": pos_rmse(pred_one, tgt),
            "one_step_prior": pos_rmse(prior_one, tgt),
            "one_step_zero_vel": pos_rmse(s_last, tgt),
            "rollout": {f"k={k + 1}": pos_rmse(pred_roll[k], rollout_tgt[k]) for k in range(K)},
            "yao_acc": float(yao_hits / max(yao_n, 1)),
        }
        model.train()
        return metrics

    def run(self):
        args = self.args
        model = self.model
        device = self.device

        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{self.log_tag}] seed={args.seed} 参数量 {n_params:,} (可训练 {n_train:,}) | "
              f"窗口 {len(self.train_ds):,}/{len(self.val_ds):,} | device={device}", flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        wpe = len(self.train_ds)

        t0_all = time.time()
        for epoch in range(args.epochs):
            model.train()
            perm = np.random.permutation(wpe)[:wpe]
            agg, nb = {}, 0
            t0e = time.time()

            for j in range(0, wpe, args.batch_size):
                idx = perm[j:j + args.batch_size]
                batch = self.train_ds.gather(idx)
                xb = batch["x"].to(device, non_blocking=True)
                xb2 = batch["x_next"].to(device, non_blocking=True)

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                    h, c = model.encode_sequence(xb)
                    fine = model.evidence(c)
                    _, c_next = model.encode_sequence(xb2)
                    fine_next = model.evidence(c_next)

                    loss, parts = self.compute_loss(
                        model, batch, h, c, fine, fine_next, self.feat_w, device)

                    if args.w_roll > 0:
                        afb = batch["af"].to(device, non_blocking=True)
                        xf = batch["x_fut"].to(device, non_blocking=True)
                        ss = min(args.ss_max, args.ss_max * epoch / max(args.ss_warmup, 1))
                        abs_preds, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=ss)
                        roll = sum((0.7 ** k) * F.huber_loss(
                            abs_preds[k].float().clamp(-50, 50),
                            xf[:, k, :OBJ_DIMS].float(), reduction="mean")
                            for k in range(self.K)) / self.K
                        loss = loss + args.w_roll * roll
                        parts = dict(parts, roll=roll.item())

                if not torch.isfinite(loss):
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_bad = any(not torch.isfinite(p.grad.sum())
                               for p in model.parameters() if p.grad is not None)
                if grad_bad:
                    optimizer.zero_grad()
                    continue
                optimizer.step()
                model.ema_prototypes(c.detach().float(), fine.argmax(-1).detach())
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0.0) + v
                nb += 1

            sched.step()
            msg = " ".join(f"{k}={v / max(nb, 1):.4f}" for k, v in agg.items())
            print(f"  Epoch {epoch + 1}/{args.epochs} [{msg}] ({time.time() - t0e:.0f}s)")

            if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
                m = self.validate_fn(model, self.val_ds, device, self.K)
                print(f"    [val] one-step: 模型 {m['one_step_model']:.3f} | "
                      f"先验 {m['one_step_prior']:.3f} | "
                      f"零速度 {m['one_step_zero_vel']:.3f}px")
                print(f"    [val] rollout: " +
                      " ".join(f"{k}:{v:.2f}" for k, v in m["rollout"].items()) +
                      f" | 爻acc {m['yao_acc']:.3f}")
                ckpt = dict(model_state_dict=model.state_dict(),
                            val_metrics=m, epoch=epoch,
                            args=vars(args))
                torch.save(ckpt, os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"))
                if m["one_step_model"] < self.best:
                    self.best = m["one_step_model"]
                    torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
                    print(f"    [保存最佳] val one-step = {self.best:.3f}px")

        print(f"\n[{self.log_tag}] 完成 ({time.time() - t0_all:.0f}s). "
              f"最佳 val one-step = {self.best:.3f}px")
