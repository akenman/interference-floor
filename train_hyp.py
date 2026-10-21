# -*- coding: utf-8 -*-
"""
多假设预测训练与判决 (train_hyp.py)
==================================================
T11 出路 + "生生之谓易"计算化。K=8 假设头，winner 软分配训练。

预注册判据（对照单点模型 checkpoints-hinge/best.pt）：
  A: best-of-8 @ k=5 RMSE < 单点 k=5 × 0.5
  B: k=5 假设间平均两两距离 > 单点 k=5 误差（假设真分叉且覆盖不确定性）
  C: oracle 命中率（任一假设 ≤40px）@ k=5 > 单点对应比例

用法：python train_hyp.py --data_path trajectories_hinge.npz --epochs 40
"""

import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from layout_h import OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, feat_weights as feat_weights_h
from model_h import WorldModelH, MultiHypWorldModelH
from train_hinge import HingeData, pos_rmse

INPUT_DIM = OBJ_DIMS + CONTACT_DIMS
POS_IDX = [1, 2, 12, 13]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--baseline_ckpt", type=str, default="checkpoints-hinge/best.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--future_steps", type=int, default=5)
    ap.add_argument("--num_prototypes", type=int, default=16)
    ap.add_argument("--n_hyp", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w_state", type=float, default=1.0)
    ap.add_argument("--w_yao", type=float, default=0.1)
    ap.add_argument("--w_proto", type=float, default=0.05)
    ap.add_argument("--w_commit", type=float, default=0.01)
    ap.add_argument("--w_usage", type=float, default=0.02)
    ap.add_argument("--w_conf", type=float, default=0.3)
    ap.add_argument("--windows_per_epoch", type=int, default=400000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints-hyp")
    args = ap.parse_args()
    device = torch.device(args.device)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    L, K = args.seq_len, args.future_steps

    print("=" * 68)
    print(f"多假设预测训练（K={args.n_hyp}，winner 软分配，beam 滚动）")
    print("=" * 68)

    d = np.load(args.data_path)
    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = states.shape[0]
    rng = np.random.RandomState(args.seed)
    segs = np.arange(N); rng.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]
    tr_eps = segs[int(N * 0.1):]
    obj_flat = states[tr_eps].reshape(-1, OBJ_DIMS).astype(np.float64)
    mean_obj = obj_flat.mean(0).astype(np.float32)
    std_obj = obj_flat.std(0).astype(np.float32)

    train_ds = HingeData(states, contact, actions, yao, tr_eps, L, K, mean_obj, std_obj)
    val_ds = HingeData(states, contact, actions, yao, val_eps, L, K, mean_obj, std_obj)

    model = MultiHypWorldModelH(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=ACTION_DIM,
                                hidden_dim=args.hidden_dim, num_prototypes=args.num_prototypes,
                                n_hyp=args.n_hyp).to(device)
    model.set_stats(mean_obj, std_obj)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    fw = feat_weights_h(OBJ_DIMS)
    feat_w_t = torch.from_numpy(fw).to(device)

    # ---- 单点基线（对照） ----
    bck = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
    base = WorldModelH(**bck["config"]).to(device)
    base.load_state_dict(bck["model_state_dict"], strict=False)
    base.eval()

    n_train = len(train_ds)
    wpe = args.windows_per_epoch if args.windows_per_epoch > 0 else n_train
    T_soft = 1.0

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_train)[:wpe]
        agg, nb = {}, 0
        for j in range(0, wpe, args.batch_size):
            idx = perm[j:j + args.batch_size]
            b = train_ds.gather(idx)
            xb = b["x"].to(device, non_blocking=True)
            xb2 = b["x_next"].to(device, non_blocking=True)
            ab = b["a0"].to(device, non_blocking=True)
            db = b["d_norm"].to(device, non_blocking=True)
            yaob = b["yao"].to(device, non_blocking=True)

            h, c = model.encode_sequence(xb)
            fine = model.evidence(c)
            prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab)
            tgt_abs = prior_n + db                              # (B, 22) 绝对目标
            hyps, conf = model.predict_hyps(h, ab)              # 残差 (B,K,22)
            abs_h = prior_n.unsqueeze(1) + hyps                 # (B,K,22) 绝对假设
            err2 = ((abs_h - tgt_abs.unsqueeze(1)) ** 2).mean(dim=-1)   # (B,K)
            soft = F.softmax(-err2 / T_soft, dim=-1)
            hub = F.huber_loss(abs_h, tgt_abs.unsqueeze(1).expand_as(abs_h),
                               reduction="none").mean(dim=-1)
            hyp_loss = (soft.detach() * hub).sum(dim=-1).mean()
            conf_loss = F.cross_entropy(conf, soft.detach(), reduction="mean")

            yao_log = model.predict_yao(h)
            yao_loss = F.binary_cross_entropy_with_logits(yao_log, yaob.float(), reduction="mean")
            pfine = model.predict_proto(h, ab)
            _, c_next = model.encode_sequence(xb2)
            fine_next = model.evidence(c_next)
            proto_loss = F.kl_div(F.log_softmax(pfine, dim=-1),
                                  F.softmax(fine_next, dim=-1), reduction="batchmean")
            assigned = model.prototypes.detach()[fine.argmax(dim=-1)]
            commit_loss = (1.0 - F.cosine_similarity(assigned, c, dim=-1)).mean()
            assignment = F.softmax(fine.float(), dim=-1)
            bavg = assignment.mean(0).clamp(min=1e-8)
            usage_loss = (math.log(model.num_prototypes)
                          - (-(bavg * bavg.clamp(min=1e-8).log()).sum())) / math.log(model.num_prototypes)

            loss = (args.w_state * hyp_loss + args.w_yao * yao_loss
                    + args.w_proto * proto_loss + args.w_commit * commit_loss
                    + args.w_usage * usage_loss + 0.3 * conf_loss)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if any(not torch.isfinite(p.grad.sum()) for p in model.parameters() if p.grad is not None):
                optimizer.zero_grad()
                continue
            optimizer.step()
            model.ema_prototypes(c.detach().float(), fine.argmax(-1).detach())
            for k, v in dict(state=hyp_loss.item(), conf=conf_loss.item(),
                             yao=yao_loss.item()).items():
                agg[k] = agg.get(k, 0.0) + v
        sched.step()
        msg = " ".join(f"{k}={v / max(nb, 1):.4f}" for k, v in agg.items())
        print(f"Epoch {epoch + 1}/{args.epochs} [{msg}]")

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            m = evaluate(model, base, val_ds, device, K, 20000, POS_IDX)
            print(f"  [val] k=1: best-of-K {m['hyp_best_k1']:.2f} vs 单点 {m['base_k1']:.2f} | "
                  f"k=5: best {m['hyp_best_k5']:.2f} vs 单点 {m['base_k5']:.2f}")
            print(f"  [val] 分叉度 k=5: {m['spread_k5']:.2f}px | "
                  f"oracle命中(≤40px) k=5: {m['oracle_k5']:.1%} vs 单点 {m['base_hit_k5']:.1%}")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(dict(model_state_dict=model.state_dict(),
                             config=dict(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS,
                                         action_dim=ACTION_DIM, hidden_dim=args.hidden_dim,
                                         num_prototypes=args.num_prototypes, n_hyp=args.n_hyp),
                             norm_stats=dict(mean_obj=mean_obj, std_obj=std_obj),
                             args=vars(args), epoch=epoch, val_metrics=m),
                       os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"))


@torch.no_grad()
def evaluate(model, base, ds, device, K, max_windows, pos_idx):
    """判决指标（逐 k 正确目标帧）：best-of-K / 分叉度 / oracle 命中，物理单位。"""
    model.eval(); base.eval()
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rs = np.random.RandomState(0)
        idx_all = np.sort(rs.choice(idx_all, size=max_windows, replace=False))
    M = len(idx_all); bs = 2048
    n_hyp = model.n_hyp
    hyp_err = np.zeros((M, K, n_hyp))
    spread = np.zeros((M, K))
    base_err = np.zeros((M, K))
    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = b["x"].to(device)
        afb = b["af"].to(device)
        xf = b["x_fut"].to(device)          # (B, K, 25) 真值未来帧（归一化）
        # 多假设 beam 滚动
        steps, _ = model.rollout_hyp(xb, afb)
        # 单点基线滚动
        bsteps, _ = base.rollout(xb, afb)
        tgt_all = xf[:, :, :OBJ_DIMS].float().cpu().numpy()   # (B, K, 22)
        for k in range(K):
            raw = steps[k].float().cpu().numpy()              # (B, n_hyp, 22)
            hyp_err[sl, k] = np.abs(raw[:, :, pos_idx] - tgt_all[:, k, pos_idx][:, None, :]).mean(axis=2)
            pr = raw[:, :, pos_idx]
            d2 = 0.0; cnt = 0
            for i in range(n_hyp):
                for h2 in range(i + 1, n_hyp):
                    d2 += np.hypot(pr[:, i, 0] - pr[:, h2, 0], pr[:, i, 1] - pr[:, h2, 1]).mean()
                    cnt += 1
            spread[sl, k] = d2 / max(cnt, 1)
            raw_b = bsteps[k].float().cpu().numpy()
            base_err[sl, k] = np.abs(raw_b[:, pos_idx] - tgt_all[:, k, pos_idx]).mean(axis=1)

    pos_std = float(np.mean(ds.std_obj[POS_IDX]))
    out = {"pos_std_px": pos_std}
    for k in range(K):
        best_k = hyp_err[:, k].min(axis=1) * pos_std          # best-of-K(px)
        out[f"hyp_best_k{k+1}"] = float(np.sqrt((best_k**2).mean()))
        out[f"base_k{k+1}"] = float(np.sqrt((base_err[:, k]**2).mean())) * pos_std
        out[f"spread_k{k+1}"] = float(spread[:, k].mean())
        out[f"oracle_k{k+1}"] = float((best_k <= 20).mean())   # 最优假设 MAE ≤20px
        out[f"base_hit_k{k+1}"] = float((base_err[:, k] * pos_std <= 20).mean())
    return out


if __name__ == "__main__":
    main()
