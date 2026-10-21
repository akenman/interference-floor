# -*- coding: utf-8 -*-
"""
E1b 训练脚本 (train_e1b.py) — 相态容量分配三臂
==================================================
三臂（同参数量、仅分配方式不同）：
  uniform  : 标准 WorldModelH 状态主头 256 维（ours-base）
  phase    : PhaseCapacityWorldModelH — 8 个相态专用子头，卦门控软路由
  priority : WorldModelH + PriorityHLoss — 损失按卦难度加权

用法：
  python train_e1b.py --variant uniform --seed 42 --epochs 60
  python train_e1b.py --variant phase   --seed 42 --epochs 60
  python train_e1b.py --variant priority --seed 42 --epochs 60 --gua_weights 1.0,1.0,...

配置与 E0 统一：epochs=60, batch=256, lr=1e-3(AdamW wd=1e-5),
  w_state=1.0 w_yao=0.1 w_proto=0.05 w_commit=0.01 w_usage=0.02,
  ss_max=0.5 线性升温 20 epochs, AMP bf16, 输出层零初始化。
"""

import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from layout_h import (OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, NUM_OBJ, PER_OBJ,
                      DT as DT_H, feat_weights as feat_weights_h)
from model_h_e1b import PhaseCapacityWorldModelH, PriorityHLoss
from model_h import WorldModelH, HLoss

INPUT_DIM = OBJ_DIMS + CONTACT_DIMS   # 22 + 3 = 25
POS_IDX = [1, 2, 12, 13]              # 肘(x,y) 手(x,y)
GUA_NAMES = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]


def parse_args():
    p = argparse.ArgumentParser(description="E1b 相态容量分配三臂训练")
    p.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    p.add_argument("--variant", type=str, choices=["uniform", "phase", "priority"],
                   default="uniform")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--sub_head_dim", type=int, default=0,
                   help="0 -> auto-select so B-phase param count matches B-uniform (within 1 pct)")
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--future_steps", type=int, default=5)
    p.add_argument("--num_prototypes", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--w_state", type=float, default=1.0)
    p.add_argument("--w_yao", type=float, default=0.1)
    p.add_argument("--w_proto", type=float, default=0.05)
    p.add_argument("--w_commit", type=float, default=0.01)
    p.add_argument("--w_usage", type=float, default=0.02)
    p.add_argument("--w_roll", type=float, default=0.3)
    p.add_argument("--ss_max", type=float, default=0.5)
    p.add_argument("--windows_per_epoch", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints-e1b")
    p.add_argument("--log_dir", type=str, default="runs/e1b")
    p.add_argument("--gua_weights", type=str, default="",
                   help="8 个相对误差值，逗号分隔，仅 priority 臂使用")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def match_phase_params(base_params, n_gua=8, action_dim=1, obj_dim=22):
    """求子头隐藏维 d，使 8×(Linear(257,d)+Linear(d,22)) ≈ base_params。"""
    for d in range(8, 257):
        got = n_gua * (((256 + action_dim) * d + d) + (d * obj_dim + obj_dim))
        if abs(got - base_params) / base_params <= 0.01:
            return d, got
    # 兜兜：取最接近的
    best = min(range(8, 257), key=lambda d: abs(
        n_gua * (((256 + action_dim) * d + d) + (d * obj_dim + obj_dim)) - base_params))
    got = n_gua * (((256 + action_dim) * best + best) + (best * obj_dim + obj_dim))
    return best, got


def gua_idx_from_yao(yao_t):
    """真值爻标签 → 手端(物体1) 卦索引 (B,)。

    yao_t: (B, 6) 来自 train_hinge 的 reshape(B,3,2)（原数组 (3 轴, 2 物体) 行主序）:
        [轴0_物0, 轴0_物1, 轴1_物0, 轴1_物1, 轴2_物0, 轴2_物1]
    手端（物体1）三爻 = 下标 1, 3, 5。
    """
    bits = yao_t[:, [1, 3, 5]].long()                       # (B, 3) = [初, 中, 上] of hand
    return bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]


class HingeData:
    """惰性窗口数据源（铰链世界），与 train_hinge 同源。"""

    def __init__(self, states, contact, actions, yao, episodes, L, K,
                 mean_obj, std_obj):
        self.L, self.K = L, K
        self.mean_obj, self.std_obj = mean_obj, std_obj
        episodes = np.asarray(episodes)
        T = states.shape[1]
        E = len(episodes)
        self.obj_norm = ((states[episodes] - mean_obj) / (std_obj + 1e-8)).astype(np.float32)
        self.act_norm = actions[episodes].astype(np.float32)
        self.contact = contact[episodes].astype(np.float32)
        self.yaos = yao[episodes]                              # (E, T, 3, 2)
        n_win = T - L - K + 1
        self.ep_idx = np.repeat(np.arange(E), n_win)
        self.win_start = np.tile(np.arange(n_win), E)

    def __len__(self):
        return len(self.ep_idx)

    def gather(self, idx):
        ep = self.ep_idx[idx]
        i0 = self.win_start[idx]
        L, K = self.L, self.K
        B = len(idx)
        off = np.arange(L + K)
        frm = self.obj_norm[ep[:, None], i0[:, None] + off[None, :]]
        ct = self.contact[ep[:, None], i0[:, None] + off[None, :]]

        def assemble(obj_seg, ct_seg, n):
            return np.concatenate([obj_seg, ct_seg], axis=2)

        x = assemble(frm[:, :L], ct[:, :L], L)
        x_next = assemble(frm[:, 1:L + 1], ct[:, 1:L + 1], L)
        x_fut = assemble(frm[:, L:L + K], ct[:, L:L + K], K)
        a0 = self.act_norm[ep, i0 + L]
        af = self.act_norm[ep[:, None], i0[:, None] + L + np.arange(K)[None, :]]
        prior_n = self._prior_np(frm[:, L - 1], a0)
        d_norm = frm[:, L] - prior_n
        y_raw = frm[:, L] * (self.std_obj + 1e-8) + self.mean_obj
        s_last = frm[:, L - 1] * (self.std_obj + 1e-8) + self.mean_obj
        yao_t = self.yaos[ep, i0 + L].reshape(B, 6)
        ct_t = self.contact[ep, i0 + L]

        to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return dict(x=to(x), x_next=to(x_next), a0=to(a0), af=to(af),
                    d_norm=to(d_norm), y_raw=to(y_raw), yao=to(yao_t),
                    s_last=to(s_last), ct=to(ct_t), x_fut=to(x_fut))

    def _prior_np(self, obj_norm_batch, a_norm):
        from layout_h import L1, L2, M1, M2, TAU_MAX
        raw = obj_norm_batch * (self.std_obj + 1e-8) + self.mean_obj
        b1, b2 = 0, 11
        th1 = np.arctan2(raw[:, b1 + 3], raw[:, b1 + 4])
        th2 = np.arctan2(raw[:, b2 + 3], raw[:, b2 + 4])
        w1 = raw[:, b1 + 7]
        w2 = raw[:, b2 + 7]
        tau = a_norm[:, 0] * TAU_MAX
        c2 = np.cos(th2); s2 = np.sin(th2)
        M11 = (M1 + M2) * L1**2 + M2 * L2**2 + 2 * M2 * L1 * L2 * c2
        M12 = M2 * L2**2 + M2 * L1 * L2 * c2
        M22 = M2 * L2**2
        h1 = (-M2 * L1 * L2 * (2 * w1 + w2) * w2 * s2
              + (M1 + M2) * 900.0 * L1 * np.sin(th1)
              + M2 * 900.0 * L2 * np.sin(th1 + th2))
        h2 = (M2 * L1 * L2 * w1**2 * s2 + M2 * 900.0 * L2 * np.sin(th1 + th2)
              - tau)
        det = M11 * M22 - M12**2
        det = np.where(np.abs(det) < 1e-9, np.sign(det) * 1e-9, det)
        a1 = (M22 * (-h1) - M12 * (-h2)) / det
        a2 = (-M12 * (-h1) + M11 * (-h2)) / det
        w1n = w1 + a1 * 0.05
        w2n = w2 + a2 * 0.05
        th1n = th1 + w1n * 0.05
        th2n = th2 + w2n * 0.05
        out = raw.copy()
        x1 = L1 * np.sin(th1n); y1 = -L1 * np.cos(th1n)
        x2 = x1 + L2 * np.sin(th1n + th2n); y2 = y1 - L2 * np.cos(th1n + th2n)
        vx1 = L1 * w1n * np.cos(th1n); vy1 = L1 * w1n * np.sin(th1n)
        vx2 = vx1 + L2 * (w1n + w2n) * np.cos(th1n + th2n)
        vy2 = vy1 + L2 * (w1n + w2n) * np.sin(th1n + th2n)
        out[:, b1 + 1], out[:, b1 + 2] = x1, y1
        out[:, b2 + 1], out[:, b2 + 2] = x2, y2
        out[:, b1 + 3], out[:, b1 + 4] = np.sin(th1n), np.cos(th1n)
        out[:, b2 + 3], out[:, b2 + 4] = np.sin(th2n), np.cos(th2n)
        out[:, b1 + 5], out[:, b1 + 6] = vx1, vy1
        out[:, b2 + 5], out[:, b2 + 6] = vx2, vy2
        out[:, b1 + 7], out[:, b2 + 7] = w1n, w2n
        return (out - self.mean_obj) / (self.std_obj + 1e-8)


def pos_rmse(pred_raw, tgt_raw):
    err = (pred_raw[:, POS_IDX] - tgt_raw[:, POS_IDX]) ** 2
    return float(np.sqrt(err.mean()))


def per_gua_rmse(pred_raw, tgt_raw, gua_idx, n_gua=8):
    """返回 (n_gua,) 各卦位置 RMSE 与 (n_gua,) 各卦样本数。"""
    errs = np.zeros(n_gua, dtype=np.float64)
    counts = np.zeros(n_gua, dtype=np.int64)
    e = np.sqrt(((pred_raw[:, POS_IDX] - tgt_raw[:, POS_IDX]) ** 2).mean(axis=1))
    for g in range(n_gua):
        m = gua_idx == g
        if m.any():
            errs[g] = float(e[m].mean())
            counts[g] = int(m.sum())
    return errs, counts


@torch.no_grad()
def validate(model, ds, device, K, max_windows, variant):
    model.eval()
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rng = np.random.RandomState(0)
        idx_all = np.sort(rng.choice(idx_all, size=max_windows, replace=False))
    M = len(idx_all)
    bs = 1024
    pred_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    prior_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    pred_roll = [np.zeros((M, OBJ_DIMS), dtype=np.float32) for _ in range(K)]
    yao_hits = 0
    yao_n = 0
    gua_idx_all = np.zeros(M, dtype=np.int64)
    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = b["x"].to(device)
        ab = b["a0"].to(device)
        h, c = model.encode_sequence(xb)
        if variant == "phase":
            delta, _ = model.forward_state(h, ab)
        else:
            delta = model.predict_state(h, ab)
        delta = delta.float().cpu().numpy()
        prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab).float().cpu().numpy()
        pred_one[sl] = (prior_n + delta) * (ds.std_obj + 1e-8) + ds.mean_obj
        prior_one[sl] = prior_n * (ds.std_obj + 1e-8) + ds.mean_obj
        yao_log = model.predict_yao(h)
        yp = (torch.sigmoid(yao_log.float()) > 0.5).cpu().numpy()
        yt = b["yao"].numpy()
        yao_hits += (yp == yt).mean() * len(idx)
        yao_n += len(idx)
        afb = b["af"].to(device)
        xf = b["x_fut"].to(device)
        abs_preds, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=0.0)
        for k in range(K):
            pred_roll[k][sl] = abs_preds[k].float().cpu().numpy() * (ds.std_obj + 1e-8) + ds.mean_obj
        gua_idx_all[sl] = gua_idx_from_yao(b["yao"]).numpy()

    frm_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L]
    tgt = frm_last * (ds.std_obj + 1e-8) + ds.mean_obj
    rollout_tgt = [
        ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L + k]
        * (ds.std_obj + 1e-8) + ds.mean_obj
        for k in range(K)
    ]
    s_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L - 1] \
        * (ds.std_obj + 1e-8) + ds.mean_obj

    gua_err, gua_cnt = per_gua_rmse(pred_one, tgt, gua_idx_all)

    metrics = {
        "one_step_model": pos_rmse(pred_one, tgt),
        "one_step_prior": pos_rmse(prior_one, tgt),
        "one_step_zero_vel": pos_rmse(s_last, tgt),
        "rollout": {f"k={k + 1}": pos_rmse(pred_roll[k], rollout_tgt[k]) for k in range(K)},
        "yao_acc": float(yao_hits / max(yao_n, 1)),
        "gua_err": {f"g{g} ({GUA_NAMES[g]})": float(gua_err[g]) for g in range(8)},
        "gua_count": {f"g{g}": int(gua_cnt[g]) for g in range(8)},
    }
    model.train()
    return metrics


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    L, K = args.seq_len, args.future_steps

    print("=" * 72)
    print(f"E1b 训练 · variant={args.variant} seed={args.seed}")
    print(f"  数据: {args.data_path} | 设备: {device} | L={L} K={K}")
    print("=" * 72)

    d = np.load(args.data_path)
    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = states.shape[0]
    rng = np.random.RandomState(args.seed)
    segs = np.arange(N)
    rng.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]
    tr_eps = segs[int(N * 0.1):]

    obj_flat = states[tr_eps].reshape(-1, OBJ_DIMS).astype(np.float64)
    mean_obj = obj_flat.mean(0).astype(np.float32)
    std_obj = obj_flat.std(0).astype(np.float32)
    print(f"  片段: 训练 {len(tr_eps)} / 验证 {len(val_eps)}")

    train_ds = HingeData(states, contact, actions, yao, tr_eps, L, K, mean_obj, std_obj)
    val_ds = HingeData(states, contact, actions, yao, val_eps, L, K, mean_obj, std_obj)
    print(f"  窗口: 训练 {len(train_ds):,} / 验证 {len(val_ds):,}")

    # ---- 模型 -------------------------------------------------------
    base_params = sum(p.numel() for p in WorldModelH(
        input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim, num_prototypes=args.num_prototypes).parameters())

    if args.variant == "phase":
        if args.sub_head_dim <= 0:
            # 选子头隐藏维，使 B-phase 总参数量与 B-uniform 差距 ≤1%
            for d in range(8, 257):
                got = sum(p.numel() for p in PhaseCapacityWorldModelH(
                    hidden_dim=args.hidden_dim, num_prototypes=args.num_prototypes,
                    sub_head_dim=d).parameters())
                if abs(got - base_params) / base_params <= 0.01:
                    args.sub_head_dim = d
                    break
            else:
                args.sub_head_dim = min(range(8, 257), key=lambda d: abs(
                    sum(p.numel() for p in PhaseCapacityWorldModelH(
                        hidden_dim=args.hidden_dim, num_prototypes=args.num_prototypes,
                        sub_head_dim=d).parameters()) - base_params))
        model = PhaseCapacityWorldModelH(
            input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=ACTION_DIM,
            hidden_dim=args.hidden_dim, num_prototypes=args.num_prototypes,
            sub_head_dim=args.sub_head_dim).to(device)
        got = sum(p.numel() for p in model.parameters())
        print(f"  子头隐藏维: {args.sub_head_dim} → 参数量 {got:,} (基线 {base_params:,}, "
              f"{(got / base_params - 1) * 100:+.1f}%)")
    else:
        model = WorldModelH(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=ACTION_DIM,
                            hidden_dim=args.hidden_dim,
                            num_prototypes=args.num_prototypes).to(device)
    model.set_stats(mean_obj, std_obj)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,} (基线 {base_params:,})")

    # ---- 损失 -------------------------------------------------------
    if args.variant == "priority":
        if not args.gua_weights:
            raise ValueError("--gua_weights 要求 8 个相对误差值（priority 臂）")
        rel = [float(x) for x in args.gua_weights.split(",")]
        if len(rel) != 8:
            raise ValueError("--gua_weights 必须有 8 个值")
        print(f"  卦相对误差: {rel}")
        print(f"  卦权重 (归一化后): "
              + ", ".join(f"{GUA_NAMES[g]}={w:.3f}" for g, w in
                          enumerate(PriorityHLoss(rel)._gua_weight.tolist())))
        criterion = PriorityHLoss(rel, args.w_state, args.w_yao, args.w_proto,
                                  args.w_commit, args.w_usage).to(device)
    else:
        criterion = HLoss(args.w_state, args.w_yao, args.w_proto, args.w_commit, args.w_usage).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    fw = feat_weights_h(OBJ_DIMS)
    feat_w_t = torch.from_numpy(fw).to(device)

    best = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    n_train = len(train_ds)
    wpe = args.windows_per_epoch if args.windows_per_epoch > 0 else n_train
    use_amp = device.type == "cuda"

    import time
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_train)[:wpe]
        agg, nb = {}, 0
        t0e = time.time()
        for j in range(0, wpe, args.batch_size):
            idx = perm[j:j + args.batch_size]
            batch = train_ds.gather(idx)
            xb = batch["x"].to(device, non_blocking=True)
            xb2 = batch["x_next"].to(device, non_blocking=True)
            ab = batch["a0"].to(device, non_blocking=True)
            afb = batch["af"].to(device, non_blocking=True)
            db = batch["d_norm"].to(device, non_blocking=True)
            yaob = batch["yao"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                h, c = model.encode_sequence(xb)
                fine = model.evidence(c)
                _, c_next = model.encode_sequence(xb2)
                fine_next = model.evidence(c_next)
                kwargs = {}
                if args.variant == "priority":
                    kwargs["gua_idx"] = gua_idx_from_yao(batch["yao"]).to(device)
                loss, parts, _ = criterion(model, h, c, fine, ab, db, yaob,
                                           fine_next, feat_w_t, **kwargs)
                if args.w_roll > 0:
                    xf = batch["x_fut"].to(device, non_blocking=True)
                    ss = min(args.ss_max, 0.025 * (epoch + 1))   # 线性升温 20 epochs
                    abs_preds, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=ss)
                    roll = sum((0.7**k) * F.huber_loss(
                        abs_preds[k].float().clamp(-50, 50),
                        xf[:, k, :OBJ_DIMS].float(), reduction="mean") for k in range(K)) / K
                    loss = loss + args.w_roll * roll
                    parts = dict(parts, roll=roll.item())
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_bad = any(
                not torch.isfinite(p.grad.sum()) for p in model.parameters()
                if p.grad is not None)
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
        print(f"Epoch {epoch + 1}/{args.epochs} [{msg}] ({time.time() - t0e:.0f}s)")

        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            m = validate(model, val_ds, device, K, 20000, args.variant)
            ge = m["gua_err"]
            print(f"  [val] one-step: 模型 {m['one_step_model']:.3f} | 先验 {m['one_step_prior']:.3f}"
                  f" | 零速 {m['one_step_zero_vel']:.3f} px")
            print(f"  [val] 卦误差: " + " ".join(f"{n}={v:.2f}" for n, v in ge.items()))
            print(f"  [val] 爻acc {m['yao_acc']:.3f} | rollout "
                  + " ".join(f"{k}:{v:.2f}" for k, v in m["rollout"].items()))
            ckpt = dict(model_state_dict=model.state_dict(),
                        config=dict(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS,
                                    action_dim=ACTION_DIM, hidden_dim=args.hidden_dim,
                                    num_prototypes=args.num_prototypes,
                                    variant=args.variant, sub_head_dim=args.sub_head_dim,
                                    gua_weights=args.gua_weights),
                        norm_stats=dict(mean_obj=mean_obj, std_obj=std_obj),
                        args=vars(args), epoch=epoch, val_metrics=m)
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"))
            if m["one_step_model"] < best:
                best = m["one_step_model"]
                torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
                print(f"  [保存最佳] val one-step = {best:.3f}px")

    print(f"\n[OK] {args.variant} seed={args.seed} 完成。最佳验证 one-step = {best:.3f}px")


if __name__ == "__main__":
    main()
