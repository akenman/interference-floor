# -*- coding: utf-8 -*-
"""
E1a 训练 (train_e1a.py) — 7 变体 × 3 种子 × {有门控, 无门控}
========================================================
用法：
  python train_e1a.py --variant V3 --seed 0 [--no_gua]
变体表（见 model_e1a.py 头部）：V1 V2a V2b V2c V3 V4 Vfree
"""

import os
import sys
import json
import math
import random
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F

from layout_h import (OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, NUM_OBJ, DT as DT_H,
                      feat_weights as feat_weights_h, yao_labels_h)
from model_h import WorldModelH
from model_e1a import (E1aModel, E1aLoss, compute_energy_median,
                       energy_level_labels)

INPUT_DIM = OBJ_DIMS + CONTACT_DIMS   # 25
POS_IDX = [1, 2, 12, 13]
L, K = 8, 5

VARIANTS = {
    "V1":    dict(axes=(0,),       free_moe=False, n_gua_free=8),
    "V2a":   dict(axes=(0, 1),     free_moe=False, n_gua_free=8),
    "V2b":   dict(axes=(0, 2),     free_moe=False, n_gua_free=8),
    "V2c":   dict(axes=(1, 2),     free_moe=False, n_gua_free=8),
    "V3":    dict(axes=(0, 1, 2),  free_moe=False, n_gua_free=8),
    "V4":    dict(axes=(0, 1, 2, 3), free_moe=False, n_gua_free=8),
    "Vfree": dict(axes=(0, 1, 2),  free_moe=True,  n_gua_free=8),
}

# --------------------------------------------------------------------------
# 解耦消融臂（E1a-D）
# -----------------------------------------------------------------------
# 动机：历史 7 变体同时改变三件事 —— (a) 爻监督的轴数/内容、
# (b) 门控能读到的信息通路（gua_gate 恒读全部 6 维 → 被"砍掉"的轴仍经
# state loss 拿到梯度）、(c) 门控容量（n_gua = n_phases² = 4^n_axes）。
# 三者共变导致 V1 vs V3 无法归因，且所有 Δ 仅 2–4%（效应被稀释）。
#
# 下面一组把容量固定为 V3 的 64 专家，只改 (a) 与 (b)：
#   A1x64  监督 1 轴(初爻) + 门控只读初爻   → 真正的"单爻"模型
#   A2x64  监督 2 轴(初+中) + 门控只读这 2 轴
#   A3g1   监督 3 轴(全)     + 门控只读初爻 → 分离"监督"与"门控输入"
#   A3x64  与 V3 逐位等价（gate_axes=(0,1,2) 时 _select_axes 返回完整 6 维，
#          n_gua=64=n_phases² 故仍走外积）→ 仅作回归校验，正式分析以 V3 为基准，
#          无需单独跑（V3 由 Stage A 直接给出 n=10）。
VARIANTS_DECOUPLED = {
    "A1x64": dict(axes=(0,),      free_moe=False, n_gua_free=8,
                  n_gua_override=64, gate_axes=(0,)),
    "A2x64": dict(axes=(0, 1),    free_moe=False, n_gua_free=8,
                  n_gua_override=64, gate_axes=(0, 1)),
    "A3g1":  dict(axes=(0, 1, 2), free_moe=False, n_gua_free=8,
                  n_gua_override=64, gate_axes=(0,)),
    "A3x64": dict(axes=(0, 1, 2), free_moe=False, n_gua_free=8,
                  n_gua_override=64, gate_axes=(0, 1, 2)),
}
VARIANTS.update(VARIANTS_DECOUPLED)


def parse_args():
    p = argparse.ArgumentParser(description="E1a 变体训练")
    p.add_argument("--variant", type=str, required=True, choices=list(VARIANTS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--no_gua", action="store_true", help="无门控基线（M3 用）")
    p.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    p.add_argument("--heldout_path", type=str,
                   default="experiments/trajectories_hinge_heldout.npz")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--early_stop_patience", type=int, default=8,
                   help="连续多少次验证无提升后早停（每 2 个 epoch 验证一次）")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--windows_per_epoch", type=int, default=0,
                   help="每轮随机采样窗口数；0 表示使用全部训练窗口")
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
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints-e1a")
    p.add_argument("--resume", type=str, default="",
                   help="从 checkpoint 恢复训练（通常为 checkpoint_latest.pt 的路径）；"
                        "恢复 model/optimizer/scheduler/epoch/best。"
                        "注意：2026-09-21 之前的旧 checkpoint 无 optimizer 状态，"
                        "只恢复权重（optimizer 从零热启动，已如实标注）。")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_amp", action="store_true",
                   help="禁用 bf16 自动混合精度（数值稳定性排查，与 train_hinge/train_e0 一致）")
    return p.parse_args()


class HingeDataE1a:
    """惰性窗口数据源（E1a 版）：支持第4轴能量标签。"""

    def __init__(self, states, contact, actions, yao, targets, episodes, L, K,
                 mean_obj, std_obj, energy_median=None):
        self.L, self.K = L, K
        self.energy_median = energy_median
        self.mean_obj, self.std_obj = mean_obj, std_obj
        episodes = np.asarray(episodes)
        T = states.shape[1]
        E = len(episodes)
        self.obj_norm = ((states[episodes] - mean_obj) /
                         (std_obj + 1e-8)).astype(np.float32)
        self.act_norm = actions[episodes].astype(np.float32)
        self.contact = contact[episodes].astype(np.float32)
        self.yaos = yao[episodes]                      # (E, T, 3, 2)
        self.targets = targets[episodes]               # (E, 2)
        self.states = states[episodes]                 # (E, T, 22)
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

        # 第4轴（V4）：以窗口首帧前一帧（t = i0+L-1）的总动能划分
        e4 = np.zeros(B, dtype=np.int8)
        if self.energy_median is not None:
            e4 = energy_level_labels(self.states, ep, i0, L, self.energy_median)
            e4 = np.asarray(e4).reshape(-1).astype(np.int8)

        to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return dict(x=to(x), x_next=to(x_next), a0=to(a0), af=to(af),
                    d_norm=to(d_norm), y_raw=to(y_raw), yao=to(yao_t),
                    e4=to(e4), s_last=to(s_last), x_fut=to(x_fut))

    def _prior_np(self, obj_norm_batch, a_norm):
        from layout_h import L1, L2, M1, M2, TAU_MAX
        raw = obj_norm_batch * (self.std_obj + 1e-8) + self.mean_obj
        b1, b2 = 0, 11
        th1 = np.arctan2(raw[:, b1 + 3], raw[:, b1 + 4])
        th2 = np.arctan2(raw[:, b2 + 3], raw[:, b2 + 4])
        w1 = raw[:, b1 + 7]; w2 = raw[:, b2 + 7]
        tau = a_norm[:, 0] * TAU_MAX
        c2 = np.cos(th2); s2 = np.sin(th2)
        M11 = (M1 + M2) * L1 ** 2 + M2 * L2 ** 2 + 2 * M2 * L1 * L2 * c2
        M12 = M2 * L2 ** 2 + M2 * L1 * L2 * c2
        M22 = M2 * L2 ** 2
        h1 = (-M2 * L1 * L2 * (2 * w1 + w2) * w2 * s2
              + (M1 + M2) * 900.0 * L1 * np.sin(th1)
              + M2 * 900.0 * L2 * np.sin(th1 + th2))
        h2 = (M2 * L1 * L2 * w1 ** 2 * s2
              + M2 * 900.0 * L2 * np.sin(th1 + th2)
              - tau)
        det = M11 * M22 - M12 ** 2
        a1 = (M22 * (-h1) - M12 * (-h2)) / det
        a2 = (-M12 * (-h1) + M11 * (-h2)) / det
        w1n = w1 + a1 * 0.05; w2n = w2 + a2 * 0.05
        th1n = th1 + w1n * 0.05; th2n = th2 + w2n * 0.05
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


@torch.no_grad()
def validate(model, ds, device, K, max_windows=20000):
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
        delta, yao_log = model.forward_state(h, ab)
        delta = delta.float().cpu().numpy()
        prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab).float().cpu().numpy()
        pred_one[sl] = (prior_n + delta) * (ds.std_obj + 1e-8) + ds.mean_obj
        prior_one[sl] = prior_n * (ds.std_obj + 1e-8) + ds.mean_obj
        if yao_log is None:
            yao_log = model.predict_yao(h)
        yp = (torch.sigmoid(yao_log.float()) > 0.5).cpu().numpy()
        yt = b["yao"].numpy()
        # 只算模型实际学习的轴（V4 第4轴为静态阈值，无学习头）
        axes = model.learned_axes
        sel = np.concatenate([yt[:, 2 * a:2 * a + 2] for a in axes], axis=1)
        yp_sel = np.concatenate([yp[:, 2 * a:2 * a + 2] for a in axes], axis=1)
        yao_hits += (yp_sel == sel).mean() * len(idx)
        yao_n += len(idx)
        afb = b["af"].to(device); xf = b["x_fut"].to(device)
        abs_preds, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=0.0)
        for k in range(K):
            pred_roll[k][sl] = (abs_preds[k].float().cpu().numpy()
                                * (ds.std_obj + 1e-8) + ds.mean_obj)
    frm_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L]
    tgt = frm_last * (ds.std_obj + 1e-8) + ds.mean_obj
    rollout_tgt = [
        ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L + k]
        * (ds.std_obj + 1e-8) + ds.mean_obj
        for k in range(K)
    ]
    s_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L - 1] \
        * (ds.std_obj + 1e-8) + ds.mean_obj
    metrics = {
        "one_step_model": pos_rmse(pred_one, tgt),
        "one_step_prior": pos_rmse(prior_one, tgt),
        "one_step_zero_vel": pos_rmse(s_last, tgt),
        "rollout": {f"k={k + 1}": pos_rmse(pred_roll[k], rollout_tgt[k]) for k in range(K)},
        "yao_acc": float(yao_hits / max(yao_n, 1)),
    }
    model.train()
    return metrics


def build_model(variant, no_gua, energy_median, device, args):
    cfg = VARIANTS[variant]
    m = E1aModel(axes=cfg["axes"], free_moe=cfg["free_moe"],
                 no_gua=no_gua, n_gua_free=cfg["n_gua_free"],
                 energy_median=energy_median,
                 input_dim=INPUT_DIM, obj_dim=OBJ_DIMS,
                 action_dim=ACTION_DIM, hidden_dim=args.hidden_dim,
                 num_prototypes=args.num_prototypes,
                 # 解耦消融专用（历史变体不含这两键，值为 None = 保持原行为）
                 n_gua_override=cfg.get("n_gua_override"),
                 gate_axes=cfg.get("gate_axes")).to(device)
    return m


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    variant, no_gua = args.variant, args.no_gua

    print("=" * 68)
    print(f"E1a 训练 | 变体 {variant}{' (no-gua)' if no_gua else ''} | seed {args.seed}")
    print(f"  数据: {args.data_path} | 设备: {device} | L={L} K={K}")
    print("=" * 68)

    d = np.load(args.data_path)
    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
    targets = d.get("targets", np.zeros((len(states), 2)))
    N = states.shape[0]

    # 数据划分固定种子 42（协议：seed 42 数据 + args.seed 训练随机性）——
    # 修复 BUG-E1a-2：此前用 args.seed 划分，不同种子的验证集不同
    # （V1_s43 先验 2.887 vs V1_s44 先验 2.346 即是症状），种子间比较混入验证集难度差。
    rng = np.random.RandomState(42)
    segs = np.arange(N)
    rng.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]
    tr_eps = segs[int(N * 0.1):]

    # 第4轴阈值只从训练片段计算，避免验证集信息泄漏。
    energy_median = compute_energy_median(states[tr_eps]) if "V4" in (variant,) else None
    if args.variant == "V4":
        print(f"  能量中位数（训练集，第4轴阈值）: {energy_median:.1f}")

    obj_flat = states[tr_eps].reshape(-1, OBJ_DIMS).astype(np.float64)
    mean_obj = obj_flat.mean(0).astype(np.float32)
    std_obj = obj_flat.std(0).astype(np.float32)
    print(f"  片段: 训练 {len(tr_eps)} / 验证 {len(val_eps)}")

    train_ds = HingeDataE1a(states, contact, actions, yao, targets, tr_eps,
                            L, K, mean_obj, std_obj, energy_median)
    val_ds = HingeDataE1a(states, contact, actions, yao, targets, val_eps,
                          L, K, mean_obj, std_obj, energy_median)
    print(f"  窗口: 训练 {len(train_ds):,} / 验证 {len(val_ds):,}")

    model = build_model(variant, no_gua, energy_median, device, args)
    model.set_stats(mean_obj, std_obj)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,} (可训练 {n_train:,}) | 相位数 {model.n_phases}"
          f" | 门控基数 {model.n_gua if model.has_gate else '-'}")

    criterion = E1aLoss(args.w_state, args.w_yao, args.w_proto,
                        args.w_commit, args.w_usage)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=args.epochs)
    feat_w_t = torch.from_numpy(feat_weights_h(OBJ_DIMS)).to(device)

    best = float("inf")
    no_improve_checks = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        rs = torch.load(args.resume, map_location=device, weights_only=False)
        protocol = rs.get("config", {}).get("protocol_version")
        if protocol != "e1a_corrected_v2":
            raise RuntimeError(
                "拒绝恢复旧协议 checkpoint（缺少 protocol_version=e1a_corrected_v2）；"
                "请使用新的 checkpoint_dir 重新训练。"
            )
        model.load_state_dict(rs["model_state_dict"])
        if "optimizer_state_dict" in rs:
            optimizer.load_state_dict(rs["optimizer_state_dict"])
            sched.load_state_dict(rs["scheduler_state_dict"])
            resume_mode = "完整（含 optimizer/scheduler）"
        else:
            resume_mode = "仅权重（旧格式 checkpoint 无 optimizer 状态，热启动）"
        start_epoch = int(rs.get("epoch", -1)) + 1
        best = float(rs.get("best", float("inf")))
        if best == float("inf"):
            # 旧格式 checkpoint 无 best 记录：从既有 best.pt 恢复，防止更差的 val
            # 覆盖历史最优（2026-09-22 修复）
            bp = os.path.join(args.checkpoint_dir, "best.pt")
            if os.path.exists(bp):
                try:
                    bck = torch.load(bp, map_location="cpu", weights_only=False)
                    if isinstance(bck.get("best"), float):
                        best = float(bck["best"])
                    elif bck.get("val_metrics", {}).get("one_step_model") is not None:
                        best = float(bck["val_metrics"]["one_step_model"])
                    print(f"[resume] best 从既有 best.pt 恢复 = {best:.3f}px")
                except Exception as e:
                    print(f"[resume] best.pt 恢复失败（{e}），以 inf 继续")
        print(f"[resume] 从 {args.resume} 恢复：{resume_mode}；"
              f"从 epoch {start_epoch} 继续，best={best:.3f}px")
    n_train_win = len(train_ds)
    wpe = (n_train_win if args.windows_per_epoch <= 0
           else min(n_train_win, args.windows_per_epoch))
    print(f"  每轮窗口: {wpe:,} / {n_train_win:,}")
    use_amp = (device.type == "cuda") and not args.no_amp

    t0_all = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        perm = np.random.permutation(n_train_win)[:wpe]
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
            yaob = batch["yao"].to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                h, c = model.encode_sequence(xb)
                fine = model.evidence(c)
                _, c_next = model.encode_sequence(xb2)
                fine_next = model.evidence(c_next)
                loss, parts, _ = criterion(model, h, c, fine, ab, db, yaob,
                                           fine_next, feat_w_t)
                if args.w_roll > 0:
                    xf = batch["x_fut"].to(device, non_blocking=True)
                    ss = min(args.ss_max, args.ss_max * epoch / 20.0)  # 20 epoch 线性升温
                    abs_preds, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=ss)
                    roll = sum((0.7 ** k) * F.huber_loss(
                        abs_preds[k].float().clamp(-50, 50),
                        xf[:, k, :OBJ_DIMS].float(), reduction="mean")
                        for k in range(K)) / K
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
        print(f"Epoch {epoch + 1}/{args.epochs} [{msg}] ({time.time() - t0e:.0f}s)")

        if (epoch + 1) % 2 == 0 or epoch == args.epochs - 1:
            m = validate(model, val_ds, device, K)
            print(f"  [val] one-step: 模型 {m['one_step_model']:.3f} | 先验 "
                  f"{m['one_step_prior']:.3f} | 零速度 {m['one_step_zero_vel']:.3f}px")
            print(f"  [val] rollout: " + " ".join(f"{k}:{v:.2f}" for k, v in m["rollout"].items())
                  + f" | 爻acc {m['yao_acc']:.3f}")
            improved = m["one_step_model"] < best
            best = min(best, m["one_step_model"])
            no_improve_checks = 0 if improved else no_improve_checks + 1
            ckpt = dict(model_state_dict=model.state_dict(),
                        config=dict(variant=variant, no_gua=no_gua, seed=args.seed,
                                    protocol_version="e1a_corrected_v2",
                                    axes=list(model.axes), free_moe=model.free_moe,
                                    n_gua=model.n_gua, n_phases=model.n_phases,
                                    input_dim=INPUT_DIM, obj_dim=OBJ_DIMS,
                                    action_dim=ACTION_DIM,
                                    hidden_dim=args.hidden_dim,
                                    num_prototypes=args.num_prototypes,
                                    energy_median=energy_median),
                        norm_stats=dict(mean_obj=mean_obj, std_obj=std_obj),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=sched.state_dict(),
                        best=best,
                        args=vars(args), epoch=epoch, val_metrics=m)
            torch.save(ckpt, os.path.join(args.checkpoint_dir,
                                          "checkpoint_latest.pt"))
            if improved:
                torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
                print(f"  [保存最佳] val one-step = {best:.3f}px")

            # 崩溃早停：曾收敛良好(best<5)但当前验证发散(>10px)，判定为训练崩溃并终止本 run
            # best checkpoint 已在上面保存，数据不丢；本行作为 E1a 崩溃率统计的标记
            if m["one_step_model"] > 10.0 and best < 5.0:
                print(f"[EARLY-STOP] collapse=1 epoch={epoch + 1} "
                      f"val={m['one_step_model']:.1f}px best={best:.3f}px")
                break
            if (args.early_stop_patience > 0 and
                    no_improve_checks >= args.early_stop_patience):
                print(f"[EARLY-STOP] no_improve={no_improve_checks} epoch={epoch + 1} "
                      f"val={m['one_step_model']:.3f}px best={best:.3f}px")
                break

    print(f"\n[OK] 完成 ({time.time() - t0_all:.0f}s). 最佳 val one-step = {best:.3f}px")


if __name__ == "__main__":
    main()
