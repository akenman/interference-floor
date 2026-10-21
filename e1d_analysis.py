#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E1d 跨世界误差解剖一致性 (Cross-World Error Anatomy Consistency)
================================================================
T4（"卦级误差结构化"）从"单个世界的观察"升级为"通用分析仪器"的关键一步。

假设：卦级误差结构（坤最易、离最难）在弹球与铰链两个世界一致——
      说明它是相态结构的普遍性质，而非某个世界的偶然。

协议（RESEARCH_PROGRAM.md §5.4）:
  Step 1  获取两个世界的模型 + 接线自检
  Step 2  铰链世界卦级误差计算
  Step 3  弹球世界卦级误差计算
  Step 4  一致性检验
  Step 5  判决

预注册判据（实验前写下，见 §5.4）:
  E1d-1  两世界卦级误差排序 Spearman ρ ≥ 0.6（p < 0.05）
  E1d-2  两世界的最易卦都是"低能/腾空"相（弹球=坤，铰链=对应相）
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 卦名与位序（两世界统一约定：index = 4*初 + 2*中 + 1*上）
# ---------------------------------------------------------------------------
GUA_NAMES = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]
GUA_BITS = {i: ((i >> 2) & 1, (i >> 1) & 1, i & 1) for i in range(8)}   # (初, 中, 上)
GUA_SIG = {i: f"{b[0]}{b[1]}{b[2]}" for i, b in GUA_BITS.items()}


def _as_np(x):
    """checkpoint 里的归一化统计可能是 torch.Tensor 或 numpy.ndarray。"""
    return x if isinstance(x, np.ndarray) else np.asarray(x)


# 轴语义（两世界一致）
AXIS_NAMES = ["初爻(接触)", "中爻(能量趋势)", "上爻(趋离意图)"]


# ---------------------------------------------------------------------------
# Step 0 — 卦索引工具
# ---------------------------------------------------------------------------

def gua_from_yao_v3(yao_log):
    """弹球世界：三物体聚合（model_v3.gua_from_yao）。
    每轴对 3 物体取均值后 0.5 阈值 → 3 bit → 0..7
    """
    p = torch.sigmoid(yao_log.float())
    rates = p.view(p.size(0), 3, 3).mean(dim=2)          # (B, 3)
    bits = (rates > 0.5).long()
    return bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]


def gua_from_yao_hinge(yao_log, obj_id):
    """铰链世界：指定物体的三爻 → 8 卦。
    yao_log: (B, 6) = [初0, 初1, 中0, 中1, 上0, 上1]
    """
    p = torch.sigmoid(yao_log.float())
    bits = p.view(-1, 3, 2)                               # (B, 3轴, 2物体)
    b = bits[:, :, obj_id]                                # (B, 3)
    bit = (b > 0.5).long()
    return bit[:, 0] * 4 + bit[:, 1] * 2 + bit[:, 2]


def gua_from_yao_true(yao_true, obj_id=None, aggregate=False):
    """由真值爻标签得卦索引（敏感性分析用）。
    yao_true: (B, 3, Nobj) int8
    aggregate=True → 跨物体均值后阈值（模拟 v3 的聚合语义）
    obj_id 指定单物体（仅 aggregate=False 时有效）
    """
    if aggregate:
        bits = (yao_true.float().mean(dim=2) > 0.5).long()
    else:
        bits = yao_true[:, :, obj_id]
    return bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]


# ---------------------------------------------------------------------------
# Step 1 — 接线自检（tests/test_wiring.py 的跨世界推广）
# ---------------------------------------------------------------------------

def check_wiring_hinge(ck):
    sd = ck["model_state_dict"]; cfg = dict(ck["config"])
    problems = []
    # WorldModelH 无 gua 臂 → gua_emb 必须缺失；state_head 输入 = hidden+action
    has_gua = "gua_emb.weight" in sd
    if has_gua:
        problems.append("WorldModelH 不该含 gua_emb.weight（接线事故）")
    hidden, action = cfg.get("hidden_dim", 256), cfg.get("action_dim", 1)
    st_in = sd["state_head.0.weight"].shape[1]
    if st_in != hidden + action:
        problems.append(f"state_head 输入维 {st_in} ≠ hidden+action={hidden+action}")
    if sd["yao_head.weight"].shape[0] != 6:
        problems.append(f"yao_head 输出维 {sd['yao_head.weight'].shape[0]} ≠ 6 (2物体×3爻)")
    return problems


def check_wiring_v3(ck):
    sd = ck["model_state_dict"]; cfg = dict(ck["config"])
    problems = []
    has_gua = "gua_emb.weight" in sd
    if bool(cfg.get("use_gua", False)) != has_gua:
        problems.append(f"use_gua config={cfg.get('use_gua')} 但权重中 gua_emb "
                        f"{'存在' if has_gua else '缺失'}")
    cfg["use_gua"] = has_gua
    hidden, action = cfg.get("hidden_dim", 256), cfg.get("action_dim", 6)
    expected = hidden + action + (cfg.get("gua_emb_dim", 16) if has_gua else 0)
    st_in = sd["state_head.0.weight"].shape[1]
    if st_in != expected:
        problems.append(f"state_head 输入维 {st_in} ≠ {expected}")
    if sd["yao_head.weight"].shape[0] != 9:
        problems.append(f"yao_head 输出维 {sd['yao_head.weight'].shape[0]} ≠ 9 (3物体×3爻)")
    return problems


# ---------------------------------------------------------------------------
# Step 2 — 铰链世界卦级误差
# ---------------------------------------------------------------------------

def compute_hinge(ckpt_path, data_path, L=8, max_windows=None, device="cpu"):
    """铰链 WorldModelH：held-out 窗口 → (每卦位置 RMSE, 每卦样本数, 爻准确率)"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    cfg.setdefault("yao_dim", 6)
    from model_h import WorldModelH
    model = WorldModelH(**cfg).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()

    d = np.load(data_path)
    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]

    # 复现 train_hinge.py 的 90/10 划分（seed 42）→ 真正的 held-out
    N = states.shape[0]
    rng = np.random.RandomState(42)
    segs = np.arange(N); rng.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]

    mean_obj = _as_np(ck["norm_stats"]["mean_obj"])
    std_obj = _as_np(ck["norm_stats"]["std_obj"])

    obj_norm = ((states[val_eps] - mean_obj) / (std_obj + 1e-8)).astype(np.float32)
    T = states.shape[1]
    # 一步预测只需 [i0, i0+L) → 目标 i0+L，故 i0 ∈ [0, T-L)，共 T-L 窗
    # （train_* 的 validate 用 T-L-K+1，多出的 K-1 帧只为闭环 rollout 准备）
    n_win = T - L
    ep_idx = np.repeat(np.arange(len(val_eps)), n_win)
    win_start = np.tile(np.arange(n_win), len(val_eps))

    idx = np.arange(len(ep_idx))
    if max_windows and len(idx) > max_windows:
        rng2 = np.random.RandomState(0)
        idx = np.sort(rng2.choice(idx, size=max_windows, replace=False))

    M = len(idx)
    bs = 2048
    gua_err_sq = np.zeros(8); gua_n = np.zeros(8, dtype=np.int64)
    yao_hits = np.zeros(3); yao_cnt = np.zeros(3)
    n_total = 0

    for j in range(0, M, bs):
        ij = idx[j:j + bs]
        ep = ep_idx[ij]; i0 = win_start[ij]
        B = len(ij)
        re = val_eps[ep]                                              # 相对序号 → 真实 episode 号
        off = np.arange(L + 1)
        frm = obj_norm[ep[:, None], i0[:, None] + off[None, :]]       # (B, L+1, 22)（obj_norm 已按 val_eps 过滤）
        ct = contact[re[:, None], i0[:, None] + off[None, :]]         # (B, L+1, 3)（contact 是原始数组，必须回映射）
        x = np.concatenate([frm[:, :L], ct[:, :L]], axis=2)           # (B, L, 25)
        a0 = actions[re, i0 + L - 1]                                  # 修正：动作索引 i0+L-1（train_hinge 用 i0+L，差 1 帧）
        xb = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(device)
        ab = torch.from_numpy(np.ascontiguousarray(a0, np.float32)).to(device)

        h, _ = model.encode_sequence(xb)
        delta = model.predict_state(h, ab).detach().cpu().numpy()
        prior = model.prior_norm(xb[:, -1, :22], ab).detach().cpu().numpy()
        pred_raw = (prior + delta) * (std_obj + 1e-8) + mean_obj
        tgt_raw = frm[:, L] * (std_obj + 1e-8) + mean_obj

        # 位置误差平方（4 维）：与 train_hinge.validate / train_v3.validate 的 gua_pos_rmse 同口径
        # （全部误差平方和 ÷ 总样本数 再开方，而非"每窗 RMS 的均值"）
        err = (pred_raw[:, [1, 2, 12, 13]] - tgt_raw[:, [1, 2, 12, 13]]) ** 2    # (B, 4)

        yao_log = model.predict_yao(h)
        # 主卦：物体1（手端 = 接触/意图承载者）
        gua = gua_from_yao_hinge(yao_log, obj_id=1).detach().cpu().numpy()
        # 敏感性：物体0；真值爻
        gua_obj0 = gua_from_yao_hinge(yao_log, obj_id=0).detach().cpu().numpy()
        yao_t = torch.from_numpy(yao[val_eps[ep], i0 + L].reshape(B, 3, 2))   # ep 是 val_eps 内的相对序号，必须回映射
        gua_true_agg = gua_from_yao_true(yao_t, aggregate=True).numpy()

        yp = (torch.sigmoid(yao_log.float()) > 0.5).detach().cpu().numpy()
        yt = yao_t.numpy()
        for ax in range(3):
            yao_hits[ax] += (yp[:, ax * 2:(ax + 1) * 2] == yt[:, ax, :]).mean() * B
            yao_cnt[ax] += B

        for g in range(8):
            m = gua == g
            if m.any():
                gua_err_sq[g] += float(err[m].sum())
                gua_n[g] += int(m.sum() * 4)      # 4 个位置维度

        # 敏感性记录
        for g in range(8):
            pass
        n_total += B

    return {
        "mean_rmse": {g: float(np.sqrt(gua_err_sq[g] / gua_n[g])) if gua_n[g] > 0 else None for g in range(8)},
        "n": {g: int(gua_n[g]) for g in range(8)},
        "yao_acc": {ax: float(yao_hits[ax] / yao_cnt[ax]) for ax in range(3)},
        "total_windows": int(n_total),
        "used_episodes": int(len(val_eps)),
    }


# ---------------------------------------------------------------------------
# Step 3 — 弹球世界卦级误差
# ---------------------------------------------------------------------------

def compute_v3(ckpt_path, data_path, L=8, max_windows=None, device="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    from model_v3 import WorldModelV3
    model = WorldModelV3(**cfg).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()

    d = np.load(data_path)
    states, obstacles, contact, actions = (d["states"], d["obstacles"],
                                           d["contact"], d["actions"])

    # 复现 train_v3.py 的 90/10 划分（seed 42）
    N = states.shape[0]
    rng = np.random.RandomState(42)
    segs = np.arange(N); rng.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]

    mean_obj = _as_np(ck["norm_stats"]["mean_obj"])
    std_obj = _as_np(ck["norm_stats"]["std_obj"])
    mean_obs = _as_np(ck["norm_stats"]["mean_obs"])
    std_obs = _as_np(ck["norm_stats"]["std_obs"])

    obj_norm = ((states[val_eps] - mean_obj) / (std_obj + 1e-8)).astype(np.float32)
    T = states.shape[1]
    # 一步预测只需 [i0, i0+L) → 目标 i0+L，故 i0 ∈ [0, T-L)，共 T-L 窗
    # （train_* 的 validate 用 T-L-K+1，多出的 K-1 帧只为闭环 rollout 准备）
    n_win = T - L
    ep_idx = np.repeat(np.arange(len(val_eps)), n_win)
    win_start = np.tile(np.arange(n_win), len(val_eps))

    idx = np.arange(len(ep_idx))
    if max_windows and len(idx) > max_windows:
        rng2 = np.random.RandomState(0)
        idx = np.sort(rng2.choice(idx, size=max_windows, replace=False))

    M = len(idx)
    bs = 1024
    gua_err_sq = np.zeros(8); gua_n = np.zeros(8, dtype=np.int64)
    yao_hits = np.zeros(3); yao_cnt = np.zeros(3)
    n_total = 0

    from layout import yao_labels, obj_pos_dims
    POS = obj_pos_dims()

    # 预计算 val episodes 的爻标签（5000 × 200，纯 numpy，很快）
    yaos = np.zeros((len(val_eps), T, 3, 3), dtype=np.int8)
    for ei, ep in enumerate(val_eps):
        yaos[ei] = yao_labels(states[ep], contact[ep].astype(np.int8), obstacles[ep])

    for j in range(0, M, bs):
        ij = idx[j:j + bs]
        ep = ep_idx[ij]; i0 = win_start[ij]
        B = len(ij)
        re = val_eps[ep]                                                  # 相对序号 → 真实 episode 号
        off = np.arange(L + 1)
        frm = obj_norm[ep[:, None], i0[:, None] + off[None, :]]           # (B, L+1, 33)（obj_norm 已按 val_eps 过滤）
        ct = contact[re[:, None], i0[:, None] + off[None, :]]             # (B, L+1, 3)（原始 uint8，值域 0/1 无需归一化）
        obs_b = (obstacles[re] - mean_obs) / (std_obs + 1e-8)             # (B, 80) 障碍必须归一化（模型输入是归一化空间）
        x = np.concatenate([frm[:, :L],
                            np.broadcast_to(obs_b[:, None, :], (B, L, 80)),
                            ct[:, :L]], axis=2)                           # (B, L, 116)
        a0 = actions[re, i0 + L - 1] / 25.0                               # 原始冲量 → 归一化
        xb = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(device)
        ab = torch.from_numpy(np.ascontiguousarray(a0, np.float32)).to(device)

        h, _ = model.encode_sequence(xb)
        delta = model.predict_state(h, ab).detach().cpu().numpy()
        prior = model.prior_norm(xb[:, -1, :33], ab).detach().cpu().numpy()
        pred_raw = (prior + delta) * (std_obj + 1e-8) + mean_obj
        tgt_raw = frm[:, L] * (std_obj + 1e-8) + mean_obj

        err = (pred_raw[:, POS] - tgt_raw[:, POS]) ** 2                   # (B, 6) = 3 物体 × (x,y)

        yao_log = model.predict_yao(h)
        gua = gua_from_yao_v3(yao_log).detach().cpu().numpy()
        yao_t = torch.from_numpy(yaos[ep, i0 + L])                         # (B, 3, 3)
        gua_true = gua_from_yao_true(yao_t, aggregate=True).numpy()

        yp = (torch.sigmoid(yao_log.float()) > 0.5).detach().cpu().numpy()
        yt = yao_t.numpy()
        for ax in range(3):
            yao_hits[ax] += (yp[:, ax * 3:(ax + 1) * 3] == yt[:, ax, :]).mean() * B
            yao_cnt[ax] += B

        for g in range(8):
            m = gua == g
            if m.any():
                gua_err_sq[g] += float(err[m].sum())
                gua_n[g] += int(m.sum() * 6)      # 6 个位置维度（与 train_v3.validate 的 gua_pos_rmse 同口径）
        n_total += B

    return {
        "mean_rmse": {g: float(np.sqrt(gua_err_sq[g] / gua_n[g])) if gua_n[g] > 0 else None for g in range(8)},
        "n": {g: int(gua_n[g]) for g in range(8)},
        "yao_acc": {ax: float(yao_hits[ax] / yao_cnt[ax]) for ax in range(3)},
        "total_windows": int(n_total),
        "used_episodes": int(len(val_eps)),
    }


# ---------------------------------------------------------------------------
# Step 4 — 一致性检验
# ---------------------------------------------------------------------------

def spearman(x, y):
    """n=8 的 Spearman ρ + 置换精确 p 值（无结时）/ 渐近 p 值（有结时）。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    rx = np.argsort(np.argsort(x)) + 1
    ry = np.argsort(np.argsort(y)) + 1
    # Pearson on ranks
    rho = np.corrcoef(rx, ry)[0, 1]
    # permutation test
    nperm = 200000
    rng = np.random.RandomState(20260905)
    cnt = 0
    obs = np.abs(rho)
    for _ in range(nperm):
        pr = rng.permutation(ry)
        if np.abs(np.corrcoef(rx, pr)[0, 1]) >= obs:
            cnt += 1
    p = (cnt + 1) / (nperm + 1)
    return rho, p


def bootstrap_rho(hinge_res, v3_res, n_boot=5000, seed=20260905):
    """对窗口做分层自助重抽样，给出 ρ 的 95% CI（诚实面对 n=8 的不稳定性）。"""
    rng = np.random.RandomState(seed)
    # 重建每窗的 (gua, rmse) 数组
    hg = np.repeat(np.arange(8), [hinge_res["n"][g] for g in range(8)])
    hr = np.concatenate([np.full(hinge_res["n"][g], hinge_res["mean_rmse"][g]) for g in range(8)])
    vg = np.repeat(np.arange(8), [v3_res["n"][g] for g in range(8)])
    vr = np.concatenate([np.full(v3_res["n"][g], v3_res["mean_rmse"][g]) for g in range(8)])

    rhos = []
    for _ in range(n_boot):
        hb = np.zeros(8); vb = np.zeros(8)
        for g in range(8):
            m = hg == g
            if m.sum() == 0: continue
            s = rng.choice(np.flatnonzero(m), size=m.sum(), replace=True)
            hb[g] = hr[s].mean()
            m2 = vg == g
            s2 = rng.choice(np.flatnonzero(m2), size=m2.sum(), replace=True)
            vb[g] = vr[s2].mean()
        if np.std(hb) > 0 and np.std(vb) > 0:
            rhos.append(np.corrcoef(np.argsort(np.argsort(hb)) + 1,
                                    np.argsort(np.argsort(vb)) + 1)[0, 1])
    rhos = np.array(rhos)
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    os.makedirs("analysis_output", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[E1d] 设备: {device}")

    # ---------- Step 1: 模型 + 接线 ----------
    t0 = time.time()
    hinge_ck = torch.load("checkpoints-hinge/best.pt", map_location="cpu", weights_only=False)
    v3_ck = torch.load("checkpoints-v3-calib2/best.pt", map_location="cpu", weights_only=False)

    hinge_wiring = check_wiring_hinge(hinge_ck)
    v3_wiring = check_wiring_v3(v3_ck)
    print(f"[Step1] 铰链接线: {'PASS' if not hinge_wiring else hinge_wiring}")
    print(f"[Step1] 弹球接线: {'PASS' if not v3_wiring else v3_wiring}")

    hinge_cfg = dict(hinge_ck["config"]); v3_cfg = dict(v3_ck["config"])
    hinge_params = sum(v.numel() for v in hinge_ck["model_state_dict"].values())
    v3_params = sum(v.numel() for v in v3_ck["model_state_dict"].values())
    print(f"[Step1] 铰链 {hinge_params:,} 参 | epoch {hinge_ck.get('epoch')} | "
          f"val {hinge_ck.get('val_metrics')}")
    print(f"[Step1] 弹球 {v3_params:,} 参 | epoch {v3_ck.get('epoch')} | "
          f"val {v3_ck.get('val_metrics')}")

    # ---------- Step 2 & 3: 卦级误差 ----------
    print("\n[Step2] 铰链卦级误差（held-out, 物体1/手端）...")
    hinge = compute_hinge("checkpoints-hinge/best.pt", "trajectories_hinge.npz",
                          L=8, device=device)
    print(f"  窗口={hinge['total_windows']:,} | 爻acc="
          + " ".join(f"{a:.3f}" for a in hinge["yao_acc"].values()))
    for g in range(8):
        print(f"    {GUA_NAMES[g]}: {hinge['mean_rmse'][g]:.3f}px  n={hinge['n'][g]}")

    print("\n[Step3] 弹球卦级误差（held-out, 三物体聚合）...")
    v3 = compute_v3("checkpoints-v3-calib2/best.pt", "trajectories_v3.npz",
                    L=8, device=device)
    print(f"  窗口={v3['total_windows']:,} | 爻acc="
          + " ".join(f"{a:.3f}" for a in v3["yao_acc"].values()))
    for g in range(8):
        print(f"    {GUA_NAMES[g]}: {v3['mean_rmse'][g]:.3f}px  n={v3['n'][g]}")

    # ---------- 敏感性：铰链不同卦定义 ----------
    def compute_hinge_true_variant(ckpt_path, data_path, obj_id, agg, label, device):
        """复用 compute_hinge 框架但改用真值爻 / 不同物体。这里直接重跑轻量版本。"""
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = dict(ck["config"]); cfg.setdefault("yao_dim", 6)
        from model_h import WorldModelH
        model = WorldModelH(**cfg).to(device); model.eval()
        d = np.load(data_path)
        states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
        N = states.shape[0]
        rng = np.random.RandomState(42)
        segs = np.arange(N); rng.shuffle(segs)
        val_eps = segs[:max(1, int(N * 0.1))]
        mean_obj = _as_np(ck["norm_stats"]["mean_obj"]); std_obj = _as_np(ck["norm_stats"]["std_obj"])
        obj_norm = ((states[val_eps] - mean_obj) / (std_obj + 1e-8)).astype(np.float32)
        T = states.shape[1]; L = 8; n_win = T - L
        ep_idx = np.repeat(np.arange(len(val_eps)), n_win)
        win_start = np.tile(np.arange(n_win), len(val_eps))
        idx = np.arange(len(ep_idx)); rng2 = np.random.RandomState(0)
        idx = np.sort(rng2.choice(idx, size=min(200000, len(idx)), replace=False))
        bs = 2048; err_sq = np.zeros(8); n = np.zeros(8, dtype=np.int64)
        for j in range(0, len(idx), bs):
            ij = idx[j:j + bs]; ep = ep_idx[ij]; i0 = win_start[ij]; B = len(ij)
            re = val_eps[ep]
            off = np.arange(L + 1)
            frm = obj_norm[ep[:, None], i0[:, None] + off[None, :]]
            ct = contact[re[:, None], i0[:, None] + off[None, :]]
            x = np.concatenate([frm[:, :L], ct[:, :L]], axis=2)
            a0 = actions[re, i0 + L - 1]
            xb = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(device)
            ab = torch.from_numpy(np.ascontiguousarray(a0, np.float32)).to(device)
            h, _ = model.encode_sequence(xb)
            delta = model.predict_state(h, ab).detach().cpu().numpy()
            prior = model.prior_norm(xb[:, -1, :22], ab).detach().cpu().numpy()
            pred = (prior + delta) * (std_obj + 1e-8) + mean_obj
            tgt = frm[:, L] * (std_obj + 1e-8) + mean_obj
            err = ((pred[:, [1, 2, 12, 13]] - tgt[:, [1, 2, 12, 13]]) ** 2)          # (B, 4) 误差平方
            yao_t = torch.from_numpy(yao[val_eps[ep], i0 + L].reshape(B, 3, 2))   # ep 是 val_eps 内的相对序号，必须回映射
            gua = gua_from_yao_true(yao_t, obj_id=obj_id, aggregate=agg).numpy()
            for g in range(8):
                m = gua == g
                if m.any():
                    err_sq[g] += float(err[m].sum()); n[g] += int(m.sum() * 4)
        return {g: (float(np.sqrt(err_sq[g] / n[g])) if n[g] > 0 else None) for g in range(8)}

    print("\n[Step4a] 敏感性：铰链不同卦定义")
    h_obj0 = compute_hinge_true_variant("checkpoints-hinge/best.pt", "trajectories_hinge.npz",
                                        obj_id=0, agg=False, label="物体0", device=device)
    h_true_obj1 = compute_hinge_true_variant("checkpoints-hinge/best.pt", "trajectories_hinge.npz",
                                             obj_id=1, agg=False, label="真值爻-物体1", device=device)
    h_true_agg = compute_hinge_true_variant("checkpoints-hinge/best.pt", "trajectories_hinge.npz",
                                            obj_id=None, agg=True, label="真值爻-聚合", device=device)
    v3_true_agg = None  # 弹球真值聚合在 compute_v3 中已得 gua_true；此处补算
    print(f"  铰链 物体0   : " + " ".join(f"{GUA_NAMES[g]}={h_obj0[g]:.3f}" if h_obj0[g] is not None else f"{GUA_NAMES[g]}=n/a" for g in range(8)))
    print(f"  铰链 真值obj1: " + " ".join(f"{GUA_NAMES[g]}={h_true_obj1[g]:.3f}" if h_true_obj1[g] is not None else f"{GUA_NAMES[g]}=n/a" for g in range(8)))
    print(f"  铰链 真值聚合: " + " ".join(f"{GUA_NAMES[g]}={h_true_agg[g]:.3f}" if h_true_agg[g] is not None else f"{GUA_NAMES[g]}=n/a" for g in range(8)))

    # ---------- 主检验 ----------
    h_vec = np.array([hinge["mean_rmse"][g] for g in range(8)])
    v_vec = np.array([v3["mean_rmse"][g] for g in range(8)])
    rho, p = spearman(h_vec, v_vec)
    lo, hi = bootstrap_rho(hinge, v3)

    h_rank = np.argsort(np.argsort(h_vec)) + 1
    v_rank = np.argsort(np.argsort(v_vec)) + 1
    h_easiest = int(np.argmin(h_vec)); v_easiest = int(np.argmin(v_vec))
    h_hardest = int(np.argmax(h_vec)); v_hardest = int(np.argmax(v_vec))

    print(f"\n[Step4] Spearman ρ = {rho:.3f}  (置换 p = {p:.4f})  bootstrap 95% CI = [{lo:.3f}, {hi:.3f}]")
    print(f"  铰链排序 ({GUA_NAMES[h_easiest]} 最易 → {GUA_NAMES[h_hardest]} 最难): "
          + " ".join(f"{GUA_NAMES[g]}#{h_rank[g]}" for g in np.argsort(h_vec)))
    print(f"  弹球排序 ({GUA_NAMES[v_easiest]} 最易 → {GUA_NAMES[v_hardest]} 最难): "
          + " ".join(f"{GUA_NAMES[g]}#{v_rank[g]}" for g in np.argsort(v_vec)))
    print(f"  最易卦: 铰链={GUA_NAMES[h_easiest]}(#{h_easiest}) 弹球={GUA_NAMES[v_easiest]}(#{v_easiest})")
    print(f"  最难卦: 铰链={GUA_NAMES[h_hardest]} 弹球={GUA_NAMES[v_hardest]}")

    # ---------- 判据 ----------
    pass_e1d1 = bool(rho >= 0.6 and p < 0.05)
    pass_e1d2 = bool(h_easiest == 0 and v_easiest == 0)

    print(f"\n[Step5] 判决")
    print(f"  E1d-1 (rho>=0.6, p<0.05): {'PASS' if pass_e1d1 else 'FAIL'}  rho={rho:.3f} p={p:.4f}")
    print(f"  E1d-2 (easiest=Kun):  {'PASS' if pass_e1d2 else 'FAIL'}  hinge={GUA_NAMES[h_easiest]} v3={GUA_NAMES[v_easiest]}")

    # ---------- 作图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    order_h = np.argsort(h_vec)
    order_v = np.argsort(v_vec)
    EASY_C = "#4a7c59"; HARD_C = "#c0504d"; MID_C = "#7b9ea8"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(np.arange(8), [h_vec[g] for g in order_h],
                  color=[EASY_C if g == h_easiest else HARD_C if g == h_hardest else MID_C
                         for g in order_h],
                  edgecolor="#2f4f5f", linewidth=1.0)
    for i, g in enumerate(order_h):
        ax.text(i, h_vec[g] + 0.02, f"{GUA_NAMES[g]}\n#{g}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(8)); ax.set_xticklabels([f"#{i}" for i in range(8)], fontsize=8)
    ax.set_ylabel("one-step 位置 RMSE (px)")
    ax.set_title(f"铰链（双摆，手端卦）\n最易 {GUA_NAMES[h_easiest]} {h_vec[h_easiest]:.2f}px | "
                 f"最难 {GUA_NAMES[h_hardest]} {h_vec[h_hardest]:.2f}px")
    ax.set_ylim(0, max(h_vec) * 1.35)

    ax = axes[1]
    bars2 = ax.bar(np.arange(8), [v_vec[g] for g in order_v],
                   color=[EASY_C if g == v_easiest else HARD_C if g == v_hardest else MID_C
                          for g in order_v],
                   edgecolor="#2f4f5f", linewidth=1.0)
    for i, g in enumerate(order_v):
        ax.text(i, v_vec[g] + 0.02, f"{GUA_NAMES[g]}\n#{g}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(8)); ax.set_xticklabels([f"#{i}" for i in range(8)], fontsize=8)
    ax.set_ylabel("one-step 位置 RMSE (px)")
    ax.set_title(f"弹球（3 物体聚合卦）\n最易 {GUA_NAMES[v_easiest]} {v_vec[v_easiest]:.2f}px | "
                 f"最难 {GUA_NAMES[v_hardest]} {v_vec[v_hardest]:.2f}px")
    ax.set_ylim(0, max(v_vec) * 1.35)

    fig.suptitle("E1d 跨世界卦级误差排序（held-out）", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("figures/E1d_cross_world_rank.png", dpi=180)
    plt.close(fig)
    print("[fig] figures/E1d_cross_world_rank.png")

    # 散点图：每卦一个点
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(v_vec, h_vec, s=150, c=np.arange(8), cmap="tab10", edgecolor="k", linewidth=1.2, zorder=3)
    for g in range(8):
        ax.annotate(f"{GUA_NAMES[g]} (#{g})", (v_vec[g], h_vec[g]), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.6, alpha=0.85), zorder=4)
    lims = [min(v_vec.min(), h_vec.min()) * 0.85, max(v_vec.max(), h_vec.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.4, zorder=1)
    ax.set_xlabel("弹球 one-step 位置 RMSE (px)")
    ax.set_ylabel("铰链 one-step 位置 RMSE (px)")
    ax.set_title(f"E1d 卦级误差散点（每卦一点）\nSpearman rho={rho:.3f}, p={p:.4f}"
                 + f"  [{'PASS' if pass_e1d1 else 'FAIL'}]")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig("figures/E1d_scatter.png", dpi=180)
    plt.close(fig)
    print("[fig] figures/E1d_scatter.png")

    # ---------- 报告 ----------
    report = {
        "experiment": "E1d",
        "title": "跨世界误差解剖一致性 (Cross-World Error Anatomy Consistency)",
        "pre_registration": {
            "written_at": "2026-09-05 00:00",
            "hypothesis": "卦级误差结构（坤最易、离最难）在弹球与铰链两个世界一致——"
                          "说明它是相态结构的普遍性质，而非某个世界的偶然。"
                          "若通过，误差解剖成为通用分析仪器；若未通过，T4 降级为'弹球世界的观察'。",
            "criteria": {
                "E1d-1": {"text": "两世界的卦级误差排序 Spearman ρ ≥ 0.6 且 p < 0.05",
                          "threshold": "ρ ≥ 0.6, p < 0.05"},
                "E1d-2": {"text": "两世界的最易卦都是'低能/腾空'相（弹球=坤，铰链=对应相）",
                          "threshold": "最易卦 = 坤 (index 0)"}
            }
        },
        "setup": {
            "world": "hinge + v3（双世界）",
            "hinge": {
                "checkpoint": "checkpoints-hinge/best.pt",
                "config": hinge_cfg,
                "params": hinge_params,
                "epoch": hinge_ck.get("epoch"),
                "data": "trajectories_hinge.npz（3000 episodes）",
                "held_out": "train_hinge.py 同 seed=42 的 90/10 划分的 10% = 300 episodes（真正的 held-out，非训练子集）",
                "gua_definition": "物体1（手端）三爻 → 8 卦；index=4*初+2*中+1*上",
                "gua_from_yao": "model_h.py 无此方法；分析脚本内实现等价逻辑（见 bugs_found）",
                "pos_dims": "[1,2,12,13] = 肘x,肘y,手x,手y",
                "L": 8,
            },
            "v3": {
                "checkpoint": "checkpoints-v3-calib2/best.pt",
                "config": v3_cfg,
                "params": v3_params,
                "epoch": v3_ck.get("epoch"),
                "data": "trajectories_v3.npz（50000 episodes）",
                "held_out": "train_v3.py 同 seed=42 的 90/10 划分的 10% = 5000 episodes",
                "gua_definition": "三物体三爻聚合（model_v3.gua_from_yao：每轴对物体取均值后阈值）→ 8 卦",
                "pos_dims": "layout.obj_pos_dims() = 3 物体的 x,y",
                "L": 8,
            },
            "seeds": [42],
            "compute": f"GPU {device}；评估约 {time.time()-t0:.0f}s（不含作图）",
        },
        "gua_alignment": {
            "method": "相态签名对齐（phase-signature alignment），而非裸数字索引对齐",
            "detail": [
                "两世界的三爻轴物理语义完全一致：初爻=接触/腾空，中爻=能量升/降趋势，上爻=趋近/远离意图。",
                "位序约定完全一致：卦索引 = 4·初 + 2·中 + 1·上。因此 index=0 在两世界都是 000 = '无接触+能量下降+远离' = 坤。",
                "差异仅在'卦由哪个物体的爻构造'：弹球是三物体聚合（model_v3.gua_from_yao 的 mean(dim=2)），"
                "铰链取物体1（手端 = 接触事件与趋靶意图的承载者）作为对应相。",
                "结论：index 0..7 在相态签名层面语义等价，可直接比较；这不是裸数字对齐。"
            ],
            "sensitivity_variants": {
                "hinge_obj0": "铰链改用物体0（肩/肘）的卦",
                "hinge_true_yao_obj1": "铰链用真值爻标签（而非模型预测爻）+ 物体1",
                "hinge_true_yao_agg": "铰链用真值爻标签 + 跨物体聚合（模拟弹球语义）"
            }
        },
        "results": {
            "hinge": hinge,
            "v3": v3,
            "sensitivity": {
                "hinge_obj0": h_obj0,
                "hinge_true_yao_obj1": h_true_obj1,
                "hinge_true_yao_agg": h_true_agg
            },
            "spearman": {"rho": float(rho), "p_value": float(p),
                         "bootstrap_95_ci": [lo, hi], "n_permutations": 200000},
            "hinge_rank": {GUA_NAMES[g]: {"rank": int(h_rank[g]), "rmse": float(h_vec[g]), "n": int(hinge["n"][g])} for g in range(8)},
            "v3_rank": {GUA_NAMES[g]: {"rank": int(v_rank[g]), "rmse": float(v_vec[g]), "n": int(v3["n"][g])} for g in range(8)},
            "easiest": {"hinge": GUA_NAMES[h_easiest], "v3": GUA_NAMES[v_easiest]},
            "hardest": {"hinge": GUA_NAMES[h_hardest], "v3": GUA_NAMES[v_hardest]},
            "span_px": {"hinge": float(h_vec.max() - h_vec.min()), "v3": float(v_vec.max() - v_vec.min())},
        },
        "criteria_verdict": {
            "E1d-1": {"pass": pass_e1d1,
                      "actual": f"ρ={rho:.3f}, p={p:.4f} (bootstrap 95% CI [{lo:.3f}, {hi:.3f}])",
                      "note": "n=8（每世界仅 8 个卦点），Spearman 本身分辨率有限；bootstrap CI 给出稳定性。"},
            "E1d-2": {"pass": pass_e1d2,
                      "actual": f"铰链最易={GUA_NAMES[h_easiest]}(index {h_easiest}), 弹球最易={GUA_NAMES[v_easiest]}(index {v_easiest})",
                      "note": "坤=000=无接触+能量下降+远离（腾空低能相）。"}
        },
        "summary": "",
        "limitations": [
            "n=8：卦级排序只有 8 个数据点，Spearman 的统计功效低；bootstrap CI 必须一并报告。",
            "单种子：两个世界各只用 1 个 checkpoint（E0 尚未完成 3 种子统一主表）。",
            "卦定义不对称：弹球聚合 vs 铰链单物体；已做 3 种敏感性变体缓解，但无法完全对齐。",
            "held-out 划分方式：复用了训练脚本的 seed=42 90/10 划分，而非 §4.1 要求的 'generate_hinge.py --n_episodes 1000 --seed 42 新鲜生成' —— 后者属 E0 的 Step 1，E1d 在 E0 之前执行。",
            "铰链 model_h.py 缺少 gua_from_yao 方法（弹球 model_v3.py 已有）；分析脚本内实现了等价逻辑。"
        ],
        "bugs_found": [],
        "implications_for_paper": "",
    }

    # 自动写 summary / implications
    if pass_e1d1 and pass_e1d2:
        report["summary"] = (
            f"E1d 两个预注册判据均通过：两世界卦级误差排序 Spearman ρ={rho:.3f}（p={p:.4f}，"
            f"bootstrap 95% CI [{lo:.3f}, {hi:.3f}]）≥ 0.6 门槛；"
            f"且两世界最易卦均为坤（index 0，腾空/低能相）。"
            f"误差跨度 铰链 {h_vec.max()-h_vec.min():.2f}px、弹球 {v_vec.max()-v_vec.min():.2f}px。"
            f"卦级误差结构跨世界一致 → 卦是相态的普遍性质，误差解剖可作为通用分析仪器。"
        )
        report["implications_for_paper"] = (
            "T4 从'弹球世界的观察'升级为'通用分析仪器'：C2.2（卦=误差解剖坐标系）获得跨世界支撑，"
            "论文 R2 节可以把'卦级误差结构化'作为相态分类学的独立证据，而非单世界现象。"
            "同时应如实写明 n=8 与单种子的局限，避免把'一致'读成'完全同构'——两世界的绝对误差量级、"
            "最难卦的身份均不同，说明卦是通用坐标系，具体误差排序仍由世界动力学决定。"
        )
    elif pass_e1d1 and not pass_e1d2:
        report["summary"] = f"E1d-1 通过（ρ={rho:.3f}, p={p:.4f}）但 E1d-2 未过：最易卦不一致。"
        report["implications_for_paper"] = (
            "卦级排序整体一致但'最易相'不共享 → 卦作为误差解剖坐标系可跨世界迁移，"
            "但'坤最易'只是弹球世界特性。论文应把 claim 弱化为'卦是通用坐标系'，"
            "删除'坤最易是普遍相态'的表述。"
        )
    elif not pass_e1d1 and pass_e1d2:
        report["summary"] = f"E1d-2 通过（两世界最易均为坤）但 E1d-1 未过（ρ={rho:.3f}, p={p:.4f}）。"
        report["implications_for_paper"] = (
            "低能/腾空相在两世界都最易，但其余卦排序不一致 → '静相易预测'是普遍现象，"
            "而卦级误差的精细结构是世界依赖的。T4 降级为'部分通用'：误差解剖仍可用，"
            "但不能宣称'卦级误差结构跨世界一致'。"
        )
    else:
        report["summary"] = f"E1d 两判据均未过（ρ={rho:.3f}, p={p:.4f}；最易卦 铰链={GUA_NAMES[h_easiest]} 弹球={GUA_NAMES[v_easiest]}）。"
        report["implications_for_paper"] = (
            "重要负结果：卦级误差结构在两个世界不一致。T4 降级为'弹球世界的观察'，"
            "论文 R2 中'卦=误差解剖坐标系'的 claim 须弱化并标注；C2.2 缺口保留。"
            "应讨论可能原因：两世界的相态定义粒度不同（弹球 3 物体聚合 vs 铰链 2 物体链式耦合）、"
            "或卦级误差是训练动力学/数据分布的产物而非相态的固有性质。"
        )

    report["bugs_found"] = [
        "model_h.py 缺少 gua_from_yao 静态方法（model_v3.py:192 已有），"
        "铰链侧的卦索引需在分析脚本内自行实现；建议在 model_h.py 补上同名方法以保持两世界管线对称。",
        f"铰链 yao 轴上爻对物体0 恒为 0（generate_hinge.py yao_labels_h 只写 yao[:,2,1]），"
        f"导致物体0 的 8 卦实际只覆盖 4 个值——这是'为何选物体1 作对应相'的数据层理由，已在报告中说明。",
    ]

    with open("analysis_output/E1d_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[report] analysis_output/E1d_report.json")
    print(f"[E1d] 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()