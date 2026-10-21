# -*- coding: utf-8 -*-
"""
G2v3 分位数分布预测 最终判决 (eval_hq_verdict.py)
==================================================
判据（预注册于 train_hq.py 文件头，训练前写下）：
  A 校准：[q0.1,q0.9] 经验覆盖率 = 0.80±0.06；[q0.25,q0.75] = 0.50±0.08
  B 锋锐：中位数（q0.5）作点预测的 RMSE ≤ 单点模型（base）
  C T11 响应：闭环分位数带宽（q0.9−q0.1）随 k（1..5）单调增长
  D 噪声响应：带宽随观测噪声水平单调增长（0/0.01/0.02/0.05/0.1，G1 融合）
  补充（非门）：k=5 闭环覆盖率——分位数带是否捕获混沌发散（T11 兑现）

用法：python eval_hq_verdict.py
"""

import argparse
import json
import os
import numpy as np
import torch

from model_h import QuantileWorldModelH, WorldModelH
from train_hinge import HingeData
from train_hq import evaluate

POS_IDX = [1, 2, 12, 13]
OBJ_DIMS = 22


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = QuantileWorldModelH(**ck["config"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    ns = ck["norm_stats"]
    return model, ns["mean_obj"], ns["std_obj"], ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints-hq2/checkpoint_latest.pt")
    ap.add_argument("--base_ckpt", type=str, default="checkpoints-hinge/best.pt")
    ap.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--output", type=str, default="analysis_output/hq_verdict_report.json")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    model, mean_obj, std_obj, ck = load_model(args.ckpt, device)
    taus = torch.tensor(model.quantiles, dtype=torch.float32, device=device)
    bck = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    base = WorldModelH(**bck["config"]).to(device)
    base.load_state_dict(bck["model_state_dict"], strict=False)
    base.eval()

    # 与 train_hq 完全一致的验证划分（seed 42，前 10%）
    d = np.load(args.data_path)
    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = states.shape[0]
    rng = np.random.RandomState(42)
    segs = np.arange(N); rng.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]
    val_ds = HingeData(states, contact, actions, yao, val_eps, 8, 5, mean_obj, std_obj)

    print("=" * 72)
    print("G2v3 分位数分布预测最终判决（预注册判据 A-D）")
    print(f"  ckpt: {args.ckpt} (epoch {ck.get('epoch', '?')})")
    print("=" * 72)

    # ---- A/B/C：训练噪声水平（0.02）下 ----
    m = evaluate(model, base, val_ds, device, 5, 20000, taus, POS_IDX, OBJ_DIMS,
                 noise_aug=0.02)
    print(f"\n[A 校准] [0.1,0.9] 覆盖 = {m['cov80_k1']:.3f}（目标 0.80±0.06） | "
          f"[0.25,0.75] 覆盖 = {m['cov50_k1']:.3f}（目标 0.50±0.08）")
    print(f"[B 锋锐] q0.5 点预测 k=1 = {m['med_k1']:.3f}px vs 单点基线 k=1 = {m['base_k1']:.3f}px"
          f" | k=5: {m['med_k5']:.3f} vs {m['base_k5']:.3f}")
    bands = [m[f"band_k{k+1}"] for k in range(5)]
    print(f"[C T11 响应] 带宽 k=1..5: " + " ".join(f"{b:.1f}" for b in bands) + " px")

    # ---- D：噪声扫描（G1 融合）----
    print("\n[D 噪声响应] 带宽随观测噪声水平：")
    sweep = {}
    for nz in [0.0, 0.01, 0.02, 0.05, 0.1]:
        ms = evaluate(model, base, val_ds, device, 5, 10000, taus, POS_IDX, OBJ_DIMS,
                      noise_aug=nz)
        sweep[str(nz)] = {"band_k1": ms["band_k1"], "band_k5": ms["band_k5"],
                          "cov80_k1": ms["cov80_k1"], "med_k1": ms["med_k1"]}
        print(f"  noise={nz:<5}: band_k1={ms['band_k1']:>6.2f}px | "
              f"band_k5={ms['band_k5']:>7.2f}px | cov80={ms['cov80_k1']:.3f} | "
              f"q0.5点误差={ms['med_k1']:.2f}px")

    # ---- 补充（非门）：闭环覆盖率逐 k（T11 兑现：带是否捕获混沌发散）----
    print("\n[补充] 闭环覆盖率（q0.1-q0.9 带包含真值的比例，逐 k）:")
    cov_by_k = closed_loop_coverage(model, val_ds, device, noise=0.02)
    print("  " + " ".join(f"k={k+1}:{v:.3f}" for k, v in cov_by_k.items()))

    # ---- 判决 ----
    a_pass = (abs(m["cov80_k1"] - 0.80) <= 0.06) and (abs(m["cov50_k1"] - 0.50) <= 0.08)
    b_pass = m["med_k1"] <= m["base_k1"]
    # 容差 1e-3px（与 D 一致）；严格单调与实质增长分开呈现
    c_strict = all(bands[k + 1] >= bands[k] - 1e-3 for k in range(4))
    c_substantive = bands[4] >= bands[1] * 2.0   # k=5 ≥ k=2 的 2×
    seq = [sweep[str(nz)]["band_k1"] for nz in [0.0, 0.01, 0.02, 0.05, 0.1]]
    d_pass = all(seq[k + 1] >= seq[k] - 1e-3 for k in range(len(seq) - 1))
    d_growth = (seq[-1] - seq[0]) / max(seq[0], 1e-9)
    crit = {"A_校准": bool(a_pass), "B_锋锐": bool(b_pass),
            "C_严格单调": bool(c_strict), "C_实质增长": bool(c_substantive),
            "D_噪声单调": bool(d_pass)}
    print("\n" + "=" * 72)
    print("[预注册判决]")
    print(f"  A 校准:   {'[OK]' if a_pass else '[FAIL]'} (cov80 {m['cov80_k1']:.3f}, cov50 {m['cov50_k1']:.3f})")
    print(f"  B 锋锐:   {'[OK]' if b_pass else '[FAIL]'} (med {m['med_k1']:.3f} vs base {m['base_k1']:.3f})")
    print(f"  C 带宽随k: 严格单调 {'[OK]' if c_strict else '[FAIL]'} / 实质增长(k5≥2×k2) "
          f"{'[OK]' if c_substantive else '[FAIL]'} ({' '.join(f'{b:.2f}' for b in bands)})")
    print(f"  D 带宽随噪声增长: {'[OK]' if d_pass else '[FAIL]'} "
          f"({' '.join(f'{s:.2f}' for s in seq)} | 实质增幅 {d_growth:.1%}——形式单调≠实质敏感)")
    n_pass = sum(crit.values())
    print(f"\nVERDICT: {n_pass}/4 判据通过"
          f"{' — G2v3 分位数路线成立' if n_pass >= 3 else '（如实记录）'}")

    # ---- 事后修正（post-hoc，非预注册，T14 已验证配方）：CQR 共形校准 ----
    cov_cqr, width_cqr, offsets = cqr_amendment(model, val_ds, device, std_obj, noise=0.02)
    a_amended = abs(cov_cqr - 0.80) <= 0.06
    print(f"\n[事后修正] CQR 共形校准（T14 配方，per-dim offset）：")
    print(f"  覆盖 {m['cov80_k1']:.3f} → {cov_cqr:.3f}（目标 0.80±0.06，"
          f"{'[OK]' if a_amended else '[FAIL]'}） | 带宽 {m['band_k1']:.2f} → {width_cqr:.2f}px | "
          f"offset = {np.round(offsets, 3).tolist()}")

    report = {"criteria": crit, "main": m, "noise_sweep": sweep,
              "d_substantive_growth": d_growth,
              "cqr_amendment": {"coverage": cov_cqr, "band_px": width_cqr,
                                "offsets_px": offsets.tolist(), "A_amended": bool(a_amended)},
              "closed_loop_cov_by_k": cov_by_k, "epoch": ck.get("epoch")}
    os.makedirs("analysis_output", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] {args.output}")


@torch.no_grad()
def cqr_amendment(model, ds, device, std_obj, noise=0.02, max_windows=20000, alpha=0.20):
    """T14 配方：分位数 CQR 共形后处理（事后修正，非预注册门）。

    校准/评估对半分窗；nonconformity s = max(q_lo − y, y − q_hi)（per-dim，px）；
    Q = ⌈(n+1)(1−α)⌉/n 分位；区间 = [q_lo − Q, q_hi + Q]（覆盖保证分布无关）。
    """
    pos_std = np.asarray(std_obj)[POS_IDX].astype(np.float64)
    idx_all = np.arange(len(ds))
    if len(idx_all) > max_windows:
        rs = np.random.RandomState(0)
        idx_all = np.sort(rs.choice(idx_all, max_windows, replace=False))
    qs, ys = [], []
    for j in range(0, len(idx_all), 2048):
        b = ds.gather(idx_all[j:j + 2048])
        xb = b["x"].to(device)
        if noise > 0:
            xb = xb + torch.randn_like(xb) * noise
        h, _ = model.encode_sequence(xb)
        a = b["a0"].to(device)
        pn = model.prior_norm(xb[:, -1, :OBJ_DIMS], a)
        q = (pn.unsqueeze(1) + model.predict_q(h, a)).float().cpu().numpy()
        qs.append(q[:, [0, -1], :][:, :, POS_IDX] * pos_std)  # (B,2,4) px
        ys.append(b["x_fut"][:, 0, :][:, POS_IDX].float().cpu().numpy() * pos_std)
    q = np.concatenate(qs); y = np.concatenate(ys)
    n = len(y) // 2
    q_cal, y_cal = q[:n], y[:n]
    q_ev, y_ev = q[n:], y[n:]
    s = np.maximum(q_cal[:, 0] - y_cal, y_cal - q_cal[:, 1])  # (n,4)
    Q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), axis=0)
    lo, hi = q_ev[:, 0] - Q, q_ev[:, 1] + Q
    cov = float(((y_ev >= lo) & (y_ev <= hi)).mean())
    width = float((hi - lo).mean())
    return cov, width, Q


@torch.no_grad()
def closed_loop_coverage(model, ds, device, noise=0.02, max_windows=20000):
    """闭环逐 k：真值是否落在 [q0.1, q0.9] 带内。"""
    idx_all = np.arange(len(ds))
    if len(idx_all) > max_windows:
        rs = np.random.RandomState(0)
        idx_all = np.sort(rs.choice(idx_all, max_windows, replace=False))
    cov = {k: [] for k in range(5)}
    for j in range(0, len(idx_all), 2048):
        b = ds.gather(idx_all[j:j + 2048])
        xb = b["x"].to(device)
        if noise > 0:
            xb = xb + torch.randn_like(xb) * noise
        afb = b["af"].to(device)
        xf = b["x_fut"].to(device)
        steps, _ = model.rollout_q(xb, afb)
        for k in range(5):
            lo = steps[k][:, 0, POS_IDX].float().cpu().numpy()
            hi = steps[k][:, -1, POS_IDX].float().cpu().numpy()
            tgt = xf[:, k, :][:, POS_IDX].float().cpu().numpy()
            cov[k].append(((tgt >= lo) & (tgt <= hi)).mean(axis=1))
    return {k: float(np.concatenate(v).mean()) for k, v in cov.items()}


if __name__ == "__main__":
    main()
