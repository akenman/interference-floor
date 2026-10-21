# -*- coding: utf-8 -*-
"""SLOT-1 解析版：弹球世界先验阶梯（0 参数免训练；判据以预注册为准）。

预注册：experiments/prereg_S1_ball_ladder.json
四档解析先验（0 参数，numpy）：
  parabola005  layout.physics_prior 逐位复用（现行 legacy，闭式连续精确解）
  euler005     半隐式欧拉 @0.05×1（恒力 a=g+F/m）
  rk4005       RK4 @0.05×1（连续精确解，含力位移项）
  euler001x5   半隐式欧拉 @0.01×5（=引擎复现：pymunk 逐子步恒力，源码 L180-184）
引擎对拍解析：引擎位置增益 15·a·dt²，连续精确 12.5·a·dt²，SI@0.05 25·a·dt²
  ⇒ 对引擎差 2.5/−2.5/+10 个 a·dt² = 0.225/0.225/0.9 px @a=900（T1 的 0.23px）。

切片（预注册）：FF（自由飞行转移，cap 500k）/ FF-0（零动作子集）/ closed-loop
（连续 6 帧无接触起点，cap 50k）/ mixed（30k 随机窗口，只报告）。
动作对齐：A[t+1] 驱动 t→t+1（源码 L211-214，与铰链管线同约定）。

用法：python tmp_s1_ball_ladder.py [--max-ff 500000]
输出：控制台判决 + analysis_output/S1_ball_ladder.json + figures/S1_ball_ladder.png
"""
import argparse
import io
import json
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from layout import NUM_OBJ, OBJ_DIMS, DT_V3, PHYSICS_DT, physics_prior  # noqa: E402

G = -900.0
TAU = DT_V3                    # 0.05
DT = PHYSICS_DT                # 0.01
POS_DIMS = [o * 11 + 1 for o in range(NUM_OBJ)] + [o * 11 + 2 for o in range(NUM_OBJ)]
TIERS = ["parabola005", "euler005", "rk4005", "euler001x5"]
# 预注册解析预测（FF-0 纯重力，a=900：per-y-dim 0.225/0.9 px → pos_rmse = /√2）
PRED_FF0 = {"parabola005": 0.225 / np.sqrt(2), "euler005": 0.9 / np.sqrt(2),
            "rk4005": 0.225 / np.sqrt(2), "euler001x5": 0.0}


def masses_of(x0):
    """(M,33) → (M,3) 每物体质量（legacy 同规则：≤1e-3 视为 1）。"""
    m = x0[:, [o * 11 + 10 for o in range(NUM_OBJ)]].copy()
    return np.where(m > 1e-3, m, 1.0)


def step_euler_expl(x0, acts, dt, n_sub):
    """显式欧拉 n_sub 子步（x += v·dt 先于 v += a·dt）——数据实测的引擎形式：
    真自由飞行转移 dy−vy·τ = 10·a·dt²（=Σ_{k=0..4}k·a·dt²），非半隐式的 15·a·dt²。
    （协议修正依据见 DEEP_DIVE 10.19.23；chipmunk 6.x 位置先积分、速度后更新。）"""
    m = masses_of(x0)                                    # (M,3)
    out = x0.copy()
    pos = x0[:, [o * 11 + f for o in range(NUM_OBJ) for f in (1, 2)]].copy()   # (M,6)
    vel = x0[:, [o * 11 + f for o in range(NUM_OBJ) for f in (5, 6)]].copy()   # (M,6)
    acc = np.empty_like(vel)
    for o in range(NUM_OBJ):
        acc[:, 2 * o] = acts[:, 2 * o] / (m[:, o] * TAU)
        acc[:, 2 * o + 1] = acts[:, 2 * o + 1] / (m[:, o] * TAU) + G
    for _ in range(n_sub):
        pos += vel * dt
        vel += acc * dt
    for o in range(NUM_OBJ):
        b = o * 11
        out[:, b + 1] = pos[:, 2 * o]
        out[:, b + 2] = pos[:, 2 * o + 1]
        out[:, b + 5] = vel[:, 2 * o]
        out[:, b + 6] = vel[:, 2 * o + 1]
        # 角：无力矩 → ω 常数，θ' = θ + ω·τ（与 legacy 同式）
        th = np.arctan2(x0[:, b + 3], x0[:, b + 4]) + x0[:, b + 7] * TAU
        out[:, b + 3] = np.sin(th)
        out[:, b + 4] = np.cos(th)
    return out


def step_rk4(x0, acts):
    """RK4 @0.05×1，双积分器常加速（解析上=连续精确，含力位移项）。"""
    M = x0.shape[0]
    m = masses_of(x0)
    out = x0.copy()
    pos = x0[:, [o * 11 + f for o in range(NUM_OBJ) for f in (1, 2)]].copy()
    vel = x0[:, [o * 11 + f for o in range(NUM_OBJ) for f in (5, 6)]].copy()
    acc = np.empty_like(vel)
    for o in range(NUM_OBJ):
        acc[:, 2 * o] = acts[:, 2 * o] / (m[:, o] * TAU)
        acc[:, 2 * o + 1] = acts[:, 2 * o + 1] / (m[:, o] * TAU) + G
    k1v = acc
    k1x = vel
    k2v = acc
    k2x = vel + 0.5 * TAU * k1v
    k3v = acc
    k3x = vel + 0.5 * TAU * k2v
    k4v = acc
    k4x = vel + TAU * k3v
    pos = pos + TAU * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
    vel = vel + TAU * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
    for o in range(NUM_OBJ):
        b = o * 11
        out[:, b + 1] = pos[:, 2 * o]
        out[:, b + 2] = pos[:, 2 * o + 1]
        out[:, b + 5] = vel[:, 2 * o]
        out[:, b + 6] = vel[:, 2 * o + 1]
        th = np.arctan2(x0[:, b + 3], x0[:, b + 4]) + x0[:, b + 7] * TAU
        out[:, b + 3] = np.sin(th)
        out[:, b + 4] = np.cos(th)
    return out


def tier_predict(tier, x0, acts):
    if tier == "parabola005":
        return physics_prior(x0, acts, tau=TAU, g=G)
    if tier == "euler005":
        return step_euler_expl(x0, acts, TAU, 1)     # 同族粗档：显式 @0.05×1
    if tier == "rk4005":
        return step_rk4(x0, acts)
    if tier == "euler001x5":
        return step_euler_expl(x0, acts, DT, 5)      # =引擎复现（显式 @0.01×5）
    raise KeyError(tier)


def pos_rmse(pred, tgt):
    err = (pred[:, POS_DIMS] - tgt[:, POS_DIMS]) ** 2
    return float(np.sqrt(err.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ff", type=int, default=500_000)
    ap.add_argument("--max-cl", type=int, default=50_000)
    args = ap.parse_args()

    print("载入 trajectories_v3.npz …", flush=True)
    d = np.load("trajectories_v3.npz")
    S, A, C = d["states"], d["actions"], d["contact"]
    N, T, _ = S.shape
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    val_eps = perm[:N // 10]                     # held-out 10%（与 e0_common 同构）
    print(f"episodes={N} held-out={len(val_eps)}", flush=True)

    # ---------- 转移级数据清洁过滤（协议修正 #2） ----------
    # metadata known_approximation：静态压紧接触可能漏记 ⇒ 贴地/滑行物体混入 FF。
    # 引擎速度关系 |Δv − (I/m + g)·τ| < 1 px/s 逐物体成立者才算干净弹道转移。
    eps_idx = np.repeat(val_eps, T - 1)
    t_idx = np.tile(np.arange(T - 1), len(val_eps))
    cf = (C[val_eps, :-1].max(axis=2) == 0).reshape(-1) & \
         (C[val_eps, 1:].max(axis=2) == 0).reshape(-1)
    vel_dims = [o * 11 + f for o in range(NUM_OBJ) for f in (5, 6)]
    vel_all = S[:, :, vel_dims]                       # (N,T,6)
    dvy = vel_all[:, 1:, :] - vel_all[:, :-1, :]      # (N,T-1,6)
    mdims = [o * 11 + 10 for o in range(NUM_OBJ)]
    m_all = np.where(S[:, :-1, mdims] > 1e-3, S[:, :-1, mdims], 1.0)   # (N,T-1,3) 转移起点质量
    a_pred = np.empty((N, T - 1, 6), dtype=np.float64)
    for o in range(NUM_OBJ):
        m_o = m_all[:, :, o]                             # (N,T-1) 转移起点质量
        a_pred[:, :, 2 * o] = A[:, 1:, 2 * o] / m_o
        a_pred[:, :, 2 * o + 1] = A[:, 1:, 2 * o + 1] / m_o + G
    clean_grid = (np.abs(dvy - a_pred * TAU).max(axis=2) < 1.0)   # (N,T-1)
    del dvy, a_pred, vel_all

    # ---------- FF 转移索引 ----------
    ff_ep, ff_t = eps_idx[cf], t_idx[cf]
    ff_clean = clean_grid[ff_ep, ff_t]
    print(f"FF 转移 {len(ff_t):,}（其中速度关系干净 {int(ff_clean.sum()):,}）", flush=True)
    if len(ff_t) > args.max_ff:
        sel = rng.choice(len(ff_t), size=args.max_ff, replace=False)
        ff_ep, ff_t, ff_clean = ff_ep[sel], ff_t[sel], ff_clean[sel]
        print(f"  子采样至 {len(ff_t):,}", flush=True)
    acts_ff = A[ff_ep, ff_t + 1]                 # A[t+1] 驱动 t→t+1
    ff0 = np.abs(acts_ff).max(axis=1) < 1e-6
    print(f"FF-0（零动作）{int(ff0.sum()):,}（干净 {int((ff0 & ff_clean).sum()):,}）",
          flush=True)

    x0 = S[ff_ep, ff_t]
    tgt = S[ff_ep, ff_t + 1]
    one = {}
    for tier in TIERS:
        pred = tier_predict(tier, x0, acts_ff)
        one[tier] = dict(ff=pos_rmse(pred, tgt),
                         ff_clean=pos_rmse(pred[ff_clean], tgt[ff_clean]),
                         ff0=pos_rmse(pred[ff0], tgt[ff0]) if ff0.sum() else float("nan"),
                         ff0_clean=pos_rmse(pred[ff0 & ff_clean], tgt[ff0 & ff_clean])
                         if (ff0 & ff_clean).sum() else float("nan"))
        print(f"  [{tier}] FF={one[tier]['ff']:.4f}  FF-clean={one[tier]['ff_clean']:.4f}"
              f"  FF-0={one[tier]['ff0']:.4f}  FF-0-clean={one[tier]['ff0_clean']:.4f}"
              f"  (FF-0-clean 理论 {PRED_FF0[tier]:.4f})", flush=True)

    # ---------- 闭环 ----------
    ep_r = np.repeat(val_eps, T - 5)
    t_r = np.tile(np.arange(T - 5), len(val_eps))
    c6 = np.stack([C[val_eps, k:T - 5 + k].max(axis=2) == 0 for k in range(6)])
    cl5 = np.stack([clean_grid[val_eps, k:T - 5 + k] for k in range(5)]).all(axis=0)
    cl_mask = ((c6.sum(axis=0) == 6) & cl5).reshape(-1)
    cl_ep, cl_t = ep_r[cl_mask], t_r[cl_mask]
    if len(cl_t) > args.max_cl:
        sel = rng.choice(len(cl_t), size=args.max_cl, replace=False)
        cl_ep, cl_t = cl_ep[sel], cl_t[sel]
    print(f"闭环起点 {len(cl_t):,}", flush=True)
    roll = {t: {} for t in TIERS}
    for tier in TIERS:
        cur = S[cl_ep, cl_t]
        errs = [[] for _ in range(5)]
        for k in range(5):
            cur = tier_predict(tier, cur, A[cl_ep, cl_t + 1 + k])
            errs[k].append(pos_rmse(cur, S[cl_ep, cl_t + 1 + k]))
        for k in range(5):
            roll[tier][f"k={k + 1}"] = errs[k][0]
        print(f"  [{tier}] " + "  ".join(f"k{k + 1}={errs[k][0]:.3f}" for k in range(5)),
              flush=True)

    # ---------- mixed（只报告） ----------
    mix_ep = np.repeat(val_eps, T - 1)
    mix_t = np.tile(np.arange(T - 1), len(val_eps))
    sel = rng.choice(len(mix_t), size=30_000, replace=False)
    mix_ep, mix_t = mix_ep[sel], mix_t[sel]
    mixed = {}
    for tier in TIERS:
        pred = tier_predict(tier, S[mix_ep, mix_t], A[mix_ep, mix_t + 1])
        mixed[tier] = pos_rmse(pred, S[mix_ep, mix_t + 1])
    print("mixed（含接触，只报告）: " +
          "  ".join(f"{t}={mixed[t]:.3f}" for t in TIERS), flush=True)

    # ---------- 判决（预注册顺序；干净切片，协议修正见 DEEP_DIVE 10.19.23） ----------
    out = {"prereg": "experiments/prereg_S1_ball_ladder.json",
           "n_ff": int(len(ff_t)), "n_ff_clean": int(ff_clean.sum()),
           "n_ff0": int(ff0.sum()), "n_ff0_clean": int((ff0 & ff_clean).sum()),
           "n_closed_loop": int(len(cl_t)),
           "one_step": one, "rollout": roll,
           "mixed_report_only": mixed, "pred_ff0": PRED_FF0,
           "protocol_fix": "引擎实测为显式欧拉形式（dy−vy·τ=10·a·dt²），四档实现按此修正；"
                           "FF 切片加速度关系清洁过滤（漏记持续接触，metadata 已声明）"}
    v = {}
    v["S1-1"] = one["euler001x5"]["ff_clean"] <= 0.01
    v["S1-2"] = (one["euler001x5"]["ff_clean"] < min(one["parabola005"]["ff_clean"],
                                                     one["rk4005"]["ff_clean"])
                 < one["euler005"]["ff_clean"]
                 and one["parabola005"]["ff_clean"] < one["euler005"]["ff_clean"]
                 and abs(one["parabola005"]["ff0_clean"]
                         - one["rk4005"]["ff0_clean"]) <= 0.02)
    v["S1-3"] = ((ff0 & ff_clean).sum() >= 5000
                 and 0.12 <= one["parabola005"]["ff0_clean"] <= 0.20
                 and 0.12 <= one["rk4005"]["ff0_clean"] <= 0.20
                 and 0.50 <= one["euler005"]["ff0_clean"] <= 0.80)
    v["S1-4"] = (roll["euler001x5"]["k=5"] < min(roll["parabola005"]["k=5"],
                                                 roll["rk4005"]["k=5"])
                 and roll["euler005"]["k=5"] > max(roll["parabola005"]["k=5"],
                                                   roll["rk4005"]["k=5"]))
    out["verdicts"] = v
    print("\n=== 判决（预注册顺序，干净切片）===")
    print(f"  S1-1 引擎对拍 (≤0.01px): {'[OK]' if v['S1-1'] else '[FAIL]'}"
          f"  ({one['euler001x5']['ff_clean']:.5f}px)")
    print(f"  S1-2 计算类距离排序:     {'[OK]' if v['S1-2'] else '[FAIL]'}")
    print(f"  S1-3 FF-0-clean 量值命中: {'[OK]' if v['S1-3'] else '[FAIL]'}"
          f"  (n={int((ff0 & ff_clean).sum())})")
    print(f"  S1-4 闭环同序:           {'[OK]' if v['S1-4'] else '[FAIL]'}")
    print(f"  总判决: {'4/4 —— 地板定律跨引擎成立' if all(v.values()) else '见上'}")

    with open("analysis_output/S1_ball_ladder.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/S1_ball_ladder.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        xs = np.arange(len(TIERS))
        for ax, (tag, lab) in zip(axes, (("ff0", "one-step, free-flight no-action (px)"),
                                         ("k=5", "closed-loop k=5, free-flight (px)"))):
            vals = [(one[t][tag] if tag in one[t] else roll[t][tag]) for t in TIERS]
            ax.bar(xs, vals, color="#4C72B0")
            ax.set_xticks(xs)
            ax.set_xticklabels(TIERS, rotation=20, fontsize=8)
            ax.set_title(lab)
            if tag == "ff0":
                ax.axhline(0.159, ls="--", c="r", lw=0.8, label="analytic 0.159")
                ax.axhline(0.636, ls=":", c="g", lw=0.8, label="analytic 0.636")
                ax.legend(fontsize=7)
        fig.suptitle("SLOT-1: ball-world prior ladder (0-param, engine = semi-implicit @0.01x5)")
        fig.tight_layout()
        fig.savefig("figures/S1_ball_ladder.png", dpi=150)
        print("[save] figures/S1_ball_ladder.png")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 出图失败（不影响判决）: {e}")


if __name__ == "__main__":
    main()
