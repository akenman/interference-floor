# -*- coding: utf-8 -*-
"""先验阶梯零成本试跑（审计 + P0-a 预演）。

三件事一次做完：
 1. 对齐颗粒度：比对三处先验实现（layout_h.acrobot_step / e0_common._prior_np /
    model_h.prior_step_torch）在相同输入下的差异；
 2. 给每一档先验的**真实**单步与闭环 k=1..5 误差（同 e0_common.validate 的
    POS_IDX + feat_weights 口径，同一 val 划分，同 30k 窗口抽样）；
 3. 判定「精确积分器能否救闭环」：数据生成的 τ 是**每 0.01s 物理步**取值
    （generate_hinge.make_tau_schedule），而动作标签每 0.05s 才记录一次 →
    区间内 τ 变化是不可观测输入，任何先验都有不可约下限。按 Δa≠0 / Δa=0
    分层统计即可证明该下限是否存在。

用法：python tmp_prior_ladder_pilot.py
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from layout_h import (OBJ_DIMS, ACTION_DIM, PHYSICS_DT, SAMPLE_INTERVAL, DT,
                      TAU_MAX, L1, L2, M1, M2, G, obj_base, F as LF,
                      feat_weights, acrobot_rk4)

POS_IDX = [1, 2, 12, 13]
L, K, N_EP_VAL, N_WIN = 8, 5, 300, 30000
B1, B2 = obj_base(0), obj_base(1)
GP = -G


# ---------------- 先验：向量化 numpy 实现（与 model_h.prior_step_torch 同构） ----------------
def prior_euler(raw, a_norm, dt=DT, n_sub=1):
    """半隐式欧拉。n_sub=1,dt=0.05 → E0 现行；n_sub=5,dt=0.01 → T19 M-F 细化版。"""
    th1 = np.arctan2(raw[:, B1 + LF["sin_t"]], raw[:, B1 + LF["cos_t"]])
    th2 = np.arctan2(raw[:, B2 + LF["sin_t"]], raw[:, B2 + LF["cos_t"]])
    w1 = raw[:, B1 + LF["w"]].copy()
    w2 = raw[:, B2 + LF["w"]].copy()
    tau = a_norm * TAU_MAX
    for _ in range(n_sub):
        c2, s2 = np.cos(th2), np.sin(th2)
        M11 = (M1 + M2) * L1**2 + M2 * L2**2 + 2 * M2 * L1 * L2 * c2
        M12 = M2 * L2**2 + M2 * L1 * L2 * c2
        M22 = M2 * L2**2
        h1 = (-M2 * L1 * L2 * (2 * w1 + w2) * w2 * s2
              + (M1 + M2) * GP * L1 * np.sin(th1) + M2 * GP * L2 * np.sin(th1 + th2))
        h2 = M2 * L1 * L2 * w1**2 * s2 + M2 * GP * L2 * np.sin(th1 + th2) - tau
        det = M11 * M22 - M12**2
        det = np.where(np.abs(det) < 1e-9, np.sign(det) * 1e-9 + 1e-12, det)
        a1 = (M22 * (-h1) - M12 * (-h2)) / det
        a2 = (-M12 * (-h1) + M11 * (-h2)) / det
        w1 = w1 + a1 * dt
        w2 = w2 + a2 * dt
        th1 = th1 + w1 * dt
        th2 = th2 + w2 * dt
    return _rebuild(raw, th1, th2, w1, w2)


def _rk4_step(th1, th2, w1, w2, tau, dt):
    def dyn(t1, t2, v1, v2):
        c2, s2 = np.cos(t2), np.sin(t2)
        M11 = (M1 + M2) * L1**2 + M2 * L2**2 + 2 * M2 * L1 * L2 * c2
        M12 = M2 * L2**2 + M2 * L1 * L2 * c2
        M22 = M2 * L2**2
        h1 = (-M2 * L1 * L2 * (2 * v1 + v2) * v2 * s2
              + (M1 + M2) * GP * L1 * np.sin(t1) + M2 * GP * L2 * np.sin(t1 + t2))
        h2 = M2 * L1 * L2 * v1**2 * s2 + M2 * GP * L2 * np.sin(t1 + t2) - tau
        det = M11 * M22 - M12**2
        return (v1, v2, (M22 * (-h1) - M12 * (-h2)) / det,
                (-M12 * (-h1) + M11 * (-h2)) / det)
    k1 = dyn(th1, th2, w1, w2)
    k2 = dyn(th1 + .5 * dt * k1[0], th2 + .5 * dt * k1[1], w1 + .5 * dt * k1[2], w2 + .5 * dt * k1[3])
    k3 = dyn(th1 + .5 * dt * k2[0], th2 + .5 * dt * k2[1], w1 + .5 * dt * k2[2], w2 + .5 * dt * k2[3])
    k4 = dyn(th1 + dt * k3[0], th2 + dt * k3[1], w1 + dt * k3[2], w2 + dt * k3[3])
    return (th1 + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6,
            th2 + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6,
            w1 + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6,
            w2 + dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6)


def prior_rk4(raw, a_norm, dt=DT, n_sub=1):
    th1 = np.arctan2(raw[:, B1 + LF["sin_t"]], raw[:, B1 + LF["cos_t"]])
    th2 = np.arctan2(raw[:, B2 + LF["sin_t"]], raw[:, B2 + LF["cos_t"]])
    w1 = raw[:, B1 + LF["w"]].copy()
    w2 = raw[:, B2 + LF["w"]].copy()
    tau = a_norm * TAU_MAX
    for _ in range(n_sub):
        th1, th2, w1, w2 = _rk4_step(th1, th2, w1, w2, tau, dt)
    return _rebuild(raw, th1, th2, w1, w2)


def _rebuild(raw, th1, th2, w1, w2):
    out = raw.copy()
    x1 = L1 * np.sin(th1); y1 = -L1 * np.cos(th1)
    x2 = x1 + L2 * np.sin(th1 + th2); y2 = y1 - L2 * np.cos(th1 + th2)
    vx1 = L1 * w1 * np.cos(th1); vy1 = L1 * w1 * np.sin(th1)
    vx2 = vx1 + L2 * (w1 + w2) * np.cos(th1 + th2)
    vy2 = vy1 + L2 * (w1 + w2) * np.sin(th1 + th2)
    out[:, B1 + LF["x"]], out[:, B1 + LF["y"]] = x1, y1
    out[:, B2 + LF["x"]], out[:, B2 + LF["y"]] = x2, y2
    out[:, B1 + LF["sin_t"]], out[:, B1 + LF["cos_t"]] = np.sin(th1), np.cos(th1)
    out[:, B2 + LF["sin_t"]], out[:, B2 + LF["cos_t"]] = np.sin(th2), np.cos(th2)
    out[:, B1 + LF["vx"]], out[:, B1 + LF["vy"]] = vx1, vy1
    out[:, B2 + LF["vx"]], out[:, B2 + LF["vy"]] = vx2, vy2
    out[:, B1 + LF["w"]], out[:, B2 + LF["w"]] = w1, w2
    return out


W = feat_weights(OBJ_DIMS)[POS_IDX]


def pos_rmse(pred, tgt, mask=None):
    e = (pred[:, POS_IDX] - tgt[:, POS_IDX]) ** 2
    if mask is not None:
        return float(np.sqrt((e[mask] * W).mean()))
    return float(np.sqrt((e * W).mean()))


def main():
    d = np.load("trajectories_hinge.npz", allow_pickle=True)
    states, actions = d["states"], d["actions"]
    n_ep = states.shape[0]
    rng = np.random.RandomState(42)                       # 与 e0_common.make_datasets 同口径
    segs = np.arange(n_ep)
    rng.shuffle(segs)
    val_eps = np.sort(segs[:max(1, int(n_ep * 0.1))])
    S = states[val_eps]                                   # (300, 200, 22)
    A = actions[val_eps]
    n_win = S.shape[1] - L - K + 1
    ep_i = np.repeat(np.arange(len(val_eps)), n_win)
    w0 = np.tile(np.arange(n_win), len(val_eps))
    idx = np.arange(len(ep_i))
    if len(idx) > N_WIN:
        idx = np.sort(np.random.RandomState(0).choice(idx, size=N_WIN, replace=False))
    ep_i, w0 = ep_i[idx], w0[idx]
    print(f"val episodes={len(val_eps)}  窗口={len(ep_i)}  (L={L}, K={K})\n")

    def frame(off):
        return S[ep_i, w0 + off].astype(np.float64)

    last = frame(L - 1)                                   # 输入末帧
    tgt1 = frame(L)[:, :OBJ_DIMS] if frame(L).shape[-1] > OBJ_DIMS else frame(L)
    tgt1 = frame(L)[:, :OBJ_DIMS].copy()
    a0 = A[ep_i, w0 + L].astype(np.float64).reshape(-1)
    af = np.stack([A[ep_i, w0 + L + k].astype(np.float64).reshape(-1) for k in range(K)], 1)
    tgt_k = [frame(L + k)[:, :OBJ_DIMS].copy() for k in range(K)]

    # ---- 区间内动作变化（生成器 τ 每 0.01s 取值，标签每 0.05s 记录一次）----
    # 正确口径：比较相邻两个**采样帧**的记录动作 a(t) 与 a(t+1)；
    # 生成器 τ 在 0.1s 线性过渡内每 0.01s 变一次，故 Δa≠0 的采样帧即"处于过渡段"。
    a_next = A[ep_i, w0 + L + 1].astype(np.float64).reshape(-1)
    d_a = np.abs(a_next - a0)
    ramp = d_a > 1e-6
    print(f"处于力矩过渡段的窗口占比 = {ramp.mean()*100:.1f}%  "
          f"（这些帧的 |Δa| 中位数 {np.median(d_a[ramp]) if ramp.any() else 0:.4f}，"
          f"最大 {d_a.max():.4f}）")
    print("  说明：生成器在每个 0.05s 区间内用了 5 个不同的 τ，而标签只记录末步 τ；\n"
          "  该不可观测差异构成任何先验的**不可约下限**（下面按 Δa 分层估计其大小）。\n")

    specs = [
        ("zero-vel（无动力学）", None, None),
        ("euler@0.05×1（E0 现行）", prior_euler, dict(dt=DT, n_sub=1)),
        ("euler@0.05×5（错误用法对照）", prior_euler, dict(dt=DT / 5, n_sub=5)),
        ("euler@0.01×5（T19 M-F 细化）", prior_euler, dict(dt=PHYSICS_DT, n_sub=5)),
        ("rk4@0.05×1", prior_rk4, dict(dt=DT, n_sub=1)),
        ("rk4@0.01×5（精确积分器）", prior_rk4, dict(dt=PHYSICS_DT, n_sub=5)),
    ]

    print(f"{'先验':<30}{'单步':>8}{'k=2':>8}{'k=3':>8}{'k=4':>8}{'k=5':>8}"
          f"{'单步(Δa=0)':>12}{'单步(Δa≠0)':>12}")
    print("-" * 110)
    for name, fn, kw in specs:
        if fn is None:
            one = pos_rmse(last, tgt1)
            roll = [pos_rmse(last, tgt_k[k]) for k in range(K)]
            one0 = pos_rmse(last, tgt1, ~ramp)
            one1 = pos_rmse(last, tgt1, ramp)
        else:
            one = pos_rmse(fn(last, a0, **kw), tgt1)
            one0 = pos_rmse(fn(last, a0, **kw), tgt1, ~ramp)
            one1 = pos_rmse(fn(last, a0, **kw), tgt1, ramp)
            cur = last
            roll = []
            for k in range(K):
                cur = fn(cur, af[:, k], **kw)
                roll.append(pos_rmse(cur, tgt_k[k]))
        print(f"{name:<30}{one:>8.3f}" + "".join(f"{v:>8.2f}" for v in roll)
              + f"{one0:>12.3f}{one1:>12.3f}")

    # ---- 三处实现等价性对拍（颗粒度审计）----
    print("\n=== 三处实现等价性对拍（同输入，raw 空间 max|Δ|）===")
    from e0_common import make_datasets, E0Data           # noqa: E402
    import torch                                          # noqa: E402
    from model_h import prior_step_torch                  # noqa: E402
    from layout_h import acrobot_step                     # noqa: E402
    sub = np.arange(0, len(ep_i), max(1, len(ep_i) // 2000))[:2000]
    r = last[sub].copy()
    a = a0[sub]
    ref = prior_euler(r, a, dt=DT, n_sub=1)
    # 1) layout_h.acrobot_step(dt=0.05, substeps=1)
    lh = []
    for i in range(len(sub)):
        t1 = np.arctan2(r[i, B1 + LF["sin_t"]], r[i, B1 + LF["cos_t"]])
        t2 = np.arctan2(r[i, B2 + LF["sin_t"]], r[i, B2 + LF["cos_t"]])
        w1, w2 = r[i, B1 + LF["w"]], r[i, B2 + LF["w"]]
        t1, t2, w1, w2 = acrobot_step(t1, t2, w1, w2, a[i] * TAU_MAX, dt=DT, substeps=1)
        lh.append(_rebuild(r[i:i + 1], np.array([t1]), np.array([t2]),
                           np.array([w1]), np.array([w2]))[0])
    lh = np.array(lh)
    print(f"  layout_h.acrobot_step(0.05,1)  vs 本脚本 euler@0.05×1 : max|Δ| = "
          f"{np.abs(lh - ref).max():.3e}")
    # 2) model_h.prior_step_torch（走归一化空间）
    _, val_ds, mu, sd = make_datasets("trajectories_hinge.npz", L=L, K=K, seed=42)
    raw_t = torch.tensor(r, dtype=torch.float32)
    an_t = torch.tensor(a, dtype=torch.float32).reshape(-1, 1)
    outs = prior_step_torch(raw_t[:, B1 + LF["sin_t"]].double(), raw_t[:, B1 + LF["cos_t"]].double(),
                            raw_t[:, B2 + LF["sin_t"]].double(), raw_t[:, B2 + LF["cos_t"]].double(),
                            raw_t[:, B1 + LF["w"]].double(), raw_t[:, B2 + LF["w"]].double(),
                            an_t.double())
    th1n = torch.atan2(outs[0], outs[1]).numpy()
    th2n = torch.atan2(outs[2], outs[3]).numpy()
    tw = _rebuild(r, th1n, th2n, outs[4].numpy(), outs[5].numpy())
    print(f"  model_h.prior_step_torch       vs 本脚本 euler@0.05×1 : max|Δ| = "
          f"{np.abs(tw - ref).max():.3e}")
    # 3) e0_common.E0Data._prior_np（训练时构造残差目标用的就是它）
    ds = E0Data(states, d["contact"], actions, d["yao"], val_eps, L=L, K=K,
                mean_obj=mu, std_obj=sd)
    pn = ds._prior_np((r - mu) / (sd + 1e-8), a.reshape(-1, 1).astype(np.float32))
    np_raw = pn * (sd + 1e-8) + mu
    print(f"  e0_common._prior_np（残差目标） vs 本脚本 euler@0.05×1 : max|Δ| = "
          f"{np.abs(np_raw - ref).max():.3e}")


if __name__ == "__main__":
    main()
