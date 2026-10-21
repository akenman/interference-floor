# -*- coding: utf-8 -*-
"""摆世界精确档残余下限的成因工件（W9）：区间内扭矩粒度不可观测。

背景：生成器（generate_hinge.make_tau_schedule）在 0.1 s 线性过渡期内每 0.01 s
更新一次 τ，而动作标签每 0.05 s 只记录末步值 ⇒ 过渡区间内的 τ 变化对任何先验
都是**不可观测输入**。本脚本把 tmp_prior_ladder_pilot.py 的 Δa 分层读数落成
JSON（此前只在预注册 aliasng_floor 的散文中），供正文 §4 引用与守卫绑定。

口径（与 tmp_prior_ladder_pilot.py 逐位一致）：
  - val 划分：RandomState(42) shuffle episodes 取前 10%（= e0_common.make_datasets 同构）；
  - 窗口：L=8, K=5, N_WIN=30000（RandomState(0) 子采样）；
  - 过渡判据：|a(t+1) − a(t)| > 1e-6（相邻采样帧的记录动作）；
  - 误差：POS_IDX=[1,2,12,13] + feat_weights 加权 RMSE（与 e0_common.validate 同口径）；
  - 先验实现：physics_prior_spec.prior_raw_np（单一来源，非本脚本重实现）。

用法：python paper/tables/build_torque_granularity.py
写：analysis_output/torque_granularity.json
"""
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from layout_h import OBJ_DIMS, feat_weights, obj_base, F as LF  # noqa: E402
from physics_prior_spec import prior_raw_np  # noqa: E402

POS_IDX = [1, 2, 12, 13]
L, K, N_EP_VAL, N_WIN = 8, 5, 300, 30000
B1, B2 = obj_base(0), obj_base(1)
W = feat_weights(OBJ_DIMS)[POS_IDX]


def pos_rmse(pred, tgt, mask=None):
    e = (pred[:, POS_IDX] - tgt[:, POS_IDX]) ** 2
    if mask is not None:
        return float(np.sqrt((e[mask] * W).mean()))
    return float(np.sqrt((e * W).mean()))


def main():
    d = np.load(os.path.join(ROOT, "trajectories_hinge.npz"), allow_pickle=True)
    states, actions = d["states"], d["actions"]
    n_ep = states.shape[0]
    rng = np.random.RandomState(42)
    segs = np.arange(n_ep)
    rng.shuffle(segs)
    val_eps = np.sort(segs[:max(1, int(n_ep * 0.1))])
    S = states[val_eps]
    A = actions[val_eps]
    n_win = S.shape[1] - L - K + 1
    ep_i = np.repeat(np.arange(len(val_eps)), n_win)
    w0 = np.tile(np.arange(n_win), len(val_eps))
    idx = np.arange(len(ep_i))
    if len(idx) > N_WIN:
        idx = np.sort(np.random.RandomState(0).choice(idx, size=N_WIN, replace=False))
    ep_i, w0 = ep_i[idx], w0[idx]

    last = S[ep_i, w0 + L - 1].astype(np.float64)
    tgt1 = S[ep_i, w0 + L][:, :OBJ_DIMS].astype(np.float64)
    a0 = A[ep_i, w0 + L].astype(np.float64).reshape(-1)
    a_next = A[ep_i, w0 + L + 1].astype(np.float64).reshape(-1)
    ramp = np.abs(a_next - a0) > 1e-6

    tiers = {}
    for spec in ("euler005", "euler001x5", "rk4005", "rk4001x5"):
        pred = prior_raw_np(spec, last, a0)
        tiers[spec] = dict(
            one_step=pos_rmse(pred, tgt1),
            one_step_da0=pos_rmse(pred, tgt1, ~ramp),
            one_step_dane0=pos_rmse(pred, tgt1, ramp),
        )

    exact = tiers["rk4001x5"]
    out = dict(
        note=("Exact-tier residual floor in the pendulum world: the generator updates "
              "tau every 0.01 s inside a 0.1 s ramp, the action label records only the "
              "last value of each 0.05 s interval; windows sitting inside a ramp carry "
              "an unobservable intra-interval torque schedule."),
        protocol=dict(val_episodes=int(len(val_eps)), window_seed=0, n_windows=int(len(ep_i)),
                      L=L, K=K, pos_idx=POS_IDX,
                      prior_impl="physics_prior_spec.prior_raw_np",
                      ramp_criterion="|a(t+1)-a(t)| > 1e-6"),
        ramp_fraction=float(ramp.mean()),
        n_ramp=int(ramp.sum()),
        tiers=tiers,
        exact_tier_gap_px=float(exact["one_step_dane0"] - exact["one_step_da0"]),
        prereg_reference=dict(
            file="experiments/prereg_P0a_prior_ladder.json",
            field="aliasing_floor",
            quoted=dict(ramp_fraction=0.188, da0_px=0.018, dane0_px=0.079,
                        floor_upper_bound_px=0.06)),
    )

    assert abs(out["ramp_fraction"] - 0.188) < 0.002
    assert abs(exact["one_step_da0"] - 0.018) < 0.001
    assert abs(exact["one_step_dane0"] - 0.079) < 0.001
    assert abs(exact["one_step"] - 0.038) < 0.001
    assert abs(tiers["rk4005"]["one_step"] - 0.167) < 0.002
    assert abs(tiers["euler005"]["one_step"] - 3.630) < 0.002

    p = os.path.join(ROOT, "analysis_output", "torque_granularity.json")
    with io.open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("[save]", p)
    print(f"  ramp windows: {out['ramp_fraction'] * 100:.1f}% (n={out['n_ramp']:,})")
    for spec, t in tiers.items():
        print(f"  [{spec:11s}] one-step={t['one_step']:.4f}  "
              f"da=0 {t['one_step_da0']:.4f}  da!=0 {t['one_step_dane0']:.4f}")
    print(f"  exact-tier gap (da!=0 - da=0): {out['exact_tier_gap_px'] * 1000:.1f}e-3 px")


if __name__ == "__main__":
    main()
