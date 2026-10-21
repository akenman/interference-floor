# -*- coding: utf-8 -*-
"""弹球世界先验积分器注册表（physics_prior_spec_v3.py）——v3 管线的先验唯一来源。

背景（SLOT-1，2026-10-03）：弹球引擎（pymunk）的实测积分器形式为**显式欧拉**
（位置先于速度更新；真自由飞行转移 dy−vy·τ = 10·a·dt²，DEEP_DIVE 10.19.23）。
本模块把 v3 的先验从单档（layout.physics_prior 闭式抛物线）扩展为 4 档阶梯：

    spec          实现                     计算类                      对引擎单步差（FF，a=900）
    parabola005   layout.physics_prior     连续精确（缺力位移项）       0.225px/维（T1）
    euler005      显式欧拉 @0.05×1         同族粗档                    0.9px/维
    rk4005        RK4 @0.05×1              连续精确（含力位移项）       0.225px/维
    euler001x5    显式欧拉 @0.01×5         =引擎复现                    ~0（float32）

parabola005 为默认档，逐位复用 layout.physics_prior（历史 checkpoint 行为不变，
tests/test_prior_spec_v3.py 断言）。动作语义与引擎一致：恒力 F = I/τ 逐子步施加。
"""

import numpy as np
import torch

from layout import (NUM_OBJ, DT_V3, PHYSICS_DT, F as LF, obj_base,
                    physics_prior as _legacy_prior)

G_V3 = -900.0          # 与 layout/generate_trajectories_v3 一致（px/s²）
TAU_V3 = DT_V3         # 0.05
DT_E = PHYSICS_DT      # 0.01

SPECS = {
    "parabola005": ("legacy", TAU_V3, 1),
    "euler005":    ("euler_expl", TAU_V3, 1),
    "rk4005":      ("rk4", TAU_V3, 1),
    "euler001x5":  ("euler_expl", DT_E, 5),
}
DEFAULT_SPEC = "parabola005"

# 交错布局（与 _accel_*/_rebuild_* 的 [vx0,vy0,vx1,vy1,...] 约定一致）
POS_DIMS = [d for o in range(NUM_OBJ)
            for d in (obj_base(o) + LF["x"], obj_base(o) + LF["y"])]
VEL_DIMS = [d for o in range(NUM_OBJ)
            for d in (obj_base(o) + LF["vx"], obj_base(o) + LF["vy"])]
MASS_DIMS = [obj_base(o) + LF["mass"] for o in range(NUM_OBJ)]


def spec_of(name):
    if name not in SPECS:
        raise KeyError(f"未知先验 spec: {name}（可选 {sorted(SPECS)}）")
    return SPECS[name]


def _accel_np(x0, impulse_raw, tau):
    """每物体常加速度 a = I/(m·τ) + g（恒力 F=I/τ）。返回 (M,6) [vx0,vy0,...]。"""
    m = x0[:, MASS_DIMS]
    m = np.where(m > 1e-3, m, 1.0)
    acc = np.empty((x0.shape[0], 2 * NUM_OBJ), dtype=np.float64)
    for o in range(NUM_OBJ):
        acc[:, 2 * o] = impulse_raw[:, 2 * o] / (m[:, o] * tau)
        acc[:, 2 * o + 1] = impulse_raw[:, 2 * o + 1] / (m[:, o] * tau) + G_V3
    return acc


def _rebuild_np(out, x0, pos, vel):
    """把平铺 (M,6) 的 pos/vel 写回 33 维布局；角 θ'=θ+ω·τ（无力矩，全档相同）。"""
    for o in range(NUM_OBJ):
        b = obj_base(o)
        out[:, b + LF["x"]] = pos[:, 2 * o]
        out[:, b + LF["y"]] = pos[:, 2 * o + 1]
        out[:, b + LF["vx"]] = vel[:, 2 * o]
        out[:, b + LF["vy"]] = vel[:, 2 * o + 1]
        th = np.arctan2(x0[:, b + LF["sin_t"]], x0[:, b + LF["cos_t"]]) \
            + x0[:, b + LF["w"]] * TAU_V3
        out[:, b + LF["sin_t"]] = np.sin(th)
        out[:, b + LF["cos_t"]] = np.cos(th)
    return out


def prior_raw_np(spec, s_last_raw, impulse_raw):
    """原始单位状态 (M,33) + 原始净冲量 (M,6) → 下一帧原始状态 (M,33)。"""
    method, dt, n_sub = spec_of(spec)
    if method == "legacy":
        return _legacy_prior(s_last_raw, impulse_raw, tau=TAU_V3, g=G_V3)
    x64 = s_last_raw.astype(np.float64)
    acc = _accel_np(x64, impulse_raw.astype(np.float64), TAU_V3)
    pos = x64[:, POS_DIMS].astype(np.float64).copy()
    vel = x64[:, VEL_DIMS].astype(np.float64).copy()
    if method == "euler_expl":
        for _ in range(n_sub):
            pos = pos + vel * dt
            vel = vel + acc * dt
    elif method == "rk4":
        k1x, k1v = vel, acc
        k2x, k2v = vel + 0.5 * dt * k1v, acc
        k3x, k3v = vel + 0.5 * dt * k2v, acc
        k4x, k4v = vel + dt * k3v, acc
        pos = pos + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        vel = vel + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
    else:
        raise ValueError(method)
    out = s_last_raw.copy()
    return _rebuild_np(out, s_last_raw, pos, vel)


# ============================================================ torch 实现
def _accel_torch(x0, impulse, tau):
    m = x0[:, MASS_DIMS]
    m = torch.where(m > 1e-3, m, torch.ones_like(m))
    acc = torch.empty(x0.shape[0], 2 * NUM_OBJ, dtype=x0.dtype, device=x0.device)
    for o in range(NUM_OBJ):
        acc[:, 2 * o] = impulse[:, 2 * o] / (m[:, o] * tau)
        acc[:, 2 * o + 1] = impulse[:, 2 * o + 1] / (m[:, o] * tau) + G_V3
    return acc


def _rebuild_torch(out, x0, pos, vel, tau):
    for o in range(NUM_OBJ):
        b = obj_base(o)
        out[:, b + LF["x"]] = pos[:, 2 * o]
        out[:, b + LF["y"]] = pos[:, 2 * o + 1]
        out[:, b + LF["vx"]] = vel[:, 2 * o]
        out[:, b + LF["vy"]] = vel[:, 2 * o + 1]
        th = torch.atan2(x0[:, b + LF["sin_t"]], x0[:, b + LF["cos_t"]]) \
            + x0[:, b + LF["w"]] * tau
        out[:, b + LF["sin_t"]] = torch.sin(th)
        out[:, b + LF["cos_t"]] = torch.cos(th)
    return out


def prior_raw_torch(spec, s_last_raw, impulse_raw):
    """torch 版 prior_raw_np（float32 路径；与 numpy 一致性由测试断言 ≤1e-4）。"""
    method, dt, n_sub = spec_of(spec)
    if method == "legacy":
        legacy = _legacy_prior(s_last_raw.cpu().numpy(), impulse_raw.cpu().numpy(),
                               tau=TAU_V3, g=G_V3)
        return torch.from_numpy(np.ascontiguousarray(legacy))
    acc = _accel_torch(s_last_raw, impulse_raw, TAU_V3)
    pos = s_last_raw[:, POS_DIMS].clone()
    vel = s_last_raw[:, VEL_DIMS].clone()
    if method == "euler_expl":
        for _ in range(n_sub):
            pos = pos + vel * dt
            vel = vel + acc * dt
    elif method == "rk4":
        k1x, k1v = vel, acc
        k2x, k2v = vel + 0.5 * dt * k1v, acc
        k3x, k3v = vel + 0.5 * dt * k2v, acc
        k4x, k4v = vel + dt * k3v, acc
        pos = pos + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        vel = vel + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
    out = s_last_raw.clone()
    return _rebuild_torch(out, s_last_raw, pos, vel, TAU_V3)


def prior_norm_torch(spec, obj_norm, action_norm, mean_obj, std_obj, impulse_max):
    """归一化空间先验：obj_norm (B,33) + 归一化动作 (B,6) → (B,33) 归一化。"""
    raw = obj_norm * std_obj + mean_obj
    imp = action_norm * impulse_max
    out = prior_raw_torch(spec, raw, imp)
    return (out - mean_obj) / (std_obj + 1e-8)


def assert_dir_not_foreign_v3(ckpt_dir, prior_spec, seed):
    """覆盖保护（v3 版）：目录里已有 best.pt 且 prior_spec/seed 不同 → 拒训。
    （E0 覆盖事故的制度化教训在 v3 管线的最小落地。）"""
    import os
    pt = os.path.join(ckpt_dir, "best.pt")
    if not os.path.exists(pt):
        return
    try:
        ck = torch.load(pt, map_location="cpu", weights_only=False)
    except Exception as e:                                  # noqa: BLE001
        print(f"[warn] 无法读取已有 {pt}（{e}），跳过覆盖保护", flush=True)
        return
    old = ck.get("args", {}) or {}
    if "prior_spec" not in old and "seed" not in old:
        return                                              # 早期 checkpoint，容忍
    if old.get("prior_spec", DEFAULT_SPEC) != prior_spec or old.get("seed") != seed:
        raise SystemExit(
            f"[拒训] {ckpt_dir} 已有 prior_spec={old.get('prior_spec')} "
            f"seed={old.get('seed')} 的 checkpoint，与本进程 "
            f"prior_spec={prior_spec} seed={seed} 不一致。请换 --checkpoint_dir。")
