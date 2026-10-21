# -*- coding: utf-8 -*-
"""先验积分器注册表（physics_prior_spec.py）——先验的**唯一**实现来源。

背景（2026-10-01 审计）：先验此前有**三处**独立实现，颗粒度各不相同：
  · layout_h.acrobot_step    （numpy，半隐式欧拉，dt=0.01，substeps 可调）
  · model_h.prior_step_torch （torch，半隐式欧拉，dt=0.05，硬编码 G=900）
  · e0_common.E0Data._prior_np（numpy，半隐式欧拉，dt=0.05，用于构造残差目标）
审计对拍结果：三者在同输入下 max|Δ| = 0.0 / 1e-12 / 1.1e-6（数值一致，但"三份代码"
意味着任何一次改动都可能只改一处 → 训练目标与评估先验静默错配的风险）。

本模块把三者收敛为**一个** spec 表 + 一份实现，并对 euler005 保持逐位兼容：

    spec          method  dt      n_sub  单步(px)  闭环 k=5(px)   说明
    euler005      euler   0.05    1      3.630     16.27         E0 现行（默认，兼容）
    euler001x5    euler   0.01    5      0.532      3.35         T19 M-F 细化版
    rk4005        rk4     0.05    1      0.167      2.13         —
    rk4001x5      rk4     0.01    5      0.038      0.50         = 数据生成器（"精确"）
    （数字为 tmp_prior_ladder_pilot.py 在 30k 窗口 / E0 val 划分上的实测）

τ 的颗粒度说明：生成器每 0.01s 取一次 τ（0.1s 线性过渡），而动作标签每 0.05s
只记录末步值 → 区间内 τ 变化不可观测。实测其贡献 ≤0.06px（精确先验在 Δa≠0 帧
0.079px vs Δa=0 帧 0.018px），故对结论无影响，但任何先验都受此微小下限约束。
"""

import numpy as np
import torch

from layout_h import (L1, L2, M1, M2, G, TAU_MAX, PHYSICS_DT, DT,
                      obj_base, F as LF)

B1, B2 = obj_base(0), obj_base(1)
_G = -G          # layout_h 的 G 为负（世界定义），进入方程取正
TAU_DT = DT      # 兼容旧名（model_h.TAU_DT）

# spec → (method, dt, n_sub)
SPECS = {
    "euler005":   ("euler", DT, 1),                 # E0 现行（默认）
    "euler001x5": ("euler", PHYSICS_DT, 5),         # T19 M-F 细化
    "rk4005":     ("rk4", DT, 1),
    "rk4001x5":   ("rk4", PHYSICS_DT, 5),           # 与数据生成器同积分器
}
DEFAULT_SPEC = "euler005"

# 报告/图里的人类可读标签（文档与代码共用，避免两边各写一套）
SPEC_LABELS = {
    "euler005":   "euler@0.05×1",
    "euler001x5": "euler@0.01×5",
    "rk4005":     "rk4@0.05",
    "rk4001x5":   "rk4@0.01×5",
}


def spec_of(name):
    if name not in SPECS:
        raise KeyError(f"未知先验 spec: {name}（可选 {sorted(SPECS)}）")
    return SPECS[name]


# ============================================================ numpy 实现
def _accel_np(th1, th2, w1, w2, tau):
    c2, s2 = np.cos(th2), np.sin(th2)
    M11 = (M1 + M2) * L1 * L1 + M2 * L2 * L2 + 2.0 * M2 * L1 * L2 * c2
    M12 = M2 * L2 * L2 + M2 * L1 * L2 * c2
    M22 = M2 * L2 * L2
    h1 = (-M2 * L1 * L2 * (2.0 * w1 + w2) * w2 * s2
          + (M1 + M2) * _G * L1 * np.sin(th1) + M2 * _G * L2 * np.sin(th1 + th2))
    h2 = (M2 * L1 * L2 * w1 * w1 * s2 + M2 * _G * L2 * np.sin(th1 + th2) - tau)
    det = M11 * M22 - M12 * M12
    det = np.where(np.abs(det) < 1e-9, np.sign(det) * 1e-9, det)
    a1 = (M22 * (-h1) - M12 * (-h2)) / det
    a2 = (-M12 * (-h1) + M11 * (-h2)) / det
    return a1, a2


def integrate_np(method, dt, n_sub, th1, th2, w1, w2, tau):
    for _ in range(n_sub):
        if method == "euler":
            a1, a2 = _accel_np(th1, th2, w1, w2, tau)
            w1 = w1 + a1 * dt
            w2 = w2 + a2 * dt
            th1 = th1 + w1 * dt
            th2 = th2 + w2 * dt
        elif method == "rk4":
            k1 = (w1, w2) + _accel_np(th1, th2, w1, w2, tau)
            k2 = (w1 + .5 * dt * k1[2], w2 + .5 * dt * k1[3]) + _accel_np(
                th1 + .5 * dt * k1[0], th2 + .5 * dt * k1[1],
                w1 + .5 * dt * k1[2], w2 + .5 * dt * k1[3], tau)
            k3 = (w1 + .5 * dt * k2[2], w2 + .5 * dt * k2[3]) + _accel_np(
                th1 + .5 * dt * k2[0], th2 + .5 * dt * k2[1],
                w1 + .5 * dt * k2[2], w2 + .5 * dt * k2[3], tau)
            k4 = (w1 + dt * k3[2], w2 + dt * k3[3]) + _accel_np(
                th1 + dt * k3[0], th2 + dt * k3[1],
                w1 + dt * k3[2], w2 + dt * k3[3], tau)
            th1 = th1 + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
            th2 = th2 + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
            w1 = w1 + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
            w2 = w2 + dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6.0
        else:
            raise ValueError(f"未知积分方法: {method}")
    return th1, th2, w1, w2


def rebuild_np(raw, th1, th2, w1, w2):
    """由角度/角速度重建状态（位置、sin/cos、vx/vy、w）；其余字段保持原值。"""
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


def prior_raw_np(spec, raw, a_norm):
    """原始单位状态 (N,22) + 归一化力矩 (N,) → 下一帧原始状态 (N,22)。"""
    method, dt, n_sub = spec_of(spec)
    th1 = np.arctan2(raw[:, B1 + LF["sin_t"]], raw[:, B1 + LF["cos_t"]])
    th2 = np.arctan2(raw[:, B2 + LF["sin_t"]], raw[:, B2 + LF["cos_t"]])
    w1 = raw[:, B1 + LF["w"]]
    w2 = raw[:, B2 + LF["w"]]
    th1, th2, w1, w2 = integrate_np(method, dt, n_sub, th1, th2, w1, w2,
                                    np.asarray(a_norm).reshape(-1) * TAU_MAX)
    return rebuild_np(raw, th1, th2, w1, w2)


# ============================================================ torch 实现
def _accel_torch(th1, th2, w1, w2, tau):
    c2 = torch.cos(th2); s2 = torch.sin(th2)
    M11 = (M1 + M2) * L1 ** 2 + M2 * L2 ** 2 + 2 * M2 * L1 * L2 * c2
    M12 = M2 * L2 ** 2 + M2 * L1 * L2 * c2
    M22 = M2 * L2 ** 2
    h1 = (-M2 * L1 * L2 * (2 * w1 + w2) * w2 * s2
          + (M1 + M2) * _G * L1 * torch.sin(th1) + M2 * _G * L2 * torch.sin(th1 + th2))
    h2 = (M2 * L1 * L2 * w1 ** 2 * s2 + M2 * _G * L2 * torch.sin(th1 + th2) - tau)
    det = M11 * M22 - M12 ** 2
    det = torch.where(det.abs() < 1e-9, det.sign() * 1e-9, det)
    a1 = (M22 * (-h1) - M12 * (-h2)) / det
    a2 = (-M12 * (-h1) + M11 * (-h2)) / det
    return a1, a2


def integrate_torch(method, dt, n_sub, th1, th2, w1, w2, tau):
    for _ in range(n_sub):
        if method == "euler":
            a1, a2 = _accel_torch(th1, th2, w1, w2, tau)
            w1 = w1 + a1 * dt
            w2 = w2 + a2 * dt
            th1 = th1 + w1 * dt
            th2 = th2 + w2 * dt
        elif method == "rk4":
            def dyn(t1, t2, v1, v2):
                return (v1, v2) + _accel_torch(t1, t2, v1, v2, tau)
            k1 = dyn(th1, th2, w1, w2)
            k2 = dyn(th1 + .5 * dt * k1[0], th2 + .5 * dt * k1[1],
                     w1 + .5 * dt * k1[2], w2 + .5 * dt * k1[3])
            k3 = dyn(th1 + .5 * dt * k2[0], th2 + .5 * dt * k2[1],
                     w1 + .5 * dt * k2[2], w2 + .5 * dt * k2[3])
            k4 = dyn(th1 + dt * k3[0], th2 + dt * k3[1],
                     w1 + dt * k3[2], w2 + dt * k3[3])
            th1 = th1 + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
            th2 = th2 + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
            w1 = w1 + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
            w2 = w2 + dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6.0
        else:
            raise ValueError(f"未知积分方法: {method}")
    return th1, th2, w1, w2


def prior_step_torch_spec(spec, sin1, cos1, sin2, cos2, w1, w2, a_norm):
    """兼容签名：返回 (sin1, cos1, sin2, cos2, w1, w2)。"""
    method, dt, n_sub = spec_of(spec)
    th1 = torch.atan2(sin1, cos1)
    th2 = torch.atan2(sin2, cos2)
    tau = a_norm.squeeze(-1) * TAU_MAX
    th1, th2, w1, w2 = integrate_torch(method, dt, n_sub, th1, th2, w1, w2, tau)
    return (torch.sin(th1), torch.cos(th1), torch.sin(th2), torch.cos(th2), w1, w2)


def prior_norm_torch(spec, obj_norm, action, mean_obj, std_obj, dtype=None):
    """归一化空间先验：obj_norm (B,22) + action (B,1) → 下一帧归一化状态 (B,22)。

    dtype 给定时先把角度/角速度/动作转成该精度再积分（用于 float64 对拍）。
    """
    mean = torch.as_tensor(mean_obj, dtype=obj_norm.dtype, device=obj_norm.device)
    std = torch.as_tensor(std_obj, dtype=obj_norm.dtype, device=obj_norm.device)
    raw = obj_norm * (std + 1e-8) + mean
    b1, b2 = B1, B2
    sin1, cos1 = raw[:, b1 + LF["sin_t"]], raw[:, b1 + LF["cos_t"]]
    sin2, cos2 = raw[:, b2 + LF["sin_t"]], raw[:, b2 + LF["cos_t"]]
    w1, w2 = raw[:, b1 + LF["w"]], raw[:, b2 + LF["w"]]
    if dtype is not None:
        sin1, cos1 = sin1.to(dtype), cos1.to(dtype)
        sin2, cos2 = sin2.to(dtype), cos2.to(dtype)
        w1, w2, action = w1.to(dtype), w2.to(dtype), action.to(dtype)
    s1n, c1n, s2n, c2n, w1n, w2n = prior_step_torch_spec(
        spec, sin1, cos1, sin2, cos2, w1, w2, action)
    th1n = torch.atan2(s1n, c1n)
    th2n = torch.atan2(s2n, c2n)
    x1 = L1 * torch.sin(th1n); y1 = -L1 * torch.cos(th1n)
    x2 = x1 + L2 * torch.sin(th1n + th2n); y2 = y1 - L2 * torch.cos(th1n + th2n)
    vx1 = L1 * w1n * torch.cos(th1n); vy1 = L1 * w1n * torch.sin(th1n)
    vx2 = vx1 + L2 * (w1n + w2n) * torch.cos(th1n + th2n)
    vy2 = vy1 + L2 * (w1n + w2n) * torch.sin(th1n + th2n)
    out = raw.clone()
    out[:, b1 + LF["x"]], out[:, b1 + LF["y"]] = x1, y1
    out[:, b2 + LF["x"]], out[:, b2 + LF["y"]] = x2, y2
    out[:, b1 + LF["sin_t"]], out[:, b1 + LF["cos_t"]] = s1n, c1n
    out[:, b2 + LF["sin_t"]], out[:, b2 + LF["cos_t"]] = s2n, c2n
    out[:, b1 + LF["vx"]], out[:, b1 + LF["vy"]] = vx1, vy1
    out[:, b2 + LF["vx"]], out[:, b2 + LF["vy"]] = vx2, vy2
    out[:, b1 + LF["w"]], out[:, b2 + LF["w"]] = w1n, w2n
    return (out - mean) / (std + 1e-8)
