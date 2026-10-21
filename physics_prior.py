# -*- coding: utf-8 -*-
"""
铰链世界物理先验 — 统一实现（消除 4 处重复）
================================================
提供 torch 和 numpy 两个版本的双摆动力学：
  - prior_step_torch(): 归一化空间，批量，用于模型 forward / rollout
  - prior_step_np():    物理空间，批量，用于数据集窗口构建
  - prior_full_torch(): 从完整 obj 向量预测下一帧（含位置重构）
  - prior_full_np():    numpy 版

所有文件（model_h.py, model_hp.py, core_model.py, train_*.py）
应统一 import 此模块，不再各自实现。
"""

import numpy as np
import torch

# ── 物理常量 ──
L1, L2 = 80.0, 80.0          # 杆长 (px)
M1, M2 = 2.0, 1.0            # 质量 (kg)
G_PRIOR = 900.0               # 重力加速度 (px/s²)
TAU_MAX = 28000.0             # 最大肘力矩 (N·m)
TAU_DT = 0.05                 # 时间步长 (s)
OBJ_DIMS = 22                 # 物体状态维度
B1, B2 = 0, 11               # 物体1/2在 obj 向量中的起始索引


# ═══════════════════════════════════════════════════════════════
# Torch 版本（归一化空间输入输出，批量）
# ═══════════════════════════════════════════════════════════════

def prior_step_torch(sin1, cos1, sin2, cos2, w1, w2, a_norm):
    """无接触双摆先验一步（归一化空间，torch 批量版）。

    输入输出均为角度的 sin/cos 与角速度（避免回绕问题）。
    a_norm: (B,1) 归一化肘力矩。
    返回: (sin1n, cos1n, sin2n, cos2n, w1n, w2n)
    """
    th1 = torch.atan2(sin1, cos1)
    th2 = torch.atan2(sin2, cos2)
    tau = a_norm.squeeze(-1) * TAU_MAX
    c2 = torch.cos(th2)
    s2 = torch.sin(th2)
    M11 = (M1 + M2) * L1**2 + M2 * L2**2 + 2 * M2 * L1 * L2 * c2
    M12 = M2 * L2**2 + M2 * L1 * L2 * c2
    M22 = M2 * L2**2
    h1 = (-M2 * L1 * L2 * (2 * w1 + w2) * w2 * s2
          + (M1 + M2) * G_PRIOR * L1 * torch.sin(th1)
          + M2 * G_PRIOR * L2 * torch.sin(th1 + th2))
    h2 = (M2 * L1 * L2 * w1**2 * s2
          + M2 * G_PRIOR * L2 * torch.sin(th1 + th2)
          - tau)
    det = M11 * M22 - M12**2
    det = torch.where(det.abs() < 1e-9, det.sign() * 1e-9, det)
    a1 = (M22 * (-h1) - M12 * (-h2)) / det
    a2 = (-M12 * (-h1) + M11 * (-h2)) / det
    w1n = w1 + a1 * TAU_DT
    w2n = w2 + a2 * TAU_DT
    th1n = th1 + w1n * TAU_DT
    th2n = th2 + w2n * TAU_DT
    return torch.sin(th1n), torch.cos(th1n), torch.sin(th2n), torch.cos(th2n), w1n, w2n


def prior_full_torch(obj_norm, action, mean_obj, std_obj):
    """完整先验：从归一化 obj 向量预测下一帧（含位置/速度重构）。

    obj_norm: (B, 22) 归一化物体状态
    action: (B, 1) 归一化力矩
    返回: (B, 22) 归一化预测下一帧
    """
    raw = obj_norm * std_obj + mean_obj
    s1n, c1n, s2n, c2n, w1n, w2n = prior_step_torch(
        raw[:, B1 + 3], raw[:, B1 + 4], raw[:, B2 + 3], raw[:, B2 + 4],
        raw[:, B1 + 7], raw[:, B2 + 7], action)
    out = raw.clone()
    th1n = torch.atan2(s1n, c1n)
    th2n = torch.atan2(s2n, c2n)
    x1 = L1 * torch.sin(th1n); y1 = -L1 * torch.cos(th1n)
    x2 = x1 + L2 * torch.sin(th1n + th2n); y2 = y1 - L2 * torch.cos(th1n + th2n)
    vx1 = L1 * w1n * torch.cos(th1n); vy1 = L1 * w1n * torch.sin(th1n)
    vx2 = vx1 + L2 * (w1n + w2n) * torch.cos(th1n + th2n)
    vy2 = vy1 + L2 * (w1n + w2n) * torch.sin(th1n + th2n)
    out[:, B1 + 1], out[:, B1 + 2] = x1, y1
    out[:, B2 + 1], out[:, B2 + 2] = x2, y2
    out[:, B1 + 3], out[:, B1 + 4] = s1n, c1n
    out[:, B2 + 3], out[:, B2 + 4] = s2n, c2n
    out[:, B1 + 5], out[:, B1 + 6] = vx1, vy1
    out[:, B2 + 5], out[:, B2 + 6] = vx2, vy2
    out[:, B1 + 7], out[:, B2 + 7] = w1n, w2n
    return (out - mean_obj) / (std_obj + 1e-8)


# ═══════════════════════════════════════════════════════════════
# Numpy 版本（物理空间输入输出，批量，用于数据集构建）
# ═══════════════════════════════════════════════════════════════

def prior_step_np(sin1, cos1, sin2, cos2, w1, w2, a_norm):
    """无接触双摆先验一步（numpy 批量版，与 torch 版完全对称）。

    a_norm: (B,) 或 (B,1) 归一化肘力矩。
    返回: (sin1n, cos1n, sin2n, cos2n, w1n, w2n)
    """
    th1 = np.arctan2(sin1, cos1)
    th2 = np.arctan2(sin2, cos2)
    tau = np.asarray(a_norm).squeeze(-1) * TAU_MAX
    c2 = np.cos(th2); s2 = np.sin(th2)
    M11 = (M1 + M2) * L1**2 + M2 * L2**2 + 2 * M2 * L1 * L2 * c2
    M12 = M2 * L2**2 + M2 * L1 * L2 * c2
    M22 = M2 * L2**2
    h1 = (-M2 * L1 * L2 * (2 * w1 + w2) * w2 * s2
          + (M1 + M2) * G_PRIOR * L1 * np.sin(th1)
          + M2 * G_PRIOR * L2 * np.sin(th1 + th2))
    h2 = (M2 * L1 * L2 * w1**2 * s2
          + M2 * G_PRIOR * L2 * np.sin(th1 + th2)
          - tau)
    det = M11 * M22 - M12**2
    det = np.where(np.abs(det) < 1e-9, np.sign(det) * 1e-9, det)
    a1 = (M22 * (-h1) - M12 * (-h2)) / det
    a2 = (-M12 * (-h1) + M11 * (-h2)) / det
    w1n = w1 + a1 * TAU_DT; w2n = w2 + a2 * TAU_DT
    th1n = th1 + w1n * TAU_DT; th2n = th2 + w2n * TAU_DT
    return np.sin(th1n), np.cos(th1n), np.sin(th2n), np.cos(th2n), w1n, w2n


def prior_full_np(obj_norm_batch, a_norm, mean_obj, std_obj):
    """完整先验：numpy 版，从归一化 obj 向量预测下一帧。

    obj_norm_batch: (B, 22) 归一化物体状态
    a_norm: (B, 1) 归一化力矩
    返回: (B, 22) 归一化预测下一帧
    """
    raw = obj_norm_batch * (std_obj + 1e-8) + mean_obj
    b1, b2 = B1, B2
    th1 = np.arctan2(raw[:, b1 + 3], raw[:, b1 + 4])
    th2 = np.arctan2(raw[:, b2 + 3], raw[:, b2 + 4])
    w1 = raw[:, b1 + 7]; w2 = raw[:, b2 + 7]
    tau = np.asarray(a_norm).squeeze(-1) * TAU_MAX
    c2 = np.cos(th2); s2 = np.sin(th2)
    M11 = (M1 + M2) * L1**2 + M2 * L2**2 + 2 * M2 * L1 * L2 * c2
    M12 = M2 * L2**2 + M2 * L1 * L2 * c2
    M22 = M2 * L2**2
    h1 = (-M2 * L1 * L2 * (2 * w1 + w2) * w2 * s2
          + (M1 + M2) * G_PRIOR * L1 * np.sin(th1)
          + M2 * G_PRIOR * L2 * np.sin(th1 + th2))
    h2 = (M2 * L1 * L2 * w1**2 * s2
          + M2 * G_PRIOR * L2 * np.sin(th1 + th2)
          - tau)
    det = M11 * M22 - M12**2
    det = np.where(np.abs(det) < 1e-9, np.sign(det) * 1e-9, det)
    a1 = (M22 * (-h1) - M12 * (-h2)) / det
    a2 = (-M12 * (-h1) + M11 * (-h2)) / det
    w1n = w1 + a1 * TAU_DT; w2n = w2 + a2 * TAU_DT
    th1n = th1 + w1n * TAU_DT; th2n = th2 + w2n * TAU_DT
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
    return (out - mean_obj) / (std_obj + 1e-8)
