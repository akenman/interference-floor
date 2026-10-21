# -*- coding: utf-8 -*-
"""
铰链试验场布局 (layout_h.py) — 新试验场的状态布局单一事实源
==================================================
吸取的全部教训直接进设计：
  - T7：可控域由执行器权限决定 → 关节力矩驱动（权限明确、无摩擦钉死问题）
  - T8：接触须是调度对象 → 摆锤手端接触地面/目标即事件（初爻本质化）
  - 爻=相位：摆动相位 φ = atan2 的连续变量 + 摆动方向，是步态相位的直系前身
  - T1：残差-先验基座 → 先验 = 关节空间半隐式欧拉（无接触假设）

世界定义（Acrobot 式欠驱动双连杆摆）：
  连杆1：长 L1，质量 m1（集中在端点），关节在原点 O（肩）
  连杆2：长 L2，质量 m2（集中在端点），关节在连杆1末端（肘）
  控制：仅肘关节力矩 τ ∈ [-τ_max, τ_max]（肩自由——欠驱动，能量靠摆动积蓄）
  重力 g = -900（与 2D 自由刚体世界一致的尺度语义）

状态（每物体段布局，兼容 v3 管线物体段语义）：
  [presence, x, y, sinθ, cosθ, vx, vy, ω, type, size, mass] × 2 物体
    物体1 = 连杆1末端（肘），物体2 = 连杆2末端（手）
    θ_i = 连杆 i 与竖直向下方向的夹角；ω_i = 角速度
    手端另记关节角（肘角 = θ2 - θ1）与摆动相位 φ
  动作 = [τ_norm]（1 维，肘力矩归一化）

接触通道（3 维，与 v3 兼容）：[手触地, 手在靶区, 肘超限]
  初爻 = 手触地/触靶（接触调度对象）
  中爻 = 系统总能量趋势
  上爻 = 手端朝靶运动（趋近意图）

三爻在此世界的物理语义（对应八卦）：
  初爻：手接触目标/地面（任务接触）
  中爻：能量升降（摆动积蓄/耗散）——欠驱动控制的核心变量
  上爻：手端朝向目标（意图）
"""

import math
import numpy as np

# ---------------- 布局常量（与 v3 兼容的物体段语义） ----------------
PER_OBJ = 11
NUM_OBJ = 2
OBJ_DIMS = PER_OBJ * NUM_OBJ          # 22
CONTACT_DIMS = 3
ACTION_DIM = 1                        # 肘力矩（归一化）

F = {"presence": 0, "x": 1, "y": 2, "sin_t": 3, "cos_t": 4,
     "vx": 5, "vy": 6, "w": 7, "type": 8, "size": 9, "mass": 10}

def obj_base(oid):
    return oid * PER_OBJ

def pos_dims():
    return [obj_base(o) + F["x"] for o in range(NUM_OBJ)] + \
           [obj_base(o) + F["y"] for o in range(NUM_OBJ)]

# ---------------- 物理常量（世界定义） ----------------
L1, L2 = 80.0, 80.0        # 连杆长（px）
M1, M2 = 2.0, 1.0          # 端点集中质量
G = -900.0                 # 重力（与 v3 世界一致）
TAU_MAX = 28000.0          # 肘力矩上限（使 ability to hold 连杆2 水平 ≈ m2*g*L2 的 ~1.6 倍）
PHYSICS_DT = 0.01
SAMPLE_INTERVAL = 5        # 20Hz（与 v3 一致）
DT = PHYSICS_DT * SAMPLE_INTERVAL   # 0.05s

def feat_weights(dim):
    """位置/速度 2.0，角度 1.0，常量属性 0.25（归一化均值 1）。"""
    w = np.ones(dim, dtype=np.float32)
    for o in range(NUM_OBJ):
        b = obj_base(o)
        for f in ("x", "y", "vx", "vy"):
            w[b + F[f]] = 2.0
        for f in ("presence", "type", "size", "mass"):
            w[b + F[f]] = 0.25
    return w / w.mean()


# ---------------- 物理先验（无接触半隐式欧拉，关节空间） ----------------

def forward_kinematics(th1, th2):
    """θ 相对竖直向下。返回肘 (x1,y1) 与手 (x2,y2)。y 向下为负（与 v3 一致，重力 -y）。"""
    x1 = L1 * math.sin(th1)
    y1 = -L1 * math.cos(th1)
    x2 = x1 + L2 * math.sin(th1 + th2)
    y2 = y1 - L2 * math.cos(th1 + th2)
    return x1, y1, x2, y2


def _dynamics(th1, th2, w1, w2, tau):
    """导数向量 (ω1, ω2, α1, α2)。拉格朗日方程（端点集中质量双摆）。"""
    g = -G   # 900
    c2 = math.cos(th2)
    s2 = math.sin(th2)
    M11 = (M1 + M2) * L1 * L1 + M2 * L2 * L2 + 2.0 * M2 * L1 * L2 * c2
    M12 = M2 * L2 * L2 + M2 * L1 * L2 * c2
    M22 = M2 * L2 * L2
    Q1, Q2 = 0.0, tau
    h1 = (-M2 * L1 * L2 * (2.0 * w1 + w2) * w2 * s2
          + (M1 + M2) * g * L1 * math.sin(th1)
          + M2 * g * L2 * math.sin(th1 + th2) - Q1)
    h2 = (M2 * L1 * L2 * w1 * w1 * s2
          + M2 * g * L2 * math.sin(th1 + th2) - Q2)
    det = M11 * M22 - M12 * M12
    if abs(det) < 1e-9:
        det = 1e-9 * (1 if det >= 0 else -1)
    a1 = (M22 * (-h1) - M12 * (-h2)) / det
    a2 = (-M12 * (-h1) + M11 * (-h2)) / det
    return w1, w2, a1, a2


def acrobot_rk4(th1, th2, w1, w2, tau, dt=PHYSICS_DT):
    """RK4 积分（4 阶）——数据生成/真值引擎用。10s 能量漂移 <0.1% 量级。"""
    k1 = _dynamics(th1, th2, w1, w2, tau)
    k2 = _dynamics(th1 + 0.5 * dt * k1[0], th2 + 0.5 * dt * k1[1],
                   w1 + 0.5 * dt * k1[2], w2 + 0.5 * dt * k1[3], tau)
    k3 = _dynamics(th1 + 0.5 * dt * k2[0], th2 + 0.5 * dt * k2[1],
                   w1 + 0.5 * dt * k2[2], w2 + 0.5 * dt * k2[3], tau)
    k4 = _dynamics(th1 + dt * k3[0], th2 + dt * k3[1],
                   w1 + dt * k3[2], w2 + dt * k3[3], tau)
    th1 += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
    th2 += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
    w1 += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
    w2 += dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6.0
    return th1, th2, w1, w2


def acrobot_step(th1, th2, w1, w2, tau, dt=PHYSICS_DT, substeps=1):
    """半隐式欧拉积分双摆动力学（端点集中质量，标准拉格朗日方程）。

    θ 相对竖直向下。M θ̈ + h = Q：
      M11 = (m1+m2)L1² + m2L2² + 2m2L1L2·cosθ2
      M12 = m2L2² + m2L1L2·cosθ2
      M22 = m2L2²
      h1  = m2L1L2·(2ω1+ω2)·ω2·sinθ2 + (m1+m2)g·L1·sinθ1 + m2·g·L2·sin(θ1+θ2) − Q1
      h2  = m2L1L2·ω1²·sinθ2 + m2·g·L2·sin(θ1+θ2) − Q2
    τ 只作用于肘（θ2 关节）。g 取正值 900（势能最低点 = 双杆垂下）。
    返回 (th1, th2, w1, w2)。
    """
    g = -G   # 900，正数进入方程
    for _ in range(substeps):
        c2 = math.cos(th2)
        s2 = math.sin(th2)
        M11 = (M1 + M2) * L1 * L1 + M2 * L2 * L2 + 2.0 * M2 * L1 * L2 * c2
        M12 = M2 * L2 * L2 + M2 * L1 * L2 * c2
        M22 = M2 * L2 * L2
        Q1, Q2 = 0.0, tau
        # 科氏项符号修正（2026-09-03 取证）：h 内应为 −m2L1L2(2ω1+ω2)ω2·sinθ2。
        # 原版 +号与 _dynamics（sympy 5.7e-14 对拍 + 能量守恒背书）相反，
        # 数值表现为能量泵入/衰减（dt=0.05 时 NaN）。管线的 torch 先验
        # （model_h/OursParamModel/HingeData._prior_np）均为正确符号，未受影响。
        h1 = (-M2 * L1 * L2 * (2.0 * w1 + w2) * w2 * s2
              + (M1 + M2) * g * L1 * math.sin(th1)
              + M2 * g * L2 * math.sin(th1 + th2) - Q1)
        h2 = (M2 * L1 * L2 * w1 * w1 * s2
              + M2 * g * L2 * math.sin(th1 + th2) - Q2)
        det = M11 * M22 - M12 * M12
        if abs(det) < 1e-9:
            det = 1e-9 * (1 if det >= 0 else -1)
        a1 = (M22 * (-h1) - M12 * (-h2)) / det
        a2 = (-M12 * (-h1) + M11 * (-h2)) / det
        w1 += a1 * dt
        w2 += a2 * dt
        th1 += w1 * dt
        th2 += w2 * dt
    return th1, th2, w1, w2


def physics_prior_state(state, action_norm):
    """无接触物理先验（T1 基座）：状态 → 下一帧（0.05s，动作恒定）。

    state: (22,) 原始单位。action_norm: 标量或 (1,) 归一化肘力矩。
    """
    b1, b2 = obj_base(0), obj_base(1)
    th1 = math.atan2(state[b1 + F["sin_t"]], state[b1 + F["cos_t"]])
    th2 = math.atan2(state[b2 + F["sin_t"]], state[b2 + F["cos_t"]])
    w1 = state[b1 + F["w"]]
    w2 = state[b2 + F["w"]]
    tau = float(np.clip(action_norm, -1, 1)) * TAU_MAX
    th1, th2, w1, w2 = acrobot_step(th1, th2, w1, w2, tau, dt=DT)
    out = state.copy()
    x1, y1, x2, y2 = forward_kinematics(th1, th2)
    # 速度用差分（0.05s 大步长差分 vs 瞬时角速度——先验自洽即可，残差学修正）
    th1_p, th2_p, w1_p, w2_p = th1, th2, w1, w2
    x1p, y1p, x2p, y2p = forward_kinematics(th1_p, th2_p)
    out[b1 + F["x"]], out[b1 + F["y"]] = x1, y1
    out[b2 + F["x"]], out[b2 + F["y"]] = x2, y2
    out[b1 + F["sin_t"]], out[b1 + F["cos_t"]] = math.sin(th1), math.cos(th1)
    out[b2 + F["sin_t"]], out[b2 + F["cos_t"]] = math.sin(th2), math.cos(th2)
    out[b1 + F["vx"]] = (x1p - state[b1 + F["x"]]) / DT
    out[b1 + F["vy"]] = (y1p - state[b1 + F["y"]]) / DT
    out[b2 + F["vx"]] = (x2p - state[b2 + F["x"]]) / DT
    out[b2 + F["vy"]] = (y2p - state[b2 + F["y"]]) / DT
    out[b1 + F["w"]], out[b2 + F["w"]] = w1, w2
    return out


def yao_labels_h(states, contact, target=None):
    """三爻标签（铰链世界语义）。

    states: (T, 22) 原始单位；contact: (T, 3) 引擎接触真值；target: (2,) 或 None。
    返回 (T, 3, NUM_OBJ)：[轴, 物体]。轴0=初爻接触（广播到手端事件到两物体），
    轴1=能量趋势，轴2=趋近意图。
    """
    T = states.shape[0]
    yao = np.zeros((T, 3, NUM_OBJ), dtype=np.int8)
    # 手端事件（contact 通道0）作为初爻，广播到两物体（M2 随肘被动运动）；
    # 通道1/2（靶区/警戒）不进爻标签——它们是任务事件而非相态。
    yao[:, 0, :] = contact[:, 0:1].astype(np.int8)
    # 中爻：系统总动能趋势（逐物体 |v|²）
    for o in range(NUM_OBJ):
        b = obj_base(o)
        v2 = states[:, b + F["vx"]] ** 2 + states[:, b + F["vy"]] ** 2
        if T > 1:
            yao[1:, 1, o] = (np.diff(v2) > 0).astype(np.int8)
            yao[0, 1, o] = yao[1, 1, o]
    # 上爻：手端朝靶（target 给定时），否则手端速度方向一致性
    b2 = obj_base(1)
    if target is not None:
        d = np.hypot(states[:, b2 + F["x"]] - target[0],
                     states[:, b2 + F["y"]] - target[1])
        if T > 1:
            yao[1:, 2, 1] = (np.diff(d) < 0).astype(np.int8)
            yao[0, 2, 1] = yao[1, 2, 1]
    return yao


def yao_labels_h_v2(states, contact, target=None):
    """三爻标签 v2（W2：修物体0 上爻恒 0 的标签缺陷）。

    与 yao_labels_h 的唯一差异 —— **上爻改为逐物体定义**：
      v1: 上爻只写物体1（手端趋近靶），物体0 的上爻恒 0 ⇒ 物体0 只有
          4/8 个卦可达 ⇒ GuaMoE 的 64 专家中至少 32 个构造上永不可达。
      v2: 上爻对两个物体都用「该物体的端点是否在趋近靶」定义
          （物体0 = 肘端趋近靶，物体1 = 手端趋近靶），使 8 卦全部可达。

    刻意**未改动**的部分（保持与 v1 的可比性）：
      - 轴0 初爻仍是 contact[:,0] 广播到两物体（物体0 的初爻描述的其实是手）。
        这是独立的语义瑕疵，但不影响 8 卦可达性，留作后续 W2b。
      - 轴1 中爻动能趋势、通道1/2 不进爻，均与 v1 一致。

    v1 保持原样不改动：E0/E1a 全部已落盘数据均由 v1 生成，改写会破坏
    已预注册实验的可复现性。

    states: (T, 22)；contact: (T, 3)；target: (2,) 或 None
    返回 (T, 3, NUM_OBJ)
    """
    T = states.shape[0]
    yao = np.zeros((T, 3, NUM_OBJ), dtype=np.int8)
    yao[:, 0, :] = contact[:, 0:1].astype(np.int8)          # 初爻：同 v1
    for o in range(NUM_OBJ):                                # 中爻：同 v1
        b = obj_base(o)
        v2 = states[:, b + F["vx"]] ** 2 + states[:, b + F["vy"]] ** 2
        if T > 1:
            yao[1:, 1, o] = (np.diff(v2) > 0).astype(np.int8)
            yao[0, 1, o] = yao[1, 1, o]
    for o in range(NUM_OBJ):                                # 上爻：v2 逐物体
        b = obj_base(o)
        if target is not None:
            d = np.hypot(states[:, b + F["x"]] - target[0],
                         states[:, b + F["y"]] - target[1])
        else:                                                # 无靶：速度方向一致性
            d = -np.hypot(states[:, b + F["vx"]], states[:, b + F["vy"]])
        if T > 1:
            yao[1:, 2, o] = (np.diff(d) < 0).astype(np.int8)
            yao[0, 2, o] = yao[1, 2, o]
    return yao
