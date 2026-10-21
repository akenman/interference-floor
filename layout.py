# -*- coding: utf-8 -*-
"""
状态布局单一事实源 (layout.py)
==================================================
所有涉及"状态向量每个维度是什么"的代码（生成器 / 数据集 / 训练 / 评估）
都应从本文件取布局常量，禁止再散落硬编码索引。

v2（旧，兼容保留）
  state 18 维 = 3 物体 × [x, y, angle, vx, vy, ω]，不足 3 物体零填充
  action 6 维 = 3 物体 × [fx, fy]，单步力，归一化除以 FORCE_MAX=500

v3（新）
  组装后每帧 116 维 = 物体 33 + 障碍 80 + 接触 3
    物体（每物体 11 维）:
      [presence, x, y, sinθ, cosθ, vx, vy, ω, type, size, mass]
      type: 0=空槽 1=ball 2=box；size: ball=半径 / box=边长
    障碍（每槽 10 维 × 8 槽，广播到每帧）:
      [is_segment, is_polygon, is_ramp, is_circle, cx, cy, dir_sin, dir_cos, p_len, p_rad]
      segment/ramp: 锚点(cx,cy) + 方向(dir_sin,dir_cos) + p_len=长度
      circle_pin:   圆心(cx,cy) + p_rad=半径
      polygon:      近似编码 cx,cy=质心, p_len=外接圆半径, p_rad=n_verts/6
                    （顶点级精确形状留给 v4，此为已知近似——见 metadata）
      空槽: 全零
    接触（每物体 1 维）: 上一采样间隔内该物体是否发生任何接触（初爻，引擎真实标签）
  action 6 维 = 3 物体 × [impulse_x, impulse_y]，采样间隔内净冲量 ΣF·dt，
      归一化除以 IMPULSE_MAX_V3 = FORCE_MAX × PHYSICS_DT × SAMPLE_INTERVAL = 25
  三爻（辅助监督标签，由数据集按需计算/读取）:
      初爻 接触/腾空   = contact 通道（引擎真值）
      中爻 能量趋势    = sign(d|v|²/dt)（由速度事后计算）
      上爻 意图方向    = sign(d·最近障碍距离/dt)（由状态+障碍布局事后计算）
"""

import numpy as np

# ---------------- v2（旧口径，只读兼容） ----------------
V2 = {
    "per_obj": 6,
    "num_obj": 3,
    "state_dim": 18,
    "pos_dims": [0, 1, 6, 7, 12, 13],
    "vel_dims": [3, 4, 9, 10, 15, 16],
    "action_dim": 6,
    "force_max": 500.0,
}

# ---------------- v3 ----------------
PER_OBJ = 11
NUM_OBJ = 3
OBJ_DIMS = PER_OBJ * NUM_OBJ              # 33
OBS_SLOTS = 8
OBS_PER_SLOT = 10
OBS_DIMS = OBS_SLOTS * OBS_PER_SLOT       # 80
CONTACT_DIMS = NUM_OBJ                    # 3
ASSEMBLED_DIM = OBJ_DIMS + OBS_DIMS + CONTACT_DIMS   # 116

OBJ_FIELDS = ["presence", "x", "y", "sin_t", "cos_t",
              "vx", "vy", "w", "type", "size", "mass"]
TYPE_NONE, TYPE_BALL, TYPE_BOX = 0, 1, 2

# 每物体字段索引（槽内偏移）
F = {name: i for i, name in enumerate(OBJ_FIELDS)}

ACTION_DIM = 6
FORCE_MAX = 500.0          # 与 v2 一致的最大单步力
SAMPLE_INTERVAL_V3 = 5     # 20Hz（v2 为 10）
PHYSICS_DT = 0.01
DT_V3 = PHYSICS_DT * SAMPLE_INTERVAL_V3   # 0.05s 每采样步
IMPULSE_MAX_V3 = FORCE_MAX * PHYSICS_DT * SAMPLE_INTERVAL_V3   # 25.0


def obj_base(oid):
    """第 oid 个物体在物体段(33 维)中的起始下标。"""
    return oid * PER_OBJ


def obj_pos_dims():
    """v3 物体段中的位置维度（x,y × 3 物体）——用于位置 RMSE 等指标。"""
    return [obj_base(o) + F["x"] for o in range(NUM_OBJ)] + \
           [obj_base(o) + F["y"] for o in range(NUM_OBJ)]


def obj_vel_dims():
    """v3 物体段中的平动速度维度（vx,vy × 3 物体）。"""
    return [obj_base(o) + F["vx"] for o in range(NUM_OBJ)] + \
           [obj_base(o) + F["vy"] for o in range(NUM_OBJ)]


def assemble_frame(obj_states, obstacles, contact):
    """(33,) + (80,) + (3,) → (116,)。obstacles 为该片段的槽位矩阵（展平）。"""
    return np.concatenate([obj_states, obstacles, contact])


def assemble_window(obj_win, obstacles_flat, contact_win):
    """(T,33) + (80,) + (T,3) → (T,116)，障碍槽位广播到每帧。"""
    T = obj_win.shape[0]
    return np.concatenate([obj_win,
                           np.repeat(obstacles_flat[None, :], T, axis=0),
                           contact_win], axis=1)


def feat_weights(assembled_dim=ASSEMBLED_DIM):
    """v3 重构/回归的逐维权重的槽内偏移）
    - presence/type/size/mass 与障碍槽位为片段级常量，重构它们是平凡任务 → 0.25
    - 速度物理意义重要 → 2.0
    - 位置/角度/接触 → 1.0
    权重归一化到均值 1（不改变损失总尺度）。
    """
    w = np.ones(assembled_dim, dtype=np.float32)
    for o in range(NUM_OBJ):
        b = obj_base(o)
        w[b + F["presence"]] = 0.25
        w[b + F["type"]] = 0.25
        w[b + F["size"]] = 0.25
        w[b + F["mass"]] = 0.25
        w[b + F["vx"]] = 2.0
        w[b + F["vy"]] = 2.0
    obs_start = OBJ_DIMS
    w[obs_start:obs_start + OBS_DIMS] = 0.25
    w = w / w.mean()
    return w


def physics_prior(s_last_raw, impulse_raw, tau=None, g=-900.0):
    """无接触牛顿动力学先验（含动作冲量与重力）——M1d 残差基座。

    参数:
        s_last_raw:  (M, 33) 上帧物体状态（原始单位）
        impulse_raw: (M, 6)  本间隔净冲量（原始单位，未归一化）
        tau:         采样间隔秒数（默认 DT_V3=0.05）
        g:           重力（像素/s²）
    返回:
        prior_raw: (M, 33) 先验状态
          x' = x + vx·τ            y' = y + vy·τ + ½gτ²
          θ' = θ + ω·τ (sin/cos)   vx' = vx + Ix/m   vy' = vy + Iy/m + g·τ
          ω'/presence/type/size/mass 复制
    无接触区间上该先验与引擎真值之差为 0（已由冲量-Δv 交叉验证背书）；
    残差网络只需学碰撞/接触修正。
    """
    if tau is None:
        tau = DT_V3
    out = s_last_raw.copy()
    for o in range(NUM_OBJ):
        b = obj_base(o)
        x = s_last_raw[:, b + F["x"]]
        y = s_last_raw[:, b + F["y"]]
        st = s_last_raw[:, b + F["sin_t"]]
        ct_ = s_last_raw[:, b + F["cos_t"]]
        vx = s_last_raw[:, b + F["vx"]]
        vy = s_last_raw[:, b + F["vy"]]
        w = s_last_raw[:, b + F["w"]]
        mass = np.where(s_last_raw[:, b + F["mass"]] > 1e-3,
                        s_last_raw[:, b + F["mass"]], 1.0)
        theta2 = np.arctan2(st, ct_) + w * tau
        out[:, b + F["x"]] = x + vx * tau
        out[:, b + F["y"]] = y + vy * tau + 0.5 * g * tau * tau
        out[:, b + F["sin_t"]] = np.sin(theta2)
        out[:, b + F["cos_t"]] = np.cos(theta2)
        out[:, b + F["vx"]] = vx + impulse_raw[:, 2 * o] / mass
        out[:, b + F["vy"]] = vy + impulse_raw[:, 2 * o + 1] / mass + g * tau
    return out


def yao_labels(obj_win_raw, contact_win, obstacles_flat, eps=1e-6):
    """计算三爻标签（逐物体，供爻头监督 / 分析）。

    参数:
        obj_win_raw:   (T, 33) 原始单位物体状态（v3 布局）
        contact_win:   (T, 3)  每帧接触通道（引擎真值）
        obstacles_flat: (80,) 片段障碍槽位
    返回:
        yao: (T, 3, NUM_OBJ) int8，[爻轴, 物体]
          轴0 初爻: 1=接触 / 0=腾空（引擎真值）
          轴1 中爻: 1=动能升 / 0=动能降（该物体 |v|² 差分符号）
          轴2 上爻: 1=趋近 / 0=远离（该物体到最近激活障碍距离的差分符号）
    展平为 9 维顺序 [初(3物体) | 中(3) | 上(3)]，与 model_v3.yao_head 输出一致。
    """
    T = obj_win_raw.shape[0]
    yao = np.zeros((T, 3, NUM_OBJ), dtype=np.int8)
    yao[:, 0, :] = contact_win.astype(np.int8)

    # 中爻：逐物体动能趋势
    v2 = np.zeros((T, NUM_OBJ))
    for o in range(NUM_OBJ):
        b = obj_base(o)
        v2[:, o] = obj_win_raw[:, b + F["vx"]] ** 2 + obj_win_raw[:, b + F["vy"]] ** 2
    de = np.diff(v2, axis=0)
    if T > 1:
        yao[1:, 1, :] = (de > 0).astype(np.int8)
        yao[0, 1, :] = yao[1, 1, :]

    # 上爻：逐物体到最近激活障碍的距离趋势
    obs = obstacles_flat.reshape(OBS_SLOTS, OBS_PER_SLOT)
    active = obs[:, :4].sum(axis=1) > 0
    if active.any():
        cx, cy = obs[active, 4], obs[active, 5]
        for o in range(NUM_OBJ):
            b = obj_base(o)
            pres = obj_win_raw[:, b + F["presence"]] > 0.5
            if not pres.any():
                continue
            d = np.sqrt((obj_win_raw[:, b + F["x"]][:, None] - cx[None, :]) ** 2 +
                        (obj_win_raw[:, b + F["y"]][:, None] - cy[None, :]) ** 2).min(axis=1)
            dn = np.diff(d)
            if T > 1:
                yao[1:, 2, o] = (dn < 0).astype(np.int8)
                yao[0, 2, o] = yao[1, 2, o]
    return yao


if __name__ == "__main__":
    print("v3 组装维度 =", ASSEMBLED_DIM)
    print("位置维度 =", obj_pos_dims())
    print("速度维度 =", obj_vel_dims())
    print("净冲量归一化常数 IMPULSE_MAX_V3 =", IMPULSE_MAX_V3)
    print("feat_weights 均值 =", feat_weights().mean())
