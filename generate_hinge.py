# -*- coding: utf-8 -*-
"""
铰链世界数据生成器 (generate_hinge.py)
==================================================
RK4 真值引擎（已验证：能量漂移 1e-3 级）+ 随机力矩策略序列。
输出 trajectories_hinge.npz:
    states    (N, T, 22)  物体段（layout_h 布局，原始单位）
    actions   (N, T, 1)   归一化肘力矩
    contact   (N, T, 3)   三爻接触通道真值（手触地 / 手在靶区 / 能量警戒）
    metadata  json        布局与生成配置

力矩策略（保证数据覆盖可控域）：
  分段常数 + 平滑过渡：每 0.25~0.75s 切换目标力矩档位 ∈ {-1,-0.5,0,0.5,1}，
  线性过渡 0.1s——产生泵能/卸能/保持的多样轨迹。
接触事件（layout_h 语义）：
  手触地（y >= ground_y）        → 初爻=1
  手在靶区（距靶 < target_r）    → 事件通道
  系统能量 > energy_warn         → 警戒通道（高速域）

用法：python generate_hinge.py --episodes 3000 --output trajectories_hinge.npz
"""

import os
import json
import math
import argparse
import numpy as np

from layout_h import (OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, NUM_OBJ, PER_OBJ, F as LF,
                      L1, L2, M1, M2, TAU_MAX, PHYSICS_DT, SAMPLE_INTERVAL, DT,
                      acrobot_rk4, forward_kinematics, yao_labels_h)

SEQ_LEN = 200          # 10s @ 20Hz（与 v3 一致）
GROUND_Y = -150.0      # 地面高度：手端最低 -(L1+L2)=-160，地面设 -150，
                       # 触地 = y2 <= GROUND_Y（手端摆到离最低点 10px 内 = 接近接触）
TARGET_R = 40.0


def make_tau_schedule(rng, n_phys_steps):
    """分段常数 + 线性过渡的力矩序列（n_phys_steps 个物理步）。"""
    tau = np.zeros(n_phys_steps, dtype=np.float64)
    i = 0
    level = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
    while i < n_phys_steps:
        hold = int(rng.uniform(0.25, 0.75) / PHYSICS_DT)
        trans = int(0.1 / PHYSICS_DT)
        new_level = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
        for j in range(min(trans, n_phys_steps - i)):
            tau[i + j] = level + (new_level - level) * (j + 1) / trans
        i += trans
        for j in range(min(hold, n_phys_steps - i)):
            tau[i + j] = new_level
        i += hold
        level = new_level
    return np.clip(tau, -1, 1) * TAU_MAX


def run_episode(rng, target):
    """单条轨迹：RK4 真值 + 20Hz 采样。返回 (states, contact, actions)。"""
    total_phys = SEQ_LEN * SAMPLE_INTERVAL
    tau_seq = make_tau_schedule(rng, total_phys)

    th1 = rng.uniform(0.4, 2.6)
    th2 = rng.uniform(-1.2, 1.2)
    w1 = rng.uniform(-2.0, 2.0)
    w2 = rng.uniform(-2.0, 2.0)

    states = np.zeros((SEQ_LEN, OBJ_DIMS), dtype=np.float32)
    contact = np.zeros((SEQ_LEN, CONTACT_DIMS), dtype=np.float32)
    actions = np.zeros((SEQ_LEN, ACTION_DIM), dtype=np.float32)

    s_i = 0
    for step in range(total_phys):
        th1, th2, w1, w2 = acrobot_rk4(th1, th2, w1, w2, tau_seq[step], dt=PHYSICS_DT)
        if (step + 1) % SAMPLE_INTERVAL == 0:
            x1, y1, x2, y2 = forward_kinematics(th1, th2)
            b1, b2 = 0, PER_OBJ
            states[s_i, b1 + LF["presence"]] = 1.0
            states[s_i, b1 + LF["x"]], states[s_i, b1 + LF["y"]] = x1, y1
            states[s_i, b1 + LF["sin_t"]], states[s_i, b1 + LF["cos_t"]] = math.sin(th1), math.cos(th1)
            # 速度差分（相对上一采样帧；首帧用小步长差分近似）
            states[s_i, b2 + LF["presence"]] = 1.0
            states[s_i, b2 + LF["x"]], states[s_i, b2 + LF["y"]] = x2, y2
            states[s_i, b2 + LF["sin_t"]], states[s_i, b2 + LF["cos_t"]] = math.sin(th2), math.cos(th2)
            states[s_i, b1 + LF["w"]], states[s_i, b2 + LF["w"]] = w1, w2
            states[s_i, b1 + LF["mass"]], states[s_i, b2 + LF["mass"]] = M1, M2
            states[s_i, b1 + LF["size"]], states[s_i, b2 + LF["size"]] = L1, L2
            states[s_i, b1 + LF["type"]], states[s_i, b2 + LF["type"]] = 1, 2
            # 归一化力矩 -> 动作。TAU_MAX<=0（无力矩上限）时物理上无驱动，
            # 动作定义为 0，避免 0/0 -> NaN 污染整条轨迹。
            actions[s_i, 0] = (tau_seq[step] / TAU_MAX) if TAU_MAX > 0 else 0.0
            # 接触通道
            contact[s_i, 0] = 1.0 if y2 <= GROUND_Y else 0.0                 # 手触地（近地面）
            contact[s_i, 1] = 1.0 if math.hypot(x2 - target[0], y2 - target[1]) <= TARGET_R else 0.0
            v2 = states[s_i, b2 + LF["vx"]] ** 2 + states[s_i, b2 + LF["vy"]] ** 2
            contact[s_i, 2] = 1.0 if v2 > 4e4 else 0.0                        # 手速警戒 (>200px/s)
            s_i += 1

    # 速度差分回填（vx,vy = 位置差分 / DT；末帧复制前帧）
    for o in range(NUM_OBJ):
        b = o * PER_OBJ
        states[1:, b + LF["vx"]] = (states[1:, b + LF["x"]] - states[:-1, b + LF["x"]]) / DT
        states[1:, b + LF["vy"]] = (states[1:, b + LF["y"]] - states[:-1, b + LF["y"]]) / DT
        states[0, b + LF["vx"]] = states[1, b + LF["vx"]]
        states[0, b + LF["vy"]] = states[1, b + LF["vy"]]
    return states, contact, actions


def main():
    parser = argparse.ArgumentParser(description="铰链世界数据生成器")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="trajectories_hinge.npz")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print("=" * 60)
    print("铰链世界数据生成（RK4 真值引擎）")
    print(f"  片段: {args.episodes} | 序列: {SEQ_LEN} @ 20Hz | 力矩档位 ±{{0.5,1}}×τmax")
    print("=" * 60)

    all_states = np.zeros((args.episodes, SEQ_LEN, OBJ_DIMS), dtype=np.float32)
    all_contact = np.zeros((args.episodes, SEQ_LEN, CONTACT_DIMS), dtype=np.float32)
    all_actions = np.zeros((args.episodes, SEQ_LEN, ACTION_DIM), dtype=np.float32)
    all_targets = np.zeros((args.episodes, 2), dtype=np.float32)

    for ep in range(args.episodes):
        target = (rng.uniform(-120, 120), rng.uniform(-140, -40))
        all_targets[ep] = target
        st, ct, ac = run_episode(rng, target)
        all_states[ep] = st
        all_contact[ep] = ct
        all_actions[ep] = ac
        if (ep + 1) % 500 == 0:
            print(f"  [{(ep + 1) / args.episodes * 100:5.1f}%] {ep + 1:,}")

    # 三爻标签（中爻/上爻）
    yao = np.zeros((args.episodes, SEQ_LEN, 3, NUM_OBJ), dtype=np.int8)
    for ep in range(args.episodes):
        yao[ep] = yao_labels_h(all_states[ep], all_contact[ep], all_targets[ep])

    np.savez_compressed(args.output, states=all_states, contact=all_contact,
                        actions=all_actions, yao=yao, targets=all_targets)
    mb = os.path.getsize(args.output) / (1024 ** 2)
    print(f"\n[OK] 保存至 {args.output} ({mb:.1f} MB)")
    print(f"     states {all_states.shape} | contact {all_contact.shape} | actions {all_actions.shape}")
    print(f"     手触地帧占比: {all_contact[:, :, 0].mean():.1%} | 靶区命中帧: {all_contact[:, :, 1].mean():.1%}"
          f" | 高速警戒帧: {all_contact[:, :, 2].mean():.1%}")

    meta = {
        "version": "hinge-v1",
        "episodes": args.episodes, "seq_length": SEQ_LEN, "sample_rate_hz": 20.0,
        "layout": {"obj_dims": OBJ_DIMS, "contact_dims": CONTACT_DIMS, "action_dim": ACTION_DIM},
        "physics": {"L1": L1, "L2": L2, "M1": M1, "M2": M2, "g": 900.0,
                    "tau_max": TAU_MAX, "integrator": "RK4@0.01s", "dt_sample": DT},
        "contact_channels": ["手触地", "手在靶区", "手速警戒"],
        "seed": args.seed,
    }
    with open(args.output.replace(".npz", "_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
