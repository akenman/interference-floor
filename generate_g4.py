# -*- coding: utf-8 -*-
"""
G4 真动力学偏移数据生成器 v2 (generate_g4.py)
==================================================
T5/G4 的检验场：先验（硬编码 g=900、方程类无阻力/摩擦）在偏移世界中的
资产/负债属性。两个偏移世界（观测通道保持干净——与 T16 正交）：

  g125 : g = 1125（+25%）      —— 参数类偏移，NM-5 方程类内可辨识
  drag : cd=5e-3, cf=3000      —— 加性类偏移，NM-5 类外、NM-7（+cd/cf）类内

预检确立的两个设计事实（2026-09-03）：
  ① 加性 v² 阻力自限：系统减速 → 阻力自缩 → 单步先验误差反而变小
     （剂量-响应只在 rollout 域存在：clean 0 / g125 2.76px / drag 19.24px @k=5，
      单步仅 0.15/0.56px——T11 混沌放大是参数误差的显示器）
  ② layout_h.acrobot_step 的科氏项符号错误已修（_dynamics 为准：
     sympy 5.7e-14 对拍 + 能量守恒背书；本引擎与 acrobot_rk4 逐位对拍）

用法：python generate_g4.py --episodes 600
"""

import os
import json
import math
import argparse
import numpy as np

from layout_h import (OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, NUM_OBJ, PER_OBJ, F as LF,
                      L1, L2, M1, M2, TAU_MAX, PHYSICS_DT, SAMPLE_INTERVAL, DT,
                      acrobot_rk4, forward_kinematics, yao_labels_h)

SEQ_LEN = 200
GROUND_Y = -150.0
TARGET_R = 40.0

WORLDS = {
    "g125": {"g": 1125.0, "cd": 0.0, "cf": 0.0},
    "drag": {"g": 900.0, "cd": 5e-3, "cf": 3000.0},
}


def _dynamics_ext(th1, th2, w1, w2, tau, g, cd, cf):
    """扩展动力学导数（与 layout_h._dynamics 严格同构 + 参数化 g + 摩擦/阻力）。"""
    c2, s2 = math.cos(th2), math.sin(th2)
    M11 = (M1 + M2) * L1 * L1 + M2 * L2 * L2 + 2.0 * M2 * L1 * L2 * c2
    M12 = M2 * L2 * L2 + M2 * L1 * L2 * c2
    M22 = M2 * L2 * L2
    h1 = (-M2 * L1 * L2 * (2.0 * w1 + w2) * w2 * s2
          + (M1 + M2) * g * L1 * math.sin(th1)
          + M2 * g * L2 * math.sin(th1 + th2))
    h2 = (M2 * L1 * L2 * w1 * w1 * s2
          + M2 * g * L2 * math.sin(th1 + th2))
    c1, s1 = math.cos(th1), math.sin(th1)
    c12, s12 = math.cos(th1 + th2), math.sin(th1 + th2)
    v1x, v1y = L1 * w1 * c1, L1 * w1 * s1
    v2x, v2y = v1x + L2 * (w1 + w2) * c12, v1y + L2 * (w1 + w2) * s12
    if cd > 0.0:
        f1x = -cd * math.hypot(v1x, v1y) * v1x
        f1y = -cd * math.hypot(v1x, v1y) * v1y
        f2x = -cd * math.hypot(v2x, v2y) * v2x
        f2y = -cd * math.hypot(v2x, v2y) * v2y
        qd1 = f1x * (L1 * c1) + f1y * (L1 * s1) + f2x * (L2 * c12) + f2y * (L2 * s12)
        qd2 = f2x * (L2 * c12) + f2y * (L2 * s12)
    else:
        qd1 = qd2 = 0.0
    Q1 = -cf * w1 + qd1
    Q2 = tau - cf * w2 + qd2
    h1 -= Q1
    h2 -= Q2
    det = M11 * M22 - M12 * M12
    if abs(det) < 1e-9:
        det = 1e-9 * (1 if det >= 0 else -1)
    a1 = (M22 * (-h1) - M12 * (-h2)) / det
    a2 = (-M12 * (-h1) + M11 * (-h2)) / det
    return w1, w2, a1, a2


def acrobot_rk4_ext(th1, th2, w1, w2, tau, g, cd, cf, dt=PHYSICS_DT):
    k1 = _dynamics_ext(th1, th2, w1, w2, tau, g, cd, cf)
    k2 = _dynamics_ext(th1 + 0.5 * dt * k1[0], th2 + 0.5 * dt * k1[1],
                       w1 + 0.5 * dt * k1[2], w2 + 0.5 * dt * k1[3], tau, g, cd, cf)
    k3 = _dynamics_ext(th1 + 0.5 * dt * k2[0], th2 + 0.5 * dt * k2[1],
                       w1 + 0.5 * dt * k2[2], w2 + 0.5 * dt * k2[3], tau, g, cd, cf)
    k4 = _dynamics_ext(th1 + dt * k3[0], th2 + dt * k3[1],
                       w1 + dt * k3[2], w2 + dt * k3[3], tau, g, cd, cf)
    th1 += dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
    th2 += dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
    w1 += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
    w2 += dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6.0
    return th1, th2, w1, w2


def make_tau_schedule(rng, n_phys_steps):
    """与 generate_hinge 完全一致的力矩策略。"""
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


def run_episode(rng, world):
    """单条轨迹：扩展 RK4 真值 + 20Hz 采样（布局与 generate_hinge 一致）。"""
    g, cd, cf = world["g"], world["cd"], world["cf"]
    total_phys = SEQ_LEN * SAMPLE_INTERVAL
    tau_seq = make_tau_schedule(rng, total_phys)

    th1 = rng.uniform(0.4, 2.6)
    th2 = rng.uniform(-1.2, 1.2)
    w1 = rng.uniform(-2.0, 2.0)
    w2 = rng.uniform(-2.0, 2.0)

    states = np.zeros((SEQ_LEN, OBJ_DIMS), dtype=np.float32)
    contact = np.zeros((SEQ_LEN, CONTACT_DIMS), dtype=np.float32)
    actions = np.zeros((SEQ_LEN, ACTION_DIM), dtype=np.float32)
    target = (rng.uniform(-120, 120), rng.uniform(-140, -40))

    s_i = 0
    for step in range(total_phys):
        th1, th2, w1, w2 = acrobot_rk4_ext(th1, th2, w1, w2, tau_seq[step], g, cd, cf)
        if (step + 1) % SAMPLE_INTERVAL == 0:
            x1, y1, x2, y2 = forward_kinematics(th1, th2)
            b1, b2 = 0, PER_OBJ
            states[s_i, b1 + LF["presence"]] = 1.0
            states[s_i, b1 + LF["x"]], states[s_i, b1 + LF["y"]] = x1, y1
            states[s_i, b1 + LF["sin_t"]], states[s_i, b1 + LF["cos_t"]] = math.sin(th1), math.cos(th1)
            states[s_i, b2 + LF["presence"]] = 1.0
            states[s_i, b2 + LF["x"]], states[s_i, b2 + LF["y"]] = x2, y2
            states[s_i, b2 + LF["sin_t"]], states[s_i, b2 + LF["cos_t"]] = math.sin(th2), math.cos(th2)
            states[s_i, b1 + LF["w"]], states[s_i, b2 + LF["w"]] = w1, w2
            states[s_i, b1 + LF["mass"]], states[s_i, b2 + LF["mass"]] = M1, M2
            states[s_i, b1 + LF["size"]], states[s_i, b2 + LF["size"]] = L1, L2
            states[s_i, b1 + LF["type"]], states[s_i, b2 + LF["type"]] = 1, 2
            actions[s_i, 0] = (tau_seq[step] / TAU_MAX) if TAU_MAX > 0 else 0.0
            contact[s_i, 0] = 1.0 if y2 <= GROUND_Y else 0.0
            contact[s_i, 1] = 1.0 if math.hypot(x2 - target[0], y2 - target[1]) <= TARGET_R else 0.0
            s_i += 1

    for o in range(NUM_OBJ):
        b = o * PER_OBJ
        states[1:, b + LF["vx"]] = (states[1:, b + LF["x"]] - states[:-1, b + LF["x"]]) / DT
        states[1:, b + LF["vy"]] = (states[1:, b + LF["y"]] - states[:-1, b + LF["y"]]) / DT
        states[0, b + LF["vx"]] = states[1, b + LF["vx"]]
        states[0, b + LF["vy"]] = states[1, b + LF["vy"]]
    return states, contact, actions, target


def pilot_check():
    """对拍 + rollout 域剂量-响应（先于任何全量生成）。"""
    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(50):
        th1, th2 = rng.uniform(-3, 3), rng.uniform(-3, 3)
        w1, w2 = rng.uniform(-30, 30), rng.uniform(-30, 30)
        tau = rng.uniform(-1, 1) * TAU_MAX
        a1 = acrobot_rk4(th1, th2, w1, w2, tau)
        a2 = acrobot_rk4_ext(th1, th2, w1, w2, tau, 900.0, 0.0, 0.0)
        max_diff = max(max_diff, max(abs(x - y) for x, y in zip(a1, a2)))
    print(f"[对拍] g=900,cd=cf=0 vs layout_h.acrobot_rk4: max|Δ| = {max_diff:.3e} "
          f"{'[OK]' if max_diff < 1e-9 else '[FAIL]'}")

    def prior_rollout_err(world, n=40):
        """固定 clean 先验（g=900,无阻力）迭代 rollout k=5 的位置误差。"""
        errs = []
        for _ in range(n):
            th1, th2 = rng.uniform(0.4, 2.6), rng.uniform(-1.2, 1.2)
            w1, w2 = rng.uniform(-6, 6), rng.uniform(-6, 6)
            t1, t2, u1, u2 = th1, th2, w1, w2
            p1, p2, v1, v2 = th1, th2, w1, w2
            for k in range(5):
                t1, t2, u1, u2 = acrobot_rk4_ext(t1, t2, u1, u2, 0.0,
                                                 world["g"], world["cd"], world["cf"])
                p1, p2, v1, v2 = acrobot_rk4_ext(p1, p2, v1, v2, 0.0, 900.0, 0.0, 0.0)
                if k == 0:
                    e1 = 0.5 * (abs(L1 * math.sin(p1) - L1 * math.sin(t1))
                                + abs(L1 * math.sin(p1) + L2 * math.sin(p1 + p2)
                                      - L1 * math.sin(t1) - L2 * math.sin(t1 + t2)))
            x_t = np.array([L1 * math.sin(t1), L1 * math.sin(t1) + L2 * math.sin(t1 + t2)])
            x_p = np.array([L1 * math.sin(p1), L1 * math.sin(p1) + L2 * math.sin(p1 + p2)])
            errs.append((np.abs(x_t - x_p).mean(), e1))
        e5 = float(np.mean([e[0] for e in errs]))
        e1 = float(np.mean([e[1] for e in errs]))
        return e5, e1

    print("[rollout 剂量-响应] 固定 clean 先验迭代 k=5（无动作，自由摆）：")
    e5c, e1c = prior_rollout_err({"g": 900.0, "cd": 0.0, "cf": 0.0})
    print(f"  clean   k5={e5c:8.2f}px | k1={e1c:.2f}px")
    for name, w in WORLDS.items():
        e5, e1 = prior_rollout_err(w)
        print(f"  {name:<8} k5={e5:8.2f}px | k1={e1:.2f}px")


def main():
    ap = argparse.ArgumentParser(description="G4 真动力学偏移数据生成")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_pilot", action="store_true")
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()
    # 铁律 12（SLOT-2 首轮启动时抓出）：--outdir 不建目录，savez 在子目录上崩
    os.makedirs(args.outdir, exist_ok=True)

    if not args.skip_pilot:
        pilot_check()

    for name, world in WORLDS.items():
        rng = np.random.default_rng(args.seed)
        out = os.path.join(args.outdir, f"g4_{name}.npz")
        all_states = np.zeros((args.episodes, SEQ_LEN, OBJ_DIMS), dtype=np.float32)
        all_contact = np.zeros((args.episodes, SEQ_LEN, CONTACT_DIMS), dtype=np.float32)
        all_actions = np.zeros((args.episodes, SEQ_LEN, ACTION_DIM), dtype=np.float32)
        all_targets = np.zeros((args.episodes, 2), dtype=np.float32)
        for ep in range(args.episodes):
            st, ct, ac, tg = run_episode(rng, world)
            all_states[ep], all_contact[ep] = st, ct
            all_actions[ep], all_targets[ep] = ac, tg
            if (ep + 1) % 200 == 0:
                print(f"  [{name}] {ep + 1}/{args.episodes}")
        yao = np.zeros((args.episodes, SEQ_LEN, 3, NUM_OBJ), dtype=np.int8)
        for ep in range(args.episodes):
            yao[ep] = yao_labels_h(all_states[ep], all_contact[ep], all_targets[ep])
        np.savez_compressed(out, states=all_states, contact=all_contact,
                            actions=all_actions, yao=yao, targets=all_targets)
        meta = {"version": "g4-hinge-v2", "world": name, **world,
                "episodes": args.episodes, "seq_length": SEQ_LEN,
                "physics": {"L1": L1, "L2": L2, "M1": M1, "M2": M2,
                            "tau_max": TAU_MAX, "integrator": "RK4@0.01s(+drag/friction)"},
                "seed": args.seed}
        with open(out.replace(".npz", "_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[OK] {out}")


if __name__ == "__main__":
    main()
