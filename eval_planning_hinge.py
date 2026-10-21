# -*- coding: utf-8 -*-
"""
铰链决策实验：摆锤击靶 (eval_planning_hinge.py)
==================================================
T8 修正条件的检验：铰链世界的"接触/能量调度是任务关键路径"设计。
任务：欠驱动双摆——只控制肘力矩，让手端（连杆2末端）以 ≤40px 距离掠过靶点。
关键路径=能量调度：必须先摆动积蓄能量（中爻）再在正确相位（初爻）出手——
纯位置反馈（reactive）在这类欠驱动任务上著名的无能（能量不可直接由位置推），
而世界模型 MPC 的能量预测（残差头所学）正是先验也有的东西……
所以这里的判别力来自：模型学会了大步长先验 vs 真值引擎的积分差 + 相位时机。

三执行者：
  prior     纯物理先验 CEM（layout_h.acrobot_step 0.05s 欧拉，与模型先验同源）
  model     世界模型 MPC（model_h.rollout，先验+残差）
  energy    能量整形基线（无模型：泵能到目标能量值后保持——经典 Acrobot 技巧）
预注册判据（同 M-D v1/v2 协议）：
  判据 A：model 成功率 ≥ prior + 10pp → 学习残差具有决策价值
  判据 B：model 成功率 > energy（经典启发式基线）→ 可证实用性

用法：python eval_planning_hinge.py --episodes 60
"""

import argparse
import json
import os
import numpy as np
import torch

from layout_h import (OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, DT as DT_H,
                      TAU_MAX, PHYSICS_DT, acrobot_step, acrobot_rk4)
from model_h import WorldModelH

K = 10            # 规划视野（0.5s）
N_CAND = 256
N_ELITE = 32
CEM_ITERS = 2
CTRL_STEPS = 150  # 7.5s
WARMUP = 8
HIT_DIST = 40.0   # 手端掠过靶点 ≤40px 即命中


def target_dist(state, target):
    return float(np.hypot(state[13] - target[0], state[14] - target[1]))


class PriorPlanner:
    """纯先验 CEM：用 layout_h.acrobot_step（0.05s 欧拉）滚动，与模型先验同源。"""
    name = "prior"

    def __init__(self, rng):
        self.rng = rng

    def reset(self):
        pass

    def plan(self, th1, th2, w1, w2, target):
        mean = np.zeros(K)
        std = np.ones(K)
        for _ in range(CEM_ITERS):
            cand = np.clip(mean + std * self.rng.standard_normal((N_CAND, K)), -1, 1)
            t1 = np.full(N_CAND, th1); t2 = np.full(N_CAND, th2)
            v1 = np.full(N_CAND, w1); v2 = np.full(N_CAND, w2)
            dmin = np.full(N_CAND, np.inf)
            for k in range(K):
                # 逐候选滚动（0.05s 欧拉）
                tau = cand[:, k] * TAU_MAX
                # 向量化单步（复制 layout_h.acrobot_step 的方程）
                c2 = np.cos(t2); s2 = np.sin(t2)
                M11 = (M1 + M2) * L1 ** 2 + M2 * L2 ** 2 + 2 * M2 * L1 * L2 * c2
                M12 = M2 * L2 ** 2 + M2 * L1 * L2 * c2
                M22 = M2 * L2 ** 2
                h1 = (-M2 * L1 * L2 * (2 * v1 + v2) * v2 * s2
                      + (M1 + M2) * 900.0 * L1 * np.sin(t1)
                      + M2 * 900.0 * L2 * np.sin(t1 + t2))
                h2 = (M2 * L1 * L2 * v1 ** 2 * s2 + M2 * 900.0 * L2 * np.sin(t1 + t2))
                det = M11 * M22 - M12 ** 2
                a1 = (M22 * (-h1) - M12 * (-h2)) / det
                a2 = (-M12 * (-h1) + M11 * (-h2)) / det
                v1n = v1 + a1 * 0.05; v2n = v2 + a2 * 0.05
                t1n = t1 + v1n * 0.05; t2n = t2 + v2n * 0.05
                # 手端位置
                x2 = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
                y2 = -(L1 * np.cos(t1) + L2 * np.cos(t1 + t2))
                dmin = np.minimum(dmin, np.hypot(x2 - target[0], y2 - target[1]))
                t1, t2, v1, v2 = t1n, t2n, v1n, v2n
            d = dmin + 0.01 * (cand ** 2).sum(axis=1)
            elite = np.argsort(d)[:N_ELITE]
            mean = cand[elite].mean(axis=0)
            std = cand[elite].std(axis=0) + 0.05
        return float(np.clip(mean[0], -1, 1))


from layout_h import L1, L2, M1, M2


class ModelPlanner:
    """世界模型 MPC：model_h.rollout（先验+残差，K=10）。"""
    name = "model"

    def __init__(self, ckpt_path, device, rng):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = WorldModelH(**ck["config"]).to(device)
        self.model.load_state_dict(ck["model_state_dict"], strict=False)
        self.model.eval()
        self.mean_obj = ck["norm_stats"]["mean_obj"]
        self.std_obj = ck["norm_stats"]["std_obj"]
        self.device = device
        self.rng = rng
        self.L = 8
        self.history = None

    def reset(self, frame_norm):
        self.history = [frame_norm] * self.L

    def observe(self, frame_norm):
        self.history.append(frame_norm)
        self.history.pop(0)

    def plan(self, obj_raw, target):
        x = np.stack(self.history)
        xt = torch.from_numpy(np.repeat(x[None], N_CAND, axis=0)).to(self.device)
        mean = np.zeros(K)
        std = np.ones(K)
        with torch.no_grad():
            for _ in range(CEM_ITERS):
                cand = np.clip(mean + std * self.rng.standard_normal((N_CAND, K)), -1, 1)
                at = torch.from_numpy(cand.astype(np.float32)).unsqueeze(-1).to(self.device)
                abs_preds, _ = self.model.rollout(xt, at)
                dmin = np.full(N_CAND, np.inf)
                for k in range(K):
                    raw = abs_preds[k].float().cpu().numpy() * (self.std_obj + 1e-8) + self.mean_obj
                    dmin = np.minimum(dmin, np.hypot(raw[:, 13] - target[0],
                                                     raw[:, 14] - target[1]))
                d = dmin + 0.01 * (cand ** 2).sum(axis=1)
                elite = np.argsort(d)[:N_ELITE]
                mean = cand[elite].mean(axis=0)
                std = cand[elite].std(axis=0) + 0.05
        return float(np.clip(mean[0], -1, 1))


class EnergyPlanner:
    """能量整形基线（经典 Acrobot 启发式，无模型）：
    目标能量 E*（对应手端达靶所需高度），τ = k·(E*−E)·sign(ω2)（能量泵）。"""
    name = "energy"

    def __init__(self):
        self.target = None
        self.kp = 8e-4

    def reset(self, target=None):
        self.target = target

    def plan(self, th1, th2, w1, w2):
        g = 900.0
        y_hand = -(L1 * np.cos(th1) + L2 * np.cos(th1 + th2))
        y_hand_max = L1 + L2
        E_now = (-g * (M1 * L1 * np.cos(th1) + M2 * (L1 * np.cos(th1) + L2 * np.cos(th1 + th2)))
                 + 0.5 * M1 * (L1 * w1) ** 2
                 + 0.5 * M2 * ((L1 * w1) ** 2 + (L2 * (w1 + w2)) ** 2
                               + 2 * L1 * L2 * w1 * (w1 + w2) * np.cos(th2)))
        # 手端在靶点高度时的势能 → 目标能量
        y_t = self.target[1]
        E_star = -g * (M1 * L1 * np.cos(th1) + M2 * (L1 * np.cos(th1) + L2 * np.cos(th1 + th2))) \
            + g * M2 * max(0.0, y_t - y_hand)   # 粗略：M2 抬到靶高所需
        dE = E_star - E_now
        tau = np.clip(self.kp * dE * np.sign(w2 + 1e-9), -1, 1)
        # 相位出手：能量够了就按靶点方向调整
        return float(tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--model_ckpt", type=str, default="checkpoints-hinge/best.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_report", type=str,
                    default="analysis_output/md_hinge_report.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    print("=" * 72)
    print("铰链决策实验：摆锤击靶（欠驱动能量调度任务，判据 A/B 预注册）")
    print("=" * 72)

    model_planner = ModelPlanner(args.model_ckpt, device, np.random.default_rng(7))
    planners = [("prior", PriorPlanner(np.random.default_rng(8)), False),
                ("model", model_planner, True),
                ("energy", EnergyPlanner(), False)]

    results = {n: {"hit": 0, "min_dists": []} for n, _, _ in planners}
    n_eff = 0
    for ep in range(args.episodes):
        seed = 2000 + ep
        rng = np.random.default_rng(seed)
        th1 = rng.uniform(1.2, 2.2)      # 手端低位起始
        th2 = rng.uniform(-0.6, 0.6)
        w1 = w2 = 0.0
        # 靶点：手端初始可达域内随机（距肩 60-150px，下半平面）
        ang = rng.uniform(-0.25 * np.pi, 1.25 * np.pi)
        r = rng.uniform(60, 150)
        target = (r * np.sin(ang), -r * np.cos(ang))

        # 每执行者独立跑同一初始条件
        for name, planner, is_model in planners:
            t1, t2, v1, v2 = th1, th2, w1, w2
            min_d = float("inf")
            hit = False
            if name == "energy":
                planner.reset(target=target)
            if is_model:
                frame = np.zeros(OBJ_DIMS, dtype=np.float32)
                b1, b2 = 0, 11
                x1, y1, x2, y2 = (L1 * np.sin(t1), -L1 * np.cos(t1),
                                  L1 * np.sin(t1) + L2 * np.sin(t1 + t2),
                                  -(L1 * np.cos(t1) + L2 * np.cos(t1 + t2)))
                frame[b1 + 1], frame[b1 + 2], frame[b2 + 1], frame[b2 + 2] = x1, y1, x2, y2
                frame[b1 + 3], frame[b1 + 4] = np.sin(t1), np.cos(t1)
                frame[b2 + 3], frame[b2 + 4] = np.sin(t2), np.cos(t2)
                frame[b1 + 7], frame[b2 + 7] = v1, v2
                frame[b1], frame[b2] = 1, 1
                frame[b1 + 10], frame[b2 + 10] = M1, M2
                model_planner.reset(model_frame(frame))
            for t in range(CTRL_STEPS):
                b1, b2 = 0, 11
                state_vec = np.zeros(OBJ_DIMS, dtype=np.float32)
                x1, y1, x2, y2 = (L1 * np.sin(t1), -L1 * np.cos(t1),
                                  L1 * np.sin(t1) + L2 * np.sin(t1 + t2),
                                  -(L1 * np.cos(t1) + L2 * np.cos(t1 + t2)))
                state_vec[b1 + 1], state_vec[b1 + 2] = x1, y1
                state_vec[b2 + 1], state_vec[b2 + 2] = x2, y2
                state_vec[b1 + 3], state_vec[b1 + 4] = np.sin(t1), np.cos(t1)
                state_vec[b2 + 3], state_vec[b2 + 4] = np.sin(t2), np.cos(t2)
                state_vec[b1 + 7], state_vec[b2 + 7] = v1, v2
                state_vec[b1], state_vec[b2] = 1, 1
                state_vec[b1 + 10], state_vec[b2 + 10] = M1, M2

                if name == "prior":
                    a = planner.plan(t1, t2, v1, v2, target)
                elif name == "model":
                    model_planner.observe(model_frame(state_vec))
                    a = model_planner.plan(state_vec, target)
                else:
                    a = planner.plan(t1, t2, v1, v2)

                tau = a * TAU_MAX
                # 真值引擎：RK4 @0.01s × 5 子步（0.05s 决策间隔）
                for _ in range(5):
                    t1, t2, v1, v2 = acrobot_rk4(t1, t2, v1, v2, tau, dt=PHYSICS_DT)
                x2, y2 = (L1 * np.sin(t1) + L2 * np.sin(t1 + t2),
                          -(L1 * np.cos(t1) + L2 * np.cos(t1 + t2)))
                d = float(np.hypot(x2 - target[0], y2 - target[1]))
                min_d = min(min_d, d)
                if min_d <= HIT_DIST and not hit:
                    hit = True
            results[name]["hit"] += int(hit)
            results[name]["min_dists"].append(min_d)
        n_eff += 1
        if n_eff % 10 == 0:
            line = " | ".join(f"{n} {results[n]['hit']}/{n_eff}" for n, _, _ in planners)
            print(f"  [{n_eff}/{args.episodes}] {line}")

    print("\n" + "=" * 72)
    print(f"{'执行者':<10}{'命中率':>10}{'平均最近距(px)':>16}")
    summary = {}
    for n, _, _ in planners:
        s = results[n]["hit"] / max(n_eff, 1)
        md = float(np.mean(results[n]["min_dists"]))
        summary[n] = {"hit_rate": s, "mean_min_dist": md}
        print(f"{n:<12}{s:>10.1%}{md:>16.2f}")

    a_pp = (summary["model"]["hit_rate"] - summary["prior"]["hit_rate"]) * 100
    b_pp = (summary["model"]["hit_rate"] - summary["energy"]["hit_rate"]) * 100
    print(f"\n判据 A（model − prior ≥ +10pp）: {a_pp:+.1f}pp → {'[OK]' if a_pp >= 10 else '[FAIL]'}")
    print(f"判据 B（model > energy 经典启发式）: {b_pp:+.1f}pp → {'[OK]' if b_pp > 0 else '[FAIL]'}")

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump({"episodes": n_eff, "summary": summary,
                   "criterion_A_pp": float(a_pp), "criterion_B_pp": float(b_pp)}, f,
                  indent=2, ensure_ascii=False)
    print(f"[OK] 报告: {args.output_report}")


def model_frame(state_vec, contact=(0.0, 0.0, 0.0)):
    """组装帧 = [obj(22) | contact(3)]（与训练 25 维一致）。"""
    return np.concatenate([state_vec.copy(),
                           np.asarray(contact, dtype=np.float32)])


if __name__ == "__main__":
    main()
