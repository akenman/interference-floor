# -*- coding: utf-8 -*-
"""
E0 决策价值评估 (eval_decision_e0.py) — E0-G 判据
==================================================
在 λ=1.0（最大能量需求）场景上，测 model-MPC − prior-CEM 的成功率差。
配对设计：同一场景同时跑 prior / model / energy（T8 方法学）。

λ 定义（§6.2）：
  E_needed  = 手端到达靶点所需的最小机械能（靶高对应的势能，初态静止）
  E_initial = 初始状态机械能
  E_pumpable= 7.5s 内以最大力矩可泵入的最大能量（数值模拟）
  λ = clamp((E_needed − E_initial) / E_pumpable, 0, 1)

用法：python eval_decision_e0.py --ckpt checkpoints-E0/ours_base_s42/best.pt --lambda 1.0
      python eval_decision_e0.py --all   # 全部 ours_base 种子
"""

import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from layout_h import (OBJ_DIMS, L1, L2, M1, M2, TAU_MAX, PHYSICS_DT,
                      acrobot_rk4, forward_kinematics)
from model_h import WorldModelH


def mechanical_energy(th1, th2, w1, w2):
    """机械能 = PE + KE（g=900，θ 相对竖直向下）。"""
    g = 900.0
    pe = -g * (M1 * L1 * np.cos(th1) + M2 * (L1 * np.cos(th1) + L2 * np.cos(th1 + th2)))
    ke = (0.5 * M1 * (L1 * w1) ** 2
          + 0.5 * M2 * ((L1 * w1) ** 2 + (L2 * (w1 + w2)) ** 2
                        + 2 * L1 * L2 * w1 * (w1 + w2) * np.cos(th2)))
    return pe + ke


def estimate_pumpable(dt=0.05, horizon=7.5):
    """数值估计 E_pumpable：从底部静止出发，最大力矩交替泵能，取峰值能量。"""
    th1 = th2 = 0.0; w1 = w2 = 0.0
    e0 = mechanical_energy(th1, th2, w1, w2)
    peak = e0
    n = int(horizon / dt)
    phase = 0
    for i in range(n):
        # 在摆动过竖直时同向加力（简化泵能策略）
        tau = TAU_MAX if (w2 > 0) else -TAU_MAX
        th1, th2, w1, w2 = acrobot_rk4(th1, th2, w1, w2, tau, dt=dt)
        peak = max(peak, mechanical_energy(th1, th2, w1, w2))
    return peak - e0


PUMPABLE = estimate_pumpable()
E_BOTTOM = mechanical_energy(0.0, 0.0, 0.0, 0.0)


def target_from_lambda(lam, rng, horizontal_spread=15.0):
    """由 λ 反推靶点高度（竖直方向为主），加水平扰动。"""
    y_hand_initial = -(L1 + L2)                     # 底部静止
    y_target = y_hand_initial + (PUMPABLE * lam) / (900.0 * M2)
    y_target = np.clip(y_target, -(L1 + L2) + 5, (L1 + L2) - 5)
    x = rng.uniform(-horizontal_spread, horizontal_spread)
    return float(x), float(y_target)


def lambda_of_target(target, th1=0.0, th2=0.0, w1=0.0, w2=0.0):
    e_initial = mechanical_energy(th1, th2, w1, w2)
    # 到达靶高所需的最小势能（手端抬到 target[1]）
    e_needed = mechanical_energy(th1, th2, 0, 0) + 900.0 * M2 * max(0.0, target[1] - (-(L1 + L2)))
    return float(np.clip((e_needed - e_initial) / PUMPABLE, 0, 1))


# ---------------------------------------------------------------- 三执行者
K = 10
N_CAND, N_ELITE, CEM_ITERS = 256, 32, 2
CTRL_STEPS = 150          # 7.5s @ 0.05s
HIT_DIST = 40.0


class PriorPlanner:
    def __init__(self, rng):
        self.rng = rng

    def plan(self, th1, th2, w1, w2, target):
        mean = np.zeros(K); std = np.ones(K)
        for _ in range(CEM_ITERS):
            cand = np.clip(mean + std * self.rng.standard_normal((N_CAND, K)), -1, 1)
            t1, t2, v1, v2 = (np.full(N_CAND, x) for x in (th1, th2, w1, w2))
            dmin = np.full(N_CAND, np.inf)
            for k in range(K):
                tau = cand[:, k] * TAU_MAX
                c2 = np.cos(t2); s2 = np.sin(t2)
                M11 = (M1 + M2) * L1 ** 2 + M2 * L2 ** 2 + 2 * M2 * L1 * L2 * c2
                M12 = M2 * L2 ** 2 + M2 * L1 * L2 * c2
                M22 = M2 * L2 ** 2
                h1 = (-M2 * L1 * L2 * (2 * v1 + v2) * v2 * s2
                      + (M1 + M2) * 900.0 * L1 * np.sin(t1) + M2 * 900.0 * L2 * np.sin(t1 + t2))
                h2 = (M2 * L1 * L2 * v1 ** 2 * s2 + M2 * 900.0 * L2 * np.sin(t1 + t2))
                det = M11 * M22 - M12 ** 2
                a1 = (M22 * (-h1) - M12 * (-h2)) / det
                a2 = (-M12 * (-h1) + M11 * (-h2)) / det
                v1n = v1 + a1 * 0.05; v2n = v2 + a2 * 0.05
                t1n = t1 + v1n * 0.05; t2n = t2 + v2n * 0.05
                x2 = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
                y2 = -(L1 * np.cos(t1) + L2 * np.cos(t1 + t2))
                dmin = np.minimum(dmin, np.hypot(x2 - target[0], y2 - target[1]))
                t1, t2, v1, v2 = t1n, t2n, v1n, v2n
            d = dmin + 0.01 * (cand ** 2).sum(axis=1)
            elite = np.argsort(d)[:N_ELITE]
            mean, std = cand[elite].mean(axis=0), cand[elite].std(axis=0) + 0.05
        return float(np.clip(mean[0], -1, 1))


class ModelPlanner:
    def __init__(self, ckpt_path, device, rng):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = WorldModelH(**ck["config"]).to(device)
        self.model.load_state_dict(ck["model_state_dict"], strict=False)
        self.model.eval()
        self.mean_obj = ck["norm_stats"]["mean_obj"]
        self.std_obj = ck["norm_stats"]["std_obj"]
        self.device = device
        self.rng = rng
        self.history = None

    def reset(self, frame_norm):
        self.history = [frame_norm] * 8

    def observe(self, frame_norm):
        self.history.append(frame_norm); self.history.pop(0)

    def plan(self, target):
        x = np.stack(self.history)
        xt = torch.from_numpy(np.repeat(x[None], N_CAND, axis=0)).to(self.device)
        mean = np.zeros(K); std = np.ones(K)
        with torch.no_grad():
            for _ in range(CEM_ITERS):
                cand = np.clip(mean + std * self.rng.standard_normal((N_CAND, K)), -1, 1)
                at = torch.from_numpy(cand.astype(np.float32)).unsqueeze(-1).to(self.device)
                abs_preds, _ = self.model.rollout(xt, at)
                dmin = np.full(N_CAND, np.inf)
                for k in range(K):
                    raw = abs_preds[k].float().cpu().numpy() * (self.std_obj + 1e-8) + self.mean_obj
                    dmin = np.minimum(dmin, np.hypot(raw[:, 13] - target[0], raw[:, 14] - target[1]))
                d = dmin + 0.01 * (cand ** 2).sum(axis=1)
                elite = np.argsort(d)[:N_ELITE]
                mean, std = cand[elite].mean(axis=0), cand[elite].std(axis=0) + 0.05
        return float(np.clip(mean[0], -1, 1))


def make_frame(t1, t2, v1, v2):
    """组装 25 维训练帧（obj 22 + contact 3），与 generate_hinge.py 约定一致。"""
    f = np.zeros(OBJ_DIMS, dtype=np.float32)
    b1, b2 = 0, 11
    x1, y1 = L1 * np.sin(t1), -L1 * np.cos(t1)
    x2, y2 = x1 + L2 * np.sin(t1 + t2), y1 - L2 * np.cos(t1 + t2)
    f[b1 + 0], f[b1 + 1], f[b1 + 2] = 1.0, x1, y1
    f[b1 + 3], f[b1 + 4] = np.sin(t1), np.cos(t1)
    f[b1 + 5], f[b1 + 6] = L1 * v1 * np.cos(t1), L1 * v1 * np.sin(t1)
    f[b1 + 7], f[b1 + 8], f[b1 + 9], f[b1 + 10] = v1, 1, L1, M1
    f[b2 + 0], f[b2 + 1], f[b2 + 2] = 1.0, x2, y2
    f[b2 + 3], f[b2 + 4] = np.sin(t2), np.cos(t2)
    f[b2 + 5], f[b2 + 6] = (L1 * v1 * np.cos(t1) + L2 * (v1 + v2) * np.cos(t1 + t2)), \
                           (L1 * v1 * np.sin(t1) + L2 * (v1 + v2) * np.sin(t1 + t2))
    f[b2 + 7], f[b2 + 8], f[b2 + 9], f[b2 + 10] = v2, 2, L2, M2
    return f


def model_frame(state_vec, contact=(0.0, 0.0, 0.0)):
    return np.concatenate([state_vec.copy(), np.asarray(contact, dtype=np.float32)])


def run_scene(target, prior_planner, model_planner, th1=0.0, th2=0.0, w1=0.0, w2=0.0):
    """同一场景跑 prior / model / energy，返回 {name: (hit, min_dist)}."""
    results = {}
    # prior
    t1, t2, v1, v2 = th1, th2, w1, w2
    min_d = float("inf"); hit = False
    for _ in range(CTRL_STEPS):
        a = prior_planner.plan(t1, t2, v1, v2, target)
        tau = a * TAU_MAX
        for _ in range(5):
            t1, t2, v1, v2 = acrobot_rk4(t1, t2, v1, v2, tau, dt=PHYSICS_DT)
        x2, y2 = forward_kinematics(t1, t2)
        d = float(np.hypot(x2 - target[0], y2 - target[1]))
        min_d = min(min_d, d)
        if min_d <= HIT_DIST and not hit:
            hit = True
    results["prior"] = (hit, min_d)
    # model
    t1, t2, v1, v2 = th1, th2, w1, w2
    min_d = float("inf"); hit = False
    model_planner.reset(model_frame(make_frame(t1, t2, v1, v2)))
    for _ in range(CTRL_STEPS):
        model_planner.observe(model_frame(make_frame(t1, t2, v1, v2)))
        a = model_planner.plan(target)
        tau = a * TAU_MAX
        for _ in range(5):
            t1, t2, v1, v2 = acrobot_rk4(t1, t2, v1, v2, tau, dt=PHYSICS_DT)
        x2, y2 = forward_kinematics(t1, t2)
        d = float(np.hypot(x2 - target[0], y2 - target[1]))
        min_d = min(min_d, d)
        if min_d <= HIT_DIST and not hit:
            hit = True
    results["model"] = (hit, min_d)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--lambda", type=float, default=1.0, dest="lam")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--all", action="store_true", help="评估全部 ours_base 种子")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    print(f"E_pumpable(7.5s 最大泵能) = {PUMPABLE:.1f} J  → λ=1.0 靶高 y = {(-(L1+L2) + PUMPABLE/(900*M2)):.1f} px")

    checkpoints = []
    if args.all:
        import glob
        for p in sorted(glob.glob("checkpoints-E0/ours_base_s*/best.pt")):
            checkpoints.append(p)
    elif args.ckpt:
        checkpoints = [args.ckpt]
    else:
        checkpoints = ["checkpoints-E0/ours_base_s42/best.pt"]

    out = {}
    for ckpt in checkpoints:
        rng = np.random.default_rng(2024 + hash(ckpt) % 1000)
        prior_planner = PriorPlanner(np.random.default_rng(8))
        model_planner = ModelPlanner(ckpt, device, np.random.default_rng(9))
        targets = [target_from_lambda(args.lam, rng) for _ in range(args.episodes)]
        lambdas = [lambda_of_target(t) for t in targets]
        hits = {"prior": 0, "model": 0}; sums = {"prior": 0.0, "model": 0.0}
        for i, target in enumerate(targets):
            res = run_scene(target, prior_planner, model_planner)
            for name in ("prior", "model"):
                h, md = res[name]
                hits[name] += int(h); sums[name] += md
            if (i + 1) % 10 == 0:
                print(f"  [{i + 1}/{args.episodes}] prior={hits['prior']}/{i + 1} model={hits['model']}/{i + 1}", flush=True)
        rec = dict(n=args.episodes, lambda_target=args.lam,
                   lambda_actual_mean=float(np.mean(lambdas)), lambda_actual_min=float(np.min(lambdas)),
                   prior_hit_rate=hits["prior"] / args.episodes,
                   model_hit_rate=hits["model"] / args.episodes,
                   diff_pp=(hits["model"] - hits["prior"]) / args.episodes * 100,
                   prior_mean_dist=sums["prior"] / args.episodes,
                   model_mean_dist=sums["model"] / args.episodes)
        out[ckpt] = rec
        print(f"\n[OK] {ckpt}")
        print(f"  λ 实际均值={rec['lambda_actual_mean']:.3f} 最小={rec['lambda_actual_min']:.3f}")
        print(f"  prior 成功率 {rec['prior_hit_rate']:.1%} | model {rec['model_hit_rate']:.1%}")
        print(f"  判据 E0-G（model − prior ≥ +10pp）: {rec['diff_pp']:+.1f}pp "
              f"→ {'[OK]' if rec['diff_pp'] >= 10 else '[FAIL]'}")

    os.makedirs("analysis_output", exist_ok=True)
    with open("analysis_output/E0_decision.json", "w", encoding="utf-8") as f:
        json.dump(dict(pumpable=PUMPABLE, results=out), f, indent=2, ensure_ascii=False)
    print(f"[OK] analysis_output/E0_decision.json")


if __name__ == "__main__":
    main()