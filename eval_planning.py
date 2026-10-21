# -*- coding: utf-8 -*-
"""
M-D v2 决策实验：接触丰富任务（吸取 T8 教训）
==================================================
M-D v1 判决：腾空低速导引域里 model=prior（+0pp）——残差（接触修正）无出场机会。
v2 设计原则：把接触变成任务的关键路径——
  任务"绕墙导引"：靶区与球之间放置一道垂直隔板，直线弹道被挡死，
  唯一路径是利用边界墙反弹绕行（反弹点选择 = 接触预测问题）。
  prior（无墙/障碍知识）的 CEM 只会把球推向隔板；model 的接触预测成为
  唯一优势来源——判据 A 的检验力最大化设计。
三执行者与预注册判据不变：
  判据 A：model 成功率 ≥ prior + 10pp → 学习残差具有决策价值
  判据 B：model 成功率 > reactive → 世界模型优于无模型反馈控制

用法：
  python eval_planning.py --episodes 60      # pilot: --episodes 10
"""

import argparse
import copy
import json
import os
import random
import numpy as np
import torch

from generate_trajectories import generate_scene

from layout import (OBJ_DIMS, OBS_DIMS, ASSEMBLED_DIM, ACTION_DIM, DT_V3,
                    IMPULSE_MAX_V3, PER_OBJ, F as LF, physics_prior)
from model_v3 import WorldModelV3
from env_task import PushTargetEnv

K = 10           # 规划视野（步数，0.5s）
N_CAND = 256     # CEM 候选数
N_ELITE = 32
CEM_ITERS = 2
CTRL_STEPS = 200  # 控制步数（10s @ 20Hz）——绕行需要更长时程
WARMUP = 8        # 历史填充步（零动作，三执行者统一）
MIN_DIST_SUCCESS = 50.0   # 途经靶区（最近距离 ≤50px）即成功
MAX_SCENE_RETRY = 40      # 可控域筛选重试上限（T7：接地不可控 + 高速不可控）
MAX_OBJ0_SPEED = 150.0    # 控制起点物体 0 速度上限


# ---------------- 执行者 1：纯物理先验（解析滚动，无网络） ----------------

class PriorPlanner:
    name = "prior"

    def __init__(self, rng):
        self.rng = rng

    def reset(self, **kw):
        pass

    def plan(self, obj_raw, target, history=None):
        mean = np.zeros((K, 2))
        std = np.ones((K, 2))
        for _ in range(CEM_ITERS):
            cand = np.clip(mean + std * self.rng.standard_normal((N_CAND, K, 2)), -1, 1)
            states = np.repeat(obj_raw[None, :], N_CAND, axis=0)
            dist_min = np.full(N_CAND, np.inf)
            for k in range(K):
                act = np.zeros((N_CAND, 6), dtype=np.float32)
                act[:, 0:2] = cand[:, k]
                states = physics_prior(states, act * IMPULSE_MAX_V3)
                dist_min = np.minimum(dist_min, np.hypot(states[:, LF["x"]] - target[0],
                                                         states[:, LF["y"]] - target[1]))
            d = dist_min + 0.01 * (cand ** 2).sum(axis=(1, 2))
            elite = np.argsort(d)[:N_ELITE]
            mean = cand[elite].mean(axis=0)
            std = cand[elite].std(axis=0) + 0.05
        return np.clip(mean[0], -1, 1)


# ---------------- 执行者 2：世界模型 MPC ----------------

class ModelPlanner:
    name = "model"

    def __init__(self, ckpt_path, device):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = dict(ck["config"])
        if "gua_emb.weight" not in ck["model_state_dict"]:
            cfg["use_gua"] = False     # 以权重为准（接线自检教训）
        cfg.setdefault("tau", DT_V3)
        cfg.setdefault("gravity", -900.0)
        cfg.setdefault("impulse_max", IMPULSE_MAX_V3)
        self.model = WorldModelV3(**cfg).to(device)
        self.model.load_state_dict(ck["model_state_dict"], strict=False)
        self.model.eval()
        self.mean_obj = ck["norm_stats"]["mean_obj"]
        self.std_obj = ck["norm_stats"]["std_obj"]
        self.mean_obs = ck["norm_stats"]["mean_obs"]
        self.std_obs = ck["norm_stats"]["std_obs"]
        self.device = device
        self.L = 8
        self.history = None
        self.obs_norm = None
        self.rng = np.random.default_rng(0)

    def reset(self, obj_raw, obs_raw, contact):
        self.obs_norm = (obs_raw - self.mean_obs) / (self.std_obs + 1e-8)
        frame = self._frame_norm(obj_raw, contact)
        self.history = [frame] * self.L

    def _frame_norm(self, obj_raw, contact):
        return np.concatenate([
            (obj_raw - self.mean_obj) / (self.std_obj + 1e-8),
            self.obs_norm,
            np.asarray(contact, dtype=np.float32)])

    def observe(self, obj_raw, contact):
        """每执行一步后推入真实观测帧（推理时可获得的传感器）。"""
        self.history.append(self._frame_norm(obj_raw, contact))
        self.history.pop(0)

    def plan(self, obj_raw, target, history=None):
        x = np.stack(self.history)                                   # (L,116)
        xt = torch.from_numpy(np.repeat(x[None], N_CAND, axis=0)).to(self.device)
        obb = torch.from_numpy(np.repeat(self.obs_norm[None], N_CAND, axis=0)).to(self.device)
        mean = np.zeros((K, 2))
        std = np.ones((K, 2))
        with torch.no_grad():
            for _ in range(CEM_ITERS):
                cand = np.clip(mean + std * self.rng.standard_normal((N_CAND, K, 2)), -1, 1)
                act = np.zeros((N_CAND, K, 6), dtype=np.float32)
                act[:, :, 0:2] = cand
                at = torch.from_numpy(act).to(self.device)
                abs_preds, _ = self.model.rollout(xt, at, obb)
                dist_min = np.full(N_CAND, np.inf)
                for k in range(K):
                    raw = abs_preds[k].float().cpu().numpy() * (self.std_obj + 1e-8) + self.mean_obj
                    dist_min = np.minimum(dist_min, np.hypot(raw[:, LF["x"]] - target[0],
                                                             raw[:, LF["y"]] - target[1]))
                d = dist_min + 0.01 * (cand ** 2).sum(axis=(1, 2))
                elite = np.argsort(d)[:N_ELITE]
                mean = cand[elite].mean(axis=0)
                std = cand[elite].std(axis=0) + 0.05
        return np.clip(mean[0], -1, 1)


# ---------------- 执行者 3：无模型 P-D 反馈 ----------------

class ReactivePlanner:
    name = "reactive"

    def __init__(self, kp=3.0, kd=0.6):
        self.kp, self.kd = kp, kd

    def reset(self, **kw):
        pass

    def plan(self, obj_raw, target, history=None):
        pos = obj_raw[LF["x"]:LF["x"] + 2]
        vel = obj_raw[LF["vx"]:LF["vx"] + 2]
        F = self.kp * (np.asarray(target) - pos) - self.kd * vel
        n = np.linalg.norm(F)
        if n > 500:
            F = F / n * 500
        return (F / 500.0).astype(np.float32)   # 归一化冲量：F·τ/25 = F/500


# ---------------- 主协议 ----------------

def assemble_norm(obj_raw, obs_raw, contact, stats):
    return np.concatenate([
        (obj_raw - stats["mean_obj"]) / (stats["std_obj"] + 1e-8),
        (obs_raw - stats["mean_obs"]) / (stats["std_obs"] + 1e-8),
        np.asarray(contact, dtype=np.float32)])


def _segment_blocks(a, b, seg_a, seg_b, margin=8.0):
    """判断线段 ab 与隔板 seg_a-seg_b 是否相交（隔板两端各外扩 margin）。"""
    d1 = (b[0] - a[0]) * (seg_a[1] - a[1]) - (b[1] - a[1]) * (seg_a[0] - a[0])
    d2 = (b[0] - a[0]) * (seg_b[1] - a[1]) - (b[1] - a[1]) * (seg_b[0] - a[0])
    d3 = (seg_b[0] - seg_a[0]) * (a[1] - seg_a[1]) - (seg_b[1] - seg_a[1]) * (a[0] - seg_a[0])
    d4 = (seg_b[0] - seg_a[0]) * (b[1] - seg_a[1]) - (seg_b[1] - seg_a[1]) * (b[0] - seg_a[0])
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def select_scene(seed):
    """确定性场景筛选（M-D v2 接触丰富域）：
    ①可控域：物体 0 腾空 + 速度 <150px/s（T7）；
    ②接触关键路径：球-靶连线被垂直隔板遮挡 → 直线弹道不可达，
      唯一路径是利用边界墙反弹绕行（反弹点选择 = 接触预测问题）。"""
    for attempt in range(MAX_SCENE_RETRY):
        rng_seed = seed * 100 + attempt
        random.seed(rng_seed)
        scene = generate_scene()
        probe = PushTargetEnv()
        rng = np.random.default_rng(seed + 500000 + attempt)
        probe.reset(scene_config=copy.deepcopy(scene), rng=rng)
        ct0 = np.zeros(3, dtype=np.float32)
        obj = None
        for _ in range(WARMUP):
            (obj, _, ct), _, _, _ = probe.step(np.zeros((3, 2)))
            ct0 = ct
        speed0 = float(np.hypot(obj[LF["vx"]], obj[LF["vy"]]))
        if ct0[0] >= 0.5 or speed0 >= MAX_OBJ0_SPEED:
            continue

        # 构造隔板：x = bx 处的垂直墙（不与原有障碍重叠即可——重叠也无碍，静态叠加）
        bx = float(rng.uniform(280, 420))
        gap_frac = rng.uniform(0.28, 0.45)     # 隔板占墙高的比例（留通道需绕行）
        seg_a = (bx, 400.0)                     # 从地板到 gap_frac*400 高度
        seg_b = (bx, 400.0 * (1.0 - gap_frac))
        # 靶区在隔板另一侧、与球异侧
        side = 1.0 if obj[LF["x"]] < bx else -1.0
        tx = float(np.clip(bx + side * rng.uniform(90, 180), 60, 540))
        ty = float(rng.uniform(80, 320))
        if not _segment_blocks((obj[LF["x"]], obj[LF["y"]]), (tx, ty),
                               seg_a, seg_b, margin=15.0):
            continue   # 隔板必须真的挡住直线弹道
        scene["obstacles"].append({"type": "segment", "a": seg_a, "b": seg_b, "radius": 4.0})
        return scene, (tx, ty), attempt
    return None, None, MAX_SCENE_RETRY


def run_episode(planner, scene, target, stats, model_planner, seed,
                align_target=True):
    env = PushTargetEnv(max_steps=CTRL_STEPS)
    obj, obs_raw, ct = env.reset(scene_config=copy.deepcopy(scene), rng=np.random.default_rng(seed))
    if align_target:
        # 2026-09-21 修复：原实现忽略 select_scene 返回的靶点（env.reset 内部另行
        # 随机生成），导致隔板遮挡校验与真实靶点不一致——M-D v2 / T8 二审数字受影响，
        # 复核用 --legacy_target 可复现旧行为。
        env.target = (float(target[0]), float(target[1]))
    if isinstance(planner, ModelPlanner):
        planner.reset(obj, obs_raw, ct)
    dists = []
    min_dist = float("inf")
    success_step = None
    # 预热：统一零动作填充历史
    for _ in range(WARMUP):
        (obj, obs_raw, ct), d, _, _ = env.step(np.zeros((3, 2)))
        dists.append(d)
        if isinstance(planner, ModelPlanner):
            planner.observe(obj, ct)
    for t in range(CTRL_STEPS):
        a0 = planner.plan(obj, env.target)
        a = np.zeros((3, 2), dtype=np.float32)
        a[0] = a0
        (obj, obs_raw, ct), d, _, _ = env.step(a)
        dists.append(d)
        min_dist = min(min_dist, d)
        if isinstance(planner, ModelPlanner):
            planner.observe(obj, ct)
        if success_step is None and min_dist <= MIN_DIST_SUCCESS:
            success_step = t + 1
    return {"success": success_step is not None,
            "success_step": success_step,
            "min_dist": min_dist,
            "final_dist": dists[-1]}


def main():
    parser = argparse.ArgumentParser(description="M-D 决策实验：三个执行者配对同台")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--model_ckpt", type=str, default="checkpoints-v3-nogua/best.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_report", type=str, default="analysis_output/md_planning_report.json")
    parser.add_argument("--legacy_target", action="store_true",
                        help="复现旧行为（不把 select_scene 靶点对齐到 env，供 T8 复核对照）")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 72)
    print(f"M-D 决策实验：推球入靶 | {args.episodes} 个配对场景 | 判据 A/B 预注册")
    print("=" * 72)

    model_planner = ModelPlanner(args.model_ckpt, device)
    planners = [PriorPlanner(np.random.default_rng(7)),
                model_planner,
                ReactivePlanner()]

    results = {p.name: {"success": 0, "steps": [], "min_dists": []} for p in planners}
    skipped = 0
    for ep in range(args.episodes):
        seed = 1000 + ep
        scene, target, attempt = select_scene(seed)
        if scene is None:
            skipped += 1
            continue
        for p in planners:
            r = run_episode(p, scene, target, None, model_planner, seed + 900000,
                            align_target=not args.legacy_target)
            results[p.name]["success"] += int(r["success"])
            if r["success"]:
                results[p.name]["steps"].append(r["success_step"])
            results[p.name]["min_dists"].append(r["min_dist"])
        if (ep + 1) % 10 == 0:
            line = " | ".join(f"{p.name} {results[p.name]['success']}/{ep + 1}" for p in planners)
            print(f"  [{ep + 1}/{args.episodes}] {line}")

    n_eff = args.episodes - skipped
    print("\n" + "=" * 72)
    print(f"{'执行者':<12}{'导引成功率':>12}{'中位成功步':>12}{'平均最近距(px)':>16}"
          f"{'跳过场景':>10}")
    summary = {}
    for p in planners:
        s = results[p.name]["success"]
        med = float(np.median(results[p.name]["steps"])) if results[p.name]["steps"] else float("nan")
        md = float(np.mean(results[p.name]["min_dists"]))
        summary[p.name] = {"success_rate": s / max(n_eff, 1), "median_success_step": med,
                           "mean_min_dist": md}
        print(f"{p.name:<14}{s / max(n_eff, 1):>12.1%}{med:>12.0f}{md:>16.2f}{skipped:>10}")

    a = (summary["model"]["success_rate"] - summary["prior"]["success_rate"]) * 100
    b = summary["model"]["success_rate"] - summary["reactive"]["success_rate"]
    print(f"\n判据 A（model − prior ≥ +10pp）: {a:+.1f}pp → {'[OK] 通过' if a >= 10 else '[FAIL] 未通过'}")
    print(f"判据 B（model > reactive 导引成功率）: {b * 100:+.1f}pp → {'[OK] 通过' if b > 0 else '[FAIL] 未通过'}")

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump({"episodes": args.episodes, "summary": summary,
                   "criterion_A_pp": float(a), "criterion_B_pp": float(b),
                   "pre_registered": "A: model-prior>=10pp; B: model>reactive"}, f,
                  indent=2, ensure_ascii=False)
    print(f"[OK] 报告已保存: {args.output_report}")


if __name__ == "__main__":
    main()
