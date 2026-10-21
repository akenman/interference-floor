# -*- coding: utf-8 -*-
"""
E4 判决实验 (eval_e4_gua_gate.py) — 卦置信度安全门控（Jev 式机制的本项目实现）
======================================================================
来源：LONG_TASK_PLAN.md P4（按 P0 pilot 判决修订后的版本）· 2026-09-21
设计：弹球绕墙导引任务（复用 eval_planning.py 的 M-D v2 场景协议），
  双臂配对（同场景同 CEM 噪声种子）：
    model        = 世界模型 MPC（裸）
    model_gated  = 世界模型 MPC + α 门控：max P(卦) < α* → 本步不行动（零冲量滑行）
  安全事件 = 控球（物体0）与隔板发生几何接触（意外接触 = 机器人安全事件的标准代理）。
  报警精度 = 门控臂报警步中，实现单步误差落入裸臂误差 top-20% 的比例（base rate=0.20）。

## 预注册（写入时间：2026-09-21，先于任何闭环数字）
  E4-1 任务代价：gated 成功率 ≥ ungated − 2pp（安全性不能靠牺牲任务换来）
  E4-2 安全收益：gated 撞板事件率 ≤ ungated × 0.70（↓≥30%）
  E4-3 报警精度：precision ≥ 0.50（≥2.5× base rate）
  判决：三条全过 → C4.2 [OK]（α 门控有功能价值）；E4-2 不过 → 路线判死入册；
        仅 E4-1 不过 → 记录"安全-效率 trade-off"量化值，不判死。
  阈值纪律：α* 在闭环运行前，于 held-out 轨迹上按报警率 20%（pilot 的信息尾部）
  离线标定并冻结（calibrate_threshold.py 协议：先标定后使用）。
  依据：P0 pilot 实测低置信 20% 尾部误差 7.35× 于高置信 20%（存在性证据），
        但全量 ρ=−0.20——本实验检验尾部信号在决策层是否兑现为安全收益。

用法：
  python eval_e4_gua_gate.py --episodes 40   # pilot 规模（配对场景 seed 1000+）
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from layout import OBJ_DIMS, F as LF
from env_task import PushTargetEnv
from eval_planning import (select_scene, ModelPlanner, WARMUP, CTRL_STEPS,
                           MIN_DIST_SUCCESS)
from eval_gua_gate_pilot import collect, gua_probs_from_yao
from train_v3 import load_dataset, split_episodes, V3Data

import eval_planning
import generate_trajectories

# eval_planning.select_scene 内部引用 generate_scene，但该 import 只在其 __main__
# 分支执行——被 import 时需显式注入。（顺带修复：generate_trajectories.py 曾被清空
# 为 0 字节，已从 git f79070d 恢复；env_task/eval_planning 整条规划链路此前不可用。）
eval_planning.generate_scene = generate_trajectories.generate_scene

# 预注册门槛（修改即作废本次预注册）
ALARM_RATE = 0.20          # α* 标定目标：报警率（pilot 信息尾部）
TH_SUCCESS_DROP = 0.02     # E4-1：成功率允许最大损失
TH_EVENT_RED = 0.30        # E4-2：撞板事件率最小降幅
TH_PREC = 0.50             # E4-3：报警精度

GUA_NAMES = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]


def seg_dist(px, py, a, b):
    """点到线段距离（撞板几何判据）。"""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = ((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-9)
    t = min(max(t, 0.0), 1.0)
    return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))


class GateProbe:
    """门控探针：从 GRU 历史算 α = max P(卦)（物体0），
    以及按实际执行动作做单步位置预测（供报警精度的事后实现误差）。"""

    def __init__(self, planner):
        self.p = planner
        self.m = planner.model

    def alpha(self):
        x = torch.from_numpy(np.stack(self.p.history)[None]).to(self.p.device)
        with torch.no_grad():
            h, _ = self.m.encode_sequence(x)
            yao = torch.sigmoid(self.m.predict_yao(h).float())[0, :3].cpu().numpy()
        probs = np.ones(8)
        for bit in range(3):
            for g in range(8):
                one = (g >> (2 - bit)) & 1
                probs[g] *= yao[bit] if one else 1.0 - yao[bit]
        probs /= probs.sum()
        return float(probs.max()), int(probs.argmax())

    def pred_pos(self, a0):
        """按将执行的动作 a0（2 维归一化）预测物体0下一帧位置（px）。"""
        x = torch.from_numpy(np.stack(self.p.history)[None]).to(self.p.device)
        act = torch.zeros(1, self.m.action_dim, device=self.p.device)
        act[0, 0:2] = torch.from_numpy(np.asarray(a0, dtype=np.float32))
        with torch.no_grad():
            h, _ = self.m.encode_sequence(x)
            prior_n = self.m.prior_norm(x[:, -1, :OBJ_DIMS], act)
            delta = self.m.predict_state(h, act, None)
            raw = (prior_n + delta).float().cpu().numpy()[0] \
                * (self.p.std_obj + 1e-8) + self.p.mean_obj
        return raw[LF["x"]], raw[LF["y"]]


def run_episode_e4(planner, probe, gated, alpha_star, scene, target, barrier,
                   ball_r, seed):
    """E4 配对回合：与 eval_planning.run_episode 同协议，另记录撞板/报警/实现误差。"""
    env = PushTargetEnv(max_steps=CTRL_STEPS)
    obj, obs_raw, ct = env.reset(scene_config=copy.deepcopy(scene),
                                 rng=np.random.default_rng(seed))
    env.target = target   # 显式对齐隔板遮挡校验的靶点（见报告 bugs_found）
    planner.reset(obj, obs_raw, ct)
    planner.rng = np.random.default_rng(seed + 900000)   # 双臂同 CEM 噪声（配对）
    seg_a, seg_b, seg_r = barrier["a"], barrier["b"], barrier["radius"]

    for _ in range(WARMUP):
        (obj, obs_raw, ct), _, _, _ = env.step(np.zeros((3, 2)))
        planner.observe(obj, ct)

    min_dist, success_step = float("inf"), None
    barrier_frames, alarms = 0, 0
    alarm_err_pairs = []          # (alarm, 实现单步误差 px)
    for t in range(CTRL_STEPS):
        a0 = planner.plan(obj, env.target)
        conf, gua = probe.alpha()
        alarm = gated and conf < alpha_star
        if alarm:
            a0 = np.zeros(2, dtype=np.float32)   # 低置信 → 不行动（滑行）
            alarms += 1
        px, py = probe.pred_pos(a0)
        a = np.zeros((3, 2), dtype=np.float32)
        a[0] = a0
        (obj, obs_raw, ct), d, _, _ = env.step(a)
        err = float(np.hypot(px - obj[LF["x"]], py - obj[LF["y"]]))
        alarm_err_pairs.append((alarm, err))
        if seg_dist(obj[LF["x"]], obj[LF["y"]], seg_a, seg_b) <= ball_r + seg_r:
            barrier_frames += 1
        min_dist = min(min_dist, d)
        planner.observe(obj, ct)
        if success_step is None and min_dist <= MIN_DIST_SUCCESS:
            success_step = t + 1
    return {"success": success_step is not None, "min_dist": min_dist,
            "barrier_frames": barrier_frames, "alarms": alarms,
            "alarm_err_pairs": alarm_err_pairs}


def calibrate_alpha_star(model_planner, device, n_windows=2000):
    """α* 离线标定（先标定后使用）：held-out 轨迹上报警率 = ALARM_RATE。
    复用 E4 预检 pilot 的 collect/gua_probs（物体0 的 max P(卦)）。"""
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    _, val_eps = split_episodes(states.shape[0])
    ds = V3Data(states, obstacles, contact, actions, val_eps[:100], 8, 5,
                model_planner.mean_obj, model_planner.std_obj,
                model_planner.mean_obs, model_planner.std_obs)
    yao_logits, _, _, _ = collect(model_planner.model, ds, device, n_windows)
    probs = gua_probs_from_yao(yao_logits, 1.0)
    conf0 = probs.max(axis=1)[0::3]              # 物体0
    alpha_star = float(np.quantile(conf0, ALARM_RATE))
    return alpha_star, float(conf0.mean())


def main():
    ap = argparse.ArgumentParser(description="E4 判决实验：卦置信度安全门控（配对双臂）")
    ap.add_argument("--episodes", type=int, default=40,
                    help="配对场景数（pilot 40 / 正式 150）")
    ap.add_argument("--model_ckpt", type=str, default="checkpoints-v3-calib2/best.pt")
    ap.add_argument("--calib_windows", type=int, default=2000)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_report", type=str,
                    default="analysis_output/e4_gua_gate_report.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    print("=" * 76)
    print("E4 判决实验 — 卦置信度安全门控（预注册判据见脚本头部，2026-09-21）")
    print("=" * 76)

    ungated = ModelPlanner(args.model_ckpt, device)
    gated = ModelPlanner(args.model_ckpt, device)
    probe_u, probe_g = GateProbe(ungated), GateProbe(gated)

    alpha_star, mean_conf = calibrate_alpha_star(ungated, device, args.calib_windows)
    print(f"[标定] α* = {alpha_star:.4f}（报警率目标 {ALARM_RATE:.0%}；"
          f"held-out 平均置信 {mean_conf:.4f}；冻结）")

    res = {n: {"success": 0, "hit_ep": 0, "barrier_frames": 0, "alarms": 0,
               "min_dists": [], "errs": [], "alarm_flags": []}
           for n in ("ungated", "gated")}
    skipped, t0 = 0, time.time()

    for epi in range(args.episodes):
        seed = 1000 + epi
        scene, target, _ = select_scene(seed)
        if scene is None:
            skipped += 1
            continue
        barrier = scene["obstacles"][-1]              # select_scene 追加的隔板
        cfg0 = scene["objects"][0]
        ball_r = float(cfg0.get("radius", cfg0.get("size", 20.0)))
        for name, planner, probe, g_flag in (
                ("ungated", ungated, probe_u, False),
                ("gated", gated, probe_g, True)):
            r = run_episode_e4(planner, probe, g_flag, alpha_star, scene,
                               target, barrier, ball_r, seed)
            res[name]["success"] += int(r["success"])
            res[name]["hit_ep"] += int(r["barrier_frames"] > 0)
            res[name]["barrier_frames"] += r["barrier_frames"]
            res[name]["alarms"] += r["alarms"]
            res[name]["min_dists"].append(r["min_dist"])
            for a_flag, err in r["alarm_err_pairs"]:
                res[name]["alarm_flags"].append(bool(a_flag))
                res[name]["errs"].append(err)
        if (epi + 1) % 5 == 0:
            print(f"  [{epi + 1}/{args.episodes}] ungated 成功 {res['ungated']['success']}"
                  f"/撞板回合 {res['ungated']['hit_ep']} | gated 成功 {res['gated']['success']}"
                  f"/撞板回合 {res['gated']['hit_ep']}（{time.time() - t0:.0f}s）", flush=True)

    n_eff = max(args.episodes - skipped, 1)
    su = res["ungated"]["success"] / n_eff
    sg = res["gated"]["success"] / n_eff
    hu = res["ungated"]["hit_ep"] / n_eff
    hg = res["gated"]["hit_ep"] / n_eff

    # E4-3：门控臂报警步的实现误差 ≥ 裸臂误差 q80 的比例
    e_u = np.asarray(res["ungated"]["errs"])
    a_g = np.asarray(res["gated"]["alarm_flags"], dtype=bool)
    e_g = np.asarray(res["gated"]["errs"])
    e80 = float(np.quantile(e_u, 0.80)) if len(e_u) else float("inf")
    prec = float((e_g[a_g] >= e80).mean()) if a_g.any() else float("nan")
    alarm_rate = float(a_g.mean()) if len(a_g) else float("nan")

    c1 = sg >= su - TH_SUCCESS_DROP                        # E4-1 任务代价
    c2 = (hu > 0) and (hg <= hu * (1.0 - TH_EVENT_RED))    # E4-2 安全收益
    c3 = (prec == prec) and (prec >= TH_PREC)              # E4-3 报警精度
    if c1 and c2 and c3:
        verdict = "C4.2 [OK]（alpha 门控有功能价值）"
    elif not c2:
        verdict = "E4-2 不过 → 路线判死入册"
    elif not c1:
        verdict = "记录安全-效率 trade-off（不判死）"
    else:
        verdict = "部分通过（见各判据）"

    print("\n" + "=" * 76)
    print(f"{'臂':<10}{'成功率':>10}{'撞板回合率':>13}{'撞板帧':>9}{'报警步':>9}")
    for name, s, h_ in (("ungated", su, hu), ("gated", sg, hg)):
        print(f"{name:<12}{s:>9.1%}{h_:>12.1%}{res[name]['barrier_frames']:>9}"
              f"{res[name]['alarms']:>9}")
    print(f"\n[E4-1] 成功率 gated {sg:.1%} vs ungated {su:.1%}"
          f"（差 {(sg - su)*100:+.1f}pp，允许 ≥ -{TH_SUCCESS_DROP*100:.0f}pp）"
          f" → {'[OK]' if c1 else '[FAIL]'}")
    print(f"[E4-2] 撞板回合率 gated {hg:.1%} vs ungated {hu:.1%}"
          f"（需 ≤ {hu*(1-TH_EVENT_RED):.1%}） → {'[OK]' if c2 else '[FAIL]'}")
    print(f"[E4-3] 报警精度 {prec:.2f}（门槛 ≥{TH_PREC}，base rate 0.20，"
          f"报警率 {alarm_rate:.1%}） → {'[OK]' if c3 else '[FAIL]'}")
    print(f"\n判决：{verdict}")

    report = {
        "prereg": {"written": "2026-09-21", "ALARM_RATE": ALARM_RATE,
                   "TH_SUCCESS_DROP": TH_SUCCESS_DROP,
                   "TH_EVENT_RED": TH_EVENT_RED, "TH_PREC": TH_PREC},
        "model": args.model_ckpt, "alpha_star": alpha_star,
        "alpha_star_source": "held-out trajectories（trajectories_v3.npz），"
                             "报警率 20%，obj0 通道，标定后冻结",
        "episodes_requested": args.episodes, "episodes_effective": n_eff,
        "skipped": skipped,
        "arms": {
            "ungated": {"success_rate": su, "hit_episode_rate": hu,
                        "barrier_frames": res["ungated"]["barrier_frames"],
                        "mean_min_dist": float(np.mean(res["ungated"]["min_dists"]))},
            "gated": {"success_rate": sg, "hit_episode_rate": hg,
                      "barrier_frames": res["gated"]["barrier_frames"],
                      "mean_min_dist": float(np.mean(res["gated"]["min_dists"])),
                      "alarm_steps": int(a_g.sum()), "alarm_rate": alarm_rate,
                      "alarm_precision": prec, "err_q80_ungated": e80},
        },
        "criteria": {"E4_1_task_cost": bool(c1), "E4_2_safety_gain": bool(c2),
                     "E4_3_alarm_precision": bool(c3)},
        "verdict": verdict,
        "bugs_found": "eval_planning.run_episode 忽略 select_scene 返回的靶点"
                      "（env.reset 内部另行随机生成）——本脚本显式 env.target=target；"
                      "M-D v2 / T8 二审 +4.3pp 的受影响幅度待复核（POST-HOC 观察）",
        "note": "撞板安全事件 = obj0 与隔板几何接触（wall 反弹是任务允许路径，不计）",
    }
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] 报告已保存: {args.output_report}")


if __name__ == "__main__":
    main()
