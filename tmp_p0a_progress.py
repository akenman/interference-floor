# -*- coding: utf-8 -*-
"""P0-a 训练进度与即时结果（读 checkpoint 的 val_metrics，不重跑评估）。"""
import glob
import json
import os
import re
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8")

# 训练期 best val（与 E0 的 one_step_model 同口径、同 per-seed 划分）
rows = []
for dd in sorted(glob.glob("checkpoints-E0/ours_base@*")):
    pt = os.path.join(dd, "best.pt")
    if not os.path.exists(pt):
        continue
    name = os.path.basename(dd)
    arm, spec = name.split("@", 1)
    spec = spec.rsplit("_s", 1)[0]
    seed = int(name.rsplit("_s", 1)[1])
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    vm = ck.get("val_metrics", {})
    rows.append(dict(name=name, spec=spec, seed=seed, epoch=ck.get("epoch") + 1,
                     one=vm.get("one_step_model"), prior=vm.get("one_step_prior"),
                     roll5=(vm.get("rollout") or {}).get("k=5"),
                     args_spec=(ck.get("args") or {}).get("prior_spec")))

# euler005 档：复用已有 E0 checkpoint
old = json.load(open("analysis_output/E0_report.json", encoding="utf-8"))["arms"]
base = {}
for r in old.get("ours_base", []):
    base[r["seed"]] = r

print("=== P0-a 训练进度（best val = 训练期最优单步）===\n")
print(f"{'run':<34}{'spec(args)':<12}{'epoch':>6}{'best one-step':>15}{'其先验':>9}{'闭环k=5':>10}")
print("-" * 90)
for r in sorted(rows, key=lambda x: (x["spec"], x["seed"])):
    print(f"{r['name']:<34}{str(r['args_spec']):<12}{r['epoch']:>6}"
          f"{r['one']:>15.3f}{(r['prior'] if r['prior'] == r['prior'] else float('nan')):>9.3f}"
          f"{(r['roll5'] if r['roll5'] is not None else float('nan')):>10.2f}")

print("\n=== 对照：已完成的 3 个 run 全部到 60 epoch，最后一档 euler001x5 在跑 ===\n")
print("=== 阶梯对比（同架构 ours_base，只换先验积分器；同 seed 配对）===\n")
print(f"{'seed':>5}  {'euler005(旧)':>13}{'euler001x5':>12}{'rk4005':>10}{'rk4001x5':>11}"
      f"   {'rk4001x5 的先验':>16}")
specs = ["euler001x5", "rk4005", "rk4001x5"]
for seed in sorted(base):
    line = f"{seed:>5}  {base[seed]['one_step_model']:>13.3f}"
    for s in specs:
        m = [r for r in rows if r["spec"] == s and r["seed"] == seed]
        line += f"{m[0]['one']:>12.3f}" if m else f"{'—':>12}"
    pr = [r for r in rows if r["spec"] == "rk4001x5" and r["seed"] == seed]
    if pr and pr[0]["prior"] == pr[0]["prior"]:
        line += f"   {pr[0]['prior']:>16.3f}"
    print(line)

print("\n=== 闭环 k=5（同 seed 配对）===\n")
print(f"{'seed':>5}  {'euler005(旧)':>13}{'euler001x5':>12}{'rk4005':>10}{'rk4001x5':>11}"
      f"   {'解析先验下界':>13}")
for seed in sorted(base):
    line = f"{seed:>5}  {base[seed]['rollout']['k=5']:>13.2f}"
    for s in specs:
        m = [r for r in rows if r["spec"] == s and r["seed"] == seed]
        line += f"{m[0]['roll5']:>12.2f}" if (m and m[0]["roll5"] is not None) else f"{'—':>12}"
    line += f"   {'0.49':>13}"
    print(line)

n_done = sum(1 for r in rows if r["epoch"] >= 60)
print(f"\n进度：{n_done}/9 run 完成；当前在跑 euler001x5 的 s42/s43")
