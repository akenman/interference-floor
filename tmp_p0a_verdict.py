# -*- coding: utf-8 -*-
"""P0-a 正式判据核算（n=3 全档位，同 seed 配对 + ITT）。只读 checkpoint 的 val_metrics。"""
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

RUNGS = [("euler005", "ours_base", "euler@0.05×1"),
         ("euler001x5", "ours_base@euler001x5", "euler@0.01×5"),
         ("rk4005", "ours_base@rk4005", "rk4@0.05"),
         ("rk4001x5", "ours_base@rk4001x5", "rk4@0.01×5")]
PRIOR = {"euler005": (3.373, 15.804), "euler001x5": (0.522, 3.24),
         "rk4005": (0.147, 1.68), "rk4001x5": (0.038, 0.49)}

R = {}
for spec, pat, _ in RUNGS:
    R[spec] = {}
    for s in (42, 43, 44):
        ck = torch.load(f"checkpoints-E0/{pat}_s{s}/best.pt", map_location="cpu",
                        weights_only=False)
        vm = ck["val_metrics"]
        R[spec][s] = (vm["one_step_model"], vm["rollout"]["k=5"])

print("=== 四档 × 3 seed（训练期 best val，与报告同协议）===\n")
print(f"{'先验':<15}{'seed':>5}{'one-step':>11}{'其先验':>9}{'k=5':>9}{'其先验':>9}")
for spec, _, lab in RUNGS:
    for s in (42, 43, 44):
        o, k = R[spec][s]
        print(f"{lab:<15}{s:>5}{o:>11.4f}{PRIOR[spec][0]:>9.3f}{k:>9.2f}{PRIOR[spec][1]:>9.2f}")
    o = np.mean([R[spec][s][0] for s in (42, 43, 44)])
    k = np.mean([R[spec][s][1] for s in (42, 43, 44)])
    print(f"{'  → 3-seed 均值':<15}{'':>5}{o:>11.4f}{PRIOR[spec][0]:>9.3f}{k:>9.2f}{PRIOR[spec][1]:>9.2f}"
          f"   (k=5/先验 = {k/PRIOR[spec][1]:.2f}×)\n")

print("=== 判据 ===\n")
m1 = [np.mean([R[s][k][0] for k in (42, 43, 44)]) for s, _, _ in RUNGS]
m5 = [np.mean([R[s][k][1] for k in (42, 43, 44)]) for s, _, _ in RUNGS]
labs = [lab for _, _, lab in RUNGS]
print("P0a-1 单调性（先验精度↑ ⇒ 误差↓）")
print("   单步: " + " > ".join(f"{v:.4f}" for v in m1))
print("   k=5 : " + " > ".join(f"{v:.2f}" for v in m5))
mono1 = all(m1[i] > m1[i + 1] for i in range(3))
mono5 = all(m5[i] > m5[i + 1] for i in range(3))
print(f"   → 单步单调 {mono1}；闭环单调 {mono5} → {'PASS' if mono1 and mono5 else 'FAIL'}\n")

# P0a-2 / P0a-4：rk4001x5
o = np.array([R["rk4001x5"][s][0] for s in (42, 43, 44)])
k = np.array([R["rk4001x5"][s][1] for s in (42, 43, 44)])
print(f"P0a-2 单步 ≤0.10px 且 k=5 ≤1.0px：one={o.mean():.4f}±{o.std():.4f}, "
      f"k5={k.mean():.3f}±{k.std():.3f} → {'PASS' if o.mean() <= .1 and k.mean() <= 1 else 'FAIL'}")
print(f"P0a-4 单步 ≤2× 先验({PRIOR['rk4001x5'][0]:.3f}={2*PRIOR['rk4001x5'][0]:.3f})："
      f"{o.mean():.4f} = {o.mean()/PRIOR['rk4001x5'][0]:.2f}× → "
      f"{'PASS' if o.mean() <= 2*PRIOR['rk4001x5'][0] else 'FAIL'}")

e0 = json.load(open("analysis_output/E0_report.json", encoding="utf-8"))["arms"]
sm = np.array([x["rollout"]["k=5"] for x in e0["strong_mlp"]])

# P0a-3：与无先验基线（strong_mlp）
t, p = stats.ttest_rel(k, sm)
print(f"P0a-3 k=5 ≤ strong_mlp({sm.mean():.3f})：{k.mean():.3f}（{sm.mean()/k.mean():.1f}× 更优），"
      f"配对 t={t:+.2f} p={p:.3f}，方向 {int((k < sm).sum())}/3 → "
      f"{'PASS' if k.mean() < sm.mean() else 'FAIL'}")

print("\n=== 关键对比（同架构同 seed 只换先验）===")
for ref, lab in (("euler005", "euler@0.05×1 → rk4@0.01×5"),):
    for i, nm in ((0, "one-step"), (1, "k=5")):
        x = np.array([R[ref][s][i] for s in (42, 43, 44)])
        y = np.array([R["rk4001x5"][s][i] for s in (42, 43, 44)])
        t, p = stats.ttest_rel(x, y)
        print(f"   {nm:<9} {x.mean():8.4f} → {y.mean():7.4f}  "
              f"（{x.mean()/y.mean():5.1f}× 改善，配对 t={t:+.2f} p={p:.4f}，3/3 同向）")

print("\n=== 残差头相对自身先验的贡献（k=5）===")
for spec, _, lab in RUNGS:
    km = np.mean([R[spec][s][1] for s in (42, 43, 44)])
    pr = PRIOR[spec][1]
    print(f"   {lab:<15} 训练 {km:7.2f} vs 先验 {pr:6.2f}  →  "
          f"{'改善' if km < pr else '恶化'} {abs(km/pr-1)*100:5.1f}%")

print("\n=== 闭环 k=5 的种子离散（先验越准越稳？）===")
for spec, _, lab in RUNGS:
    v = np.array([R[spec][s][1] for s in (42, 43, 44)])
    print(f"   {lab:<15} sd={v.std():6.2f}  (值 {'/'.join('%.2f' % x for x in v)})")
