# -*- coding: utf-8 -*-
"""P0-b A/B 配对比较（读 checkpoint val_metrics；A=单步配方，B=闭环配方）。"""
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")


def load(root):
    out = {}
    for d in sorted(glob.glob(f"{root}/*")):
        pt = os.path.join(d, "best.pt")
        if not os.path.exists(pt):
            continue
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        vm = ck["val_metrics"]
        out[os.path.basename(d)] = dict(
            one=vm["one_step_model"], k5=vm["rollout"]["k=5"],
            k2=vm["rollout"]["k=2"], epoch=ck.get("epoch"),
            spec=(ck.get("args") or {}).get("prior_spec") or "euler005")
    return out


A, B = load("checkpoints-P0bA"), load("checkpoints-P0bB")
print(f"配方A {len(A)} 个 run；配方B {len(B)} 个 run\n")

print("=== 逐 run：A（单步配方） vs B（闭环配方）===\n")
print(f"{'run':<28}{'A one':>8}{'B one':>8}{'Δone':>8}{'A k=5':>9}{'B k=5':>9}{'Δk5':>9}")
for key in sorted(set(A) & set(B)):
    a, b = A[key], B[key]
    print(f"{key:<28}{a['one']:>8.4f}{b['one']:>8.4f}{b['one']-a['one']:>+8.4f}"
          f"{a['k5']:>9.2f}{b['k5']:>9.2f}{b['k5']-a['k5']:>+9.2f}")

print("\n=== 判据（同 seed 配对）===\n")


def block(name, keys, thresh, better="lower"):
    ks = [k for k in keys if k in A and k in B]
    if not ks:
        print(f"{name}: 数据未就绪"); return
    a = np.array([A[k][thresh[1]] for k in ks])
    b = np.array([B[k][thresh[1]] for k in ks])
    if len(ks) > 1:
        t, p = stats.ttest_rel(b, a)
    else:
        t, p = float("nan"), float("nan")
    print(f"{name}")
    print(f"   n={len(ks)}  A={a.mean():8.3f} → B={b.mean():8.3f}  "
          f"（{a.mean()/max(b.mean(),1e-9):5.1f}× 改善，配对 t={t:+.2f} p={p:.3f}）")
    print(f"   逐 seed: " + " / ".join(f"A{a[i]:.3f}→B{b[i]:.3f}" for i in range(len(ks))))
    return a.mean(), b.mean()


def find(pat):
    return [k for k in A if k.startswith(pat)]


print("--- P0b-1 核心威胁：strong_mlp 闭环 k=5 是否 ≤0.49px（精确先验水平）---")
r = block("  strong_mlp", find("strong_mlp"), (None, "k5"), None)
if r:
    print(f"   → B 的 k=5 均值 = {r[1]:.3f}px，{'≤0.49 [OK] 威胁成立（先验可被配方替代）' if r[1] <= 0.49 else '>0.49 [FAIL] 威胁不成立（先验仍不可替代）'}")
print()
print("--- P0b-2：noprior 闭环 k=5 是否 ≤1.0px（A 为 ~54px）---")
r = block("  noprior", find("noprior"), (None, "k5"), None)
if r:
    print(f"   → B = {r[1]:.2f}px，{'≤1.0 [OK] 纯学习可自救' if r[1] <= 1.0 else '>1.0 [FAIL] 纯学习无法自救'}")
print()
print("--- P0b-3：ours_base（粗先验）闭环 k=5 是否 ≤2.19px（rk4@0.05 档）---")
r = block("  ours_base(euler005)", [k for k in find("ours_base") if "@" not in k],
          (None, "k5"), None)
if r:
    print(f"   → B = {r[1]:.2f}px，{'≤2.19 [OK] 配方补偿了粗先验' if r[1] <= 2.19 else '>2.19 [FAIL] 配方补偿不足'}")
print()
print("--- P0b-4：ours_base@rk4001x5 是否 ≤0.54px（不损害精确先验臂）---")
r = block("  ours_base@rk4001x5", find("ours_base@rk4001x5"), (None, "k5"), None)
if r:
    print(f"   → B = {r[1]:.3f}px，{'≤0.54 [OK] 未被损害' if r[1] <= 0.54 else '>0.54 [FAIL] 被损害'}")
else:
    print("   仍在训练中（108s/epoch，最慢的一档）")
