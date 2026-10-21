# -*- coding: utf-8 -*-
"""P0-b 三格比较：A(w_roll=0,lr=1e-3) / B(w_roll=.05,lr=5e-4) / C(w_roll=.05,lr=1e-3)。"""
import glob
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
        if ck.get("epoch", -1) < 10:            # 未跑完的不计入
            continue
        vm = ck["val_metrics"]
        out[os.path.basename(d)] = dict(one=vm["one_step_model"],
                                        k5=vm["rollout"]["k=5"], epoch=ck["epoch"])
    return out


A, B, C = load("checkpoints-P0bA"), load("checkpoints-P0bB"), load("checkpoints-P0bC")
print(f"可用 run：A={len(A)}  B={len(B)}  C={len(C)}\n")

print("=== 逐 run 三格对照（one-step / 闭环 k=5）===\n")
print(f"{'run':<26}{'A(wr0,lr1e-3)':>18}{'B(wr.05,lr5e-4)':>18}{'C(wr.05,lr1e-3)':>18}")
for key in sorted(set(A) | set(B) | set(C)):
    def fmt(d):
        return f"{d[key]['one']:.4f}/{d[key]['k5']:.2f}" if key in d else "—"
    print(f"{key:<26}{fmt(A):>18}{fmt(B):>18}{fmt(C):>18}")

print("\n=== 判据 ===\n")


def pairs(pat, d1, d2):
    ks = [k for k in d1 if k.startswith(pat) and k in d2]
    if not ks:
        return None
    a = np.array([d1[k]["k5"] for k in ks])
    b = np.array([d2[k]["k5"] for k in ks])
    a1 = np.array([d1[k]["one"] for k in ks])
    b1 = np.array([d2[k]["one"] for k in ks])
    return ks, a, b, a1, b1


def show(tag, pat, d1, d2, n1, n2):
    r = pairs(pat, d1, d2)
    if not r:
        print(f"{tag}: 数据未就绪"); return None
    ks, a, b, a1, b1 = r
    t, p = (stats.ttest_rel(b, a) if len(ks) > 1 else (np.nan, np.nan))
    print(f"{tag}  (n={len(ks)})")
    print(f"   {n1} k=5 {a.mean():8.2f} → {n2} k=5 {b.mean():8.2f}  "
          f"({'改善' if b.mean() < a.mean() else '恶化'} {abs(a.mean()/b.mean()-1)*100 if b.mean() else 0:5.1f}%"
          f"{'' if len(ks) < 2 else f'，配对 t={t:+.2f} p={p:.3f}'})")
    print(f"   {n1} one {a1.mean():8.4f} → {n2} one {b1.mean():8.4f}  "
          f"(Δ={b1.mean()-a1.mean():+.4f})")
    return b.mean()


print("--- P0c-1 核心威胁重测：strong_mlp@C 闭环 k=5 ≤0.49px ---")
v = show("strong_mlp  C vs A（w_roll 净效应）", "strong_mlp", A, C, "A", "C")
if v is not None:
    print(f"   → strong_mlp@C = {v:.3f}px，{'≤0.49 [OK] 威胁成立' if v <= 0.49 else '>0.49 [FAIL] 威胁不成立（P0-a 加固）'}")
v = show("strong_mlp  B vs C（lr 减半效应）", "strong_mlp", C, B, "C", "B")

print("\n--- P0c-2：noprior@C ≤1.0px ---")
v = show("noprior  C vs A", "noprior", A, C, "A", "C")
if v is not None:
    print(f"   → noprior@C = {v:.2f}px，{'≤1.0 [OK]' if v <= 1.0 else '>1.0 [FAIL]'}")

print("\n--- P0c-3：ours_base（粗先验）三格 ---")
show("ours_base  C vs A（w_roll 净效应）", "ours_base_s", A, C, "A", "C")
show("ours_base  C vs B（lr 减半效应）", "ours_base_s", C, B, "C", "B")
show("ours_base  A vs B（合并效应）", "ours_base_s", A, B, "A", "B")

print("\n--- 防守臂：ours_base@rk4001x5 三格（C 格按预注册不跑）---")
show("rk4001x5  A vs B", "ours_base@rk4001x5", A, B, "A", "B")
