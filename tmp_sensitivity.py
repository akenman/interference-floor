# -*- coding: utf-8 -*-
"""崩溃剔除口径敏感性：同一判据在「剔崩溃」与「全量」下的结论是否一致。"""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

d = __import__("json").load(open("analysis_output/E1a_corrected_report.json",
                                 encoding="utf-8"))
S = d["summary"]

print("各臂稳定均值 vs 全量均值（相对 V3）\n")
print(f"{'臂':<8}{'n':>4}{'崩溃':>5}{'稳定均值':>10}{'全量均值':>10}"
      f"{'Δ稳定':>9}{'Δ全量':>9}{'结论一致?':>11}")
print("-" * 70)
V3s, V3a = S["V3"]["mean_stable"], S["V3"]["mean"]
for v in ["V1", "V2a", "V2b", "V2c", "V4", "Vfree"]:
    if v not in S:
        continue
    ms, ma = S[v]["mean_stable"], S[v]["mean"]
    if not (np.isfinite(ms) and np.isfinite(V3s)):
        continue
    ds, da = (ms - V3s) / V3s, (ma - V3a) / V3a
    # 判据方向：Δ ≥ 门槛(5%) 才 PASS；两者同侧才叫结论一致
    same = (ds >= 0.05) == (da >= 0.05) if np.isfinite(da) else "?"
    print(f"{v:<8}{S[v]['n']:>4}{S[v]['n_collapse']:>5}{ms:>10.3f}{ma:>10.3f}"
          f"{ds:>+9.1%}{da:>+9.1%}{str(same):>11}")

print("\n崩溃 run 的 best 与稳定 run 的 best（是否系统性更差）")
for v in ["V1", "V2a", "V2b"]:
    if v not in S:
        continue
    det = S[v]["detail"]
    col = [r["best"] for r in det.values() if r["collapse"] and r["done"]]
    st = [r["best"] for r in det.values() if not r["collapse"] and r["done"]]
    if not col:
        print(f"  {v:<5} 无崩溃 run（n_stable={len(st)}）")
        continue
    print(f"  {v:<5} 崩溃 best={sorted(col)}  → 均值 {np.mean(col):.3f}"
          f"  | 稳定 n={len(st)} 均值 {np.mean(st):.3f}"
          f"  | {'崩溃 run 系统性更差→剔除有偏' if np.mean(col) > np.mean(st) else '无系统差异'}")
