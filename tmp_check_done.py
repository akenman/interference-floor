# -*- coding: utf-8 -*-
"""验证报告是否把未完成（无 [OK]）的 run 计入统计。"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
d = json.load(open("analysis_output/E1a_corrected_report.json", encoding="utf-8"))
S = d["summary"]
runs = d["runs"]
inc = [r for r in runs if not r["done"]]
print(f"runs 总数 {len(runs)}，未完成 {len(inc)}")
for r in inc:
    print(f"  未完成: {r['variant']}_s{r['seed']} best={r['best']:.3f} "
          f"n_vals={r['n_vals']} collapse={r['collapse']}")
print()
for v in ["V1", "V2a", "V2b", "V3", "Vfree"]:
    if v in S:
        s = S[v]
        print(f"  {v:<6} n={s['n']} n_done={s['n_done']} "
              f"stable_n={s['n_stable']} mean={s['mean']:.3f} "
              f"mean_stable={s['mean_stable']:.3f}")
print()
print("结论: 若 n != n_done，则统计被未完成 run 污染")
bad = [v for v in S if S[v]["n"] != S[v]["n_done"]]
print("被污染的臂:", bad if bad else "无")
