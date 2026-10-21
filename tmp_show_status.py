# -*- coding: utf-8 -*-
"""打印各臂汇总与预注册判据当前状态。"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
d = json.load(open("analysis_output/E1a_corrected_report.json", encoding="utf-8"))
S = d["summary"]

print("臂      n  崩溃  稳定n   均值    稳定均值")
for v in ["V1", "V2a", "V2b", "V3", "Vfree", "V2c", "V4"]:
    if v in S:
        s = S[v]
        print(f"{v:<7} {s['n']:>2} {s['n_collapse']:>5} {s['n_stable']:>5}"
              f"  {s['mean']:.3f}  {s['mean_stable']:.3f}")

print("\n预注册判据（两崩溃口径对照）:")
for k, c in d["criteria"].items():
    def f(ci):
        return "   n/a  " if not ci else f"[{ci[0]:+.3f},{ci[1]:+.3f}]"
    print(f"  {k:<6} {c['variant']:<6} Δ={c['delta']:+.3f} n={c['n_paired']}"
          f" thr={c['threshold']:.2f} -> {c['verdict']}")
    print(f"           剔崩溃 {f(c.get('ci95_stable'))} = {c.get('verdict_stable')}")
    print(f"           全量   {f(c.get('ci95_all'))} = {c.get('verdict_all')}"
          f"   [{c.get('method')}]")

print("\n完整性:", d["status"])
print("进行中:", [(r["variant"], r["seed"]) for r in d.get("incomplete_runs", [])])
