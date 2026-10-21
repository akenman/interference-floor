# -*- coding: utf-8 -*-
"""打印解耦消融（Stage D）判据 + 基准 V3 对比。"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
path = "analysis_output/E1a_stageD_report.json"
d = json.load(open(path, encoding="utf-8"))
S = d["summary"]

print("Stage D 解耦消融（固定 64 专家容量，只变监督轴/门控可见轴）")
print("臂        n  崩溃  稳定n   均值    稳定均值   vs V3(稳定0.829)")
v3 = S.get("V3", {}).get("mean_stable")
for v in ["A1x64", "A2x64", "A3g1", "V3"]:
    if v in S:
        s = S[v]
        rel = ""
        if v3:
            rel = "  %+.1f%%" % ((s["mean_stable"] - v3) / v3 * 100)
        print(f"{v:<8} {s['n']:>2} {s['n_collapse']:>4} {s['n_stable']:>6}"
              f"  {s['mean']:.3f}  {s['mean_stable']:.3f}{rel}")

cd = d.get("criteria_decoupled_exploratory", {})
print("\n解耦判据（探索性，基准 V3）:")
for k, c in cd.items():
    ci = c.get("ci95")
    cis = "n/a" if not ci else f"[{ci[0]:+.3f},{ci[1]:+.3f}]"
    print(f"  {k:<8} {c['variant']:<6} Δ={c['delta']:+.3f} CI95={cis}"
          f" n={c.get('n_paired')} -> {c.get('verdict')}")
    print(f"           text: {c.get('text', c.get('threshold'))}")

print("\n(Δ>0 = 该臂比 V3 更差 / one-step 误差更高)")
print("完整性:", d["status"], "| completed:", d["completed_runs"], "/", d["expected_runs"])
