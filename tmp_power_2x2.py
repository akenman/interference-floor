# -*- coding: utf-8 -*-
"""用 Stage B 的配对实测数据，反推「语义 × 分离度」2×2 设计的可检出效应量。

只读 analysis_output/E1a_stageB_report.json（137 个 run 的 per-seed best），
不做任何训练。输出：配对差、配对 SD、以及 80% 功效下各 n 的最小可检出效应。
"""
import json
import sys

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

D = json.load(open("analysis_output/E1a_stageB_report.json", encoding="utf-8"))
runs = D["runs"]

# --- 配对：同 variant 同 seed 的 gua vs no_gua ---
pairs = {}
for r in runs:
    if not r.get("done") or r.get("best") is None:
        continue
    key = (r["variant"], r["seed"])
    pairs.setdefault(key, {})[bool(r["no_gua"])] = r

rows = []
for (v, s), d in sorted(pairs.items()):
    if True in d and False in d:
        g, ng = d[False], d[True]          # g = 有门控(gua), ng = 无门控(no_gua)
        rows.append(dict(variant=v, seed=s, gua=g["best"], nogua=ng["best"],
                         d=ng["best"] - g["best"],                      # >0 = 门控有害
                         rel=(ng["best"] - g["best"]) / g["best"],
                         collapse=bool(g["collapse"] or ng["collapse"])))

print(f"配对样本：{len(rows)} 对（同 variant 同 seed）\n")
print(f"{'variant':<8} {'seed':>5} {'有门控':>8} {'无门控':>8} {'差(无-有)':>10} {'相对':>8}  崩溃")
for r in rows:
    print(f"{r['variant']:<8} {r['seed']:>5} {r['gua']:>8.3f} {r['nogua']:>8.3f} "
          f"{r['d']:>10.3f} {r['rel']*100:>7.1f}%  {'Y' if r['collapse'] else ''}")


def report(tag, sel):
    d = np.array([r["d"] for r in sel], dtype=float)
    rel = np.array([r["rel"] for r in sel], dtype=float)
    if len(d) < 2:
        print(f"\n[{tag}] 样本不足")
        return None
    n = len(d)
    sd = d.std(ddof=1)
    t, p = stats.ttest_rel([r["nogua"] for r in sel], [r["gua"] for r in sel])
    print(f"\n[{tag}] n={n}  平均差={d.mean():+.3f}px (相对 {rel.mean()*100:+.1f}%)  "
          f"配对SD={sd:.3f}px  配对t={t:+.2f} p={p:.2g}")
    z = 1.96 + 0.8416
    for n2 in (7, 10, 13, 21, 31):
        print(f"    n={n2:>3} 时 80% 功效最小可检出 |δ| = {z*sd/np.sqrt(n2):.3f}px")
    for delta in (0.20, 0.10, 0.05, 0.02, 0.01):
        need = (z * sd / delta) ** 2
        print(f"    要检出 |δ|={delta:.2f}px → 需 n={np.ceil(need):.0f} 对")
    return dict(n=n, sd=sd, mean=d.mean())


print("\n" + "=" * 72)
a = report("全部配对（含崩溃 run 的 best）", rows)
b = report("剔除崩溃配对", [r for r in rows if not r["collapse"]])

# --- 2x2 交互项的样本量代价 ---
if a:
    sd = a["sd"]
    z = 1.96 + 0.8416
    print("\n" + "=" * 72)
    print("2×2 交互项（语义/无语义 × 低/高分离度）的样本量代价：")
    print("  同一批 seed 做四格配对时，交互项 = (A-B)-(C-D)，其 per-seed 方差约为 4·σ²")
    print(f"  取 σ={sd:.3f}px（Stage B 实测配对 SD），要检出交互 δ_int 需要：")
    for dint in (0.20, 0.10, 0.05, 0.02):
        n_cell = (z * 2 * sd / dint) ** 2
        print(f"    δ_int={dint:.2f}px → 每格 n≈{np.ceil(n_cell):.0f}（四格合计 "
              f"{4*np.ceil(n_cell):.0f} 个 run）")
    print("\n  参照：单臂 E0/E1a run ≈ 40-60 min（单卡）。")
    for dint in (0.20, 0.10, 0.05, 0.02):
        n_cell = np.ceil((z * 2 * sd / dint) ** 2)
        print(f"    δ_int={dint:.2f}px → 四格 {4*n_cell:.0f} run ≈ "
              f"{4*n_cell*50/60:.0f} GPU·小时")
