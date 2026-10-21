# -*- coding: utf-8 -*-
"""Vfree（自由学习门控，无三爻结构）是否也是负担？——这决定「非语义门控」是否已被检验。

只读 analysis_output/E1a_stageB_report.json。
"""
import json
import sys

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
S = json.load(open("analysis_output/E1a_stageB_report.json", encoding="utf-8"))["summary"]

rows = ["V1", "V1_nogua", "V2a", "V2a_nogua", "V2b", "V2b_nogua", "V2c",
        "V2c_nogua", "V3", "V3_nogua", "V4", "V4_nogua", "Vfree"]
print(f"{'臂':<12}{'n':>3}{'mean':>9}{'stable':>9}{'n_st':>6}{'崩溃率':>8}")
for k in rows:
    s = S.get(k)
    if not s:
        print(f"{k:<12}  (无数据)")
        continue
    print(f"{k:<12}{s['n']:>3}{s['mean']:>9.3f}{s.get('mean_stable', float('nan')):>9.3f}"
          f"{s.get('n_stable', 0):>6}{s.get('collapse_rate', 0):>8.0%}")


def pair(a, b, label):
    sv, sn = S[a]["seeds"], S[b]["seeds"]
    common = sorted(set(sv) & set(sn), key=int)
    x = np.array([sv[s] for s in common])
    y = np.array([sn[s] for s in common])
    t, p = stats.ttest_rel(x, y)
    print(f"\n{label}  (n={len(common)})")
    for s, xx, yy in zip(common, x, y):
        print(f"   seed {s}: {a}={xx:.3f}  {b}={yy:.3f}  diff={xx-yy:+.3f}")
    print(f"   mean diff={x.mean()-y.mean():+.3f}px  配对 t={t:+.2f}  p={p:.3g}")
    return x.mean() - y.mean(), p


print("\n" + "=" * 70)
pair("Vfree", "V3_nogua", "自由学习门控 vs 无门控（同监督 V3 体系）")
pair("V3", "V3_nogua", "爻结构化门控 vs 无门控（基准对照）")
pair("Vfree", "V3", "自由门控 vs 爻门控  ← T10 原「17.7% 优势」所指的对比")
