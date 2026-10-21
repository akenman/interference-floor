# -*- coding: utf-8 -*-
"""决定性检验：门控分离度能否预测同 seed 的 gating 增益？

前面的 run 级相关几乎全为 0，但 run 级相关混入了：
  - 架构差异（n_gua=4/16/64/256、轴数）
  - 种子难度差异
  - no_gua 基线差异

正确做法：**配对**。每个 gated run 都有同变体同 seed 的 no_gua 对照，
于是 gain = best(gated) - best(no_gua)（>0 = 门控变差），
把架构与种子难度全部消掉，再看 sep_state 是否解释 gain。

同时检验机制的两个可分离预测：
  P1  分离度高 → gain 更大（更正）   [机制核心]
  P2  存活组多 / eff_n 高 → gain 更小（专家坍塌有害）
"""
import io
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

rows = json.load(open("analysis_output/gate_real.json", encoding="utf-8"))

# 从 checkpoint 报告补 no_gua 的 best（gate_real.json 已含 gated）
import build_e1a_report as B
import os
ctrl = {}
for tag in os.listdir("checkpoints-e1a-corrected"):
    lp = os.path.join("logs-corrected", tag + ".log")
    if not os.path.exists(lp):
        continue
    r = B.parse_log(lp)
    if not (r.get("done") and r.get("best") is not None):
        continue
    if r.get("no_gua"):
        ctrl[(r["variant"], r["seed"])] = r["best"]

pairs = []
for r in rows:
    if r["collapse"]:
        continue                       # 主分析剔崩溃
    key = (r["variant"], r["seed"])
    if key not in ctrl:
        continue
    pairs.append(dict(r, gain=r["best"] - ctrl[key], ctrl_best=ctrl[key]))
print("配对数（剔崩溃）: %d" % len(pairs))

y = np.array([p["gain"] for p in pairs])
print("gain 统计: mean %+.4f  median %+.4f  (n=%d, >0 表示门控变差)"
      % (y.mean(), np.median(y), len(y)))
print()

from scipy.stats import wilcoxon
w = wilcoxon(y)
print("Wilcoxon 配对检验 gain≠0: stat=%.1f p=%.4f\n" % (w.statistic, w.pvalue))


def rep(key, expect):
    s = np.array([p[key] for p in pairs])
    ok = np.isfinite(s) & np.isfinite(y)
    s2, y2 = s[ok], y[ok]
    if len(s2) < 5 or np.std(s2) == 0:
        print("  %-11s: 方差为 0，跳过" % key)
        return
    pe = float(np.corrcoef(s2, y2)[0, 1])
    sp = float(spearmanr(s2, y2).statistic)
    ok_sign = "一致" if np.sign(pe) == np.sign(expect) else "相反"
    print("  %-11s vs gain: Pearson %+.3f  Spearman %+.3f  [期望 %+d → %s]"
          % (key, pe, sp, expect, ok_sign))


print("机制预测 vs 配对增益：")
rep("sep_state", +1)      # 分离度高 → 门控更差
rep("sep_hidden", +1)
rep("eff_n", -1)          # 有效专家多 → 门控更差
rep("ngrp", -1)
rep("ent", -1)
print()

# 只在同 n_gua 内做，消掉专家数共变
print("按 n_gua 分层（消专家数共变）：")
by = defaultdict(list)
for p in pairs:
    by[p["n_gua"]].append(p)
for ng, ps in sorted(by.items()):
    if len(ps) < 4:
        print("  n_gua=%-4d n=%d 样本不足" % (ng, len(ps)))
        continue
    s = np.array([p["sep_state"] for p in ps])
    g = np.array([p["gain"] for p in ps])
    print("  n_gua=%-4d n=%2d  sep_state vs gain  Pearson %+.3f  Spearman %+.3f"
          % (ng, len(ps), float(np.corrcoef(s, g)[0, 1]),
             float(spearmanr(s, g).statistic)))
print()

# 全量（含崩溃）敏感性
allp = []
for r in rows:
    key = (r["variant"], r["seed"])
    if key in ctrl:
        allp.append(dict(r, gain=r["best"] - ctrl[key]))
y2 = np.array([p["gain"] for p in allp])
s2 = np.array([p["sep_state"] for p in allp])
print("全量敏感性（含崩溃, n=%d）: sep_state vs gain Pearson %+.3f Spearman %+.3f"
      % (len(y2), float(np.corrcoef(s2, y2)[0, 1]),
         float(spearmanr(s2, y2).statistic)))

json.dump(pairs, open("analysis_output/gate_paired.json", "w",
                      encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n[save] analysis_output/gate_paired.json")
