# -*- coding: utf-8 -*-
"""严谨性检查：配对相关是否显著？效应来自臂内还是臂间？"""
import io
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

pairs = json.load(open("analysis_output/gate_paired.json", encoding="utf-8"))


def test(name, s, y):
    pr = pearsonr(s, y)
    sr = spearmanr(s, y)
    print("  %-22s Pearson r=%+.3f p=%.3f | Spearman ρ=%+.3f p=%.3f"
          % (name, pr.statistic, pr.pvalue, sr.statistic, sr.pvalue))
    return pr.pvalue


y = np.array([p["gain"] for p in pairs])
ss = np.array([p["sep_state"] for p in pairs])
en = np.array([p["eff_n"] for p in pairs])
ng = np.array([p["n_gua"] for p in pairs])
arms = np.array([p["variant"] for p in pairs])

print("n = %d\n=== 显著性 ===" % len(pairs))
test("sep_state ~ gain", ss, y)
test("eff_n ~ gain", en, y)
print()
print("  → r=0.248 在 n=41 下 p≈0.11，**未达 0.05**")
print()

# 组内（within-arm）vs 组间（between-arm）分解
print("=== 效应来源分解 ===")
mu_s = defaultdict(list); mu_y = defaultdict(list)
for a, s, v in zip(arms, ss, y):
    mu_s[a].append(s); mu_y[a].append(v)
arm_names = sorted(mu_s)
bs = np.array([np.mean(mu_s[a]) for a in arm_names])
by = np.array([np.mean(mu_y[a]) for a in arm_names])
print("  臂间（%d 个臂均值）: sep_state~gain Pearson r=%+.3f p=%.3f"
      % (len(arm_names), *np.round(pearsonr(bs, by)[:2], 4)))
ws, wy = [], []
for a in arm_names:
    s = np.array(mu_s[a]); v = np.array(mu_y[a])
    if len(s) > 1:
        ws.append(s - s.mean()); wy.append(v - v.mean())
ws = np.concatenate(ws); wy = np.concatenate(wy)
print("  臂内（去均值后 %d 点）: Pearson r=%+.3f p=%.3f"
      % (len(ws), *np.round(pearsonr(ws, wy)[:2], 4)))
print()
print("  臂均值表:")
print("  %-9s %3s %9s %9s" % ("臂", "n", "sep_state", "gain"))
for i, a in enumerate(arm_names):
    print("  %-9s %3d %9.3f %+9.4f"
          % (a, len(mu_s[a]), bs[i], by[i]))
print()

# 部分相关：控制 n_gua 后 sep_state 还剩多少
print("=== 控制 n_gua 后的偏相关 ===")
def partial(x, yv, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = yv - np.polyval(np.polyfit(z, yv, 1), z)
    return pearsonr(rx, ry)
pr = partial(ss, y, ng.astype(float))
print("  sep_state ~ gain | n_gua: r=%+.3f p=%.3f" % (pr.statistic, pr.pvalue))
pr2 = partial(ss, y, np.log(en))
print("  sep_state ~ gain | eff_n: r=%+.3f p=%.3f" % (pr2.statistic, pr2.pvalue))
print()

# 机制的可证伪预测：门控无益是否与分离度无关（分离度只解释有害程度）
print("=== 总结论 ===")
w = wilcoxon(y)
print("  1. 门控确实有害: gain=%.4f, Wilcoxon p=%.4f" % (y.mean(), w.pvalue))
print("  2. 分离度与有害程度相关 r=+0.248，但 p≈%.2f（未显著）" % pearsonr(ss, y).pvalue)
print("  3. 臂内效应 r=%+.3f 弱于臂间，说明主要由架构差异驱动" % pearsonr(ws, wy).statistic)
print("  → 机制**未被证实也未被证伪**：方向一致但效应量小、不显著、且 n_gua=64 层反向")
print("  → 不能据此启动 n=13 的 GPU 因果实验；先做零成本的设计合理性检查")
