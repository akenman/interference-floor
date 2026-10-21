# -*- coding: utf-8 -*-
"""门控增益的**配对**重分析（Stage B 原始 per-seed 数据）。

文档 10.19.2/10.19.4 的「6/6 变体、两口径、12/12 为负，-15%~-29%」用的是
「各臂均值比」（剔崩溃 / 全量两口径），从未做配对检验。但每个 variant 的
gua/nogua 用的是同一批 seed（同一 val 划分），配对是免费的、且更正确。

本脚本给出：逐 variant 配对检验 + 以 variant 为重复单元的元分析。
只读 analysis_output/E1a_stageB_report.json。
"""
import json
import sys

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
S = json.load(open("analysis_output/E1a_stageB_report.json", encoding="utf-8"))["summary"]

VARIANTS = ["V1", "V2a", "V2b", "V2c", "V3", "V4", "Vfree"]
PAIR = {"V1": "V1_nogua", "V2a": "V2a_nogua", "V2b": "V2b_nogua",
        "V2c": "V2c_nogua", "V3": "V3_nogua", "V4": "V4_nogua"}

print("=== 逐 variant 配对检验（全部 seed，含崩溃 run = ITT 口径）===")
print(f"{'variant':<8}{'n':>4}{'配对差(门控-无门控)':>22}{'SD':>8}{'t':>7}{'p':>9}{'文档口径':>10}")
effs, ses = [], []
for v, nv in PAIR.items():
    sv, sn = S[v]["seeds"], S[nv]["seeds"]
    common = sorted(set(sv) & set(sn), key=int)
    d = np.array([sv[s] - sn[s] for s in common])
    t, p = stats.ttest_rel([sv[s] for s in common], [sn[s] for s in common])
    doc = (S[nv]["mean_stable"] - S[v]["mean_stable"]) / S[v]["mean_stable"] * 100
    print(f"{v:<8}{len(d):>4}{d.mean():>+22.3f}{d.std(ddof=1):>8.3f}{t:>+7.2f}{p:>9.3f}{doc:>9.1f}%")
    effs.append(d.mean())
    ses.append(d.std(ddof=1) / np.sqrt(len(d)))

print("\n=== 自由门控对照（无配对 nogua 臂，用 V3_nogua 作对照）===")
sv, sn = S["Vfree"]["seeds"], S["V3_nogua"]["seeds"]
common = sorted(set(sv) & set(sn), key=int)
d = np.array([sv[s] - sn[s] for s in common])
t, p = stats.ttest_rel([sv[s] for s in common], [sn[s] for s in common])
print(f"Vfree vs V3_nogua: 配对差 {d.mean():+.3f}px  SD {d.std(ddof=1):.3f}  t={t:+.2f}  p={p:.3f}  n={len(d)}")
effs.append(d.mean())
ses.append(d.std(ddof=1) / np.sqrt(len(d)))

# --- 以 variant 为重复单元的元分析（固定效应，逆方差加权）---
effs, ses = np.array(effs), np.array(ses)
w = 1 / ses ** 2
mu = (w * effs).sum() / w.sum()
se_mu = 1 / np.sqrt(w.sum())
z = mu / se_mu
p_mu = 2 * (1 - stats.norm.cdf(abs(z)))
q = (w * (effs - mu) ** 2).sum()
df = len(effs) - 1
print(f"\n=== 元分析（k={len(effs)} 个 variant 为重复单元，固定效应）===")
print(f"  合并效应 = {mu:+.3f}px  (SE {se_mu:.3f}, 95%CI [{mu-1.96*se_mu:+.3f}, {mu+1.96*se_mu:+.3f}])")
print(f"  z = {z:+.2f}  p = {p_mu:.3f}")
print(f"  异质性 Cochran Q = {q:.2f} (df={df}, p={1-stats.chi2.cdf(q, df):.3f})")
print(f"  逐 variant 效应：{', '.join(f'{e:+.3f}' for e in effs)}")
neg = int((effs < 0).sum())
print(f"  方向为负（门控更好）的 variant 数：{neg}/{len(effs)}")

# 落盘：论文 §5.2 引用合并效应与其 95% CI，必须可溯源（评审 P1-3）
with open("analysis_output/gate_paired_meta.json", "w", encoding="utf-8") as f:
    json.dump(dict(
        k=int(len(effs)), pooled=float(mu), se=float(se_mu),
        ci_low=float(mu - 1.96 * se_mu), ci_high=float(mu + 1.96 * se_mu),
        z=float(z), p=float(p_mu), q=float(q), df=int(df),
        effects=[float(e) for e in effs], n_negative=neg,
        unit="variant (7 gating variants, shared seeds and architecture)",
        caveat="fixed-effect combination over non-independent units; z is "
               "anticonservative",
        source="analysis_output/E1a_stageB_report.json",
    ), f, ensure_ascii=False, indent=1)
print("  [save] analysis_output/gate_paired_meta.json")

# --- 用配对 SD 反推：要把 0.1px 的效应做到 80% 功效需要多少 seed ---
z_ = 1.96 + 0.8416
print("\n=== 要判定门控效应，需要多少 seed（配对，ITT 口径）===")
sd_pool = np.mean([np.std([S[v]['seeds'][s] - S[PAIR[v]]['seeds'][s]
                           for s in sorted(set(S[v]['seeds']) & set(S[PAIR[v]]['seeds']), key=int)],
                          ddof=1) for v in PAIR])
print(f"  平均配对 SD = {sd_pool:.3f}px")
for delta in (0.20, 0.15, 0.10, 0.05):
    print(f"  δ={delta:.2f}px → 需 n={np.ceil((z_*sd_pool/delta)**2):.0f} 个 seed（一对两臂）")
