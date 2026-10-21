# -*- coding: utf-8 -*-
"""E1a / Stage D 功效分析（零训练成本，numpy 向量化）。

目的：定量回答「现有数据能支撑多大的结论」，为论文方法节提供可复现的功效
声明，而不是事后追加 seed（见 DEEP_DIVE_REPORT 10.14 约束）。

判定规则与 build_e1a_report.judge_pair 完全一致：
    PASS      当 CI 下界 >= threshold
    FAIL      当 CI 上界 <  threshold
    UNDECIDED 其余
故「功效」= 真实效应恰为 threshold 时 P(PASS)；「假阳性率」= 真实效应为 0 时
P(判出 PASS 或 FAIL)。
"""
import io
import json
import math
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_SIM = 200000
RNG = np.random.default_rng(20260929)

# 95% t 临界值（双侧），df 1..40 查表 + 大 df 正态近似
_T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
      8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
      14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
      20: 2.086, 24: 2.064, 29: 2.045, 40: 2.021}


def tcrit(df):
    if df in _T:
        return _T[df]
    if df < 1:
        return 12.706
    if df > 40:
        return 1.959964
    ks = sorted(_T)
    for a, b in zip(ks, ks[1:]):
        if a <= df <= b:
            return _T[a] + (_T[b] - _T[a]) * (df - a) / (b - a)
    return 1.959964


def mde(n, sd, alpha=0.05, power=0.80):
    """最小可检测效应（配对 t，双侧）。"""
    from statistics import NormalDist
    za = NormalDist().inv_cdf(1 - alpha / 2)
    zb = NormalDist().inv_cdf(power)
    return (za + zb) * sd / math.sqrt(n)


def simulate(n, sd, true_delta, thr, n_sim=N_SIM):
    """向量化模拟报告判定规则，返回 (PASS, FAIL, UNDECIDED) 比例。"""
    tc = tcrit(max(1, n - 1))
    xs = RNG.normal(true_delta, sd, size=(n_sim, n))
    m = xs.mean(axis=1)
    se = xs.std(axis=1, ddof=1) / math.sqrt(n)
    half = tc * se
    n_pass = int(np.sum(m - half >= thr))
    n_fail = int(np.sum(m + half < thr))
    return n_pass / n_sim, n_fail / n_sim, 1 - (n_pass + n_fail) / n_sim


def need_n(sd, true_delta, thr, target=0.80, cap=600):
    """达到目标功效所需最小 n：从解析估计起步二分细化。"""
    lo, hi = 4, cap
    if simulate(hi, sd, true_delta, thr, n_sim=4000)[0] < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if simulate(mid, sd, true_delta, thr, n_sim=8000)[0] >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def load(path, only_prefix=None):
    d = json.load(open(path, encoding="utf-8"))
    crit = dict(d.get("criteria", {}))
    crit.update(d.get("criteria_decoupled_exploratory", {}))
    out = []
    for k, c in sorted(crit.items()):
        # D 报告里 criteria 也含 E1a-* 行；按前缀去重，避免同一判据重复统计
        if only_prefix and not k.startswith(only_prefix):
            continue
        pd = c.get("paired_deltas") or []
        n = len(pd)
        if n < 2:
            continue
        delta = c["delta"]
        sd = float(np.std(pd, ddof=1))
        out.append(dict(key=k, var=c["variant"], n=n, delta=delta, sd=sd,
                        dz=delta / sd if sd else float("nan"),
                        thr=c.get("threshold", 0.05),
                        ci=c.get("ci95"), verdict=c.get("verdict")))
    return out


def show(title, rows):
    print("=" * 96)
    print(title)
    print("=" * 96)
    print("%-8s %-6s %3s %9s %9s %7s %10s %11s %12s" %
          ("判据", "臂", "n", "Δ观测", "sd配对", "dz", "MDE@n", "功效@门槛", "假阳率@0"))
    for r in rows:
        pw, _, _ = simulate(r["n"], r["sd"], r["thr"], r["thr"], n_sim=40000)
        fp, ff, _ = simulate(r["n"], r["sd"], 0.0, r["thr"], n_sim=40000)
        print("%-8s %-6s %3d %+9.4f %9.4f %7.3f %10.4f %10.1f%% %11.1f%%" %
              (r["key"], r["var"], r["n"], r["delta"], r["sd"], r["dz"],
               mde(r["n"], r["sd"]), pw * 100, (fp + ff) * 100))
    print()
    return rows


def main():
    a = show("Stage A 预注册爻轴消融（门槛 5% / 3%）",
             load("analysis_output/E1a_corrected_report.json", only_prefix="E1a-"))
    b = show("Stage D 解耦消融（探索性，基准 V3）",
             load("analysis_output/E1a_stageD_report.json", only_prefix="E1a-D"))
    allr = a + b

    print("=" * 96)
    print("达到 80% 功效所需的配对 n —— 真实效应 = 预注册门槛（即「刚好要判 PASS」）")
    print("=" * 96)
    print("%-8s %-6s %8s %10s %10s %10s" %
          ("判据", "臂", "门槛", "sd配对", "当前n", "需要n"))
    for r in allr:
        print("%-8s %-6s %8.3f %10.4f %10d %10s" %
              (r["key"], r["var"], r["thr"], r["sd"], r["n"],
               need_n(r["sd"], r["thr"], r["thr"]) or ">600"))

    print()
    print("=" * 96)
    print("若观测效应为真，需多少 n 才能 80% 功效检出")
    print("=" * 96)
    print("%-8s %-6s %10s %8s %10s" % ("判据", "臂", "Δ观测", "dz", "需要n"))
    for r in allr:
        if abs(r["delta"]) < 5e-3:
            print("%-8s %-6s %+10.4f %8.3f %10s" %
                  (r["key"], r["var"], r["delta"], r["dz"], "效应≈0，见上表"))
            continue
        print("%-8s %-6s %+10.4f %8.3f %10s" %
              (r["key"], r["var"], r["delta"], r["dz"],
               need_n(r["sd"], abs(r["delta"]), r["thr"]) or ">600"))

    print()
    print("=" * 96)
    print("对照：E0 主表 n=3 距离显著有多远（各臂 vs noprior，独立两样本）")
    print("=" * 96)
    e0 = json.load(open("analysis_output/E0_report.json", encoding="utf-8"))
    arms = {k: v for k, v in e0["arms"].items() if isinstance(v, list) and v}
    base = np.array([x["one_step_model"] for x in arms["noprior"]])
    bsd = base.std(ddof=1)
    print("%-11s %8s %8s %8s %12s %10s" %
          ("臂", "Δ", "sd合并", "dz", "需n(80%)", "现功效"))
    for k, recs in arms.items():
        if k == "noprior":
            continue
        v = np.array([x["one_step_model"] for x in recs])
        delta = v.mean() - base.mean()
        sp = np.sqrt((v.std(ddof=1) ** 2 + bsd ** 2) / 2)
        n_req = max(4, int(round(2 * (1.96 + 0.8416) ** 2 * sp ** 2 / delta ** 2)))
        # 现状功效：双样本 t（每组 n=3），df=4，临界 2.776
        se = np.sqrt(sp ** 2 / 3 + sp ** 2 / 3)
        t = abs(delta) / se
        pw = 1 - 0.0  # 近似：用正态近似粗估
        pw = float(1 - 0.5 * (1 + math.erf((2.776 - t) / math.sqrt(2))))
        print("%-11s %+8.3f %8.3f %8.2f %12d %9.1f%%" %
              (k, delta, sp, delta / sp, n_req, pw * 100))


if __name__ == "__main__":
    main()
