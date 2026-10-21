# -*- coding: utf-8 -*-
"""SLOT-0b 判决脚本：减法是否被地板吸收（判据以预注册为准，不得事后更改）。

预注册：experiments/prereg_S0b_floor_absorption.json
输入：analysis_output/S0b_report.json（新跑 ours_gua@rk4001x5 × 3 seed）
      + analysis_output/E0_report.json（既有 ours_base@rk4001x5 与 ours_gua@euler005）

口径：同 seed 配对（n=3），ITT（无 best.pt 记配对缺失）；按预注册顺序报告。
判据（门槛以绝对 px 写死，均 ≥5× 噪声底）：
  S0b-1 主判据（单步）：Δone = one(gua@exact) − one(base@exact)
        ≤ +0.010px ⇒ 吸收；≥ +0.050px ⇒ 不吸收；其间 ⇒ UNDECIDED
  S0b-2 闭环同向：k5(gua@exact) / k5(base@exact)
        ≤ ×1.10 ⇒ 同向吸收；≥ ×1.30 ⇒ 不吸收同向；其间 ⇒ UNDECIDED
  S0b-3 效应量比（只报告）：ratio = Δone@exact / Δone@coarse（粗档分母 +0.272px 复算）
次口径：rollout 曲线、yao_acc、gua occupancy 审计、崩溃率。

用法：python tmp_s0b_compare.py [--no-figure]
输出：控制台判决 + analysis_output/S0b_compare.json + figures/S0b_floor_absorption.png
"""
import argparse
import io
import json
import sys

import numpy as np
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

S0B_REPORT = "analysis_output/S0b_report.json"
E0_REPORT = "analysis_output/E0_report.json"
EXPECTED_SEEDS = [42, 43, 44]
ABS_LOW, ABS_HIGH = 0.010, 0.050      # S0b-1 门槛（px）
REL_LOW, REL_HIGH = 1.10, 1.30        # S0b-2 门槛（×）
COARSE_DENOM_EXPECT = 0.272           # 粗档效应（E0 复算应≈此值）


def load_runs(path, key):
    r = json.load(open(path, encoding="utf-8"))
    return {x["seed"]: x for x in r["arms"][key]}


def paired_diff(a, b, seeds, fn):
    """按 seed 配对 a−b；缺 seed 记 NaN。返回 (per-seed array, 有效数)。"""
    d = np.array([fn(a[s]) - fn(b[s]) if (s in a and s in b) else np.nan
                  for s in seeds])
    return d, int(np.isfinite(d).sum())


def paired_ratio(a, b, seeds, fn):
    r = np.array([fn(a[s]) / fn(b[s]) if (s in a and s in b) and fn(b[s]) else np.nan
                  for s in seeds])
    return r, int(np.isfinite(r).sum())


def ttest(d):
    ok = np.isfinite(d)
    if ok.sum() < 2:
        return float("nan"), float("nan")
    t = stats.ttest_1samp(d[ok], 0.0)   # 配对差对 0 的单样本 t（符号 = 差的方向）
    return float(t.statistic), float(t.pvalue)


def verdict_abs(delta):
    if np.isfinite(delta):
        if delta <= ABS_LOW:
            return "吸收（主轴精确化 4 成立）"
        if delta >= ABS_HIGH:
            return "不吸收（减法含地板无关通道）"
    return "UNDECIDED"


def verdict_rel(ratio):
    if np.isfinite(ratio):
        if ratio <= REL_LOW:
            return "闭环同向吸收"
        if ratio >= REL_HIGH:
            return "不吸收同向"
    return "UNDECIDED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    gua_x = load_runs(S0B_REPORT, "ours_gua@rk4001x5")
    base_x = load_runs(E0_REPORT, "ours_base@rk4001x5")
    gua_c = load_runs(E0_REPORT, "ours_gua")
    base_c = load_runs(E0_REPORT, "ours_base")

    for name, d in (("gua@exact", gua_x), ("base@exact", base_x),
                    ("gua@coarse", gua_c), ("base@coarse", base_c)):
        missing = [s for s in EXPECTED_SEEDS if s not in d]
        print(f"[{name}] seeds={sorted(d)} 缺失={missing}")

    one = lambda r: r["one_step_model"]
    k5 = lambda r: r["rollout"]["k=5"]
    out = {"prereg": "experiments/prereg_S0b_floor_absorption.json",
           "thresholds": {"abs_px": [ABS_LOW, ABS_HIGH], "rel_x": [REL_LOW, REL_HIGH]}}

    # ---- S0b-1 主判据 ----
    print("\n=== S0b-1 主判据（单步配对差，门槛 +0.010 吸收 / +0.050 不吸收）===")
    d_exact, n1 = paired_diff(gua_x, base_x, EXPECTED_SEEDS, one)
    eff1 = float(np.nanmean(d_exact)) if n1 else float("nan")
    t1, p1 = ttest(d_exact)
    same1 = int(np.sum(d_exact[np.isfinite(d_exact)] > 0))
    out["S0b-1"] = dict(per_seed=[float(x) if np.isfinite(x) else None for x in d_exact],
                        effect=eff1, n=n1, paired_t=t1, p=p1,
                        direction=f"{same1}/{n1}", verdict=verdict_abs(eff1))
    print(f"  Δone@exact = {eff1:+.4f}px  t={t1:+.2f} p={p1:.4f}  方向 {same1}/{n1}"
          f"  ⇒ {verdict_abs(eff1)}")

    # ---- S0b-2 闭环同向 ----
    print("=== S0b-2 闭环同向（k=5 配对比值，门槛 ×1.10 / ×1.30）===")
    r_exact, n2 = paired_ratio(gua_x, base_x, EXPECTED_SEEDS, k5)
    eff2 = float(np.nanmean(r_exact)) if n2 else float("nan")
    out["S0b-2"] = dict(per_seed=[float(x) if np.isfinite(x) else None for x in r_exact],
                        ratio=eff2, n=n2, verdict=verdict_rel(eff2))
    print(f"  k5 ratio@exact = ×{eff2:.3f}  ⇒ {verdict_rel(eff2)}")

    # ---- S0b-3 效应量比 ----
    d_coarse, n3 = paired_diff(gua_c, base_c, EXPECTED_SEEDS, one)
    eff3 = float(np.nanmean(d_coarse)) if n3 else float("nan")
    ratio = eff1 / eff3 if (np.isfinite(eff1) and np.isfinite(eff3) and eff3) else float("nan")
    out["S0b-3"] = dict(coarse_effect=eff3, coarse_n=n3,
                        coarse_expect=COARSE_DENOM_EXPECT, ratio=ratio)
    print(f"=== S0b-3 效应量比 ===\n  Δone@coarse = {eff3:+.3f}px（预期 ≈{COARSE_DENOM_EXPECT}）"
          f"  ratio = {ratio:.4f}（<0.1 强吸收 / ≈1 无吸收）")

    # ---- 次口径 ----
    print("\n=== 次口径 ===")
    for name, d in (("gua@exact", gua_x), ("base@exact", base_x)):
        ks = [d[s]["rollout"][f"k={k}"] for s in sorted(d) for k in (1, 5)]
        ya = np.mean([d[s]["yao_acc"] for s in sorted(d)])
        print(f"  {name}: one={np.mean([one(d[s]) for s in sorted(d)]):.4f}  "
              f"k1={ks[0]:.3f}  k5={ks[1]:.3f}  yao_acc={ya:.3f}")
        out.setdefault("cells", {})[name] = dict(
            one=float(np.mean([one(d[s]) for s in sorted(d)])),
            k1=float(ks[0]), k5=float(ks[1]), yao_acc=float(ya))
    occ = gua_x[sorted(gua_x)[0]].get("gua_occupancy", {})
    if occ:
        alive = sum(1 for v in occ.values() if v > 0)
        minf = min(occ.values()) if occ else 0.0
        out["gua_occupancy_audit"] = dict(alive_groups=alive, min_frac=float(minf))
        print(f"  gua occupancy：存活组 {alive}/64，最小组占比 {minf:.4f}")

    crash = (len(EXPECTED_SEEDS) - len(gua_x)) / len(EXPECTED_SEEDS)
    out["crash_rate_ITT"] = crash
    print(f"崩溃率(ITT): {crash:.0%}")

    with open("analysis_output/S0b_compare.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/S0b_compare.json")

    if not args.no_figure:
        fig = None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            for ax, (tag, fn, lab) in zip(
                    axes, (("one-step (px)", one, "one"), ("rollout k=5 (px)", k5, "k5"))):
                labels = ["base@exact", "gua@exact", "base@coarse", "gua@coarse"]
                means = [np.mean([fn(d[s]) for s in sorted(d)])
                         for d in (base_x, gua_x, base_c, gua_c)]
                ax.bar(labels, means, color="#4C72B0")
                for i, d in enumerate((base_x, gua_x, base_c, gua_c)):
                    ax.plot([i] * len(d), [fn(d[s]) for s in sorted(d)], "k.", alpha=0.4)
                ax.set_title(lab)
                ax.set_yscale("log")
            fig.suptitle("SLOT-0b: structural subtraction x prior floor "
                         "(dots = seeds; thresholds +0.010/+0.050 px)")
            fig.tight_layout()
            fig.savefig("figures/S0b_floor_absorption.png", dpi=150)
            print("[save] figures/S0b_floor_absorption.png")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 出图失败（不影响判决）: {e}")
        finally:
            if fig is not None:
                plt.close(fig)


if __name__ == "__main__":
    main()
