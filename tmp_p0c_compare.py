# -*- coding: utf-8 -*-
"""P0-c 辅助监督 2×2 配对分析（T5 步骤 2；判据以预注册为准，不得事后更改）。

预注册：experiments/prereg_P0c_aux_supervision.json
输入：analysis_output/P0c_{C00,C01,C10,C11}.json
      （由 build_e0_report.py --checkpoint_root checkpoints-P0c/<格> --skip_analytic
        --out analysis_output/P0c_<格>.json 生成，四份都要先存在）

口径：同 seed 配对 + ITT（缺 seed 即记崩溃，崩溃率入报告）。
主判据（单步 one_step_model；门槛 ±0.23px = n=6 MDE，来自 E0 效应 0.56px 与配对 SD≈0.2px）：
  P0c-1 爻监督主效应   = mean(C10, C11) − mean(C00, C01)
  P0c-2 原型家族主效应 = mean(C01, C11) − mean(C00, C10)
  P0c-3 交互项         = (C11 − C10) − (C01 − C00)   （只报告，不设门槛）
  >+0.23 ⇒ 有害；<−0.23 ⇒ 有益；其间 ⇒ 无影响（该配方下）
次口径（描述性）：闭环 k=5；yao_acc 监督生效核验（C10/C11 预期 ≈0.96x，
  C00/C01 预期 ≈0.7x——若 C10/C11 的 yao_acc 不升，判"监督未生效"，实验作废重查）。

用法：python tmp_p0c_compare.py [--no-figure]
输出：控制台判决 + analysis_output/P0c_compare.json + figures/P0c_two_by_two.png
"""
import argparse
import io
import json
import sys

import numpy as np
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CELLS = ["C00", "C01", "C10", "C11"]
REPORTS = {c: f"analysis_output/P0c_{c}.json" for c in CELLS}
EXPECTED_SEEDS = [42, 43, 44, 45, 46, 47]
THRESH = 0.23          # 预注册主效应门槛（px）
EXPECT_YAO = {"on": 0.90, "off": 0.80}   # 监督生效核验的宽松界（0.96x vs ~0.7x 之间留裕度）


def load_cell(cell):
    try:
        r = json.load(open(REPORTS[cell], encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"[缺报告] {REPORTS[cell]} 不存在——先运行 build_e0_report.py 四份"
                 f"（见 prereg_P0c_aux_supervision.json / HANDOFF T5）")
    runs = r["arms"]["ours_base"]
    return {x["seed"]: x for x in runs}, r


def metric(run, name):
    if name == "one":
        return run["one_step_model"]
    if name == "k5":
        return run["rollout"]["k=5"]
    if name == "yao_acc":
        return run["yao_acc"]
    raise KeyError(name)


def cell_vectors(per, seeds, name):
    """按 seed 取每格指标；缺 seed 记 NaN（ITT 崩溃）。"""
    return {c: np.array([metric(per[c][s], name) if s in per[c] else np.nan
                         for s in seeds]) for c in CELLS}


def verdict(eff):
    if eff > THRESH:
        return "有害"
    if eff < -THRESH:
        return "有益"
    return "无影响（该配方下）"


def main_effect(vals, on_cells, off_cells, seeds, label):
    """主效应 = mean(on) − mean(off)，同 seed 配对（每 seed 两格各取均值）。
    返回 dict；NaN（崩溃缺失）按可用 seed 对计算并记录。"""
    on = np.nanmean(np.stack([vals[c] for c in on_cells]), axis=0)
    off = np.nanmean(np.stack([vals[c] for c in off_cells]), axis=0)
    d = on - off
    ok = np.isfinite(d)
    n_ok = int(ok.sum())
    eff = float(np.mean(d[ok]))
    if n_ok >= 2:
        t = stats.ttest_rel(on[ok], off[ok])
        t_stat, p = float(t.statistic), float(t.pvalue)
        sd = float(np.std(d[ok], ddof=1))
        # 配对 t 区间：df = n-1（6 对时临界值约 2.571，不是正态的 1.96）
        tcrit = stats.t.ppf(0.975, n_ok - 1)
        ci = float(tcrit * sd / np.sqrt(n_ok)) if n_ok > 1 else float("nan")
    else:
        t_stat = p = sd = ci = float("nan")
    same_dir = int(np.sum((d[ok] > 0))) if eff > 0 else int(np.sum((d[ok] < 0)))
    return dict(label=label, n_pairs=n_ok, n_missing=int((~ok).sum()),
                effect=eff, threshold=THRESH, verdict=verdict(eff),
                paired_t=t_stat, p=p, ci95_half_width=float(ci),
                direction_consistency=f"{same_dir}/{n_ok}",
                per_seed=[float(x) if np.isfinite(x) else None for x in d],
                seeds=list(seeds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    per, raw = {}, {}
    for c in CELLS:
        per[c], raw[c] = load_cell(c)
        got = sorted(per[c])
        missing = [s for s in EXPECTED_SEEDS if s not in per[c]]
        print(f"[{c}] seeds={got} 缺失(记崩溃)={missing}")

    seeds = EXPECTED_SEEDS
    crash_rate = {c: (len(EXPECTED_SEEDS) - len(per[c])) / len(EXPECTED_SEEDS)
                  for c in CELLS}

    out = {"prereg": "experiments/prereg_P0c_aux_supervision.json",
           "threshold_px": THRESH, "crash_rate_ITT": crash_rate}

    # ---- 主判据（按预注册顺序）----
    print("\n=== 主判据（单步 one_step_model，同 seed 配对，门槛 ±0.23px）===")
    one = cell_vectors(per, seeds, "one")
    for label, on_c, off_c in (
            ("P0c-1 爻监督主效应  mean(C10,C11)−mean(C00,C01)", ("C10", "C11"), ("C00", "C01")),
            ("P0c-2 原型家族主效应 mean(C01,C11)−mean(C00,C10)", ("C01", "C11"), ("C00", "C10"))):
        r = main_effect(one, on_c, off_c, seeds, label)
        out[label.split(" ")[0]] = r
        print(f"  {label}\n    effect={r['effect']:+.3f}px  [±{r['ci95_half_width']:.3f}]  "
              f"t={r['paired_t']:+.2f} p={r['p']:.4f} 方向一致 {r['direction_consistency']}"
              f"  ⇒ {r['verdict']}")

    # 交互项须用差对差：手工按 seed 算（main_effect 只做 on/off 两格均值）
    inter = ((one["C11"] - one["C10"]) - (one["C01"] - one["C00"]))
    ok = np.isfinite(inter)
    inter_eff = float(np.mean(inter[ok])) if ok.any() else float("nan")
    out["P0c-3 交互项"] = dict(effect=inter_eff, n_pairs=int(ok.sum()),
                               per_seed=[float(x) if np.isfinite(x) else None for x in inter],
                               note="只报告，不设门槛")
    print(f"  P0c-3 交互项 (C11−C10)−(C01−C00) = {inter_eff:+.3f}px（只报告）")

    # ---- 次口径 ----
    print("\n=== 次口径 ===")
    k5 = cell_vectors(per, seeds, "k5")
    ya = cell_vectors(per, seeds, "yao_acc")
    for c in CELLS:
        print(f"  {c}: one={np.nanmean(one[c]):.3f}  k5={np.nanmean(k5[c]):.2f}  "
              f"yao_acc={np.nanmean(ya[c]):.3f}")
    out["cells_summary"] = {
        c: dict(one=float(np.nanmean(one[c])), k5=float(np.nanmean(k5[c])),
                yao_acc=float(np.nanmean(ya[c]))) for c in CELLS}

    # 监督生效核验：爻监督开的格（C10/C11）yao_acc 必须显著高于关的格（C00/C01）
    yao_on = np.nanmean(np.concatenate([ya["C10"], ya["C11"]]))
    yao_off = np.nanmean(np.concatenate([ya["C00"], ya["C01"]]))
    ok_sup = yao_on >= EXPECT_YAO["on"] and yao_off <= EXPECT_YAO["off"]
    out["yao_supervision_check"] = dict(yao_on=float(yao_on), yao_off=float(yao_off),
                                        passed=bool(ok_sup))
    print(f"  监督生效核验: yao_acc(on)={yao_on:.3f} (≥{EXPECT_YAO['on']}) vs "
          f"off={yao_off:.3f} (≤{EXPECT_YAO['off']}) ⇒ "
          f"{'PASS' if ok_sup else 'FAIL——监督未生效，实验作废重查'}")

    print(f"\n崩溃率(ITT): " + ", ".join(f"{c}={crash_rate[c]:.0%}" for c in CELLS))

    with open("analysis_output/P0c_compare.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/P0c_compare.json")

    if not args.no_figure:
        fig = None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            for ax, (name, tag) in zip(axes, (("one-step (px)", "one"), ("rollout k=5 (px)", "k5"))):
                v = cell_vectors(per, seeds, tag)
                means = [np.nanmean(v[c]) for c in CELLS]
                ax.bar(CELLS, means, color="#4C72B0")
                for i, s in enumerate(seeds):
                    ax.plot(CELLS, [v[c][i] if np.isfinite(v[c][i]) else np.nan
                                    for c in CELLS], "k.", alpha=0.35)
                ax.set_title(name)
                ax.set_xlabel("爻监督 → / 原型家族 ↑   (C00 无, C01 仅原型, C10 仅爻, C11 全开)")
            fig.suptitle("P0-c 辅助监督 2×2（点 = 各 seed；预注册门槛 ±0.23px）")
            fig.tight_layout()
            fig.savefig("figures/P0c_two_by_two.png", dpi=150)
            print("[save] figures/P0c_two_by_two.png")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 出图失败（不影响判决）: {e}")
        finally:
            if fig is not None:
                plt.close(fig)


if __name__ == "__main__":
    main()
