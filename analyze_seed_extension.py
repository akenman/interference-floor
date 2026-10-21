# -*- coding: utf-8 -*-
"""Ten-seed summary of the pendulum prior ladder (E0 seeds 42-48 + extension 49-51).

Reads:
  analysis_output/noise_floor_matrix.json   seeds 42-48 (R1 own-split)
  analysis_output/e0_extra_seeds.json       seeds 49-51 (same protocol)

Writes:
  analysis_output/seed_extension_summary.json

Computes, per tier and metric: 7-seed and 10-seed mean/sd/CV, the change in the
estimate, and a per-seed monotonicity count over all ten seeds (the ladder is
ordered euler005 > euler001x5 > rk4005 > rk4001x5 in both columns).

Descriptive only: all ten seeds share the same source dataset and protocol, and
the extension seeds reuse the same recipe. Validation splits differ per seed.

Usage: python -X utf8 analyze_seed_extension.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
AO = os.path.join(ROOT, "analysis_output")
TIERS = ["euler005", "euler001x5", "rk4005", "rk4001x5"]
BASE = list(range(42, 49))
METRICS = [("one_step", "one_step", "one-step"), ("k=5", "k5", "k=5")]


def main():
    ext = json.load(open(os.path.join(AO, "noise_floor_matrix.json"), encoding="utf-8"))["tiers"]
    new = json.load(open(os.path.join(AO, "e0_extra_seeds.json"), encoding="utf-8"))
    new_seeds = sorted(int(k) for k in new)
    all_seeds = BASE + new_seeds

    def value(tier, seed, bcol, ncol):
        if seed in BASE:
            return float(ext[tier]["R1_own_split"][str(seed)][bcol])
        return float(new[str(seed)][tier][ncol])

    out = {"base_seeds": BASE, "extension_seeds": new_seeds, "all_seeds": all_seeds,
           "protocol": "e0_common.validate, per-seed own split", "tiers": {}}

    print(f"{'tier':11s} {'metric':9s} {'n=7 mean+-sd':>19s} {'n=10 mean+-sd':>19s} "
          f"{'CV7':>7s} {'CV10':>7s} {'min':>8s} {'max':>8s}")
    for tier in TIERS:
        out["tiers"][tier] = {}
        for bcol, ncol, lab in METRICS:
            v7 = np.array([value(tier, s, bcol, ncol) for s in BASE])
            v10 = np.array([value(tier, s, bcol, ncol) for s in all_seeds])
            rec = dict(metric=lab, n7=7, n10=10,
                       mean7=float(v7.mean()), sd7=float(v7.std(ddof=1)),
                       cv7_pct=float(100 * v7.std(ddof=1) / v7.mean()),
                       mean10=float(v10.mean()), sd10=float(v10.std(ddof=1)),
                       cv10_pct=float(100 * v10.std(ddof=1) / v10.mean()),
                       min10=float(v10.min()), max10=float(v10.max()),
                       extension_values={str(s): float(value(tier, s, bcol, ncol))
                                         for s in new_seeds})
            out["tiers"][tier][lab] = rec
            print(f"{tier:11s} {lab:9s} {rec['mean7']:8.4f}+-{rec['sd7']:.4f} "
                  f"{rec['mean10']:8.4f}+-{rec['sd10']:.4f} {rec['cv7_pct']:6.1f}% "
                  f"{rec['cv10_pct']:6.1f}% {rec['min10']:8.4f} {rec['max10']:8.4f}")

    # per-seed monotonicity over the ten seeds, both columns
    mono = {}
    for bcol, ncol, lab in METRICS:
        ok, detail = 0, {}
        for s in all_seeds:
            v = [value(t, s, bcol, ncol) for t in TIERS]
            good = all(v[i] > v[i + 1] for i in range(len(v) - 1))
            detail[str(s)] = good
            ok += int(good)
        mono[lab] = dict(n_monotone=ok, n_seeds=len(all_seeds), per_seed=detail)
        print(f"\nper-seed monotone ({lab}): {ok}/{len(all_seeds)}")
        for s in all_seeds:
            if not detail[str(s)]:
                print(f"  NOT monotone: s{s} -> "
                      + ", ".join(f"{value(t, s, bcol, ncol):.4f}" for t in TIERS))
    out["monotonicity"] = mono

    # ladder separation on the ten-seed means
    sep = {}
    for lab in ("one-step", "k=5"):
        a = out["tiers"]["euler005"][lab]["mean10"]
        b = out["tiers"]["rk4001x5"][lab]["mean10"]
        sep[lab] = dict(coarse_mean10=a, exact_mean10=b, ratio=a / b)
        print(f"10-seed ladder separation ({lab}): {a:.4f} / {b:.4f} = {a/b:.1f}x")
    out["ladder_separation_mean10"] = sep

    path = os.path.join(AO, "seed_extension_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\n[save] {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
