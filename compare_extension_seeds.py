# -*- coding: utf-8 -*-
"""Compare extension seeds (49+) against E0 seeds 42-48 on the same protocol.

Reads:
  analysis_output/noise_floor_matrix.json   E0 seeds 42-48 (R1 own-split)
  analysis_output/e0_extra_seeds.json       extension seeds (eval_e0_extra_seeds.py)

Writes:
  analysis_output/extension_seed_comparison.json

The comparison is descriptive: extension seeds are checked against the min-max
range and the mean±sd of the seven E0 seeds, tier by tier. Being outside that
range is not an error; it means the seven-seed spread did not cover that value.

Usage: python -X utf8 compare_extension_seeds.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
AO = os.path.join(ROOT, "analysis_output")
TIERS = ["euler005", "euler001x5", "rk4005", "rk4001x5"]
BASE = list(range(42, 49))
# tolerance for calling a value "at the boundary" rather than outside the range
BOUNDARY_TOL_PX = 1e-4


def main():
    ext = json.load(open(os.path.join(AO, "noise_floor_matrix.json"), encoding="utf-8"))["tiers"]
    new = json.load(open(os.path.join(AO, "e0_extra_seeds.json"), encoding="utf-8"))
    ext_seeds = sorted(int(k) for k in new)

    rows, flags = [], []
    for spec in TIERS:
        runs = ext[spec]["R1_own_split"]
        for bcol, ncol, lab in (("one_step", "one_step", "one-step"),
                                ("k=5", "k5", "k=5")):
            b = np.array([runs[str(s)][bcol] for s in BASE], dtype=float)
            lo, hi, m, sd = b.min(), b.max(), b.mean(), b.std(ddof=1)
            rec = dict(tier=spec, metric=lab, base_mean=float(m), base_sd=float(sd),
                       base_min=float(lo), base_max=float(hi), extension={})
            for s in ext_seeds:
                v = float(new[str(s)][spec][ncol])
                if v < lo - BOUNDARY_TOL_PX or v > hi + BOUNDARY_TOL_PX:
                    status = "outside_min_max"
                    flags.append((spec, lab, s, v, lo, hi))
                elif v < lo or v > hi:
                    status = "at_boundary"
                else:
                    status = "inside"
                rec["extension"][str(s)] = dict(value=v, status=status,
                                                z_vs_base=(v - m) / sd if sd else None)
            rows.append(rec)

    print(f"{'tier':11s} {'metric':9s} {'base mean+-sd':>20s} {'[min,max]':>20s}  "
          + "  ".join(f"s{s}" for s in ext_seeds))
    for r in rows:
        vals = "  ".join(f"{r['extension'][str(s)]['value']:.4f}" for s in ext_seeds)
        sts = ",".join(r["extension"][str(s)]["status"][:3] for s in ext_seeds)
        print(f"{r['tier']:11s} {r['metric']:9s} {r['base_mean']:8.4f}+-{r['base_sd']:.4f} "
              f"[{r['base_min']:.4f},{r['base_max']:.4f}]  {vals}  ({sts})")

    print("\n[outside seven-seed range beyond tolerance]")
    if flags:
        for spec, lab, s, v, lo, hi in flags:
            print(f"  {spec:11s} {lab:9s} s{s}: {v:.6f} outside [{lo:.6f},{hi:.6f}]")
    else:
        print("  none")

    out = dict(base_seeds=BASE, extension_seeds=ext_seeds,
               boundary_tol_px=BOUNDARY_TOL_PX,
               note=("Descriptive only. Extension seeds reuse the same data and protocol; "
                     "values outside the seven-seed range mean that range under-covered "
                     "the spread, not that the run is invalid."),
               rows=rows, outside_range=[dict(tier=a, metric=b, seed=c, value=d,
                                              base_min=e, base_max=f)
                                         for a, b, c, d, e, f in flags])
    path = os.path.join(AO, "extension_seed_comparison.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\n[save] {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
