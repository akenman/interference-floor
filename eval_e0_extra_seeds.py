# -*- coding: utf-8 -*-
"""Evaluate extension seeds (49, 50, 51, ...) on all four prior tiers.

Uses the same protocol as build_e0_report.py: per-seed own validation split
(make_datasets(seed)) and e0_common.validate(..., K=5, max_windows=30000).

Reads checkpoints from checkpoints-E0-extra/. For each checkpoint it asserts the
prior_spec recorded inside the file matches the tier directory name, so a
mislabeled directory cannot silently enter the ladder.

Usage: python -X utf8 eval_e0_extra_seeds.py --seeds 49 50 51
Output: analysis_output/e0_extra_seeds.json
"""
import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from e0_common import make_datasets, validate      # noqa: E402
import model_e0                                     # noqa: E402

DATA = os.path.join(ROOT, "trajectories_hinge.npz")
EXTRA = os.path.join(ROOT, "checkpoints-E0-extra")
TIERS = [("euler005", "ours_base"),
         ("euler001x5", "ours_base@euler001x5"),
         ("rk4005", "ours_base@rk4005"),
         ("rk4001x5", "ours_base@rk4001x5")]


def evaluate(seed, spec, key):
    path = os.path.join(EXTRA, f"{key}_s{seed}", "best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    rec = (ck.get("args") or {}).get("prior_spec")
    if rec != spec:
        raise SystemExit(f"{key}_s{seed}: recorded prior_spec={rec!r} != tier {spec!r}")

    _, val_ds, mo, so = make_datasets(DATA, L=8, K=5, seed=seed, prior_spec=spec)
    model = model_e0.make_model("ours_base", prior_spec=spec).to("cpu")
    model.load_state_dict(ck["model_state_dict"])
    model.set_stats(mo, so)
    model.eval()
    vm = validate(model, val_ds, torch.device("cpu"), K=5, max_windows=30000)
    return dict(key=key, seed=seed, prior_spec=rec, epoch=ck.get("epoch"),
                one_step=float(vm["one_step_model"]),
                k2=float(vm["rollout"]["k=2"]),
                k5=float(vm["rollout"]["k=5"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", default="e0_extra_seeds.json")
    args = ap.parse_args()

    out = {}
    for seed in args.seeds:
        out[str(seed)] = {}
        for spec, key in TIERS:
            rec = evaluate(seed, spec, key)
            out[str(seed)][spec] = rec
            print(f"s{seed} {spec:11s} one={rec['one_step']:.4f} "
                  f"k2={rec['k2']:.4f} k5={rec['k5']:.3f} epoch={rec['epoch']}")

    path = os.path.join(ROOT, "analysis_output", args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[save] {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
