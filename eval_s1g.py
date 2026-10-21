# -*- coding: utf-8 -*-
"""S1g evaluation: the mechanism-ablation arm (w_ff=0) on the S1c protocol.

Why this script exists
----------------------
The paper quotes "excluding free-flight windows from the gradient entirely
(w_ff=0) leaves the noise at 0.0213 against the S1c baseline 0.0186". The number
0.0213 exists in no stored JSON: S1g was preregistered
(experiments/prereg_S1g_mechanism_ablation.json) and trained
(checkpoints-S1g/wff0_s42) but never evaluated into a report.

This script evaluates that checkpoint with the *same* machinery as
tmp_s1c_compare.py: same dataset, same seed-42 validation split, same window
subsampling (val_windows=20000, RandomState(0)), and the same head-RMS-on-FF
definition. It also re-evaluates the two comparators on the same protocol so the
three numbers are directly comparable.

Comparators, exactly as the paper names them:
  baseline : checkpoints-S1b/euler001x5_s42   (no reweighting)
  wff10    : checkpoints-S1c/wff10_s42        (free-flight windows x10)

Usage: python -X utf8 eval_s1g.py
Output: analysis_output/S1g_report.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from tmp_s1c_compare import (head_rms_on_ff, load_model, eval_model,   # noqa: E402
                             transfer_clean_mask)
from train_v3 import V3Data, load_dataset, split_episodes              # noqa: E402

SEED = 42
ARMS = {
    "baseline": "checkpoints-S1b/euler001x5_s{seed}/best.pt",
    "wff10": "checkpoints-S1c/wff10_s{seed}/best.pt",
    "wff0": "checkpoints-S1g/wff0_s{seed}/best.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-windows", type=int, default=20000)
    ap.add_argument("--out", default="S1g_report.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading trajectories_v3.npz ...", flush=True)
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]

    res = {}
    for arm, pat in ARMS.items():
        path = os.path.join(ROOT, pat.format(seed=SEED))
        model, ck = load_model(path)
        ds_args = ck["args"]
        _, val_eps = split_episodes(N, SEED)
        ns = ck["norm_stats"]
        ds = V3Data(states, obstacles, contact, actions, val_eps,
                    ds_args["seq_len"], ds_args["future_steps"],
                    ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"],
                    prior_spec=ck["config"]["prior_spec"])
        idx_all = np.arange(len(ds))
        if args.val_windows and len(idx_all) > args.val_windows:
            idx_all = np.sort(np.random.RandomState(0).choice(
                idx_all, size=args.val_windows, replace=False))
        r = eval_model(model.to(device), ds, device, idx_all, ds_args["future_steps"])
        r["head_rms_ff"], r["n_ff"] = head_rms_on_ff(model, ds, device, idx_all)
        r["prior_spec"] = ck["config"]["prior_spec"]
        r["val_windows"] = int(len(idx_all))
        res[arm] = r
        print(f"[{arm}] mixed={r['one_mixed']:.4f}  FF={r['one_ff_clean']:.4f}  "
              f"head_rms_ff={r['head_rms_ff']:.4f}  n_ff={r['n_ff']}", flush=True)
        del model
        torch.cuda.empty_cache()

    b, w10, w0 = res["baseline"], res["wff10"], res["wff0"]
    shift_pct = 100.0 * (w0["head_rms_ff"] - b["head_rms_ff"]) / b["head_rms_ff"]
    out = {
        "prereg": "experiments/prereg_S1g_mechanism_ablation.json",
        "protocol": ("same dataset, seed-42 val split, val_windows=20000 subsample "
                     "(RandomState(0)) and head-RMS-on-FF definition as tmp_s1c_compare.py"),
        "seed": SEED,
        "arms": res,
        "comparison": {
            "baseline_head_rms_ff": b["head_rms_ff"],
            "wff10_head_rms_ff": w10["head_rms_ff"],
            "wff0_head_rms_ff": w0["head_rms_ff"],
            "wff0_vs_baseline_shift_pct": shift_pct,
            "prereg_interpretation": (
                "if ~0.019: gradient crosstalk confirmed; "
                "if ~0.010: mixed; if <0.005: FF self-loss drives the noise"),
        },
    }
    path = os.path.join(ROOT, "analysis_output", args.out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nwff0 vs baseline head RMS: {w0['head_rms_ff']:.4f} vs "
          f"{b['head_rms_ff']:.4f}  ({shift_pct:+.1f}%)")
    print(f"[save] analysis_output/{args.out}")


if __name__ == "__main__":
    main()
