# -*- coding: utf-8 -*-
"""Recompute pendulum head/target ratios on each seed's own held-out split (R1).

The original M4 ratio script, m4_floor_partial.py, fixes the split at seed 42 for all
checkpoints. That means most of this common validation set overlaps the training data
for seeds 43--51. This script uses each checkpoint's own make_datasets(seed) split,
so its validation episodes are excluded from that seed's training statistics/data.
It produces a descriptive sensitivity summary, not ten independent trials: seed
validation splits overlap each other, and the data-generating episodes are shared.

Checkpoints are read from checkpoints-E0 (seeds 42--48) and checkpoints-E0-extra
(seeds 49--51), in that order.

CPU-only; no training/GPU jobs are started.

Usage: python -X utf8 paper/tables/build_ladder_floor_r1.py
Output: analysis_output/ladder_floor_r1.json
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
AO = os.path.join(ROOT, "analysis_output")

from e0_common import make_datasets                              # noqa: E402
from layout_h import OBJ_DIMS, feat_weights                       # noqa: E402
from model_e0 import make_model                                   # noqa: E402

ARM = "ours_base"
SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
CKPT_ROOTS = (os.path.join(ROOT, "checkpoints-E0"),
              os.path.join(ROOT, "checkpoints-E0-extra"))
L, K, MAXW = 8, 5, 30000
POS = [1, 2, 12, 13]
W_POS = feat_weights(OBJ_DIMS)[POS]
TIERS = [
    ("euler005", "coarse Euler@.05x1"),
    ("euler001x5", "Euler@.01x5"),
    ("rk4005", "RK4@.05x1"),
    ("rk4001x5", "exact RK4@.01x5"),
]


def key(spec):
    return ARM if spec == "euler005" else f"{ARM}@{spec}"


def ckpt_path(spec, seed):
    """First existing checkpoint across the configured roots."""
    name = f"{key(spec)}_s{seed}"
    for root in CKPT_ROOTS:
        p = os.path.join(root, name, "best.pt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name}: not found in {[os.path.basename(r) for r in CKPT_ROOTS]}")


def main():
    data_path = os.path.join(ROOT, "trajectories_hinge.npz")
    per_tier = {}
    print("R1: each checkpoint evaluated on its own seed's held-out split")
    print("tier                 seed       target_RMS      head_RMS       ratio      n_FF")
    for spec, label in TIERS:
        per = {}
        for seed in SEEDS:
            path = ckpt_path(spec, seed)
            train_ds, val_ds, mo, so = make_datasets(
                data_path, L=L, K=K, seed=seed, prior_spec=spec)
            model = make_model(ARM, prior_spec=spec).to("cpu")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model_state_dict"])
            model.set_stats(mo, so)
            model.eval()

            idx = np.arange(len(val_ds))
            if len(idx) > MAXW:
                idx = np.sort(np.random.RandomState(0).choice(idx, MAXW, replace=False))
            ep, i0 = val_ds.ep_idx[idx], val_ds.win_start[idx]
            ff_mask = ((val_ds.contact[ep, i0 + L - 1].max(axis=1) < 0.5)
                       & (val_ds.contact[ep, i0 + L].max(axis=1) < 0.5))
            sub = idx[ff_mask]
            batch = val_ds.gather(sub)
            with torch.no_grad():
                h, _ = model.encode_sequence(batch["x"].to("cpu"))
                delta = model.predict_state(h, batch["a0"].to("cpu")).float().numpy()
            target = batch["d_norm"].numpy()
            head_rms = float(np.sqrt((delta[:, POS] ** 2).mean()))
            target_rms = float(np.sqrt((target[:, POS] ** 2).mean()))
            ratio = head_rms / target_rms if target_rms else float("inf")
            per[str(seed)] = dict(head_rms=head_rms, target_rms=target_rms,
                                  ratio=ratio, n_free_flight=int(len(sub)),
                                  n_val_episodes=int(len(np.unique(val_ds.ep_idx))),
                                  best_epoch=ck.get("epoch"))
            print(f"{label:20s} {seed:>4d} {target_rms:14.7g} {head_rms:14.7g} "
                  f"{ratio:10.4f} {len(sub):8d}")
            del model, train_ds, val_ds
        ratios = np.asarray([per[str(s)]["ratio"] for s in SEEDS], dtype=float)
        targets = np.asarray([per[str(s)]["target_rms"] for s in SEEDS], dtype=float)
        heads = np.asarray([per[str(s)]["head_rms"] for s in SEEDS], dtype=float)
        per_tier[spec] = dict(
            label=label, n=len(ratios), per_seed=per,
            ratio_mean=float(ratios.mean()), ratio_sd=float(ratios.std(ddof=1)),
            ratio_cv_pct=float(100 * ratios.std(ddof=1) / ratios.mean()),
            head_mean=float(heads.mean()), target_mean=float(targets.mean()),
            descriptive_only=True)

    coarse = per_tier["euler005"]
    exact = per_tier["rk4001x5"]
    pooled_scale = float(np.sqrt(coarse["ratio_sd"]**2 + exact["ratio_sd"]**2))
    z_descriptive = float((coarse["ratio_mean"] - exact["ratio_mean"]) / pooled_scale)
    out = dict(
        protocol="per-seed own held-out split via e0_common.make_datasets(seed); same e0_common E0 evaluation convention",
        seeds=list(SEEDS), tiers=per_tier,
        split_warning=("Descriptive extension only: splits across seeds overlap in episodes; "
                       "the ten values are not independent replications."),
        shared_split_legacy=dict(
            source="analysis_output/m4_floor_partial.json",
            warning=("Legacy common-split ratios reuse seed-42 validation episodes; most overlap "
                     "training data for seeds 43-51 and are not used for the revised inferential wording.")),
        coarse_vs_exact_descriptive_standardized_gap=z_descriptive,
        standardized_gap_is_inferential=False)
    with open(os.path.join(AO, "ladder_floor_r1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n=== descriptive R1 summary (not an independent-seed inference) ===")
    for spec, _ in TIERS:
        r = per_tier[spec]
        print(f"{spec:12s} ratio={r['ratio_mean']:.4f} ± {r['ratio_sd']:.4f} "
              f"(CV {r['ratio_cv_pct']:.1f}%; n={r['n']})")
    print(f"coarse-vs-exact pooled-SD descriptive gap = {z_descriptive:.2f} (not inferential)")
    print("[save] analysis_output/ladder_floor_r1.json")


if __name__ == "__main__":
    main()
