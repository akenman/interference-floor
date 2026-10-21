# -*- coding: utf-8 -*-
"""Provenance probe: free-flight head Jacobian norm (the paper's 1.18 / 1.15 claim).

Reuses tmp_crosstalk_law.py's collect()/head_jacobian_norm() and the same six
arms / seeds, but requests lip=True on the FREE-FLIGHT window set (the original
script only computed it on contact). Writes analysis_output/ff_jacobian.json.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tmp_crosstalk_law as C  # noqa: E402

device = torch.device("cpu")
states, obstacles, contact, actions = C.load_dataset("trajectories_v3.npz")
N = states.shape[0]

out = {}
for seed in (42, 43, 44):
    train_eps, val_eps = C.split_episodes(N, seed)
    mean_obj, std_obj, mean_obs, std_obs = C.norm_stats(states, obstacles, train_eps)
    sub = val_eps[:C.N_EP]
    for arm, tmpl in C.ARMS.items():
        path = tmpl.format(s=seed)
        if not os.path.exists(path):
            print("  [skip]", arm, seed)
            continue
        model, ck = C.load_model(path)
        model = model.to(device)
        prior = ck["config"].get("prior_spec")
        ds = C.V3Data(states, obstacles, contact, actions, sub, 8, 3,
                      mean_obj, std_obj, mean_obs, std_obs, prior_spec=prior)
        rng = np.random.RandomState(seed + 7)
        wff = ds.win_ff.reshape(-1) > 0.5
        ia = np.arange(len(ds))
        ff_idx = rng.choice(ia[wff], size=min(C.N_WIN_FF, int(wff.sum())),
                            replace=False)
        ct_pool = ia[~wff]
        ct_idx = rng.choice(ct_pool, size=min(C.N_WIN_CT, len(ct_pool)),
                            replace=False)
        ff = C.collect(model, ds, ff_idx, device, want_lip=True)
        ct = C.collect(model, ds, ct_idx, device, want_lip=True)
        out["%s_s%d" % (arm, seed)] = {
            "lip_ff": float(ff["lip"].mean()),
            "lip_ct": float(ct["lip"].mean()),
            "n_ff": int(len(ff_idx)), "n_ct": int(len(ct_idx)),
        }
        print("  %-10s s%d  ff=%.4f  ct=%.4f" % (arm, seed,
              out["%s_s%d" % (arm, seed)]["lip_ff"],
              out["%s_s%d" % (arm, seed)]["lip_ct"]), flush=True)
        del ds, model

with open(os.path.join("analysis_output", "ff_jacobian.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print("\n=== seed-42 free-flight Jacobian norm, six arms ===")
ffs = [out["%s_s42" % a]["lip_ff"] for a in C.ARMS if "%s_s42" % a in out]
cts = [out["%s_s42" % a]["lip_ct"] for a in C.ARMS if "%s_s42" % a in out]
for a in C.ARMS:
    k = "%s_s42" % a
    if k in out:
        print("  %-10s ff=%.4f ct=%.4f ratio=%.2f"
              % (a, out[k]["lip_ff"], out[k]["lip_ct"],
                 out[k]["lip_ff"] / out[k]["lip_ct"]))
print("  mean ff (6 arms) = %.4f" % (sum(ffs) / len(ffs)))
print("  mean ct (6 arms) = %.4f" % (sum(cts) / len(cts)))
print("  exact arm (euler001x5) ff = %.4f" % out["euler001x5_s42"]["lip_ff"])
print("[save] analysis_output/ff_jacobian.json")
