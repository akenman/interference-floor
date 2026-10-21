# -*- coding: utf-8 -*-
"""M4 检查 (a) 的**增量**报告：现有 checkpoint 上的先验阶梯 floor 比值。

与 build_m4_multiseed.py 的区别：本脚本对**缺失的档/种子**容错，
所以可以在扩种子训练进行中随时跑，看已到位的部分。

用法：python -X utf8 paper/tables/m4_floor_partial.py
输出：analysis_output/m4_floor_partial.json
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
AO = os.path.join(ROOT, "analysis_output")

from e0_common import E0Data                                      # noqa: E402
from layout_h import OBJ_DIMS, feat_weights                       # noqa: E402
from model_e0 import make_model                                   # noqa: E402

POS = [1, 2, 12, 13]
W_POS = feat_weights(OBJ_DIMS)[POS]
L, K = 8, 5
SEEDS = (42, 43, 44, 45, 46, 47, 48)
MAXW = 30000
TIERS = [
    ("ours_base", "euler005", "粗档"),
    ("ours_base@euler001x5", "euler001x5", "euler@.01x5"),
    ("ours_base@rk4005", "rk4005", "rk4@.05"),
    ("ours_base@rk4001x5", "rk4001x5", "精确档"),
]


def main():
    d = np.load(os.path.join(ROOT, "trajectories_hinge.npz"))
    st, ct, ac, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = st.shape[0]
    segs = np.arange(N)
    np.random.RandomState(42).shuffle(segs)
    val, tr = segs[:300], segs[300:]
    obj_flat = st[tr].reshape(-1, OBJ_DIMS).astype(np.float64)
    mo, so = obj_flat.mean(0), obj_flat.std(0)

    out = {}
    print(f"{'tier':14s} " + " ".join(f"{s:>8d}" for s in SEEDS) +
          "   mean±sd   n")
    for arm_key, spec, label in TIERS:
        arm = arm_key.split("@")[0]
        per = {}
        for seed in SEEDS:
            ckpt = os.path.join(ROOT, "checkpoints-E0", f"{arm_key}_s{seed}", "best.pt")
            if not os.path.exists(ckpt):
                continue
            model = make_model(arm, prior_spec=spec).to("cpu")
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            try:
                model.load_state_dict(ck["model_state_dict"])
            except Exception as e:
                print(f"  [skip] {arm_key} s{seed}: {e}")
                continue
            model.eval()
            model.set_stats(mo, so)
            ds = E0Data(st, ct, ac, yao, val, L, K, mo, so, prior_spec=spec)
            idx = np.arange(len(ds))
            if len(idx) > MAXW:
                idx = np.sort(np.random.RandomState(0).choice(
                    idx, size=MAXW, replace=False))
            ep, i0 = ds.ep_idx[idx], ds.win_start[idx]
            ff = ((ds.contact[ep, i0 + L - 1].max(axis=1) < 0.5)
                  & (ds.contact[ep, i0 + L].max(axis=1) < 0.5))
            sub = idx[ff]
            b = ds.gather(sub)
            with torch.no_grad():
                h, _ = model.encode_sequence(b["x"].to("cpu"))
                delta = model.predict_state(h, b["a0"].to("cpu")).float().numpy()
            tgt = b["d_norm"].numpy()
            hn = float(np.sqrt((delta[:, POS] ** 2).mean()))
            tn = float(np.sqrt((tgt[:, POS] ** 2).mean()))
            per[seed] = dict(head=hn, target=tn, ratio=hn / tn if tn else float("inf"),
                             n_ff=int(len(sub)))
        vals = [per[s]["ratio"] for s in SEEDS if s in per]
        if not vals:
            continue
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[spec] = dict(label=label, per_seed={str(k): v for k, v in per.items()},
                         mean=mean, sd=sd, n=len(vals))
        row = " ".join(f"{per[s]['ratio']:8.3f}" if s in per else f"{'—':>8s}"
                       for s in SEEDS)
        print(f"{label:14s} {row}   {mean:.3f}±{sd:.3f}  {len(vals)}")

    # 精确档 vs 粗档的分离
    if "rk4001x5" in out and "euler005" in out:
        a, b_ = out["rk4001x5"], out["euler005"]
        if a["n"] > 1 and b_["n"] > 1:
            pooled = np.sqrt(a["sd"] ** 2 + b_["sd"] ** 2)
            z = (b_["mean"] - a["mean"]) / pooled if pooled else float("inf")
            out["separation_z"] = float(z)
            print(f"\n精确档 {a['mean']:.3f} (n={a['n']}) vs 粗档 {b_['mean']:.3f} "
                  f"(n={b_['n']}):  z = {z:+.2f}")
            print(f"判据 z>3 -> {'方向稳健' if z > 3 else '尚不稳健/样本不足'}")
    else:
        print("\n精确档或粗档数据不足，暂不能算 z")

    with open(os.path.join(AO, "m4_floor_partial.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/m4_floor_partial.json")


if __name__ == "__main__":
    main()
