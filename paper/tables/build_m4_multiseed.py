# -*- coding: utf-8 -*-
"""M4 汇总：两处单种子结论的多 seed 检验。

输入
  checkpoints-E0/*_s{42..48}/best.pt          先验阶梯四档（73 个 checkpoint 中的相关部分）
  analysis_output/e5_multiseed.json           E5 消融的 seeds 45-48

输出
  analysis_output/m4_multiseed.json
  （并打印判读：哪一处结论在多 seed 下站得住，哪一处站不住）

用法：python -X utf8 paper/tables/build_m4_multiseed.py
"""
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
AO = os.path.join(ROOT, "analysis_output")

import physics_prior_spec as P                                    # noqa: E402
from e0_common import E0Data                                      # noqa: E402
from layout_h import OBJ_DIMS, feat_weights                       # noqa: E402
from model_e0 import make_model, ARM_SPECS                        # noqa: E402

POS = [1, 2, 12, 13]
W_POS = feat_weights(OBJ_DIMS)[POS]
L, K = 8, 5
SEEDS = (42, 43, 44, 45, 46, 47, 48)
MAXW = 30000
TIERS = {
    "ours_base@euler001x5": "euler001x5",
    "ours_base@rk4005": "rk4005",
    "ours_base@rk4001x5": "rk4001x5",
    "ours_base": "euler005",
}


def J(name):
    with open(os.path.join(AO, name), encoding="utf-8") as f:
        return json.load(f)


def floor_ratios():
    """先验阶梯：每档每 seed 的 头输出/目标 比值（自由飞行窗口）。"""
    d = np.load(os.path.join(ROOT, "trajectories_hinge.npz"))
    st, ct, ac, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = st.shape[0]
    segs = np.arange(N)
    np.random.RandomState(42).shuffle(segs)
    val, tr = segs[:300], segs[300:]
    obj_flat = st[tr].reshape(-1, OBJ_DIMS).astype(np.float64)
    mo, so = obj_flat.mean(0), obj_flat.std(0)
    res = {}
    for arm_key, spec in TIERS.items():
        arm = arm_key.split("@")[0]
        res[spec] = {}
        for seed in SEEDS:
            ckpt = os.path.join(ROOT, "checkpoints-E0", f"{arm_key}_s{seed}", "best.pt")
            if not os.path.exists(ckpt):
                continue
            model = make_model(arm, prior_spec=spec).to("cpu")
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            try:
                model.load_state_dict(ck["model_state_dict"])
            except Exception:
                continue
            model.eval()
            model.set_stats(mo, so)
            ds = E0Data(st, ct, ac, yao, val, L, K, mo, so, prior_spec=spec)
            idx = np.arange(len(ds))
            if len(idx) > MAXW:
                idx = np.sort(np.random.RandomState(0).choice(idx, size=MAXW,
                                                              replace=False))
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
            res[spec][str(seed)] = dict(head=hn, target=tn,
                                        ratio=hn / tn if tn > 0 else float("inf"))
    return res


def main():
    out = {}
    print("=== 检验 1：地板比值（头/目标）在先验阶梯上的多 seed 复现 ===")
    fr = floor_ratios()
    out["floor_ratios"] = fr
    print(f"{'tier':12s} " + " ".join(f"{s:>8d}" for s in SEEDS) + "   mean±sd")
    summ = {}
    for spec, per in fr.items():
        vals = [per[str(s)]["ratio"] for s in SEEDS if str(s) in per]
        tgts = [per[str(s)]["target"] for s in SEEDS if str(s) in per]
        summ[spec] = dict(n=len(vals), mean=float(np.mean(vals)),
                          sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                          target_mean=float(np.mean(tgts)))
        print(f"{spec:12s} " + " ".join(f"{v:8.3f}" for v in vals) +
              f"   {summ[spec]['mean']:.3f}±{summ[spec]['sd']:.3f}")
    out["floor_summary"] = summ
    ex = summ.get("rk4001x5", {})
    co = summ.get("euler005", {})
    print(f"\n  精确档 mean ratio = {ex.get('mean', float('nan')):.3f} "
          f"(sd {ex.get('sd', float('nan')):.3f}, n={ex.get('n')})")
    print(f"  粗档   mean ratio = {co.get('mean', float('nan')):.3f} "
          f"(sd {co.get('sd', float('nan')):.3f}, n={co.get('n')})")
    if ex and co:
        sep = (co["mean"] - ex["mean"]) / max(np.sqrt(co["sd"] ** 2 + ex["sd"] ** 2), 1e-9)
        out["floor_separation_z"] = float(sep)
        print(f"  两档分离 z = {sep:+.2f}  -> "
              f"{'方向稳健' if sep > 3 else '方向不稳健'}")

    # ---------------- 检验 2：消融的多 seed ----------------
    p = os.path.join(AO, "e5_multiseed.json")
    if os.path.exists(p):
        ms = J("e5_multiseed.json")
        out["ablation_multiseed"] = ms
        print("\n=== 检验 2：frozen_match vs full（多 seed）===")
        for cond in ("g1125", "drag"):
            print(f"  [{cond}]")
            rows = []
            for seed, rec in sorted(ms.items(), key=lambda kv: int(kv[0])):
                if cond not in rec:
                    continue
                v = rec[cond]["variants"]
                pri = rec[cond]["prior_one"]
                rows.append((seed, pri, v["full"]["roll"], v["frozen_match"]["roll"]))
                print(f"    s{seed}: prior={pri:.4f} full_roll={v['full']['roll']:.4f} "
                      f"frozen_roll={v['frozen_match']['roll']:.4f}  "
                      f"Δ={100*(1-v['frozen_match']['roll']/v['full']['roll']):+.1f}%")
            if len(rows) >= 2:
                f_ = np.array([r[2] for r in rows])
                g_ = np.array([r[3] for r in rows])
                d_ = 1 - g_ / f_
                out.setdefault("ablation_summary", {})[cond] = dict(
                    n=len(rows), full_mean=float(f_.mean()),
                    frozen_mean=float(g_.mean()),
                    delta_mean=float(d_.mean()),
                    delta_sd=float(d_.std(ddof=1)),
                    direction_consistent=int(sum(d_ > 0)))
                s = out["ablation_summary"][cond]
                print(f"    -> full {s['full_mean']:.4f} vs frozen "
                      f"{s['frozen_mean']:.4f}; frozen 更好 {s['direction_consistent']}"
                      f"/{s['n']} seed, Δmean={100*s['delta_mean']:+.1f}% "
                      f"± {100*s['delta_sd']:.1f}%")
    else:
        print("\n[e5_multiseed.json 尚未生成，跳过检验 2]")

    with open(os.path.join(AO, "m4_multiseed.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/m4_multiseed.json")


if __name__ == "__main__":
    main()
