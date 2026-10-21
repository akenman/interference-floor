# -*- coding: utf-8 -*-
"""干扰地板：第四轮——先验档位依赖性，3 seed（评审动作 5）。

第三轮（tmp_crosstalk_law3.py，seed 42）给出一个与论文表述相反的结果：
在同一批 clean-FF 窗口上，头的输出随先验档位单调上升：
    euler001x5 0.01967  parabola005 0.02026  euler005 0.02420  rk4005 0.02699
即粗档先验上噪声更大，而不是"每一档都一样"。

本脚本把该比较扩到 3 seed，并给出：
  * 每一档的 per-seed head RMS（同一窗口集合）与组内离散度
  * 组间差 / 组内 sd 的 z 值（判断该差异是否超出同配置重跑的噪声）
用法：python -X utf8 tmp_crosstalk_law4.py
输出：analysis_output/crosstalk_law4.json
"""
import json
import os
import statistics as st
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tmp_s1b_compare import load_model, transfer_clean_mask   # noqa: E402
from train_v3 import load_dataset, split_episodes, norm_stats, V3Data   # noqa: E402

POS = [1, 2, 12, 13, 23, 24]
MAX_BATCH = 2048
VAL_WINDOWS = 20000
SEEDS = (42, 43, 44)
PRIORS = ("euler001x5", "parabola005", "rk4005", "euler005")


def head_rms(model, ds, idx, device):
    sq_h, sq_t = [], []
    for j in range(0, len(idx), MAX_BATCH):
        b = ds.gather(idx[j:j + MAX_BATCH])
        xb, ab = b["x"].to(device), b["a0"].to(device)
        h, _ = model.encode_sequence(xb)
        delta = model.predict_state(h, ab)
        dn = b["d_norm"].to(device)
        sq_h.append((delta[:, POS] ** 2).mean().item())
        sq_t.append((dn[:, POS] ** 2).mean().item())
    return float(np.sqrt(np.mean(sq_h))), float(np.sqrt(np.mean(sq_t)))


def main():
    device = torch.device("cpu")
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    out = {}
    for seed in SEEDS:
        tr, val_eps = split_episodes(N, seed)
        mo, so, moo, soo = norm_stats(states, obstacles, tr)
        ds0 = V3Data(states, obstacles, contact, actions, val_eps, 8, 3,
                     mo, so, moo, soo, prior_spec="euler001x5")
        rng = np.random.RandomState(0)
        idx_all = np.arange(len(ds0))
        idx_all = np.sort(rng.choice(idx_all, size=VAL_WINDOWS, replace=False))
        clean_idx = idx_all[transfer_clean_mask(ds0, idx_all, 1)]
        del ds0
        row = {}
        for prior in PRIORS:
            path = f"checkpoints-S1b/{prior}_s{seed}/best.pt"
            if not os.path.exists(path):
                print(f"[skip] {prior} s{seed}")
                continue
            model, ck = load_model(path)
            ds = V3Data(states, obstacles, contact, actions, val_eps, 8, 3,
                        mo, so, moo, soo, prior_spec=ck["config"]["prior_spec"])
            hr, tr_ = head_rms(model, ds, clean_idx, device)
            row[prior] = dict(head_rms=hr, target_rms=tr_, n=int(len(clean_idx)))
            print(f"  s{seed} {prior:12s} head={hr:.5f} target={tr_:.7f} "
                  f"ratio={hr/max(tr_, 1e-12):.1f}")
            del ds, model
        out[str(seed)] = row

    with open("analysis_output/crosstalk_law4.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/crosstalk_law4.json")

    print("\n" + "=" * 78)
    print("同一 clean-FF 窗口集合上的头输出（per seed）")
    print(f"{'prior':14s} {'s42':>9s} {'s43':>9s} {'s44':>9s} {'mean':>9s} {'sd':>8s}")
    summary = {}
    for prior in PRIORS:
        vals = [out[str(s)][prior]["head_rms"] for s in SEEDS if prior in out[str(s)]]
        if not vals:
            continue
        summary[prior] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
        print(f"{prior:14s} " + " ".join(f"{v:9.5f}" for v in vals) +
              f" {np.mean(vals):9.5f} {np.std(vals, ddof=1):8.5f}")
    print("\n组间 vs 组内：")
    if "euler001x5" in summary and "rk4005" in summary:
        m1, s1 = summary["euler001x5"]
        m4, s4 = summary["rk4005"]
        pooled = float(np.sqrt((s1 ** 2 + s4 ** 2) / 2))
        print(f"  rk4005 - euler001x5 = {m4 - m1:+.5f} "
              f"({100*(m4-m1)/m1:+.1f}%), pooled sd={pooled:.5f}, "
              f"z={(m4-m1)/pooled:+.2f}")
    if "euler001x5" in summary and "euler005" in summary:
        m1, s1 = summary["euler001x5"]
        m3, s3 = summary["euler005"]
        pooled = float(np.sqrt((s1 ** 2 + s3 ** 2) / 2))
        print(f"  euler005 - euler001x5 = {m3 - m1:+.5f} "
              f"({100*(m3-m1)/m1:+.1f}%), pooled sd={pooled:.5f}, "
              f"z={(m3-m1)/pooled:+.2f}")
    m_all = [summary[p][0] for p in summary]
    print(f"  跨四档极差 = {max(m_all)/min(m_all):.2f}x")
    print(f"  同档跨 seed 的 CV 中位 = "
          f"{st.median([100*summary[p][1]/summary[p][0] for p in summary]):.1f}%")


if __name__ == "__main__":
    main()
