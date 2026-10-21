# -*- coding: utf-8 -*-
"""复现论文的头输出口径，并分解自由飞行窗口的异质性（评审动作 5 前置校验）。

论文口径（tmp_s1c_compare.head_rms_on_ff）：
  * 完整 val_eps（= split_episodes(N, seed) 的前 10%），不是抽样片段
  * 窗口先随机抽 20000 条（rng = RandomState(0)，sort 后取）
  * 在其上取 transfer_clean_mask(..., 1)：接触双帧为 0 且速度关系
    |Δv − (I/m + g)τ| < 1
  * 头输出 predict_state 在位置维 [1,2,12,13,23,24] 上的 RMS（归一化空间）

本脚本：复现该值；再把"接触双帧为 0"的窗口（纸面意义上的自由飞行）按
速度一致性分成 clean / non-clean 两桶，看头输出 RMS 与其残差目标各是多少。
用法：python -X utf8 tmp_probe_replicate.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tmp_s1b_compare import load_model, transfer_clean_mask   # noqa: E402
from train_v3 import load_dataset, split_episodes, norm_stats, V3Data   # noqa: E402

POS = [1, 2, 12, 13, 23, 24]
MAX_BATCH = 2048
VAL_WINDOWS = 20000
ARMS = {
    "euler001x5_s42": "checkpoints-S1b/euler001x5_s42/best.pt",
    "wff0_s42": "checkpoints-S1g/wff0_s42/best.pt",
    "wff10_s42": "checkpoints-S1c/wff10_s42/best.pt",
    "parabola005_s42": "checkpoints-S1b/parabola005_s42/best.pt",
}


def head_and_target_on(model, ds, idx, device):
    """返回 (head_rms, target_rms, per-window head 平方均值)。"""
    sq_h, sq_t = [], []
    for j in range(0, len(idx), MAX_BATCH):
        b = ds.gather(idx[j:j + MAX_BATCH])
        xb, ab = b["x"].to(device), b["a0"].to(device)
        h, _ = model.encode_sequence(xb)
        delta = model.predict_state(h, ab)
        dn = b["d_norm"].to(device)
        sq_h.append((delta[:, POS] ** 2).mean(dim=1).detach().cpu().numpy())
        sq_t.append((dn[:, POS] ** 2).mean(dim=1).detach().cpu().numpy())
    sq_h = np.concatenate(sq_h)
    sq_t = np.concatenate(sq_t)
    return float(np.sqrt(sq_h.mean())), float(np.sqrt(sq_t.mean())), sq_h, sq_t


def main():
    device = torch.device("cpu")
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    _, val_eps = split_episodes(N, 42)
    tr, val_eps = split_episodes(N, 42)
    mo, so, moo, soo = norm_stats(states, obstacles, tr)

    report = {}
    for name, path in ARMS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}")
            continue
        model, ck = load_model(path)
        model = model.to(device)
        ds = V3Data(states, obstacles, contact, actions, val_eps, 8, 3,
                    mo, so, moo, soo, prior_spec=ck["config"]["prior_spec"])
        rng = np.random.RandomState(0)
        idx_all = np.arange(len(ds))
        idx_all = np.sort(rng.choice(idx_all, size=VAL_WINDOWS, replace=False))
        # 论文口径
        mask = transfer_clean_mask(ds, idx_all, 1)
        sub = idx_all[mask]
        hr, tr_, _, _ = head_and_target_on(model, ds, sub, device)
        # 未过滤口径：只要求接触双帧为 0
        ep, t0 = ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L - 1
        c0 = ds.contact[ep, t0].max(axis=1) == 0
        c1 = ds.contact[ep, t0 + 1].max(axis=1) == 0
        anyff = c0 & c1
        hr_any, tr_any, sqh_any, sqt_any = head_and_target_on(
            model, ds, idx_all[anyff], device)
        report[name] = dict(
            paper_protocol_head_rms=hr, paper_protocol_target_rms=tr_,
            n_paper=int(mask.sum()),
            anyff_head_rms=hr_any, anyff_target_rms=tr_any, n_anyff=int(anyff.sum()),
            clean_head_rms=hr, clean_target_rms=tr_,
            p50_head=float(np.sqrt(np.median(sqh_any))),
            p90_head=float(np.sqrt(np.percentile(sqh_any, 90))),
            max_head=float(np.sqrt(sqh_any.max())),
            p50_target=float(np.sqrt(np.median(sqt_any))),
            p90_target=float(np.sqrt(np.percentile(sqt_any, 90))),
        )
        print(f"\n{name}  prior={ck['config']['prior_spec']}")
        print(f"  论文口径（clean 掩码, n={int(mask.sum())}）: "
              f"head={hr:.5f}  target={tr_:.6f}")
        print(f"  仅接触双帧=0（n={int(anyff.sum())}）: head={hr_any:.5f}  "
              f"target={tr_any:.6f}")
        print(f"  该集合内 head 分布 p50={report[name]['p50_head']:.5f} "
              f"p90={report[name]['p90_head']:.5f} max={report[name]['max_head']:.4f}")
        print(f"  该集合内 target 分布 p50={report[name]['p50_target']:.6f} "
              f"p90={report[name]['p90_target']:.5f}")
        del model, ds
    with open("analysis_output/probe_replicate.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/probe_replicate.json")
    print("\n论文声称：exact 0.0197 / legacy 0.0203（S1b_probe_cached.json），"
          "S1e baseline 三 seed 均值 0.0186")


if __name__ == "__main__":
    main()
