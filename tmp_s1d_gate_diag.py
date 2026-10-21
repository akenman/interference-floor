# -*- coding: utf-8 -*-
"""SLOT-1d 机制归因探针：监督门为何买不到构造性静默？

对每个 sgate seed，在与判决同口径的 FF-clean 子集上分解闭环 FF 误差：
  g      = σ(gate_logits(h, a))          门开度
  r      = 无门残差输出（WorldModelV3.predict_state）
  gated  = g · r                         实际残差输出
报告 g 分布、|r| 与 |g·r| 的 RMS（位置维、归一化口径），并对照同子集上的
非门控臂（baseline 复用）头输出 RMS——判定"误差来自门泄漏"还是"来自头本身"。

用法：python tmp_s1d_gate_diag.py [--val-windows 20000]
输出：控制台 + analysis_output/S1d_gate_diag.json
"""
import argparse
import json

import numpy as np
import torch

from layout import OBS_DIMS, OBJ_DIMS
from tmp_s1b_compare import load_model, transfer_clean_mask
from tmp_s1d_compare import SEEDS, gate_probe
from model_v3 import WorldModelV3
from train_v3 import V3Data, load_dataset, split_episodes

POS = [1, 2, 12, 13, 23, 24]


@torch.no_grad()
def decompose(model, ds, device, idx_all, mask, bs=2048):
    """在 mask 选中的窗口上分解 g / r / g·r（RMS 归一化位置维）。"""
    idx = idx_all[mask]
    if len(idx) == 0:
        return None
    gs, rs, gs_rs = [], [], []
    for j in range(0, len(idx), bs):
        sub = idx[j:j + bs]
        b = ds.gather(sub)
        xb = b["x"].to(device)
        ab = b["a0"].to(device)
        gua = b["gua"].to(device)
        h, _ = model.encode_sequence(xb)
        g = torch.sigmoid(model.gate_logits(h, ab).float())
        base = WorldModelV3.predict_state(model, h, ab, gua).float()
        gs.append(g.cpu().numpy())
        rs.append(base.cpu().numpy())
        gs_rs.append((g.unsqueeze(-1) * base).cpu().numpy())
    g = np.concatenate(gs)
    r = np.concatenate(rs)
    gr = np.concatenate(gs_rs)
    rms = lambda a: float(np.sqrt((a[:, POS] ** 2).mean()))
    return dict(n=int(len(g)), g_mean=float(g.mean()), g_median=float(np.median(g)),
                g_p95=float(np.percentile(g, 95)), g_max=float(g.max()),
                frac_closed=float((g < 0.5).mean()),
                frac_g_le_0_05=float((g <= 0.05).mean()),
                rms_r=rms(r), rms_gated=rms(gr),
                rms_r_pos_exact=float(np.sqrt((r[:, POS] ** 2).mean())))


@torch.no_grad()
def head_rms_plain(model, ds, device, idx, bs=2048):
    """非门控臂在同子集上的残差输出 RMS（位置维）。"""
    vals = []
    for j in range(0, len(idx), bs):
        sub = idx[j:j + bs]
        b = ds.gather(sub)
        h, _ = model.encode_sequence(b["x"].to(device))
        d = model.predict_state(h, b["a0"].to(device),
                                b["gua"].to(device)).float().cpu().numpy()
        vals.append(d)
    d = np.concatenate(vals) if vals else np.zeros((0, OBJ_DIMS), np.float32)
    return float(np.sqrt((d[:, POS] ** 2).mean())) if len(d) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-windows", type=int, default=20000)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("载入 trajectories_v3.npz …", flush=True)
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]

    out = {"prereg": "experiments/prereg_S1d_supervised_gate.json", "seeds": {}}
    for s in SEEDS:
        ds_cache = None
        row = {}
        for arm, path in (("sgate", f"checkpoints-S1d/sgate_s{s}/best.pt"),
                          ("baseline", f"checkpoints-S1b/euler001x5_s{s}/best.pt")):
            model, ck = load_model(path)
            a, ns = ck["args"], ck["norm_stats"]
            _, val_eps = split_episodes(N, s)
            ds = V3Data(states, obstacles, contact, actions, val_eps,
                        a["seq_len"], a["future_steps"],
                        ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"],
                        prior_spec=ck["config"]["prior_spec"])
            rng = np.random.RandomState(0)
            idx_all = np.arange(len(ds))
            if args.val_windows and len(idx_all) > args.val_windows:
                idx_all = np.sort(rng.choice(idx_all, size=args.val_windows,
                                             replace=False))
            m1 = transfer_clean_mask(ds, idx_all, 1)      # R1d 子集
            mK = transfer_clean_mask(ds, idx_all, a["future_steps"])   # R3d 子集
            model = model.to(device)
            if arm == "sgate":
                row["one_step_ffclean"] = decompose(model, ds, device, idx_all, m1)
                row["closed_loop_ffclean"] = decompose(model, ds, device, idx_all, mK)
                row["gate_all_windows"] = gate_probe(model, ds, device, idx_all)
            else:
                row["baseline_rms_r_one"] = head_rms_plain(model, ds, device,
                                                           idx_all[m1])
                row["baseline_rms_r_cl"] = head_rms_plain(model, ds, device,
                                                          idx_all[mK])
            ds_cache = ds
            del model
            torch.cuda.empty_cache()
        out["seeds"][s] = row
        print(f"\n[seed {s}]")
        for tag in ("one_step_ffclean", "closed_loop_ffclean"):
            d = row[tag]
            print(f"  {tag}: n={d['n']}  ḡ={d['g_mean']:.4f}  g_中位={d['g_median']:.4f}"
                  f"  g_p95={d['g_p95']:.4f}  g_max={d['g_max']:.4f}"
                  f"  关门率={d['frac_closed']:.3f}  g≤0.05={d['frac_g_le_0_05']:.3f}")
            print(f"      RMS|r|={d['rms_r']:.4f}  →  RMS|g·r|={d['rms_gated']:.4f}")
        print(f"  对照 baseline 头 RMS: one-step 子集={row['baseline_rms_r_one']:.4f}"
              f"  闭环子集={row['baseline_rms_r_cl']:.4f}")

    with open("analysis_output/S1d_gate_diag.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/S1d_gate_diag.json")


if __name__ == "__main__":
    main()
