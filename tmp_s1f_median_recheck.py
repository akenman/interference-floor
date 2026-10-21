# -*- coding: utf-8 -*-
"""W13 复核：在 CPU 上重算 S1f verdict_B 的稳健读数（median / all-silenced）。

背景：adversarial review 第 13 条要求报告 S1f 预注册的稳健读数。读数已在
S1f_report.json 的 results 层存储（每条记录含 ff_median / ff_all_silenced），
本脚本独立地在 CPU 上重算这几组数并逐项对照存储值，作为该读数可复现的证据。

重算范围：baseline 三个 seed（τ*：0.75/0.80/0.75）+ F1 seed 42（τ*=0.85）。
数据、窗口、τ* 全部取自存储报文，不重新选择阈值。

用法：python tmp_s1f_median_recheck.py
输出：analysis_output/S1f_median_recheck.json + 控制台对照表
"""
import json
import os

import numpy as np
import torch

from tmp_s1f_compare import eval_tau
from tmp_s1b_compare import load_model
from train_v3 import V3Data, load_dataset, split_episodes

CELLS = [("baseline", 42), ("baseline", 43), ("baseline", 44), ("F1", 42)]
ARMS = {"baseline": "checkpoints-S1b/euler001x5_s{s}/best.pt",
        "F1": "checkpoints-S1f/F1_s{s}/best.pt"}
KEYS = ["n_ff", "ff_rms", "ff_median", "ff_all_silenced", "false_silence",
        "contact_rms", "protect_free"]


def main():
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    with open("analysis_output/S1f_report.json", encoding="utf-8") as f:
        stored = json.load(f)["results"]
    print("载入 trajectories_v3.npz …", flush=True)
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]

    out = {"note": "CPU recompute of stored verdict_B robustness readings "
                   "(W13); taus taken from S1f_report.json, not reselected",
           "cells": {}}
    for arm, s in CELLS:
        row = stored[arm][str(s)]
        tau_star = row["tau_star"]
        model, ck = load_model(ARMS[arm].format(s=s))
        a, ns = ck["args"], ck["norm_stats"]
        _, val_eps = split_episodes(N, s)
        ds = V3Data(states, obstacles, contact, actions, val_eps,
                    a["seq_len"], a["future_steps"],
                    ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"],
                    prior_spec=ck["config"]["prior_spec"])
        rng = np.random.RandomState(0)
        idx_all = np.arange(len(ds))
        if len(idx_all) > 20000:
            idx_all = np.sort(rng.choice(idx_all, size=20000, replace=False))
        half = len(idx_all) // 2
        idx_B = idx_all[half:]
        re = eval_tau(model, ds, "cpu", idx_B, a["future_steps"], tau_star)
        ref = row["verdict_B"]
        diffs = {k: (float(re[k]) if np.isfinite(re[k]) else None,
                     float(ref[k]) if ref[k] is not None else None)
                 for k in KEYS}
        rel = {k: (abs(v[0] - v[1]) / max(abs(v[1]), 1e-12) if v[0] is not None else None)
               for k, v in diffs.items()}
        ok = all(v is None or v < 1e-3 for v in rel.values())
        out["cells"][f"{arm}_s{s}"] = {"tau_star": tau_star, "n_B": len(idx_B),
                                       "cpu": diffs, "rel_diff": rel, "ok": ok}
        print(f"[{arm} s{s}] tau*={tau_star:.2f} n_B={len(idx_B)} "
              f"median {diffs['ff_median'][0]:.3e} vs {diffs['ff_median'][1]:.3e} "
              f"| allSil {diffs['ff_all_silenced'][0]:.2f} vs {diffs['ff_all_silenced'][1]:.2f} "
              f"| ffRms {diffs['ff_rms'][0]:.3e} vs {diffs['ff_rms'][1]:.3e} "
              f"| maxRel {max(v for v in rel.values() if v is not None):.2e} "
              f"{'[OK]' if ok else '[FAIL]'}", flush=True)
    out["all_ok"] = all(c["ok"] for c in out["cells"].values())
    with open("analysis_output/S1f_median_recheck.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/S1f_median_recheck.json")
    if not out["all_ok"]:
        raise SystemExit("CPU 重算与存储读数不一致（见 rel_diff）")


if __name__ == "__main__":
    main()
