# -*- coding: utf-8 -*-
"""SLOT-1d 冒烟探针：核对落盘 args 全字段 + 1-epoch 门行为（H1 方向早验）。

用法：python tmp_s1d_smoke_probe.py [checkpoint] [n_windows]
"""
import json
import sys

import numpy as np
import torch

from tmp_s1b_compare import load_model
from tmp_s1d_compare import gate_probe
from train_v3 import V3Data, load_dataset, split_episodes

ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints-S1d/_smoke_g/best.pt"
n_win = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

ck = torch.load(ckpt, map_location="cpu", weights_only=False)
print("ARGS:", json.dumps(ck["args"], ensure_ascii=False, sort_keys=True))
print("CONFIG:", json.dumps(ck["config"], ensure_ascii=False, sort_keys=True))

states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
N = states.shape[0]
model, ck = load_model(ckpt)
a, ns = ck["args"], ck["norm_stats"]
_, val_eps = split_episodes(N, a.get("seed", 42))
ds = V3Data(states, obstacles, contact, actions, val_eps, a["seq_len"], a["future_steps"],
            ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"],
            prior_spec=ck["config"]["prior_spec"])
rng = np.random.RandomState(0)
idx_all = np.arange(len(ds))
if n_win and len(idx_all) > n_win:
    idx_all = np.sort(rng.choice(idx_all, size=n_win, replace=False))
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GATE PROBE ({ckpt}, seed={a.get('seed')}, {len(idx_all)} 窗):")
print(json.dumps(gate_probe(model.to(dev), ds, dev, idx_all),
                 ensure_ascii=False, indent=1))
