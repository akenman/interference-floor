# -*- coding: utf-8 -*-
"""M1 复现地板（球世界，k=3 闭环）：重训轴方差测量（待训练完成后运行）。

背景（审稿 M1）：§8 的 scope 句宣称 "every closed-loop claim in this paper is a
29x or larger separation, or a 34% reversal"，但 §5.9 解码规则的接触损伤
x1.12--1.21 同样是闭环读数（球世界 k=3），其重训复现地板此前从未测量。本脚本
在**按原配方重训**的 baseline checkpoint 上逐项重算 S1f 协议读数（RS(0)、
20000 窗、A/B 对半、τ 网格 0.50--0.95、误关 ≤2% 选 τ*），与 S1f_report.json
的原始读数逐项比较，给出重训轴相对差，供 §8 决定 scope 句的收窄或扩写。

重训配方（与原 checkpoint ck["args"] 逐项一致；train_v3.py 当时尚无
w_ff/gate 等后续参数，现默认值即中性设置）：

  python -X utf8 train_v3.py --epochs 60 --batch_size 1024 --lr 3e-4 \\
    --weight_decay 1e-5 --hidden_dim 256 --seq_len 8 --future_steps 3 \\
    --num_prototypes 32 --num_coarse 8 --seed 42 --w_state 1.0 --w_yao 0.1 \\
    --w_proto 0.05 --w_commit 0.01 --w_usage 0.02 --w_roll 0.3 --w_cal 0.2 \\
    --noise_aug 0.0 --ss_max 0.5 --windows_per_epoch 150000 --val_windows 20000 \\
    --checkpoint_dir checkpoints-M1repro/euler001x5_s42 --prior_spec euler001x5

对照原读数（S1f_report.json results.baseline）：seed 42 damage x1.2103、
seed 43 x1.1341、seed 44 x1.1221（τ* = 0.75，误关 ~2.0%）。

用法：
  python -X utf8 tmp_m1_reproduction.py \\
      --ckpt checkpoints-M1repro/euler001x5_s42/best.pt --seed 42
输出：analysis_output/m1_reproduction.json
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import tmp_s1f_compare as C  # noqa: E402


def run_one(ckpt, seed, device, val_windows=20000):
    """在单个 checkpoint 上复算 S1f 协议（与 C.main 的单臂单 seed 分支逐行等价）。"""
    states, obstacles, contact, actions = C.load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    model, ck = C.load_model(ckpt)
    a, ns = ck["args"], ck["norm_stats"]
    _, val_eps = C.split_episodes(N, seed)
    ds = C.V3Data(states, obstacles, contact, actions, val_eps,
                  a["seq_len"], a["future_steps"],
                  ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"],
                  prior_spec=ck["config"]["prior_spec"])
    idx_all = np.arange(len(ds))
    if val_windows and len(idx_all) > val_windows:
        idx_all = np.sort(np.random.RandomState(0).choice(
            idx_all, size=val_windows, replace=False))
    half = len(idx_all) // 2
    idx_A, idx_B = idx_all[:half], idx_all[half:]
    model = model.to(device)

    base = C.eval_model(model, ds, device, idx_all, a["future_steps"])
    row = dict(mixed_roll_k3=base["roll_mixed"]["k=3"], mixed_one=base["one_mixed"])
    row["unmasked_B"] = C.eval_tau(model, ds, device, idx_B,
                                   K=a["future_steps"], tau=0.0)
    grid_A, grid_B = {}, {}
    for tau in C.TAU_GRID:
        grid_A[str(tau)] = C.eval_tau(model, ds, device, idx_A,
                                      a["future_steps"], tau)
        grid_B[str(tau)] = C.eval_tau(model, ds, device, idx_B,
                                      a["future_steps"], tau)
    row["grid_A"], row["grid_B"] = grid_A, grid_B
    ok = [t for t in C.TAU_GRID
          if grid_A[str(t)]["false_silence"] <= C.FALSE_SILENCE_CAP]
    row["tau_star"] = max(ok) if ok else C.TAU_GRID[0]
    row["tau_star_rule_ok"] = bool(ok)
    row["verdict_B"] = C.eval_tau(model, ds, device, idx_B,
                                  a["future_steps"], row["tau_star"])
    u, v = row["unmasked_B"], row["verdict_B"]
    row["damage"] = v["contact_rms"] / max(u["contact_rms"], 1e-9)
    row["R1f"] = bool(v["ff_rms"] <= C.R1F_THRESHOLD)
    row["R2f"] = bool(v["contact_rms"] <= u["contact_rms"] * C.R2F_MARGIN)
    row["best_epoch"] = ck.get("epoch")
    return row


def get(row, path):
    v = row
    for k in path:
        v = v[k]
    return float(v)


DELTA_KEYS = [
    ("mixed_one", ("mixed_one",)),
    ("mixed_roll_k3", ("mixed_roll_k3",)),
    ("unmasked_B.ff_rms", ("unmasked_B", "ff_rms")),
    ("unmasked_B.contact_rms", ("unmasked_B", "contact_rms")),
    ("verdict_B.contact_rms", ("verdict_B", "contact_rms")),
    ("verdict_B.false_silence", ("verdict_B", "false_silence")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints-M1repro/euler001x5_s42/best.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="m1_reproduction.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with io.open(os.path.join(ROOT, "analysis_output", "S1f_report.json"),
                 encoding="utf-8") as f:
        rep = json.load(f)
    old = rep["results"]["baseline"][str(args.seed)]
    old_damage = old["verdict_B"]["contact_rms"] / old["unmasked_B"]["contact_rms"]

    print("载入 trajectories_v3.npz ...", flush=True)
    new = run_one(os.path.join(ROOT, args.ckpt), args.seed, device)

    deltas = {}
    for name, path in DELTA_KEYS:
        o, n = get(old, path), get(new, path)
        deltas[name] = dict(old=o, new=n,
                            rel_pct=(n - o) / o * 100 if abs(o) > 1e-12 else None)
    deltas["damage"] = dict(old=old_damage, new=new["damage"],
                            rel_pct=(new["damage"] - old_damage) / old_damage * 100)
    deltas["verdict_B.ff_rms(abs)"] = dict(
        old=get(old, ("verdict_B", "ff_rms")), new=get(new, ("verdict_B", "ff_rms")),
        rel_pct=None)

    old_dmg = {s: rep["results"]["baseline"][s]["verdict_B"]["contact_rms"]
               / rep["results"]["baseline"][s]["unmasked_B"]["contact_rms"]
               for s in ("42", "43", "44")}
    out = dict(
        note=("Ball-world retraining-axis reproduction floor for the S1f decode-rule "
              "readings (reviewer M1). Same recipe as the original baseline "
              "checkpoints (args copied from the checkpoint); identical eval protocol "
              "(tmp_s1f_compare machinery: RS(0) 20000 windows, A/B halves, "
              "tau grid, 2% false-silence cap)."),
        seed=args.seed, ckpt=args.ckpt,
        protocol=dict(val_windows=20000, tau_grid=C.TAU_GRID,
                      false_silence_cap=C.FALSE_SILENCE_CAP,
                      r2f_margin=C.R2F_MARGIN, r1f_threshold=C.R1F_THRESHOLD),
        old=dict(tau_star=old["tau_star"], mixed_one=old["mixed_one"],
                 mixed_roll_k3=old["mixed_roll_k3"],
                 unmasked_B=old["unmasked_B"], verdict_B=old["verdict_B"],
                 damage=old_damage,
                 source="analysis_output/S1f_report.json results.baseline.%d" % args.seed),
        new=new,
        deltas_pct=deltas,
        old_three_seed_damage=old_dmg,
    )
    p = os.path.join(ROOT, "analysis_output", args.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[save] analysis_output/{args.out}")
    print(f"  tau*: {old['tau_star']:.2f} -> {new['tau_star']:.2f}")
    print(f"  damage: x{old_damage:.4f} -> x{new['damage']:.4f} "
          f"({deltas['damage']['rel_pct']:+.1f}%)")
    for name, _ in DELTA_KEYS:
        d = deltas[name]
        if d["rel_pct"] is not None:
            print(f"  {name:26s} {d['old']:.5g} -> {d['new']:.5g} "
                  f"({d['rel_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
