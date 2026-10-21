# -*- coding: utf-8 -*-
"""干扰地板：第六轮——静默的可实现性上限（评审动作 5 的决定性实验，公平版）。

第五轮（tmp_silence_feasible.py）的问题：重训的头欠拟合
（训练集 L2 0.080 vs 原头 0.0209），因此 "FF 只降到 0.0127" 不能归因于表示。

本脚本修正三点：
  1. 更长的训练（cosine 衰减、lr 1e-3、最多 3000 epoch）+ 更大的训练窗口
     （w_ff=1 这一档算到误差 <0.03 才算公平）；
  2. w_ff 网格扩到 0.1/1/3/10/30/100，看**表示侧的静默上限**在哪里、
     以及接触精度在哪个权重下开始被牺牲（论文 §6.1 声称"通过共享头买到的东西
     一定在接触窗口上付账"）；
  3. 同时报告原头在**训练窗口**上的拟合值，作为"该特征上可达的拟合水平"参照。

口径：trunk 冻结（checkpoints-S1b/euler001x5_s42），只重训 state_head；
损失与 train_v3 的 state 项同形（对 d_norm 的 33 维做逐样本 L2），
逐窗口权重 w_ff 只作用于 win_ff 窗口。
评估：clean-FF 掩码 73 窗口 + 同规模接触窗口（与 crosstalk_law3/4 同集合）。
用法：python -X utf8 tmp_silence_ceiling.py
输出：analysis_output/silence_ceiling.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tmp_s1b_compare import load_model, transfer_clean_mask   # noqa: E402
from train_v3 import load_dataset, split_episodes, norm_stats, V3Data   # noqa: E402

CKPT = "checkpoints-S1b/euler001x5_s42/best.pt"
POS = [1, 2, 12, 13, 23, 24]
MAX_BATCH = 4096
VAL_WINDOWS = 20000
N_TRAIN = 400_000
MAX_EPOCHS = 1200
LR = 1e-3
W_FF_GRID = (0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0)
BATCH = 8192
SEED = 42
EVAL_AT = (100, 300, 600, 1200)


def build_head(model):
    """Clone the head's real module sequence: Linear-GELU-Linear-GELU-Linear.

    An earlier version appended an extra GELU at the end (it cut the list on a GELU
    instead of after the last Linear), which changes the head's expressiveness. The
    round-6 numbers were produced with that bug; round 7 (tmp_trunk_isolation.py)
    uses the true architecture. Rebuild faithfully here so both agree.
    """
    net = nn.Sequential(*list(model.state_head))
    with torch.no_grad():
        net.load_state_dict(model.state_head.state_dict())
    return net


def collect(model, ds, idx, device):
    H, A, D, W = [], [], [], []
    for j in range(0, len(idx), MAX_BATCH):
        b = ds.gather(idx[j:j + MAX_BATCH])
        xb, ab = b["x"].to(device), b["a0"].to(device)
        with torch.no_grad():
            h, _ = model.encode_sequence(xb)
        H.append(h.cpu().numpy()); A.append(ab.cpu().numpy())
        D.append(b["d_norm"].numpy()); W.append(b["w_ff"].numpy())
    return (np.concatenate(H), np.concatenate(A), np.concatenate(D), np.concatenate(W))


def infer(net, H, A):
    dev = next(net.parameters()).device
    with torch.no_grad():
        return net(torch.from_numpy(np.concatenate([H, A], axis=1)).to(dev)).cpu().numpy()


def rms_pos(x):
    return float(np.sqrt((x[:, POS] ** 2).mean()))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    tr, val_eps = split_episodes(N, SEED)
    mo, so, moo, soo = norm_stats(states, obstacles, tr)

    model, ck = load_model(CKPT)
    model = model.to(device)
    prior = ck["config"]["prior_spec"]

    dsv = V3Data(states, obstacles, contact, actions, val_eps, 8, 3,
                 mo, so, moo, soo, prior_spec=prior)
    rng = np.random.RandomState(0)
    iv = np.arange(len(dsv))
    iv = np.sort(rng.choice(iv, size=VAL_WINDOWS, replace=False))
    clean_idx = iv[transfer_clean_mask(dsv, iv, 1)]
    ep, t0 = dsv.ep_idx[iv], dsv.win_start[iv] + dsv.L - 1
    ct_idx = np.random.RandomState(1).choice(
        iv[dsv.contact[ep, t0].max(axis=1) > 0.5], size=len(clean_idx), replace=False)
    Hc, Ac, Dc, _ = collect(model, dsv, clean_idx, device)
    Hk, Ak, Dk, _ = collect(model, dsv, ct_idx, device)

    base_ff = rms_pos(infer(model.state_head, Hc, Ac))
    base_ct = rms_pos(infer(model.state_head, Hk, Ak))
    print(f"baseline head: FF={base_ff:.5f}  contact={base_ct:.5f}  "
          f"(targets {rms_pos(Dc):.2e} / {rms_pos(Dk):.4f})")

    dst = V3Data(states, obstacles, contact, actions, tr, 8, 3,
                 mo, so, moo, soo, prior_spec=prior)
    it = np.arange(len(dst))
    if len(it) > N_TRAIN:
        it = np.sort(np.random.RandomState(11).choice(it, size=N_TRAIN, replace=False))
    Ht, At, Dt, Wt = collect(model, dst, it, device)
    Xt = torch.from_numpy(np.concatenate([Ht, At], axis=1)).to(device)
    Yt = torch.from_numpy(Dt).to(device)
    Wt_t = torch.from_numpy(Wt.astype(np.float32)).to(device)
    print(f"train windows {len(it)}  FF share {Wt.mean()*100:.2f}%")
    # 原头在训练窗口上的拟合（可达水平参照）
    sub = torch.randperm(len(Xt), device=device)[:60000]

    def huber_mean(net, X, Y):
        with torch.no_grad():
            return float(torch.nn.functional.huber_loss(
                net(X[sub]), Y[sub], reduction="mean").item())
    base_train_fit = huber_mean(model.state_head, Xt, Yt)
    print(f"原头在训练窗口上的 Huber = {base_train_fit:.5f}")

    out = dict(baseline=dict(head_ff_rms=base_ff, head_ct_rms=base_ct,
                             target_ff_rms=rms_pos(Dc), target_ct_rms=rms_pos(Dk),
                             head_train_fit=base_train_fit,
                             n_clean=int(len(clean_idx)), n_contact=int(len(ct_idx))),
               n_train=int(len(it)), max_epochs=MAX_EPOCHS, lr=LR, grid={})

    for w_ff in W_FF_GRID:
        torch.manual_seed(0)
        net = build_head(model).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        sample_w = 1.0 + (w_ff - 1.0) * Wt_t
        traj = []
        for ep_i in range(1, MAX_EPOCHS + 1):
            perm = torch.randperm(len(Xt), device=device)
            tot = 0.0
            for j in range(0, len(perm), BATCH):
                b = perm[j:j + BATCH]
                hu = torch.nn.functional.huber_loss(
                    net(Xt[b]), Yt[b], reduction="none").mean(dim=1)
                loss = (hu * sample_w[b]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.item()) * len(b)
            sched.step()
            if ep_i in EVAL_AT:
                fit = huber_mean(net, Xt, Yt)
                ff = rms_pos(infer(net, Hc, Ac))
                ct = rms_pos(infer(net, Hk, Ak))
                traj.append(dict(epoch=ep_i, train_fit=fit,
                                 head_ff_rms=ff, head_ct_rms=ct))
                print(f"  w_ff={w_ff:<6g} ep{ep_i:<5d} fit={fit:.5f} "
                      f"FF={ff:.5f} (×{ff/base_ff:.3f})  ct={ct:.5f} "
                      f"(×{ct/base_ct:.3f})", flush=True)
        out["grid"][f"{w_ff:g}"] = traj

    with open("analysis_output/silence_ceiling.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/silence_ceiling.json")
    print("\n=== 末次评估（epoch %d）汇总 ===" % EVAL_AT[-1])
    print(f"{'w_ff':>7s} {'train_fit':>10s} {'FF_rms':>9s} {'FF/base':>8s} "
          f"{'ct_rms':>9s} {'ct/base':>8s}")
    for k, traj in out["grid"].items():
        last = traj[-1]
        print(f"{k:>7s} {last['train_fit']:10.5f} {last['head_ff_rms']:9.5f} "
              f"{last['head_ff_rms']/base_ff:8.3f} {last['head_ct_rms']:9.5f} "
              f"{last['head_ct_rms']/base_ct:8.3f}")


if __name__ == "__main__":
    main()
