# -*- coding: utf-8 -*-
"""干扰地板：第五轮——静默在表示上是否可实现（评审动作 5 的决定性实验）。

逻辑
----
论文把干扰地板归因于"损失几何"（共享头、regime 配比），但从未证明静默在
表示上是可实现的。这一点可以直接检验：**冻结 trunk**，只用相同的接触损失目标
重新拟合一个 head，扫描自由飞行损失权重 w_ff。

  * 若在很小的 w_ff 下就能把自由飞行头输出压到接近 0（而接触精度基本不变），
    则"特征里已经含有静默所需的全部信息"，地板是损失几何/优化问题 →
    论文的机制归因成立，且 §8 的 "no theory" 可以换成一条可操作的分隔判据。
  * 若即使把 w_ff 调大到 1.0（甚至 10）也压不下去，则地板是**表示**地板，
    论文的机制叙述需要加上这条限制。

协议（全部沿用论文口径）
  * trunk：checkpoints-S1b/euler001x5_s42/best.pt，冻结
  * 训练窗口：完整训练划分（seed 42），上采样到 ≤2e5 条
  * 损失：与 train_v3 的 state 损失同形，逐样本 L2（对 d_norm 的 33 维不取位置子集
    ——位置 RMSE 只用于评估），w_ff 只作为逐样本权重
  * 评估：clean-FF 掩码 73 窗口（头输出 RMS）+ 同一批接触窗口（头输出 RMS）
    与"未重训的头"在**完全相同窗口**上的同一指标对照

用法：python -X utf8 tmp_silence_feasible.py
输出：analysis_output/silence_feasible.json
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
MAX_BATCH = 2048
VAL_WINDOWS = 20000
N_TRAIN = 200_000
EPOCHS = 400
LR = 5e-4
W_FF_GRID = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0)
SEED = 42


def head_from(model):
    """复制原头的结构（Sequential: Linear-GELU-Linear-GELU-Linear）。"""
    h = model.state_head
    in_dim = h[0].in_features
    layers = []
    for m in h:
        if isinstance(m, nn.Linear):
            layers += [nn.Linear(m.in_features, m.out_features)]
            layers.append(nn.GELU())
    layers = layers[:-1]                      # 去掉末尾多余 GELU
    net = nn.Sequential(*layers)
    with torch.no_grad():
        net[-1].weight.normal_(0, 0.01)
        net[-1].bias.zero_()
    return net


def collect_hidden(model, ds, idx, device):
    """返回 (h, a, d_norm, w_ff) 的 numpy，供冻结特征上重训头。"""
    H, A, D, W = [], [], [], []
    for j in range(0, len(idx), MAX_BATCH):
        b = ds.gather(idx[j:j + MAX_BATCH])
        xb, ab = b["x"].to(device), b["a0"].to(device)
        with torch.no_grad():
            h, _ = model.encode_sequence(xb)
        H.append(h.cpu().numpy())
        A.append(ab.cpu().numpy())
        D.append(b["d_norm"].numpy())
        W.append(b["w_ff"].numpy())
    return (np.concatenate(H), np.concatenate(A),
            np.concatenate(D), np.concatenate(W))


def eval_head(net, H, A, POS_):
    dev = next(net.parameters()).device
    with torch.no_grad():
        out = net(torch.from_numpy(np.concatenate([H, A], axis=1)).to(dev)).cpu().numpy()
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    tr, val_eps = split_episodes(N, SEED)
    mo, so, moo, soo = norm_stats(states, obstacles, tr)

    model, ck = load_model(CKPT)
    model = model.to(device)
    prior = ck["config"]["prior_spec"]

    # ---------------- 评估窗口（论文口径） ----------------
    dsv = V3Data(states, obstacles, contact, actions, val_eps, 8, 3,
                 mo, so, moo, soo, prior_spec=prior)
    rng = np.random.RandomState(0)
    iv = np.arange(len(dsv))
    iv = np.sort(rng.choice(iv, size=VAL_WINDOWS, replace=False))
    clean_idx = iv[transfer_clean_mask(dsv, iv, 1)]
    ep, t0 = dsv.ep_idx[iv], dsv.win_start[iv] + dsv.L - 1
    ct_pool = iv[dsv.contact[ep, t0].max(axis=1) > 0.5]
    ct_idx = np.random.RandomState(1).choice(ct_pool, size=len(clean_idx), replace=False)

    Hc, Ac, Dc, _ = collect_hidden(model, dsv, clean_idx, device)
    Hk, Ak, Dk, _ = collect_hidden(model, dsv, ct_idx, device)

    def rms(x):
        return float(np.sqrt((x[:, POS] ** 2).mean()))

    base_ff_out = eval_head(model.state_head, Hc, Ac, POS)
    base_ct_out = eval_head(model.state_head, Hk, Ak, POS)
    base = dict(head_ff_rms=rms(base_ff_out), head_ct_rms=rms(base_ct_out),
                target_ff_rms=rms(Dc), target_ct_rms=rms(Dk),
                n_clean=int(len(clean_idx)), n_contact=int(len(ct_idx)))
    print(f"未重训的头（这些窗口上的基线）: FF head={base['head_ff_rms']:.5f} "
          f"ct head={base['head_ct_rms']:.5f}")

    # ---------------- 训练窗口 ----------------
    dst = V3Data(states, obstacles, contact, actions, tr, 8, 3,
                 mo, so, moo, soo, prior_spec=prior)
    it = np.arange(len(dst))
    rngt = np.random.RandomState(11)
    if len(it) > N_TRAIN:
        it = np.sort(rngt.choice(it, size=N_TRAIN, replace=False))
    Ht, At, Dt, Wt = collect_hidden(model, dst, it, device)
    print(f"训练窗口 {len(it)}  自由飞行占比 {Wt.mean()*100:.2f}%")
    Xt = torch.from_numpy(np.concatenate([Ht, At], axis=1)).to(device)
    Yt = torch.from_numpy(Dt).to(device)
    Wt_t = torch.from_numpy(Wt.astype(np.float32)).to(device)

    results = {}
    for w_ff in W_FF_GRID:
        torch.manual_seed(0)
        net = head_from(model).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        sample_w = 1.0 + (w_ff - 1.0) * Wt_t          # FF 窗口权重 = w_ff
        for ep_i in range(EPOCHS):
            perm = torch.randperm(len(Xt), device=device)
            for j in range(0, len(perm), 8192):
                b = perm[j:j + 8192]
                pred = net(Xt[b])
                per = ((pred - Yt[b]) ** 2).mean(dim=1)
                loss = (per * sample_w[b]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            fit_ct = float(np.sqrt(((net(Xt) - Yt) ** 2).mean().item()))
        ff_out = eval_head(net, Hc, Ac, POS)
        ct_out = eval_head(net, Hk, Ak, POS)
        rec = dict(w_ff=w_ff,
                   head_ff_rms=rms(ff_out), head_ct_rms=rms(ct_out),
                   fit_contact_rms=fit_ct)
        results[f"{w_ff:g}"] = rec
        print(f"  w_ff={w_ff:<6g} FF head={rec['head_ff_rms']:.5f} "
              f"(基线 {base['head_ff_rms']:.5f}, "
              f"×{rec['head_ff_rms']/base['head_ff_rms']:.3f})  "
              f"ct head={rec['head_ct_rms']:.5f} "
              f"(基线 {base['head_ct_rms']:.5f})  trainfit={fit_ct:.5f}")

    out = dict(baseline=base, grid=results, n_train=int(len(it)),
               epochs=EPOCHS, lr=LR, ckpt=CKPT)
    with open("analysis_output/silence_feasible.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/silence_feasible.json")


if __name__ == "__main__":
    main()
