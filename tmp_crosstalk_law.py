# -*- coding: utf-8 -*-
"""干扰地板的定量预测（评审动作 5）。

问题
----
论文 §6.1 测量到：残差头在"先验已经精确"的自由飞行窗口上输出 0.0197（归一化 RMS）
的噪声，且该噪声与先验档位无关。§8 自称 "we have no theory that predicts its
magnitude from the regime ratio"。

本脚本给出并检验两个定量说法。

P1（参数化预测）
    残差头是共享 trunk 隐状态 h 的函数，且主要用接触窗口的损失拟合。
    自由飞行隐状态落在接触隐状态子空间的"外部"，接触拟合的函数在那里产生
    系统性偏离：

        noise_ff ≈ Lip_contact × rep_mismatch × target_contact_rms
        Lip_contact   = 头在接触点上 ‖∂head/∂h‖_F 的均值
        rep_mismatch  = 自由飞行隐状态相对接触中心、在"与头部输出最相关的
                        接触方向"上的相对偏移 = RMS(P_k(h_ff − h̄_c)) / std(h_ff)
        target_contact_rms = 接触窗口残差目标的 RMS（同一归一化空间）

P2（非参数预测，不依赖 Lipschitz）
    只用接触窗口的隐状态拟合一个线性代理 head（岭回归），拿到自由飞行窗口上
    求值。它在接触上拟合得很好，在自由飞行上的输出幅度就是对干扰地板的预测。

检验：跨 6 个训练臂 × 3 seed 比较预测与实测（log-log 相关 + 比值区间）。

用法：python -X utf8 tmp_crosstalk_law.py
输出：analysis_output/crosstalk_law.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tmp_s1b_compare import load_model          # noqa: E402
from train_v3 import (load_dataset, split_episodes, norm_stats, V3Data,   # noqa: E402
                      POS_DIMS_OBJ)

SEEDS = (42, 43, 44)
ARMS = {
    "euler001x5": "checkpoints-S1b/euler001x5_s{s}/best.pt",
    "wff10": "checkpoints-S1c/wff10_s{s}/best.pt",
    "wff0": "checkpoints-S1g/wff0_s{s}/best.pt",
    "sgate": "checkpoints-S1d/sgate_s{s}/best.pt",
    "W1": "checkpoints-S1e/W1_s{s}/best.pt",
    "S": "checkpoints-S1e/S_s{s}/best.pt",
}
N_EP = 60             # 验证片段抽样数
N_WIN_FF = 3000       # 自由飞行窗口抽样
N_WIN_CT = 3000       # 接触窗口抽样
MAX_BATCH = 512
K_PCA = 40            # 保留的接触相关方向数
POS = POS_DIMS_OBJ


def head_jacobian_norm(model, h, a):
    """‖∂head/∂h‖_F 的逐样本值。"""
    hh = h.detach().clone().requires_grad_(True)
    out = model.predict_state(hh, a)
    total = torch.zeros(out.shape[0], device=h.device)
    for j in range(out.shape[1]):
        grad = torch.autograd.grad(out[:, j].sum(), hh,
                                   retain_graph=(j < out.shape[1] - 1))[0]
        total += (grad ** 2).sum(dim=1)
    return total.sqrt().detach().cpu().numpy()


def collect(model, ds, idx, device, want_lip=False):
    """收集一个窗口集合的 (target 位置 RMS, head 位置输出 RMS, H, D, lip)。"""
    sq_t, sq_d, H, D, LIP = [], [], [], [], []
    for j in range(0, len(idx), MAX_BATCH):
        sub = idx[j:j + MAX_BATCH]
        b = ds.gather(sub)
        xb = b["x"].to(device)
        ab = b["a0"].to(device)
        h, _ = model.encode_sequence(xb)
        dn = b["d_norm"].to(device)
        delta = model.predict_state(h, ab)
        sq_t.append((dn[:, POS] ** 2).mean().item())
        sq_d.append((delta[:, POS] ** 2).mean().item())
        H.append(h.detach().cpu().numpy())
        D.append(delta[:, POS].detach().cpu().numpy())
        if want_lip:
            LIP.append(head_jacobian_norm(model, h, ab))
    out = dict(target_rms=float(np.sqrt(np.mean(sq_t))),
               head_rms=float(np.sqrt(np.mean(sq_d))),
               H=np.concatenate(H, axis=0), D=np.concatenate(D, axis=0))
    if want_lip:
        out["lip"] = np.concatenate(LIP, axis=0)
    return out


def measure(model, ds, ff_idx, ct_idx, device):
    ff = collect(model, ds, ff_idx, device, want_lip=False)
    ct = collect(model, ds, ct_idx, device, want_lip=True)
    rec = dict(target_contact_rms=ct["target_rms"],
               target_ff_rms=ff["target_rms"],
               head_ff_rms=ff["head_rms"],
               head_contact_rms=ct["head_rms"],
               lip_contact=float(ct["lip"].mean()),
               lip_contact_p90=float(np.percentile(ct["lip"], 90)))

    Hc, Hf = ct["H"], ff["H"]
    mu_c = Hc.mean(axis=0, keepdims=True)
    Hc_c, Hf_c = Hc - mu_c, Hf - mu_c          # 都以接触中心为原点

    # 与头部输出最相关的接触方向（对 Hc 做 SVD，再按与 D 的相关性排序取前 K）
    U, S, Vt = np.linalg.svd(Hc_c, full_matrices=False)
    V = Vt[:200]
    d_ref = ct["D"][:, 0]
    proj = Hc_c @ V.T
    corr = np.array([abs(np.corrcoef(proj[:, i], d_ref)[0, 1])
                     if np.isfinite(np.corrcoef(proj[:, i], d_ref)[0, 1]) else 0.0
                     for i in range(V.shape[0])])
    Vk = V[np.argsort(-corr)[:K_PCA]]
    proj_ff = Hf_c @ Vk.T
    h_ff_std = float(np.sqrt((Hf_c ** 2).sum(axis=1).mean()))
    rep = float(np.sqrt((proj_ff ** 2).sum(axis=1).mean()) / max(h_ff_std, 1e-9))
    rec["h_ff_std"] = h_ff_std
    rec["rep_mismatch"] = rep

    # P2：接触拟合的线性代理
    Hc1 = np.concatenate([Hc, np.ones((len(Hc), 1), dtype=Hc.dtype)], axis=1)
    Hf1 = np.concatenate([Hf, np.ones((len(Hf), 1), dtype=Hf.dtype)], axis=1)
    Yc = ct["D"]
    lam = 1e-2 * len(Hc)
    A = Hc1.T @ Hc1 + lam * np.eye(Hc1.shape[1])
    W = np.linalg.solve(A, Hc1.T @ Yc)
    rec["ridge_fit_contact_rms"] = float(np.sqrt(((Hc1 @ W - Yc) ** 2).mean()))
    rec["ridge_pred_ff_rms"] = float(np.sqrt(((Hf1 @ W) ** 2).mean()))
    rec["head_contact_rms_direct"] = rec["head_contact_rms"]

    rec["predicted_ff_rms"] = rec["lip_contact"] * rep * rec["target_contact_rms"]
    return rec


def main():
    device = torch.device("cpu")
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    out = {}
    for seed in SEEDS:
        train_eps, val_eps = split_episodes(N, seed)
        mean_obj, std_obj, mean_obs, std_obs = norm_stats(states, obstacles, train_eps)
        sub = val_eps[:N_EP]
        for arm, tmpl in ARMS.items():
            path = tmpl.format(s=seed)
            if not os.path.exists(path):
                print(f"  [skip] {arm} s{seed}: missing {path}")
                continue
            model, ck = load_model(path)
            model = model.to(device)
            prior = ck["config"].get("prior_spec")
            ds = V3Data(states, obstacles, contact, actions, sub,
                        8, 3, mean_obj, std_obj, mean_obs, std_obs, prior_spec=prior)
            rng = np.random.RandomState(seed + 7)
            wff = ds.win_ff.reshape(-1) > 0.5
            ia = np.arange(len(ds))
            ff_idx = rng.choice(ia[wff], size=min(N_WIN_FF, int(wff.sum())), replace=False)
            ct_pool = ia[~wff]
            ct_idx = rng.choice(ct_pool, size=min(N_WIN_CT, len(ct_pool)), replace=False)
            rec = measure(model, ds, ff_idx, ct_idx, device)
            rec.update(prior_spec=prior, n_ff=int(len(ff_idx)), n_ct=int(len(ct_idx)))
            out[f"{arm}_s{seed}"] = rec
            print(f"  {arm:10s} s{seed} prior={prior:12s} "
                  f"ct_tgt={rec['target_contact_rms']:.4f} "
                  f"ff_tgt={rec['target_ff_rms']:.5f} ff_head={rec['head_ff_rms']:.4f} "
                  f"lip={rec['lip_contact']:.3f} rep={rec['rep_mismatch']:.3f} "
                  f"P1={rec['predicted_ff_rms']:.4f} P2={rec['ridge_pred_ff_rms']:.4f}")
            del ds, model
    with open(os.path.join("analysis_output", "crosstalk_law.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/crosstalk_law.json")

    print("\n" + "=" * 84)
    print(f"{'arm/seed':16s} {'actual':>8s} {'P1':>8s} {'P1/act':>7s} "
          f"{'P2':>8s} {'P2/act':>7s} {'ct_tgt':>7s}")
    a, p1, p2 = [], [], []
    for k, v in out.items():
        act = v["head_ff_rms"]
        print(f"{k:16s} {act:8.4f} {v['predicted_ff_rms']:8.4f} "
              f"{v['predicted_ff_rms']/act:7.2f} {v['ridge_pred_ff_rms']:8.4f} "
              f"{v['ridge_pred_ff_rms']/act:7.2f} {v['target_contact_rms']:7.3f}")
        a.append(act); p1.append(v["predicted_ff_rms"]); p2.append(v["ridge_pred_ff_rms"])
    for name, p in (("P1 Lip*rep*target", p1), ("P2 ridge(contact)->FF", p2)):
        a_, p_ = np.array(a), np.array(p)
        r = float(np.corrcoef(np.log(a_), np.log(p_))[0, 1])
        ratio = p_ / a_
        print(f"{name:26s}: log-log r={r:+.3f}  median ratio={np.median(ratio):.2f}  "
              f"range=[{ratio.min():.2f}, {ratio.max():.2f}]")


if __name__ == "__main__":
    main()
