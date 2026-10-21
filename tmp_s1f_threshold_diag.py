# -*- coding: utf-8 -*-
"""SLOT-1f 前置诊断：判决式硬静默的 regime 指示器阈值扫描（标定用）。

背景（SLOT-1e 上限探针）：硬静默掩码在 FF-clean 子集上**中位 0.0001px**（门关对=解析
恒等）而 RMS 0.33–0.68——误差全来自"该关没关"的窗（指示器假阳性 15–53%）。
本脚本扫阈值 τ（silent ⟺ 爻头预测接触概率 max < τ），给出两侧代价曲线：
  A) FF-clean 闭环 k=3：RMS / 中位 / 未关门窗占比      ← R3f 主判据
  B) 接触 regime 闭环 k=3：RMS / 误关率                ← R2f 不伤接触
用于按预注册规则选 τ*，并判定"是否需要训练更好的指示器"。

用法：python tmp_s1f_threshold_diag.py [--val-windows 20000] [--arms a,b] [--taus .5,.9]
输出：控制台 + analysis_output/S1f_threshold_diag.json
"""
import argparse
import json

import numpy as np
import torch

from layout import OBJ_DIMS, OBS_DIMS
from tmp_s1b_compare import load_model, transfer_clean_mask
from train_v3 import V3Data, load_dataset, split_episodes

SEEDS = [42, 43, 44]
ARMS = {"baseline": "checkpoints-S1b/euler001x5_s{s}/best.pt",
        "sgate": "checkpoints-S1d/sgate_s{s}/best.pt",
        "W2": "checkpoints-S1e/W2_s{s}/best.pt",
        "S": "checkpoints-S1e/S_s{s}/best.pt"}
TAUS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
POS6 = [1, 2, 12, 13, 23, 24]


@torch.no_grad()
def rollout_masked(model, x_seq, actions_fut, obstacles_last, gua0, tau,
                   per_object=False):
    """硬静默 rollout：预测接触 < τ → 跳过残差头。

    per_object=False：窗口级（max over 3 物体 < τ 才静默，SLOT-1e 现状）；
    per_object=True ：**逐物体**（每个物体用自己的接触概率判决）。
    返回 (preds, silenced(B,K) 窗口级, silenced_obj(B,3,K) 逐物体)。
    """
    B = x_seq.size(0)
    h = torch.zeros(B, model.hidden_dim, device=x_seq.device)
    for t in range(x_seq.size(1)):
        h = model.gru_cell(x_seq[:, t], h)
    h = model.ln_h(h)
    obj_last = x_seq[:, -1, :model.obj_dim]
    K = actions_fut.size(1)
    cur_gua = gua0
    preds = []
    sil = torch.zeros(B, K, device=x_seq.device)
    sil_obj = torch.zeros(B, 3, K, device=x_seq.device)
    probs = []
    for k in range(K):
        a_k = actions_fut[:, k]
        prior_k = model.prior_norm(obj_last, a_k)
        delta = model.predict_state(h, a_k, cur_gua)
        yao_log = model.predict_yao(h)
        ct = torch.sigmoid(yao_log[:, 0:3].float())
        if per_object:
            so = (ct < tau)                                   # (B,3) True=关门（静默）
            keep = (~so).float()                              # 保留残差的门（勿用 so 当乘子！）
            mask = torch.zeros(B, delta.size(1), device=x_seq.device)
            for o in range(3):
                mask[:, o * 11:(o + 1) * 11] = keep[:, o].unsqueeze(-1)
            obj_next = prior_k + mask * delta
            contact_fb = ct * keep
        else:
            silent = (ct.max(dim=1).values < tau).float()
            so = silent.unsqueeze(-1).expand(B, 3).bool()
            obj_next = prior_k + (1.0 - silent).unsqueeze(-1) * delta
            contact_fb = ct * (1.0 - silent).unsqueeze(-1)
        sil[:, k] = so.all(dim=1).float()
        sil_obj[:, :, k] = so.float()
        probs.append(ct)
        preds.append(obj_next)
        x_next = model.assemble_next_input(obj_next, obstacles_last, contact_fb)
        h = model.ln_h(model.gru_cell(x_next, h))
        obj_last = obj_next
        if model.use_gua:
            cur_gua = model.gua_from_yao(yao_log.detach())
    return preds, sil, sil_obj, torch.stack(probs, dim=1)


@torch.no_grad()
def eval_tau(model, ds, device, idx_all, K, tau, per_object=False, bs=2048):
    M = len(idx_all)
    pred = [np.zeros((M, OBJ_DIMS), np.float32) for _ in range(K)]
    sil = np.zeros((M, K), bool)
    sil_obj = np.zeros((M, 3, K), bool)
    ct_prob = np.zeros((M, K, 3), np.float32)
    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = b["x"].to(device)
        obb = xb[:, -1, OBJ_DIMS:OBJ_DIMS + OBS_DIMS].contiguous()
        ps, s, so, cp = rollout_masked(model, xb, b["af"].to(device), obb,
                                       b["gua"].to(device), tau, per_object)
        for k in range(K):
            pred[k][sl] = ps[k].float().cpu().numpy() * (ds.std_obj + 1e-8) + ds.mean_obj
        sil[sl] = s.cpu().numpy().astype(bool)
        sil_obj[sl] = so.cpu().numpy().astype(bool)
        ct_prob[sl] = cp.float().cpu().numpy()
    ep = ds.ep_idx[idx_all]
    t0 = ds.win_start[idx_all] + ds.L
    tgt = ds.obj_norm[ep, t0 + K - 1] * (ds.std_obj + 1e-8) + ds.mean_obj
    per_win = np.sqrt(((pred[K - 1][:, POS6] - tgt[:, POS6]) ** 2).mean(axis=1))
    ff = transfer_clean_mask(ds, idx_all, K)                       # FF-clean 子集
    ct_reg = ds.contact[ep, t0].max(axis=1) > 0.5                  # 目标帧有接触
    all_sil = sil.all(axis=1)                                      # 全步关门
    # 逐（窗, 步, 物体）单元的真值接触：指示器的 ROC 就在这一层
    ct_gt = ds.contact[ep[:, None], t0[:, None] + np.arange(K)[None, :]] > 0.5
    free_cells, ct_cells = ~ct_gt, ct_gt
    sil_t = sil_obj.transpose(0, 2, 1)                    # (M,K,3) 与 ct_gt 同轴序
    per_k = {}
    for k in range(K):
        fk, ck = ~ct_gt[:, k, :], ct_gt[:, k, :]
        per_k[f"k={k+1}"] = dict(
            n_free=int(fk.sum()), n_contact=int(ck.sum()),
            prob_free=float(ct_prob[:, k, :][fk].mean()) if fk.any() else float("nan"),
            prob_contact=float(ct_prob[:, k, :][ck].mean()) if ck.any() else float("nan"),
            protect_rate_free=float(sil_t[:, k, :][fk].mean()) if fk.any() else float("nan"),
            false_silence_contact=float(sil_t[:, k, :][ck].mean()) if ck.any() else float("nan"))
    out = {"cells": dict(
        n_free=int(free_cells.sum()), n_contact=int(ct_cells.sum()),
        protect_rate_free=float(sil_t[free_cells].mean()) if free_cells.any() else float("nan"),
        false_silence_contact=float(sil_t[ct_cells].mean()) if ct_cells.any() else float("nan")),
        "per_k": per_k}
    if ff.any():
        e = per_win[ff]
        full = all_sil[ff]
        ff_cells = sil_t[ff]                                   # (n_ff, K, 3)
        out["ff"] = dict(n=int(ff.sum()), rms=float(np.sqrt((e ** 2).mean())),
                         median=float(np.median(e)),
                         frac_all_silenced=float(full.mean()),
                         cell_sil_rate=float(ff_cells.mean()),
                         n_full=int(full.sum()),
                         rms_full=float(np.sqrt((e[full] ** 2).mean())) if full.any()
                         else float("nan"),
                         rms_partial=float(np.sqrt((e[~full] ** 2).mean())) if (~full).any()
                         else float("nan"))
    if ct_reg.any():
        e = per_win[ct_reg]
        out["contact"] = dict(n=int(ct_reg.sum()), rms=float(np.sqrt((e ** 2).mean())),
                              median=float(np.median(e)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-windows", type=int, default=20000)
    ap.add_argument("--arms", type=str, default="baseline,sgate,W2,S")
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--taus", type=str, default=",".join(str(t) for t in TAUS))
    ap.add_argument("--out", type=str, default="analysis_output/S1f_threshold_diag.json")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    taus = [float(t) for t in args.taus.split(",") if t]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("载入 trajectories_v3.npz …", flush=True)
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]

    out = {"note": "SLOT-1f 前置诊断：硬静默阈值扫描", "arms": {}}
    for arm in arms:
        out["arms"][arm] = {}
        for s in seeds:
            model, ck = load_model(ARMS[arm].format(s=s))
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
            model = model.to(device)
            row = {}
            for mode in (False, True):
                for tau in taus:
                    row[f"{'obj' if mode else 'win'}|{tau}"] = eval_tau(
                        model, ds, device, idx_all, a["future_steps"], tau, mode)
            out["arms"][arm][s] = row
            print(f"\n[{arm} s{s}]  (win=窗口级 max判决 / obj=逐物体判决)")
            print(f"  {'模式/τ':>10s} | {'FF n':>5s} {'FF RMS':>8s} {'FF中位':>8s} "
                  f"{'全步关门':>8s} | {'自由单元保护':>11s} {'接触单元误关':>11s} | "
                  f"{'接触RMS':>8s}")
            for mode in (False, True):
                for tau in taus:
                    r = row[f"{'obj' if mode else 'win'}|{tau}"]
                    f = r.get("ff", {})
                    c = r.get("contact", {})
                    cl = r["cells"]
                    print(f"  {'obj' if mode else 'win'}/{tau:<5.2f} | "
                          f"{f.get('n', 0):5d} {f.get('rms', float('nan')):8.4f} "
                          f"{f.get('median', float('nan')):8.4f} "
                          f"{f.get('frac_all_silenced', float('nan')):8.1%} | "
                          f"{cl['protect_rate_free']:11.1%} "
                          f"{cl['false_silence_contact']:11.1%} | "
                          f"{c.get('rms', float('nan')):8.4f}")
            if not mode:
                print("   逐步剖面（逐物体单元，p̄=预测接触概率均值，保护/误关=门关占比）：")
                for k, v in row["win|0.5"]["per_k"].items():
                    print(f"     {k}: 自由 n={v['n_free']:6d} p̄={v['prob_free']:.3f} "
                          f"保护={v['protect_rate_free']:6.1%} | 接触 n={v['n_contact']:6d} "
                          f"p̄={v['prob_contact']:.3f} 误关={v['false_silence_contact']:6.1%}")
            del model
            torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
