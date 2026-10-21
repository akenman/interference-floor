# -*- coding: utf-8 -*-
"""SLOT-1e 前置诊断：闭环 FF 误差预算分解（0 GPU，判据标定用）。

问题（SLOT-1d 探针引出）：sgate 把闭环 FF 子集上的头噪声削了 3.27×（RMS 0.0124→0.0038），
闭环 k=3 误差却只降 1.28×（2.042→1.596）。⇒ 闭环 FF 误差的主成分可能不是头噪声，
而是**接触反馈通路 + 状态自回归复合**。若成立，任何"静默/损失质量"操纵都买不到 R3，
阈值也会标错（10.19.x 已四次栽在阈值标定上）。本脚本用已有 checkpoint 把它测出来。

条件矩阵（delta × 状态通路 × 接触反馈）：
  delta  : "head"（模型残差头）| "zero"（关掉头 = 纯解析先验）
  state  : "self"（自身预测回灌）| "gt"（真值帧回灌 = 无复合）
  contact: "soft"（爻头软概率，现状）| "hard"（爻头硬判决）| "freeze"（冻结末观测）
           | "gt"（真值接触 = 反馈通路完美）
关键参照：zero-gt-gt = 解析恒等（应复现 SLOT-1 的 0.00003px）；
  zero-self-soft − zero-self-gt = **纯接触反馈通道误差（零头噪声）**。

用法：python tmp_s1e_error_budget.py [--val-windows 20000]
输出：控制台 + analysis_output/S1e_error_budget.json
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
        "wff10": "checkpoints-S1c/wff10_s{s}/best.pt",
        "sgate": "checkpoints-S1d/sgate_s{s}/best.pt",
        "W1": "checkpoints-S1e/W1_s{s}/best.pt",
        "W2": "checkpoints-S1e/W2_s{s}/best.pt",
        "S": "checkpoints-S1e/S_s{s}/best.pt"}
POS6 = [1, 2, 12, 13, 23, 24]
CONDITIONS = [
    ("head", "self", "soft"), ("head", "self", "hard"), ("head", "self", "freeze"),
    ("head", "self", "gt"), ("head", "gt", "soft"), ("head", "gt", "gt"),
    ("zero", "self", "soft"), ("zero", "self", "gt"), ("zero", "gt", "gt"),
]


@torch.no_grad()
def rollout_diag(model, x_seq, actions_fut, obstacles_last, x_fut_gt, gua0,
                 delta_mode, state_mode, contact_mode):
    """rollout 的显式分解版（逐项可换），语义与 model_v3.rollout 对齐。"""
    B = x_seq.size(0)
    h = torch.zeros(B, model.hidden_dim, device=x_seq.device)
    for t in range(x_seq.size(1)):
        h = model.gru_cell(x_seq[:, t], h)
    h = model.ln_h(h)
    obj_last = x_seq[:, -1, :model.obj_dim]
    ct_frozen = x_seq[:, -1, -3:]
    K = actions_fut.size(1)
    cur_gua = gua0
    preds = []
    for k in range(K):
        a_k = actions_fut[:, k]
        prior_k = model.prior_norm(obj_last, a_k)
        delta = (torch.zeros_like(prior_k) if delta_mode == "zero"
                 else model.predict_state(h, a_k, cur_gua))
        yao_log = model.predict_yao(h)
        obj_next = prior_k + delta
        preds.append(obj_next)
        if contact_mode == "hard":
            contact_fb = (torch.sigmoid(yao_log[:, 0:3].float()) > 0.5).float()
        elif contact_mode == "freeze":
            contact_fb = ct_frozen
        elif contact_mode == "gt":
            contact_fb = x_fut_gt[:, k, -3:]
        else:
            contact_fb = torch.sigmoid(yao_log[:, 0:3].float())
        if state_mode == "gt":
            x_next = x_fut_gt[:, k]
            obj_last = x_fut_gt[:, k, :model.obj_dim]
        else:
            x_next = model.assemble_next_input(obj_next, obstacles_last, contact_fb)
            obj_last = obj_next
        h = model.ln_h(model.gru_cell(x_next, h))
        if model.use_gua:
            cur_gua = model.gua_from_yao(yao_log.detach())
    return preds


@torch.no_grad()
def eval_conditions(model, ds, device, idx_all, K, bs=2048):
    """返回 {cond: {k=1,2,3: pos6 RMSE px（闭环 FF-clean 子集）}}。"""
    M = len(idx_all)
    out = {f"{d}-{s}-{c}": [np.zeros((M, OBJ_DIMS), np.float32) for _ in range(K)]
           for d, s, c in CONDITIONS}
    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = b["x"].to(device)
        afb = b["af"].to(device)
        obb = xb[:, -1, OBJ_DIMS:OBJ_DIMS + OBS_DIMS].contiguous()   # 障碍段
        xf = b["x_fut"].to(device)
        gua0 = b["gua"].to(device)
        for d, s, c in CONDITIONS:
            preds = rollout_diag(model, xb, afb, obb, xf, gua0, d, s, c)
            for k in range(K):
                out[f"{d}-{s}-{c}"][k][sl] = \
                    preds[k].float().cpu().numpy() * (ds.std_obj + 1e-8) + ds.mean_obj

    ep = ds.ep_idx[idx_all]
    t0 = ds.win_start[idx_all] + ds.L
    frm = [ds.obj_norm[ep, t0 + k] * (ds.std_obj + 1e-8) + ds.mean_obj
           for k in range(K)]
    cl = transfer_clean_mask(ds, idx_all, K)
    one = transfer_clean_mask(ds, idx_all, 1)

    res = {}
    for key, plist in out.items():
        row = {}
        for k in range(K):
            p, t = plist[k], frm[k]
            row[f"k={k+1}_ffclean"] = float(
                np.sqrt(((p[cl][:, POS6] - t[cl][:, POS6]) ** 2).mean())) if cl.any() \
                else float("nan")
            if k == 0:
                row["k=1_oneclean"] = float(
                    np.sqrt(((p[one][:, POS6] - t[one][:, POS6]) ** 2).mean())) \
                    if one.any() else float("nan")
        res[key] = row
    res["_n"] = {"n_cl_clean": int(cl.sum()), "n_one_clean": int(one.sum())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-windows", type=int, default=20000)
    ap.add_argument("--arms", type=str, default=",".join(ARMS),
                    help="逗号分隔的臂名（冒烟用）")
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--out", type=str, default="analysis_output/S1e_error_budget.json")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("载入 trajectories_v3.npz …", flush=True)
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]

    out = {"note": "SLOT-1e 前置诊断：闭环 FF 误差预算分解（0 GPU）", "arms": {}}
    for arm in arms:
        pat = ARMS[arm]
        out["arms"][arm] = {}
        for s in seeds:
            model, ck = load_model(pat.format(s=s))
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
            r = eval_conditions(model.to(device), ds, device, idx_all, a["future_steps"])
            out["arms"][arm][s] = r
            print(f"\n[{arm} s{s}] n_cl={r['_n']['n_cl_clean']} "
                  f"n_one={r['_n']['n_one_clean']}")
            for key in [f"{d}-{s2}-{c}" for d, s2, c in CONDITIONS]:
                row = r[key]
                print(f"  {key:20s} k=1 {row['k=1_oneclean']:8.4f}(one-clean) "
                      f"| 闭环 k=1 {row['k=1_ffclean']:8.4f} "
                      f"k=2 {row['k=2_ffclean']:8.4f} "
                      f"k=3 {row['k=3_ffclean']:8.4f}")
            del model
            torch.cuda.empty_cache()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
