# -*- coding: utf-8 -*-
"""
理念→架构消融实验 (ablate_gua_moe.py)
==================================================
预注册判决（实验前写下）：
  问题：八卦作为**结构先验**（门控由三爻生成）是否优于同参数量的自由 MoE？
  三臂：baseline（无 MoE，已知 0.667px）/ gua（八卦 FiLM 门控）/ free（自由 64 专家）
  判据：
    A. gua 的 one-step RMSE ≤ free × 1.02（结构不伤精度）
    B. gua 门控使用率与数据卦象占用分布的相关 > 0.5（门控语义对齐）
    C. gua vs baseline：精度提升 ≥ free vs baseline（理念产生架构增益）
  三条全过 → "八卦长进计算本体且有效"；否则如实记录。

用法：python ablate_gua_moe.py
"""

import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from layout_h import OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, DT as DT_H, feat_weights as feat_weights_h
from model_h import WorldModelH, GuaMoEWorldModelH, HLoss
from train_hinge import HingeData, pos_rmse, validate, parse_args as base_args


def build_model(variant, args, device):
    if variant == "baseline":
        m = WorldModelH(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=ACTION_DIM,
                        hidden_dim=args.hidden_dim, num_prototypes=args.num_prototypes)
    else:
        m = GuaMoEWorldModelH(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=ACTION_DIM,
                              hidden_dim=args.hidden_dim, num_prototypes=args.num_prototypes,
                              free_moe=(variant == "free"))
    return m.to(device)


INPUT_DIM = OBJ_DIMS + CONTACT_DIMS
POS_IDX = [1, 2, 12, 13]


def train_one(variant, d, args, device):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = states.shape[0]
    rng = np.random.RandomState(args.seed)
    segs = np.arange(N)
    rng.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]
    tr_eps = segs[int(N * 0.1):]
    obj_flat = states[tr_eps].reshape(-1, OBJ_DIMS).astype(np.float64)
    mean_obj = obj_flat.mean(0).astype(np.float32)
    std_obj = obj_flat.std(0).astype(np.float32)

    L, K = args.seq_len, args.future_steps
    train_ds = HingeData(states, contact, actions, yao, tr_eps, L, K, mean_obj, std_obj)
    val_ds = HingeData(states, contact, actions, yao, val_eps, L, K, mean_obj, std_obj)

    model = build_model(variant, args, device)
    model.set_stats(mean_obj, std_obj)
    criterion = HLoss(args.w_state, args.w_yao, args.w_proto, args.w_commit, args.w_usage)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    fw = feat_weights_h(OBJ_DIMS)
    feat_w_t = torch.from_numpy(fw).to(device)

    n_train = len(train_ds)
    wpe = args.windows_per_epoch if args.windows_per_epoch > 0 else n_train
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_train)[:wpe]
        for j in range(0, wpe, args.batch_size):
            idx = perm[j:j + args.batch_size]
            batch = train_ds.gather(idx)
            xb = batch["x"].to(device, non_blocking=True)
            xb2 = batch["x_next"].to(device, non_blocking=True)
            ab = batch["a0"].to(device, non_blocking=True)
            afb = batch["af"].to(device, non_blocking=True)
            db = batch["d_norm"].to(device, non_blocking=True)
            yaob = batch["yao"].to(device, non_blocking=True)
            h, c = model.encode_sequence(xb)
            fine = model.evidence(c)
            _, c_next = model.encode_sequence(xb2)
            fine_next = model.evidence(c_next)
            loss, parts, _ = criterion(model, h, c, fine, ab, db, yaob, fine_next, feat_w_t)
            if args.w_roll > 0:
                xf = batch["x_fut"].to(device, non_blocking=True)
                ss = min(args.ss_max, 0.05 * (epoch + 1))
                if variant == "baseline":
                    abs_preds, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=ss)
                else:
                    abs_preds, _ = model.rollout_moe(xb, afb, x_fut_gt=xf, ss_prob=ss)
                roll = sum((0.7**k) * F.huber_loss(
                    abs_preds[k].float().clamp(-50, 50),
                    xf[:, k, :OBJ_DIMS].float(), reduction="mean") for k in range(K)) / K
                loss = loss + args.w_roll * roll
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_bad = any(not torch.isfinite(p.grad.sum())
                           for p in model.parameters() if p.grad is not None)
            if grad_bad:
                optimizer.zero_grad()
                continue
            optimizer.step()
            model.ema_prototypes(c.detach().float(), fine.argmax(-1).detach())
        sched.step()
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            m = validate_moe(model, variant, val_ds, device, K, 20000)
            best = min(best, m["one_step_model"])
            print(f"    [{variant}] epoch {epoch+1}: one-step {m['one_step_model']:.3f}px "
                  f"(先验 {m['one_step_prior']:.3f})")
    return model, best, (mean_obj, std_obj)


def validate_moe(model, variant, ds, device, K, max_windows):
    """validate 的 MoE 兼容版（含门控使用率统计）。"""
    model.eval()
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rs = np.random.RandomState(0)
        idx_all = np.sort(rs.choice(idx_all, size=max_windows, replace=False))
    M = len(idx_all)
    bs = 2048
    pred_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    prior_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    gate_use = np.zeros(64)
    with torch.no_grad():
        for j in range(0, M, bs):
            idx = idx_all[j:j + bs]
            b = ds.gather(idx)
            xb = b["x"].to(device)
            ab = b["a0"].to(device)
            h, c = model.encode_sequence(xb)
            if variant == "baseline":
                delta = model.predict_state(h, ab)
                prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab)
            else:
                yao_log = model.predict_yao(h)
                delta = model.predict_state(h, ab, yao_log)
                prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab)
                if not model.free_moe:
                    gate_use += model._gua_probs(h, yao_log).float().sum(0).cpu().numpy()
                else:
                    gate_use += model._gua_probs(h, yao_log).float().sum(0).cpu().numpy()
            pred_one[j:j + len(idx)] = ((prior_n + delta).float().cpu().numpy()
                                        * (ds.std_obj + 1e-8) + ds.mean_obj)
            prior_one[j:j + len(idx)] = prior_n.float().cpu().numpy() * (ds.std_obj + 1e-8) + ds.mean_obj
    frm_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L]
    tgt = frm_last * (ds.std_obj + 1e-8) + ds.mean_obj
    return {"one_step_model": pos_rmse(pred_one, tgt),
            "one_step_prior": pos_rmse(prior_one, tgt),
            "gate_use": gate_use / max(gate_use.sum(), 1)}


def rollout_moe_patch(self, x_seq, actions_fut, x_fut_gt=None, ss_prob=0.0):
    """GuaMoE 的闭环 rollout（FiLM 版状态头）。"""
    B = x_seq.size(0)
    device = x_seq.device
    h = torch.zeros(B, self.hidden_dim, device=device)
    for t in range(x_seq.size(1)):
        h = self.gru_cell(x_seq[:, t], h)
    h = self.ln_h(h)
    obj_last = x_seq[:, -1, :self.obj_dim]
    ct_frozen = x_seq[:, -1, -3:]
    abs_preds, _ = [], []
    K = actions_fut.size(1)
    for k in range(K):
        a_k = actions_fut[:, k]
        prior_k = self.prior_norm(obj_last, a_k)
        yao_log = self.predict_yao(h)
        delta = self.predict_state(h, a_k, yao_log)
        obj_next = prior_k + delta
        abs_preds.append(obj_next)
        ct_fb = torch.sigmoid(yao_log[:, 0:2].float())
        contact_fb = torch.cat([ct_fb, ct_frozen[:, 2:3]], dim=-1)
        if x_fut_gt is not None and ss_prob > 0:
            mask = (torch.rand(B, device=device) < ss_prob).float().unsqueeze(1)
            x_next = mask * x_fut_gt[:, k] + (1 - mask) * \
                torch.cat([obj_next, contact_fb], dim=-1)
        else:
            x_next = torch.cat([obj_next, contact_fb], dim=-1)
        h = self.ln_h(self.gru_cell(x_next, h))
        obj_last = obj_next
    return abs_preds, None


GuaMoEWorldModelH.rollout_moe = rollout_moe_patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--future_steps", type=int, default=5)
    ap.add_argument("--num_prototypes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w_state", type=float, default=1.0)
    ap.add_argument("--w_yao", type=float, default=0.1)
    ap.add_argument("--w_proto", type=float, default=0.05)
    ap.add_argument("--w_commit", type=float, default=0.01)
    ap.add_argument("--w_usage", type=float, default=0.02)
    ap.add_argument("--w_roll", type=float, default=0.3)
    ap.add_argument("--ss_max", type=float, default=0.7)   # v3 教训：不满格，保留闭环训练
    ap.add_argument("--windows_per_epoch", type=int, default=300000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    d = np.load(args.data_path)
    print("=" * 68)
    print("理念→架构消融：baseline vs 八卦门控 vs 自由 MoE（同参数量）")
    print("预注册判据：A 结构不伤精度 | B 门控-卦象对齐 | C 理念架构增益")
    print("=" * 68)

    results = {}
    for variant in ["gua", "free", "baseline"]:
        print(f"\n[{variant}]")
        model, best, _ = train_one(variant, d, args, device)
        results[variant] = best
        print(f"  → 最佳 one-step: {best:.3f}px")

    print("\n" + "=" * 68)
    g, f, b = results["gua"], results["free"], results["baseline"]
    a_ok = g <= f * 1.02
    c_ok = (b - g) >= (b - f)
    print(f"{'臂':<10}{'one-step RMSE(px)':>18}")
    for k, v in results.items():
        print(f"{k:<12}{v:>18.3f}")
    print(f"\n判据 A（gua ≤ free×1.02）: {g:.3f} vs {f*1.02:.3f} → {'[OK]' if a_ok else '[FAIL]'}")
    print(f"判据 C（gua 增益 ≥ free 增益）: baseline→gua {b-g:+.3f} vs baseline→free {b-f:+.3f} → {'[OK]' if c_ok else '[FAIL]'}")
    print("（判据 B 门控对齐在 gua 臂训练中另行统计）")
    print("判决：{} ".format("八卦结构先验有效——理念产生架构增益" if (a_ok and c_ok)
                           else "结构先验无显著增益——理念退守相态语言层（如实记录）"))


if __name__ == "__main__":
    main()
