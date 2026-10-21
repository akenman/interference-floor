# -*- coding: utf-8 -*-
"""
三臂公平对决 (benchmark_arms.py)
==================================================
同一数据、同一训练预算、同一评估协议：
  Ours      = 物理先验基座 + 残差头（理念架构）
  NoPrior   = 同 GRU + 直接状态头（无物理先验）
  JEPALike  = 同 GRU + latent 预测 + EMA target（表征范式）

预注册判据：
  A: Ours 的 one-step RMSE < NoPrior 和 JEPALike（物理先验+理念有架构增益）
  B: Ours 的数据效率最高（1k 数据时 Ours 优势最大）
  C: Ours 的 rollout k=5 最优（长期预测优势）

用法：python benchmark_arms.py --episodes 3000 --epochs 60
"""

import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from layout_h import OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, DT as DT_H, feat_weights as feat_weights_h, L1, L2, M1, M2, TAU_MAX
from core_model import OursModel, NoPriorModel, JEPALikeModel

INPUT_DIM = OBJ_DIMS + CONTACT_DIMS
POS_IDX = [1, 2, 12, 13]
G_PRIOR = 900.0
TAU_DT = 0.05


def prior_step_np(obj_raw, a_norm):
    """物理先验 numpy 单步（与 core_model._euler_step_torch 一致）。"""
    th1 = math.atan2(obj_raw[3], obj_raw[4])
    th2 = math.atan2(obj_raw[14], obj_raw[15])
    w1 = obj_raw[7]; w2 = obj_raw[18]
    tau = a_norm * TAU_MAX
    c2 = math.cos(th2); s2 = math.sin(th2)
    M11 = (M1+M2)*L1**2 + M2*L2**2 + 2*M2*L1*L2*c2
    M12 = M2*L2**2 + M2*L1*L2*c2
    M22 = M2*L2**2
    h1 = -M2*L1*L2*(2*w1+w2)*w2*s2 + (M1+M2)*g*L1*math.sin(th1) + M2*g*L2*math.sin(th1+th2)
    h2 = M2*L1*L2*w1**2*s2 + M2*g*L2*math.sin(th1+th2) - tau
    det = M11*M22 - M12**2
    det = abs(det) if abs(det) >= 1e-9 else 1e-9 * (1 if det >= 0 else -1)
    a1 = (M22*(-h1) - M12*(-h2))/det
    a2 = (-M12*(-h1) + M11*(-h2))/det
    w1n = w1 + a1*0.05; w2n = w2 + a2*0.05
    th1n = th1 + w1n*0.05; th2n = th2 + w2n*0.05
    out = obj_raw.copy()
    out[1] = L1*math.sin(th1n); out[2] = -L1*math.cos(th1n)
    out[12] = out[1] + L2*math.sin(th1n+th2n); out[13] = out[2] - L2*math.cos(th1n+th2n)
    out[3] = math.sin(th1n); out[4] = math.cos(th1n)
    out[14] = math.sin(th2n); out[15] = math.cos(th2n)
    out[5] = L1*w1n*math.cos(th1n); out[6] = L1*w1n*math.sin(th1n)
    out[16] = out[5] + L2*(w1n+w2n)*math.cos(th1n+th2n)
    out[17] = out[6] + L2*(w1n+w2n)*math.sin(th1n+th2n)
    out[7] = w1n; out[18] = w2n
    return out

g = G_PRIOR


class HingeWindowDS:
    """简化的铰链窗口数据集（仅训练用，无惰性优化——pilot 规模够用）。"""

    def __init__(self, states, contact, actions, episodes, L, K, mean_obj, std_obj, noise=0.0):
        self.L, self.K = L, K
        self.noise = noise
        eps = np.asarray(episodes)
        T = states.shape[1]
        n_win = T - L - K + 1
        # 预归一化
        self.obj_n = ((states[eps] - mean_obj) / (std_obj + 1e-8)).astype(np.float32)
        self.act = actions[eps].astype(np.float32)
        self.ct = contact[eps].astype(np.float32)
        self.ep_idx = np.repeat(np.arange(len(eps)), n_win)
        self.win_start = np.tile(np.arange(n_win), len(eps))
        # 真值目标（绝对归一化状态，每 k 步）
        self.targets = []
        for k in range(K):
            self.targets.append(self.obj_n[:, L+k:L+k+1, :])  # (E, 1, 22)

    def __len__(self):
        return len(self.ep_idx)

    def gather(self, idx, device):
        ep = self.ep_idx[idx]; i0 = self.win_start[idx]
        B = len(idx); L, K = self.L, self.K
        off = np.arange(L + K)
        frm = self.obj_n[ep[:, None], i0[:, None] + off[None, :]]   # (B, L+K, 22)
        ct = self.ct[ep[:, None], i0[:, None] + off[None, :]]       # (B, L+K, 3)
        # 噪声注入（G1）
        if self.noise > 0:
            s = self.noise * np.random.rand(B, 1, 1).astype(np.float32)
            frm = frm + np.random.randn(*frm.shape).astype(np.float32) * s
        x = np.concatenate([frm[:, :L], ct[:, :L]], axis=2)          # (B, L, 25)
        x_next = np.concatenate([frm[:, 1:L+1], ct[:, 1:L+1]], axis=2)
        a0 = self.act[ep, i0 + L]                                    # (B, 1)
        # 每 k 步绝对目标（归一化）
        tgts = [self.obj_n[ep, i0 + L + k] for k in range(K)]        # K × (B, 22)
        x_fut = np.stack([np.concatenate([self.obj_n[ep, i0+L+k],
                          self.ct[ep, i0+L+k]], axis=-1) for k in range(K)], axis=1)
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
        return dict(x=to(x), x_next=to(x_next), a0=to(a0), x_fut=to(x_fut),
                    tgts=[to(t) for t in tgts])


def train_arm(variant, ds, args, device):
    """训练单臂，返回模型。"""
    n_train = len(ds)
    wpe = args.windows_per_epoch

    if variant == "ours":
        model = OursModel(INPUT_DIM, OBJ_DIMS, ACTION_DIM, args.hidden_dim).to(device)
    elif variant == "noprior":
        model = NoPriorModel(INPUT_DIM, OBJ_DIMS, ACTION_DIM, args.hidden_dim).to(device)
    elif variant == "jepa":
        model = JEPALikeModel(INPUT_DIM, OBJ_DIMS, ACTION_DIM, args.hidden_dim,
                              latent_dim=args.hidden_dim).to(device)

    model.set_stats(ds.mean_obj, ds.std_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_train)[:wpe]
        total_loss = 0.0; nb = 0
        for j in range(0, wpe, args.batch_size):
            idx = perm[j:j + args.batch_size]
            b = ds.gather(idx, device)
            xb, ab = b["x"], b["a0"]
            xf = b["x_fut"]  # (B, K, 25)
            tgts = b["tgts"]  # K × (B, 22)

            if variant == "ours":
                h = model.trunk(xb)
                obj_norm_last = xb[:, -1, :OBJ_DIMS]
                prior_n = model.prior_norm(obj_norm_last, ab)
                delta = model.predict(h, ab)
                pred = prior_n + delta                          # (B, 22) 残差参数化
                loss = F.huber_loss(pred, tgts[0], reduction="mean")
            elif variant == "noprior":
                h = model.trunk(xb)
                pred = model.predict(h, ab)                     # (B, 22) 直接预测
                losses_k = [F.huber_loss(pred, tgts[0], reduction="mean")]
                loss = sum(losses_k) / len(losses_k)
            elif variant == "jepa":
                # JEPA: latent 预测 + EMA target
                pred_latent = model.forward_latent(xb, ab)      # (B, latent_dim)
                with torch.no_grad():
                    tgt_latent = model.forward_target(
                        torch.cat([xb[:, 1:], xf[:, 0:1]], dim=1))  # (B, hidden)
                loss = F.mse_loss(pred_latent, tgt_latent)
                # probe loss（训练 linear probe）
                probe_pred = model.predict_state(pred_latent, ab)
                probe_tgt = tgts[0]
                probe_loss = F.huber_loss(probe_pred, probe_tgt, reduction="mean")
                loss = loss + 0.5 * probe_loss

            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); nb += 1
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"    [{variant}] epoch {epoch+1}: loss={total_loss/max(nb,1):.4f}")

    return model


@torch.no_grad()
def eval_arm(variant, model, ds, device, K=5, max_windows=5000):
    """统一 one-step 评估（预注册判据 A 的核心指标）。"""
    model.eval()
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rs = np.random.RandomState(0)
        idx_all = np.sort(rs.choice(idx_all, size=max_windows, replace=False))
    M = len(idx_all)
    pos_std = float(np.mean(ds.std_obj[POS_IDX]))
    med_err = np.zeros(M)
    prior_err = np.zeros(M)
    for j in range(0, M, 2048):
        idx = idx_all[j:j + 2048]
        b = ds.gather(idx, device)
        sl = slice(j, j + len(idx))
        xb, ab = b["x"], b["a0"]
        tgts = b["tgts"]

        if variant == "ours":
            h = model.trunk(xb)
            obj_last = xb[:, -1, :OBJ_DIMS]
            prior_n = model.prior_norm(obj_last, ab)
            delta = model.predict(h, ab)
            pred_n = prior_n + delta
        elif variant == "noprior":
            h = model.trunk(xb)
            pred_n = model.predict(h, ab)
            prior_n = torch.zeros_like(pred_n)  # 无先验（计数用）
        elif variant == "jepa":
            pred_latent = model.forward_latent(xb, ab)
            pred_n = model.predict_state(pred_latent, ab)
            prior_n = torch.zeros_like(pred_n)

        tgt = tgts[0]
        e = np.abs(pred_n.float().cpu().numpy()[:, POS_IDX] - tgt.cpu().numpy()[:, POS_IDX]).mean(axis=1)
        med_err[sl] = e
        if variant == "ours":
            e_p = np.abs(prior_n.float().cpu().numpy()[:, POS_IDX] - tgt.cpu().numpy()[:, POS_IDX]).mean(axis=1)
            prior_err[sl] = e_p

    out = {"k1": float(np.sqrt((med_err**2).mean())) * pos_std}
    if variant == "ours":
        out["prior_k1"] = float(np.sqrt((prior_err**2).mean())) * pos_std
    return out



def main():
    ap = argparse.ArgumentParser(description="三臂公平对决")
    ap.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--future_steps", type=int, default=5)
    ap.add_argument("--num_prototypes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--windows_per_epoch", type=int, default=300000)
    ap.add_argument("--noise_aug", type=float, default=0.02)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_report", type=str, default="analysis_output/benchmark_arms.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    print("=" * 68)
    print("三臂公平对决：Ours vs NoPrior vs JEPALike")
    print(f"数据: {args.data_path} | {args.epochs} epochs | seed {args.seed}")
    print("预注册判据 A: Ours < NoPrior 且 Ours < JEPALike（one-step）")
    print("=" * 68)

    d = np.load(args.data_path)
    states = d["states"][:args.episodes]
    contact = d["contact"][:args.episodes]
    actions = d["actions"][:args.episodes]
    N = states.shape[0]
    rng = np.random.RandomState(args.seed)
    segs = np.arange(N); rng.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]
    tr_eps = segs[int(N * 0.1):]
    obj_flat = states[tr_eps].reshape(-1, OBJ_DIMS).astype(np.float64)
    mean_obj = obj_flat.mean(0).astype(np.float32)
    std_obj = obj_flat.std(0).astype(np.float32)

    ds = HingeWindowDS(states, contact, actions, tr_eps, args.seq_len,
                       args.future_steps, mean_obj, std_obj, noise=args.noise_aug)
    ds.mean_obj = mean_obj; ds.std_obj = std_obj

    results = {}
    for variant in ["ours", "noprior", "jepa"]:
        print(f"\n=== 臂: {variant} ===")
        model = train_arm(variant, ds, args, device)
        m = eval_arm(variant, model, ds, device, args.future_steps, 5000)
        results[variant] = m
        print(f"  → " + " | ".join(f"{k}: {v:.2f}px" for k, v in m.items()))

    print("\n" + "=" * 68)
    print(f"{'臂':<12}{'k=1 位置RMSE(px)':>18}")
    for v in ["ours", "noprior", "jepa"]:
        print(f"{v:<14}{results[v]['k1']:>18.3f}")

    ok_a = (results["ours"]["k1"] < results["noprior"]["k1"]
            and results["ours"]["k1"] < results["jepa"]["k1"])
    print(f"\n判据 A: Ours 一步最优 → {'[OK] 通过' if ok_a else '[FAIL] 未通过'}")

    report = {"results": results, "criteria_A": bool(ok_a), "seed": args.seed,
              "epochs": args.epochs, "episodes": args.episodes}
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] {args.output_report}")


if __name__ == "__main__":
    import json
    main()
