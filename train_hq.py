# -*- coding: utf-8 -*-
"""
G2v3 分位数分布预测训练与判决 (train_hq.py)
==================================================
预注册判据（实验前写下）：
  A 校准：[q0.1,q0.9] 经验覆盖率 = 0.80±0.06；[q0.25,q0.75] = 0.50±0.08
  B 锋锐：中位数（q0.5）作点预测的 RMSE ≤ 单点模型（0.667px 量级）
  C T11 响应：闭环滚动中分位数带宽（q0.9−q0.1）随 k 单调增长
  D 噪声响应：带宽随观测噪声水平单调增长（G1 融合）

T17 修补轮（checkpoints-hq3，2026-09-03）预注册判据（对照 T17: med_k5 8.35
vs base 2.94、闭环覆盖 k5 0.289）：
  R-A 闭环锋锐：med_k5 ≤ 1.5 × base_k5（T17 为 2.84×）
  R-B 闭环覆盖：k5 闭环 cov80 ≥ 0.5（T17 为 0.289）
  R-C 单步不回归：med_k1 ≤ 1.1 × T17 的 1.186px
  R-D 校准不回归：cov50 ∈ [0.42, 0.58]（T17 0.456）
用法：python train_hq.py --data_path trajectories_hinge.npz --epochs 40
"""

import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from layout_h import OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, feat_weights as feat_weights_h
from model_h import WorldModelH, QuantileWorldModelH, pinball_loss
from train_hinge import HingeData, pos_rmse

INPUT_DIM = OBJ_DIMS + CONTACT_DIMS
POS_IDX = [1, 2, 12, 13]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--baseline_ckpt", type=str, default="checkpoints-hinge/best.pt")
    ap.add_argument("--epochs", type=int, default=40)
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
    ap.add_argument("--w_mono", type=float, default=0.1)
    ap.add_argument("--w_roll", type=float, default=0.3,
                    help="闭环 pinball rollout 损失权重（T17 修补：GRU 学消费自生成分位数输入）")
    ap.add_argument("--ss_max", type=float, default=0.5,
                    help="scheduled sampling 上限（闭环训练逐步喂自生成中位数 beam）")
    ap.add_argument("--noise_aug", type=float, default=0.02,
                   help="观测噪声（训练+评估同水平）——分布宽度的真实来源（G1/T13 教训）")
    ap.add_argument("--windows_per_epoch", type=int, default=400000)
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints-hq")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    L, K = args.seq_len, args.future_steps

    print("=" * 68)
    print("G2v3 分位数分布预测（pinball + 闭环分位数带）")
    print("=" * 68)

    d = np.load(args.data_path)
    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = states.shape[0]
    rng = np.random.RandomState(args.seed)
    segs = np.arange(N); rng.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]
    tr_eps = segs[int(N * 0.1):]
    obj_flat = states[tr_eps].reshape(-1, OBJ_DIMS).astype(np.float64)
    mean_obj = obj_flat.mean(0).astype(np.float32)
    std_obj = obj_flat.std(0).astype(np.float32)

    train_ds = HingeData(states, contact, actions, yao, tr_eps, L, K, mean_obj, std_obj)
    val_ds = HingeData(states, contact, actions, yao, val_eps, L, K, mean_obj, std_obj)

    model = QuantileWorldModelH(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=ACTION_DIM,
                                hidden_dim=args.hidden_dim, num_prototypes=args.num_prototypes).to(device)
    model.set_stats(mean_obj, std_obj)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    fw = feat_weights_h(OBJ_DIMS)
    feat_w_t = torch.from_numpy(fw).to(device)
    taus = torch.tensor(model.quantiles, dtype=torch.float32, device=device)

    bck = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
    base = WorldModelH(**bck["config"]).to(device)
    base.load_state_dict(bck["model_state_dict"], strict=False)
    base.eval()

    n_train = len(train_ds)
    wpe = args.windows_per_epoch if args.windows_per_epoch > 0 else n_train

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_train)[:wpe]
        agg, nb = {}, 0
        for j in range(0, wpe, args.batch_size):
            idx = perm[j:j + args.batch_size]
            b = train_ds.gather(idx)
            xb = b["x"].to(device, non_blocking=True)
            if args.noise_aug > 0:
                s = args.noise_aug * torch.rand(xb.size(0), 1, 1, device=device)
                xb = xb + torch.randn_like(xb) * s
            xb2 = b["x_next"].to(device, non_blocking=True)
            ab = b["a0"].to(device, non_blocking=True)
            afb = b["af"].to(device, non_blocking=True)
            db = b["d_norm"].to(device, non_blocking=True)
            yaob = b["yao"].to(device, non_blocking=True)
            xf = b["x_fut"].to(device, non_blocking=True)

            h, c = model.encode_sequence(xb)
            fine = model.evidence(c)
            _, c_next = model.encode_sequence(xb2)
            fine_next = model.evidence(c_next)
            prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab)
            tgt_abs = prior_n + db                                  # (B,22) 单步绝对目标

            q_res = model.predict_q(h, ab)                          # (B,n_q,22) 残差分位数
            q_abs = prior_n.unsqueeze(1) + q_res                    # 绝对分位数
            pb = pinball_loss(tgt_abs, q_abs, taus)                 # 单步 pinball
            mono = sum(F.relu(q_res[:, i] - q_res[:, i + 1]).pow(2).mean()
                       for i in range(model.n_q - 1))               # 分位数单调性
            yao_log = model.predict_yao(h)
            yao_loss = F.binary_cross_entropy_with_logits(yao_log, yaob.float(), reduction="mean")
            pfine = model.predict_proto(h, ab)
            proto_loss = F.kl_div(F.log_softmax(pfine, dim=-1),
                                  F.softmax(fine_next, dim=-1), reduction="batchmean")
            assigned = model.prototypes.detach()[fine.argmax(dim=-1)]
            commit_loss = (1.0 - F.cosine_similarity(assigned, c, dim=-1)).mean()
            assignment = F.softmax(fine.float(), dim=-1)
            bavg = assignment.mean(0).clamp(min=1e-8)
            usage_loss = (math.log(model.num_prototypes)
                          - (-(bavg * bavg.clamp(min=1e-8).log()).sum())) / math.log(model.num_prototypes)

            loss = (args.w_state * pb + args.w_yao * yao_loss + args.w_proto * proto_loss
                    + args.w_commit * commit_loss + args.w_usage * usage_loss
                    + args.w_mono * mono)
            # T17 修补：闭环 pinball rollout 损失（分位数带在自生成反馈下保持校准）
            if args.w_roll > 0:
                xf = b["x_fut"].to(device, non_blocking=True)
                ss = min(args.ss_max, 0.05 * (epoch + 1))
                roll_steps, _ = model.rollout_q(xb, afb, x_fut_gt=xf, ss_prob=ss)
                roll = sum(
                    (0.7 ** k) * pinball_loss(
                        xf[:, k, :OBJ_DIMS].float().clamp(-50, 50),
                        roll_steps[k].clamp(-50, 50), taus)
                    for k in range(K)) / K
                loss = loss + args.w_roll * roll
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if any(not torch.isfinite(p.grad.sum()) for p in model.parameters() if p.grad is not None):
                optimizer.zero_grad()
                continue
            optimizer.step()
            model.ema_prototypes(c.detach().float(), fine.argmax(-1).detach())
            for k, v in dict(pb=pb.item(), mono=mono.item(), yao=yao_loss.item()).items():
                agg[k] = agg.get(k, 0.0) + v
        sched.step()
        msg = " ".join(f"{k}={v / max(nb, 1):.4f}" for k, v in agg.items())
        print(f"Epoch {epoch + 1}/{args.epochs} [{msg}]")

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            m = evaluate(model, base, val_ds, device, K, 20000, taus, POS_IDX, OBJ_DIMS,
                         noise_aug=args.noise_aug)
            print(f"  [val] 校准: [0.1,0.9]覆盖 {m['cov80_k1']:.3f}(目标0.8±0.06) | "
                  f"[0.25,0.75]覆盖 {m['cov50_k1']:.3f}(目标0.5±0.08)")
            print(f"  [val] 锋锐: q0.5 点预测 k=1 {m['med_k1']:.3f} / k=5 {m['med_k5']:.3f} px"
                  f" | 单点基线 k=1 {m['base_k1']:.3f} px")
            print(f"  [val] 带宽(k=1→{K}): " + " ".join(f"{m[f'band_k{k+1}']:.1f}" for k in range(K))
                  + " px（应随 k 单调增长，T11 响应）")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(dict(model_state_dict=model.state_dict(),
                             config=dict(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS,
                                         action_dim=ACTION_DIM, hidden_dim=args.hidden_dim,
                                         num_prototypes=args.num_prototypes),
                             norm_stats=dict(mean_obj=mean_obj, std_obj=std_obj),
                             args=vars(args), epoch=epoch, val_metrics=m),
                       os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"))


@torch.no_grad()
def evaluate(model, base, ds, device, K, max_windows, taus, pos_idx, obj_dim, noise_aug=0.0):
    model.eval(); base.eval()
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rs = np.random.RandomState(0)
        idx_all = np.sort(rs.choice(idx_all, size=max_windows, replace=False))
    M = len(idx_all); bs = 2048
    n_q = model.n_q
    cov80 = []; cov50 = []
    med_err = np.zeros((M, K)); band = np.zeros((M, K))
    base_err = np.zeros((M, K))
    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = b["x"].to(device)
        afb = b["af"].to(device)
        xf = b["x_fut"].to(device)
        if noise_aug > 0:
            xb = xb + torch.randn_like(xb) * noise_aug
        h, _ = model.encode_sequence(xb)
        prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], b["a0"].to(device))
        q_res = model.predict_q(h, b["a0"].to(device))
        q_abs = (prior_n.unsqueeze(1) + q_res).float().cpu().numpy()   # (B,n_q,22)
        tgt = xf[:, :, :obj_dim].float().cpu().numpy()                  # (B,K,22)
        # 单步校准（k=0）
        lo, hi = 0, model.n_q - 1
        e_lo = tgt[:, 0, pos_idx] - q_abs[:, lo, pos_idx]
        e_hi = q_abs[:, hi, pos_idx] - tgt[:, 0, pos_idx]
        cov80.append(((e_lo <= 0) & (e_hi >= 0)).mean())
        lo, hi = 1, model.n_q - 2
        e_lo = tgt[:, 0, pos_idx] - q_abs[:, lo, pos_idx]
        e_hi = q_abs[:, hi, pos_idx] - tgt[:, 0, pos_idx]
        cov50.append(((e_lo <= 0) & (e_hi >= 0)).mean())
        # 闭环：每 k 的分位数带与中位数误差
        steps, _ = model.rollout_q(xb, afb)
        for k in range(K):
            raw = steps[k].float().cpu().numpy()                        # (B,n_q,22)
            med = raw[:, model.q_idx_med, :]
            med_err[sl, k] = np.abs(med[:, pos_idx] - tgt[:, k, pos_idx]).mean(axis=1)
            band[sl, k] = (raw[:, model.n_q - 1, pos_idx] - raw[:, 0, pos_idx]).mean(axis=1)
        bsteps, _ = base.rollout(xb, afb)
        for k in range(K):
            raw_b = bsteps[k].float().cpu().numpy()
            base_err[sl, k] = np.abs(raw_b[:, pos_idx] - tgt[:, k, pos_idx]).mean(axis=1)

    pos_std = float(np.mean(ds.std_obj[POS_IDX]))
    out = {"pos_std_px": pos_std}
    out["cov80_k1"] = float(np.mean(cov80)); out["cov50_k1"] = float(np.mean(cov50))
    for k in range(K):
        out[f"med_k{k+1}"] = float(np.sqrt((med_err[:, k]**2).mean())) * pos_std
        out[f"band_k{k+1}"] = float(band[:, k].mean()) * pos_std
        out[f"base_k{k+1}"] = float(np.sqrt((base_err[:, k]**2).mean())) * pos_std
    return out


if __name__ == "__main__":
    main()
