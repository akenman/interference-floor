# -*- coding: utf-8 -*-
"""
G3 部分可观测实验 (train_hp.py) — 周易认识论融合
==================================================
遮蔽梯度（真实机器人观测的分级模拟）：
  L0 全观测（对照，已知 0.667px）
  L1 遮蔽速度（vx,vy,ω → 0）：真实编码器给位置不给速度
  L2 遮蔽速度+接触输入：接触传感器缺失，接触状态须由历史推断
  L3 L2 + 位置噪声 0.02：完整真实观测模拟

预注册预言：
  P3.1 速度推断可行：ω̂ MAE < 1.0 rad/s；L1 模型 one-step < 先验（先验在遮蔽下退化）
  P3.2 信念（内卦）准确率随遮蔽深度优雅退化（L2 ≤ L1 ≤ L0）
  P3.3 知幾：信念准确率 > 多数类基线 ≥5pp（模型确实从历史学到了隐藏状态）
  P3.4 内卦条件化增益：FiLM(信念) 模型优于无条件化模型（T10 POMDP 重赛）

用法：python train_hp.py --mask_tier 1 --epochs 30
"""

import os
import sys
import math
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Windows PowerShell 重定向 stdout 为 UTF-16，且控制台代码页非 UTF-8；
# 强制 UTF-8 避免 Epoch 结束时的 print 把 UnicodeEncodeError 当训练崩溃
if sys.stdout.encoding.upper() not in ("UTF-8", "UTF8"):
    sys.stdout.reconfigure(encoding="utf-8")

from layout_h import OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, feat_weights as feat_weights_h
from model_hp import HingePOMDPModel, VEL_DIMS
from train_hinge import HingeData, pos_rmse

INPUT_DIM = OBJ_DIMS + CONTACT_DIMS
POS_IDX = [1, 2, 12, 13]
CONTACT_IN_DIMS = [22, 23, 24]   # 组装帧中的接触输入通道


def apply_mask(xb, tier):
    """L1: 零化速度维；L2: +零化接触输入；L3: +位置噪声。xb: (B, L, 25)。"""
    if tier >= 1:
        for d in VEL_DIMS:
            xb[..., d] = 0.0
    if tier >= 2:
        for d in CONTACT_IN_DIMS:
            xb[..., d] = 0.0
    if tier >= 3:
        xb = xb + torch.randn_like(xb) * 0.02
    return xb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--mask_tier", type=int, default=1)
    ap.add_argument("--use_belief_film", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--future_steps", type=int, default=5)
    ap.add_argument("--num_prototypes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w_yao", type=float, default=0.1)
    ap.add_argument("--w_proto", type=float, default=0.05)
    ap.add_argument("--w_usage", type=float, default=0.02)
    ap.add_argument("--w_commit", type=float, default=0.01)
    ap.add_argument("--windows_per_epoch", type=int, default=300000)
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints-hp")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_amp", action="store_true",
                    help="禁用 bf16 自动混合精度（与 train_hinge/train_e0/train_e1a 一致）")
    args = ap.parse_args()
    device = torch.device(args.device)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    L, K = args.seq_len, args.future_steps

    print("=" * 68)
    print(f"G3 部分可观测实验：遮蔽层级 L{args.mask_tier} | 内卦条件化 {bool(args.use_belief_film)}")
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

    model = HingePOMDPModel(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=ACTION_DIM,
                            hidden_dim=args.hidden_dim, num_prototypes=args.num_prototypes,
                            use_belief_film=bool(args.use_belief_film)).to(device)
    model.set_stats(mean_obj, std_obj)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    fw = feat_weights_h(OBJ_DIMS)
    feat_w_t = torch.from_numpy(fw).to(device)

    # 多数类基线（信念判据 P3.3 用）：初爻恒 0 或恒 1 的最大类频率
    yao_majority = float(max((yao[:, :, 0, :] == 0).mean(),
                             (yao[:, :, 0, :] == 1).mean()))

    n_train = len(train_ds)
    wpe = args.windows_per_epoch if args.windows_per_epoch > 0 else n_train
    best = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tag = f"L{args.mask_tier}{'F' if args.use_belief_film else 'N'}"
    ckpt_dir = os.path.join(args.checkpoint_dir, tag)

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_train)[:wpe]
        agg, nb = {}, 0
        for j in range(0, wpe, args.batch_size):
            idx = perm[j:j + args.batch_size]
            b = train_ds.gather(idx)
            xb = apply_mask(b["x"].to(device), args.mask_tier)
            xb2 = apply_mask(b["x_next"].to(device), args.mask_tier)
            ab = b["a0"].to(device, non_blocking=True)
            db = b["d_norm"].to(device, non_blocking=True)
            yaob = b["yao"].to(device, non_blocking=True)

            h, c = model.encode_sequence(xb)
            fine = model.evidence(c)
            belief_log = model.predict_belief(h)
            # 速度估计（真值监督：未被遮蔽的目标中的 ω）
            # 量纲修正：b["x"] 在归一化空间，estimate_omega / prior_norm 工作在 raw rad/s；
            # 必须反归一化，否则速度头学的是无量纲数，注入先验后等价于零速度。
            _s7, _s18 = train_ds.std_obj[7], train_ds.std_obj[18]
            _m7, _m18 = train_ds.mean_obj[7], train_ds.mean_obj[18]
            w_true_np = np.stack([b["x"][:, -1, 7] * _s7 + _m7,
                                  b["x"][:, -1, 18] * _s18 + _m18],
                                 axis=-1).astype(np.float32)
            w_true = torch.from_numpy(w_true_np).to(device)
            w_hat = model.estimate_omega(h)
            vel_var = float(np.mean(np.array([train_ds.std_obj[d] for d in VEL_DIMS]) ** 2))
            vel_loss = F.mse_loss(w_hat, w_true) / vel_var   # 按数据集速度方差归一化，保持量级
            prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab, w_hat=w_hat)
            tgt_abs = prior_n + db
            delta = model.predict_state(h, ab, belief_log)
            state_loss = (F.huber_loss(delta, db, reduction="none") * feat_w_t.view(1, -1)).mean()

            yao_loss = F.binary_cross_entropy_with_logits(belief_log, yaob.float(), reduction="mean")
            _, c_next = model.encode_sequence(apply_mask(b["x_next"].to(device), args.mask_tier))
            fine_next = model.evidence(c_next)
            pfine = model.predict_proto(h, ab)
            proto_loss = F.kl_div(F.log_softmax(pfine, dim=-1),
                                  F.softmax(fine_next, dim=-1), reduction="batchmean")
            assigned = model.prototypes.detach()[fine.argmax(dim=-1)]
            commit_loss = (1.0 - F.cosine_similarity(assigned, c, dim=-1)).mean()
            assignment = F.softmax(fine.float(), dim=-1)
            bavg = assignment.mean(0).clamp(min=1e-8)
            usage_loss = (math.log(model.num_prototypes)
                          - (-(bavg * bavg.clamp(min=1e-8).log()).sum())) / math.log(model.num_prototypes)

            loss = (1.0 * state_loss + args.w_yao * yao_loss + args.w_proto * proto_loss
                    + 0.01 * commit_loss + args.w_usage * usage_loss
                    + 0.3 * vel_loss)
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
            for k, v in dict(state=state_loss.item(), yao=yao_loss.item(),
                             vel=vel_loss.item()).items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        sched.step()
        msg = " ".join(f"{k}={v / max(nb, 1):.4f}" for k, v in agg.items())
        print(f"Epoch {epoch + 1}/{args.epochs} [{msg}]")

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            m = evaluate(model, val_ds, device, args.mask_tier, K, 20000, yao_majority)
            print(f"  [val] one-step: 模型 {m['model']:.3f} | 先验(ω̂) {m['prior']:.3f}"
                  f" | 先验(零速) {m['prior_zerovel']:.3f} | 零速度 {m['zero_vel']:.3f} px")
            print(f"  [val] ω̂ 推断 MAE: {m['vel_mae']:.3f} rad/s | 内卦信念 acc: {m['belief_acc']:.3f}"
                  f"（多数类 {yao_majority:.3f}）")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(dict(model_state_dict=model.state_dict(),
                             config=dict(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS,
                                         action_dim=ACTION_DIM, hidden_dim=args.hidden_dim,
                                         num_prototypes=args.num_prototypes,
                                         use_belief_film=bool(args.use_belief_film)),
                             norm_stats=dict(mean_obj=mean_obj, std_obj=std_obj),
                             args=vars(args), epoch=epoch, val_metrics=m),
                       os.path.join(ckpt_dir, "checkpoint_latest.pt"))
            if m["model"] < best:
                best = m["model"]
                torch.save(dict(model_state_dict=model.state_dict(),
                                 config=dict(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS,
                                             action_dim=ACTION_DIM, hidden_dim=args.hidden_dim,
                                             num_prototypes=args.num_prototypes,
                                             use_belief_film=bool(args.use_belief_film)),
                                 norm_stats=dict(mean_obj=mean_obj, std_obj=std_obj),
                                 args=vars(args), epoch=epoch, val_metrics=m),
                           os.path.join(ckpt_dir, "best.pt"))
                print(f"  [保存最佳] {best:.3f}px")

    print(f"[OK] 完成。最佳 {best:.3f}px | 报告见 {ckpt_dir}")


@torch.no_grad()
def evaluate(model, ds, device, tier, K, max_windows, yao_majority):
    model.eval()
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rs = np.random.RandomState(0)
        idx_all = np.sort(rs.choice(idx_all, size=max_windows, replace=False))
    M = len(idx_all); bs = 2048
    pred_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    prior_est = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    prior_zv = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    zero_vel = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    vel_err, belief_hits, belief_n = 0.0, 0, 0
    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = apply_mask(b["x"].clone().to(device), tier)
        ab = b["a0"].to(device)
        h, _ = model.encode_sequence(xb)
        w_hat = model.estimate_omega(h)
        # 量纲修正（同上）：b["x"] 归一化 → raw rad/s
        _s7, _s18 = ds.std_obj[7], ds.std_obj[18]
        _m7, _m18 = ds.mean_obj[7], ds.mean_obj[18]
        w_true_np = np.stack([b["x"][:, -1, 7] * _s7 + _m7,
                              b["x"][:, -1, 18] * _s18 + _m18],
                             axis=-1).astype(np.float32)
        w_true = torch.from_numpy(w_true_np).to(device)
        vel_err += float((w_hat - w_true).abs().mean()) * len(idx)
        belief_log = model.predict_belief(h)
        bp = (torch.sigmoid(belief_log.float()) > 0.5).cpu().numpy()
        yt = b["yao"].numpy()
        belief_hits += (bp == yt).mean() * len(idx)
        belief_n += len(idx)
        delta = model.predict_state(h, ab, belief_log).float().cpu().numpy()
        p_est = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab, w_hat=w_hat).float().cpu().numpy()
        p_zv = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab, w_hat=torch.zeros_like(w_hat)).float().cpu().numpy()
        std, mean = ds.std_obj + 1e-8, ds.mean_obj
        pred_one[sl] = (p_est + delta) * std + mean
        prior_est[sl] = p_est * std + mean
        prior_zv[sl] = p_zv * std + mean
        zero_vel[sl] = b["s_last"].float().numpy()
    frm_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L]
    tgt = frm_last * (ds.std_obj + 1e-8) + ds.mean_obj
    return {"model": pos_rmse(pred_one, tgt),
            "prior": pos_rmse(prior_est, tgt),
            "prior_zerovel": pos_rmse(prior_zv, tgt),
            "zero_vel": pos_rmse(zero_vel, tgt),
            "vel_mae": vel_err / max(M, 1),
            "belief_acc": belief_hits / max(belief_n, 1)}


if __name__ == "__main__":
    main()
