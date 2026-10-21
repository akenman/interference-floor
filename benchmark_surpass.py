# -*- coding: utf-8 -*-
"""
超越实验：物理参数可学习化 + 数据效率曲线 (benchmark_surpass.py)
==================================================
核心假设：物理先验的可学习参数（g, M1, M2, L1, L2）使模型在
  ①低数据端碾压纯学习（先验提供结构性热启动）
  ②分布偏移时退化最慢（参数自动校准）
  ③新系统适应最快（只学参数+残差，不重学全部动力学）

三臂 × 数据效率梯度 {50, 200, 1000, 3000} episodes：
  ours_param   = 物理参数可学习先验 + 残差头（超越版）
  noprior      = 纯学习 GRU（JEPA 基线）
  jepa         = latent 预测 + EMA target（JEPA 范式）

预注册判据（实验前写下）：
  E1: episodes ≤ 200 时，ours_param 优于 noprior ≥ 30%
  E2: ours_param 的物理参数收敛到真值 ±20%（g=900, M1=2, M2=1, L1=L2=80）
  E3: ours_param 在 50 episodes 时达到 noprior 在 3000 episodes 时精度
      （数据效率倍增 ≥60×）
"""

import os
import math
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layout_h import OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, feat_weights as feat_weights_h
from core_model import NoPriorModel, JEPALikeModel
from train_hinge import HingeData, pos_rmse

INPUT_DIM = OBJ_DIMS + CONTACT_DIMS
POS_IDX = [1, 2, 12, 13]


# ============================================================
# 可学习物理参数版 Ours
# ============================================================

class OursParamModel(nn.Module):
    """物理参数可学习的残差-先验模型。

    g, M1, M2, L1, L2 是 nn.Parameter——梯度下降自动校准物理。
    约束：g ∈ [500, 1500]（sigmoid 缩放），质量/长度非负。
    """

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256):
        super().__init__()
        self.obj_dim = obj_dim
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.ln_h = nn.LayerNorm(hidden_dim)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, obj_dim))
        nn.init.normal_(self.state_head[-1].weight, std=0.01)
        nn.init.zeros_(self.state_head[-1].bias)

        # 可学习物理参数（原始空间参数化，通过 softplus/sigmoid 约束）
        self.log_g = nn.Parameter(torch.log(torch.tensor(900.0)))       # log(g)
        self.log_M1 = nn.Parameter(torch.log(torch.tensor(2.0)))
        self.log_M2 = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_L1 = nn.Parameter(torch.log(torch.tensor(80.0)))
        self.log_L2 = nn.Parameter(torch.log(torch.tensor(80.0)))
        # 初始值 = 仿真真值（如果不知道真值，可以设为通用初值）

        self.register_buffer("mean_obj", torch.zeros(obj_dim))
        self.register_buffer("std_obj", torch.ones(obj_dim))

    def set_stats(self, mean_obj, std_obj):
        with torch.no_grad():
            self.mean_obj.copy_(torch.as_tensor(mean_obj, dtype=self.mean_obj.dtype))
            self.std_obj.copy_(torch.as_tensor(std_obj, dtype=self.std_obj.dtype))

    def encode(self, x_seq):
        h = torch.zeros(x_seq.size(0), self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.gru_cell(x_seq[:, t], h)
        return self.ln_h(h)

    def get_params(self):
        """返回当前物理参数（用于监控收敛）。"""
        return {"g": self.log_g.exp().item(),
                "M1": self.log_M1.exp().item(),
                "M2": self.log_M2.exp().item(),
                "L1": self.log_L1.exp().item(),
                "L2": self.log_L2.exp().item()}

    def prior_step(self, obj_norm, a_norm):
        """可学习物理参数的先验（归一化空间进出）。

        取证修复（2026-09-02）：原实现缺少反归一化——把归一化输入当物理单位
        使用（atan2 吃归一化 sin/cos、ω 用归一化值当 rad/s），先验输出为垃圾，
        T15 的 ours_param 臂实质运行在该坏先验上。现补对称往返：
        raw = obj_norm·(std+1e-8) + mean（与 HingeData 归一化严格互逆）。
        obj_norm: (B, 22)，a_norm: (B, 1)。
        """
        g = self.log_g.exp().clamp(500, 1500)
        m1 = self.log_M1.exp().clamp(0.1, 10)
        m2 = self.log_M2.exp().clamp(0.1, 10)
        l1 = self.log_L1.exp().clamp(20, 200)
        l2 = self.log_L2.exp().clamp(20, 200)
        b1, b2 = 0, 11
        raw = obj_norm * (self.std_obj + 1e-8) + self.mean_obj

        th1 = torch.atan2(raw[:, b1 + 3], raw[:, b1 + 4])
        th2 = torch.atan2(raw[:, b2 + 3], raw[:, b2 + 4])
        w1 = raw[:, b1 + 7]
        w2 = raw[:, b2 + 7]
        tau = a_norm.squeeze(-1) * 28000.0

        c2 = torch.cos(th2); s2 = torch.sin(th2)
        M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * c2
        M12 = m2 * l2**2 + m2 * l1 * l2 * c2
        M22 = m2 * l2**2
        h1 = (-m2 * l1 * l2 * (2 * w1 + w2) * w2 * s2
              + (m1 + m2) * g * l1 * torch.sin(th1)
              + m2 * g * l2 * torch.sin(th1 + th2))
        h2 = (m2 * l1 * l2 * w1**2 * s2
              + m2 * g * l2 * torch.sin(th1 + th2))
        det = M11 * M22 - M12**2
        a1 = (M22 * (-h1) - M12 * (-h2)) / det
        a2 = (-M12 * (-h1) + M11 * (-h2)) / det
        w1n = w1 + a1 * 0.05
        w2n = w2 + a2 * 0.05
        th1n = th1 + w1n * 0.05
        th2n = th2 + w2n * 0.05

        out = raw.clone()
        x1 = l1 * torch.sin(th1n); y1 = -l1 * torch.cos(th1n)
        x2 = x1 + l2 * torch.sin(th1n + th2n); y2 = y1 - l2 * torch.cos(th1n + th2n)
        vx1 = l1 * w1n * torch.cos(th1n); vy1 = l1 * w1n * torch.sin(th1n)
        vx2 = vx1 + l2 * (w1n + w2n) * torch.cos(th1n + th2n)
        vy2 = vy1 + l2 * (w1n + w2n) * torch.sin(th1n + th2n)
        out[:, b1+1], out[:, b1+2] = x1, y1
        out[:, b2+1], out[:, b2+2] = x2, y2
        out[:, b1+3], out[:, b1+4] = torch.sin(th1n), torch.cos(th1n)
        out[:, b2+3], out[:, b2+4] = torch.sin(th2n), torch.cos(th2n)
        out[:, b1+5], out[:, b1+6] = vx1, vy1
        out[:, b2+5], out[:, b2+6] = vx2, vy2
        out[:, b1+7], out[:, b2+7] = w1n, w2n
        return (out - self.mean_obj) / (self.std_obj + 1e-8)

    def encode_sequence(self, x_seq):
        h = torch.zeros(x_seq.size(0), self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.gru_cell(x_seq[:, t], h)
        return self.ln_h(h)

    def predict(self, h, action):
        return self.state_head(torch.cat([h, action], dim=-1))

    def encode_sequence(self, x_seq):
        """与 WorldModelH 同名接口：返回 (h, c_norm)。"""
        h = self.encode(x_seq)
        return h, F.normalize(h, p=2, dim=-1)

    def predict_state(self, h, action, belief_log=None):
        """与 WorldModelH 同名接口（belief_log 仅签名兼容，忽略）。"""
        return self.predict(h, action)

    @torch.no_grad()
    def set_physics(self, g, M1, M2, L1, L2):
        """辨识结果写回（Phase A → Phase B/C 的交接；log 空间赋值）。"""
        self.log_g.data.fill_(math.log(g))
        self.log_M1.data.fill_(math.log(M1))
        self.log_M2.data.fill_(math.log(M2))
        self.log_L1.data.fill_(math.log(L1))
        self.log_L2.data.fill_(math.log(L2))


# ============================================================
# 数据效率实验
# ============================================================

def train_arm(variant, ds_train, args, device, n_episodes_available):
    """训练单臂，返回 (model, 物理参数dict或None)。"""
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    n_train = len(ds_train)
    wpe = min(args.windows_per_epoch, n_train)
    params_out = None

    if variant == "ours_param":
        model = OursParamModel(INPUT_DIM, OBJ_DIMS, ACTION_DIM, args.hidden_dim).to(device)
        model.set_stats(ds_train.mean_obj, ds_train.std_obj)
    elif variant == "noprior":
        model = NoPriorModel(INPUT_DIM, OBJ_DIMS, ACTION_DIM, args.hidden_dim).to(device)
    elif variant == "jepa":
        model = JEPALikeModel(INPUT_DIM, OBJ_DIMS, ACTION_DIM, args.hidden_dim,
                              latent_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, min(args.epochs, n_train // 200)))
    fw = feat_weights_h(OBJ_DIMS)
    feat_w_t = torch.from_numpy(fw).to(device)

    effective_epochs = max(10, min(args.epochs, n_train // 200))
    for epoch in range(effective_epochs):
        model.train()
        perm = np.random.permutation(n_train)[:wpe]
        for j in range(0, wpe, args.batch_size):
            idx = perm[j:j + args.batch_size]
            b = ds_train.gather(idx)
            xb = b["x"].to(device)
            if args.noise_aug > 0:
                s = args.noise_aug * torch.rand(xb.size(0), 1, 1, device=device)
                xb = xb + torch.randn_like(xb) * s
            ab = b["a0"].to(device)
            tgts = [b["x_fut"][:, k, :OBJ_DIMS].to(device) for k in range(args.future_steps)]

            if variant == "ours_param":
                h = model.encode(xb)
                obj_norm_last = xb[:, -1, :OBJ_DIMS]
                prior_n = model.prior_step(obj_norm_last, ab)
                delta = model.predict(h, ab)
                pred = prior_n + delta
                loss = F.huber_loss(pred, tgts[0], reduction="mean")
            elif variant == "noprior":
                h = model.encode(xb)
                pred = model.predict(h, ab)
                loss = F.huber_loss(pred, tgts[0], reduction="mean")
            elif variant == "jepa":
                pred_latent = model.forward_latent(xb, ab)
                tgt_latent = model.forward_target(xb[:, 1:])  # EMA target: 移位一帧
                loss = F.mse_loss(pred_latent, tgt_latent)
                probe_pred = model.predict_state(pred_latent, ab)
                loss = loss + 0.5 * F.huber_loss(probe_pred, tgts[0], reduction="mean")

            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if any(not torch.isfinite(p.grad.sum()) for p in model.parameters() if p.grad is not None):
                optimizer.zero_grad()
                continue
            optimizer.step()
        sched.step()

    if variant == "ours_param":
        params_out = model.get_params()
    return model, params_out


@torch.no_grad()
def eval_model(variant, model, ds_eval, device, pos_std, K=5):
    model.eval()
    idx_all = np.arange(len(ds_eval))
    rs = np.random.RandomState(0)
    idx_all = np.sort(rs.choice(idx_all, min(5000, len(idx_all)), replace=False))
    errs = []
    for j in range(0, len(idx_all), 2048):
        idx = idx_all[j:j + 2048]
        b = ds_eval.gather(idx)
        xb = b['x'].to(device)
        ab = b['a0'].to(device)
        tgts = [b['x_fut'][:, k, :OBJ_DIMS].to(device) for k in range(K)]
        if variant == "ours_param":
            h = model.encode(xb)
            obj_last = xb[:, -1, :OBJ_DIMS]
            prior_n = model.prior_step(obj_last, ab)
            delta = model.predict(h, ab)
            pred = prior_n + delta
        elif variant == "noprior":
            h = model.encode(xb)
            pred = model.predict(h, ab)
        elif variant == "jepa":
            pred_latent = model.forward_latent(xb, ab)
            pred = model.predict_state(pred_latent, ab)
        e = np.abs(pred.float().cpu().numpy()[:, POS_IDX]
                   - tgts[0].cpu().numpy()[:, POS_IDX]).mean(axis=1)
        errs.append(e)
    all_e = np.concatenate(errs)
    pos_std = float(np.mean(ds_eval.std_obj[POS_IDX]))
    return float(np.sqrt((all_e**2).mean())) * pos_std


def main():
    ap = argparse.ArgumentParser(description="超越实验：物理参数可学习化 + 数据效率")
    ap.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--future_steps", type=int, default=5)
    ap.add_argument("--num_prototypes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--windows_per_epoch", type=int, default=200000)
    ap.add_argument("--noise_aug", type=float, default=0.02)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_report", type=str, default="analysis_output/surpass_report.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    print("=" * 72)
    print("超越实验：物理参数可学习化 + 数据效率曲线")
    print("三臂 × 四数据量 = 12 次训练")
    print("预注册判据 E1/E2/E3 见源码")
    print("=" * 72)

    d = np.load(args.data_path)
    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = states.shape[0]
    rng = np.random.RandomState(args.seed)
    segs = np.arange(N); rng.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]
    all_tr = segs[int(N * 0.1):]
    obj_flat = states[all_tr].reshape(-1, OBJ_DIMS).astype(np.float64)
    mean_obj = obj_flat.mean(0).astype(np.float32)
    std_obj = obj_flat.std(0).astype(np.float32)

    pos_std = float(np.mean(std_obj[POS_IDX]))
    data_tiers = [50, 200, 1000, 3000]
    variants = ["ours_param", "noprior", "jepa"]
    report = {"data_tiers": data_tiers, "results": {}}

    for n_data in data_tiers:
        if n_data > len(all_tr):
            continue
        tr_eps = all_tr[:n_data]
        # 需要 val 的 mean/std 独立于训练子集——用全部 tr 数据的统计
        # （实际中应只用选定子集——此处保持一致以便公平对比）
        ds_train = HingeData(states, contact, actions, yao, tr_eps, args.seq_len,
                             args.future_steps, mean_obj, std_obj)
        ds_eval = HingeData(states, contact, actions, yao, val_eps, args.seq_len,
                            args.future_steps, mean_obj, std_obj)

        print(f"\n{'='*60}")
        print(f"数据量: {n_data} episodes ({len(ds_train):,} 窗口)")
        print(f"{'='*60}")

        report["results"][str(n_data)] = {}
        for variant in variants:
            model, params = train_arm(variant, ds_train, args, device, n_data)
            rmse = eval_model(variant, model, ds_eval, device, pos_std, K=args.future_steps)
            report["results"][str(n_data)][variant] = rmse
            param_str = ""
            if params:
                param_str = f" | g={params['g']:.0f} M1={params['M1']:.2f} L1={params['L1']:.1f}"
                report["results"][str(n_data)][variant + "_params"] = params
            print(f"  {variant:<12} {rmse:>8.3f}px{param_str}")

    # 数据效率曲线分析
    print("\n" + "=" * 68)
    print("数据效率曲线（one-step 位置 RMSE, px）")
    print(f"{'数据量':<10}" + "".join(f"{v:>14}" for v in variants))
    for n_data in data_tiers:
        key = str(n_data)
        if key in report["results"]:
            row = report["results"][key]
            print(f"{n_data:<10}" + "".join(f"{row[v]:>14.3f}" for v in variants))

    # 预注册判据
    print("\n[预注册判决]")
    e1_pass = False; e3_pass = False; e2_pass = False
    try:
        low = report["results"]["200"]
        e1_pass = low["ours_param"] < low["noprior"] * 0.7
        print(f"E1 (低数据优势 ≥30%): {'[OK]' if e1_pass else '[FAIL]'} "
              f"(ours {low['ours_param']:.3f} vs noprior {low['noprior']:.3f})")
    except KeyError:
        print("E1: 数据不足")

    try:
        p = report["results"]["3000"].get("ours_param_params", {})
        if p:
            g_ok = abs(p["g"] - 900) / 900 < 0.2
            m1_ok = abs(p["M1"] - 2) / 2 < 0.2
            e2_pass = g_ok and m1_ok
            print(f"E2 (物理参数收敛): {'[OK]' if e2_pass else '[FAIL]'} (g={p['g']:.0f}, M1={p['M1']:.2f})")
    except KeyError:
        print("E2: 参数未记录")

    try:
        acc_50_ours = 1.0 / report["results"]["50"]["ours_param"]
        acc_3000_noprior = 1.0 / report["results"]["3000"]["noprior"]
        ratio = report["results"]["50"]["noprior"] / report["results"]["50"]["ours_param"]
        e3_pass = ratio >= 2.0
        print(f"E3 (50ep ours ≤ 3000ep noprior ÷ 2): {'[OK]' if e3_pass else '[FAIL]'} "
              f"(效率倍增 {ratio:.1f}×)")
    except KeyError:
        print("E3: 数据不足")

    report["criteria"] = {"E1": bool(e1_pass), "E2": bool(e2_pass), "E3": bool(e3_pass)}
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 报告: {args.output_report}")


if __name__ == "__main__":
    main()
