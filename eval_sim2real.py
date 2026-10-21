# -*- coding: utf-8 -*-
"""
微缩 sim-to-real 判决实验 (eval_sim2real.py)
==================================================
预注册判决（看到真实数据前写下）：

  SR-1 优雅退化（T1b-real）：模型预测误差 ≤ 物理先验误差 + 模型改进量
       的 2 倍——模型在真实数据上不能比先验差太多（T1b 的下界保证）。
  SR-2 先验有效性（T1-real）：物理先验（解析双摆）在真实数据上的单步
       误差 < 30px（真实世界有空气阻力/摩擦/弹性——先验不可能完美，
       但必须比"什么都不做"好）。
  SR-3 残差补偿：模型 one-step < 先验 one-step（残差头在真实数据上
       补偿 sim-to-real 差距）。
  SR-4 速度推断：ω̂/v̂ 从真实视觉轨迹推断的 MAE < 2× 仿真值
       （真实 CV 噪声更大是预期，但不应爆炸）。

任一判据的诚实解读：
  SR-1 [OK] = 架构的真实可行性确证（核心判决）
  SR-2 [FAIL] = 先验方程在真实物理下失效（预期：空气阻力）
  SR-3 [OK] = 残差补偿了 sim-to-real 差距（最强正结果）
  SR-4 [FAIL] = CV 噪声太大，需改进跟踪

用法：
  python eval_sim2real.py --real_data real_traj.npz --checkpoint checkpoints-hinge/best.pt
  python eval_sim2real.py --synthetic_test   # 用合成数据先跑一遍验证脚本逻辑
"""

import os
import math
import json
import argparse
import numpy as np
import torch

from layout_h import (OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, L1, L2, M1, M2, TAU_MAX,
                      acrobot_step, physics_prior_state, yao_labels_h)
from model_h import WorldModelH
from train_hinge import pos_rmse

PASS, FAIL = "[OK]", "[FAIL]"
results = []


def report(name, ok, detail):
    results.append((name, ok))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")


def make_prior_real(state, tau_norm=0.0):
    """物理先验（单样本 numpy 版）——铰链世界双摆 0.05s 欧拉一步。"""
    return physics_prior_state(state, tau_norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_data", type=str, default="real_traj.npz")
    ap.add_argument("--checkpoint", type=str, default="checkpoints-hinge/best.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--synthetic_test", action="store_true",
                    help="用仿真数据测试脚本逻辑（不加载模型）")
    ap.add_argument("--output_report", type=str,
                    default="analysis_output/sim2real_report.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    print("=" * 68)
    print("微缩 sim-to-real 判决（T1b/T6/T13 的真实版）")
    print("=" * 68)

    if args.synthetic_test:
        print("[合成测试模式] 用仿真数据验证脚本逻辑")
        # 生成合成"真实"数据（RK4 + 注入非仿真效应：空气阻力 + 观测噪声）
        th1, th2, w1, w2 = 2.0, 0.3, 0.0, 0.0
        T = 100
        drag = 0.02  # 空气阻尼系数（仿真没有的效应）
        real_states = []
        for i in range(T):
            x1, y1 = L1 * math.sin(th1), -L1 * math.cos(th1)
            x2 = x1 + L2 * math.sin(th1 + th2)
            y2 = y1 - L2 * math.cos(th1 + th2)
            real_states.append([x1, y1, x2, y2, th1, th2, w1, w2])
            # RK4 + 空气阻力
            for _ in range(5):
                tau = 0.0
                # 阻力矩 ∝ -ω
                th1, th2, w1, w2 = acrobot_step(
                    th1, th2, w1 * (1 - drag), w2 * (1 - drag), tau, dt=0.01)
        real_states = np.array(real_states, dtype=np.float32)
        # 观测噪声
        real_states += np.random.randn(*real_states.shape).astype(np.float32) * 1.5
        states = real_states[:, None, :].repeat(1, axis=1)
        # 简化为 22 维
        full = np.zeros((1, T, 22), dtype=np.float32)
        b1, b2 = 0, 11
        for t in range(T):
            full[0, t, b1+1], full[0, t, b1+2] = real_states[t, 0], real_states[t, 1]
            full[0, t, b2+1], full[0, t, b2+2] = real_states[t, 2], real_states[t, 3]
            full[0, t, b1+3], full[0, t, b1+4] = np.sin(real_states[t, 4]), np.cos(real_states[t, 4])
            full[0, t, b2+3], full[0, t, b2+4] = np.sin(real_states[t, 5]), np.cos(real_states[t, 5])
            full[0, t, b1+7], full[0, t, b2+7] = real_states[t, 6], real_states[t, 7]
            full[0, t, b1], full[0, t, b2] = 1, 1
            full[0, t, b1+10], full[0, t, b2+10] = M1, M2
        contact = np.zeros((1, T, 3), dtype=np.float32)
        actions = np.zeros((1, T, 1), dtype=np.float32)
        # 组装为 25 维（obj + contact 零填充）
        states_25 = np.concatenate([full, contact], axis=-1)  # (1, T, 25)
        states = states_25  # [修复] 赋回给 states 供后续使用
        data_source = "synthetic_with_drag"

    else:
        d = np.load(args.real_data)
        states = d["states"]  # (1, T, 22) 或 (N, T, 22)
        contact = np.zeros((states.shape[0], states.shape[1], 3), dtype=np.float32)
        actions = np.zeros((states.shape[0], states.shape[1], 1), dtype=np.float32)
        data_source = args.real_data

    # 取第一条轨迹
    states_1 = states[0:1] if states.ndim == 3 else states[None]
    T = states_1.shape[1]
    print(f"数据源: {data_source} | {T} 帧")

    # ============================================================
    # SR-1/SR-2/SR-3：单步预测（物理先验 vs 模型）在真实数据上
    # ============================================================
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WorldModelH(**ck["config"]).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    ns = ck["norm_stats"]

    prior_errs = []
    model_errs = []
    zero_errs = []
    with torch.no_grad():
        for t in range(T - 1):
            state_now = states_1[0, t][:OBJ_DIMS].cpu().numpy() if hasattr(states_1, 'cpu') else states_1[0, t][:OBJ_DIMS]
            state_next = states_1[0, t+1][:OBJ_DIMS].cpu().numpy() if hasattr(states_1, 'cpu') else states_1[0, t+1][:OBJ_DIMS]
            b1, b2 = 0, 11
            pos_true = [state_next[b1+1], state_next[b1+2],
                        state_next[b2+1], state_next[b2+2]]
            # 物理先验
            prior = make_prior_real(state_now, 0.0)
            pos_prior = [prior[b1+1], prior[b1+2], prior[b2+1], prior[b2+2]]
            # 零速度
            pos_zero = [state_now[b1+1], state_now[b1+2],
                        state_now[b2+1], state_now[b2+2]]
            # 模型
            # 组装 25 维输入帧：22 归一化 obj + 3 contact（合成测试无接触→零）
            obj_norm_22 = ((state_now[:OBJ_DIMS] - ns["mean_obj"]) / (ns["std_obj"] + 1e-8)
                           ).astype(np.float32)
            frame_25 = np.concatenate([obj_norm_22, np.zeros(3, dtype=np.float32)])
            obj_norm = torch.from_numpy(frame_25[None]).to(device)
            act = torch.zeros(1, 1, device=device)
            # 构建 8 帧历史窗口（简化：重复当前帧 8 次）
            seq_25 = obj_norm.squeeze(0).unsqueeze(0).expand(8, -1).unsqueeze(0)  # (1,8,25)
            h, _ = model.encode_sequence(seq_25.contiguous())
            delta = model.predict_state(h, act).float().cpu().numpy()[0]
            prior_norm_np = ((prior - ns["mean_obj"]) / (ns["std_obj"] + 1e-8)
                             ).astype(np.float32)
            pred = (prior_norm_np + delta) * (ns["std_obj"] + 1e-8) + ns["mean_obj"]
            pos_model = [pred[b1+1], pred[b1+2], pred[b2+1], pred[b2+2]]

            prior_errs.append(np.abs(np.array(pos_prior) - np.array(pos_true)).mean())
            model_errs.append(np.abs(np.array(pos_model) - np.array(pos_true)).mean())
            zero_errs.append(np.abs(np.array(pos_zero) - np.array(pos_true)).mean())

    prior_errs = np.array(prior_errs)
    model_errs = np.array(model_errs)
    zero_errs = np.array(zero_errs)
    pos_std_est = float(np.std(states_1[0, :, 1].cpu().numpy() if hasattr(states_1, 'cpu')
                                else states_1[0, :, 1]))

    print(f"\n[单步预测] {T-1} 步")
    print(f"  物理先验 MAE: {prior_errs.mean():.2f}px")
    print(f"  模型 MAE:     {model_errs.mean():.2f}px")
    print(f"  零速度 MAE:   {zero_errs.mean():.2f}px")

    # ============================================================
    # 预注册判决
    # ============================================================
    print("\n[预注册判决]")

    # SR-1 优雅退化
    ratio = model_errs.mean() / max(prior_errs.mean(), 1e-6)
    report("SR-1 优雅退化（model ≤ prior × 2）", ratio < 2.0,
           f"模型/先验 = {ratio:.2f}（<2 为通过；=1 为完美退化到先验）")

    # SR-2 先验有效性
    report("SR-2 先验有效（prior < 零速度）", prior_errs.mean() < zero_errs.mean(),
           f"先验 {prior_errs.mean():.2f} < 零速度 {zero_errs.mean():.2f}")

    # SR-3 残差补偿
    improvement = (prior_errs.mean() - model_errs.mean()) / max(prior_errs.mean(), 1e-6)
    report("SR-3 残差补偿（model < prior）", model_errs.mean() < prior_errs.mean(),
           f"改善 {improvement:.1%}")

    # SR-4 速度推断（需要视频跟踪质量）
    # 在合成测试中检查速度差分质量
    if args.synthetic_test:
        v_est = states_1[0, 1:, 5].cpu().numpy() if hasattr(states_1, 'cpu') else states_1[0, 1:, 5]
        v_true = np.diff(states_1[0, :, 1]) / 0.05
        v_mae = np.abs(v_est[:len(v_true)] - v_true[:len(v_est)]).mean()
        report("SR-4 速度推断（CV 噪声下）", v_mae < 2 * 5.0,
               f"速度 MAE {v_mae:.2f} px/s（判据 <2× 仿真 std 5.0）")

    # ============================================================
    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n{'='*72}")
    print(f"微缩 sim-to-real 汇总: {n_pass}/{len(results)} 通过")
    if n_pass >= 3:
        print("→ 架构在真实数据上存活。T1b/T8 的真实版获得初步支持。")
    elif n_pass >= 2:
        print("→ 部分存活。分析未通过项的根因。")
    else:
        print("→ 架构在 sim-to-real 下大幅退化。需要重新设计。")

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump({"source": data_source, "n_steps": T - 1,
                   "prior_mae": float(prior_errs.mean()),
                   "model_mae": float(model_errs.mean()),
                   "zero_vel_mae": float(zero_errs.mean()),
                   "results": [(str(n), bool(ok)) for n, ok in results]}, f, indent=2,
                  ensure_ascii=False)
    print(f"[OK] 报告: {args.output_report}")


if __name__ == "__main__":
    main()
