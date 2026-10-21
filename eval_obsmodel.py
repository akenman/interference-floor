# -*- coding: utf-8 -*-
"""
观测模型显式化判决 (eval_obsmodel.py) — T19 遗留合流 T13/G3
==================================================
问题：T13 判决"速度观测是部署硬需求"的依据是 prior(ω̂差分)=12.8px vs
prior(真ω)=2.5px——但那个 ω̂ 只是朴素差分。T19 发现采样器-物理混叠
（细化先验类大幅恢复 g），同一逻辑问：**显式观测模型能否无学习地闭合
速度观测缺口？**

五臂（ω 估计器 → 注入先验 → one-step 预测误差 px，终点指标）：
  true_ω      真值角速度（上限参照，等价 L0）
  naive_diff  朴素差分 ω̂=(θ_t−θ_{t−1})/0.05（T13 病根臂）
  savgol      Savitzky-Golay 5 点 2 阶差分（经典去噪差分）
  ekf         5×0.01s 细化双摆动力学 EKF（4 维状态 [θ1,ω1,θ2,ω2]，
              预测=细化欧拉，更新=θ1/θ2 位置观测，Q/R 手调）
  gru         checkpoints-hp/L1F 学习头（T13 的 12.8px 参照臂）

预注册判决（实验前写下）：
  O-A (T13 病根复现): L1 下 naive_diff 先验 ≥ 2 × true_ω 先验
  O-B (SG 有效):      savgol 先验 < naive_diff 先验（L1 与 L3 双档成立）
  O-C (EKF 闭合):     ekf 先验 ≤ 1.5 × true_ω 先验（部署缺口闭合判据）
  O-D (EKF vs GRU):   ekf 先验 < gru 先验（无学习 ≥ 学习头——若过，
                      "速度观测硬需求"改写为"显式观测模型硬需求"）
  O-E (噪声鲁棒):     L3（+位置噪声 0.02）下 O-B/C/D 同向成立
  O-F (GRU 现状):     gru 先验 ≤ naive_diff 先验（T13 的学习头至少打过病根）

用法：
  python eval_obsmodel.py            # L1 协议（速度遮蔽，T13 语义）
  python eval_obsmodel.py --l0       # 旧协议（不遮蔽，仅用于与 2026-09-04 旧报告对拍）

 2026-09-16 修法记录（三个 bug，详见 E3_report.json）：
  (1) prior_eval 从未遮蔽速度维 → L1 语义落空，实际跑的是 L0。已加 mask_vel。
  (2) gru 臂用 encode_sequence 返回的最后一帧 h 广播到全段 → ω̂ 常数。
      已改为逐帧 gru_cell + ln_h 推进。
  (3) vel_head 训练靶是归一化的 b["x_next"]，输出归一化单位；prior_norm 期望
      原始 rad/s。已反归一化（×std+mean）后再注入。
"""

import os
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

from layout_h import OBJ_DIMS, ACTION_DIM, L1 as LH_L1, L2 as LH_L2, M1, M2, G as G_NEG, TAU_MAX
from model_hp import HingePOMDPModel, VEL_DIMS

L, K = 8, 5
POS = [1, 2, 12, 13]
N_EPS, N_WIN = 100, 60
NOISE_PX = {"L1": 0.0, "L3": 0.02}     # 归一化位置噪声（观测侧）
DT_SAMPLE = 0.05
DT_SUB = 0.01
N_SUB = 5
G_PRIOR = 900.0


# ---------------- ω 估计器（各臂） ----------------

def est_true_omega(states):
    """真值 ω 通道（上限参照）。states: (T,22) raw。"""
    return states[:, 7].copy(), states[:, 18].copy()


def est_naive_diff(theta_hist):
    """朴素差分（T13 病根）。theta_hist: (T,) rad。"""
    w = np.zeros_like(theta_hist)
    w[1:] = (theta_hist[1:] - theta_hist[:-1]) / DT_SAMPLE
    w[0] = w[1]
    return w


def est_savgol(theta_hist):
    """Savitzky-Golay 5 点 2 阶导数（窗口内二次拟合的斜率）。"""
    # 5 点 SG 一阶导系数（窗口 [-2..2], polyorder 2）: [-2,-1,0,1,2]/(10*dt)
    c = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / (10.0 * DT_SAMPLE)
    T = len(theta_hist)
    w = np.zeros(T)
    pad = np.concatenate([[theta_hist[0]] * 2, theta_hist, [theta_hist[-1]] * 2])
    for t in range(T):
        w[t] = np.dot(c, pad[t:t + 5])
    return w


def unwrap_angle(a):
    return np.unwrap(a)


class DoublePendulumEKF:
    """4 维 [θ1, ω1, θ2, ω2] EKF，动力学 = 5×0.01s 细化半隐式欧拉双摆。

    观测 = [θ1, θ2]（由肘/手标记 atan2 推得）；F 雅可比由有限差分。
    """

    def __init__(self, q_omega=4.0, r_theta=(2.0e-3, 2.0e-3)):
        self.q_w = q_omega
        self.r = np.array(r_theta)
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1.0

    def _step(self, x, tau=0.0):
        th1, w1, th2, w2 = x
        dt = DT_SUB
        for _ in range(N_SUB):
            c2, s2 = math.cos(th2), math.sin(th2)
            M11 = (M1 + M2) * LH_L1 ** 2 + M2 * LH_L2 ** 2 + 2 * M2 * LH_L1 * LH_L2 * c2
            M12 = M2 * LH_L2 ** 2 + M2 * LH_L1 * LH_L2 * c2
            M22 = M2 * LH_L2 ** 2
            h1 = (-M2 * LH_L1 * LH_L2 * (2 * w1 + w2) * w2 * s2
                  + (M1 + M2) * G_PRIOR * LH_L1 * math.sin(th1)
                  + M2 * G_PRIOR * LH_L2 * math.sin(th1 + th2))
            h2 = (M2 * LH_L1 * LH_L2 * w1 ** 2 * s2
                  + M2 * G_PRIOR * LH_L2 * math.sin(th1 + th2))
            det = M11 * M22 - M12 ** 2
            a1 = (M22 * (-h1) - M12 * (-h2)) / det
            a2 = (-M12 * (-h1) + M11 * (-h2)) / det
            w1 += a1 * dt
            w2 += a2 * dt
            th1 += w1 * dt
            th2 += w2 * dt
        return np.array([th1, w1, th2, w2])

    def initialize(self, th1, th2):
        self.x = np.array([th1, 0.0, th2, 0.0])
        self.P = np.eye(4) * 1.0

    def step(self, th1_obs, th2_obs, tau=0.0):
        # 预测
        F = np.zeros((4, 4))
        eps = 1e-4
        for i in range(4):
            xp = self.x.copy(); xp[i] += eps
            xm = self.x.copy(); xm[i] -= eps
            F[:, i] = (self._step(xp, tau) - self._step(xm, tau)) / (2 * eps)
        x_pred = self._step(self.x, tau)
        Q = np.diag([1e-6, self.q_w, 1e-6, self.q_w])
        self.P = F @ self.P @ F.T + Q
        # 更新（θ1, θ2 观测）
        H = np.zeros((2, 4)); H[0, 0] = 1.0; H[1, 2] = 1.0
        y = np.array([th1_obs, th2_obs]) - H @ x_pred
        y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + np.diag(self.r)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.x[0] = (self.x[0] + np.pi) % (2 * np.pi) - np.pi
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.x.copy()


def est_ekf(states, actions=None):
    """EKF ω 轨迹；actions 为归一化力矩，按目标帧注入动力学。"""
    ekf = DoublePendulumEKF()
    th1_raw = np.arctan2(states[:, 1], -states[:, 2])         # 连杆1 角（相对竖直向下）
    x1 = states[:, 1]; y1 = states[:, 2]
    x2 = states[:, 12]; y2 = states[:, 13]
    th12 = np.arctan2(x2 - x1, -(y2 - y1))
    th2_raw = th12 - th1_raw                                   # 肘角
    out1 = np.zeros(len(states)); out2 = np.zeros(len(states))
    ekf.initialize(th1_raw[0], th2_raw[0])
    out1[0], out2[0] = 0.0, 0.0
    for t in range(1, len(states)):
        tau = 0.0
        if actions is not None:
            tau = float(np.asarray(actions[t]).reshape(-1)[0]) * TAU_MAX
        x = ekf.step(th1_raw[t], th2_raw[t], tau=tau)
        out1[t], out2[t] = x[1], x[3]
    return out1, out2


# ---------------- 先验注入评估 ----------------

def prior_eval(model, states, contact, actions, w1_seq, w2_seq, device,
               mask_vel=True, pos_noise=0.0, n_eps=N_EPS, n_win=N_WIN):
    """以 (w1_seq, w2_seq) 注入 prior_norm 的 one-step 位置 MAE(px)。

    states 含观测噪声（pos_noise 归一化），目标 = 干净真值帧。
    mask_vel=True 实现 L1 语义：把 obj 段的速度维（vx,vy,ω ×2）置 0 后再喂先验。
    """
    std_obj = model.std_obj.cpu().numpy()
    mean_obj = model.mean_obj.cpu().numpy()
    errs = []
    for ep in range(n_eps):
        obs = states[ep]
        if pos_noise > 0:
            obs = obs + np.random.RandomState(1000 + ep).randn(*obs.shape) * pos_noise
        idx = np.arange(n_win)
        t_now = idx + L - 1
        t_next = idx + L
        xn_raw = obs[t_now].astype(np.float64).copy()
        if mask_vel:
            for v in VEL_DIMS:
                xn_raw[:, v] = 0.0
        xn = torch.from_numpy(((xn_raw - mean_obj) / (std_obj + 1e-8))
                              .astype(np.float32)).to(device)
        an = torch.from_numpy(actions[ep, t_next].astype(np.float32)).to(device)
        w_hat = torch.from_numpy(np.stack([w1_seq[ep, t_now],
                                           w2_seq[ep, t_now]], axis=1
                                          ).astype(np.float32)).to(device)
        p = model.prior_norm(xn, an, w_hat=w_hat).float().cpu().numpy()
        p_full = p * (std_obj + 1e-8) + mean_obj
        pred = p_full[:, POS]
        tgt = states[ep, t_next][:, POS]
        errs.append(np.abs(pred - tgt).mean(axis=1))
    return float(np.concatenate(errs).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--hp_ckpt", type=str, default="checkpoints-hp/L1F/best.pt")
    ap.add_argument("--output", type=str, default="analysis_output/obsmodel_report.json")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--l0", action="store_true",
                    help="关闭 L1 速度遮蔽（仅用于与旧报告对拍，非 L1 语义）")
    args = ap.parse_args()
    device = torch.device(args.device)
    np.random.seed(42)

    d = np.load(args.clean)
    states_clean = d["states"][:500].astype(np.float64)
    contact, actions = d["contact"][:500], d["actions"][:500]

    ck = torch.load(args.hp_ckpt, map_location=device, weights_only=False)
    model = HingePOMDPModel(**ck["config"]).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()

    # 观测侧：干净的 θ 加位置噪声（两档）；速度通道全遮蔽（L1 语义）
    arms = ["true_ω", "naive_diff", "savgol", "ekf", "gru"]
    results = {}
    for level, nz in NOISE_PX.items():
        print("=" * 72)
        print(f"[{level}] 位置观测噪声（归一化）= {nz} | 速度遮蔽(L1) = {not args.l0}")
        # 预计算观测轨迹（θ 加噪一次，五臂共享）
        # σ 语义 = 归一化空间（与 T6/G1 一致）→ raw px 须 ×std 逐维换算
        obs_states = states_clean.copy()
        if nz > 0:
            std_x = states_clean[100:].reshape(-1, OBJ_DIMS).std(0)
            for ep in range(len(obs_states)):
                n4 = np.random.RandomState(2000 + ep).randn(200, 4) * nz
                obs_states[ep][:, [1, 2, 12, 13]] += n4 * std_x[[1, 2, 12, 13]]
        # 各臂 ω 序列（按 episode 独立计算）
        w_seqs = {}
        w_seqs["true_ω"] = (states_clean[:, :, 7], states_clean[:, :, 18])
        w_seqs["naive_diff"] = _batch(export_naive, obs_states)
        w_seqs["savgol"] = _batch(export_savgol, obs_states)
        w_seqs["ekf"] = _batch(export_ekf, obs_states, actions)
        # gru 臂：vel_head 吃 GRU 历史（用观测帧编码）
        #
        # 修法 1（逐帧演化）：encode_sequence 只返回最后一帧的 h（shape (1,256)），
        #   把它当 (T,256) 用会让整段 ω̂ 为常数。改为每步 gru_cell + ln_h 推进。
        # 修法 2（量纲）：vel_head 在训练时以归一化的 b["x_next"][:,[7,18]] 为靶，
        #   输出是**归一化单位**；prior_norm 期望原始 rad/s（raw=obj_norm*std+mean），
        #   须反归一化后再注入。
        std_obj = model.std_obj.cpu().numpy()
        mean_obj = model.mean_obj.cpu().numpy()
        gru_w = np.zeros((N_EPS, 200, 2))
        with torch.no_grad():
            for ep in range(N_EPS):
                frames = np.concatenate([
                    (obs_states[ep] - mean_obj) / (std_obj + 1e-8),
                    contact[ep].astype(np.float32)], axis=1)          # (200, 25)
                h = torch.zeros(1, model.hidden_dim, device=device)
                for t in range(frames.shape[0]):
                    # 注意：encode_sequence 只在最后做一次 ln_h，中间步不能 ln_h
                    #（否则 GRU 递归看到的分布与训练时不一致）
                    h = model.gru_cell(
                        torch.from_numpy(frames[t:t + 1].astype(np.float32)).to(device), h)
                    w_norm = model.estimate_omega(model.ln_h(h)).cpu().numpy()[0]
                    gru_w[ep, t, 0] = w_norm[0] * std_obj[7] + mean_obj[7]
                    gru_w[ep, t, 1] = w_norm[1] * std_obj[18] + mean_obj[18]
        w_seqs["gru"] = (gru_w[:, :, 0], gru_w[:, :, 1])
        res = {}
        for arm in arms:
            w1s, w2s = w_seqs[arm]
            mae = prior_eval(model, obs_states, contact, actions, w1s, w2s,
                             device, mask_vel=not args.l0, pos_noise=0.0)
            res[arm] = mae
            print(f"  {arm:<12} prior MAE = {mae:8.2f}px")
        # ω 估计精度（对真值，统一前 N_EPS 集）
        tw1, tw2 = w_seqs["true_ω"][0][:N_EPS], w_seqs["true_ω"][1][:N_EPS]
        for arm in ["naive_diff", "savgol", "ekf", "gru"]:
            w1s, w2s = w_seqs[arm][0][:N_EPS], w_seqs[arm][1][:N_EPS]
            e = np.abs(w1s - tw1).mean(), np.abs(w2s - tw2).mean()
            res[arm + "_ωerr"] = [float(e[0]), float(e[1])]
            print(f"  {arm:<12} omega_hat error = {e[0]:6.2f} / {e[1]:6.2f} rad/s")
        results[level] = res

    # ---------------- 预注册判决 ----------------
    r1, r3 = results["L1"], results["L3"]
    o_a = r1["naive_diff"] >= 2.0 * r1["true_ω"]
    o_b = r1["savgol"] < r1["naive_diff"] and r3["savgol"] < r3["naive_diff"]
    o_c = r1["ekf"] <= 1.5 * r1["true_ω"]
    o_d = r1["ekf"] < r1["gru"]
    o_e = (r3["savgol"] < r3["naive_diff"] and r3["ekf"] <= 1.5 * r3["true_ω"]
           and r3["ekf"] < r3["gru"])
    o_f = r1["gru"] <= r1["naive_diff"]
    crit = {"O_A_病根复现": bool(o_a), "O_B_SG有效": bool(o_b), "O_C_EKF闭合": bool(o_c),
            "O_D_EKF超GRU": bool(o_d), "O_E_噪声鲁棒": bool(o_e), "O_F_GRU现状": bool(o_f)}
    print("\n" + "=" * 72)
    print("[预注册判决]")
    print(f"  O-A 病根复现（naive ≥ 2×true）:    {'[OK]' if o_a else '[FAIL]'} "
          f"({r1['naive_diff']:.1f} vs {r1['true_ω']:.1f})")
    print(f"  O-B SG 有效（双档 < naive）:        {'[OK]' if o_b else '[FAIL]'}")
    print(f"  O-C EKF 闭合（≤ 1.5×true）:         {'[OK]' if o_c else '[FAIL]'} "
          f"({r1['ekf']:.1f} vs {r1['true_ω']:.1f})")
    print(f"  O-D EKF 超 GRU:                     {'[OK]' if o_d else '[FAIL]'} "
          f"({r1['ekf']:.1f} vs {r1['gru']:.1f})")
    print(f"  O-E L3 噪声鲁棒（同向）:            {'[OK]' if o_e else '[FAIL]'}")
    print(f"  O-F GRU 现状（≤ naive）:            {'[OK]' if o_f else '[FAIL]'} "
          f"({r1['gru']:.1f} vs {r1['naive_diff']:.1f})")
    print(f"\nVERDICT: {sum(crit.values())}/6")

    os.makedirs("analysis_output", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"results": results, "criteria": crit}, f, indent=2, ensure_ascii=False)
    print(f"[OK] {args.output}")


def _batch(fn, obs_states, actions=None):
    """按 episode 批量算 ω 轨迹。"""
    N = 100
    out1 = np.zeros((N, obs_states.shape[1]))
    out2 = np.zeros((N, obs_states.shape[1]))
    for ep in range(N):
        if actions is None:
            out1[ep], out2[ep] = fn(obs_states[ep])
        else:
            out1[ep], out2[ep] = fn(obs_states[ep], actions[ep])
    return out1, out2


def export_naive(states):
    th1 = np.arctan2(states[:, 1], -states[:, 2])
    x1, y1 = states[:, 1], states[:, 2]
    th12 = np.arctan2(states[:, 12] - x1, -(states[:, 13] - y1))
    th2 = th12 - th1
    return est_naive_diff(unwrap_angle(th1)), est_naive_diff(unwrap_angle(th2))


def export_savgol(states):
    th1 = np.arctan2(states[:, 1], -states[:, 2])
    x1, y1 = states[:, 1], states[:, 2]
    th12 = np.arctan2(states[:, 12] - x1, -(states[:, 13] - y1))
    th2 = th12 - th1
    return est_savgol(unwrap_angle(th1)), est_savgol(unwrap_angle(th2))


def export_ekf(states, actions=None):
    return est_ekf(states, actions)


if __name__ == "__main__":
    main()
