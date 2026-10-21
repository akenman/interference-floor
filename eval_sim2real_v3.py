# -*- coding: utf-8 -*-
"""
微缩 sim-to-real 判决 v3 (eval_sim2real_v3.py) — 修复 v2 协议两处缺陷后重判
==================================================
取证结论（2026-09-02，先于本实验确认）：
  ① v2 报告 85.7px"伪真实崩塌"的全部根因 = E1 速度重算的索引轴错位
    （`noisy[1:, b+vdim]` 在 (N,T,22) 布局写的是时间步整行 + 跨 episode
    差分÷0.05）——已修复。崩溃与域偏移、噪声鲁棒性无关。
  ② v2 协议自身两处缺陷：动作通道被置零（先验/模型都没吃到真实力矩）、
    预测目标用了被污染的伪真实帧（测的是传感器不是预测器，与 G1 协议
    "噪声观测 + 干净目标"的项目先例相悖）。本版均修复。

协议（v3）：
  观测 = 伪真实帧（逐效应注入），目标 = 干净真值帧，动作 = 数据真实力矩。
  数据集 = clean 对拍 + E1..E9 单效应（内存生成，seed 42）+ 全效应组合。
  臂 = 物理先验(真值参数) / 模型(checkpoints-hinge 零样本) / 零速度；
  E9 与组合数据集追加 prior+ω滤波；组合数据集追加 NM 重辨识先验 / 微调模型。

预注册判决（实验前写下）：
  SR-A 协议可信性：clean 上 model ≤ 0.7px 且 prior ≤ 3px（canonical 量级）
  SR-B 归因（PR-2'）：逐效应 MAE 排序；组合 model ≤ 3 × 最大单效应 model
  SR-C T1b-real（PR-1'）：组合上 model ≤ prior × 2 且 model ≤ zero-vel
  SR-D 适配有效（PR-3'）：NM 重辨识 prior < 固定 prior；微调 model < 零样本
  SR-E T13-real：E9 使 prior 劣化 ≥ 3×（vs clean），且 ω 三帧均值滤波
    把 E9 的先验劣化恢复 ≥ 50%

用法：python eval_sim2real_v3.py
"""

import math
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d

from model_h import WorldModelH
from layout_h import OBJ_DIMS, L1, L2, M1, M2, TAU_MAX, feat_weights as feat_weights_h
from make_pseudo_real import EFFECTS
from benchmark_surpass import OursParamModel

L, K = 8, 5
POS = [1, 2, 12, 13]
N_EPS, N_WIN = 100, 60          # 评测集：前 100 集 × 前 60 窗
ADAPT_EPS = (100, 400)          # 适配训练用集（与评测集隔离）

TRUTH = {"g": 900.0, "M1": 2.0, "M2": 1.0, "L1": 80.0, "L2": 80.0}
OFF_INIT = {"g": 1200.0, "M1": 3.0, "M2": 1.5, "L1": 110.0, "L2": 110.0}
LOG_BOUNDS = [(math.log(500), math.log(1500)), (math.log(0.1), math.log(10)),
              (math.log(0.1), math.log(10)), (math.log(20), math.log(200)),
              (math.log(20), math.log(200))]
P_KEYS = ["g", "M1", "M2", "L1", "L2"]
CONST_DIMS = [0, 8, 9, 10, 11, 19, 20, 21]
DYN_IDX = [i for i in range(OBJ_DIMS) if i not in CONST_DIMS]


def prior_one_step(state22, tau_norm):
    """物理先验单步（0.05s 半隐式欧拉，真值参数，raw 空间）。"""
    g = TRUTH["g"]
    th1 = math.atan2(state22[3], state22[4])
    th2 = math.atan2(state22[14], state22[15])
    w1 = float(state22[7]); w2 = float(state22[18])
    tau = float(np.clip(np.asarray(tau_norm, dtype=np.float64).ravel()[0], -1, 1)) * TAU_MAX
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
    out = state22.copy()
    out[1] = L1*math.sin(th1n); out[2] = -L1*math.cos(th1n)
    out[12] = out[1] + L2*math.sin(th1n+th2n)
    out[13] = out[2] - L2*math.cos(th1n+th2n)
    return out


def omega_filtered(states):
    """ω 三帧均值滤波（真实管线对差分速度的标准平滑）。"""
    out = states.copy()
    for d in (7, 18):
        out[:, :, d] = uniform_filter1d(states[:, :, d], 3, axis=1)
    return out


@torch.no_grad()
def eval_dataset(states_obs, states_tgt, contact, actions, model, mean_obj, std_obj,
                 device, use_model=None):
    """单数据集三臂评测。观测=states_obs，目标=states_tgt（干净），动作=真实。

    返回 (prior_mae, model_mae, zero_vel_mae, n)。use_model 缺省=主模型，
    传入微调后的模型可复用同一协议评估适配臂。
    """
    mdl = use_model if use_model is not None else model
    N, T = states_obs.shape[:2]
    n_win = min(N_WIN, T - L)
    idx = np.arange(n_win)
    t_now, t_next = idx + L - 1, idx + L
    prior_e, model_e, zero_e = [], [], []
    for ep in range(min(N, N_EPS)):
        acts = actions[ep]
        now22 = states_obs[ep, t_now]                        # (W,22) 观测
        tgt22 = states_tgt[ep, t_next]                       # (W,22) 干净真值
        a_now = acts[t_next].astype(np.float32)              # (W,1) 真实动作

        pr = np.stack([prior_one_step(states_obs[ep, t], a)
                       for t, a in zip(t_now, a_now)])
        prior_e.append(np.abs(pr[:, POS] - tgt22[:, POS]).mean(axis=1))
        zero_e.append(np.abs(now22[:, POS] - tgt22[:, POS]).mean(axis=1))

        frames = np.concatenate([
            (states_obs[ep] - mean_obj) / (std_obj + 1e-8),
            contact[ep].astype(np.float32)], axis=1)          # (T,25)
        seq = torch.from_numpy(
            np.stack([frames[i:i+L] for i in idx]).astype(np.float32)).to(device)
        h, _ = mdl.encode_sequence(seq)
        act = torch.from_numpy(a_now).to(device)
        pred_n = mdl.prior_norm(seq[:, -1, :OBJ_DIMS], act) + mdl.predict_state(h, act)
        pred22 = pred_n.float().cpu().numpy() * (std_obj + 1e-8) + mean_obj
        model_e.append(np.abs(pred22[:, POS] - tgt22[:, POS]).mean(axis=1))

    return (float(np.concatenate(prior_e).mean()),
            float(np.concatenate(model_e).mean()),
            float(np.concatenate(zero_e).mean()),
            sum(len(x) for x in prior_e))


def nm_reidentify(states_obs, states_tgt, actions, mean_obj, std_obj, device,
                  n_windows=8192, maxiter=400):
    """在伪真实观测上重辨识物理参数（观测=伪真实帧，目标=干净真值）。"""
    model = OursParamModel(OBJ_DIMS + 3, OBJ_DIMS, 1, 256).to(device)
    model.set_stats(mean_obj, std_obj)
    rng = np.random.RandomState(123)
    eps = rng.randint(*ADAPT_EPS, size=n_windows)
    t0 = rng.randint(0, 100, size=n_windows)
    xn = torch.from_numpy(((states_obs[eps, t0] - mean_obj) / (std_obj + 1e-8)
                           ).astype(np.float32)).to(device)
    an = torch.from_numpy(actions[eps, t0 + 1].astype(np.float32)).to(device)
    tn = torch.from_numpy(((states_tgt[eps, t0 + 1] - mean_obj) / (std_obj + 1e-8)
                           ).astype(np.float32)).to(device)
    fw = feat_weights_h(OBJ_DIMS)[DYN_IDX]
    fw = (fw / fw.mean()).astype(np.float32)
    loss_w = torch.from_numpy(fw).to(device)
    nb = math.ceil(n_windows / 2048)

    def objective(log_x):
        model.set_physics(*(float(v) for v in np.exp(log_x)))
        with torch.no_grad():
            tot = 0.0
            for j in range(0, n_windows, 2048):
                p = model.prior_step(xn[j:j+2048], an[j:j+2048])
                tot += (F.huber_loss(p[:, DYN_IDX], tn[j:j+2048][:, DYN_IDX],
                                     reduction="none") * loss_w.view(1, -1)).mean().item()
        return tot / nb

    res = minimize(objective, [math.log(OFF_INIT[k]) for k in P_KEYS],
                   method="Nelder-Mead", bounds=LOG_BOUNDS,
                   options=dict(maxiter=maxiter, xatol=5e-4, fatol=1e-8))
    params = {k: float(v) for k, v in zip(P_KEYS, np.exp(res.x))}
    model.set_physics(**params)
    return model, params


def finetune_model(model, states_obs, states_tgt, contact, actions, mean_obj, std_obj,
                   device, epochs=5, lr=1e-4, windows_per_epoch=20000):
    """在伪真实观测(输入)→干净真值(目标)上微调 canonical 模型。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    fw = feat_weights_h(OBJ_DIMS)[DYN_IDX]
    fw = (fw / fw.mean()).astype(np.float32)
    loss_w = torch.from_numpy(fw).to(device)
    rng = np.random.RandomState(7)
    model.train()
    for _ in range(epochs):
        eps = rng.randint(*ADAPT_EPS, size=windows_per_epoch)
        t0 = rng.randint(0, 100, size=windows_per_epoch)
        for j in range(0, windows_per_epoch, 2048):
            e_s, t_s = eps[j:j+2048], t0[j:j+2048]
            frames = np.concatenate([
                (states_obs[e_s] - mean_obj) / (std_obj + 1e-8),
                contact[e_s].astype(np.float32)], axis=2)     # (B,L,25)
            seq = torch.from_numpy(np.stack([
                frames[b, t:t+L] for b, t in enumerate(t_s)
            ]).astype(np.float32)).to(device)
            act = torch.from_numpy(actions[e_s, t_s + L].astype(np.float32)).to(device)
            tgt = torch.from_numpy(((states_tgt[e_s, t_s + L] - mean_obj)
                                    / (std_obj + 1e-8)).astype(np.float32)).to(device)
            h, _ = model.encode_sequence(seq)
            pred = model.prior_norm(seq[:, -1, :OBJ_DIMS], act) + model.predict_state(h, act)
            loss = (F.huber_loss(pred[:, DYN_IDX], tgt[:, DYN_IDX], reduction="none")
                    * loss_w.view(1, -1)).mean()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def eval_custom_prior(prior_fn, states_obs, states_tgt, actions, n_eps=N_EPS):
    """以 raw 空间 prior_fn(state22_obs, a_norm)->state22 评估先验臂。"""
    errs = []
    for ep in range(n_eps):
        t_now = np.arange(N_WIN) + L - 1
        for t, a in zip(t_now, actions[ep][t_now + 1]):
            p = prior_fn(states_obs[ep, t], a)
            errs.append(np.abs(p[POS] - states_tgt[ep, t + 1][POS]).mean())
    return float(np.mean(errs))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--ckpt", type=str, default="checkpoints-hinge/best.pt")
    ap.add_argument("--output", type=str, default="analysis_output/sim2real_v3_report.json")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    d = np.load(args.clean)
    states_clean = d["states"][:500].astype(np.float64)
    contact = d["contact"][:500]
    actions = d["actions"][:500]

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = WorldModelH(**ck["config"]).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    ns = ck["norm_stats"]
    mean_obj, std_obj = ns["mean_obj"], ns["std_obj"]

    # 数据集：clean + 逐效应（内存生成）+ 全效应组合
    datasets = {"clean": states_clean}
    for name, fn in EFFECTS.items():
        np.random.seed(42)
        datasets[name] = fn(states_clean.copy())
    np.random.seed(42)
    combined = states_clean.copy()
    for name, fn in EFFECTS.items():
        combined = fn(combined)
    datasets["combined"] = combined

    results = {}
    print("=" * 78)
    print("微缩 sim-to-real 判决 v3（观测=伪真实，目标=干净真值，动作=真实力矩）")
    print("=" * 78)
    for name, obs in datasets.items():
        pr, mo, zv, n = eval_dataset(obs, states_clean, contact, actions, model,
                                     mean_obj, std_obj, device)
        row = {"prior": pr, "model": mo, "zero_vel": zv, "n": n}
        if name in ("E9_realistic_omega", "combined"):
            prf, _, _, _ = eval_dataset(omega_filtered(obs), states_clean, contact,
                                        actions, model, mean_obj, std_obj, device)
            row["prior_wfilt"] = prf
        results[name] = row
        extra = f" | prior+ω滤波 {row['prior_wfilt']:.2f}" if "prior_wfilt" in row else ""
        print(f"{name:<22} prior {pr:>8.2f} | model {mo:>8.2f} | zero-vel {zv:>8.2f}{extra}")

    # ---- 组合数据集上的适配臂（SR-D）----
    print("\n[适配臂] 组合伪真实上：NM 重辨识先验 + 模型微调")
    nm_model, nm_params = nm_reidentify(datasets["combined"], states_clean, actions,
                                        mean_obj, std_obj, device)

    @torch.no_grad()
    def nm_prior_raw(state22_obs, a_norm):
        x = torch.from_numpy(((state22_obs - mean_obj) / (std_obj + 1e-8)
                              ).astype(np.float32)[None]).to(device)
        a = torch.from_numpy(np.asarray(a_norm, dtype=np.float32)[None]).to(device)
        p = nm_model.prior_step(x, a)[0].float().cpu().numpy()
        return p * (std_obj + 1e-8) + mean_obj

    pr_nm = eval_custom_prior(nm_prior_raw, datasets["combined"], states_clean, actions)
    results["combined"]["prior_nm"] = pr_nm
    print(f"  NM 重辨识: " + " ".join(f"{k}={v:.1f}" for k, v in nm_params.items())
          + f" | prior_nm = {pr_nm:.2f}px")

    # 事后修正（post-hoc，非预注册，如实标注）：SR-E 已证 ω̂ 噪声是先验劣化
    # 主因且滤波有效——NM 辨识同样应吃滤波后的观测。预期修正 SR-D 的 NM 半边。
    nm_model_f, nm_params_f = nm_reidentify(
        omega_filtered(datasets["combined"]), states_clean, actions,
        mean_obj, std_obj, device)

    @torch.no_grad()
    def nmf_prior_raw(state22_obs, a_norm):
        x = torch.from_numpy(((state22_obs - mean_obj) / (std_obj + 1e-8)
                              ).astype(np.float32)[None]).to(device)
        a = torch.from_numpy(np.asarray(a_norm, dtype=np.float32)[None]).to(device)
        p = nm_model_f.prior_step(x, a)[0].float().cpu().numpy()
        return p * (std_obj + 1e-8) + mean_obj

    pr_nmf = eval_custom_prior(nmf_prior_raw, datasets["combined"], states_clean, actions)
    results["combined"]["prior_nm_filt"] = pr_nmf
    print(f"  NM 重辨识(ω滤波后, post-hoc): "
          + " ".join(f"{k}={v:.1f}" for k, v in nm_params_f.items())
          + f" | prior_nm_filt = {pr_nmf:.2f}px")

    model_ft = finetune_model(model, datasets["combined"], states_clean, contact,
                              actions, mean_obj, std_obj, device)
    _, mo_ft, _, _ = eval_dataset(datasets["combined"], states_clean, contact, actions,
                                  model, mean_obj, std_obj, device, use_model=model_ft)
    results["combined"]["model_ft"] = mo_ft
    print(f"  模型微调(5ep): model_ft = {mo_ft:.2f}px")

    # ================================================================
    # 预注册判决
    # ================================================================
    cl, cb = results["clean"], results["combined"]
    sr_a = cl["model"] <= 0.7 and cl["prior"] <= 3.0
    single = {k: v for k, v in results.items() if k.startswith("E") and k != "combined"}
    worst_eff = max(single, key=lambda k: single[k]["model"])
    sr_b = cb["model"] <= 3.0 * single[worst_eff]["model"]
    sr_c = cb["model"] <= 2.0 * cb["prior"] and cb["model"] <= cb["zero_vel"]
    sr_d = cb["prior_nm"] < cb["prior"] and cb["model_ft"] < cb["model"]
    sr_d_amended = (cb.get("prior_nm_filt", 1e9) < cb["prior"]
                    and cb["model_ft"] < cb["model"])
    e9 = results.get("E9_realistic_omega", {})
    filt_recovery = 0.0
    if e9:
        denom = max(e9["prior"] - cl["prior"], 1e-9)
        filt_recovery = (e9["prior"] - e9.get("prior_wfilt", e9["prior"])) / denom
    sr_e = bool(e9) and e9["prior"] >= 3.0 * cl["prior"] and filt_recovery >= 0.5
    crit = {"SR_A": bool(sr_a), "SR_B": bool(sr_b), "SR_C": bool(sr_c),
            "SR_D": bool(sr_d), "SR_D_amended_posthoc": bool(sr_d_amended),
            "SR_E": bool(sr_e),
            "worst_effect": worst_eff, "filt_recovery": round(float(filt_recovery), 3)}
    results["criteria"] = crit

    print("\n" + "=" * 78)
    print("[预注册判决]")
    print(f"  SR-A 协议可信（clean model≤0.7 且 prior≤3）:    {'[OK]' if sr_a else '[FAIL]'} "
          f"(model {cl['model']:.3f}, prior {cl['prior']:.3f})")
    print(f"  SR-B 归因（组合 ≤ 3×最大单效应 {worst_eff}）:       {'[OK]' if sr_b else '[FAIL]'} "
          f"(combined {cb['model']:.2f} vs 3×{single[worst_eff]['model']:.2f})")
    print(f"  SR-C T1b-real（model≤2×prior 且 ≤zero-vel）:    {'[OK]' if sr_c else '[FAIL]'} "
          f"(model {cb['model']:.2f}, prior {cb['prior']:.2f}, zv {cb['zero_vel']:.2f})")
    print(f"  SR-D 适配有效（NM prior 降 + FT model 降）:      {'[OK]' if sr_d else '[FAIL]'} "
          f"(prior {cb['prior']:.2f}→{pr_nm:.2f}, model {cb['model']:.2f}→{mo_ft:.2f})")
    print(f"  SR-D' 事后修正（ω滤波后 NM + FT）:              {'[OK]' if sr_d_amended else '[FAIL]'} "
          f"(prior_nm_filt {cb.get('prior_nm_filt', float('nan')):.2f}, model_ft {mo_ft:.2f})")
    print(f"  SR-E T13-real（E9 prior≥3×clean 且滤波恢复≥50%）:  {'[OK]' if sr_e else '[FAIL]'} "
          f"(E9 prior {e9.get('prior', float('nan')):.2f} vs clean {cl['prior']:.2f}, "
          f"恢复率 {filt_recovery:.0%})")
    verdict = ("[OK] 架构在伪真实条件下存活（v2 崩塌系 E1 索引 bug 伪影）"
               if (sr_a and sr_c) else "[FAIL] 仍有未过项（如实记录）")
    print(f"\nVERDICT: {verdict}")

    os.makedirs("analysis_output", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] {args.output}")


if __name__ == "__main__":
    main()
