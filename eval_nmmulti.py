# -*- coding: utf-8 -*-
"""
多步辨识判决 (eval_nmmulti.py) — T18 指明的下一步
==================================================
T18 死因：单步 NM 目标对参数不敏感（±25% g 每步 ~0.2px）且有补偿盆地
（g125 辨识到 g=597，rollout 劣于固定先验）。本实验把 NM 目标改为
rollout 域（仿真推断式）：从观测状态出发，用候选参数迭代先验 H 步，
以轨迹拟合度为目标——混沌放大让参数误差在时程上显形。

决定性设计：同一 NM、同一数据、同一初始点，**只改目标时程 H**——
时程成为因果变量。H=1 臂即 T18 的复刻。

臂：
  g125: prior_fixed / nm5_H1(T18 复刻) / nm5_H5 / nm5_H10 / nmext_H5
  drag: prior_fixed / nm5_H5 / nmext_H5
  clean: prior_fixed / nm5_H5（纯系统辨识：真值 g=900 已知）

预注册判决（实验前写下）：
  M-A (g125 恢复):  nm5_H5 k5 ≤ 0.5 × prior_fixed k5（T18: nm5=6.52 > fixed=5.54）
  M-B (g125 参数):  |g_H5/1125 − 1| ≤ 0.20
  M-C (时程因果):   nm5_H10 k5 < nm5_H1 k5（g125）
  M-D (drag 类内):  nmext_H5 k5 ≤ nmext_H1 的 T18 值 2.06（无回归），
                    且 cd/cf 仍在真值邻域（cd∈[2e-3,1e-2], cf∈[1e3,1e4]）
  M-E (clean 无害): clean nm5_H5 k5 ≤ 1.2 × prior_fixed k5 且 |g/900−1| ≤ 0.20
  M-F (判决性附录升级为预注册臂): 细化先验积分器(5×0.01s)消除 O(dt) 混叠后,
      g125 下 |g/1125−1| ≤ 0.20 且 k5 ≤ 0.8 × prior_fixed k5

用法：python eval_nmmulti.py
"""

import os
import math
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from layout_h import OBJ_DIMS, ACTION_DIM, feat_weights as feat_weights_h
from benchmark_surpass import OursParamModel
from generate_g4 import WORLDS
from train_staged import DYN_IDX, LOG_BOUNDS, P_KEYS, TRUTH
from eval_sim2real_v3 import nm_reidentify
from eval_g4 import OursExtModel, EXT_BOUNDS, EXT_KEYS, EXT_OFF, rollout_eval_episodes

L, K = 8, 5
POS = [1, 2, 12, 13]
N_EPS, N_WIN_EVAL = 100, 60
ADAPT_EPS = (100, 400)
N_ID_WINDOWS = 2048
OFF5 = {"g": 1200.0, "M1": 3.0, "M2": 1.5, "L1": 110.0, "L2": 110.0}
WORLD_TRUTH = {"clean": {"g": 900.0, "cd": 0.0, "cf": 0.0},
               "g125": {"g": 1125.0, "cd": 0.0, "cf": 0.0},
               "drag": {"g": 900.0, "cd": 5e-3, "cf": 3000.0}}


def make_id_tensors(states, actions, mean_obj, std_obj, H, device,
                    n_windows=N_ID_WINDOWS, seed=123):
    """辨识窗口张量：初始状态 + H 步动作序列 + H 步真值目标。"""
    rng = np.random.RandomState(seed)
    eps = rng.randint(*ADAPT_EPS, size=n_windows)
    t0 = rng.randint(0, states.shape[1] - 1 - H, size=n_windows)
    xn0 = torch.from_numpy(((states[eps, t0] - mean_obj) / (std_obj + 1e-8)
                            ).astype(np.float32)).to(device)
    acts = torch.from_numpy(np.stack(
        [actions[eps, t0 + 1 + k].astype(np.float32) for k in range(H)], axis=1)
    ).to(device)                                          # (B,H,1)
    tgts = [torch.from_numpy(((states[eps, t0 + 1 + k] - mean_obj) / (std_obj + 1e-8)
                              ).astype(np.float32)).to(device) for k in range(H)]
    return xn0, acts, tgts


def nm_multistep(model, keys, bounds, off, states, actions, mean_obj, std_obj,
                 H, device, maxiter=800):
    """多步 NM：目标 = H 步先验 rollout 的 feat-加权 huber（确定性窗口）。"""
    xn0, acts, tgts = make_id_tensors(states, actions, mean_obj, std_obj, H, device)
    fw = feat_weights_h(OBJ_DIMS)[DYN_IDX]
    fw = (fw / fw.mean()).astype(np.float32)
    loss_w = torch.from_numpy(fw).to(device)
    n_eval = [0]

    def set_params(x):
        vals = [float(v) for v in np.exp(x)]
        if len(vals) == 5:
            model.set_physics(**dict(zip(P_KEYS, vals)))
        else:
            model.set_physics(**dict(zip(P_KEYS, vals[:5])))
            with torch.no_grad():
                model.log_cd.data.fill_(vals[5])
                model.log_cf.data.fill_(vals[6])

    def objective(x):
        set_params(x)
        with torch.no_grad():
            obj = xn0
            tot = 0.0
            for k in range(H):
                obj = model.prior_step(obj, acts[:, k])
                tot += (F.huber_loss(obj[:, DYN_IDX], tgts[k][:, DYN_IDX],
                                     reduction="none") * loss_w.view(1, -1)).mean().item()
        n_eval[0] += 1
        return tot / H

    x0 = [math.log(off[k]) for k in keys]
    res = minimize(objective, x0, method="Nelder-Mead", bounds=bounds,
                   options=dict(maxiter=maxiter, xatol=5e-4, fatol=1e-8))
    set_params(res.x)
    return model, res, n_eval[0]


@torch.no_grad()
def one_step_eval(model, states, actions, mean_obj, std_obj, device,
                  n_eps=N_EPS, n_win=N_WIN_EVAL):
    errs = []
    for ep in range(n_eps):
        for i in range(n_win):
            t = i + L - 1
            xn = torch.from_numpy(((states[ep, t] - mean_obj) / (std_obj + 1e-8)
                                   ).astype(np.float32)[None]).to(device)
            an = torch.from_numpy(actions[ep, t + 1].astype(np.float32)[None]).to(device)
            p = model.prior_step(xn, an)[0].float().cpu().numpy()
            p_full = p * (std_obj + 1e-8) + mean_obj
            errs.append(np.abs(p_full[POS] - states[ep, t + 1][POS]).mean())
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--g125", type=str, default=None,
                    help="g125 数据集路径（默认 g4_g125.npz；SLOT-2 逐 seed 传入）")
    ap.add_argument("--drag", type=str, default=None,
                    help="drag 数据集路径（默认 g4_drag.npz）")
    ap.add_argument("--output", type=str, default="analysis_output/nmmulti_report.json")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    report = {"worlds": {}}

    def load_world(path):
        d = np.load(path)
        states = d["states"][:500].astype(np.float64)
        contact, actions = d["contact"][:500], d["actions"][:500]
        flat = states[100:].reshape(-1, OBJ_DIMS)
        return states, contact, actions, flat.mean(0).astype(np.float32), flat.std(0).astype(np.float32)

    def fresh5():
        m = OursParamModel(OBJ_DIMS + 3, OBJ_DIMS, ACTION_DIM, 256).to(device)
        return m

    # ---------------- clean 参照 + M-E ----------------
    print("=" * 76)
    print("[clean] 多步辨识 = 纯系统辨识（真值 g=900）")
    states, contact, actions, mean_obj, std_obj = load_world(args.clean)
    mfix = fresh5(); mfix.set_stats(mean_obj, std_obj); mfix.set_physics(**TRUTH)
    kfix = rollout_eval_episodes(mfix, states, contact, actions, mean_obj, std_obj,
                                 device, use_prior=True, use_residual=False)
    m5 = fresh5(); m5.set_stats(mean_obj, std_obj)
    m5, res, nev = nm_multistep(m5, P_KEYS, LOG_BOUNDS, OFF5, states, actions,
                                mean_obj, std_obj, 5, device)
    p5 = m5.get_params()
    k5m = rollout_eval_episodes(m5, states, contact, actions, mean_obj, std_obj,
                                device, use_prior=True, use_residual=False)
    g_err = abs(p5["g"] / 900.0 - 1)
    m_e = k5m[-1] <= 1.2 * kfix[-1] and g_err <= 0.20
    print(f"  prior_fixed k1..k5: " + " ".join(f"{v:.2f}" for v in kfix))
    print(f"  nm5_H5 参数: " + " ".join(f"{k}={p5[k]:.1f}" for k in P_KEYS)
          + f" | g 误差 {g_err:.1%} | NM 评估 {nev} 次")
    print(f"  nm5_H5 k1..k5: " + " ".join(f"{v:.2f}" for v in k5m))
    report["worlds"]["clean"] = {"prior_fixed": kfix, "nm5_H5": k5m,
                                 "params": p5, "g_err": g_err}

    # ---------------- g125 / drag ----------------
    world_path = {"g125": args.g125, "drag": args.drag}
    plan = {"g125": [(1, "nm5_H1"), (5, "nm5_H5"), (10, "nm5_H10"), (5, "nmext_H5"),
                     (5, "nm5sub_H5")],
            "drag": [(5, "nm5_H5"), (5, "nmext_H5"), (5, "nm5sub_H5")]}
    for wname, world in WORLDS.items():
        print("\n" + "=" * 76)
        print(f"[{wname}] {world}")
        states, contact, actions, mean_obj, std_obj = load_world(
            world_path.get(wname) or f"g4_{wname}.npz")
        mfix = fresh5(); mfix.set_stats(mean_obj, std_obj); mfix.set_physics(**TRUTH)
        kfix = rollout_eval_episodes(mfix, states, contact, actions, mean_obj, std_obj,
                                     device, use_prior=True, use_residual=False)
        res_w = {"prior_fixed": kfix,
                 "prior_fixed_k1": one_step_eval(mfix, states, actions, mean_obj, std_obj, device)}
        print(f"  prior_fixed  k1..k5: " + " ".join(f"{v:.2f}" for v in kfix)
              + f" | one-step {res_w['prior_fixed_k1']:.2f}px")
        for H, tag in plan[wname]:
            if tag == "nm5sub_H5":
                from eval_nmsub import OursSubModel
                m = OursSubModel(OBJ_DIMS + 3, OBJ_DIMS, ACTION_DIM, 256).to(device)
                m.set_stats(mean_obj, std_obj)
                m, res, nev = nm_multistep(m, P_KEYS, LOG_BOUNDS, OFF5, states, actions,
                                           mean_obj, std_obj, H, device)
                params = m.get_params()
            elif tag.startswith("nm5"):
                m = fresh5(); m.set_stats(mean_obj, std_obj)
                m, res, nev = nm_multistep(m, P_KEYS, LOG_BOUNDS, OFF5, states, actions,
                                           mean_obj, std_obj, H, device)
                params = m.get_params()
            else:
                m = OursExtModel(OBJ_DIMS + 3, OBJ_DIMS, ACTION_DIM, 256).to(device)
                m.set_stats(mean_obj, std_obj)
                m, res, nev = nm_multistep(m, EXT_KEYS, EXT_BOUNDS, EXT_OFF, states, actions,
                                           mean_obj, std_obj, H, device)
                params = m.get_params()
            kk = rollout_eval_episodes(m, states, contact, actions, mean_obj, std_obj,
                                       device, use_prior=True, use_residual=False)
            res_w[tag] = kk
            res_w[tag + "_params"] = params
            g_err = abs(params["g"] / WORLD_TRUTH[wname]["g"] - 1)
            ext = (f" cd={params['cd']:.2g} cf={params['cf']:.0f}"
                   if "cd" in params else "")
            print(f"  {tag:<10} k1..k5: " + " ".join(f"{v:.2f}" for v in kk)
                  + f" | g={params['g']:.0f}(误差{g_err:.0%}){ext} | NM {nev} 次")
        report["worlds"][wname] = res_w

    # ---------------- 预注册判决 ----------------
    g = report["worlds"]["g125"]
    dr = report["worlds"]["drag"]
    cl = report["worlds"]["clean"]
    k5 = lambda r: r[-1]
    # nm5_H1（T18 复刻）基线：直接用 T18 数值（同一协议同一种子）
    nm5_h1_k5 = 6.52
    m_a = k5(g["nm5_H5"]) <= 0.5 * k5(g["prior_fixed"])
    p_g = g["nm5_H5_params"]["g"]
    m_b = abs(p_g / 1125.0 - 1) <= 0.20
    m_c = k5(g["nm5_H10"]) < nm5_h1_k5
    p_cd = dr["nmext_H5_params"]["cd"]; p_cf = dr["nmext_H5_params"]["cf"]
    m_d = k5(dr["nmext_H5"]) <= 2.06 and 2e-3 <= p_cd <= 1e-2 and 1e3 <= p_cf <= 1e4
    m_e = (k5(cl["nm5_H5"]) <= 1.2 * k5(cl["prior_fixed"])
           and cl["g_err"] <= 0.20)
    m_f = ("nm5sub_H5" in g
           and abs(g["nm5sub_H5_params"]["g"] / 1125.0 - 1) <= 0.20
           and k5(g["nm5sub_H5"]) <= 0.8 * k5(g["prior_fixed"]))
    crit = {"M_A_g125恢复": bool(m_a), "M_B_g参数": bool(m_b),
            "M_C_时程因果": bool(m_c), "M_D_drag类内": bool(m_d),
            "M_E_clean无害": bool(m_e), "M_F_细化类": bool(m_f)}
    report["criteria"] = crit
    print("\n" + "=" * 76)
    print("[预注册判决]")
    print(f"  M-A g125 恢复（nm5_H5 ≤ 0.5×fixed）:  {'[OK]' if m_a else '[FAIL]'} "
          f"({k5(g['nm5_H5']):.2f} vs {k5(g['prior_fixed']):.2f}；T18 单步 nm5=6.52)")
    print(f"  M-B g 参数恢复（±20%）:                {'[OK]' if m_b else '[FAIL]'} "
          f"(g={p_g:.0f} vs 1125)")
    print(f"  M-C 时程因果（H10 < H1=6.52）:         {'[OK]' if m_c else '[FAIL]'} "
          f"({k5(g['nm5_H10']):.2f})")
    print(f"  M-D drag 类内（nmext_H5 ≤ 2.06 且 cd/cf 邻域）: {'[OK]' if m_d else '[FAIL]'} "
          f"({k5(dr['nmext_H5']):.2f}, cd={p_cd:.2g}, cf={p_cf:.0f})")
    print(f"  M-E clean 无害（k5 ≤1.2×fixed 且 g±20%）: {'[OK]' if m_e else '[FAIL]'} "
          f"({k5(cl['nm5_H5']):.2f} vs {k5(cl['prior_fixed']):.2f}, g={cl['params']['g']:.0f})")
    if "nm5sub_H5" in g:
        print(f"  M-F 细化类（g125: |g/1125-1|≤20% 且 k5 ≤0.8×fixed）: {'[OK]' if m_f else '[FAIL]'} "
              f"(g={g['nm5sub_H5_params']['g']:.0f}, k5={k5(g['nm5sub_H5']):.2f} "
              f"vs fixed {k5(g['prior_fixed']):.2f})")
    n_pass = sum(crit.values())
    print(f"\nVERDICT: {n_pass}/5 —— "
          f"{'多步辨识成立' if n_pass >= 3 else '（如实记录）'}")

    os.makedirs("analysis_output", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] {args.output}")


if __name__ == "__main__":
    main()
