# -*- coding: utf-8 -*-
"""
G4 真动力学偏移判决 (eval_g4.py) — T5"先验双刃剑"补完
==================================================
两个偏移世界（generate_g4.py 重仿真，观测通道干净——与 T16 正交）：
  g125 : g=1125（+25%，参数类偏移，NM-5 类内可辨识）
  drag : cd=5e-3, cf=3000（加性类偏移，NM-5 类外、NM-7 类内）
主指标 = rollout k=5 位置 MAE（预检：单步对参数误差天然鲁棒（dt=0.05 稀释），
混沌放大使偏移在部署时程显现：clean 0 / g125 2.8 / drag 19 px @k=5）。

臂（每世界）：prior_fixed / prior_nm5 / prior_nmext(+cd/cf) /
  hybrid_fixed(固定先验+残差) / hybrid_nmext / noprior(纯学习) / zero_vel

预注册判决（实验前写下）：
  G4-A (P4.1 净收益): g125 k5: prior_fixed < zero_vel_k1 且 prior_fixed ≥ 3 × clean 参照
  G4-B (P4.2 负债):   g125: hybrid_fixed ≤ noprior → 先验非负债；> → 负债成立
  G4-C (类内辨识):    g125: prior_nm5 k5 ≤ 0.5 × prior_fixed k5
  G4-D (类外扩展):    drag: prior_nmext k5 < prior_nm5 k5；且 g125 下 cd/cf ≈ 下界
  G4-E (OOS 最佳架构): drag: hybrid_nmext k5 ≤ noprior k5

用法：python eval_g4.py
"""

import os
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize

from layout_h import OBJ_DIMS, ACTION_DIM, feat_weights as feat_weights_h
from train_hinge import HingeData
from benchmark_surpass import OursParamModel
from core_model import NoPriorModel
from generate_g4 import WORLDS
from train_staged import DYN_IDX, LOG_BOUNDS, P_KEYS, TRUTH, train_residual
from eval_sim2real_v3 import nm_reidentify

L, K = 8, 5
POS = [1, 2, 12, 13]
N_EPS, N_WIN = 100, 60
ADAPT_EPS = (100, 400)

EXT_BOUNDS = LOG_BOUNDS + [(math.log(1e-6), math.log(1e-2)),
                           (math.log(1.0), math.log(1e4))]
EXT_KEYS = P_KEYS + ["cd", "cf"]
EXT_OFF = {"g": 1200.0, "M1": 3.0, "M2": 1.5, "L1": 110.0, "L2": 110.0,
           "cd": 1e-4, "cf": 50.0}


class OursExtModel(OursParamModel):
    """NM-7 扩展先验：g/M/L + 端点阻力 cd + 关节摩擦 cf（torch 向量化）。"""

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256):
        super().__init__(input_dim, obj_dim, action_dim, hidden_dim)
        self.log_cd = nn.Parameter(torch.log(torch.tensor(1e-4)))
        self.log_cf = nn.Parameter(torch.log(torch.tensor(50.0)))

    def get_params(self):
        p = super().get_params()
        p["cd"] = self.log_cd.exp().clamp(1e-6, 1e-2).item()
        p["cf"] = self.log_cf.exp().clamp(1.0, 1e4).item()
        return p

    def prior_step(self, obj_norm, a_norm):
        g = self.log_g.exp().clamp(500, 1500)
        m1 = self.log_M1.exp().clamp(0.1, 10)
        m2 = self.log_M2.exp().clamp(0.1, 10)
        l1 = self.log_L1.exp().clamp(20, 200)
        l2 = self.log_L2.exp().clamp(20, 200)
        cd = self.log_cd.exp().clamp(1e-6, 1e-2)
        cf = self.log_cf.exp().clamp(1.0, 1e4)
        raw = obj_norm * (self.std_obj + 1e-8) + self.mean_obj
        b1, b2 = 0, 11
        th1 = torch.atan2(raw[:, b1 + 3], raw[:, b1 + 4])
        th2 = torch.atan2(raw[:, b2 + 3], raw[:, b2 + 4])
        w1 = raw[:, b1 + 7]
        w2 = raw[:, b2 + 7]
        tau = a_norm.squeeze(-1) * 28000.0
        c2, s2 = torch.cos(th2), torch.sin(th2)
        M11 = (m1 + m2) * l1 ** 2 + m2 * l2 ** 2 + 2 * m2 * l1 * l2 * c2
        M12 = m2 * l2 ** 2 + m2 * l1 * l2 * c2
        M22 = m2 * l2 ** 2
        h1 = (-m2 * l1 * l2 * (2 * w1 + w2) * w2 * s2
              + (m1 + m2) * g * l1 * torch.sin(th1) + m2 * g * l2 * torch.sin(th1 + th2))
        h2 = (m2 * l1 * l2 * w1 ** 2 * s2 + m2 * g * l2 * torch.sin(th1 + th2))
        c1, s1 = torch.cos(th1), torch.sin(th1)
        c12, s12 = torch.cos(th1 + th2), torch.sin(th1 + th2)
        v1x, v1y = l1 * w1 * c1, l1 * w1 * s1
        v2x = v1x + l2 * (w1 + w2) * c12
        v2y = v1y + l2 * (w1 + w2) * s12
        s1n_ = torch.sqrt(v1x ** 2 + v1y ** 2 + 1e-9)
        s2n_ = torch.sqrt(v2x ** 2 + v2y ** 2 + 1e-9)
        f1x, f1y = -cd * s1n_ * v1x, -cd * s1n_ * v1y
        f2x, f2y = -cd * s2n_ * v2x, -cd * s2n_ * v2y
        qd1 = f1x * (l1 * c1) + f1y * (l1 * s1) + f2x * (l2 * c12) + f2y * (l2 * s12)
        qd2 = f2x * (l2 * c12) + f2y * (l2 * s12)
        Q1 = -cf * w1 + qd1
        Q2 = tau - cf * w2 + qd2
        h1 = h1 - Q1
        h2 = h2 - Q2
        det = M11 * M22 - M12 ** 2
        a1 = (M22 * (-h1) - M12 * (-h2)) / det
        a2 = (-M12 * (-h1) + M11 * (-h2)) / det
        w1n = w1 + a1 * 0.05
        w2n = w2 + a2 * 0.05
        th1n = th1 + w1n * 0.05
        th2n = th2 + w2n * 0.05
        out = raw.clone()
        x1 = l1 * torch.sin(th1n); y1 = -l1 * torch.cos(th1n)
        x2 = x1 + l2 * torch.sin(th1n + th2n)
        y2 = y1 - l2 * torch.cos(th1n + th2n)
        vx1 = l1 * w1n * torch.cos(th1n); vy1 = l1 * w1n * torch.sin(th1n)
        vx2 = vx1 + l2 * (w1n + w2n) * torch.cos(th1n + th2n)
        vy2 = vy1 + l2 * (w1n + w2n) * torch.sin(th1n + th2n)
        out[:, b1 + 1], out[:, b1 + 2] = x1, y1
        out[:, b2 + 1], out[:, b2 + 2] = x2, y2
        out[:, b1 + 3], out[:, b1 + 4] = torch.sin(th1n), torch.cos(th1n)
        out[:, b2 + 3], out[:, b2 + 4] = torch.sin(th2n), torch.cos(th2n)
        out[:, b1 + 5], out[:, b1 + 6] = vx1, vy1
        out[:, b2 + 5], out[:, b2 + 6] = vx2, vy2
        out[:, b1 + 7], out[:, b2 + 7] = w1n, w2n
        return (out - self.mean_obj) / (self.std_obj + 1e-8)


def nm_reidentify_ext(states, actions, mean_obj, std_obj, device,
                      n_windows=8192, maxiter=600):
    """NM-7：g/M/L/cd/cf 联合辨识（观测=目标=世界真值，单步对齐）。"""
    model = OursExtModel(OBJ_DIMS + 3, OBJ_DIMS, ACTION_DIM, 256).to(device)
    model.set_stats(mean_obj, std_obj)
    rng = np.random.RandomState(123)
    eps = rng.randint(*ADAPT_EPS, size=n_windows)
    t0 = rng.randint(0, 100, size=n_windows)
    xn = torch.from_numpy(((states[eps, t0] - mean_obj) / (std_obj + 1e-8)
                           ).astype(np.float32)).to(device)
    an = torch.from_numpy(actions[eps, t0 + 1].astype(np.float32)).to(device)
    tn = torch.from_numpy(((states[eps, t0 + 1] - mean_obj) / (std_obj + 1e-8)
                           ).astype(np.float32)).to(device)
    fw = feat_weights_h(OBJ_DIMS)[DYN_IDX]
    fw = (fw / fw.mean()).astype(np.float32)
    loss_w = torch.from_numpy(fw).to(device)
    nb = math.ceil(n_windows / 2048)

    def objective(x7):
        model.set_physics(*(float(v) for v in np.exp(x7[:5])))
        with torch.no_grad():
            model.log_cd.data.fill_(float(x7[5]))
            model.log_cf.data.fill_(float(x7[6]))
            tot = 0.0
            for j in range(0, n_windows, 2048):
                p = model.prior_step(xn[j:j+2048], an[j:j+2048])
                tot += (F.huber_loss(p[:, DYN_IDX], tn[j:j+2048][:, DYN_IDX],
                                     reduction="none") * loss_w.view(1, -1)).mean().item()
        return tot / nb

    x0 = [math.log(EXT_OFF[k]) for k in EXT_KEYS]
    res = minimize(objective, x0, method="Nelder-Mead", bounds=EXT_BOUNDS,
                   options=dict(maxiter=maxiter, xatol=5e-4, fatol=1e-8))
    p5 = {k: float(v) for k, v in zip(P_KEYS, np.exp(res.x[:5]))}
    model.set_physics(**p5)
    with torch.no_grad():
        model.log_cd.data.fill_(float(res.x[5]))
        model.log_cf.data.fill_(float(res.x[6]))
    return model, model.get_params()


def _step_hidden(model, x_next, h):
    """GRU 单步隐状态更新（与各模型训练时一致：ln_h 输出，不做归一化）。"""
    if hasattr(model, "gru_cell"):
        return model.ln_h(model.gru_cell(x_next, h))
    return model.trunk.ln_h(model.trunk.gru_cell(x_next, h))


@torch.no_grad()
def rollout_eval_episodes(model, states, contact, actions, mean_obj, std_obj,
                          device, K=5, use_prior=True, use_residual=True,
                          n_eps=N_EPS, n_win=N_WIN):
    """迭代 rollout（冻结接触通道反馈，与 model_h.rollout 同构）。

    model: OursParamModel/OursExtModel（encode/predict/prior_step/gru_cell/ln_h）
           或 NoPriorModel（encode/predict + trunk.gru_cell/trunk.ln_h）。
    返回逐 k 位置 MAE(px)。
    """
    model.eval()
    errs = np.zeros(K)
    cnt = 0
    for ep in range(min(n_eps, states.shape[0])):
        idx = np.arange(n_win)
        t0s = idx + L - 1
        af = actions[ep]
        frames = np.concatenate([
            (states[ep] - mean_obj) / (std_obj + 1e-8),
            contact[ep].astype(np.float32)], axis=1)               # (T,25)
        seq = torch.from_numpy(
            np.stack([frames[i - L + 1:i + 1] for i in t0s]).astype(np.float32)).to(device)
        h = model.encode(seq)
        obj_last = seq[:, -1, :OBJ_DIMS].clone()
        ct_frozen = seq[:, -1, -3:]
        B = seq.size(0)
        for k in range(K):
            a_k = torch.from_numpy(af[t0s + 1 + k].astype(np.float32)).to(device)
            if use_prior:
                obj_next = model.prior_step(obj_last, a_k) + model.predict(h, a_k)
            else:
                obj_next = model.predict(h, a_k)
            pred = obj_next.float().cpu().numpy() * (std_obj + 1e-8) + mean_obj
            tgt = states[ep, t0s + 1 + k][:, POS]
            errs[k] += np.abs(pred[:, POS] - tgt).sum()
            cnt += B
            if use_residual:
                x_next = torch.cat([obj_next, ct_frozen], dim=-1)
                h = _step_hidden(model, x_next, h)
            obj_last = obj_next
    return [float(e / cnt) for e in errs]


def zero_vel_k1(states):
    zv = []
    for ep in range(N_EPS):
        for i in range(N_WIN):
            t = i + L - 1
            zv.append(np.abs(states[ep, t, [1, 2, 12, 13]]
                             - states[ep, t + 1, [1, 2, 12, 13]]).mean())
    return float(np.mean(zv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42,
                    help="SLOT-2：训练与评估的 torch/numpy 种子（数据由 generate_g4 --seed 变异）")
    ap.add_argument("--g125", type=str, default=None,
                    help="g125 数据集路径（默认 g4_g125.npz；SLOT-2 逐 seed 传入）")
    ap.add_argument("--drag", type=str, default=None,
                    help="drag 数据集路径（默认 g4_drag.npz）")
    ap.add_argument("--output", type=str, default="analysis_output/g4_report.json")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    world_path = {"g125": args.g125, "drag": args.drag}

    fw = feat_weights_h(OBJ_DIMS)[DYN_IDX]
    fw = (fw / fw.mean()).astype(np.float32)
    loss_w = torch.from_numpy(fw).to(device)
    targs = argparse.Namespace(noise_aug=0.0, batch_size=1024, windows_per_epoch=200000)

    report = {"worlds": {}}

    # clean 参照（固定先验 rollout）
    d0 = np.load(args.clean)
    st0 = d0["states"][:500].astype(np.float64)
    flat0 = st0[100:].reshape(-1, OBJ_DIMS)
    mean0 = flat0.mean(0).astype(np.float32)
    std0 = flat0.std(0).astype(np.float32)
    mfix0 = OursParamModel(OBJ_DIMS + 3, OBJ_DIMS, ACTION_DIM, 256).to(device)
    mfix0.set_stats(mean0, std0)
    mfix0.set_physics(**TRUTH)
    k_clean = rollout_eval_episodes(mfix0, st0, d0["contact"][:500], d0["actions"][:500],
                                    mean0, std0, device, use_prior=True, use_residual=False)
    print(f"[clean 参照] prior_fixed k1..k5: " + " ".join(f"{v:.2f}" for v in k_clean))
    report["clean_prior_rollout"] = k_clean

    for wname, world in WORLDS.items():
        print("\n" + "=" * 72)
        print(f"[世界 {wname}] {world}")
        print("=" * 72)
        d = np.load(world_path.get(wname) or f"g4_{wname}.npz")
        states = d["states"][:500].astype(np.float64)
        contact, actions = d["contact"][:500], d["actions"][:500]
        flat = states[100:].reshape(-1, OBJ_DIMS)
        mean_obj = flat.mean(0).astype(np.float32)
        std_obj = flat.std(0).astype(np.float32)
        train_ds = HingeData(states, contact, actions, d["yao"][:500],
                             np.arange(100, 500), L, K, mean_obj, std_obj)

        res = {}
        mfix = OursParamModel(OBJ_DIMS + 3, OBJ_DIMS, ACTION_DIM, 256).to(device)
        mfix.set_stats(mean_obj, std_obj)
        mfix.set_physics(**TRUTH)
        res["prior_fixed"] = rollout_eval_episodes(mfix, states, contact, actions,
                                                   mean_obj, std_obj, device,
                                                   use_prior=True, use_residual=False)

        nm5, p5 = nm_reidentify(states, states, actions, mean_obj, std_obj, device)
        res["prior_nm5"] = rollout_eval_episodes(nm5, states, contact, actions,
                                                 mean_obj, std_obj, device,
                                                 use_prior=True, use_residual=False)
        res["nm5_params"] = p5
        print(f"  NM-5: " + " ".join(f"{k}={p5[k]:.1f}" for k in P_KEYS))

        nm7, p7 = nm_reidentify_ext(states, actions, mean_obj, std_obj, device)
        res["prior_nmext"] = rollout_eval_episodes(nm7, states, contact, actions,
                                                   mean_obj, std_obj, device,
                                                   use_prior=True, use_residual=False)
        res["nmext_params"] = p7
        print(f"  NM-7: " + " ".join(f"{k}={p7[k]:.3g}" for k in EXT_KEYS))

        mh = OursParamModel(OBJ_DIMS + 3, OBJ_DIMS, ACTION_DIM, 256).to(device)
        mh.set_stats(mean_obj, std_obj)
        mh.set_physics(**TRUTH)
        train_residual(mh, train_ds, targs, device, loss_w, args.epochs, 3e-4,
                       train_physics=False)
        res["hybrid_fixed"] = rollout_eval_episodes(mh, states, contact, actions,
                                                    mean_obj, std_obj, device,
                                                    use_prior=True, use_residual=True)

        me = OursExtModel(OBJ_DIMS + 3, OBJ_DIMS, ACTION_DIM, 256).to(device)
        me.set_stats(mean_obj, std_obj)
        me.load_state_dict(nm7.state_dict())
        train_residual(me, train_ds, targs, device, loss_w, args.epochs, 3e-4,
                       train_physics=False)
        res["hybrid_nmext"] = rollout_eval_episodes(me, states, contact, actions,
                                                    mean_obj, std_obj, device,
                                                    use_prior=True, use_residual=True)

        mn = NoPriorModel(OBJ_DIMS + 3, OBJ_DIMS, ACTION_DIM, 256).to(device)
        train_residual(mn, train_ds, targs, device, loss_w, args.epochs, 3e-4,
                       train_physics=False, use_prior=False)
        res["noprior"] = rollout_eval_episodes(mn, states, contact, actions,
                                               mean_obj, std_obj, device,
                                               use_prior=False, use_residual=True)

        res["zero_vel_k1"] = zero_vel_k1(states)

        for arm, vals in res.items():
            if isinstance(vals, list):
                print(f"  {arm:<14} k1..k5: " + " ".join(f"{v:7.2f}" for v in vals))
            elif isinstance(vals, dict):
                continue
            else:
                print(f"  {arm:<14} k1: {vals:.2f}")
        report["worlds"][wname] = res

    # ================================================================
    # 预注册判决
    # ================================================================
    g125 = report["worlds"]["g125"]
    drag = report["worlds"]["drag"]
    k5 = lambda r: r[-1]
    g4a = (k5(g125["prior_fixed"]) < g125["zero_vel_k1"]
           and k5(g125["prior_fixed"]) >= 3.0 * k_clean[-1])
    g4b_debt = k5(g125["hybrid_fixed"]) > k5(g125["noprior"])
    g4c = k5(g125["prior_nm5"]) <= 0.5 * k5(g125["prior_fixed"])
    g4d = (k5(drag["prior_nmext"]) < k5(drag["prior_nm5"])
           and g125["nmext_params"]["cd"] < 1.5e-4
           and g125["nmext_params"]["cf"] < 150.0)
    g4e = k5(drag["hybrid_nmext"]) <= k5(drag["noprior"])
    report["criteria"] = {"G4_A_净收益": bool(g4a), "G4_B_先验负债": bool(g4b_debt),
                          "G4_C_类内辨识": bool(g4c), "G4_D_类外扩展": bool(g4d),
                          "G4_E_OOS最佳架构": bool(g4e)}
    print("\n" + "=" * 72)
    print("[预注册判决]")
    print(f"  G4-A 净收益（g125: prior_fixed < zero_vel 且 ≥3×clean）: {'[OK]' if g4a else '[FAIL]'} "
          f"(prior {k5(g125['prior_fixed']):.1f} vs zv {g125['zero_vel_k1']:.1f} vs clean {k_clean[-1]:.1f})")
    print(f"  G4-B 先验负债（g125: hybrid_fixed vs noprior）: "
          f"{'负债成立' if g4b_debt else '先验非负债'} "
          f"(hybrid {k5(g125['hybrid_fixed']):.1f} vs noprior {k5(g125['noprior']):.1f})")
    print(f"  G4-C 类内辨识（g125: nm5 ≤ 0.5×fixed）: {'[OK]' if g4c else '[FAIL]'} "
          f"(nm5 {k5(g125['prior_nm5']):.1f} vs fixed {k5(g125['prior_fixed']):.1f})")
    print(f"  G4-D 类外扩展（drag: nmext < nm5 且 g125 下 cd/cf≈下界）: {'[OK]' if g4d else '[FAIL]'}")
    print(f"  G4-E OOS 最佳架构（drag: hybrid_nmext ≤ noprior）: {'[OK]' if g4e else '[FAIL]'} "
          f"(hybrid {k5(drag['hybrid_nmext']):.1f} vs noprior {k5(drag['noprior']):.1f})")

    os.makedirs("analysis_output", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] {args.output}")


if __name__ == "__main__":
    main()
