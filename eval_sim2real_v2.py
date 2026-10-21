# -*- coding: utf-8 -*-
"""微缩 sim-to-real 判决脚本 v2（独立文件，避免内联脚本的索引 bug 家族）。"""
import math
import json
import numpy as np
import torch
from model_h import WorldModelH
from layout_h import OBJ_DIMS, L1, L2, M1, M2, TAU_MAX

L, K = 8, 5
POS = [1, 2, 12, 13]


def prior_one_step(state22, tau_norm=0.0):
    """物理先验：0.05s 半隐式欧拉一步（与 model_h.prior_norm 同源方程）。"""
    g = 900.0
    th1 = math.atan2(state22[3], state22[4])
    th2 = math.atan2(state22[14], state22[15])
    w1 = float(state22[7]); w2 = float(state22[18])
    tau = tau_norm * TAU_MAX
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
    out[3] = math.sin(th1n); out[4] = math.cos(th1n)
    out[14] = math.sin(th2n); out[15] = math.cos(th2n)
    out[5] = L1*w1n*math.cos(th1n); out[6] = L1*w1n*math.sin(th1n)
    out[16] = out[5] + L2*(w1n+w2n)*math.cos(th1n+th2n)
    out[17] = out[6] + L2*(w1n+w2n)*math.sin(th1n+th2n)
    out[7] = w1n; out[18] = w2n
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--pseudo", type=str, default="pseudo_real_hinge.npz")
    ap.add_argument("--ckpt", type=str, default="checkpoints-hinge/best.pt")
    ap.add_argument("--max_ep", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    d_clean = np.load(args.clean)
    d_pseudo = np.load(args.pseudo)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = WorldModelH(**ck["config"]).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    ns = ck["norm_stats"]
    mean_obj = ns["mean_obj"]; std_obj = ns["std_obj"]

    @torch.no_grad()
    def eval_data(states_raw, contact_raw, label):
        N, T = states_raw.shape[:2]
        n_win = T - L - K + 1
        prior_e, model_e, zero_e = [], [], []
        for ep in range(min(N, args.max_ep)):
            frames = np.concatenate([
                (states_raw[ep] - mean_obj) / (std_obj + 1e-8),
                contact_raw[ep].astype(np.float32)
            ], axis=1)   # (T, 25)
            for wi in range(min(n_win, 30)):
                i = wi
                now22 = states_raw[ep, i + L - 1]     # (22,) raw
                next22 = states_raw[ep, i + L]        # (22,) raw
                pt = [next22[1], next22[2], next22[12], next22[13]]
                pz = [now22[1], now22[2], now22[12], now22[13]]
                pr = prior_one_step(now22, 0.0)
                pp = [pr[1], pr[2], pr[12], pr[13]]
                seq = torch.from_numpy(frames[i:i+L][None].astype(np.float32)).to(device)
                h, _ = model.encode_sequence(seq)
                obj_last_n = seq[:, -1, :OBJ_DIMS]
                act = torch.zeros(1, 1, device=device)
                delta = model.predict_state(h, act).float().cpu().numpy()[0]
                p_n = model.prior_norm(obj_last_n, act).float().cpu().numpy()[0]
                pred22 = (p_n + delta) * (std_obj + 1e-8) + mean_obj
                pm = [pred22[1], pred22[2], pred22[12], pred22[13]]
                prior_e.append(np.abs(np.array(pp) - np.array(pt)).mean())
                model_e.append(np.abs(np.array(pm) - np.array(pt)).mean())
                zero_e.append(np.abs(np.array(pz) - np.array(pt)).mean())
        return (float(np.mean(prior_e)), float(np.mean(model_e)),
                float(np.mean(zero_e)), len(prior_e))

    pr_c, mo_c, zv_c, n_c = eval_data(d_clean["states"][:500], d_clean["contact"][:500], "clean")
    pr_p, mo_p, zv_p, n_p = eval_data(d_pseudo["states"][:500], d_pseudo["contact"][:500], "pseudo")

    print("=" * 68)
    print("微缩 sim-to-real 判决：伪真实数据上的架构存活测试")
    print("=" * 68)
    print(f"\n{'来源':<14}{'先验MAE':>10}{'模型MAE':>10}{'零速MAE':>10}{'样本':>8}")
    for label, pr, mo, zv, n in [("clean", pr_c, mo_c, zv_c, n_c),
                                  ("pseudo-real", pr_p, mo_p, zv_p, n_p)]:
        print(f"{label:<14}{pr:>10.2f}{mo:>10.2f}{zv:>10.2f}{n:>8}")

    r = mo_p / max(pr_p, 1e-6)
    ok1 = r < 2.0
    ok3 = mo_p < zv_p
    deg = (mo_p - mo_c) / max(mo_c, 1e-6) * 100
    print(f"\nPR-1 graceful degradation: model/prior = {r:.2f} → {'[OK] PASS' if ok1 else '[FAIL] FAIL'}")
    print(f"PR-3 residual survival: model {mo_p:.2f} < zero-vel {zv_p:.2f} → {'[OK] PASS' if ok3 else '[FAIL] FAIL'}")
    print(f"Degradation: clean {mo_c:.2f} → pseudo-real {mo_p:.2f} px (+{deg:.0f}%)")

    if ok1 and ok3:
        print("\nVERDICT: [OK] Architecture SURVIVES pseudo-real conditions.")
    elif ok1:
        print("\nVERDICT: ◑ Graceful degradation holds, residual value lost.")
    else:
        print("\nVERDICT: [FAIL] Architecture FAILS under pseudo-real conditions.")

    report = {"clean": {"prior": pr_c, "model": mo_c, "zero_vel": zv_c},
              "pseudo": {"prior": pr_p, "model": mo_p, "zero_vel": zv_p},
              "criteria": {"PR1": bool(ok1), "PR3": bool(ok3)},
              "degradation_pct": round(float(deg), 1)}
    with open("analysis_output/sim2real_v2_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] analysis_output/sim2real_v2_report.json")


if __name__ == "__main__":
    main()
