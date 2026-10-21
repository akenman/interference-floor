# -*- coding: utf-8 -*-
"""SLOT-1b 判决脚本：弹球训练臂先验阶梯（判据以预注册为准）。

预注册：experiments/prereg_S1b_ball_trained.json
输入：checkpoints-S1b/<tier>_s<seed>/best.pt × 12（4 档 × 3 seed）
口径：同 seed 配对（val 划分随 seed，故只在同 seed 内配对）；ITT。
判据（预注册顺序）：
  S1b-1 FF 继承地板：FF-clean one-step 排序 euler001x5 < {parabola,rk4} < euler005（3/3）
  S1b-2 接触帧余量：euler001x5 mixed one-step ≤ 0.80px（先验 1.107 的 ~28% 改进）
  S1b-3 闭环 FF 贴地：euler001x5 FF-clean 闭环 k=3 ≤ 0.50px
  S1b-4 精确档不劣化：mixed one-step euler001x5 ≤ parabola005 × 1.10
次口径：mixed 闭环曲线、yao_acc、崩溃率、每档残差头贡献谱。

用法：python tmp_s1b_compare.py [--val-windows 20000]
输出：控制台判决 + analysis_output/S1b_report.json + figures/S1b_ball_trained_ladder.png
"""
import argparse
import io
import json
import sys

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from layout import OBJ_DIMS, OBS_DIMS, ASSEMBLED_DIM, ACTION_DIM, NUM_OBJ, PER_OBJ, \
    IMPULSE_MAX_V3, DT_V3, obj_pos_dims, F as LF  # noqa: E402
import physics_prior_spec_v3 as PV  # noqa: E402
from model_v3 import WorldModelV3, GatedWorldModelV3  # noqa: E402
from train_v3 import V3Data, load_dataset, split_episodes  # noqa: E402

G_V3 = -900.0
TIERS = ["parabola005", "euler005", "rk4005", "euler001x5"]
SEEDS = [42, 43, 44]
VEL_DIMS = PV.VEL_DIMS                      # 交错 [vx0,vy0,...]
POS6 = obj_pos_dims()                       # 物体段 6 位置维（px）


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, ns = ck["config"], ck["norm_stats"]
    cls = GatedWorldModelV3 if cfg.get("gate") else WorldModelV3
    m = cls(input_dim=ASSEMBLED_DIM, obj_dim=OBJ_DIMS, action_dim=ACTION_DIM,
                     hidden_dim=cfg["hidden_dim"], num_prototypes=cfg["num_prototypes"],
                     num_coarse=cfg["num_coarse"], tau=cfg["tau"], gravity=cfg["gravity"],
                     impulse_max=cfg["impulse_max"], use_gua=cfg["use_gua"],
                     gua_emb_dim=cfg["gua_emb_dim"], prior_spec=cfg.get("prior_spec"))
    m.load_state_dict(ck["model_state_dict"])
    m.set_stats(ns["mean_obj"], ns["std_obj"])
    return m.eval(), ck


def transfer_clean_mask(ds, idx, n_transfers, device_tau=DT_V3):
    """窗口的转移 (L-1→L) 起连续 n_transfers 个转移是否'干净弹道'
    （接触标志双帧为 0 且速度关系 |Δv−(I/m+g)τ|<1）。"""
    ep = ds.ep_idx[idx]
    t0 = ds.win_start[idx] + ds.L - 1
    ok = np.ones(len(idx), dtype=bool)
    for j in range(n_transfers):
        t = t0 + j
        ct = (ds.contact[ep, t].max(axis=1) == 0) & (ds.contact[ep, t + 1].max(axis=1) == 0)
        r0 = ds.obj_norm[ep, t] * (ds.std_obj + 1e-8) + ds.mean_obj
        r1 = ds.obj_norm[ep, t + 1] * (ds.std_obj + 1e-8) + ds.mean_obj
        A = ds.act_norm[ep, t + 1] * IMPULSE_MAX_V3
        m = r0[:, PV.MASS_DIMS]
        m = np.where(m > 1e-3, m, 1.0)
        pred = np.empty((len(idx), 2 * NUM_OBJ), dtype=np.float64)
        for o in range(NUM_OBJ):
            pred[:, 2 * o] = A[:, 2 * o] / m[:, o] * device_tau
            pred[:, 2 * o + 1] = A[:, 2 * o + 1] / m[:, o] * device_tau + G_V3 * device_tau
        dv = (r1[:, VEL_DIMS] - r0[:, VEL_DIMS]).astype(np.float64)
        ok &= ct & (np.abs(dv - pred).max(axis=1) < 1.0)
    return ok


@torch.no_grad()
def eval_model(model, ds, device, idx_all, K):
    """返回 mixed/FF-clean 的 one-step 与闭环 k、yao_acc（pos_rmse px）。"""
    M = len(idx_all)
    bs = 2048
    pred_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    pred_roll = [np.zeros((M, OBJ_DIMS), dtype=np.float32) for _ in range(K)]
    yao_hits = yao_n = 0
    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = b["x"].to(device)
        ab = b["a0"].to(device)
        obb = xb[:, -1, OBJ_DIMS:OBJ_DIMS + OBS_DIMS].contiguous()
        h, _ = model.encode_sequence(xb)
        delta = model.predict_state(h, ab, b["gua"].to(device)).float().cpu().numpy()
        prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab).float().cpu().numpy()
        pred_one[sl] = (prior_n + delta) * (ds.std_obj + 1e-8) + ds.mean_obj
        yao_log = model.predict_yao(h)
        yp = (torch.sigmoid(yao_log.float()) > 0.5).cpu().numpy()
        yt = b["yao"].numpy()
        yao_hits += (yp == yt).mean() * len(idx)
        yao_n += len(idx)
        afb = b["af"].to(device)
        abs_preds, _ = model.rollout(xb, afb, obb, gua0=b["gua"].to(device))
        for k in range(K):
            pred_roll[k][sl] = (abs_preds[k].float().cpu().numpy()
                                * (ds.std_obj + 1e-8) + ds.mean_obj)

    ep = ds.ep_idx[idx_all]
    t0 = ds.win_start[idx_all] + ds.L
    tgt = ds.obj_norm[ep, t0] * (ds.std_obj + 1e-8) + ds.mean_obj
    frm_fut = [ds.obj_norm[ep, t0 + k] * (ds.std_obj + 1e-8) + ds.mean_obj
               for k in range(K)]
    pos6 = lambda p, t: float(np.sqrt(((p[:, POS6] - t[:, POS6]) ** 2).mean()))

    one_clean_mask = transfer_clean_mask(ds, idx_all, 1)
    cl_clean_mask = transfer_clean_mask(ds, idx_all, K)
    out = dict(
        one_mixed=pos6(pred_one, tgt),
        one_ff_clean=pos6(pred_one[one_clean_mask], tgt[one_clean_mask])
        if one_clean_mask.any() else float("nan"),
        n_one_clean=int(one_clean_mask.sum()),
        roll_mixed={f"k={k+1}": pos6(pred_roll[k], frm_fut[k]) for k in range(K)},
        roll_ff_clean={f"k={k+1}": pos6(pred_roll[k][cl_clean_mask], frm_fut[k][cl_clean_mask])
                       for k in range(K)},
        n_cl_clean=int(cl_clean_mask.sum()),
        yao_acc=float(yao_hits / max(yao_n, 1)),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-windows", type=int, default=20000)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("载入 trajectories_v3.npz …", flush=True)
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]

    res = {}
    for tier in TIERS:
        res[tier] = {}
        for s in SEEDS:
            path = f"checkpoints-S1b/{tier}_s{s}/best.pt"
            try:
                model, ck = load_model(path)
            except FileNotFoundError:
                print(f"[缺] {path} —— 记 ITT 崩溃")
                res[tier][s] = None
                continue
            ds_args = ck["args"]
            L, K = ds_args["seq_len"], ds_args["future_steps"]
            train_eps, val_eps = split_episodes(N, s)
            ns = ck["norm_stats"]
            ds = V3Data(states, obstacles, contact, actions, val_eps, L, K,
                        ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"],
                        prior_spec=ck["config"]["prior_spec"])
            rng = np.random.RandomState(0)
            idx_all = np.arange(len(ds))
            if args.val_windows and len(idx_all) > args.val_windows:
                idx_all = np.sort(rng.choice(idx_all, size=args.val_windows, replace=False))
            res[tier][s] = eval_model(model.to(device), ds, device, idx_all, K)
            r = res[tier][s]
            print(f"[{tier} s{s}] mixed={r['one_mixed']:.3f}  FF-clean={r['one_ff_clean']:.4f}"
                  f"  roll_mixed k={K}={r['roll_mixed'][f'k={K}']:.3f}"
                  f"  roll_ff k={K}={r['roll_ff_clean'][f'k={K}']:.4f}"
                  f"  yao={r['yao_acc']:.3f}", flush=True)
            del model
            torch.cuda.empty_cache()

    # ---------- 判决（预注册顺序） ----------
    def m(t, key="one_mixed"):
        vals = [res[t][s][key] for s in SEEDS if res[t][s]]
        return float(np.mean(vals)) if vals else float("nan")

    def per_seed(t, key):
        return np.array([res[t][s][key] if res[t][s] else np.nan for s in SEEDS])

    out = {"prereg": "experiments/prereg_S1b_ball_trained.json", "results": res}
    print("\n=== 判决（预注册顺序）===")
    # S1b-1
    ok1 = []
    for s in SEEDS:
        e = res["euler001x5"][s]["one_ff_clean"]
        p, r = res["parabola005"][s]["one_ff_clean"], res["rk4005"][s]["one_ff_clean"]
        u = res["euler005"][s]["one_ff_clean"]
        ok1.append(np.isfinite([e, p, r, u]).all() and e < min(p, r) < u)
    out["S1b-1"] = dict(per_seed_ok=ok1, verdict=bool(all(ok1)))
    print(f"  S1b-1 FF 继承地板（排序 3/3）: {'[OK]' if all(ok1) else '[FAIL]'}  {ok1}")
    # S1b-2
    e_mixed = m("euler001x5")
    out["S1b-2"] = dict(euler_mixed=e_mixed, threshold=0.80,
                        verdict=bool(e_mixed <= 0.80))
    print(f"  S1b-2 接触帧余量 (≤0.80px): {'[OK]' if e_mixed <= 0.80 else '[FAIL]'}"
          f"  ({e_mixed:.3f}px)")
    # S1b-3
    e_k3 = float(np.mean([res["euler001x5"][s]["roll_ff_clean"]["k=3"]
                          for s in SEEDS if res["euler001x5"][s]]))
    out["S1b-3"] = dict(euler_roll_ff_k3=e_k3, threshold=0.50,
                        verdict=bool(e_k3 <= 0.50))
    print(f"  S1b-3 闭环 FF 贴地 (k=3 ≤0.50px): {'[OK]' if e_k3 <= 0.50 else '[FAIL]'}"
          f"  ({e_k3:.4f}px)")
    # S1b-4
    ratio = e_mixed / m("parabola005")
    out["S1b-4"] = dict(ratio=ratio, threshold=1.10, verdict=bool(ratio <= 1.10))
    print(f"  S1b-4 精确档不劣化 (≤×1.10): {'[OK]' if ratio <= 1.10 else '[FAIL]'}"
          f"  (×{ratio:.3f})")
    # 次口径
    out["summary"] = {t: dict(
        one_mixed=m(t),
        one_ff_clean=float(np.mean([res[t][s]["one_ff_clean"] for s in SEEDS
                                    if res[t][s] and np.isfinite(res[t][s]["one_ff_clean"])])),
        roll_mixed_k3=float(np.mean([res[t][s]["roll_mixed"]["k=3"] for s in SEEDS
                                     if res[t][s]])),
        yao_acc=float(np.mean([res[t][s]["yao_acc"] for s in SEEDS if res[t][s]])),
    ) for t in TIERS}
    crash = sum(res[t][s] is None for t in TIERS for s in SEEDS) / 12
    out["crash_rate_ITT"] = crash
    print("  次口径 mixed one-step: " +
          "  ".join(f"{t}={out['summary'][t]['one_mixed']:.3f}" for t in TIERS))
    print(f"  崩溃率(ITT): {crash:.0%}")

    with open("analysis_output/S1b_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/S1b_report.json")

    if not args.no_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            for ax, (tag, lab) in zip(axes, (
                    ("one_mixed", "mixed one-step (px)"),
                    ("one_ff_clean", "free-flight-clean one-step (px)"))):
                vals = [out["summary"][t][tag] for t in TIERS]
                ax.bar(np.arange(4), vals, color="#4C72B0")
                ax.set_xticks(np.arange(4))
                ax.set_xticklabels(TIERS, rotation=20, fontsize=8)
                ax.set_title(lab)
            fig.suptitle("SLOT-1b: ball-world trained prior ladder (means over 3 seeds)")
            fig.tight_layout()
            fig.savefig("figures/S1b_ball_trained_ladder.png", dpi=150)
            print("[save] figures/S1b_ball_trained_ladder.png")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 出图失败（不影响判决）: {e}")


if __name__ == "__main__":
    main()
