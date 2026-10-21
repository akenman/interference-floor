# -*- coding: utf-8 -*-
"""
G1+G2 判决实验 (eval_g1g2.py)
==================================================
G1 观测噪声鲁棒性：噪声扫描（观测注入高斯噪声+接触翻转），
   对比 噪声增强训练模型 vs 干净训练模型 vs 惯性(同样吃噪声)/零速度基线。
   预言 P1.1：噪声训练模型误差随噪声优雅退化；P1.2：测量鲁棒边界 σ*。

G2 不确定性校准（σ 头 = "术"的实体化）：
   预言 P2.1：±1σ 覆盖率 ≈0.68；P2.2：σ 在动爻窗口 > 静爻窗口；
   P2.3：σ 随观测噪声水平单调升高。

用法：
  python eval_g1g2.py
"""

import os
import json
import math
import argparse
import numpy as np
import torch

from layout import OBJ_DIMS, OBS_DIMS, ASSEMBLED_DIM, ACTION_DIM, DT_V3, IMPULSE_MAX_V3, NUM_OBJ, PER_OBJ
from model_v3 import WorldModelV3
from train_v3 import load_dataset, split_episodes, norm_stats, V3Data, pos_rmse, inertial_pred, POS_DIMS_OBJ


def apply_noise(xb, sigma):
    """与训练一致的观测噪声：非接触维高斯（逐窗 σ），接触维以 2σ 概率翻转。"""
    if sigma <= 0:
        return xb
    s = torch.full((xb.size(0), 1, 1), sigma, device=xb.device)
    noisy = xb[..., :-3] + torch.randn_like(xb[..., :-3]) * s
    flip = (torch.rand(xb.size(0), xb.size(1), 3, device=xb.device) < s * 2.0).float()
    return torch.cat([noisy, torch.abs(xb[..., -3:] - flip)], dim=-1)


@torch.no_grad()
def run_level(model, ds, device, sigma, max_windows, has_sigma, K_strata=False):
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rng = np.random.RandomState(0)
        idx_all = np.sort(rng.choice(idx_all, size=max_windows, replace=False))
    M = len(idx_all)
    bs = 2048
    pred_one = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    s_last_noisy = np.zeros((M, OBJ_DIMS), dtype=np.float32)
    sig_pos = np.zeros((M, len(POS_DIMS_OBJ)), dtype=np.float32) if has_sigma else None
    yao_t_all = np.zeros((M, 9), dtype=np.int8)
    gua_err_sq = np.zeros(8)
    gua_n = np.zeros(8, dtype=np.int64)

    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = b["x"].to(device)
        xb_n = apply_noise(xb, sigma)
        ab = b["a0"].to(device)
        gua = b["gua"].to(device)

        h, c = model.encode_sequence(xb_n)
        delta = model.predict_state(h, ab, gua).float().cpu().numpy()
        prior_n = model.prior_norm(xb_n[:, -1, :OBJ_DIMS], ab).float().cpu().numpy()
        pred_one[sl] = (prior_n + delta) * (ds.std_obj + 1e-8) + ds.mean_obj
        s_last_noisy[sl] = xb_n[:, -1, :OBJ_DIMS].float().cpu().numpy() * (ds.std_obj + 1e-8) + ds.mean_obj
        yao_t_all[sl] = b["yao"].numpy()
        if has_sigma:
            logvar = model.predict_sigma(h, ab, gua).float().cpu().numpy()
            sig_pos[sl] = np.exp(0.5 * np.clip(logvar, -8, 8))[:, POS_DIMS_OBJ]

        # 卦级 σ 解剖（任意噪声水平都收集，主报告用 σ=0）
        if has_sigma and K_strata:
            tri_true = (yao_t_all[sl][:, 0:3] * 4 + yao_t_all[sl][:, 3:6] * 2
                        + yao_t_all[sl][:, 6:9]).astype(np.int64)
            err_norm = np.abs(b["d_norm"].numpy() - delta)
            for o in range(NUM_OBJ):
                b_ = o * PER_OBJ
                e = err_norm[:, b_ + 1:b_ + 3].mean(axis=1)
                for g in range(8):
                    m = tri_true[:, o] == g
                    if m.any():
                        gua_err_sq[g] += float(np.log(np.maximum(sig_pos[sl][m, 2 * o:2 * o + 2].mean(axis=1), 1e-9)).sum())
                        gua_n[g] += int(m.sum())

    frm_last = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L]
    tgt = frm_last * (ds.std_obj + 1e-8) + ds.mean_obj

    out = {"model_rmse": pos_rmse(pred_one, tgt),
           "zero_vel_rmse": pos_rmse(s_last_noisy, tgt),
           "inertial_rmse": pos_rmse(inertial_pred(s_last_noisy), tgt)}
    if has_sigma:
        err_norm = np.abs(ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L]
                          - (pred_one - ds.mean_obj) / (ds.std_obj + 1e-8))
        ep = err_norm[:, POS_DIMS_OBJ]
        sp = sig_pos
        out["sigma_pos_mean"] = float(sp.mean())
        out["cov1"] = float((ep <= sp).mean())
        out["cov2"] = float((ep <= 2 * sp).mean())
        if K_strata:
            yao_last = ds.yaos[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L - 1].mean(axis=2)
            yao_tgt = ds.yaos[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L].mean(axis=2)
            dongyi = ((yao_last > 0.5) != (yao_tgt > 0.5)).any(axis=1)
            out["sigma_dongyao"] = float(sp[dongyi].mean()) if dongyi.any() else None
            out["sigma_jingyao"] = float(sp[~dongyi].mean()) if (~dongyi).any() else None
            out["sigma_by_gua_logmean"] = [
                float(gua_err_sq[g] / gua_n[g]) if gua_n[g] > 0 else None for g in range(8)]
    return out


def main():
    parser = argparse.ArgumentParser(description="G1+G2 判决实验")
    parser.add_argument("--data_path", type=str, default="trajectories_v3.npz")
    parser.add_argument("--noise_model", type=str, default="checkpoints-v3-calib/best.pt")
    parser.add_argument("--clean_model", type=str, default="checkpoints-v3-nogua/best.pt")
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--val_windows", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_report", type=str, default="analysis_output/g1g2_report.json")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 76)
    print("G1 噪声鲁棒性 + G2 不确定性校准 — 判决实验")
    print("=" * 76)

    states, obstacles, contact, actions = load_dataset(args.data_path)
    _, val_eps = split_episodes(states.shape[0])

    report = {"sigmas": args.sigmas, "models": {}}
    GUA_NAMES = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]

    for name, path, expect_sigma in [("噪声增强训练", args.noise_model, True),
                                     ("干净训练", args.clean_model, False)]:
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = dict(ck["config"])
        if "gua_emb.weight" not in ck["model_state_dict"]:
            cfg["use_gua"] = False
        has_sigma = "sigma_head.0.weight" in ck["model_state_dict"]
        cfg.setdefault("tau", DT_V3)
        cfg.setdefault("gravity", -900.0)
        cfg.setdefault("impulse_max", IMPULSE_MAX_V3)
        model = WorldModelV3(**cfg).to(device)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        model.eval()
        ns = ck["norm_stats"]
        ds = V3Data(states, obstacles, contact, actions, val_eps[:400], 8, 5,
                    ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"])

        print(f"\n[{name}] {path} (σ头: {'有' if has_sigma else '无'})")
        print(f"  {'σ_obs':<8}{'模型RMSE':>10}{'惯性RMSE':>10}{'零速RMSE':>10}"
              + (f"{'平均σ':>10}{'±1σ覆盖':>10}{'±2σ覆盖':>10}" if has_sigma else ""))
        report["models"][name] = {}
        for s in args.sigmas:
            r = run_level(model, ds, device, s, args.val_windows, has_sigma,
                          K_strata=(has_sigma and s == 0.0))
            report["models"][name][f"sigma_{s}"] = r
            line = f"  {s:<8}{r['model_rmse']:>10.3f}{r['inertial_rmse']:>10.3f}{r['zero_vel_rmse']:>10.3f}"
            if has_sigma:
                line += f"{r['sigma_pos_mean']:>10.4f}{r['cov1']:>10.3f}{r['cov2']:>10.3f}"
            print(line)

    # G2 分层判决（噪声训练模型，干净数据）
    calib = report["models"]["噪声增强训练"]["sigma_0.0"]
    if calib.get("sigma_dongyao") is not None:
        ratio = calib["sigma_dongyao"] / max(calib["sigma_jingyao"], 1e-9)
        print(f"\n[G2-P2.2] σ(动爻)={calib['sigma_dongyao']:.4f} vs σ(静爻)={calib['sigma_jingyao']:.4f}"
              f" → 比值 {ratio:.2f}（>1 支持'动爻即惊讶'）")
        report["g2_dongyao_ratio"] = float(ratio)
    if "sigma_by_gua_logmean" in calib:
        logs = calib["sigma_by_gua_logmean"]
        order = sorted(range(8), key=lambda g: -(logs[g] if logs[g] is not None else -9))
        print("[G2-P2.2b] σ 按卦排序（高→低，log σ 位置维均值）: "
              + " > ".join(GUA_NAMES[g] for g in order if logs[g] is not None))

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 报告已保存: {args.output_report}")


if __name__ == "__main__":
    main()
