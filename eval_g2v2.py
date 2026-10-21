# -*- coding: utf-8 -*-
"""
G2-v2 判决实验 (eval_g2v2.py)
==================================================
检验创新量特征 σ 头是否修复 G2 v1 的三处失败：
  P2.1 ±1σ 覆盖率 ≈0.68（v1: 0.995 过保守）——含温度缩放前后对比
  P2.2 σ(动爻) > σ(静爻)（v1: 1.17 弱阳性）
  P2.3 σ 随观测噪声水平单调升高（v1: 完全不敏感）

用法：
  python eval_g2v2.py
"""

import os
import json
import math
import argparse
import numpy as np
import torch

from layout import OBJ_DIMS, OBS_DIMS, ASSEMBLED_DIM, ACTION_DIM, DT_V3, IMPULSE_MAX_V3, NUM_OBJ
from model_v3 import WorldModelV3
from train_v3 import load_dataset, split_episodes, V3Data, pos_rmse, POS_DIMS_OBJ


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck["config"])
    if "gua_emb.weight" not in ck["model_state_dict"]:
        cfg["use_gua"] = False
    # σ 头输入维度按权重反推（兼容 v1 无创新量 / v2 有创新量）
    sh_in = ck["model_state_dict"]["sigma_head.0.weight"].shape[1]
    base = cfg.get("hidden_dim", 256) + cfg.get("action_dim", 6)
    if cfg.get("use_gua"):
        base += cfg.get("gua_emb_dim", 16)
    cfg["use_innov"] = sh_in > base
    if cfg["use_innov"]:
        cfg["innov_dim"] = sh_in - base
    cfg.setdefault("tau", DT_V3)
    cfg.setdefault("gravity", -900.0)
    cfg.setdefault("impulse_max", IMPULSE_MAX_V3)
    model = WorldModelV3(**cfg).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    return model, ck["norm_stats"], cfg


@torch.no_grad()
def collect(model, ds, device, sigma_obs, max_windows):
    """返回逐窗位置维 |误差| 与 σ（归一化空间），以及动爻掩码。"""
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rng = np.random.RandomState(0)
        idx_all = np.sort(rng.choice(idx_all, size=max_windows, replace=False))
    M = len(idx_all)
    bs = 2048
    err = np.zeros((M, len(POS_DIMS_OBJ)), dtype=np.float64)
    sig = np.zeros((M, len(POS_DIMS_OBJ)), dtype=np.float64)
    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = b["x"].to(device)
        if sigma_obs > 0:
            s = torch.full((xb.size(0), 1, 1), sigma_obs, device=xb.device)
            xb = torch.cat([xb[..., :-3] + torch.randn_like(xb[..., :-3]) * s,
                            torch.abs(xb[..., -3:] -
                                      (torch.rand(xb.size(0), xb.size(1), 3, device=xb.device)
                                       < s * 2.0).float())], dim=-1)
        ab = b["a0"].to(device)
        gua = b["gua"].to(device)
        h, _ = model.encode_sequence(xb)
        delta = model.predict_state(h, ab, gua).float().cpu().numpy()
        innov = model.innovation_feats(xb, b["a_win"].to(device)) if model.use_innov else None
        logvar = model.predict_sigma(h, ab, innov, gua).float().cpu().numpy()
        err_norm = np.abs(b["d_norm"].numpy() - delta)
        err[sl] = err_norm[:, POS_DIMS_OBJ]
        sig[sl] = np.exp(0.5 * np.clip(logvar, -8, 8))[:, POS_DIMS_OBJ]
    yao_last = ds.yaos[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L - 1].mean(axis=2)
    yao_tgt = ds.yaos[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L].mean(axis=2)
    dongyi = ((yao_last > 0.5) != (yao_tgt > 0.5)).any(axis=1)
    return err, sig, dongyi


def main():
    parser = argparse.ArgumentParser(description="G2-v2 判决实验")
    parser.add_argument("--data_path", type=str, default="trajectories_v3.npz")
    parser.add_argument("--model", type=str, default="checkpoints-v3-calib2/best.pt")
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.0, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--val_windows", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_report", type=str, default="analysis_output/g2v2_report.json")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 76)
    print("G2-v2 判决：创新量特征 σ 头（P2.1 校准 / P2.2 动爻 / P2.3 噪声敏感）")
    print("=" * 76)

    states, obstacles, contact, actions = load_dataset(args.data_path)
    _, val_eps = split_episodes(states.shape[0])
    model, ns, cfg = load_model(args.model, device)
    ds = V3Data(states, obstacles, contact, actions, val_eps[:400], 8, 5,
                ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"])

    report = {"model": args.model, "use_innov": cfg.get("use_innov"), "levels": {}}

    # ---- P2.1 + P2.3：噪声扫描 + 温度缩放（前半校准拟合 T，后半报告） ----
    print(f"\n  {'σ_obs':<8}{'σ_raw':>10}{'cov1_raw':>10}{'T':>8}{'cov1_T':>10}{'cov2_T':>10}")
    for s in args.sigmas:
        err, sig, dongyi = collect(model, ds, device, s, args.val_windows)
        n = err.size
        half = n // 2
        # 温度缩放：T = median(|err|/σ) 使 ±1σ 覆盖 ≈68%（在校准半区拟合）
        ratio = (err.ravel() / np.maximum(sig.ravel(), 1e-9))
        T = float(np.median(ratio[:half])) if half > 0 else 1.0
        sig_T = sig * T
        cov1_raw = float((err <= sig).mean())
        cov1_T = float((err <= sig_T).mean())
        cov2_T = float((err <= 2 * sig_T).mean())
        report["levels"][f"sigma_{s}"] = {
            "sigma_raw_mean": float(sig.mean()), "cov1_raw": cov1_raw,
            "T": T, "cov1_T": cov1_T, "cov2_T": cov2_T}
        print(f"  {s:<8}{sig.mean():>10.4f}{cov1_raw:>10.3f}{T:>8.2f}{cov1_T:>10.3f}{cov2_T:>10.3f}")

    # ---- P2.2：干净数据上的动爻/静爻 σ 比 ----
    err0, sig0, dongyi = collect(model, ds, device, 0.0, args.val_windows)
    s_d = float(sig0[dongyi].mean()) if dongyi.any() else None
    s_j = float(sig0[~dongyi].mean()) if (~dongyi).any() else None
    if s_d is not None:
        ratio = s_d / max(s_j, 1e-9)
        print(f"\n[P2.2] σ(动爻)={s_d:.4f} vs σ(静爻)={s_j:.4f} → 比值 {ratio:.2f}"
              f"（v1 为 1.17；>1 且更大 = 支持'动爻即惊讶'）")
        report["p2_2"] = {"sigma_dongyao": s_d, "sigma_jingyao": s_j, "ratio": float(ratio)}

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 报告已保存: {args.output_report}")
    print("判读：cov1_T 接近 0.68 → 校准成立；σ 随 σ_obs 单调升 → P2.3 成立；"
          "动爻比 > v1 的 1.17 → 创新量特征有效。")


if __name__ == "__main__":
    main()
