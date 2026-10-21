# -*- coding: utf-8 -*-
"""
E4 预检 pilot (eval_gua_gate_pilot.py) — 卦置信度门控的零训练生死判据
======================================================================
来源：LONG_TASK_PLAN.md P0 · 2026-09-21
动机：Jev 式"校准结构化决策"在本项目的落地——P(卦) 校准后的置信度 α
      是否携带"何时会错"的信息（高置信用学习预测 / 低置信弃权退回物理先验）。

## 预注册（写入时间：2026-09-21，先于任何数字）
  C1 仪器检查：温度缩放后 8 卦分类 ECE < 0.05
  C2 生死判据：低置信 20% 样本 RMSE / 高置信 20% 样本 RMSE ≥ 2×，
     且置信度与误差的 Spearman ρ ≤ −0.4
  C3 门控价值（模拟）：按 α 在学习头/先验间切换的最佳全体 RMSE
     严格优于"全用学习头"与"全用先验"两者的较小者，
     且接触子集（下一帧初爻=1）改善 ≥10%
  判决：C2 且 C3 过 → GO（排入 E4 全实验）；C2 不过 → 路线判死，
        E4 退回 CQR 宽度做门控信号。
  世界选择：弹球（先验是资产、卦 8/8 覆盖）；铰链爻 acc≈0.999 置信度退化，
        且先验是负债（E0），不做。

用法：
  python eval_gua_gate_pilot.py [--model checkpoints-v3-calib2/best.pt]
"""

import os
import json
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr

from layout import OBJ_DIMS, DT_V3, IMPULSE_MAX_V3, NUM_OBJ, PER_OBJ
from model_v3 import WorldModelV3
from train_v3 import load_dataset, split_episodes, V3Data

GUA_NAMES = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]

# 预注册门槛（修改即作废本次预注册）
TH_ECE = 0.05
TH_RATIO = 2.0
TH_SPEARMAN = -0.4
TH_CONTACT_GAIN = 0.10


@torch.no_grad()
def collect(model, ds, device, max_windows):
    """逐窗口收集：三爻 logits / 模型与先验的逐物体位置误差(px) / 真值三爻。"""
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rng = np.random.RandomState(0)
        idx_all = np.sort(rng.choice(idx_all, size=max_windows, replace=False))
    M = len(idx_all)
    bs = 2048
    yao_logits = np.zeros((M, NUM_OBJ * 3), dtype=np.float32)
    yao_true = np.zeros((M, NUM_OBJ * 3), dtype=np.int8)
    err2_model = np.zeros((M, NUM_OBJ), dtype=np.float32)   # 逐物体位置维均方误差(px^2)
    err2_prior = np.zeros((M, NUM_OBJ), dtype=np.float32)

    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = b["x"].to(device)
        ab = b["a0"].to(device)
        gua = b["gua"].to(device) if model.use_gua else None

        h, _ = model.encode_sequence(xb)
        yao_logits[sl] = model.predict_yao(h).float().cpu().numpy()
        yao_true[sl] = b["yao"].numpy()
        delta = model.predict_state(h, ab, gua).float().cpu().numpy()
        prior_n = model.prior_norm(xb[:, -1, :OBJ_DIMS], ab).float().cpu().numpy()

        frm = ds.obj_norm[ds.ep_idx[idx], ds.win_start[idx] + ds.L]
        tgt = frm * (ds.std_obj + 1e-8) + ds.mean_obj                      # 物理单位
        pred = (prior_n + delta) * (ds.std_obj + 1e-8) + ds.mean_obj
        prior = prior_n * (ds.std_obj + 1e-8) + ds.mean_obj
        for o in range(NUM_OBJ):
            p0 = o * PER_OBJ + 1
            d_m = pred[:, p0:p0 + 2] - tgt[:, p0:p0 + 2]
            d_p = prior[:, p0:p0 + 2] - tgt[:, p0:p0 + 2]
            err2_model[sl, o] = (d_m ** 2).mean(axis=1)
            err2_prior[sl, o] = (d_p ** 2).mean(axis=1)
    return yao_logits, yao_true, err2_model, err2_prior


def gua_probs_from_yao(yao_logits, temperature=1.0):
    """三爻独立伯努利 → 8 卦概率。(N, 9) → (N*NUM_OBJ, 8)"""
    p = 1.0 / (1.0 + np.exp(-yao_logits / max(temperature, 1e-6)))
    N = p.shape[0]
    p = p.reshape(N * NUM_OBJ, 3)                       # 每物体三爻
    probs = np.ones((N * NUM_OBJ, 8), dtype=np.float64)
    for bit in range(3):                                # bit0=初爻(权重4) bit1(2) bit2(1)
        for g in range(8):
            one = (g >> (2 - bit)) & 1
            probs[:, g] *= np.where(one, p[:, bit], 1.0 - p[:, bit])
    return probs / probs.sum(axis=1, keepdims=True)


def ece_binary_conf(conf, correct, n_bins=15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if m.any():
            ece += m.sum() / n * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def main():
    ap = argparse.ArgumentParser(description="E4 预检：卦置信度门控 pilot")
    ap.add_argument("--model", type=str, default="checkpoints-v3-calib2/best.pt")
    ap.add_argument("--data_path", type=str, default="trajectories_v3.npz")
    ap.add_argument("--val_episodes", type=int, default=400)
    ap.add_argument("--val_windows", type=int, default=10000)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output_report", type=str,
                    default="analysis_output/gua_gate_pilot_report.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    print("=" * 76)
    print("E4 预检 pilot — 卦置信度门控（预注册判据见脚本头部，2026-09-21 写入）")
    print("=" * 76)

    # ---- 加载模型（纪律：config 不可信，state_dict 反推） ----
    ck = torch.load(args.model, map_location=device, weights_only=False)
    cfg = dict(ck["config"])
    sd = ck["model_state_dict"]
    if "gua_emb.weight" not in sd:
        cfg["use_gua"] = False
    cfg.setdefault("tau", DT_V3)
    cfg.setdefault("gravity", -900.0)
    cfg.setdefault("impulse_max", IMPULSE_MAX_V3)
    assert any("yao" in k for k in sd), "无三爻头，pilot 无法进行"
    model = WorldModelV3(**cfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[模型] {args.model} | use_gua={model.use_gua} | use_innov={model.use_innov}")

    states, obstacles, contact, actions = load_dataset(args.data_path)
    _, val_eps = split_episodes(states.shape[0])
    ns = ck["norm_stats"]
    ds = V3Data(states, obstacles, contact, actions, val_eps[:args.val_episodes], 8, 5,
                ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"])

    yao_logits, yao_true, err2_m, err2_p = collect(model, ds, device, args.val_windows)
    Nw = yao_logits.shape[0]
    print(f"[数据] 窗口 {Nw} × 物体 {NUM_OBJ} = {Nw * NUM_OBJ} 个物体级样本")

    yao_t = yao_true.reshape(Nw * NUM_OBJ, 3)
    tri_true = (yao_t[:, 0] * 4 + yao_t[:, 1] * 2 + yao_t[:, 2]).astype(np.int64)
    e_m = err2_m.reshape(-1).astype(np.float64)
    e_p = err2_p.reshape(-1).astype(np.float64)
    contact_next = yao_t[:, 0].astype(bool)             # 下一帧初爻=接触

    # ---- 卦占用检查（E0 教训：先查 occupancy 再谈门控） ----
    occ = np.bincount(tri_true, minlength=8) / len(tri_true)
    print("[卦占用] " + " ".join(f"{GUA_NAMES[g]}:{occ[g]*100:.1f}%" for g in range(8)))

    # ---- 温度缩放：前半拟合 T，后半评估（防过拟合校准集） ----
    half = len(tri_true) // 2
    best_t, best_nll = 1.0, np.inf
    for t in np.arange(0.5, 5.01, 0.25):
        pr = gua_probs_from_yao(yao_logits, t)
        nll = -np.log(np.clip(pr[:half], 1e-12, 1))[
            np.arange(half), tri_true[:half]].mean()
        if nll < best_nll:
            best_nll, best_t = nll, t
    probs = gua_probs_from_yao(yao_logits, best_t)
    conf = probs.max(axis=1)
    pred_gua = probs.argmax(axis=1)
    correct = (pred_gua == tri_true).astype(np.float64)

    te = slice(half, None)
    acc = float(correct[te].mean())
    ece = ece_binary_conf(conf[te], correct[te])
    c1_pass = ece < TH_ECE
    print(f"[C1 校准] 温度 T={best_t:.2f} | 卦 acc={acc:.3f} | ECE={ece:.4f} "
          f"(门槛 <{TH_ECE}) → {'过' if c1_pass else '不过'}")

    # ---- C2 生死判据：置信度信息量 ----
    c, e = conf[te], e_m[te]
    q20, q80 = np.quantile(c, 0.2), np.quantile(c, 0.8)
    rmse_lo = float(np.sqrt(e[c <= q20].mean()))
    rmse_hi = float(np.sqrt(e[c >= q80].mean()))
    ratio = rmse_lo / max(rmse_hi, 1e-9)
    rho, pval = spearmanr(c, e)
    c2_pass = (ratio >= TH_RATIO) and (rho <= TH_SPEARMAN)
    print(f"[C2 信息量] 低置信20% RMSE={rmse_lo:.3f}px vs 高置信20% "
          f"RMSE={rmse_hi:.3f}px → 比值 {ratio:.2f}x（门槛 >={TH_RATIO}）")
    print(f"           Spearman rho={rho:.3f} (p={pval:.2e})（门槛 <={TH_SPEARMAN}）")
    print(f"           → {'过' if c2_pass else '不过'}")

    # ---- C3 门控价值模拟：α 阈值扫描，学习头/先验二选一 ----
    rmse_model = float(np.sqrt(e_m[te].mean()))
    rmse_prior = float(np.sqrt(e_p[te].mean()))
    base_best = min(rmse_model, rmse_prior)
    cm = contact_next[te]
    best = None
    for tau in np.unique(np.quantile(c, np.linspace(0, 0.95, 40))):
        use_model = c >= tau
        rmse_g = float(np.sqrt(np.where(use_model, e_m[te], e_p[te]).mean()))
        rmse_gc = float(np.sqrt(np.where(use_model[cm], e_m[te][cm],
                                         e_p[te][cm]).mean())) if cm.any() else None
        if best is None or rmse_g < best["rmse_gated"]:
            best = dict(tau=float(tau), rmse_gated=rmse_g, rmse_gated_contact=rmse_gc,
                        frac_model=float(use_model.mean()))
    rmse_mc = float(np.sqrt(e_m[te][cm].mean()))
    rmse_pc = float(np.sqrt(e_p[te][cm].mean()))
    base_c = min(rmse_mc, rmse_pc)
    gain_all = (base_best - best["rmse_gated"]) / base_best
    gain_c = (base_c - best["rmse_gated_contact"]) / max(base_c, 1e-9)
    c3_pass = (best["rmse_gated"] < base_best) and (gain_c >= TH_CONTACT_GAIN)
    print(f"[C3 门控模拟] 全体: 学习 {rmse_model:.3f} / 先验 {rmse_prior:.3f} / "
          f"门控(tau={best['tau']:.3f}, 用学习头 {best['frac_model']*100:.0f}%) "
          f"{best['rmse_gated']:.3f}px → 增益 {gain_all*100:.1f}%")
    print(f"             接触子集: 学习 {rmse_mc:.3f} / 先验 {rmse_pc:.3f} / "
          f"门控 {best['rmse_gated_contact']:.3f}px → 增益 {gain_c*100:.1f}%"
          f"（门槛 >={TH_CONTACT_GAIN*100:.0f}%）")
    print(f"             → {'过' if c3_pass else '不过'}")

    verdict = "GO" if (c2_pass and c3_pass) else "NO-GO"
    print(f"\n{'=' * 76}\n判决: {verdict}"
          f"（C1={'过' if c1_pass else '不过'} C2={'过' if c2_pass else '不过'} "
          f"C3={'过' if c3_pass else '不过'}）\n{'=' * 76}")

    report = {
        "prereg": {"written": "2026-09-21", "TH_ECE": TH_ECE, "TH_RATIO": TH_RATIO,
                   "TH_SPEARMAN": TH_SPEARMAN, "TH_CONTACT_GAIN": TH_CONTACT_GAIN},
        "model": args.model, "use_gua": model.use_gua,
        "n_windows": int(Nw), "n_samples": int(len(tri_true)),
        "gua_occupancy": {GUA_NAMES[g]: float(occ[g]) for g in range(8)},
        "C1": {"temperature": float(best_t), "gua_acc": acc, "ece": ece,
               "pass": bool(c1_pass)},
        "C2": {"rmse_low_conf20": rmse_lo, "rmse_high_conf20": rmse_hi,
               "ratio": ratio, "spearman_rho": float(rho), "p": float(pval),
               "pass": bool(c2_pass)},
        "C3": {"rmse_model": rmse_model, "rmse_prior": rmse_prior,
               "best_tau": best["tau"], "rmse_gated": best["rmse_gated"],
               "frac_model": best["frac_model"], "gain_overall": float(gain_all),
               "rmse_model_contact": rmse_mc, "rmse_prior_contact": rmse_pc,
               "rmse_gated_contact": best["rmse_gated_contact"],
               "gain_contact": float(gain_c), "pass": bool(c3_pass)},
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] 报告已保存: {args.output_report}")


if __name__ == "__main__":
    main()
