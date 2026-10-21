# -*- coding: utf-8 -*-
"""SLOT-1c 判决脚本：regime 门控静默（判据以预注册为准）。

预注册：experiments/prereg_S1c_regime_silence.json
对照臂 = SLOT-1b 的 euler001x5_s42-44（同 recipe 同 seed，零重训）；
处理臂 = wff10（regime 重加权 ×10）与 gate（静默门控头），各 3 seed。

判据（预注册顺序，同 seed 配对 vs 基线）：
  R1 静默：FF-clean one-step ≤ 0.50px（基线 0.861）
  R2 不伤接触：mixed one-step ≤ 0.628px（基线 0.571 × 1.10）
  R3 闭环缓解：FF-clean 闭环 k=3 ≤ 1.02px（基线 2.040）
  R4 探针（只报告）：FF-clean 帧头输出 RMS（基线 0.0197 归一化口径）下降倍数

用法：python tmp_s1c_compare.py [--val-windows 20000] [--no-figure]
输出：控制台判决 + analysis_output/S1c_report.json + figures/S1c_silence.png
"""
import argparse
import json
import sys

import numpy as np
import torch

# 注意：tmp_s1b_compare 在导入时会把 sys.stdout 包成 utf-8 TextIOWrapper——
# 本模块不得再包一层（双重包装会关闭底层 buffer，实测 2026-10-03）。
from layout import OBJ_DIMS  # noqa: E402
from tmp_s1b_compare import load_model, transfer_clean_mask, eval_model  # noqa: E402
from train_v3 import V3Data, load_dataset, split_episodes  # noqa: E402

SEEDS = [42, 43, 44]
ARMS = {"baseline": "checkpoints-S1b/euler001x5_s{s}/best.pt",
        "wff10": "checkpoints-S1c/wff10_s{s}/best.pt",
        "gate": "checkpoints-S1c/gate_s{s}/best.pt"}


@torch.no_grad()
def head_rms_on_ff(model, ds, device, idx_all):
    """FF-clean 单步窗口上头输出（predict_state）的 RMS（归一化，位置维）。"""
    mask = transfer_clean_mask(ds, idx_all, 1)
    sub = idx_all[mask]
    if len(sub) == 0:
        return float("nan"), 0
    b = ds.gather(sub)
    xb = b["x"].to(device)
    ab = b["a0"].to(device)
    h, _ = model.encode_sequence(xb)
    delta = model.predict_state(h, ab, b["gua"].to(device)).float().cpu().numpy()
    pos = [1, 2, 12, 13, 23, 24]
    return float(np.sqrt((delta[:, pos] ** 2).mean())), int(len(sub))


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
    for arm, pat in ARMS.items():
        res[arm] = {}
        for s in SEEDS:
            path = pat.format(s=s)
            try:
                model, ck = load_model(path)
            except FileNotFoundError:
                print(f"[缺] {path} —— ITT 崩溃")
                res[arm][s] = None
                continue
            ds_args = ck["args"]
            train_eps, val_eps = split_episodes(N, s)
            ns = ck["norm_stats"]
            ds = V3Data(states, obstacles, contact, actions, val_eps,
                        ds_args["seq_len"], ds_args["future_steps"],
                        ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"],
                        prior_spec=ck["config"]["prior_spec"])
            rng = np.random.RandomState(0)
            idx_all = np.arange(len(ds))
            if args.val_windows and len(idx_all) > args.val_windows:
                idx_all = np.sort(rng.choice(idx_all, size=args.val_windows,
                                             replace=False))
            r = eval_model(model.to(device), ds, device, idx_all, ds_args["future_steps"])
            r["head_rms_ff"], r["n_ff"] = head_rms_on_ff(model, ds, device, idx_all)
            res[arm][s] = r
            print(f"[{arm} s{s}] mixed={r['one_mixed']:.3f}  FF={r['one_ff_clean']:.4f}"
                  f"  roll_ff k3={r['roll_ff_clean']['k=3']:.4f}"
                  f"  头RMS_ff={r['head_rms_ff']:.4f}  yao={r['yao_acc']:.3f}", flush=True)
            del model
            torch.cuda.empty_cache()

    def m(arm, key):
        vals = []
        for s in SEEDS:
            r = res[arm][s]
            if not r:
                continue
            v = r.get(key, float("nan"))
            if isinstance(v, dict):                     # roll_ff_clean 是 {k: v}
                v = v.get("k=3", float("nan"))
            if np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    out = {"prereg": "experiments/prereg_S1c_regime_silence.json", "results": res}
    base = {"one": m("baseline", "one_mixed"),
            "ff": m("baseline", "one_ff_clean"),
            "k3": m("baseline", "roll_ff_clean"),
            "head": m("baseline", "head_rms_ff")}
    out["baseline"] = base
    print(f"\n基线（SLOT-1b euler001x5 复用）: mixed={base['one']:.3f} "
          f"FF={base['ff']:.4f} roll_k3={base['k3']:.4f} 头RMS={base['head']:.4f}")

    print("\n=== 判决（预注册顺序；两机制分别对照同一基线）===")
    out["verdicts"] = {}
    for arm in ("wff10", "gate"):
        ff, mx = m(arm, "one_ff_clean"), m(arm, "one_mixed")
        k3 = m(arm, "roll_ff_clean")
        hd = m(arm, "head_rms_ff")
        r1 = ff <= 0.50
        r2 = mx <= 0.628
        r3 = k3 <= 1.02
        out["verdicts"][arm] = dict(
            ff_one=ff, mixed=mx, roll_ff_k3=k3, head_rms_ff=hd,
            head_drop_factor=base["head"] / hd if hd else float("nan"),
            R1=bool(r1), R2=bool(r2), R3=bool(r3))
        print(f"  [{arm}] R1 静默(≤0.50): {'[OK]' if r1 else '[FAIL]'} ({ff:.4f}) | "
              f"R2 不伤接触(≤0.628): {'[OK]' if r2 else '[FAIL]'} ({mx:.3f}) | "
              f"R3 闭环(≤1.02): {'[OK]' if r3 else '[FAIL]'} ({k3:.4f}) | "
              f"R4 头RMS 降 {base['head'] / hd if hd else float('nan'):.2f}×")
    print(f"崩溃率(ITT): {sum(res[a][s] is None for a in ARMS for s in SEEDS) / 12:.0%}")

    with open("analysis_output/S1c_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/S1c_report.json")

    if not args.no_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            labels = ["baseline", "wff10", "gate"]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, (tag, lab, thr) in zip(axes, (
                    ("one_ff_clean", "FF-clean one-step (px)", 0.50),
                    ("one_mixed", "mixed one-step (px)", 0.628),
                    ("roll_ff_clean", "FF-clean closed-loop k=3 (px)", 1.02))):
                vals = []
                for a in labels:
                    v = [res[a][s][("roll_ff_clean" if tag == "roll_ff_clean"
                                    else tag)][("k=3" if tag == "roll_ff_clean" else None)]
                         if tag == "roll_ff_clean" else res[a][s][tag]
                         for s in SEEDS if res[a][s]]
                    vals.append(float(np.mean(v)))
                ax.bar(np.arange(3), vals, color="#4C72B0")
                ax.axhline(thr, ls="--", c="r", lw=0.8)
                ax.set_xticks(np.arange(3))
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_title(lab)
            fig.suptitle("SLOT-1c: regime silence (dashed = preregistered threshold)")
            fig.tight_layout()
            fig.savefig("figures/S1c_silence.png", dpi=150)
            print("[save] figures/S1c_silence.png")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 出图失败（不影响判决）: {e}")


if __name__ == "__main__":
    main()
