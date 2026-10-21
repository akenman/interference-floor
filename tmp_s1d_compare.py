# -*- coding: utf-8 -*-
"""SLOT-1d 判决脚本：监督式静默门（判据以预注册为准）。

预注册：experiments/prereg_S1d_supervised_gate.json
处理臂 = sgate（GatedWorldModelV3 + BCE(g_logits, 1−win_ff)），3 seed；
对照臂零重训复用：baseline = SLOT-1b euler001x5_s42-44，wff10 = SLOT-1c（次口径）。
协议沿用 SLOT-1c：同 seed 配对（val 划分随 seed）+ ITT + 崩溃率。

判据（预注册顺序）：
  R1d 完全静默：FF-clean one-step ≤ 0.10px（0.10–0.25 记部分兑现）
  R2d 不伤接触：mixed one-step ≤ 0.628px（基线 0.571 × 1.10）
  R3d 闭环贴地：FF-clean 闭环 k=3 ≤ 0.20px（解析恒等 0.00003 + 门泄漏裕度）
  R4d 门准确率：FF 帧关门 / 接触帧开门准确率（预期 ≥99%）
  R5d 混合闭环不劣化：mixed 闭环 k=3 ≤ 基线 × 1.10
  统计：同 seed 配对 vs baseline（3/3 方向一致）；wff10 为次口径报告。

用法：python tmp_s1d_compare.py [--val-windows 20000] [--no-figure]
输出：控制台判决 + analysis_output/S1d_report.json + figures/S1d_supervised_gate.png
"""
import argparse
import json

import numpy as np
import torch

# 注意：tmp_s1b_compare 导入时把 sys.stdout 包成 utf-8 TextIOWrapper，
# 本模块不得再包一层（双重包装会关闭底层 buffer，SLOT-1c 实测）。
from tmp_s1b_compare import load_model, transfer_clean_mask, eval_model  # noqa: E402
from train_v3 import V3Data, load_dataset, split_episodes  # noqa: E402

SEEDS = [42, 43, 44]
ARMS = {"baseline": "checkpoints-S1b/euler001x5_s{s}/best.pt",
        "wff10": "checkpoints-S1c/wff10_s{s}/best.pt",
        "sgate": "checkpoints-S1d/sgate_s{s}/best.pt"}

R1D, R2D, R3D, R4D, R5D_MARGIN = 0.10, 0.628, 0.20, 0.99, 1.10


@torch.no_grad()
def gate_probe(model, ds, device, idx_all, bs=2048):
    """门探针：逐窗 g=σ(gate_logits(h, a0))，按 win_ff 标签算关门/开门准确率。

    win_ff=1（末输入帧与目标帧均无接触）→ 目标门关（g<0.5）；
    win_ff=0（接触 regime）→ 目标门开（g≥0.5）。标签与训练 BCE 完全同源。
    """
    if not hasattr(model, "gate_logits"):
        return None
    M = len(idx_all)
    g_all = np.zeros(M, dtype=np.float64)
    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        h, _ = model.encode_sequence(b["x"].to(device))
        logits = model.gate_logits(h, b["a0"].to(device))
        g_all[j:j + len(idx)] = torch.sigmoid(logits.float()).cpu().numpy()
    ep, t = ds.ep_idx[idx_all], ds.win_start[idx_all]
    ff = ds.win_ff[ep, t] > 0.5
    n_ff, n_ct = int(ff.sum()), int((~ff).sum())
    acc_closed = float((g_all[ff] < 0.5).mean()) if n_ff else float("nan")
    acc_open = float((g_all[~ff] >= 0.5).mean()) if n_ct else float("nan")
    acc_all = float(((g_all < 0.5) == ff).mean()) if M else float("nan")
    return dict(n_ff=n_ff, n_contact=n_ct,
                acc_closed_ff=acc_closed, acc_open_contact=acc_open, acc_all=acc_all,
                g_mean_ff=float(g_all[ff].mean()) if n_ff else float("nan"),
                g_mean_contact=float(g_all[~ff].mean()) if n_ct else float("nan"),
                g_p95_ff=float(np.percentile(g_all[ff], 95)) if n_ff else float("nan"))


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
            gp = gate_probe(model, ds, device, idx_all)
            if gp is not None:
                r["gate"] = gp
            res[arm][s] = r
            tail = ""
            if gp is not None:
                tail = (f"  门: FF关={gp['acc_closed_ff']:.4f}"
                        f"(ḡ={gp['g_mean_ff']:.3f}) 接触开={gp['acc_open_contact']:.4f}"
                        f"(ḡ={gp['g_mean_contact']:.3f})")
            print(f"[{arm} s{s}] mixed={r['one_mixed']:.3f}  FF={r['one_ff_clean']:.4f}"
                  f"  roll_ff k3={r['roll_ff_clean']['k=3']:.4f}"
                  f"  roll_mixed k3={r['roll_mixed']['k=3']:.3f}"
                  f"  yao={r['yao_acc']:.3f}{tail}", flush=True)
            del model
            torch.cuda.empty_cache()

    def m(arm, key):
        vals = []
        for s in SEEDS:
            r = res[arm][s]
            if not r:
                continue
            v = r.get(key, float("nan"))
            if isinstance(v, dict):
                v = v.get("k=3", float("nan"))
            if np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    def seed_vals(arm, key):
        out = []
        for s in SEEDS:
            r = res[arm][s]
            if not r:
                out.append(np.nan)
                continue
            v = r.get(key, np.nan)
            if isinstance(v, dict):
                v = v.get("k=3", np.nan)
            out.append(v)
        return np.array(out, dtype=float)

    out = {"prereg": "experiments/prereg_S1d_supervised_gate.json", "results": res}
    base = {"one": m("baseline", "one_mixed"), "ff": m("baseline", "one_ff_clean"),
            "k3_ff": m("baseline", "roll_ff_clean"),
            "k3_mixed": m("baseline", "roll_mixed"), "head": m("baseline", "head_rms_ff")}
    out["baseline"] = base
    print(f"\n基线（SLOT-1b euler001x5 复用）: mixed={base['one']:.3f} "
          f"FF={base['ff']:.4f} roll_ff_k3={base['k3_ff']:.4f} "
          f"roll_mixed_k3={base['k3_mixed']:.3f}")

    print("\n=== 判决（预注册顺序；sgate 主口径，wff10 次口径）===")
    out["verdicts"] = {}
    for arm in ("sgate", "wff10"):
        ff, mx = m(arm, "one_ff_clean"), m(arm, "one_mixed")
        k3 = m(arm, "roll_ff_clean")
        k3m = m(arm, "roll_mixed")
        gp = [res[arm][s]["gate"] for s in SEEDS
              if res[arm][s] and "gate" in res[arm][s]]
        g_acc = float(np.mean([g["acc_all"] for g in gp])) if gp else float("nan")
        g_closed = float(np.mean([g["acc_closed_ff"] for g in gp])) if gp else float("nan")
        g_open = float(np.mean([g["acc_open_contact"] for g in gp])) if gp else float("nan")
        r1 = ff <= R1D
        partial = bool((not r1) and ff <= 0.25)
        r2 = mx <= R2D
        r3 = k3 <= R3D
        r4 = bool(np.isfinite(g_acc) and g_acc >= R4D)
        r5 = k3m <= base["k3_mixed"] * R5D_MARGIN
        # 同 seed 配对方向一致性（vs baseline）
        d_ff = seed_vals(arm, "one_ff_clean") - seed_vals("baseline", "one_ff_clean")
        d_k3 = seed_vals(arm, "roll_ff_clean") - seed_vals("baseline", "roll_ff_clean")
        dir_ff = int(np.nansum(d_ff < 0))
        dir_k3 = int(np.nansum(d_k3 < 0))
        out["verdicts"][arm] = dict(
            ff_one=ff, mixed=mx, roll_ff_k3=k3, roll_mixed_k3=k3m,
            gate_acc=g_acc, gate_acc_closed_ff=g_closed, gate_acc_open_contact=g_open,
            R1d=bool(r1), R1d_partial=partial, R2d=bool(r2), R3d=bool(r3),
            R4d=bool(r4), R5d=bool(r5),
            paired_delta_ff=d_ff.tolist(), paired_delta_roll_ff_k3=d_k3.tolist(),
            dir_improved_ff=f"{dir_ff}/3", dir_improved_roll_ff_k3=f"{dir_k3}/3")
        gate_txt = (f" | R4d 门准确率(≥{R4D}): {'[OK]' if r4 else '[FAIL]'} "
                    f"({g_acc:.4f}；FF关 {g_closed:.4f} / 接触开 {g_open:.4f})"
                    if np.isfinite(g_acc) else "")
        print(f"  [{arm}] R1d 静默(≤{R1D}): {'[OK]' if r1 else ('◐部分' if partial else '[FAIL]')}"
              f" ({ff:.4f}) | R2d 不伤接触(≤{R2D}): {'[OK]' if r2 else '[FAIL]'} ({mx:.3f}) | "
              f"R3d 闭环(≤{R3D}): {'[OK]' if r3 else '[FAIL]'} ({k3:.4f}) | "
              f"R5d mixed 闭环(≤×{R5D_MARGIN}): {'[OK]' if r5 else '[FAIL]'} ({k3m:.3f})"
              f"{gate_txt}")
        print(f"        配对 vs baseline: Δff={np.round(d_ff, 4).tolist()} "
              f"(改善 {dir_ff}/3) | Δroll_ff_k3={np.round(d_k3, 4).tolist()} "
              f"(改善 {dir_k3}/3)")
    crash = sum(res[a][s] is None for a in ARMS for s in SEEDS) / (len(ARMS) * len(SEEDS))
    out["crash_rate_ITT"] = crash
    print(f"崩溃率(ITT): {crash:.0%}")

    with open("analysis_output/S1d_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/S1d_report.json")

    if not args.no_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            v = out["verdicts"]
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            panels = [
                ("FF-clean one-step (px)", [base["ff"], v["wff10"]["ff_one"],
                                            v["sgate"]["ff_one"]], R1D),
                ("FF-clean closed-loop k=3 (px)", [base["k3_ff"], v["wff10"]["roll_ff_k3"],
                                                   v["sgate"]["roll_ff_k3"]], R3D),
                ("gate accuracy", [np.nan, np.nan, v["sgate"]["gate_acc"]], R4D),
            ]
            for ax, (lab, vals, thr) in zip(axes, panels):
                ax.bar(np.arange(3), vals, color=["#8C8C8C", "#4C72B0", "#C44E52"])
                if np.isfinite(thr) and lab != "gate accuracy":
                    ax.axhline(thr, ls="--", c="r", lw=0.8)
                elif lab == "gate accuracy":
                    ax.axhline(thr, ls="--", c="r", lw=0.8)
                    ax.set_ylim(0, 1.05)
                ax.set_xticks(np.arange(3))
                ax.set_xticklabels(["baseline", "wff10", "sgate"], fontsize=8)
                ax.set_title(lab, fontsize=9)
            fig.suptitle("SLOT-1d: supervised silence gate "
                         "(dashed = preregistered threshold)")
            fig.tight_layout()
            fig.savefig("figures/S1d_supervised_gate.png", dpi=150)
            print("[save] figures/S1d_supervised_gate.png")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 出图失败（不影响判决）: {e}")


if __name__ == "__main__":
    main()
