# -*- coding: utf-8 -*-
"""SLOT-1f 附件：全部九臂的 τ 曲线（预注册 reporting.extra 兑现）。

九臂 = baseline（S1b）+ wff10/gate（S1c）+ sgate（S1d）+ W1/W2/S（S1e）
+ F1/F2（S1f）。协议与 tmp_s1f_compare.py 完全一致（RS 0、20000 窗、
前 A 后 B、网格 0.50--0.95 step 0.05），并对三共有臂与 S1f_report.json
做逐点一致性检查（应逐位一致）。

写：analysis_output/S1f_allarms.json
   paper/figures/fig8_allarms_roc.pdf/.png
"""
import io
import json
import os

import numpy as np
import torch

import tmp_s1f_compare as C

ROOT = os.path.dirname(os.path.abspath(__file__))
ARMS = dict(C.ARMS)
ARMS.update({
    "wff10": "checkpoints-S1c/wff10_s{s}/best.pt",
    "gate": "checkpoints-S1c/gate_s{s}/best.pt",
    "W1": "checkpoints-S1e/W1_s{s}/best.pt",
    "S": "checkpoints-S1e/S_s{s}/best.pt",
})
ORDER = ["baseline", "wff10", "gate", "sgate", "W1", "W2", "S", "F1", "F2"]
SHARED = ("baseline", "F1", "F2")
LABELS = {"baseline": "baseline", "wff10": "reweight ×10", "gate": "e2e gate",
          "sgate": "sgate", "W1": "closed ×10", "W2": "both ×10",
          "S": "single ×100", "F1": "F1", "F2": "F2"}


def make_figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C_ = {"baseline": "#444444", "wff10": "#7f7f7f", "gate": "#8c564b",
          "sgate": "#9467bd", "W1": "#17becf", "W2": "#2ca02c",
          "S": "#d62728", "F1": "#1f77b4", "F2": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for arm in ORDER:
        xs, ys = [], []
        for tau in out["tau_grid"]:
            per_x = [out["arms"][arm][str(s)]["grid_B"][str(tau)]["false_silence"] * 100
                     for s in C.SEEDS]
            per_y = [out["arms"][arm][str(s)]["grid_B"][str(tau)]["contact_rms"]
                     / out["arms"][arm][str(s)]["unmasked_B"]["contact_rms"]
                     for s in C.SEEDS]
            xs.append(float(np.mean(per_x)))
            ys.append(float(np.mean(per_y)))
        lw = 2.0 if arm == "S" else 1.2
        ax.plot(xs, ys, marker="o", ms=2.5, lw=lw, color=C_[arm],
                label=LABELS[arm])
    ax.axhline(1.10, ls="--", c="k", lw=0.8)
    ax.set_xlabel("contact cells falsely silenced (%)")
    ax.set_ylabel("contact closed-loop damage (× ungated)")
    ax.set_title("Nine arms, one trade-off (τ sweep, means over 3 seeds)",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, ncol=3)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(ROOT, "paper", "figures",
                                 "fig8_allarms_roc" + ext), bbox_inches="tight")
    print("[save] paper/figures/fig8_allarms_roc.pdf/.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("载入 trajectories_v3.npz …", flush=True)
    states, obstacles, contact, actions = C.load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    out = {
        "note": "S1f 附件：九臂 τ 曲线（预注册 reporting.extra；与主报告同协议）",
        "tau_grid": list(C.TAU_GRID),
        "arms": {},
    }
    for arm in ORDER:
        out["arms"][arm] = {}
        for s in C.SEEDS:
            model, ck = C.load_model(ARMS[arm].format(s=s))
            a, ns = ck["args"], ck["norm_stats"]
            _, val_eps = C.split_episodes(N, s)
            ds = C.V3Data(states, obstacles, contact, actions, val_eps,
                          a["seq_len"], a["future_steps"],
                          ns["mean_obj"], ns["std_obj"], ns["mean_obs"],
                          ns["std_obs"], prior_spec=ck["config"]["prior_spec"])
            rng = np.random.RandomState(0)
            idx_all = np.arange(len(ds))
            if len(idx_all) > 20000:
                idx_all = np.sort(rng.choice(idx_all, size=20000, replace=False))
            half = len(idx_all) // 2
            idx_A, idx_B = idx_all[:half], idx_all[half:]
            model = model.to(device)
            row = {"n_A": int(len(idx_A)), "n_B": int(len(idx_B))}
            row["unmasked_B"] = C.eval_tau(model, ds, device, idx_B,
                                           a["future_steps"], 0.0)
            row["grid_A"], row["grid_B"] = {}, {}
            for tau in C.TAU_GRID:
                row["grid_A"][str(tau)] = C.eval_tau(
                    model, ds, device, idx_A, a["future_steps"], tau)
                row["grid_B"][str(tau)] = C.eval_tau(
                    model, ds, device, idx_B, a["future_steps"], tau)
            ok = [t for t in C.TAU_GRID
                  if row["grid_A"][str(t)]["false_silence"] <= C.FALSE_SILENCE_CAP]
            row["tau_star"] = max(ok) if ok else C.TAU_GRID[0]
            row["verdict_B"] = C.eval_tau(model, ds, device, idx_B,
                                          a["future_steps"], row["tau_star"])
            v, u = row["verdict_B"], row["unmasked_B"]
            row["R1f"] = bool(v["ff_rms"] <= C.R1F_THRESHOLD)
            row["R2f"] = bool(v["contact_rms"] <= u["contact_rms"] * C.R2F_MARGIN)
            out["arms"][arm][str(s)] = row
            print("%-9s s%s | tau*=%.2f | ffB=%.3e allSil=%.1f | contact x%.3f"
                  % (arm, s, row["tau_star"], v["ff_rms"],
                     v["ff_all_silenced"], v["contact_rms"] / u["contact_rms"]),
                  flush=True)
            del model
            torch.cuda.empty_cache()

    sh = J = json.load(io.open(os.path.join(ROOT, "analysis_output",
                                            "S1f_report.json"),
                               encoding="utf-8"))["results"]
    worst = 0.0
    for arm in SHARED:
        for s in map(str, C.SEEDS):
            for tau in out["arms"][arm][s]["grid_B"]:
                for k in ("ff_rms", "ff_all_silenced", "false_silence",
                          "contact_rms"):
                    a_, b_ = out["arms"][arm][s]["grid_B"][tau][k], sh[arm][s]["grid_B"][tau][k]
                    worst = max(worst, abs(a_ - b_) / max(abs(b_), 1e-30))
    out["shared_arm_consistency_max_rel"] = float(worst)
    print("[check] shared-arm max rel diff vs S1f_report: %.2e" % worst, flush=True)

    io.open(os.path.join(ROOT, "analysis_output", "S1f_allarms.json"), "w",
            encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("[save] analysis_output/S1f_allarms.json")
    make_figure(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure-only", action="store_true")
    args = ap.parse_args()
    if args.figure_only:
        make_figure(json.load(io.open(
            os.path.join(ROOT, "analysis_output", "S1f_allarms.json"),
            encoding="utf-8")))
    else:
        main()
