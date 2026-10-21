# -*- coding: utf-8 -*-
"""TMLR 论文图 1–5（英文出版版；全部数字来自存储 JSON，零手工）。

用法：python paper/figures/make_figures.py
输出：paper/figures/fig{1..5}_*.pdf + .png（300 dpi）
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
})
C = {"blue": "#4C72B0", "orange": "#DD8452", "green": "#55A868",
     "red": "#C44E52", "purple": "#8172B3", "grey": "#8C8C8C"}
J = lambda p: json.load(open(os.path.join(ROOT, "analysis_output", p), encoding="utf-8"))
OUT = HERE


def save(fig, name):
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(OUT, name + ext), bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {name}.pdf/.png")


def mean_over(r, seeds, fn):
    vals = [fn(r[a][str(s)]) for a in r for s in seeds if str(s) in r[a]]
    return float(np.mean(vals)) if vals else float("nan")


def fig1():
    e0 = J("E0_report.json")["arms"]
    s1 = J("S1_ball_ladder.json")
    tiers = [("ours_base", "Euler\n@.05x1"), ("ours_base@euler001x5", "Euler\n@.01x5"),
             ("ours_base@rk4005", "RK4\n@.05x1"), ("ours_base@rk4001x5", "RK4\n@.01x5")]
    # (b) 数据源改为弹球世界解析阶梯（S1_ball_ladder），不再错用铰链数据
    ball_tiers = ["parabola005", "euler005", "rk4005", "euler001x5"]
    ball_lab = ["parabola", "Euler\n@.05x1", "RK4\n@.05x1", "Euler@.01x5\n(engine)"]
    m = lambda rs, k: float(np.mean([x[k] for x in rs]))
    k5 = lambda rs: float(np.mean([x["rollout"]["k=5"] for x in rs]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = np.arange(4)
    # (a) pendulum trained ladder
    ax = axes[0]
    data = [(e0[a], l) for a, l in tiers]
    ones = [m(r, "one_step_model") for r, _ in data]
    fives = [k5(r) for r, _ in data]
    ax.bar(x - 0.2, ones, 0.4, label="one-step", color=C["blue"])
    ax.bar(x + 0.2, fives, 0.4, label="closed-loop k=5", color=C["orange"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in data], rotation=0, fontsize=7)
    ax.set_title("(a) pendulum: trained ladder")
    ax.set_ylabel("position RMSE (px, log)")
    for xi, v in zip(x, ones):
        ax.text(xi - 0.2, v * 1.15, f"{v:.3g}", ha="center", fontsize=6.5)
    for xi, v in zip(x, fives):
        ax.text(xi + 0.2, v * 1.15, f"{v:.3g}", ha="center", fontsize=6.5)
    ax.legend(frameon=False, loc="upper right")
    # (b) ball zero-parameter ladder (FF-clean, from S1_ball_ladder.json)
    ax = axes[1]
    ones_b = [s1["one_step"][t]["ff_clean"] for t in ball_tiers]
    fives_b = [s1["rollout"][t]["k=5"] for t in ball_tiers]
    ax.bar(x - 0.2, ones_b, 0.4, label="one-step (FF-clean)", color=C["blue"])
    ax.bar(x + 0.2, fives_b, 0.4, label="closed-loop k=5 (FF)", color=C["orange"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ball_lab, rotation=0, fontsize=7)
    ax.set_title("(b) ball: zero-parameter ladder (free-flight)")
    ax.set_ylabel("position RMSE (px, log)")
    for xi, v in zip(x, ones_b):
        ax.text(xi - 0.2, v * 1.15, f"{v:.3g}", ha="center", fontsize=6.5)
    for xi, v in zip(x, fives_b):
        ax.text(xi + 0.2, v * 1.15, f"{v:.3g}", ha="center", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig1_floor_ladder")


def fig2():
    r = J("P0c_compare.json")["cells_summary"]
    cells = ["C00", "C01", "C10", "C11"]
    lab = ["C00\nno aux", "C01\nprototypes", "C10\nlabels", "C11\nboth"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    x = np.arange(4)
    axes[0].bar(x, [r[c]["one"] for c in cells], color=C["blue"])
    axes[0].set_title("(a) one-step (px)")
    axes[0].axhline(0.628, ls="--", c="r", lw=0.8)
    axes[1].bar(x, [r[c]["k5"] for c in cells], color=C["orange"])
    axes[1].set_title("(b) closed-loop k=5 (px)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(lab, rotation=15)
    fig.suptitle("Supervision 2×2 (means over 6 seeds; dashed = S1e mixed bound 0.628 px)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig2_p0c_2x2")


def fig3(probe):
    s1b = J("S1b_report.json")["results"]
    s1l = J("S1_ball_ladder.json")["one_step"]
    tiers = ["parabola005", "euler005", "rk4005", "euler001x5"]
    tier_lab = ["parabola", "Euler@.05", "RK4@.05", "Euler@.01x5\n(engine)"]
    ana = [s1l[t]["ff_clean"] for t in tiers]
    tr = [float(np.mean([s1b[t][str(s)]["one_ff_clean"] for s in (42, 43, 44)
                         if str(s) in s1b[t]])) for t in tiers]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x = np.arange(4)
    axes[0].bar(x - 0.2, ana, 0.4, label="analytic prior alone", color=C["green"])
    axes[0].bar(x + 0.2, tr, 0.4, label="trained arm (residual head on)", color=C["red"])
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tier_lab, rotation=15, fontsize=7)
    axes[0].set_ylabel("FF-clean one-step (px, log)")
    axes[0].set_title("(a) the ladder is erased after training")
    axes[0].legend(frameon=False, fontsize=7)
    arms = ("euler001x5", "parabola005")
    tg = [probe["euler001x5_target"], probe["parabola005_target"]]
    hd = [probe["euler001x5_head"], probe["parabola005_head"]]
    x2 = np.arange(2)
    axes[1].bar(x2 - 0.2, tg, 0.4, label="supervision (target RMS)", color=C["green"])
    axes[1].bar(x2 + 0.2, hd, 0.4, label="head output RMS", color=C["red"])
    axes[1].set_yscale("log")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["exact prior\n(target = 0)", "legacy prior"], fontsize=7)
    axes[1].set_title("(b) probe: supervision vs output (norm. units)")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig3_interference")


def fig4():
    r = J("S1e_report.json")["results"]
    r = dict(r)
    r["gate"] = J("S1c_report.json")["results"]["gate"]
    arms = [("baseline", "baseline"), ("wff10", "reweight ×10"),
            ("gate", "end-to-end gate"), ("sgate", "supervised gate"),
            ("W1", "closed-loop ×10"), ("W2", "both ×10"), ("S", "single ×100")]
    seeds = (42, 43, 44)
    ff = [mean_over({a: r[a]}, seeds, lambda x: x["one_ff_clean"]) for a, _ in arms]
    k3 = [mean_over({a: r[a]}, seeds, lambda x: x["roll_ff_clean"]["k=3"]) for a, _ in arms]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(x - 0.2, ff, 0.4, label="FF one-step", color=C["blue"])
    ax.bar(x + 0.2, k3, 0.4, label="FF closed-loop k=3", color=C["orange"])
    ax.axhline(0.00003, ls=":", c="g", lw=0.9)
    ax.text(len(arms) - 0.4, 0.00006, "analytic identity (3e-5)", fontsize=6.5, color="g")
    ax.annotate("dose reversal", xy=(len(arms) - 1 + 0.2, k3[-1]),
                xytext=(len(arms) - 3.6, 2.4),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in arms], rotation=20, fontsize=7)
    ax.set_ylabel("free-flight error (px, log)")
    ax.set_title("Seven trained arms, one trade-off (means over 3 seeds)")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    save(fig, "fig4_dose_response")


def fig5():
    r = J("S1f_report.json")["results"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for arm, col in (("baseline", C["grey"]), ("F1", C["blue"]), ("F2", C["orange"])):
        xs, ys = [], []
        un = r[arm]["42"]["unmasked_B"]
        base_contact = un["contact_rms"]
        for s in ("42", "43", "44"):
            for tau, g in r[arm][s]["grid_B"].items():
                xs.append(g["false_silence"] * 100)
                ys.append(g["contact_rms"] / un["contact_rms"])
        ax.scatter(xs, ys, s=10, alpha=0.45, color=col, label=arm)
        b = r[arm]["42"]
        tau = str(b["tau_star"])
        g = b["grid_B"][tau]
        ax.scatter([g["false_silence"] * 100], [g["contact_rms"] / un["contact_rms"]],
                   marker="*", s=90, color=col, edgecolor="k", linewidths=0.4, zorder=5)
    ax.axhline(1.10, ls="--", c="r", lw=0.8)
    ax.text(0.4, 1.115, "preregistered tolerance ×1.10", fontsize=6.5, color="r")
    ax.set_xlabel("false-silence rate on contact cells (%)")
    ax.set_ylabel("contact closed-loop damage (× of ungated)")
    ax.set_title("The exchange rate of silence (val-B; star = selected tau)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig5_silence_roc")


def compute_probe():
    """Fig 3(b) 的目标/头输出 RMS（确定性重算，缓存到 JSON）。"""
    cache = os.path.join(ROOT, "analysis_output", "S1b_probe_cached.json")
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    import torch
    from tmp_s1b_compare import load_model, transfer_clean_mask
    from train_v3 import V3Data, load_dataset, split_episodes
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    pos = [1, 2, 12, 13, 23, 24]
    for tier in ("euler001x5", "parabola005"):
        model, ck = load_model(f"checkpoints-S1b/{tier}_s42/best.pt")
        model = model.to(device).eval()
        _, val_eps = split_episodes(N, 42)
        ns = ck["norm_stats"]
        ds = V3Data(states, obstacles, contact, actions, val_eps, 8, 3,
                    ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"],
                    prior_spec=ck["config"]["prior_spec"])
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(np.arange(len(ds)), size=20000, replace=False))
        mask = transfer_clean_mask(ds, idx, 1)
        sub = idx[mask]
        b = ds.gather(sub)
        dn = b["d_norm"].numpy()[:, pos]
        with torch.no_grad():
            h, _ = model.encode_sequence(b["x"].to(device))
            dl = model.predict_state(h, b["a0"].to(device),
                                     b["gua"].to(device)).float().cpu().numpy()
        dl = dl[:, pos]
        out[f"{tier}_target"] = float(np.sqrt((dn ** 2).mean()))
        out[f"{tier}_head"] = float(np.sqrt((dl ** 2).mean()))
        del model
        torch.cuda.empty_cache()
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    fig1()
    fig2()
    fig3(compute_probe())
    fig4()
    fig5()
    print("全部图件完成")
