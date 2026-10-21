# -*- coding: utf-8 -*-
"""TMLR 论文图 7：静默上限（英文出版版；数字全部来自 floor_analysis.json）。

用法：python paper/figures/make_fig_floor.py
输出：paper/figures/fig7_floor_ceiling.pdf + .png（300 dpi）
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
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
})
C = {"blue": "#4C72B0", "orange": "#DD8452", "green": "#55A868",
     "red": "#C44E52", "purple": "#8172B3", "grey": "#8C8C8C"}


def save(fig, name):
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(HERE, name + ext), bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {name}.pdf/.png")


def main():
    fa = json.load(open(os.path.join(ROOT, "analysis_output", "floor_analysis.json"),
                        encoding="utf-8"))
    sc = fa["silence_ceiling"]
    rows = sc["rows"]
    w = np.array([r["w_ff"] for r in rows], dtype=float)
    ff = np.array([r["ff_over_baseline"] for r in rows])
    ct = np.array([r["ct_over_baseline"] for r in rows])
    x = np.arange(len(w))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    ax.plot(x, ff, "o-", color=C["red"], ms=4, lw=1.4,
            label="free-flight head output")
    ax.plot(x, ct, "s--", color=C["blue"], ms=4, lw=1.4,
            label="contact head output")
    # 端点在同一配置的不同 RNG 执行顺序下会落在 [0.34, 0.58]，标注区间
    smin = fa["run_spread"].get("no_roll_min")
    if smin is not None:
        ax.annotate("", xy=(x[-1], ff[-1]), xytext=(x[-1], smin),
                    arrowprops=dict(arrowstyle="<->", color=C["grey"], lw=1.0))
        ax.text(x[-1] - 0.15, (ff[-1] + smin) / 2, f"rerun\nrange\n{smin:.2f}",
                fontsize=6.5, color=C["grey"], ha="right", va="center")
    ax.axhline(1.0, color=C["grey"], lw=0.9, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in w])
    ax.set_xlabel(r"free-flight loss weight $w_{\mathrm{ff}}$")
    ax.set_ylabel("output / unmodified head")
    ax.set_title("(a) Silence is purchasable up to a ceiling", fontsize=9)
    ax.legend(loc="center left", frameon=False)
    ax.set_ylim(0, 1.6)

    ax = axes[1]
    # 接触侧的重现底（±13%），用灰带标出；扫到 w_ff=100 时接触侧已漂出带外
    ax.axhspan(0.87, 1.13, color=C["grey"], alpha=0.16, lw=0)
    ax.axhline(1.0, color=C["grey"], lw=0.9, ls=":")
    ax.plot(ff, ct, "o-", color=C["purple"], ms=4, lw=1.4)
    for xi, f_, c_, wi in zip(x, ff, ct, w):
        ax.annotate(f"{wi:g}", (f_, c_), textcoords="offset points",
                    xytext=(5, 4), fontsize=6.5, color=C["grey"])
    r = float(np.corrcoef(ff, ct)[0, 1])
    ax.text(0.97, 0.05, f"$r={r:+.2f}$; shaded band = $\\pm13\\%$\ncontact reproduction floor",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color=C["grey"])
    ax.set_ylim(0.84, 1.22)
    ax.set_xlabel("free-flight head output / unmodified")
    ax.set_ylabel("contact head output / unmodified")
    ax.set_title("(b) The two regimes move together", fontsize=9)

    fig.tight_layout()
    save(fig, "fig7_floor_ceiling")


if __name__ == "__main__":
    main()
