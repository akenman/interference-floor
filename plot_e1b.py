# -*- coding: utf-8 -*-
"""
E1b 作图脚本 (plot_e1b.py)
==================================================
输入: analysis_output/E1b_raw.json（eval_e1b.py 的输出）
输出:
  figures/E1b_gua_error_bars.png — 三臂并列分卦误差柱状图（带误差棒）
  figures/E1b_overall.png — 整体 RMSE 对比（三臂 × 3 种子）

用法: python plot_e1b.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GUA_NAMES = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]
GUA_BITS = ["000", "001", "010", "011", "100", "101", "110", "111"]
ARM_LABELS = {"uniform": "B-uniform\n(标准头 256 维)",
              "phase": "B-phase\n(8 相态子头)",
              "priority": "B-priority\n(卦加权损失)"}
ARM_COLORS = {"uniform": "#4C78A8", "phase": "#F58518", "priority": "#E45756"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_gua_error_bars(raw, out_path):
    arms = raw["arms"]
    variants = ["uniform", "phase", "priority"]
    guas = [f"g{g} ({GUA_NAMES[g]})" for g in range(8)]

    means = np.array([[arms[v]["gua_err_mean"][g] for g in guas] for v in variants])
    stds = np.array([[arms[v]["gua_err_std"][g] for g in guas] for v in variants])

    fig, ax = plt.subplots(figsize=(13, 5.6))
    x = np.arange(8)
    width = 0.26
    offsets = {"uniform": -width, "phase": 0.0, "priority": width}

    for i, v in enumerate(variants):
        ax.bar(x + offsets[v], means[i], width, yerr=stds[i],
               capsize=4, color=ARM_COLORS[v], label=ARM_LABELS[v],
               edgecolor="white", linewidth=0.6,
               error_kw=dict(lw=1, capsize=3, capthick=1,
                             ecolor=ARM_COLORS[v]))

    ax.set_xlabel("相态（卦）", fontsize=12)
    ax.set_ylabel("one-step 位置 RMSE (px)", fontsize=12)
    ax.set_title("E1b 分卦误差对比（mean ± std, 3 种子）", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{GUA_NAMES[g]}\n{GUA_BITS[g]}" for g in range(8)], fontsize=10)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ylim = ax.get_ylim()
    ax.set_ylim(0, max(ylim[1] * 1.22, means.max() * 1.3))

    # 标注样本数
    counts = {g: arms["uniform"]["gua_count_mean"][f"g{g}"] for g in range(8)}
    for g in range(8):
        ax.text(g, ylim[1] * 0.02, f"n={int(counts[g]):,}",
                ha="center", va="bottom", fontsize=7, color="#666666")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")


def plot_overall(raw, out_path):
    arms = raw["arms"]
    variants = ["uniform", "phase", "priority"]

    means = [arms[v]["overall_mean"] for v in variants]
    stds = [arms[v]["overall_std"] for v in variants]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(variants))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=[ARM_COLORS[v] for v in variants],
                  edgecolor="white", linewidth=0.8,
                  error_kw=dict(lw=1.5, capsize=5, capthick=1.5, ecolor="#333333"))

    for i, v in enumerate(variants):
        ax.text(x[i], means[i] + stds[i] + 0.02,
                f"{means[i]:.3f} ± {stds[i]:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[v].replace("\n", " ") for v in variants], fontsize=11)
    ax.set_ylabel("整体 one-step 位置 RMSE (px)", fontsize=12)
    ax.set_title("E1b 整体 RMSE 对比（mean ± std, 3 种子）", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(means) * 1.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_path}")


def main():
    raw = load("analysis_output/E1b_raw.json")
    print("分卦误差柱状图...")
    plot_gua_error_bars(raw, "figures/E1b_gua_error_bars.png")
    print("整体 RMSE 对比...")
    plot_overall(raw, "figures/E1b_overall.png")
    print("[OK]")


if __name__ == "__main__":
    main()