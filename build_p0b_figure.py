# -*- coding: utf-8 -*-
"""P0-b 三格图：w_roll 买得到稳定性，买不到先验质量。

左：闭环 k=5（对数轴）—— 三个配方 × 四个臂
右：one-step —— 同一分组
数据来源：checkpoints-P0b{A,B,C}/*/best.pt 的 val_metrics（与报告同协议）。
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch

sys.stdout.reconfigure(encoding="utf-8")
OUT = "figures"
rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
                 "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 150,
                 "savefig.bbox": "tight", "axes.spines.top": False,
                 "axes.spines.right": False, "axes.unicode_minus": False,
                 "font.sans-serif": ["Noto Sans SC", "Microsoft YaHei", "SimHei",
                                     "DejaVu Sans"]})

CELLS = [("A", "checkpoints-P0bA", "A：单步（w_roll=0, lr=1e-3）", "#4C78A8"),
         ("B", "checkpoints-P0bB", "B：闭环（w_roll=0.05, lr=5e-4）", "#E45756"),
         ("C", "checkpoints-P0bC", "C：闭环（w_roll=0.05, lr=1e-3）", "#54A24B")]
ARMS = [("ours_base@rk4001x5", "ours-base\nrk4@0.01×5"),
        ("noprior", "noprior\n(纯 GRU)"),
        ("strong_mlp", "strong-MLP\n(无先验)"),
        ("ours_base", "ours-base\n(euler 粗先验)")]


def load(root):
    out = {}
    for d in sorted(glob.glob(f"{root}/*")):
        pt = os.path.join(d, "best.pt")
        if not os.path.exists(pt):
            continue
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        vm = ck["val_metrics"]
        out[os.path.basename(d)] = (vm["one_step_model"], vm["rollout"]["k=5"])
    return out


data = {tag: load(root) for tag, root, _, _ in CELLS}
print("可用 run：", {t: len(data[t]) for t in data})

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
for ax, idx, title, logy in ((axes[0], 1, "闭环 rollout k=5（px，越低越好）", True),
                             (axes[1], 0, "one-step（px，越低越好）", False)):
    w = 0.26
    xs = np.arange(len(ARMS))
    for j, (tag, root, lab, col) in enumerate(CELLS):
        means, errs = [], []
        for arm, _ in ARMS:
            vals = [v[idx] for k, v in data[tag].items() if k.startswith(arm + "_s")
                    or k == arm]
            # ours_base@rk4001x5 的目录名形如 ours_base@rk4001x5_s42
            vals = [v[idx] for k, v in data[tag].items()
                    if k.rsplit("_s", 1)[0] == arm]
            means.append(np.mean(vals) if vals else np.nan)
            errs.append(np.std(vals) if vals else 0.0)
        ax.bar(xs + (j - 1) * w, means, w, yerr=errs, capsize=3, label=lab,
               color=col, edgecolor="white", linewidth=0.6)
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in ARMS], fontsize=9)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, which="both")
axes[0].set_ylabel("pos_rmse (px，对数轴)")
axes[1].set_ylabel("pos_rmse (px)")
axes[1].axhline(0.49, color="#17BECF", ls="--", lw=1.2)
axes[1].text(-0.44, 0.515, "精确先验 0.49px", color="#17BECF", fontsize=8.5, ha="left")
axes[0].legend(frameon=False, loc="upper right")
fig.suptitle("P0-b：闭环训练（w_roll）买得到稳定性，买不到先验质量", fontsize=13,
             weight="bold", y=1.03)
fig.text(0.5, -0.06,
         "C−A = w_roll 的净效应（lr 固定）：noprior 53.9→1.31（41×）、ours-base 粗先验 14.15→4.89（2.9×）、"
         "strong-MLP 1.653→1.650（0%）。\n"
         "B−C = lr 减半效应：单步全面变差（+0.02~+0.09px）。"
         "全格最优的非精确先验臂 = 1.31px，仍是精确先验臂 0.49px 的 2.7×。",
         ha="center", fontsize=8.5, color="#555")
fig.savefig(f"{OUT}/P0b_three_cell.png")
print(f"[fig] {OUT}/P0b_three_cell.png")
