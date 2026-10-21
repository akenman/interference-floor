"""E0 统一主表出图（Table 1 + 6 核心图）。依赖：matplotlib 3.x，numpy。"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

GUA = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]
GUA_CODE = ["000", "001", "010", "011", "100", "101", "110", "111"]
ARM_LABELS = {
    "noprior": "noprior\n(纯 GRU)",
    "ours_base": "ours-base\n(残差+三爻)",
    "ours_gua": "ours+gua\n(+GuaMoE)",
    "ours_sigma": "ours+sigma\n(+σ+CQR)",
    "ours_yao": "ours-yao\n(去爻头)",
    "jepa": "JEPA-like",
    "strong_mlp": "strong-MLP",
    "inertia": "inertia\n(纯解析先验)",
    "zero_vel": "zero-vel\n(零速度)",
    # 先验质量阶梯（0 参数解析档，P0-a 的"地板"参照）
    "inertia_euler001x5": "inertia\neuler@0.01×5",
    "inertia_rk4005": "inertia\nrk4@0.05",
    "inertia_rk4001x5": "inertia\nrk4@0.01×5",
    # 先验质量阶梯（训练档：同架构 ours_base 只换先验积分器）
    "ours_base@euler001x5": "ours-base\neuler@0.01×5",
    "ours_base@rk4005": "ours-base\nrk4@0.05",
    "ours_base@rk4001x5": "ours-base\nrk4@0.01×5",
}
COLORS = {"noprior": "#4C78A8", "ours_base": "#F58518", "ours_gua": "#E45756",
          "ours_sigma": "#72B7B2", "ours_yao": "#54A24B", "jepa": "#B272B2",
          "strong_mlp": "#FF9DA7", "inertia": "#8C564B", "zero_vel": "#7F7F7F",
          "inertia_euler001x5": "#C49C94", "inertia_rk4005": "#E377C2",
          "inertia_rk4001x5": "#17BECF",
          "ours_base@euler001x5": "#FFD92F", "ours_base@rk4005": "#D95F02",
          "ours_base@rk4001x5": "#1B9E77"}


def _color(a, i=0):
    """未知臂回退到 matplotlib 默认色环，避免新增臂时 KeyError（已两次踩到）。"""
    return COLORS.get(a, f"C{i % 10}")
# 无物理先验的臂：其 one_step_prior 无意义（StrongMLP 的 prior_norm 是零占位），
# 一律显示 "—"，不得写成数据（旧图曾把 87.94 px 的数据集均值基线当"先验"）。
NO_PRIOR_ARMS = {"noprior", "jepa", "strong_mlp"}

rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
                 "xtick.labelsize": 10, "ytick.labelsize": 10,
                 "legend.fontsize": 9, "figure.dpi": 150,
                 "axes.spines.top": False, "axes.spines.right": False,
                 "savefig.dpi": 150, "savefig.bbox": "tight",
                 "font.sans-serif": ["Noto Sans SC", "Microsoft YaHei",
                                     "SimHei", "DejaVu Sans"],
                 "axes.unicode_minus": False})

REPORT = "analysis_output/E0_report.json"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def load():
    d = json.load(open(REPORT, encoding="utf-8"))
    return d, d["arms"]


def _vals(arms, arm, key):
    v = [r.get(key) for r in arms[arm]]
    v = [x for x in v if x is not None and (not isinstance(x, float) or x == x)]
    return np.array(v)


def _roll(arms, arm, k):
    """取某臂 rollout k 步的（3 种子）数组。"""
    v = [r.get("rollout", {}).get(f"k={k}") for r in arms[arm]]
    v = [x for x in v if x is not None and (not isinstance(x, float) or x == x)]
    return np.array(v)


def _nwin(arms):
    v = [r.get("n_windows") for a in arms for r in arms[a] if r.get("n_windows")]
    return int(np.mean(v)) if v else None


# ---------------------------------------------------------------- Fig 1: 主表
def fig_main_table(d, arms):
    rows = []
    for arm in sorted(arms, key=lambda a: _vals(arms, a, "one_step_model").mean()):
        model = _vals(arms, arm, "one_step_model")
        zv = _vals(arms, arm, "one_step_zero_vel")
        yao = _vals(arms, arm, "yao_acc")
        r2, r5 = _roll(arms, arm, 2), _roll(arms, arm, 5)
        if arm in NO_PRIOR_ARMS:
            prior_m = np.nan                    # 无物理先验 → 不写数
        else:
            pv = _vals(arms, arm, "one_step_prior")
            prior_m = pv.mean() if pv.size else np.nan
        rows.append((ARM_LABELS.get(arm, arm), len(model),
                     model.mean(), model.std(),
                     r2.mean() if r2.size else np.nan,
                     r5.mean() if r5.size else np.nan,
                     prior_m,
                     zv.mean() if zv.size else np.nan,
                     yao.mean() if yao.size else np.nan))

    fig_h = 0.45 * (len(rows) + 1) + 1.1
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    ax.axis("off")
    hdr = ["臂", "n", "one-step\nRMSE (px)", "std", "rollout\nk=2 (px)",
           "rollout\nk=5 (px)", "先验 (px)", "零速度 (px)", "yao acc"]

    def cell(c):
        if isinstance(c, str):
            return c                                    # 臂名是字符串，不能走数字分支
        if isinstance(c, float) and c == c:
            return f"{c:.3f}"
        if isinstance(c, (int, np.integer)):
            return str(c)
        return "—"

    data = [[cell(c) for c in r] for r in rows]
    table = ax.table(cellText=[hdr] + data, cellLoc="center",
                     bbox=[0.0, 0.10, 1.0, 0.80])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for j in range(len(hdr)):
        c = table[(0, j)]
        c.set_facecolor("#2C3E50"); c.set_text_props(color="w", weight="bold")
    for i in range(1, len(data) + 1):
        table[(i, 0)].set_text_props(weight="bold")
        for j in (2, 5):
            table[(i, j)].set_text_props(weight="bold")
    nw = _nwin(arms)
    ax.set_title("E0 统一主表（Table 1）— 铰链世界，held-out 300 episodes"
                 + (f"，{nw} 窗口" if nw else "") + "（按 one-step 升序）",
                 fontsize=13, weight="bold", pad=10)
    fig.text(0.5, 0.025,
             "闭环 rollout = 模型自回归闭环（接触反馈口径见 e0_common.validate）；"
             "「先验」列仅对有物理先验的臂有意义，无先验臂记 —。\n"
             "数字来源：analysis_output/E0_report.json（由 build_e0_report.py 从 checkpoint 重建）",
             ha="center", fontsize=8.5, color="#555")
    fig.savefig(f"{OUT}/E0_main_table.png")
    plt.close(fig)
    print(f"[fig] {OUT}/E0_main_table.png")


# ---------------------------------------------------------------- Fig 2: one-step 柱状图
def fig_one_step(arms):
    if not arms:
        print("[fig] one_step: 无数据，跳过"); return
    arms_ = sorted(arms, key=lambda a: _vals(arms, a, "one_step_model").mean())
    means = [_vals(arms, a, "one_step_model").mean() for a in arms_]
    stds = [_vals(arms, a, "one_step_model").std() for a in arms_]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(range(len(arms_)), means, yerr=stds, capsize=5,
                  color=[_color(a, i) for i, a in enumerate(arms_)],
                  edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(arms_)))
    ax.set_xticklabels([ARM_LABELS.get(a, a) for a in arms_], rotation=15,
                       ha="right")
    ax.set_ylabel("one-step RMSE (px)")
    ax.set_title("E0 one-step 预测（3 种子，误差棒 = 1 std）")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + max(stds) * 0.15,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9, weight="bold")
    fig.savefig(f"{OUT}/E0_one_step_bars.png")
    plt.close(fig)
    print(f"[fig] {OUT}/E0_one_step_bars.png")


# ---------------------------------------------------------------- Fig 3: rollout 曲线
def fig_rollout(arms):
    """全臂闭环 rollout 曲线。
    旧版硬编码 ("noprior","ours_base","ours_gua") 三臂，恰好排除了在闭环上
    远好于这三角的 strong_mlp（无先验）与 jepa —— 属选择性展示，已改为全臂。"""
    arms_ = [a for a in sorted(arms) if _roll(arms, a, 5).size]
    if not arms_:
        print("[fig] rollout: 无 rollout 数据，跳过"); return
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xs = np.arange(1, 6)
    k5 = []
    for i, a in enumerate(arms_):
        kvals = np.array([[r.get("rollout", {}).get(f"k={k}") for k in range(1, 6)]
                          for r in arms[a]], dtype=float)
        kvals = np.nan_to_num(kvals, nan=0.0)
        m, s = kvals.mean(0), kvals.std(0)
        k5.append(max(m[-1], 1e-6))
        ax.errorbar(xs, m, yerr=s, marker="o", capsize=4,
                    label=ARM_LABELS.get(a, a).replace("\n", " "),
                    color=_color(a, i), lw=1.8, ms=5)
    logscale = (max(k5) / min(k5)) > 25
    if logscale:
        ax.set_yscale("log")
        ax.set_ylabel("pos_rmse (px，对数轴)")
        ax.set_title("E0 闭环 rollout 曲线（全部 %d 臂，3 种子均值 ± std，对数纵轴）"
                     % len(arms_))
    else:
        ax.set_ylabel("pos_rmse (px)")
        ax.set_title("E0 闭环 rollout 曲线（全部 %d 臂，3 种子均值 ± std）" % len(arms_))
    ax.set_xticks(xs); ax.set_xticklabels([f"k={k}" for k in xs])
    ax.set_xlabel("rollout 步长")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25, which="both")
    fig.savefig(f"{OUT}/E0_rollout_curves.png")
    plt.close(fig)
    print(f"[fig] {OUT}/E0_rollout_curves.png (arms={arms_}, log={logscale})")


# ---------------------------------------------------------------- Fig 4: 卦 occupancy 热力图
def fig_gua_occ(arms):
    if "ours_base" not in arms:
        print("[fig] gua_occ: 无 ours_base 数据，跳过"); return
    ours_base = arms["ours_base"]
    occ = np.array([[r["gua_occupancy"].get(str(k), 0.0) for k in range(8)]
                    for r in ours_base])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    im = ax.imshow(occ, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"{n}\n({c})" for n, c in zip(GUA, GUA_CODE)],
                       rotation=0)
    ax.set_yticks(range(3)); ax.set_yticklabels(["s42", "s43", "s44"])
    ax.set_title("ours_base 卦 occupancy（20k 窗口）")
    for i in range(3):
        for j in range(8):
            ax.text(j, i, f"{occ[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if occ[i, j] < 0.25 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="occupancy")
    fig.savefig(f"{OUT}/E0_gua_occupancy.png")
    plt.close(fig)
    print(f"[fig] {OUT}/E0_gua_occupancy.png")


# ---------------------------------------------------------------- Fig 5: 卦 RMSE
def fig_gua_rmse(arms):
    if "ours_base" not in arms:
        print("[fig] gua_rmse: 无 ours_base 数据，跳过"); return
    g = np.array([[r["gua_rmse"].get(str(k)) for k in range(8)]
                  for r in arms["ours_base"]], dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    idx = np.arange(8)
    m = np.nanmean(g, axis=0); s = np.nanstd(g, axis=0)
    show = ~np.isnan(m)
    ax.bar(idx[show], m[show], yerr=s[show], capsize=4,
           color="#F58518", edgecolor="white")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{n}\n({c})" for n, c in zip(GUA, GUA_CODE)],
                       rotation=0)
    ax.set_ylabel("one-step RMSE (px)")
    ax.set_title("ours_base 卦级误差（3 种子；坎/巽 occupancy=0%，不可测）")
    ax.axhline(y=np.nanmean(m[show]), color="#7F7F7F", lw=1, ls=":")
    for i in range(8):
        if show[i]:
            ax.text(i, m[i] + 0.03, f"{m[i]:.3f}", ha="center", va="bottom",
                    fontsize=8.5)
    fig.savefig(f"{OUT}/E0_gua_rmse.png")
    plt.close(fig)
    print(f"[fig] {OUT}/E0_gua_rmse.png")


# ---------------------------------------------------------------- Fig 6: 基线对比
def fig_baselines(arms):
    if "ours_base" not in arms:
        print("[fig] baselines: 无 ours_base 数据，跳过"); return
    ob = arms["ours_base"]
    keys = ["one_step_model", "one_step_prior", "one_step_zero_vel"]
    LAB = {"one_step_model": "模型（残差头）", "one_step_prior": "纯先验",
           "one_step_zero_vel": "零速度"}
    vals = [np.nanmean([r.get(k) for r in ob
                        if r.get(k) is not None and
                        (not isinstance(r.get(k), float)
                         or r.get(k) == r.get(k))]) for k in keys]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    bars = ax.bar(range(3), vals, color=["#4C78A8", "#E45756", "#7F7F7F"],
                  edgecolor="white")
    ax.set_xticks(range(3))
    ax.set_xticklabels([LAB[k] for k in keys])
    ax.set_ylabel("one-step RMSE (px)")
    ax.set_title("ours-base 内部基线对比（3 种子均值）")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.03,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9.5,
                weight="bold")
    fig.savefig(f"{OUT}/E0_baselines.png")
    plt.close(fig)
    print(f"[fig] {OUT}/E0_baselines.png")


def main():
    d, arms = load()
    print(f"report: {len(arms)} arms: {sorted(arms)}")
    fig_main_table(d, arms)
    fig_one_step(arms)
    fig_rollout(arms)
    fig_gua_occ(arms)
    fig_gua_rmse(arms)
    fig_baselines(arms)
    print("[done]")


if __name__ == "__main__":
    main()