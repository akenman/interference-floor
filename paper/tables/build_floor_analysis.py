# -*- coding: utf-8 -*-
"""汇总干扰地板的最终测量，产出 analysis_output/floor_analysis.json（v2）。

这一版替换了 v1，原因是 v1 引用的是第六轮**用错头结构**（build_head 多接了一个
GELU）的静默上限曲线。第七、八轮做了三件事：
  1. 修正头结构（克隆 model.state_head 的真实模块序列）；
  2. 用统一的 2x2（训练窗口 400k/120k × rollout 关/开，其余全部固定）把
     "静默上限是否为缓存假象"和"rollout 项是否是元凶"放在同一执行路径下比较；
  3. 记录跨运行的离散度——同一配置在不同 RNG 执行顺序下 w_ff=100 给出
     0.24–0.58 baseline 的不同终值，这一点必须写进论文而不是取一个好看的数。

输入：crosstalk_law{,3,4}.json / probe_replicate.json / silence_ceiling.json /
      ceiling_unified.json / rollout_contrast.json / trunk_isolation.json
输出：analysis_output/floor_analysis.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AO = os.path.join(ROOT, "analysis_output")


def J(name):
    with open(os.path.join(AO, name), encoding="utf-8") as f:
        return json.load(f)


def last(cell):
    return cell[-1]


rep = J("probe_replicate.json")
law4 = J("crosstalk_law4.json")
law3 = J("crosstalk_law3.json")
ceil = J("silence_ceiling.json")
uni = J("ceiling_unified_partial.json")
roll = J("rollout_contrast.json")
law1 = J("crosstalk_law.json")
diag = J("crosstalk_diag.json")

# ---------------------------------------------------------------- 复现断言
assert abs(rep["euler001x5_s42"]["paper_protocol_head_rms"] - 0.0196734) < 1e-5
assert abs(rep["parabola005_s42"]["paper_protocol_head_rms"] - 0.0202643) < 1e-5

# ---------------------------------------------------------------- 档位依赖性
PRIORS = ("euler001x5", "parabola005", "rk4005", "euler005")
by_prior = {}
for p in PRIORS:
    heads = [law4[s][p]["head_rms"] for s in ("42", "43", "44")]
    tgts = [law4[s][p]["target_rms"] for s in ("42", "43", "44")]
    by_prior[p] = dict(head_rms_per_seed=heads,
                       head_rms_mean=float(np.mean(heads)),
                       head_rms_sd=float(np.std(heads, ddof=1)),
                       target_rms=float(np.mean(tgts)),
                       ratio_per_seed=[h / t for h, t in zip(heads, tgts)])
means = {p: by_prior[p]["head_rms_mean"] for p in PRIORS}
sds = {p: by_prior[p]["head_rms_sd"] for p in PRIORS}


def z(a, b):
    pooled = float(np.sqrt((sds[a] ** 2 + sds[b] ** 2) / 2))
    return float((means[b] - means[a]) / pooled)


prior_dependence = dict(
    by_prior=by_prior,
    spread_max_over_min=float(max(means.values()) / min(means.values())),
    z_coarse_vs_exact=z("euler001x5", "euler005"),
    z_rk4_vs_exact=z("euler001x5", "rk4005"),
    within_prior_cv_pct=float(np.median([100 * sds[p] / means[p] for p in PRIORS])),
    clean_ff_jac_seed42={k: law3[k]["clean_ff"]["jac_frob_mean"]
                         for k in law3 if k != "probe"},
    contact_control_seed42={k: dict(jac_frob=law3[k]["contact_ctrl"]["jac_frob_mean"],
                                    head_rms=law3[k]["contact_ctrl"]["head_rms"])
                            for k in law3 if k != "probe"},
)

# ---------------------------------------------------------------- 静默上限（修正结构）
rows = {r["w_ff"]: r for r in
        [dict(w_ff=float(k), **last(v)) for k, v in ceil["grid"].items()]}
order = sorted(rows)
base = ceil["baseline"]
ceiling = dict(
    baseline=base,
    n_train=ceil["n_train"], epochs=ceil["max_epochs"],
    head_architecture="cloned from model.state_head (no extra GELU)",
    rows=[dict(w_ff=k, head_ff_rms=rows[k]["head_ff_rms"],
               ff_over_baseline=rows[k]["head_ff_rms"] / base["head_ff_rms"],
               head_ct_rms=rows[k]["head_ct_rms"],
               ct_over_baseline=rows[k]["head_ct_rms"] / base["head_ct_rms"],
               train_huber=rows[k]["train_fit"]) for k in order],
)
ff = np.array([rows[k]["head_ff_rms"] for k in order])
ct = np.array([rows[k]["head_ct_rms"] for k in order])
ceiling["monotone_decreasing"] = bool(all(ff[i] > ff[i + 1] for i in range(len(ff) - 1)))
ceiling["best_ff_over_baseline"] = float(ff.min() / base["head_ff_rms"])
ceiling["best_w_ff"] = float(order[int(np.argmin(ff))])
ceiling["ff_dynamic_range"] = float(ff.max() / ff.min())
ceiling["contact_max_deviation_pct"] = float(100 * max(abs(ct / base["head_ct_rms"] - 1)))

# ---------------------------------------------------------------- 统一 2x2（部分完成）
cells = {k: last(v) for k, v in uni["cells"].items()}
b_ff = uni["baseline"]["head_ff_rms"]
b_ct = uni["baseline"]["head_ct_rms"]
unified = dict(
    baseline_ff=b_ff, baseline_ct=b_ct, epochs=uni["epochs"], w_ff=uni["w_ff"],
    status=uni["status"],
    cells={k: dict(ff_over_baseline=v["ff_over_baseline"],
                   ct_over_baseline=v["ct_over_baseline"],
                   train_huber=v["train_huber"],
                   ff_over_baseline_measured=v["head_ff_rms"] / b_ff)
           for k, v in cells.items()},
)
if uni.get("matched_fit"):
    unified["matched_fit"] = uni["matched_fit"]
# 与同 w_ff、同轮数、同批大小但不同训练窗口数的运行比较（样本量效应）
ceiling_no_roll = rows[100.0]["head_ff_rms"] / base["head_ff_rms"]
unified["sample_size_effect"] = dict(
    w400k_no_roll=float(cells["w400k_no_roll"]["ff_over_baseline"]),
    w400k_sweep_no_roll=float(ceiling_no_roll),
    note="同配置、同轮数，只有训练窗口数与 RNG 执行顺序不同",
)

# 120k 补跑（独立进程、独立输出文件）
if os.path.exists(os.path.join(AO, "ceiling_unified_120k.json")):
    u120 = J("ceiling_unified_120k.json")
    c120 = {k: last(v) for k, v in u120["cells"].items()}
    b_ff120 = u120["baseline"]["head_ff_rms"]
    b_ct120 = u120["baseline"]["head_ct_rms"]
    unified["cells_120k"] = {
        k: dict(ff_over_baseline=v["head_ff_rms"] / b_ff120,
                ct_over_baseline=v["head_ct_rms"] / b_ct120,
                train_huber=v["train_huber"]) for k, v in c120.items()}
    nr = c120["w120k_no_roll"]
    rr = min(u120["cells"]["w120k_roll"],
             key=lambda r: abs(r["train_huber"] - nr["train_huber"]))
    unified["matched_fit_120k"] = dict(
        no_roll_epoch=nr["epoch"], roll_epoch=rr["epoch"],
        no_roll_huber=nr["train_huber"], roll_huber=rr["train_huber"],
        no_roll_ff_over=nr["ff_over_baseline"], roll_ff_over=rr["ff_over_baseline"],
        ratio=rr["head_ff_rms"] / nr["head_ff_rms"])
    unified["sample_size_effect"]["w120k_no_roll"] = float(nr["ff_over_baseline"])
    unified["sample_size_effect"]["ratio_120k_over_400k"] = float(
        nr["ff_over_baseline"] / cells["w400k_no_roll"]["ff_over_baseline"])

# ---------------------------------------------------------------- 跨运行离散度（合并所有运行）
rc = {k: last(v) for k, v in roll["cells"].items()}
pooled_ref = [float(rows[100.0]["head_ff_rms"] / base["head_ff_rms"]),
              float(cells["w400k_no_roll"]["ff_over_baseline"]),
              float(rc["no_roll"]["head_ff_rms"] / b_ff)]
pooled_roll = [float(rc["roll"]["head_ff_rms"] / b_ff),
               float(cells["w400k_roll"]["ff_over_baseline"])]
if "cells_120k" in unified:
    pooled_ref.append(float(unified["cells_120k"]["w120k_no_roll"]["ff_over_baseline"]))
    pooled_roll.append(float(unified["cells_120k"]["w120k_roll"]["ff_over_baseline"]))
spread = dict(
    no_roll_runs=pooled_ref, roll_runs=pooled_roll,
    no_roll_min=float(min(pooled_ref)), no_roll_max=float(max(pooled_ref)),
    roll_min=float(min(pooled_roll)), roll_max=float(max(pooled_roll)),
    note="同一配置（冻结主干、w_ff=100）在不同 RNG 执行顺序 / 训练窗口数下的终值不同，"
         "说明该点在目标函数下欠定；论文因此报区间而非单点。",
)

# ---------------------------------------------------------------- 自由主干对照
iso = J("trunk_isolation.json")
free_cell = iso["cells"]["free_no_roll"][-1]
frozen_cell = iso["cells"]["frozen_no_roll"][-1]
iso_out = dict(
    free_trunk_ff_over_baseline=float(free_cell["ff_over_baseline"]),
    free_trunk_ct_over_baseline=float(free_cell["ct_over_baseline"]),
    free_trunk_epoch=free_cell["epoch"],
    frozen_trunk_ff_over_baseline=float(frozen_cell["ff_over_baseline"]),
    frozen_trunk_ct_over_baseline=float(frozen_cell["ct_over_baseline"]),
    note="冻结主干与自由主干在同一 w_ff 下的可达静默不可区分 -> 终点不是缓存特征的假象；"
         "自由主干反而把接触练得更好（0.92x），说明它在优化接触而不是静默。",
    pooled_no_roll_range=[spread["no_roll_min"], spread["no_roll_max"]],
)

falsified = dict(
    P1=dict(median_ratio=0.53, loglog_r=0.301,
            note="Lip x rep_mismatch x target_contact_rms"),
    P2=dict(median_ratio=1.96, loglog_r=0.260,
            note="contact-fit ridge extrapolated to free flight"),
    directional_leverage_r=float(np.corrcoef(
        np.log([v["lev_ff"] for v in diag.values()]),
        np.log([v["head_ff_rms"] for v in diag.values()]))[0, 1]),
)

out = dict(
    reproduction=dict(
        clean_ff_head_rms_exact=rep["euler001x5_s42"]["paper_protocol_head_rms"],
        clean_ff_head_rms_legacy=rep["parabola005_s42"]["paper_protocol_head_rms"],
        n_clean=rep["euler001x5_s42"]["n_paper"]),
    prior_dependence=prior_dependence,
    silence_ceiling=ceiling,
    unified_2x2=unified,
    rollout_contrast=dict(
        no_roll_ff_over_baseline=rc["no_roll"]["head_ff_rms"] / b_ff,
        roll_ff_over_baseline=rc["roll"]["head_ff_rms"] / b_ff,
        ratio=float(rc["roll"]["head_ff_rms"] / rc["no_roll"]["head_ff_rms"]),
        no_roll_ct_over=rc["no_roll"]["head_ct_rms"] / b_ct,
        roll_ct_over=rc["roll"]["head_ct_rms"] / b_ct),
    run_spread=spread,
    trunk_freedom_control=iso_out,
    falsified_predictions=falsified,
    protocol=dict(
        window_definition="clean-FF mask (contact-free at both frames AND "
                          "velocity-consistent); the paper's head_rms_on_ff protocol",
        position_dims=[1, 2, 12, 13, 23, 24]),
)

assert abs(out["reproduction"]["clean_ff_head_rms_exact"] - 0.0197) < 5e-4
assert prior_dependence["z_coarse_vs_exact"] > 2.0
assert ceiling["monotone_decreasing"], "修正结构后上限曲线不再单调"
assert ceiling["best_ff_over_baseline"] < 0.70, "上限过低，结论方向需重写"
assert unified["matched_fit"]["ratio"] > 1.3, "rollout 未抬高自由飞行误差"
assert roll["cells"]["roll"][-1]["head_ff_rms"] / roll["cells"]["no_roll"][-1][
    "head_ff_rms"] > 2.0, "rollout 对照不足 2x"
assert "matched_fit_120k" in unified, "120k 补跑缺失"
assert unified["matched_fit_120k"]["ratio"] > 1.3, "120k 上 rollout 方向不符"
assert abs(unified["sample_size_effect"]["ratio_120k_over_400k"] - 1.0) < 0.5, \
    "样本量不再是可忽略因素，叙述需改"
assert iso_out["free_trunk_ff_over_baseline"] > 0.20, \
    "自由主干可达完全静默——缓存假象成立，叙述需重写"
assert abs(iso_out["free_trunk_ff_over_baseline"]
           - iso_out["frozen_trunk_ff_over_baseline"]) < 0.30, \
    "冻结/自由主干差异过大，需单独成段"

with open(os.path.join(AO, "floor_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("[save] analysis_output/floor_analysis.json")
print(f"  档位依赖：跨四档 {prior_dependence['spread_max_over_min']:.2f}x，"
      f"粗档 vs exact z={prior_dependence['z_coarse_vs_exact']:+.2f}")
print(f"  静默上限（修正结构）：单调={ceiling['monotone_decreasing']}，"
      f"w_ff=100 -> {ceiling['best_ff_over_baseline']:.3f}x，"
      f"动态范围 {ceiling['ff_dynamic_range']:.2f}x，"
      f"接触最大偏离 {ceiling['contact_max_deviation_pct']:.1f}%")
print(f"  跨运行离散：w_ff=100 无 rollout 终值 "
      f"{spread['no_roll_min']:.3f}–{spread['no_roll_max']:.3f}x baseline")
print(f"  rollout 对照（统一 2x2，匹配拟合）：x{unified['matched_fit']['ratio']:.2f}")
print(f"  样本量：w_ff=100 无 rollout 终值 400k={unified['sample_size_effect']['w400k_no_roll']:.3f} "
      f"120k={unified['sample_size_effect'].get('w120k_no_roll', float('nan')):.3f} "
      f"(x{unified['sample_size_effect'].get('ratio_120k_over_400k', float('nan')):.2f})")
print(f"  rollout 对照（120k，匹配拟合）：x{unified['matched_fit_120k']['ratio']:.2f}")
print(f"  跨全部运行：无 rollout {spread['no_roll_min']:.3f}-{spread['no_roll_max']:.3f}，"
      f"有 rollout {spread['roll_min']:.3f}-{spread['roll_max']:.3f}")
print(f"  主干自由度对照：冻结 {iso_out['frozen_trunk_ff_over_baseline']:.3f} vs "
      f"自由 {iso_out['free_trunk_ff_over_baseline']:.3f}"
      f"（接触 {iso_out['frozen_trunk_ct_over_baseline']:.3f} -> "
      f"{iso_out['free_trunk_ct_over_baseline']:.3f}）")
print(f"  rollout 对照（同执行路径）："
      f"x{out['rollout_contrast']['ratio']:.2f}")
