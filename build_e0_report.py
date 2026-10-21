# -*- coding: utf-8 -*-
"""E0 / 任意实验主表报告 — 从 checkpoint 重建（Table 1 的**唯一**数字来源）。

用法：
  python build_e0_report.py                                   # E0 主表（默认）
  python build_e0_report.py --checkpoint_root checkpoints-P0b \\
                            --out analysis_output/P0b_report.json
输出：--out 指定的 JSON（build_e0_figures.py 只读 E0_report.json 出图）
注意：A0 zero-vel / A1 inertia 等解析臂是 0 参数、无 checkpoint，在此直接评估，
      与 s42 的 checkpoint 臂共用同一 val 划分，故可与 s42 逐臂对比。
      **不同训练配方必须用不同的 --checkpoint_root**（覆盖保护会比较 w_roll/lr）。
"""
import argparse
import json, glob, os, torch, numpy as np

from e0_common import make_datasets, validate
from physics_prior_spec import DEFAULT_SPEC, SPECS
import model_e0

# 解析臂（0 参数、无 checkpoint）：zero-vel + 先验质量阶梯的四个档位。
# 先验阶梯的实测（30k 窗口，单步/闭环 k=5，px）见 tmp_prior_ladder_pilot.py：
#   euler005 3.630/16.27 · euler001x5 0.532/3.35 · rk4005 0.167/2.13 · rk4001x5 0.038/0.50
ANALYTIC_ARMS = ("zero_vel", "inertia", "inertia_euler001x5", "inertia_rk4005",
                 "inertia_rk4001x5")
BASE_SEED = 42                            # 解析臂沿用 s42 的数据划分

_p = argparse.ArgumentParser()
_p.add_argument("--checkpoint_root", type=str, default="checkpoints-E0")
_p.add_argument("--out", type=str, default="analysis_output/E0_report.json")
_p.add_argument("--title", type=str, default="统一主表（Table 1）— 修复后代码重训")
_p.add_argument("--skip_analytic", action="store_true",
                help="不评估解析臂（仅当只想看该 root 的训练臂时用）")
ARGS = _p.parse_args()
CKPT_ROOT = ARGS.checkpoint_root
REPORT = ARGS.out


def parse_task(name):
    """目录名 → (arm, prior_spec)。约定 `<arm>@<spec>_s<seed>` 或 `<arm>_s<seed>`。"""
    head = name.rsplit("_s", 1)[0]
    if "@" in head:
        arm, spec = head.split("@", 1)
        return arm, spec
    return head, None


arms = {}
for dd in sorted(glob.glob(f"{CKPT_ROOT}/*")):
    if not os.path.isdir(dd):
        continue
    name = os.path.basename(dd)
    arm, spec_from_name = parse_task(name)
    pt = os.path.join(dd, "best.pt")
    if not os.path.exists(pt):
        continue
    try:
        ck = torch.load(pt, map_location="cpu", weights_only=False)
        ck_args = ck.get("args", {})
        seed = ck_args.get("seed", ck.get("config", {}).get("seed", -1))
        data_path = ck_args.get("data_path", "trajectories_hinge.npz")
        # 先验档位：checkpoint args 为准，其次目录名，最后默认
        prior_spec = ck_args.get("prior_spec") or spec_from_name or DEFAULT_SPEC
        if prior_spec not in SPECS:
            print(f"  SKIP {name}: 未知 prior_spec={prior_spec}")
            continue
        _, val_ds, mean_obj, std_obj = make_datasets(
            data_path, L=8, K=5, seed=seed, prior_spec=prior_spec)
        model = model_e0.make_model(arm, prior_spec=prior_spec)
        model.set_stats(mean_obj, std_obj)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        # 表里的"臂"键：默认档位沿用裸臂名（兼容历史 checkpoint）；
        # 非默认档位用 `<arm>@<spec>`，否则同一架构的 4 个先验档位会被合并成一条
        # （12 个 run 混在一起 → 无法做阶梯比较）。
        key = arm if prior_spec == DEFAULT_SPEC else f"{arm}@{prior_spec}"
        vm = validate(model, val_ds, torch.device("cpu"), K=5, max_windows=30000)
        rec_spec = getattr(model, "prior_spec", None)
        arms.setdefault(key, []).append(dict(seed=seed, epoch=ck.get("epoch"),
                                             prior_spec=rec_spec, task=name, **vm))
    except Exception as e:
        print("  SKIP %s: %s" % (name, e))

# ---- 解析基线（0 参数，无需训练）：直接测「先验本身」的单步与闭环行为 ----
for arm in ANALYTIC_ARMS:
    if ARGS.skip_analytic or arm in arms:
        continue
    spec = model_e0.ARM_SPECS[arm].get("prior_spec", DEFAULT_SPEC)
    _, val_ds, mean_obj, std_obj = make_datasets(
        "trajectories_hinge.npz", L=8, K=5, seed=BASE_SEED, prior_spec=spec)
    model = model_e0.make_model(arm, prior_spec=spec)
    model.set_stats(mean_obj, std_obj)
    vm = validate(model, val_ds, torch.device("cpu"), K=5, max_windows=30000)
    arms.setdefault(arm, []).append(dict(seed=BASE_SEED, epoch=0,
                                         prior_spec=getattr(model, "prior_spec", None),
                                         **vm))
    print("  [analytic] %-22s spec=%-11s 已评估（无 checkpoint）" % (arm, spec))

# 汇总
print(f"\n=== E0 主表（{len(arms)} 臂）===\n")
print(f"{'臂':<22} {'spec':<11} {'n':>3} {'one-step':>10} {'±std':>8} {'prior':>10} "
      f"{'zero-vel':>10} {'roll k=2':>10} {'roll k=5':>10}")
print("-" * 118)
ranking = []
for arm in sorted(arms):
    seeds = arms[arm]
    specs = sorted({s.get("prior_spec") or "—" for s in seeds})
    spec_txt = specs[0] if len(specs) == 1 else "MIXED"
    vals = [s.get("one_step_model", 1e9) for s in seeds if "one_step_model" in s]
    priors = [s.get("one_step_prior", 1e9) for s in seeds if "one_step_prior" in s]
    zvels = [s.get("one_step_zero_vel", 1e9) for s in seeds if "one_step_zero_vel" in s]
    r2 = [s.get("rollout", {}).get("k=2", np.nan) for s in seeds]
    r5 = [s.get("rollout", {}).get("k=5", np.nan) for s in seeds]
    if vals:
        m, s = np.mean(vals), np.std(vals)
        pm = np.mean(priors) if priors else float('nan')
        zm = np.mean(zvels) if zvels else float('nan')
        r2m = np.nanmean(r2) if np.any(np.isfinite(r2)) else float('nan')
        r5m = np.nanmean(r5) if np.any(np.isfinite(r5)) else float('nan')
        print(f"{arm:<22} {spec_txt:<11} {len(vals):>3} {m:>10.3f} {s:>8.3f} {pm:>10.3f} "
              f"{zm:>10.3f} {r2m:>10.2f} {r5m:>10.2f}")
        ranking.append((arm, m, s, len(vals)))
    else:
        print(f"{arm:<22} {spec_txt:<11}   -         -         -         -")

ranking.sort(key=lambda x: x[1])
print(f"\n排名（one-step 越小越好）:")
for i, (arm, m, s, n) in enumerate(ranking):
    print(f"  {i+1}. {arm}: {m:.3f} ± {s:.3f} (n={n})")

# 写 JSON
d = {
    "experiment": "E0" if CKPT_ROOT == "checkpoints-E0" else CKPT_ROOT,
    "title": ARGS.title,
    "checkpoint_root": CKPT_ROOT,
    "note": (f"唯一数字来源：本文件由 build_e0_report.py 从 {CKPT_ROOT}/*/best.pt 重建，"
             "图（figures/E0_*.png）只读 analysis_output/E0_report.json。"
             "闭环 rollout = 模型自回归闭环，接触反馈口径见 e0_common.validate。"
             "『先验』列仅对 physics_prior=True 的臂有意义，其余记 null。"
             "prior_spec = 先验积分器档位（physics_prior_spec.SPECS）；"
             "训练期残差目标与评估期先验必须同源，spec 记录在每个 run 上。"
             "目录名约定 `<arm>@<spec>_s<seed>`（旧目录名无 @ 时 spec 取 checkpoint args）。"
             "已知口径限制：不同 seed 使用各自的 val 划分（make_datasets(seed)），"
             "故 std 混合了训练方差与划分方差，逐 seed 配对比较才严格。"),
    "arms": {a: seeds for a, seeds in arms.items()},
    "ranking": [a for a, _, _, _ in ranking],
    "summary": {
        "n_arms": len(arms),
        "topline": " / ".join(f"{a}({m:.3f})" for a, m, _, _ in ranking[:3]),
        "analytic_arms": list(ANALYTIC_ARMS),
        "prior_specs": sorted({(r.get("prior_spec") or "—") for v in arms.values() for r in v}),
    }
}
os.makedirs(os.path.dirname(REPORT), exist_ok=True)
json.dump(d, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n已写入 {REPORT}")
