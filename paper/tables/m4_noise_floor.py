# -*- coding: utf-8 -*-
"""噪声底矩阵：换 seed × 重训 × 指标，**同一协议**。

为什么需要这个脚本
------------------
论文 §8 原有的噪声底是一串单维数字（重训差：单步 0.5%、闭环 k=2 0.2%、
闭环 k=5 15%），全部来自**一次重训**。`internal/M4_IN_PROGRESS.md` 第五节指出：
换 seed 会带来同量级的变化，所以噪声底应当是
**（重训 × 换 seed）× 指标** 的矩阵。

两条读数的地位（主 / 诊断）
---------------------------
* **R1 主口径 —— 每个 seed 用它自己的划分与统计量**
  （`e0_common.make_datasets(seed)`，与 `build_e0_report.py` 逐位一致）。
  **val 与训练集天然不重叠**，所以它测到的是"论文主表数字在换 seed 时的
  真实波动"，同时含优化方差与划分方差。这是唯一可作为噪声底引用的读数。
* **R2 诊断 —— 所有 seed 共用 s42 的划分与统计量**
  （`m4_floor_partial.py` 的口径）。它固定了量尺，但**对 s≠42 的 seed，
  这份 val 有约 90% 落在其训练集里**（脚本算出实际重叠率并落盘）。
  因此 R2 不能当"干净量尺"用，只能用来回答一个诊断问题：
  不同 seed 的指标对统一验证划分有多敏感；由于验证窗口与训练集重叠，R2 不能估计划分对总体波动的因果贡献。

两者都走 `e0_common.validate`（E0 主表的同一个评估函数）；跨脚本比较
（E5/M4 的另一套初始化与调度）在本项目已被 `M4_IN_PROGRESS.md` 明令禁止。

输出
----
  analysis_output/noise_floor_matrix.json

用法
----
  python -X utf8 paper/tables/m4_noise_floor.py            # 四档 × 7 seed
  python -X utf8 paper/tables/m4_noise_floor.py --tier euler005
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
AO = os.path.join(ROOT, "analysis_output")

from e0_common import E0Data, validate                          # noqa: E402
from layout_h import OBJ_DIMS                                   # noqa: E402
from model_e0 import make_model                                 # noqa: E402

DATA = os.path.join(ROOT, "trajectories_hinge.npz")
L, K = 8, 5
MAXW = 30000
ARM = "ours_base"
SEEDS = (42, 43, 44, 45, 46, 47, 48)
SPLIT_SEED = 42                    # R2 的固定划分 = s42 自己的划分
N_VAL_EPS = 300
METRICS = (("one_step", "one-step"), ("k=2", "closed-loop k=2"), ("k=5", "closed-loop k=5"))
TIERS = [
    ("euler005", "coarse Euler@.05x1"),
    ("euler001x5", "Euler@.01x5"),
    ("rk4005", "RK4@.05x1"),
    ("rk4001x5", "exact RK4@.01x5"),
]


def arm_key(spec):
    return ARM if spec == "euler005" else f"{ARM}@{spec}"


CKPT_ROOT = os.path.join(ROOT, "checkpoints-E0")


def ckpt_path(spec, seed, roots=None):
    """最先存在的 checkpoint；支持多个根目录（原 E0 与扩展 seed 分开放）。"""
    roots = roots or [CKPT_ROOT]
    if isinstance(roots, str):
        roots = [roots]
    name = f"{arm_key(spec)}_s{seed}"
    for r in roots:
        p = os.path.join(r, name, "best.pt")
        if os.path.exists(p):
            return p
    return os.path.join(roots[0], name, "best.pt")


def core(vm):
    return {"one_step": vm["one_step_model"],
            "k=2": vm["rollout"]["k=2"],
            "k=5": vm["rollout"]["k=5"]}


def cv_pct(vals):
    vals = [v for v in vals if v == v]
    if len(vals) < 2:
        return float("nan"), (float(np.mean(vals)) if vals else float("nan")), 0.0
    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    return (100.0 * sd / m if m else float("nan")), m, sd


def loo_cv_pct(vals):
    """留一法：去掉使 CV 最大的那个 seed 后重算 CV。

    目的：seed 轴上常有单个离群 run（本例是粗档 s43）。报告"去掉最坏点后的 CV"
    是为了说明结论**不是**由单个 seed 绑架的——若留一后仍远大于重训差，
    则"seed 主导"成立；若留一后塌回重训量级，就必须明说结论依赖单点。
    """
    vals = [v for v in vals if v == v]
    if len(vals) < 3:
        return float("nan"), None
    cvs = []
    for i in range(len(vals)):
        rest = vals[:i] + vals[i + 1:]
        c, _, _ = cv_pct(rest)
        cvs.append((c, i))
    cvs = [(c, i) for c, i in cvs if c == c]
    if not cvs:
        return float("nan"), None
    worst_cv, worst_i = min(cvs)          # 去掉后 CV 最小 = 该点最离群
    return worst_cv, worst_i


def eval_ckpt(path, spec, ds, mo, so):
    m = make_model(ARM, prior_spec=spec).to("cpu")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    m.set_stats(mo, so)
    vm = validate(m, ds, torch.device("cpu"), K=K, max_windows=MAXW)
    return core(vm), ck.get("epoch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=None, help="只跑某一档（默认四档全跑）")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="覆盖 seed 列表（默认 %s）；扩展种子需配合 --ckpt-root" % (list(SEEDS),))
    ap.add_argument("--ckpt-root", nargs="+", default=None,
                    help="一个或多个 checkpoint 根目录，按顺序查找（默认 checkpoints-E0）")
    ap.add_argument("--out", default="noise_floor_matrix.json",
                    help="输出 JSON 文件名（默认写论文的 noise_floor_matrix.json）")
    args = ap.parse_args()

    seeds = tuple(args.seeds) if args.seeds else SEEDS
    ckpt_roots = args.ckpt_root or [CKPT_ROOT]
    print(f"[config] seeds={list(seeds)} "
          f"ckpt_roots={[os.path.relpath(r, ROOT) for r in ckpt_roots]} out={args.out}")

    d = np.load(DATA)
    st, ct, ac, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = st.shape[0]

    # ---- 每个 seed 用自己的划分（复刻 make_datasets 的切法）----
    own = {}
    for s in seeds:
        seg = np.arange(N)
        np.random.RandomState(s).shuffle(seg)
        val_eps, tr_eps = seg[:N_VAL_EPS], seg[N_VAL_EPS:]
        obj_flat = st[tr_eps].reshape(-1, OBJ_DIMS).astype(np.float64)
        own[s] = dict(val=val_eps, tr=tr_eps,
                      mo=obj_flat.mean(0), so=obj_flat.std(0))

    # ---- R2 的重叠诊断 ----
    fval = set(own[SPLIT_SEED]["val"].tolist())
    overlap = {}
    for s in seeds:
        inter = len(fval & set(own[s]["tr"].tolist()))
        overlap[str(s)] = dict(n_train_episodes_inside_fixed_val=inter,
                               frac_of_fixed_val=inter / len(fval))
    print("[R2 诊断] s42 的划分被当作公共量尺时与各 seed 训练集的重叠：")
    print("          " + "  ".join(f"s{s}:{100*overlap[str(s)]['frac_of_fixed_val']:.0f}%"
                                   for s in seeds))
    print("          （重叠高 -> R2 对 s≠42 是污染读数，只能作诊断）")

    tiers = TIERS if args.tier is None else [t for t in TIERS if t[0] == args.tier]

    out = {"arm": ARM, "protocol": "e0_common.validate（与 E0 主表同一评估函数）",
           "n_val_episodes": N_VAL_EPS, "max_windows": MAXW,
           "seeds": [int(s) for s in seeds],
           "ckpt_roots": [os.path.relpath(r, ROOT).replace("\\", "/") for r in ckpt_roots],
           "readings": {
               "R1_own_split": "主口径：每个 seed 用自己 seed 的划分与统计量（= build_e0_report），val 不与训练集重叠",
               "R2_common_split": "诊断：共用 s42 的划分与统计量（= m4_floor_partial），对 s≠42 有训练集重叠"},
           "overlap_R2": overlap, "tiers": {}}

    for spec, label in tiers:
        print(f"\n=== {label} ({spec}) ===")
        # R1：各自的划分与统计量
        ds_cache = {}
        r1 = {}
        for s in seeds:
            p = ckpt_path(spec, s, ckpt_roots)
            if not os.path.exists(p):
                print(f"  [skip] s{s} 无 checkpoint")
                continue
            o = own[s]
            ds = E0Data(st, ct, ac, yao, o["val"], L, K, o["mo"], o["so"], prior_spec=spec)
            met, ep = eval_ckpt(p, spec, ds, o["mo"], o["so"])
            r1[str(s)] = dict(met, epoch=ep)
            print(f"  [R1] s{s:<3} ep={ep:<3} one-step={met['one_step']:.4f} "
                  f"k2={met['k=2']:.4f} k5={met['k=5']:.4f}")
        # R2：共用 s42 的划分与统计量
        o42 = own[SPLIT_SEED]
        ds_fix = E0Data(st, ct, ac, yao, o42["val"], L, K, o42["mo"], o42["so"],
                        prior_spec=spec)
        r2 = {}
        for s in seeds:
            p = ckpt_path(spec, s, ckpt_roots)
            if not os.path.exists(p):
                continue
            met, ep = eval_ckpt(p, spec, ds_fix, o42["mo"], o42["so"])
            r2[str(s)] = dict(met, epoch=ep)
        out["tiers"][spec] = {"label": label, "R1_own_split": r1, "R2_common_split": r2}

    # ---- 重训轴（仅粗档 s42 有 REPRO）----
    retrain = {}
    repro = os.path.join(ROOT, "checkpoints-REPRO", f"{ARM}_s42", "best.pt")
    if os.path.exists(repro) and args.tier in (None, "euler005"):
        o42 = own[SPLIT_SEED]
        ds = E0Data(st, ct, ac, yao, o42["val"], L, K, o42["mo"], o42["so"],
                    prior_spec="euler005")
        met, ep = eval_ckpt(repro, "euler005", ds, o42["mo"], o42["so"])
        retrain["42"] = dict(met, epoch=ep)
        out["retrain_axis"] = retrain
        print(f"\n[retrain] s42 epoch={ep} one-step={met['one_step']:.4f} "
              f"k2={met['k=2']:.4f} k5={met['k=5']:.4f}")

    # ---- 汇总 ----
    print("\n" + "=" * 96)
    print("噪声底矩阵（同一评估协议）")
    print("=" * 96)
    print(f"{'tier':20s} {'metric':18s} {'R1 CV%':>9s} {'R1 LOO CV%':>20s} "
          f"{'R2 CV%':>9s} {'retrain %':>10s} {'R1/retrain':>11s} {'判定':>8s}")
    print("-" * 112)
    for spec, label in tiers:
        t = out["tiers"][spec]
        t["summary"] = {}
        for key, mlab in METRICS:
            v1 = [t["R1_own_split"][s][key] for s in sorted(t["R1_own_split"], key=int)]
            v2 = [t["R2_common_split"][s][key] for s in sorted(t["R2_common_split"], key=int)]
            c1, m1, sd1 = cv_pct(v1)
            c2, m2, sd2 = cv_pct(v2)
            loo, loo_i = loo_cv_pct(v1)
            loo_seed = (sorted(t["R1_own_split"], key=int)[loo_i]
                        if loo_i is not None else None)
            ret = float("nan")
            if spec == "euler005" and "42" in retrain and "42" in t["R1_own_split"]:
                a = t["R1_own_split"]["42"][key]
                b = retrain["42"][key]
                ret = 100.0 * abs(b - a) / abs(a) if a else float("nan")
            ratio = (c1 / ret) if (ret == ret and ret > 0 and c1 == c1) else float("nan")
            verdict = ""
            if ratio == ratio:
                verdict = "seed 主导" if ratio > 1 else "重训主导"
            t["summary"][key] = dict(
                label=mlab, n=len(v1),
                R1_seed_mean=m1, R1_seed_sd=sd1, R1_seed_cv_pct=c1,
                R2_seed_mean=m2, R2_seed_sd=sd2, R2_seed_cv_pct=c2,
                retrain_pct=ret, R1_cv_over_retrain=ratio, verdict=verdict,
                R1_loo_cv_pct=loo, R1_loo_dropped_seed=loo_seed)
            loo_txt = f"{loo:6.2f} (drop s{loo_seed})" if loo == loo else "   n/a"
            print(f"{label:20s} {mlab:18s} {c1:9.2f} {loo_txt:>20s} {c2:9.2f} "
                  f"{ret:10.2f} {ratio:11.2f} {verdict:>8s}")
    print("-" * 112)
    print("R1 = 主口径（各自划分，无训练集重叠）；R2 = 受污染的公共划分敏感性诊断（s≠42 与训练数据重叠），不估计划分效应。")
    print("R1 LOO = 去掉最离群 seed 后的 R1 CV（>0 说明结论不由单个 seed 绑架）。")
    print("重训差仅粗档 s42 可测：checkpoints-REPRO 与 checkpoints-E0 为同配置两次训练。")

    with open(os.path.join(AO, args.out), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[save] analysis_output/{args.out}")


if __name__ == "__main__":
    main()
