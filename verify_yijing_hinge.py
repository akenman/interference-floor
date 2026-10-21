# -*- coding: utf-8 -*-
"""
错综对称性重测 — 铰链周期世界 (verify_yijing_hinge.py)
==================================================
2D 弹球世界的判决：错综对称性不成立（p>0.8，无周期结构）。
铰链世界第一次具备成立的物理条件：
  摆动周期中，能量升降半周天然交替（错卦 = 三爻全翻 = 半周期翻转）；
  摆动左右相天然对称（综卦 = 爻序颠倒 = 反相摆动）。

检验对象：数据层物体级三爻（与模型无关，先问世界本身）。
  相态 = [初爻 手近地, 中爻 能量升, 上爻 趋靶]（物体级三爻位模式 0..7）
判据（预注册）：
  错卦：corr(T[g→g'], T[C(g)→C(g')]) > 0.3 且 p < 0.05 → 对称结构成立
  综卦：同上 R 变换
用法：python verify_yijing_hinge.py
"""

import json
import math
import os
import argparse
import numpy as np

from layout_h import NUM_OBJ, OBJ_DIMS


TRIGRAM = {(0, 0, 0): "坤", (1, 0, 0): "震", (0, 1, 0): "坎", (0, 0, 1): "艮",
           (1, 1, 0): "兑", (1, 0, 1): "离", (0, 1, 1): "巽", (1, 1, 1): "乾"}
ALL = [(i >> 2 & 1, i >> 1 & 1, i & 1) for i in range(8)]


def rev(b):
    return (b[2], b[1], b[0])


def comp(b):
    return (1 - b[0], 1 - b[1], 1 - b[2])


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def symmetry_score(T, transform):
    pa, pb = [], []
    for ga in range(8):
        for gb in range(8):
            ra, rb = transform(ALL[ga]), transform(ALL[gb])
            ia, ib = ALL.index(ra), ALL.index(rb)
            if T[ga, gb] > 0 and T[ia, ib] > 0 and (ga, gb) != (ia, ib):
                pa.append(T[ga, gb])
                pb.append(T[ia, ib])
    if len(pa) < 4:
        return 0.0, 0
    return pearson(pa, pb), len(pa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="trajectories_hinge.npz")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--output_report", type=str,
                    default="analysis_output/yijing_hinge_report.json")
    args = ap.parse_args()

    d = np.load(args.data_path)
    states, contact, yao = d["states"], d["contact"], d["yao"]   # yao: (N,T,3,2)
    N = states.shape[0]
    print("=" * 68)
    print("错综对称性重测 — 铰链周期世界（数据层，物体级三爻）")
    print("=" * 68)

    # 物体级三爻位模式：idx = 4·初 + 2·中 + 上
    tri = (yao[:, :, 0, :].astype(int) * 4 + yao[:, :, 1, :].astype(int) * 2
           + yao[:, :, 2, :].astype(int))                        # (N, T, 2)
    # 过渡对（同一物体相邻帧）
    T8 = np.zeros((8, 8))
    n_trans = 0
    for o in range(NUM_OBJ):
        a = tri[:, :-1, o].ravel()
        b = tri[:, 1:, o].ravel()
        np.add.at(T8, (a, b), 1)
        n_trans += len(a)
    Tn = T8 / T8.sum()
    row_active = (Tn.sum(1) > 0)
    print(f"\n[转移样本] {n_trans:,} | 覆盖卦象: 行 {row_active.sum()}/8 列 {(Tn.sum(0)>0).sum()}/8")

    # 对称性检验（标签置换零分布）
    report = {"n_transitions": int(n_trans), "tests": {}}
    names = ["综卦(反相)", "错卦(对立相态)"]
    transforms = [rev, comp]
    for (name, tf) in zip(names, transforms):
        s_obs, n_pairs = symmetry_score(Tn, tf)
        null = []
        rs = np.random.RandomState(42)
        for _ in range(args.n_perm):
            perm = rs.permutation(8)
            Tp = np.zeros_like(Tn)
            for a in range(8):
                for b in range(8):
                    Tp[perm[a], perm[b]] = Tn[a, b]
            s_p, _ = symmetry_score(Tp, tf)
            null.append(s_p)
        null = np.asarray(null)
        p_val = float((null >= s_obs).mean())
        passed = bool(p_val < 0.05 and s_obs > 0.3)
        report["tests"][name] = {"score": s_obs, "n_pairs": n_pairs, "p": p_val,
                                 "null_p95": float(np.percentile(null, 95)), "passed": passed}
        print(f"[{name}] 得分 {s_obs:.3f}（{n_pairs} 对）| 零分布均值 {null.mean():.3f} "
              f"p95 {np.percentile(null, 95):.3f} | p = {p_val:.4f} → {'[OK] 成立' if passed else '[FAIL] 不成立'}")

    # 描述：最频繁卦变
    flat = [(ALL[a], ALL[b], Tn[a, b]) for a in range(8) for b in range(8) if Tn[a, b] > 1e-4]
    flat.sort(key=lambda x: -x[2])
    print("\n[Top 卦变]")
    top = []
    for a, b, p in flat[:8]:
        print(f"  {TRIGRAM[a]} → {TRIGRAM[b]}: {p:.3f}")
        top.append({"from": TRIGRAM[a], "to": TRIGRAM[b], "prob": float(p)})

    any_pass = any(v.get("passed") for v in report["tests"].values())
    report["top_transitions"] = top
    report["overall"] = "PASSED" if any_pass else "NOT_ESTABLISHED"
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 报告: {args.output_report}")
    print("判读：任一对称 p<0.05 且得分>0.3 → 八卦结构在周期世界有统计学支撑。")


if __name__ == "__main__":
    main()
