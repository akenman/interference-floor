# -*- coding: utf-8 -*-
"""λ 扫描：找出「相态熵显著改善」与「有效样本量保持」之间的平衡点。

背景：lam=0（无多样性正则）时指数梯度坍缩 —— 3000 条轨迹只剩 60 条唯一
（重复率 98%），数据集无效。本脚本扫描 lam，报告：
  物体0/物体1 的占用熵（占均匀的比例）、最大卦占比、重采样后唯一轨迹数、ESS
据此选定可交付的 λ。
"""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from make_uniform_phase_dataset import gua_counts, reweight, target_dist, ess

LAMS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def entropy(C):
    tot = C.sum(axis=0)
    p = tot / tot.sum()
    p = p[p > 0]
    return -(p * np.log(p)).sum() / np.log(8), p.max()


def main():
    reach0 = [0, 2, 4, 6]
    reach1 = list(range(8))
    print("%-6s %14s %14s %10s %12s" %
          ("lam", "obj0熵(%均匀)", "obj1熵(%均匀)", "唯一轨迹", "ESS"))
    print("-" * 62)
    base0 = base1 = None
    for lam in LAMS:
        row = [lam]
        ents = []
        ws = []
        for src, reach, obj in [("trajectories_hinge.npz", reach0, 0),
                               ("trajectories_hinge.npz", reach1, 1)]:
            d = np.load(src, allow_pickle=True)
            yao = d["yao"]
            C = gua_counts(yao, obj, yao.shape[1])
            t = target_dist(reach)
            w = reweight(C, t, lam=lam)
            # 重采样后的占用
            Cr = np.repeat(C, 1, axis=0)  # 用同一组权重聚合
            agg = (w[:, None] * C).sum(axis=0)
            Cr2 = np.tile(agg, (1, 1))     # 形状占位
            tot = agg.sum()
            p = agg / tot
            pp = p[p > 0]
            e = -(pp * np.log(pp)).sum() / np.log(8)
            ents.append(e)
            ws.append(w)
        # 唯一轨迹数：w>1/2N 的条数（系统重采样会重复，>1/N 的才有正概率被选中）
        w = 0.5 * (ws[0] + ws[1])
        w = w / w.sum()
        N = len(w)
        uniq = int((w > 1.0 / N).sum())
        print("%-6.1f %13.1f%% %13.1f%% %10d %12.0f" %
              (lam, ents[0] * 100, ents[1] * 100, uniq, ess(w)))
        if base0 is None:
            base0, base1 = ents

    # 原始基线
    d = np.load("trajectories_hinge.npz", allow_pickle=True)
    yao = d["yao"]
    print("-" * 62)
    for obj, reach in [(0, reach0), (1, reach1)]:
        C = gua_counts(yao, obj, yao.shape[1])
        e, mx = entropy(C)
        print("原始 物体%d: 熵 %.1f%% (均匀)  最大卦 %.1f%%" % (obj, e * 100, mx * 100))


if __name__ == "__main__":
    main()
