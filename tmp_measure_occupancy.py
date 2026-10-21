# -*- coding: utf-8 -*-
"""测量铰链数据集的爻/卦占用分布，判断「均匀相态」是否可达。

背景：T10 把 gua-MoE 的劣势归因于相态分布偏斜（坤41%+艮31%+离18%=89%，
坎/巽永不出现）。要做「均匀相态」对照实验，前提是这 8 个（或 64 个）
卦在动力学上**可达**。若某些爻取值从不出现，重采样无法解决，必须改世界。
"""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

GUA_NAME = ["坤", "乾", "震", "巽", "坎", "艮", "离", "兑"]  # 索引按 yao0*4+yao1*2+yao2


def occupancy(yao, obj):
    """yao: (N,3) 该物体的三爻标签 → 8 卦计数。"""
    idx = (yao[:, 0] * 4 + yao[:, 1] * 2 + yao[:, 2]).astype(int)
    return np.bincount(idx, minlength=8)


def main(path="trajectories_hinge.npz"):
    d = np.load(path, allow_pickle=True)
    yao = d["yao"]                      # (N, T, 3, 2)
    N, T = yao.shape[0], yao.shape[1]
    print("数据集: %s  shape=%s  (%d 轨迹 x %d 帧)" % (path, yao.shape, N, T))

    print("\n=== 各爻轴的边缘分布（按物体分别统计，0/1 各占比）===")
    print("%-6s %-22s %-22s" % ("轴", "物体0(肘)", "物体1(手)"))
    names = ["初爻(接触)", "中爻(能量↑)", "上爻(趋近)"]
    for a in range(3):
        r = []
        for o in range(2):
            v = yao[:, :, a, o].ravel()
            r.append("P(1)=%.4f  n1=%d" % (v.mean(), int(v.sum())))
        print("%-12s %-22s %-22s" % (names[a], r[0], r[1]))

    print("\n=== 每物体 8 卦占用（yao0*4+yao1*2+yao2）===")
    for o, oname in enumerate(["物体0(肘)", "物体1(手)"]):
        ya = yao[:, :, :, o].reshape(-1, 3)
        c = occupancy(ya, o)
        tot = c.sum()
        print("\n  %s  (总帧 %d)" % (oname, tot))
        nz = 0
        for i in range(8):
            frac = c[i] / tot
            if frac > 0:
                nz += 1
            bar = "#" * int(frac * 60)
            print("    %d %s  %7.4f  %-8s %s" %
                  (i, GUA_NAME[i], frac, "%d" % c[i], bar))
        p = c / tot
        ent = -(p[p > 0] * np.log(p[p > 0])).sum()
        print("    非零卦数 = %d/8   熵 = %.3f / %.3f (均匀=%.3f)   最大占比 = %.1f%%" %
              (nz, ent, np.log(8), ent / np.log(8) * 100, p.max() * 100))

    print("\n=== 结论：均匀相态是否可达 ===")
    for o, oname in enumerate(["物体0(肘)", "物体1(手)"]):
        ya = yao[:, :, :, o].reshape(-1, 3)
        c = occupancy(ya, o)
        if (c == 0).any():
            print("  %s: 有 %d 个卦**从未出现**（索引 %s）→ 该爻取值组合在当前"
                  "动力学/采样下不可达，纯重采样无法覆盖。"
                  % (oname, int((c == 0).sum()),
                     [i for i in range(8) if c[i] == 0]))
        else:
            print("  %s: 8 个卦全部出现 → 原则上可通过重采样/重加权拉平" % oname)
    # 初爻（接触）出现率是瓶颈
    c0 = yao[:, :, 0, :].mean()
    print("\n  瓶颈：初爻=接触 的全局出现率仅 %.4f" % c0)
    print("  若初爻恒为 0，则 8 卦中有一半（含 4 个）在动力学上不可达。")


if __name__ == "__main__":
    main()
