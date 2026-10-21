# -*- coding: utf-8 -*-
"""W2 验证：yao_labels_h_v2 是否真的解除了 32/64 死容量？

用真实生成器仿真，对比 v1/v2 的：
  - 物体0/物体1 的 8 卦可达数
  - 64 联合卦可达数
  - 占用熵（占均匀比例）
  - 物体0 上爻的 P(1)
判据：v2 下物体0 应达 8/8、联合卦显著超过 32。
"""
import io
import math
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import generate_hinge as G
from layout_h import yao_labels_h, yao_labels_h_v2
from generate_hinge import SEQ_LEN

GUA = ["坤", "乾", "震", "巽", "坎", "艮", "离", "兑"]


def sim(n_ep, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_ep):
        target = (rng.uniform(-120, 120), rng.uniform(-140, -40))
        st, ct, ac = G.run_episode(rng, target)
        out.append((st, ct, target))
    return out


def stats(eps, labelfn, tag):
    n = len(eps)
    yaos = np.stack([labelfn(st, ct, tg) for st, ct, tg in eps])
    T = yaos.shape[1]
    idx = np.stack([
        (yaos[:, :, :, o] * np.array([4, 2, 1])[None, None, :]).sum(axis=2)
        for o in range(2)], axis=-1)                      # (N,T,2)
    print("\n%s" % tag)
    for o in (0, 1):
        tot = np.bincount(idx[:, :, o].reshape(-1), minlength=8).astype(float)
        p = tot / tot.sum()
        nz = int((tot > 0).sum())
        q = p[p > 0]
        e = -(q * np.log(q)).sum() / math.log(8)
        up = float(yaos[:, :, 2, o].mean())
        print("  物体%d: 可达 %d/8  熵 %5.1f%%  最大卦 %5.1f%%  上爻P(1) %.3f"
              % (o, nz, e * 100, p.max() * 100, up))
        print("        " + "  ".join("%d%s:%.1f%%" % (g, GUA[g], p[g] * 100) for g in range(8)))
    joint = (idx[:, :, 0] * 8 + idx[:, :, 1]).reshape(-1)
    jt = np.bincount(joint, minlength=64).astype(float)
    jnz = int((jt > 0).sum())
    pj = jt[jt > 0] / jt.sum()
    ej = -(pj * np.log(pj)).sum() / math.log(64)
    print("  >>> 64 联合卦: 可达 %d/64  熵 %.1f%% (均匀)  最大 %.2f%% (均匀=1.56%%)"
          % (jnz, ej * 100, pj.max() * 100))
    return jnz, ej


def main():
    eps = sim(200, 4242)
    print("仿真 200 条真实轨迹（GROUND_Y=-150, TAU=28000 原始世界）")
    stats(eps, yao_labels_h, "=== v1（现行，已落盘数据所用）===")
    stats(eps, yao_labels_h_v2, "=== v2（W2：上爻逐物体）===")


if __name__ == "__main__":
    main()
