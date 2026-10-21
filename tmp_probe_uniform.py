# -*- coding: utf-8 -*-
"""可行性探针：能否通过抬高 GROUND_Y 把初爻（触地）频率从 5% 提到 ~40%？

初爻 = contact[:,0] = (y2 <= GROUND_Y)，即手端落到"地面"高度。
手端最低点 = -(L1+L2) = -160，当前 GROUND_Y=-150，故需接近完全下摆才触地，
频率仅 5.18%。

抬高 GROUND_Y 是纯世界几何参数（地面更高），不改动动力学、也不改变
"触地"这一爻的语义 —— 是最干净的干预。

同时报告每个 GROUND_Y 下的 8 卦占用与熵，判断均匀相态是否物理可达。
"""
import io
import math
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import generate_hinge as G
from layout_h import yao_labels_h
from generate_hinge import SEQ_LEN

GUA = ["坤", "乾", "震", "巽", "坎", "艮", "离", "兑"]


def occupancy(yao, obj, T):
    idx = (yao[:, :, :, obj] * np.array([4, 2, 1])[None, None, :]).sum(axis=2)
    tot = np.bincount(idx.reshape(-1), minlength=8).astype(float)
    return tot / tot.sum()


def entropy(p):
    q = p[p > 0]
    return -(q * np.log(q)).sum() / math.log(8)


def probe(ground_y, n_ep, seed):
    G.GROUND_Y = ground_y
    rng = np.random.default_rng(seed)
    yaos = np.zeros((n_ep, SEQ_LEN, 3, 2), dtype=np.int8)
    for ep in range(n_ep):
        target = (rng.uniform(-120, 120), rng.uniform(-140, -40))
        st, ct, ac = G.run_episode(rng, target)
        yaos[ep] = yao_labels_h(st, ct, target)
    p0 = occupancy(yaos, 0, SEQ_LEN)
    p1 = occupancy(yaos, 1, SEQ_LEN)
    return float(yaos[:, :, 0, 0].mean()), p0, p1


def main():
    n_ep = 120
    print("探针：GROUND_Y 扫描  (每点 %d 条轨迹, 10s/条)" % n_ep)
    print("手端最低点 y2min = -160 ;  GROUND_Y 越低越难触地\n")
    print("%-11s %9s %9s %9s | %-34s" %
          ("GROUND_Y", "初爻率", "obj0熵", "obj1熵", "obj0 卦分布(0,2,4,6 之外为空)"))
    print("-" * 92)
    t0 = time.time()
    rows = []
    for gy in [-150.0, -145.0, -140.0, -135.0, -130.0, -125.0, -120.0]:
        rate, p0, p1 = probe(gy, n_ep, 12345)
        e0, e1 = entropy(p0), entropy(p1)
        s = "  ".join("%.0f%%" % (p0[g] * 100) for g in range(8))
        print("%-11.0f %8.1f%% %8.1f%% %8.1f%% | %s" % (gy, rate * 100, e0 * 100, e1 * 100, s))
        rows.append((gy, rate, e0, e1, p0, p1))
    print("-" * 92)
    print("用时 %.1fs" % (time.time() - t0))
    print("\n参考：原始数据 obj0 熵 43.0% / 物体0最大卦 47.8% ; obj1 熵 72.9%")
    print("均匀目标：obj0 在 4 个可达卦上各 25%(熵 66.7%)；obj1 在 8 卦上各 12.5%(熵 100%)")

    best = max(rows, key=lambda r: r[2] + r[3])
    print("\n熵合计最高: GROUND_Y=%.0f  初爻率 %.1f%%  obj0熵 %.1f%%  obj1熵 %.1f%%"
          % (best[0], best[1] * 100, best[2] * 100, best[3] * 100))


if __name__ == "__main__":
    main()
