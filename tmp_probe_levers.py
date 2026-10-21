# -*- coding: utf-8 -*-
"""扩展探针：三条提高初爻（触地）率的杠杆，各自的上限与代价。

杠杆A：抬高 GROUND_Y —— 把"地面"抬高（会变成荒谬的几何）
杠杆B：削弱力矩 TAU_MAX —— 摆动变小，手端长时间停在垂下位置（y=-160 < -150）
杠杆C：初始条件偏向垂下姿态

判据：初爻率能否到 ~50%，且 obj0 熵能否到 66.7%（4 可达卦均匀）。
"""
import io
import math
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import generate_hinge as G
import layout_h as LH
from layout_h import yao_labels_h
from generate_hinge import SEQ_LEN

N_EP = 120
BASE_TAU = LH.TAU_MAX


def entropy(p):
    q = p[p > 0]
    return -(q * np.log(q)).sum() / math.log(8)


def occ(yaos, obj):
    idx = (yaos[:, :, :, obj] * np.array([4, 2, 1])[None, None, :]).sum(axis=2)
    tot = np.bincount(idx.reshape(-1), minlength=8).astype(float)
    return tot / tot.sum()


def run(ground_y, tau_scale, n_ep=N_EP, seed=999, hang_init=False):
    G.GROUND_Y = ground_y
    LH.TAU_MAX = BASE_TAU * tau_scale
    G.TAU_MAX = LH.TAU_MAX
    rng = np.random.default_rng(seed)
    yaos = np.zeros((n_ep, SEQ_LEN, 3, 2), dtype=np.int8)
    return


def collect(ground_y, tau_scale, n_ep=N_EP, seed=999, hang_init=False):
    G.GROUND_Y = ground_y
    LH.TAU_MAX = BASE_TAU * tau_scale
    G.TAU_MAX = LH.TAU_MAX
    orig_run = G.run_episode
    rng = np.random.default_rng(seed)
    yaos = np.zeros((n_ep, SEQ_LEN, 3, 2), dtype=np.int8)
    for ep in range(n_ep):
        target = (rng.uniform(-120, 120), rng.uniform(-140, -40))
        if hang_init:
            G.make_tau_schedule = lambda r, n: np.zeros(n)
        st, ct, ac = orig_run(rng, target)
        yaos[ep] = yao_labels_h(st, ct, target)
    return yaos


def show(tag, yaos):
    rate = float(yaos[:, :, 0, 0].mean())
    p0, p1 = occ(yaos, 0), occ(yaos, 1)
    e0, e1 = entropy(p0), entropy(p1)
    flag = "  <== 达标" if e0 >= 0.63 and e1 >= 0.90 else ""
    print("  %-34s 初爻 %5.1f%%  obj0熵 %5.1f%%  obj1熵 %5.1f%%%s"
          % (tag, rate * 100, e0 * 100, e1 * 100, flag))
    return e0, e1


def main():
    print("基线 GROUND_Y=-150, TAU=28000 → 初爻 4.4%, obj0熵 41.9%, obj1熵 71.4%")
    print("目标：初爻 ~50%, obj0熵 66.7%, obj1熵 ~100%\n")

    print("杠杆A：抬高 GROUND_Y（几何荒谬度递增）")
    for gy in [-110, -100, -90, -80, -60, -40, 0]:
        show("GROUND_Y=%d" % gy, collect(gy, 1.0))

    print("\n杠杆B：削弱力矩（摆动变小，手停在垂下位置）")
    for ts in [0.5, 0.25, 0.1, 0.05, 0.0]:
        show("TAU x%.2f (%d)" % (ts, BASE_TAU * ts), collect(-150, ts))

    print("\n杠杆C：A 与 B 组合（抬地面 + 弱力矩，找最小几何代价的解）")
    for gy, ts in [(-120, 0.25), (-110, 0.25), (-100, 0.1), (-90, 0.1), (-80, 0.05)]:
        show("GROUND_Y=%d, TAU x%.2f" % (gy, ts), collect(gy, ts))

    # 复位
    LH.TAU_MAX = G.TAU_MAX = BASE_TAU
    G.GROUND_Y = -150.0


if __name__ == "__main__":
    main()
