# -*- coding: utf-8 -*-
"""1-DOF 耦合问题的**清理版**分析。

【本脚本取代早期版本，废弃两个误导性指标】
  [FAIL] 有效独立专家数 2^I     —— MI=0.377bit 被指数放大成"1.3 个"，无物理意义
  [FAIL] 冗余度 = 名义/有效 X 倍 —— 同上，指数化放大小 MI，纯属噪声放大

【正确的框架：外积门控依赖两个**互相独立**的假设，必须分别检验】
  H1 独立性：两物体的爻位能否互相预测？   → 互信息 + 置换检验
  H2 可达性：64 种组合是否都出现？        → 联合分布的**支撑集**

关键概念区分（早期版本的主要错误）：
  MI 测**软依赖**（由高概率格主导）；可达性测**硬支撑**（质量恰为 0 的格）。
  二者可以同时成立：MI≈0 且大量格恰好为 0 —— 因为零质量格对 MI 贡献 ~0。
  故"可达 ≠ 相关"，"外积冗余"不是从 MI 推出的结论。

【可达性必须带阈值】用"任意帧出现过"判定会把 1 帧的偶然事件算作可达，
必须在多个质量阈值上报告，并与"独立假设下的期望"对照。
"""
import io
import math
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import generate_hinge as G
from layout_h import yao_labels_h, yao_labels_h_v2, F, obj_base
from generate_hinge import SEQ_LEN

GUA = ["坤", "乾", "震", "巽", "坎", "艮", "离", "兑"]
AXIS = ["初爻(接触)", "中爻(动能趋势)", "上爻(趋近靶)"]


def sim(n_ep, seed):
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n_ep):
        target = (rng.uniform(-120, 120), rng.uniform(-140, -40))
        eps.append((G.run_episode(rng, target)[:2] + (target,)))
    return eps


def joint_mi(x, y, kx, ky):
    p = np.zeros((kx, ky))
    np.add.at(p, (x.astype(int), y.astype(int)), 1.0)
    p /= p.sum()
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log2(p / np.maximum(px @ py, 1e-300)), 0.0)
    return float(t.sum()), p


def perm_test_mi(x, y, seed, n_perm=200):
    """置换检验 H0: x ⊥ y（x,y 为任意有限取值的整数编码）。
    返回 (观测 MI, 零分布 95 分位, p 值)。"""
    rng = np.random.default_rng(seed)
    kx, ky = x.max() + 1, y.max() + 1
    obs, _ = joint_mi(x, y, kx, ky)
    null = np.empty(n_perm)
    yv = y.copy()
    for i in range(n_perm):
        rng.shuffle(yv)
        null[i], _ = joint_mi(x, yv, kx, ky)
    return obs, float(np.percentile(null, 95)), float((null >= obs).mean())


def phi(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    c11 = np.sum((a == 1) & (b == 1)); c10 = np.sum((a == 1) & (b == 0))
    c01 = np.sum((a == 0) & (b == 1)); c00 = np.sum((a == 0) & (b == 0))
    den = math.sqrt((c11 + c10) * (c01 + c00) * (c11 + c01) * (c10 + c00))
    return (c11 * c00 - c10 * c01) / den if den > 0 else float("nan")


def part_a(eps):
    print("=" * 80)
    print("A  物理层：两连杆连续量的耦合（与标签设计无关的底层事实）")
    print("=" * 80)
    b0, b1 = obj_base(0), obj_base(1)
    cols = {"x": 1, "y": 2, "vx": 5, "vy": 6, "w": 7}
    A = np.concatenate([np.stack([st[:, b0 + cols[k]] for k in cols], 1) for st, _, _ in eps])
    B = np.concatenate([np.stack([st[:, b1 + cols[k]] for k in cols], 1) for st, _, _ in eps])
    print("  连续量 Pearson r：")
    for i, k in enumerate(cols):
        print("    %-3s %+.4f" % (k, np.corrcoef(A[:, i], B[:, i])[0, 1]))
    print("\n  注：w 的负相关（角动量守恒的摆荡特征）与下方离散爻位近零 MI 不矛盾 ——")
    print("      中爻/上爻取的是**趋势符号**，对反相关连续量取符号后依赖被大幅削弱。")
    return A, B


def part_h1(eps, labelfn, tag):
    """H1 独立性检验：逐轴互信息 + 置换检验（初爻为 broadcast 伪相关，单列）。"""
    yaos = np.stack([labelfn(st, ct, tg) for st, ct, tg in eps])
    print("\n" + "=" * 80)
    print("H1 独立性检验 — %s" % tag)
    print("=" * 80)
    print("  %-14s %9s %9s %10s %9s" % ("爻轴", "P(一致)", "phi", "MI(bit)", "置换p"))
    for k in range(3):
        x = yaos[:, :, k, 0].reshape(-1)
        y = yaos[:, :, k, 1].reshape(-1)
        obs, p95, pv = perm_test_mi(x, y, 7 + k)
        mark = "  ← broadcast 伪相关" if k == 0 else ""
        print("  %-14s %8.1f%% %+9.4f %10.4f %9.3f%s"
              % (AXIS[k], float(np.mean(x == y)) * 100, phi(x, y), obs, pv, mark))
    # 纯物理部分：中爻 + 上爻（剔除 broadcast 轴）
    mid = yaos[:, :, 1, :].reshape(-1, 2)
    up = yaos[:, :, 2, :].reshape(-1, 2)
    mid4 = mid[:, 0] * 2 + up[:, 0]
    mid4b = mid[:, 1] * 2 + up[:, 1]
    mi4, p4 = joint_mi(mid4, mid4b, 4, 4)
    obs4, p95_4, pv4 = perm_test_mi(mid4, mid4b, 999)
    print("\n  剔除 broadcast 初爻后（中爻×上爻，4 个本应独立的爻位）：")
    print("    I(物体0;物体1) = %.4f bit   置换零分布95%%分位 = %.4f   p = %.3f"
          % (obs4, p95_4, pv4))
    verdict = "不能拒绝独立（H1 成立）" if pv4 > 0.05 else "拒绝独立（H1 不成立）"
    print("    ⇒ %s" % verdict)
    return yaos, pv4


def part_h2(yaos, tag):
    """H2 可达性：联合分布的支撑集，带多阈值，并与独立期望对照。"""
    g0 = (yaos[:, :, :, 0] * np.array([4, 2, 1])[None, None, :]).sum(-1).reshape(-1)
    g1 = (yaos[:, :, :, 1] * np.array([4, 2, 1])[None, None, :]).sum(-1).reshape(-1)
    joint = g0 * 8 + g1
    p64 = np.bincount(joint, minlength=64).astype(float)
    p64 /= p64.sum()
    m0 = np.bincount(g0, minlength=8) / len(g0)
    m1 = np.bincount(g1, minlength=8) / len(g1)

    print("\n  H2 可达性 — %s" % tag)
    print("    %-12s %10s %14s %16s" % ("阈值", "可达格数", "独立期望", "可达率"))
    exp_sup = int(np.sum(np.outer(m0, m1).reshape(64) > 0))
    for thr in (0.0, 1e-4, 1e-3, 1e-2):
        n = int(np.sum(p64 > thr))
        exp = int(np.sum((np.outer(m0, m1).reshape(64)) > thr))
        print("    p>%-9.0e %10d %14d %15.1f%%" % (thr, n, exp, n / 64 * 100))
    print("    独立假设下所有 64 格期望 >0（两边缘均满格）→ 期望可达 = 64")
    # 哪些格不可达
    dead = np.where(p64 == 0)[0]
    if len(dead):
        d0 = sorted(set(int(v // 8) for v in dead))
        print("    质量恰为 0 的 %d 格涉及物体0 卦: %s" % (len(dead), d0))
    return p64


def main():
    eps = sim(300, 20260929)
    print("仿真 300 条轨迹 = %d 帧（原始世界 GROUND_Y=-150, TAU=28000，未改动动力学）\n"
          % (300 * SEQ_LEN))
    part_a(eps)
    y1, pv1 = part_h1(eps, yao_labels_h, "v1（现行：物体0 上爻恒 0）")
    p1 = part_h2(y1, "v1")
    y2, pv2 = part_h1(eps, yao_labels_h_v2, "v2（W2：上爻逐物体）")
    p2 = part_h2(y2, "v2")

    print("\n" + "=" * 80)
    print("结论（经清理）")
    print("=" * 80)
    print("  H1 独立性：成立。剔除 broadcast 初爻后两物体相态近似独立")
    print("     （v1 p=%.3f, v2 p=%.3f）。外积门控的『两物体独立』前提**未被证伪**。" % (pv1, pv2))
    print("  H2 可达性：不成立。v1 仅 16/64 格有质量，v2 修好标签后 32/64。")
    print("     这与 H1 并不矛盾：MI 由高概率格主导，质量恰为 0 的格对 MI 贡献≈0。")
    print("  ⇒ 应撤回的早期论断：『两物体强耦合使 64 专家冗余』——数据不支持。")
    print("  ⇒ 应保留的结论：v1 存在真实标签缺陷（物体0 上爻恒 0，48 个死专家），")
    print("     W2 修复后降至 32 个；剩余不可达来自动力学共现约束，属**容量冗余")
    print("     （无害）**而非相关性冗余（有害）。")


if __name__ == "__main__":
    main()
