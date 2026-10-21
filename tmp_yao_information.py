# -*- coding: utf-8 -*-
"""三爻语义为何无效：不用训练，直接在现有数据上做信息论测量。

E1a + Stage D 已排除「门控容量」和「标签死格」两个假设。本脚本检验第三个、
也是唯一还没被检验的可能：**三爻标签本身是否携带对 next-state 有用的信息**。

三个递进的测量，全部在 trajectories_hinge.npz 上完成：
  M1 可预测性  I(yao ; s_t)      —— 门控信号本身是否可从状态读出？
                                  若低，门控就是噪声源。
  M2 目标可分解性  I(yao ; s_{t+1}-s_t) —— 爻能否解释下一帧的**变化**？
                                  这是路由唯一能提供增益的通道。
  M3 路由收益上界  用 yao 分组预测 delta，比较组内方差/总方差。
                                  直接给出「按卦路由」的理论最大收益。

关键：M2/M3 用**连续** target 求互信息时做高斯近似
I = 0.5*log2(Var_total / Var_within)，这是可加模型下的精确值
（对角协方差假设），比分箱互信息更稳。
"""
import io
import math
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from layout_h import yao_labels_h, yao_labels_h_v2, OBJ_DIMS, DT as DT_H, F, obj_base

GUA = ["坤", "乾", "震", "巽", "坎", "艮", "离", "兑"]
AXIS = ["初爻(接触)", "中爻(动能趋势)", "上爻(趋近靶)"]


def load(path="trajectories_hinge.npz", n_ep=800):
    d = np.load(path, allow_pickle=True)
    return d["states"][:n_ep], d["yao"][:n_ep], d["actions"][:n_ep], d["targets"][:n_ep]


def gua_index(yao, obj, axes):
    idx = np.zeros(yao.shape[:2], dtype=np.int64)
    for a in axes:
        idx = idx * 2 + yao[:, :, a, obj].astype(np.int64)
    return idx


def m1_predictability(states, yao, actions):
    """I(yao ; s_t)：门控信号可预测性。用高斯近似 + 组均值解释率。"""
    print("=" * 80)
    print("M1  门控信号可预测性  I(yao ; s_t)")
    print("     —— 若门控无法从状态读出，它对模型而言是纯噪声")
    print("=" * 80)
    S = states.reshape(-1, OBJ_DIMS)
    A = actions.reshape(-1, 1)
    print("  %-16s %14s %14s %12s" % ("爻组合", "组数", "解释率 R²", "MI(bit,近似)"))
    for axes, name in [((0,), "初爻(1轴)"), ((1,), "中爻(1轴)"),
                       ((2,), "上爻(1轴)"), ((0, 1, 2), "三爻(8卦)")]:
        for obj in (0, 1):
            g = gua_index(yao, obj, axes).reshape(-1)
            ng = int(g.max()) + 1
            tot = ((S - S.mean(0)) ** 2).sum()
            within = 0.0
            for k in range(ng):
                m = g == k
                if m.sum() < 2:
                    within += ((S[m] - S.mean(0)) ** 2).sum()
                    continue
                within += ((S[m] - S[m].mean(0)) ** 2).sum()
            r2 = 1.0 - within / tot
            # 高斯近似 MI = 0.5 * n_dims * log2(1/(1-R2))
            mi = 0.5 * OBJ_DIMS * math.log2(max(1.0 / (1 - r2), 1.0))
            print("  %-16s %14d %13.2f%% %12.2f" %
                  (name + (" 物%d" % obj), ng, r2 * 100, mi))


def m2_target_information(states, yao, actions):
    """I(yao ; delta)：爻能否解释下一帧变化。路由增益的唯一可能通道。"""
    print("\n" + "=" * 80)
    print("M2  目标可分解性  I(yao ; s_{t+1}-s_t)")
    print("     —— 爻若不能解释变化，按卦路由就不可能带来增益")
    print("=" * 80)
    S = states[:, :-1]
    S1 = states[:, 1:]
    D = (S1 - S).reshape(-1, OBJ_DIMS)
    print("  delta 维度 = %d (22 维原始单位)" % OBJ_DIMS)
    print("  %-22s %12s %16s" % ("爻组合", "解释率 R²", "MI(bit,近似)"))
    res = {}
    for axes, name in [((0,), "初爻"), ((1,), "中爻"), ((2,), "上爻"),
                       ((0, 1), "初+中(4卦)"), ((0, 2), "初+上(4卦)"),
                       ((1, 2), "中+上(4卦)"), ((0, 1, 2), "三爻(8卦)")]:
        for obj in (0, 1):
            g = gua_index(yao, obj, axes)[:, :-1].reshape(-1)
            ng = int(g.max()) + 1
            mu = D.mean(0)
            tot = ((D - mu) ** 2).sum()
            within = 0.0
            for k in range(ng):
                m = g == k
                if m.sum() < 2:
                    within += ((D[m] - mu) ** 2).sum()
                    continue
                within += ((D[m] - D[m].mean(0)) ** 2).sum()
            r2 = 1.0 - within / tot
            mi = 0.5 * OBJ_DIMS * math.log2(max(1.0 / (1 - r2), 1.0))
            res[(name, obj)] = r2
            print("  %-22s %11.2f%% %16.2f" % (name + (" 物%d" % obj), r2 * 100, mi))
    return res


def m3_routing_bound(states, yao, actions):
    """路由收益上界：按卦分组的残差方差比。含「用连续状态直接回归」的对照。"""
    print("\n" + "=" * 80)
    print("M3  路由收益上界（直接可解释为「门控最多能省多少残差」）")
    print("=" * 80)
    S = states[:, :-1]
    D = (states[:, 1:] - S).reshape(-1, OBJ_DIMS)
    A = actions[:, :-1].reshape(-1, 1)
    X = np.concatenate([S.reshape(-1, OBJ_DIMS), A], axis=1)
    mu = D.mean(0)
    tot = ((D - mu) ** 2).sum()
    # 基线：全局常数预测
    base = ((D - mu) ** 2).sum()
    # 对照 A：全局线性回归（无门控）
    Xd = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(Xd, D, rcond=None)
    lin = ((D - Xd @ coef) ** 2).sum()
    # 对照 B：按卦分组常数（纯门控，无其他信息）
    print("  %-22s %14s %14s" % ("分组", "残差占比", "vs 全局线性"))
    print("  %-22s %13.1f%% %13.1f%%" % ("全局常数", base / tot * 100, 100.0))
    print("  %-22s %13.1f%% %13.1f%%" % ("全局线性(无门控)", lin / tot * 100, lin / base * 100))
    for axes, name in [((0, 1, 2), "三爻(8卦)")]:
        for obj in (0, 1):
            g = gua_index(yao, obj, axes)[:, :-1].reshape(-1)
            within = 0.0
            for k in range(int(g.max()) + 1):
                m = g == k
                if m.sum() < 2:
                    within += ((D[m] - mu) ** 2).sum()
                    continue
                within += ((D[m] - D[m].mean(0)) ** 2).sum()
            print("  %-22s %13.1f%% %13.1f%%"
                  % ("按%s物%d分组" % (name, obj), within / tot * 100, within / base * 100))
    print("\n  读法：「按卦分组」相对「全局常数」的残差下降 = 卦标签单独能解释的方差。")
    print("        该值接近 0 ⇒ 卦语义对预测无信息 ⇒ 门控无增益是**预期**的，")
    print("        而非实现缺陷。这将把 T10 的负结果重新归因。")


def main():
    st, yao, ac, tg = load()
    print("数据：trajectories_hinge.npz  截取 %d 条轨迹 = %d 帧\n"
          % (st.shape[0], st.shape[0] * st.shape[1]))
    m1_predictability(st, yao, ac)
    m2_target_information(st, yao, ac)
    m3_routing_bound(st, yao, ac)


if __name__ == "__main__":
    main()
