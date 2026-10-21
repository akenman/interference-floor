# -*- coding: utf-8 -*-
"""零成本诊断（方向 b）：kmeans_s 的 8 个簇为何贴合 delta？yao3 为何不贴合？

10.19.10 已确立 E3：GAIN 完全由**分组粒度**决定（kmeans_s 8组→2组，GAIN
31.06%→3.34%，塌回 yao3 水平）。但留下一个未解的矛盾：

    kmeans_s 与 yao3 都是「任意 8 组分段」，为何 GAIN 差一个数量级
    （31.06% vs 3.30%）？

若答案是「kmeans_s 恰好切出了贴合 delta 结构的区域」，则需要给出这个
「贴合」的一般性刻画。本脚本检验三条候选机制：

M1「轴对齐」：kmeans_s 的簇边界是否沿 delta 主方向/少数几个连续变量对齐？
    —— 若簇基本由少数变量的阈值决定，则 kmeans 退化为「沿特定轴分段」，
       而 yao3 是多轴非线性组合，边界不落在任何单一轴上。
M2「单调可分」：每个簇是否是 delta 主方向的**连续区间**（即沿主方向排序后
    簇标签分段连续）？非单调 = 簇形状破碎。
M3「边界锐度」：簇边界附近的 delta 主方向是否发生**跃变**（分段回归斜率）？
    边界锐利 => 该切分确实对准了 delta 的结构断点。

对照：yao3、random、kmeans_s（同 8 组、同样帧数）。
"""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from layout_h import OBJ_DIMS, ACTION_DIM

K = 8
N_EP = 400


def kmeans_fit(Xa, k, seed=0, iters=40):
    rng = np.random.default_rng(seed)
    mu, sd_ = Xa.mean(0), np.maximum(Xa.std(0), 1e-9)
    Xs = (Xa - mu) / sd_
    c = Xs[rng.choice(len(Xs), k, replace=False)]
    lab = np.zeros(len(Xs), dtype=np.int64)
    for _ in range(iters):
        d_ = ((Xs[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        new = d_.argmin(1)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = Xs[m].mean(0)
    return (mu, sd_), c


def kmeans_assign(param, c, Xa):
    mu, sd_ = param
    Xs = (Xa - mu) / sd_
    return ((Xs[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)


def cluster_axis_purity(Xs, g, k):
    """M1：每个簇能否用**单一坐标**的阈值近似切出？
    返回每个簇「用最佳单轴阈值切分」的纯度（1=完全单轴可分）。"""
    pur = []
    for j in range(k):
        m = g == j
        if m.sum() == 0:
            pur.append(0.0)
            continue
        best = 0.0
        for d in range(Xs.shape[1]):
            v = Xs[:, d]
            order = np.argsort(v)
            # 找最佳阈值：使「簇 j 在阈值一侧占比」最大
            cum = np.cumsum(m[order]) / m.sum()
            best = max(best, cum.max(), 1 - cum.min())
        pur.append(best)
    return float(np.mean(pur))


def monotonicity(proj, g, k):
    """M2：沿 delta 主方向排序后，簇标签是否分段连续（1=完全单调分段）。
    衡量：相邻排序位置标签跳变次数的归一化（越少越单调）。"""
    order = np.argsort(proj)
    gs = g[order]
    changes = (gs[1:] != gs[:-1]).sum()
    # 理想分段连续：k-1 次跳变；完全随机：期望 (1-1/k)*n
    ideal = k - 1
    expected = (1 - 1.0 / k) * (len(gs) - 1)
    return ideal, changes, expected


def boundary_sharpness(proj, g, k):
    """M3：簇边界处 delta 主方向是否跃变。
    对每个边界（排序后标签变化处），比较跨界相邻样本的 proj 差 vs 全局平均差。
    比率 >1 => 边界锐利。"""
    order = np.argsort(proj)
    ps, gs = proj[order], g[order]
    bnd = np.where(gs[1:] != gs[:-1])[0]
    gaps = np.abs(np.diff(ps))
    mean_gap = gaps.mean()
    sharp = gaps[bnd].mean() / (mean_gap + 1e-12)
    return len(bnd), sharp


def main():
    d = np.load("trajectories_hinge.npz", allow_pickle=True)
    S0 = d["states"][:N_EP, :-1]
    S1 = d["states"][:N_EP, 1:]
    A = d["actions"][:N_EP, :-1]
    yao = d["yao"][:N_EP, :-1]

    D_full = (S1 - S0).reshape(-1, OBJ_DIMS)
    S0f = S0.reshape(-1, OBJ_DIMS)
    Af = A.reshape(-1, ACTION_DIM)
    X_full = np.concatenate([S0f, Af], axis=1)
    n_ep, n_fr = S0.shape[0], S0.shape[1]
    live = S0f.std(0) >= 1e-9
    D = D_full[:, live]
    X = np.concatenate([S0f[:, live], Af], axis=1)
    Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)

    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(n_ep)
    tr_ep = np.sort(perm[:n_ep // 2])
    tr_m = np.zeros(len(D), dtype=bool)
    tr_m.reshape(n_ep, n_fr)[:] = np.isin(np.arange(n_ep), tr_ep)[:, None]
    ev_m = ~tr_m
    print("数据 %d 帧；eval %d 帧；delta 有效维 %d\n"
          % (len(D), ev_m.sum(), D.shape[1]))

    # delta 主方向（train 拟合）
    Dc = D[tr_m] - D[tr_m].mean(0)
    _, sv, Vt = np.linalg.svd(Dc, full_matrices=False)
    e1 = Vt[0]
    proj = D @ e1
    var1 = (Dc @ e1).var() / Dc.var(0).sum()
    print("delta 主方向解释方差 = %.2f%%；前 3 主成分累计 = %.2f%%\n"
          % (100 * var1, 100 * ((Dc @ Vt[:3].T).var(0).sum() / Dc.var(0).sum())))

    y_obj1 = yao[:, :, :, 1]
    gy = (y_obj1[:, :, 0] * 4 + y_obj1[:, :, 1] * 2
          + y_obj1[:, :, 2]).reshape(-1)

    rs = np.random.default_rng(11)
    cands = {}
    cands["random(纯随机)"] = rs.integers(0, K, len(D))
    cands["yao3(现行基线)"] = gy
    p_x, c_x = kmeans_fit(X_full[tr_m], K, seed=1)
    cands["kmeans_s(状态)"] = kmeans_assign(p_x, c_x, X_full)
    R = rs.normal(size=(X_full.shape[1], X_full.shape[1]))
    p_r, c_r = kmeans_fit((X_full[tr_m] @ R), K, seed=4)
    cands["randproj(随机投影)"] = kmeans_assign(p_r, c_r, X_full @ R)

    ev_idx = np.where(ev_m)[0]

    print("=" * 92)
    print("M1 轴对齐：每簇能否用单一坐标阈值切出（纯度 1=完全单轴可分）")
    print("=" * 92)
    print("%-22s %10s %10s" % ("标签", "单轴纯度", "组数"))
    m1 = {}
    for name, g in cands.items():
        gev = np.asarray(g)[ev_m].astype(np.int64)
        pur = cluster_axis_purity(Xs[ev_m], gev, K)
        m1[name] = pur
        print("%-22s %9.3f %10d" % (name, pur, len(np.unique(gev))))

    print()
    print("=" * 92)
    print("M2 单调可分：沿 delta 主方向排序后，簇跳变次数（理想=k-1，随机期望见下）")
    print("=" * 92)
    print("%-22s %8s %10s %14s %10s" % ("标签", "跳变数", "理想(k-1)", "随机期望", "单调比"))
    m2 = {}
    for name, g in cands.items():
        gev = np.asarray(g)[ev_m].astype(np.int64)
        ideal, changes, expected = monotonicity(proj[ev_m], gev, K)
        ratio = expected / max(changes, 1)   # >1 表示比随机更单调
        m2[name] = ratio
        print("%-22s %8d %10d %14.0f %9.2fx" %
              (name, changes, ideal, expected, ratio))

    print()
    print("=" * 92)
    print("M3 边界锐度：簇边界处 delta 主方向的跃变倍率（>1=锐利）")
    print("=" * 92)
    print("%-22s %8s %12s" % ("标签", "边界数", "锐度倍率"))
    m3 = {}
    for name, g in cands.items():
        gev = np.asarray(g)[ev_m].astype(np.int64)
        nb, sharp = boundary_sharpness(proj[ev_m], gev, K)
        m3[name] = sharp
        print("%-22s %8d %11.2fx" % (name, nb, sharp))

    print()
    print("=" * 92)
    print("联合判读：哪个机制能解释 kmeans_s >> yao3？")
    print("=" * 92)
    print("%-22s %12s %12s %12s" % ("标签", "M1单轴纯度", "M2单调比", "M3锐度"))
    for name in cands:
        print("%-22s %11.3f %11.2fx %11.2fx" % (name, m1[name], m2[name], m3[name]))
    print()
    k, y = m1["kmeans_s(状态)"], m1["yao3(现行基线)"]
    print("* M1 判据：若 kmeans_s 单轴纯度 >> yao3，则 kmeans 退化为「沿 delta 相关轴分段」")
    print("  kmeans_s=%.3f  yao3=%.3f  =>  kmeans_s %s yao3"
          % (k, y, "更" if k > y else "不更"))
    print("* M2 判据：若 kmeans_s 单调比 >> yao3，则其簇是 delta 主方向的连续区间")
    print("  kmeans_s=%.2fx  yao3=%.2fx" % (m2["kmeans_s(状态)"], m2["yao3(现行基线)"]))
    print("* M3 判据：若 kmeans_s 锐度 >> yao3，则其边界正对准 delta 结构断点")
    print("  kmeans_s=%.2fx  yao3=%.2fx" % (m3["kmeans_s(状态)"], m3["yao3(现行基线)"]))


if __name__ == "__main__":
    main()
