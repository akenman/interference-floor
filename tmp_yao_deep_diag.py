# -*- coding: utf-8 -*-
"""零成本诊断（方向 1 + 2）：为何 yao3 的路由增益尤其差？

10.19.8 已否证 E1（信息带宽低）与 E2（切分方向错配）。本脚本检验剩余两个
可零成本验证的机制：

方向 1「组不均衡」：yao3 标签熵仅 2.19，其余候选 ~3.00。少数死组可能
    使「每组独立线性」退化（死组样本少 → 回退全局 → 等价于无门控）。
    检验：把 yao3 在 eval 上重加权到熵 ~3.00（或用 train 分位重新分箱），
    看 GAIN 是否上升。若上升 => 不均衡是原因之一。

方向 2「缺交互项」：yao3 是 3 爻位**独立判定**的粗粒度编码，而 delta 主方向
    是连续量。门控若为加性/线性，3 个二值指示变量与连续主方向之间只有
    「主效应」，没有「爻位×爻位」或「爻位×状态」的交互。
    检验：(a) 给路由回归加 爻位 one-hot 交互项；(b) 给路由加 爻位×状态 交互；
    看 GAIN 是否上升。若上升 => 缺交互是原因。

同时检验 E3 的推论：若「分段粒度」是唯一开关，则把 yao3 的 8 组**重采样到
    均衡**，或把高 GAIN 候选**退化为 2 组**，GAIN 应大幅下降/上升。
"""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from layout_h import OBJ_DIMS, ACTION_DIM

K = 8


def r2(Dv, pred):
    mu = Dv.mean(0)
    return 1.0 - ((Dv - pred) ** 2).sum() / ((Dv - mu) ** 2).sum()


def route_gain(Xtr, Dtr, Xev, Dev, gtr, gev, nonlinear=False):
    """全局线性 vs 全局线性+每组模型 的 R² 增量。

    [WARN] 早期版本用「组 one-hot × 状态」交互项，**是错的**：组内 one-hot 是常量，
    其与 X 的乘积在组内线性可表示（列共线），不增加任何容量——实测加性与交互
    给出逐位相同的结果。真正的门控增强应是**组内非线性**（每组独立的 MLP/二次项），
    故 nonlinear=True 时为每组加二次特征。
    """
    Xd_tr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
    Xd_ev = np.concatenate([Xev, np.ones((len(Xev), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(Xd_tr, Dtr, rcond=None)
    r2_lin = r2(Dev, Xd_ev @ coef)

    def quad(Xd):
        # 平方项（去掉截距列与自身平方中的截距重复）
        return (Xd[:, :-1] ** 2)

    pred = np.empty_like(Dev)
    for k in np.unique(gev):
        mtr, mev = gtr == k, gev == k
        if mtr.sum() < Xd_tr.shape[1] + 2 or not mev.any():
            pred[mev] = Xd_ev[mev] @ coef
            continue
        if nonlinear:
            A_tr = np.concatenate([Xd_tr[mtr], quad(Xd_tr[mtr])], axis=1)
            A_ev = np.concatenate([Xd_ev[mev], quad(Xd_ev[mev])], axis=1)
            c, *_ = np.linalg.lstsq(A_tr, Dtr[mtr], rcond=None)
            pred[mev] = A_ev @ c
        else:
            c, *_ = np.linalg.lstsq(Xd_tr[mtr], Dtr[mtr], rcond=None)
            pred[mev] = Xd_ev[mev] @ c
    return r2_lin, r2(Dev, pred)


def entropy(g, n):
    p = np.bincount(g, minlength=n) / len(g)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def rebalance_yao(train_frames_gy, eval_gy, n_ep, n_fr, tr_m, target_ent=3.0):
    """方向 1：按 yao 原始三爻在 train 上取更细的分位，使标签熵接近 target_ent。
    不使用 eval 的 delta，只重切 yao 自身在 train 上的边际分位。"""
    return None  # 占位，实际用 main 中的重分箱


def main():
    d = np.load("trajectories_hinge.npz", allow_pickle=True)
    S0 = d["states"][:400, :-1]
    S1 = d["states"][:400, 1:]
    A = d["actions"][:400, :-1]
    yao = d["yao"][:400, :-1]

    D_full = (S1 - S0).reshape(-1, OBJ_DIMS)
    S0f = S0.reshape(-1, OBJ_DIMS)
    Af = A.reshape(-1, ACTION_DIM)
    X_full = np.concatenate([S0f, Af], axis=1)
    n_ep, n_fr = S0.shape[0], S0.shape[1]
    live = S0f.std(0) >= 1e-9
    D = D_full[:, live]
    X = np.concatenate([S0f[:, live], Af], axis=1)

    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(n_ep)
    tr_ep, ev_ep = np.sort(perm[:n_ep // 2]), np.sort(perm[n_ep // 2:])
    tr_m = np.zeros(len(D), dtype=bool)
    tr_m.reshape(n_ep, n_fr)[:] = np.isin(np.arange(n_ep), tr_ep)[:, None]
    ev_m = ~tr_m
    print("数据 %d 帧；train %d / eval %d；delta 有效维 %d\n"
          % (len(D), tr_m.sum(), ev_m.sum(), D.shape[1]))

    # yao 三个爻位（物体1）为二值指示变量
    y_obj1 = yao[:, :, :, 1]
    bits = [y_obj1[:, :, i].reshape(-1).astype(np.int64) for i in range(3)]  # 3×(N,)
    gy = bits[0] * 4 + bits[1] * 2 + bits[2]

    def show(tag, g, nonlinear=False):
        g = np.asarray(g).astype(np.int64)
        gev, gtr = g[ev_m], g[tr_m]
        if len(np.unique(gev)) < 2:
            print("%-34s 组数<2, 跳过" % tag)
            return
        r2l, r2r = route_gain(X[tr_m], D[tr_m], X[ev_m], D[ev_m],
                              gtr, gev, nonlinear=nonlinear)
        ent = entropy(gev, K)
        print("%-34s 组数=%d 熵=%.2f  R2_lin=%.2f%%  GAIN=%+.2f%%"
              % (tag, len(np.unique(gev)), ent, r2l * 100, (r2r - r2l) * 100))

    print("=" * 78)
    print("方向 1：组不均衡是否是 yao3 GAIN 低的原因？")
    print("=" * 78)
    show("yao3 原始", gy)
    # 均衡化：对每个爻位用 train 的 37.5/62.5 分位切成三档，组合出最多 27 组，
    # 再用 train 分位把 yao 组合值压回 8 组，目标提高熵。
    # 注：直接对偏斜爻位取中位会把它压成常量（熵反而更低），故用三档分位。
    score_yao = gy.astype(np.float64)
    qs = np.percentile(score_yao[tr_m], np.linspace(0, 100, K + 1)[1:-1])
    gy_bal = np.searchsorted(qs, score_yao).astype(np.int64)   # 均衡 8 组
    show("yao3 均衡化(train分位重切)", gy_bal)
    # 参照：一个天然均衡的高 GAIN 标签
    def kmeans_fit(Xa, k, seed=0, iters=40):
        rng = np.random.default_rng(seed)
        mu, sd_ = Xa.mean(0), np.maximum(Xa.std(0), 1e-9)
        Xs = (Xa - mu) / sd_
        c = Xs[rng.choice(len(Xs), k, replace=False)]
        lab = np.zeros(len(Xs), dtype=np.int64)
        for _ in range(iters):
            d_ = ((Xs[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            if np.array_equal(d_, lab) and _ > 0:
                break
            lab = d_
            for j in range(k):
                mm = lab == j
                if mm.any():
                    c[j] = Xs[mm].mean(0)
        return (mu, sd_), c

    def kmeans_assign(param, c, Xa):
        mu, sd_ = param
        Xs = (Xa - mu) / sd_
        return ((Xs[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)

    p_x, c_x = kmeans_fit(X_full[tr_m], K, seed=1)
    gks = kmeans_assign(p_x, c_x, X_full)
    show("kmeans_s(参照)", gks)

    print()
    print("=" * 78)
    print("方向 2：组内非线性不足是否是 yao3 GAIN 低的原因？")
    print("=" * 78)
    show("yao3 原始(加性路由)", gy, nonlinear=False)
    show("yao3 + 组内二次项", gy, nonlinear=True)
    show("yao3 均衡 + 组内二次项", gy_bal, nonlinear=True)
    show("kmeans_s + 组内二次项(参照)", gks, nonlinear=True)

    print()
    print("=" * 78)
    print("E3 推论检验：把高 GAIN 候选退化，看 GAIN 是否塌回 yao3 水平")
    print("=" * 78)
    # kmeans_s 退化为 2 组（粗分段）
    g2 = (gks >= 4).astype(np.int64)   # 合并成 2 组
    show("kmeans_s 合并为 2 组(粗)", g2)
    # yao3 拆得更细不可能(已 3 爻)，改看 phys 式：把 yao3 只用首爻(2 组)
    g_yao1 = bits[0]
    show("yao3 只用首爻(2 组)", g_yao1)


if __name__ == "__main__":
    main()
