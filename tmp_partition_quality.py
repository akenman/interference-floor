# -*- coding: utf-8 -*-
"""紧致度/分离度：kmeans 簇与 yao3 卦作为状态空间划分的质量对比。"""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from layout_h import OBJ_DIMS, ACTION_DIM

K = 8


def kmeans_fit_z(Xa, k, seed=0, iters=40):
    r = np.random.default_rng(seed)
    Xs = (Xa - Xa.mean(0)) / np.maximum(Xa.std(0), 1e-9)
    c = Xs[r.choice(len(Xs), k, replace=False)]
    lab = np.zeros(len(Xs), dtype=np.int64)
    for _ in range(iters):
        d2 = ((Xs[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        nd = d2.argmin(1)
        if np.array_equal(nd, lab) and _ > 0:
            break
        lab = nd
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = Xs[m].mean(0)
    return Xs, c


def assign(Xs, c):
    return ((Xs[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)


def main():
    d = np.load("trajectories_hinge.npz", allow_pickle=True)
    S0 = d["states"][:400, :-1]
    A = d["actions"][:400, :-1]
    yao = d["yao"][:400, :-1]

    S0f = S0.reshape(-1, OBJ_DIMS)
    Af = A.reshape(-1, ACTION_DIM)
    X_full = np.concatenate([S0f, Af], axis=1)
    n_ep, n_fr = S0.shape[0], S0.shape[1]

    y_obj1 = yao[:, :, :, 1]
    gy = (y_obj1[:, :, 0] * 4 + y_obj1[:, :, 1] * 2
          + y_obj1[:, :, 2]).reshape(-1)

    rng = np.random.default_rng(11)
    Xs, c_ks = kmeans_fit_z(X_full, K, seed=1)
    g_ks = assign(Xs, c_ks)

    Xr_raw = X_full @ rng.normal(size=(X_full.shape[1], X_full.shape[1]))
    Xr, c_rp = kmeans_fit_z(Xr_raw, K, seed=4)
    g_rp = assign(Xr, c_rp)

    g_rnd = rng.integers(0, K, len(Xs))
    # yao3 的爻位（用于评估「笛卡尔积」结构）
    bits = [y_obj1[:, :, i].reshape(-1).astype(np.int64) for i in range(3)]

    def metrics(g, name, space=None):
        space = Xs if space is None else space
        sizes = np.bincount(g, minlength=K) / len(g)
        p = sizes[sizes > 0]
        ent = float(-(p * np.log2(p)).sum())
        cent = np.stack([space[g == k].mean(0) if (g == k).any()
                         else np.zeros(space.shape[1]) for k in range(K)])
        d = np.linalg.norm(space - cent[g], axis=1)
        within = float(d.mean())
        cd = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
        between = float(cd[np.triu_indices(K, 1)].mean())
        print("%-18s 熵=%.2f  组内距=%.3f  组间距=%.3f  分离比=%.3f"
              % (name, ent, within, between, between / within))

    print("状态空间划分质量（%d 帧，%d 维标准化状态）" % (len(Xs), Xs.shape[1]))
    print("-" * 74)
    print("%-18s %s" % ("标签", "熵 / 紧致度 / 分离度"))
    metrics(g_rnd, "random(纯随机)")
    metrics(gy, "yao3(现行基线)")
    metrics(g_ks, "kmeans_s(状态)")
    metrics(g_rp, "randproj(随机投影)", space=Xr)

    print()
    print("爻位边际分布 yao3:")
    for i, b in enumerate(bits):
        print("   爻%d: P(1)=%.3f" % (i + 1, b.mean()))

    print()
    print("* 若 random 的 紧致度/分离度 与 kmeans_s 相当 => random 也应拿高 GAIN")
    print("  实测 random GAIN = -0.49% (10.19.7) => 与此矛盾, 说明 GAIN 不只看划分质量")
    print("* yao3 三个爻位若高度相关 => 8 卦非笛卡尔积, 组数少且形状破碎")


if __name__ == "__main__":
    main()
