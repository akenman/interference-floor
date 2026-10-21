# -*- coding: utf-8 -*-
"""W1：采样均衡数据集（不改动爻标签、不重跑物理）。

目的：检验 T10 机制假说「gua-MoE 的劣势源于相态分布偏斜」中的**频率**成分。
做法：对现有铰链轨迹做**轨迹级重要度重采样**，把每物体的 8 卦占用拉平到
可达空间上的均匀分布。不重跑物理 => 无新物理风险；不改标签 => 与既有
E1a/E0 结果语义可比。

可达性约束（由 tmp_measure_occupancy.py 实测）：
  物体0 的上爻在 layout_h.yao_labels_h 中被结构性置 0（只写 object 1），
  故其 8 卦里只有 4 个（索引 0,2,4,6）可达 —— 均匀目标定为 4 卦各 25%。
  物体1 的 8 卦全部可达 —— 均匀目标定为各 12.5%。

机制：指数梯度（exponentiated gradient）在单纯形上最小化加权占用与目标的
加权 KL 偏差，稀有卦权重更高（对数势），再做系统重采样定容。

train 与 heldout 同时均衡 —— 只均衡其一会把「训练分布偏移」与「相态覆盖」
混在一起，无法归因。
"""
import argparse

import numpy as np

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

GUA_NAME = ["坤", "乾", "震", "巽", "坎", "艮", "离", "兑"]


def gua_counts(yao, obj, T):
    """yao: (N,T,3,2) → (N,8) 每轨迹的 8 卦帧数。"""
    idx = (yao[:, :, :, obj] * np.array([4, 2, 1])[None, None, :]).sum(axis=2)
    flat = idx.reshape(-1, T)
    out = np.zeros((flat.shape[0], 8), dtype=np.float64)
    rows = np.repeat(np.arange(flat.shape[0]), T)
    np.add.at(out, (rows, flat.reshape(-1)), 1.0)
    return out


def target_dist(reachable, n_gua=8):
    t = np.zeros(n_gua)
    t[list(reachable)] = 1.0 / len(reachable)
    return t


def reweight(C, target, lam=1.0, iters=200, lr=0.5, eps=1e-12):
    """指数梯度 + 多样性正则。

    exponent = lr * [ 匹配项 - lam * log(w_i * N) ]
    匹配项推动加权占用靠近 target；lam*log(w_i*N) 是 KL(w||uniform) 的梯度，
    防止权重坍缩到少数轨迹（2026-09-29 实证：lam=0 时 3000 条只剩 60 条唯一）。
    返回 w (N,)。
    """
    N = C.shape[0]
    Cbar = C / np.maximum(C.sum(axis=1, keepdims=True), eps)
    logpot = -np.log(np.maximum(target, 1e-3))
    w = np.full(N, 1.0 / N)
    for _ in range(iters):
        agg = np.maximum(w @ Cbar, eps)
        gap = np.log(np.maximum(target, eps)) - np.log(agg)
        score = Cbar @ (logpot * gap)                 # 匹配项
        expo = lr * (score - lam * np.log(np.maximum(w * N, eps)))
        w = w * np.exp(expo - expo.max())
        w = w / w.sum()
    return w


def ess(w):
    """有效样本量 Kish ESS = 1/sum(w^2)。"""
    return float(1.0 / np.sum(w ** 2))


def systematic_resample(w, N_out, rng):
    """系统重采样（低方差），返回 N_out 个索引。"""
    pos = (rng.random() + np.arange(N_out)) / N_out
    return np.searchsorted(np.cumsum(w), pos, side="right").clip(0, len(w) - 1)


def report(tag, C, w=None, idx=None):
    if idx is not None:
        C = C[idx]
    tot = C.sum(axis=0)
    p = tot / tot.sum()
    nz = int((tot > 0).sum())
    ent = -(p[p > 0] * np.log(p[p > 0])).sum()
    print("\n  %s  非零卦 %d/8  熵 %.3f (均匀 %.3f = %.0f%%)  最大 %.1f%%" %
          (tag, nz, ent, np.log(8), ent / np.log(8) * 100, p.max() * 100))
    for g in range(8):
        bar = "#" * int(p[g] * 50)
        print("    %d %s  %6.2f%%  %-8s %s" % (g, GUA_NAME[g], p[g] * 100,
                                                "%d" % tot[g], bar))
    return p


def process(src, dst, seed, reach0, reach1):
    d = np.load(src, allow_pickle=True)
    yao = d["yao"]
    N, T = yao.shape[0], yao.shape[1]
    C0 = gua_counts(yao, 0, T)
    C1 = gua_counts(yao, 1, T)
    t0, t1 = target_dist(reach0), target_dist(reach1)

    rng = np.random.default_rng(seed)
    w0 = reweight(C0, t0)
    w1 = reweight(C1, t1)
    w = 0.5 * (w0 + w1)
    w = w / w.sum()

    print("\n" + "=" * 70)
    print("处理 %s  (N=%d, T=%d)" % (src, N, T))
    print("=" * 70)
    report("原始 物体0", C0)
    report("原始 物体1", C1)

    idx = systematic_resample(w, N, rng)
    print("\n  重采样索引：唯一 %d/%d（重复率 %.1f%%）" %
          (len(np.unique(idx)), N, (1 - len(np.unique(idx)) / N) * 100))
    report("均衡后 物体0", C0, idx=idx)
    report("均衡后 物体1", C1, idx=idx)

    out = {k: d[k][idx] for k in d.keys()}
    np.savez_compressed(dst, **out)
    print("\n  已写出: %s" % dst)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-train", default="trajectories_hinge_uniform.npz")
    ap.add_argument("--out-heldout",
                    default="experiments/trajectories_hinge_heldout_uniform.npz")
    ap.add_argument("--seed", type=int, default=20260929)
    a = ap.parse_args()

    # 物体0 可达：上爻恒 0 → idx = yao0*4 + yao1*2，{0,2,4,6}
    reach0 = [0, 2, 4, 6]
    reach1 = list(range(8))

    process("trajectories_hinge.npz", a.out_train, a.seed, reach0, reach1)
    process("experiments/trajectories_hinge_heldout.npz", a.out_heldout,
            a.seed + 1, reach0, reach1)


if __name__ == "__main__":
    main()
