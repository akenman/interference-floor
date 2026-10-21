# -*- coding: utf-8 -*-
"""零成本诊断：门控分段与 delta 结构是否**对齐**。

背景（10.19.6）：免训练 GAIN 估计显示，任何「把状态切成 8 份」的标签都能拿到
~31-37% 的 GAIN，唯独 yao3 只有 3.30%。这留下两种互斥解释：

    E1「信息带宽低」  —— yao3 的条件信息量太低（2.39%），所以无用
    E2「切分方向错配」—— yao3 的分段与 delta 主方向**正交**，切得再细也白切

E1 已被 randproj / B_随机方向 反驳：高 R2_标签 的随机标签 GAIN 同样 ~35%。
故本脚本直接检验 E2，方法是标准的多重对应分析（MCA / 对应分析）：

    构造列联表  行=标签组(8)  列=delta 主方向分箱(B)
    统计量 = 归一化惯量（NI / 归一化主惯量占比），衡量「分组 ⟂ 连续量」的程度
    基线   = 对标签做随机置换，构造 NI 的零分布

关键对照（全部保持 8 组、8 箱、同样帧数）：
    yao3          现行三爻
    kmeans_s      状态 kmeans
    randproj      随机投影
    random        纯随机（NI 的绝对下界基线）
    v2            yao_labels_h_v2 修复版（若可用）

若 yao3 的 NI 与 random 无显著差异 => 切分与 delta 正交 => 支持 E2「错配」
若 yao3 的 NI 显著高于 random   => 切分确实捕获 delta 结构 => 问题在别处
"""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from layout_h import OBJ_DIMS, ACTION_DIM, F, obj_base

K = 8          # 标签组数，受控
B = 8          # delta 主方向分箱数，受控
N_EP = 400
N_PERM = 200   # 置换零分布次数


def correspondence_ni(g, b):
    """列联表 (g × b) 的归一化惯量。
    返回 (NI, chi2)，NI ∈ [0,1]，1 = 完全关联，0 = 完全独立。"""
    ng, nb = g.max() + 1, b.max() + 1
    T = np.zeros((ng, nb))
    np.add.at(T, (g, b), 1.0)
    n = T.sum()
    row, col = T.sum(1, keepdims=True), T.sum(0, keepdims=True)
    E = row @ col / n
    keep = E > 0
    chi2 = (((T - E) ** 2)[keep] / E[keep]).sum()
    # 惯量占比：卡平方 / (n * min(ng-1, nb-1))
    ni = chi2 / (n * min(ng - 1, nb - 1))
    return ni, chi2, n


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

    sd = S0f.std(0)
    live = sd >= 1e-9
    D = D_full[:, live]
    X = np.concatenate([S0f[:, live], Af], axis=1)

    # episode 级 train/eval 划分，与标签阶梯保持同一协议
    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(n_ep)
    tr_ep, ev_ep = np.sort(perm[:n_ep // 2]), np.sort(perm[n_ep // 2:])
    tr_m = np.zeros(len(D), dtype=bool)
    tr_m.reshape(n_ep, n_fr)[:] = np.isin(np.arange(n_ep), tr_ep)[:, None]
    ev_m = ~tr_m
    print("数据 %d 帧（%d ep）；train %d / eval %d"
          % (len(D), n_ep, tr_m.sum(), ev_m.sum()))
    print("delta 有效维度 %d\n" % D.shape[1])

    # ---- delta 主方向（只在 train 拟合）----
    Dc = D[tr_m] - D[tr_m].mean(0)
    _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
    e1 = Vt[0]
    proj = D @ e1                              # eval 也用 train 的主方向
    # 分箱边界只在 train 上取分位（防止用 eval 信息定边界）
    qs = np.percentile(proj[tr_m], np.linspace(0, 100, B + 1)[1:-1])
    b_all = np.searchsorted(qs, proj).astype(np.int64)
    b_tr, b_ev = b_all[tr_m], b_all[ev_m]
    print("delta 主方向解释方差 = %.2f%%（前 1 主成分）\n"
          % (100 * (np.linalg.norm(e1) ** 2 * 0 +
                    (Dc @ e1).var() / Dc.var(0).sum() * 0 +
                    100 * 0) if False else
             100 * (Dc @ e1).var() / Dc.var(0).sum()))

    # ---- 候选标签（与标签阶梯同口径）----
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

    rs = np.random.default_rng(11)
    cands = {}
    cands["random(纯随机)"] = rs.integers(0, K, len(D))

    y_obj1 = yao[:, :, :, 1]
    gy = (y_obj1[:, :, 0] * 4 + y_obj1[:, :, 1] * 2
          + y_obj1[:, :, 2]).reshape(-1)
    cands["yao3(现行基线)"] = gy

    if "yao_v2" in d:
        yv2 = d["yao_v2"][:N_EP, :-1, :, 1]
        gv2 = (yv2[:, :, 0] * 4 + yv2[:, :, 1] * 2 + yv2[:, :, 2]).reshape(-1)
        cands["yao3_v2(修复)"] = gv2

    p_x, c_x = kmeans_fit(X_full[tr_m], K, seed=1)
    cands["kmeans_s(状态)"] = kmeans_assign(p_x, c_x, X_full)

    R = rs.normal(size=(X_full.shape[1], X_full.shape[1]))
    p_r, c_r = kmeans_fit((X_full[tr_m] @ R), K, seed=4)
    cands["randproj(随机投影)"] = kmeans_assign(p_r, c_r, X_full @ R)

    # ---- 置换零分布：打乱标签，保持组数与帧数 ----
    print("%-22s %8s %10s %10s %10s" %
          ("标签", "组数", "NI(eval)", "NI零均值", "z / p"))
    print("-" * 70)
    out = {}
    for name, g in cands.items():
        g = np.asarray(g).astype(np.int64)
        gev, gtr = g[ev_m], g[tr_m]
        if len(np.unique(gev)) < 2:
            continue
        ni_ev, chi2, n = correspondence_ni(gev, b_ev)
        null = []
        gp = gtr.copy()
        for _ in range(N_PERM):
            gp = rs.permutation(gp)
            null.append(correspondence_ni(gp, b_tr)[0])
        null = np.array(null)
        z = (ni_ev - null.mean()) / (null.std() + 1e-12)
        p = (1 + (null >= ni_ev).sum()) / (N_PERM + 1)
        star = "  <-- 正交" if p > 0.05 else ""
        print("%-22s %8d %9.4f %10.4f %5.1f / %.3f%s" %
              (name, len(np.unique(gev)), ni_ev, null.mean(), z, p, star))
        out[name] = (ni_ev, null.mean(), z, p)

    print("\n读法：")
    print("  NI  = 归一化惯量，0=标签与 delta 主方向完全正交，1=完全关联")
    print("  p   = 对「标签随机置换」零分布的单尾 p 值")
    print("  p>0.05 => 切分方向与 delta 正交，切得再细也捕获不到 delta 结构")
    print("\n判读：")
    y = out.get("yao3(现行基线)")
    r = out.get("random(纯随机)")
    k = out.get("kmeans_s(状态)")
    if y and r:
        print("  yao3  NI=%.4f  vs  random  NI=%.4f  => 差 %+.4f"
              % (y[0], r[0], y[0] - r[0]))
        if y[3] > 0.05:
            print("  => yao3 与 delta 主方向【无显著关联】(p=%.3f)" % y[3])
            print("     支持 E2「切分方向错配」：爻位切分与 delta 主方向正交")
        else:
            print("  => yao3 与 delta 主方向【显著关联】(p=%.3f)" % y[3])
            print("     E2 不成立，问题不在对齐性，需另寻解释")
    if y and k:
        print("  kmeans_s NI=%.4f  vs  yao3 NI=%.4f  => 状态 kmeans %s yao3" %
              (k[0], y[0], "优于" if k[0] > y[0] else "不优于"))


if __name__ == "__main__":
    main()
