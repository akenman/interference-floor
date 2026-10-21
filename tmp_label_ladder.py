# -*- coding: utf-8 -*-
"""决定性实验的**筛选阶段**（零 GPU 成本）：为候选门控标签建立「信息量阶梯」。

待验证假设（H）：gua-MoE 无增益，是因为门控标签对预测目标的信息量太低。
若成立，则 路由增益 应随标签信息量 R² 单调上升。

本脚本不训练，只做**免训练**的路由增益估计，且**全部按 episode 划分 train/eval**，
所有候选的聚类中心、线性系数、组均值一律**只在 train 上拟合**，在 eval 上评估：

    R2_const  = delta ~ 全局常数
    R2_lin    = delta ~ 全局线性(s_t, a_t)          ← 主干层已有能力
    R2_route  = delta ~ 全局线性 + 每组独立线性     ← 门控叠加在主干层之上
    GAIN      = R2_route - R2_lin                  ← 门控的**增量**贡献

[WARN] 协议更正（2026-09-30）：旧版本把 kmeans_p 实现为
`kmeans(concat([X, D_full*0.05]))`，即用**真实 delta** 聚类，属前视泄漏，
却被称为「预测 delta」，结论无效。现改为：先在 train 上拟合线性预测器
X -> D_hat，再对 **D_hat**（而非 D）聚类，eval 上评估。真正的 deployable 标签
必须在推理时只凭 (s_t, a_t) 可得。

[WARN] 同时，全表改为 episode 级 train/eval 划分，消除旧版的样本内（in-sample）
乐观偏差。若只修 kmeans_p 而不统一协议，会让 kmeans_p 独自变成样本外、
其余候选仍是样本内，比较不公平。

所有标签固定 8 组（专家数受控），只变信息量，隔离「专家数」与「标签质量」两个因素。

标签阶梯（deployable = 推理时只凭 s_t, a_t 可得）：
    random     随机 8 分组            —— 负对照，应 GAIN≈0
    yao3       现行三爻 8 卦          —— 当前基线
    phys       现有物理量 8 分位      —— 只用当前状态
    kmeans_s   (s_t,a_t) 的 8 类 kmeans —— 无监督当前状态
    kmeans_p   **预测** delta 的 8 类 kmeans —— 门控可实现的理想形态
    oracle     真实 delta 的 8 类 kmeans —— **前视 oracle，泄漏未来，仅作上界**
    randproj   (s_t,a_t) 随机投影后 kmeans —— 负对照：状态可得的任意分组
"""
import io
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from layout_h import OBJ_DIMS, ACTION_DIM, F, obj_base

K = 8          # 固定专家/分组数，受控
N_EP = 400


def kmeans_fit(X, k, seed=0, iters=40):
    """在给定数据上拟合 kmeans，返回 (归一化参数, 质心)。供 train 拟合 / eval 复用。"""
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-9)
    Xs = (X - mu) / sd
    c = Xs[rng.choice(len(Xs), k, replace=False)]
    lab = np.zeros(len(Xs), dtype=np.int64)
    for _ in range(iters):
        d = ((Xs[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = Xs[m].mean(0)
    return (mu, sd), c


def kmeans_assign(param, c, X):
    mu, sd = param
    Xs = (X - mu) / sd
    return ((Xs[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)


def r2(D, pred):
    mu = D.mean(0)
    return 1.0 - ((D - pred) ** 2).sum() / ((D - mu) ** 2).sum()


def eval_r2(Xtr, Dtr, Xev, Dev, gtr, gev):
    """全部拟合在 tr、评估在 ev。返回 (R2_线性, R2_路由, R2_标签)。"""
    Xd_tr = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
    Xd_ev = np.concatenate([Xev, np.ones((len(Xev), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(Xd_tr, Dtr, rcond=None)
    r2_lin = r2(Dev, Xd_ev @ coef)

    # 路由：每组独立线性，组内拟合在 train，eval 用同组系数
    pred = np.empty_like(Dev)
    for k in np.unique(gev):
        mtr, mev = gtr == k, gev == k
        if mtr.sum() < Xd_tr.shape[1] + 2 or not mev.any():
            pred[mev] = Xd_ev[mev] @ coef
            continue
        c, *_ = np.linalg.lstsq(Xd_tr[mtr], Dtr[mtr], rcond=None)
        pred[mev] = Xd_ev[mev] @ c
    r2_route = r2(Dev, pred)

    # 标签：每组目标均值（仅组均值信息，不含 s_t）
    r2_lab = 0.0
    for k in np.unique(gev):
        mtr, mev = gtr == k, gev == k
        if not mev.any():
            continue
        mu_k = Dtr[mtr].mean(0) if mtr.any() else Dtr.mean(0)
        r2_lab += ((Dev[mev] - mu_k) ** 2).sum()
    r2_lab = 1.0 - r2_lab / ((Dev - Dev.mean(0)) ** 2).sum()
    return r2_lin, r2_route, r2_lab


def label_entropy(g, n):
    p = np.bincount(g, minlength=n) / len(g)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


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

    # ---- episode 级 train/eval 划分（避免相邻帧泄漏）----
    n_ep, n_fr = S0.shape[0], S0.shape[1]
    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(n_ep)
    tr_ep, ev_ep = np.sort(perm[:n_ep // 2]), np.sort(perm[n_ep // 2:])
    tr_m = np.zeros(len(D_full), dtype=bool)
    tr_m.reshape(n_ep, n_fr)[:] = np.isin(np.arange(n_ep), tr_ep)[:, None]
    ev_m = ~tr_m
    print("数据 %d 帧（%d ep × %d 帧）；train %d 帧 / eval %d 帧（episode 级划分）"
          % (len(D_full), n_ep, n_fr, tr_m.sum(), ev_m.sum()))

    sd = S0f.std(0)
    live = sd >= 1e-9
    D = D_full[:, live]
    X = np.concatenate([S0f[:, live], Af], axis=1)
    print("目标 delta 有效维度 %d（原 %d 维含 %d 维常量）\n"
          % (D.shape[1], OBJ_DIMS, int((~live).sum())))

    rng = np.random.default_rng(0)
    # yao: (N, T, 3 爻, 2 物体) —— 取物体1 的三爻，baseline 沿用此口径
    y_obj1 = yao[:, :, :, 1]
    g_yao_full = (y_obj1[:, :, 0] * 4 + y_obj1[:, :, 1] * 2
                  + y_obj1[:, :, 2]).reshape(-1)

    b1, b2 = obj_base(0), obj_base(1)
    # 统一为 (N,1) 列向量，避免后续 phys*2+q 误广播成 (N,N)。
    # tau 取每帧动作的绝对值和（标量），与旧版 |a| 语义一致。
    yl = S0f[:, b1 + F["y"]][:, None]
    vl = np.hypot(S0f[:, b1 + F["vx"]], S0f[:, b1 + F["vy"]])[:, None]
    vl2 = np.hypot(S0f[:, b2 + F["vx"]], S0f[:, b2 + F["vy"]])[:, None]
    ek = (S0f[:, b2 + F["vx"]] ** 2 + S0f[:, b2 + F["vy"]] ** 2)[:, None]
    tau = np.abs(Af).sum(1)[:, None]

    # ---- 候选标签：返回 (train 标签, eval 标签) ----
    cands = {}
    g_rand = rng.integers(0, K, len(D))
    cands["random(负对照)"] = (g_rand[tr_m], g_rand[ev_m])

    gy = g_yao_full
    cands["yao3(现行基线)"] = (gy[tr_m], gy[ev_m])

    # 全部列向量 (N,1)，phys 初始化也用 (N,1) 以免 (N,)+(N,1) 广播成 (N,N)
    phys = np.zeros((len(D), 1), dtype=np.int64)
    for v in (yl, vl, vl2, ek, tau):
        q = np.searchsorted(np.percentile(v, [25, 50, 75]), v)
        phys = phys * 2 + q
    phys = phys.ravel()
    phys = np.unique(np.searchsorted(np.unique(phys), phys), return_inverse=True)[1]
    phys = phys % K
    cands["phys(现有物理量)"] = (phys[tr_m], phys[ev_m])

    p_x, c_x = kmeans_fit(X_full[tr_m], K, seed=1)
    cands["kmeans_s(状态)"] = (kmeans_assign(p_x, c_x, X_full[tr_m]),
                              kmeans_assign(p_x, c_x, X_full[ev_m]))

    # ---- 修正点：真正的「预测 delta」标签 ----
    # 在 train 上拟合线性预测器 X -> D_hat（全 delta 维），再对 D_hat 聚类。
    # 推理时只需 (s_t, a_t) 即可算出 D_hat，**不使用任何未来真值**。
    Xd_tr = np.concatenate([X_full[tr_m], np.ones((tr_m.sum(), 1))], axis=1)
    Xd_ev = np.concatenate([X_full[ev_m], np.ones((ev_m.sum(), 1))], axis=1)
    w_pred, *_ = np.linalg.lstsq(Xd_tr, D_full[tr_m], rcond=None)
    dhat_tr, dhat_ev = Xd_tr @ w_pred, Xd_ev @ w_pred
    p_p, c_p = kmeans_fit(dhat_tr, K, seed=2)
    cands["kmeans_p(预测delta)"] = (kmeans_assign(p_p, c_p, dhat_tr),
                                    kmeans_assign(p_p, c_p, dhat_ev))

    # oracle：仍用真实 delta 聚类（中心只在 train 拟合），**仅作上界**
    p_o, c_o = kmeans_fit(D_full[tr_m], K, seed=3)
    cands["oracle(真实delta)"] = (kmeans_assign(p_o, c_o, D_full[tr_m]),
                                 kmeans_assign(p_o, c_o, D_full[ev_m]))

    # 负对照：状态可得的随机投影分组（检验「任意状态分段」是否自带增益）
    rs = np.random.default_rng(11)
    R = rs.normal(size=(X_full.shape[1], X_full.shape[1]))
    p_r, c_r = kmeans_fit((X_full[tr_m] @ R), K, seed=4)
    cands["randproj(随机投影)"] = (kmeans_assign(p_r, c_r, X_full[tr_m] @ R),
                                   kmeans_assign(p_r, c_r, X_full[ev_m] @ R))

    # ---- 方案1 新 B 臂：按「预测 delta 的量级/方向」分桶 ----
    # 动机：修正后表显示 GAIN 度量的是**细粒度分段能力**（randproj GAIN 35.63% >
    # oracle 35.56%），而非标签的语义信息量。故需要一个「同样细粒度分段、但去掉
    # 形态语义」的标签作为 B 臂，才能与 kmeans_p（按形态聚类）分离。
    # 构造：取预测 delta 在**能量主方向**上的投影（保留 1 个连续标量 = 量级+符号），
    # 在 train 上按分位数切成 8 桶。推理时只需 (s_t,a_t) -> D_hat -> 标量 -> 桶。
    Dh_tr, Dh_ev = dhat_tr, dhat_ev
    # 主方向用 train 的 D_hat 协方差首主成分（只在 train 拟合）
    Dh_c = Dh_tr - Dh_tr.mean(0)
    _, _, Vt = np.linalg.svd(Dh_c, full_matrices=False)
    e1 = Vt[0]                                   # 能量主方向
    proj_tr, proj_ev = Dh_tr @ e1, Dh_ev @ e1
    qs = np.percentile(proj_tr, np.linspace(0, 100, K + 1)[1:-1])  # train 分位
    cands["B_量级方向(新B臂)"] = (np.searchsorted(qs, proj_tr),
                                  np.searchsorted(qs, proj_ev))

    # B 的严格对照：同样用预测 delta，但按**随机方向**投影分桶（去量级信息、
    # 保留分段），用于检验「是量级本身有用，还是任何连续标量分段都有用」。
    # 随机单位向量，长度须与 D_hat 维数一致。
    e_rand = rs.normal(size=Dh_tr.shape[1])
    e_rand /= np.linalg.norm(e_rand)
    pr_tr, pr_ev = Dh_tr @ e_rand, Dh_ev @ e_rand
    qs_r = np.percentile(pr_tr, np.linspace(0, 100, K + 1)[1:-1])
    cands["B_随机方向(对照)"] = (np.searchsorted(qs_r, pr_tr),
                                 np.searchsorted(qs_r, pr_ev))

    print("%-22s %7s %9s %10s %10s %10s" %
          ("标签", "组数", "标签熵", "R2_标签", "R2_线性", "GAIN"))
    print("-" * 80)
    rows = []
    for name, (gtr, gev) in cands.items():
        gtr = np.asarray(gtr).astype(np.int64)
        gev = np.asarray(gev).astype(np.int64)
        ngrp = len(np.unique(gev))
        if ngrp < 2:
            continue
        r2_lin, r2_route, r2_lab = eval_r2(
            X[tr_m], D[tr_m], X[ev_m], D[ev_m], gtr, gev)
        ent = label_entropy(gev, K)
        print("%-22s %7d %9.2f %9.2f%% %9.2f%% %9.2f%%" %
              (name, ngrp, ent, r2_lab * 100, r2_lin * 100,
               (r2_route - r2_lin) * 100))
        rows.append((name, r2_lab, r2_route - r2_lin, ent))

    print("\n读法：")
    print("  R2_标签 = 标签单独能解释的 delta 方差（信息量）")
    print("  R2_线性 = 主干层能力（train 拟合 / eval 评估）")
    print("  GAIN    = 门控在全局线性之上额外获得的 R²（路由增量贡献）")
    print("  关键检验：GAIN 是否随 R2_标签 单调上升。")
    print("  注：oracle 泄漏未来，只能作上界，不能作为可用方法。")
    print("  注：全部为 episode 级样本外评估；旧版为样本内，数值不可直接比较。")

    # 单调性（仅在 deployable 候选上检验，oracle 单独标注）
    dep = [r for r in rows if not r[0].startswith("oracle")]
    dep_sorted = sorted(dep, key=lambda r: r[1])
    xs = [r[1] for r in dep_sorted]
    ys = [r[2] for r in dep_sorted]
    n = len(xs)
    if n > 2 and np.std(xs) > 0 and np.std(ys) > 0:
        rho = float(np.corrcoef(xs, ys)[0, 1])
        print("\n  deployable 候选 n=%d，R2_标签 与 GAIN 的 Pearson 相关 = %+.3f"
              % (n, rho))
        print("  排序（R2_标签 由低到高）: " +
              " < ".join("%s(%.1f%%)" % (r[0], r[2] * 100) for r in dep_sorted))


if __name__ == "__main__":
    main()
