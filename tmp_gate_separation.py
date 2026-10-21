# -*- coding: utf-8 -*-
"""零成本事后检验：实际学出来的门控，其「分离度」能否预测实际 one-step 增益？

【为什么之前的数据不能回答这个问题】
1. tmp_partition_quality.py 用的是**硬标签** + 线性代理路由，与真实训练无关。
2. 真实门控读的是**学出来的** yao_log（model_e1a.py:142 → gua_gate），
   不是硬标签。但日志显示 V3 的 爻acc = 0.999，即爻头几乎完美拟合标签，
   因此 yao_log ≈ 硬标签的 ±1 编码，用标签代理是合理的。

【本脚本】
对 checkpoints-e1a-corrected 下全部 138 个 run：
  1. 读 gua_gate.weight/bias → 还原门控对 6 维爻 logits 的响应
  2. 用真实 6 维爻标签的 ±1 编码作输入 → 门控概率 g（严格复现 gate_outer 分支）
  3. argmax(g) = 实际门控分组 → 算分离度（组间质心距/组内散布）与熵
  4. 与该 run 的实际 one-step best 关联（越低越好）

【严格口径】
- 状态空间只用 eval 集（episode 级切分，固定 seed 0），标准化后计算
- 门控几何不拟合任何东西（只是把训练好的权重作用到标签上）
- 排除 free/no_gua 臂（无 gua_gate），单独统计
"""
import glob
import io
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from layout_h import OBJ_DIMS, ACTION_DIM

CKPT_DIR = "checkpoints-e1a-corrected"
LOG_DIR = "logs-corrected"
N_EP = 400


def gate_probs(W, b, yao6, n_phases):
    """复现 E1aModel._gua_probs（gate_outer 自动分支）。
    yao6: (N,6) ∈ {0,1} → logits = (±1) @ W.T + b
    gate_outer (n_gua == n_phases²) 时：reshape(-1,side,side) 双 softmax 后外积。
    """
    pm = yao6.astype(np.float32) * 2.0 - 1.0
    lg = pm @ W.T + b
    n_gua = W.shape[0]
    if n_gua == n_phases ** 2:                     # gate_outer=True
        side = n_phases
        z = lg.reshape(-1, side, side)
        e = np.exp(z - z.max(-1, keepdims=True))
        p0 = e.sum(-1); p0 /= p0.sum(-1, keepdims=True)     # softmax over cols
        p1 = e.sum(1);  p1 /= p1.sum(1, keepdims=True)      # softmax over rows
        return np.einsum("bi,bj->bij", p0, p1).reshape(-1, n_gua)
    z = lg - lg.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def separation(lab, feats):
    present = np.unique(lab)
    if len(present) < 2:
        return 0.0
    cent = np.stack([feats[lab == k].mean(0) for k in present])
    idx = np.searchsorted(present, lab)
    within = float(np.linalg.norm(feats - cent[idx], axis=1).mean())
    cd = np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(-1))
    between = float(cd[np.triu_indices(len(present), 1)].mean())
    return between / within if within > 1e-12 else 0.0


def entropy_of(lab, n):
    p = np.bincount(lab, minlength=n) / len(lab)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    d = np.load("trajectories_hinge.npz", allow_pickle=True)
    S0 = d["states"][:N_EP, :-1]
    n_ep, n_fr = S0.shape[0], S0.shape[1]
    X = np.concatenate([S0.reshape(-1, OBJ_DIMS),
                        d["actions"][:N_EP, :-1].reshape(-1, ACTION_DIM)], 1)
    yao6 = d["yao"][:N_EP, :-1].reshape(-1, 6).astype(np.float32)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n_ep)
    ev_ep = np.sort(perm[n_ep // 2:])
    ev = np.zeros(n_ep * n_fr, bool)
    ev.reshape(n_ep, n_fr)[:] = np.isin(np.arange(n_ep), ev_ep)[:, None]

    Xev = X[ev]
    Xs = (Xev - Xev.mean(0)) / np.maximum(Xev.std(0), 1e-9)
    yev = yao6[ev]
    print("eval 帧 %d | 状态维 %d\n" % (ev.sum(), Xs.shape[1]))

    import build_e1a_report as B
    rows = []
    for cp in sorted(glob.glob(os.path.join(CKPT_DIR, "*", "best.pt"))):
        tag = os.path.basename(os.path.dirname(cp))
        lp = os.path.join(LOG_DIR, tag + ".log")
        if not os.path.exists(lp):
            continue
        r = B.parse_log(lp)
        if not (r.get("done") and r.get("best") is not None):
            continue
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        rec = dict(tag=tag, variant=r["variant"], seed=r["seed"],
                   no_gua=r["no_gua"], best=r["best"],
                   collapse=r["collapse"])
        if "gua_gate.weight" in sd:
            W = sd["gua_gate.weight"].numpy()
            b = sd["gua_gate.bias"].numpy()
            n_gua = W.shape[0]
            # 还原 gate_axes：gua_gate 的输入维度决定读几根爻
            n_in = W.shape[1]
            n_axes = 3 if n_in == 6 else n_in // 2
            n_phases = 2 ** n_axes
            sub = yev if n_axes == 3 else np.concatenate(
                [yev[:, 2 * a:2 * a + 2] for a in range(n_axes)], 1)
            g = gate_probs(W, b, sub.astype(np.float32), n_phases)
            lab = g.argmax(1)
            occ = np.bincount(lab, minlength=n_gua)
            rec.update(gate="gua", n_gua=n_gua, n_axes=n_axes,
                       sep=separation(lab, Xs), ent=entropy_of(lab, n_gua),
                       ngrp=int((occ > 0).sum()),
                       min_frac=float(occ.min() / occ.sum()))
        elif "free_gate.weight" in sd:
            rec.update(gate="free", n_gua=sd["free_gate.weight"].shape[0],
                       sep=None, ent=None, ngrp=None, min_frac=None)
        else:
            rec.update(gate="none", n_gua=None, sep=None, ent=None,
                       ngrp=None, min_frac=None)
        rows.append(rec)

    print("可分析 run：%d\n" % len(rows))
    cnt = defaultdict(int)
    for r in rows:
        cnt[r["gate"]] += 1
    print("  " + " | ".join("%s=%d" % kv for kv in sorted(cnt.items())) + "\n")

    gua = [r for r in rows if r["sep"] is not None]
    per = defaultdict(list)
    for r in gua:
        per[(r["variant"], r["no_gua"])].append(r)

    print("%-8s %-7s %3s %9s %7s %7s %8s %9s" %
          ("臂", "no_gua", "n", "分离度", "有效组", "最小占比", "熵", "best均值"))
    print("-" * 66)
    for (v, ng), rs in sorted(per.items()):
        f = lambda k: np.mean([x[k] for x in rs])
        print("%-8s %-7s %3d %9.3f %7.1f %9.4f %8.2f %9.3f" %
              (v, ng, len(rs), f("sep"), f("ngrp"), f("min_frac"),
               f("ent"), f("best")))
    print()

    # 相关：分离度 vs best（分离度高应 → best 低 → r<0）
    for name, sub in (("全部门控臂", gua),
                      ("同 K=64（消专家数共变）",
                       [r for r in gua if r["n_gua"] == 64])):
        if len(sub) < 4:
            continue
        s = np.array([r["sep"] for r in sub])
        y = np.array([r["best"] for r in sub])
        ok = np.isfinite(s) & np.isfinite(y)
        s, y = s[ok], y[ok]
        if len(s) < 4:
            continue
        from scipy.stats import spearmanr
        pe = float(np.corrcoef(s, y)[0, 1])
        sp = float(spearmanr(s, y).statistic)
        print("=== %s：分离度 vs 实际 one-step best（n=%d）===" % (name, len(s)))
        print("  Pearson %+.3f | Spearman %+.3f  (期望 r<0)" % (pe, sp))
        print("  → %s\n" % ("支持：分离度高→误差低" if pe < -0.2 else
                            "反向：分离度高→误差高" if pe > 0.2 else
                            "不支持：无相关"))

    with open("analysis_output/gate_separation_check.json", "w",
              encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/gate_separation_check.json")


if __name__ == "__main__":
    main()
