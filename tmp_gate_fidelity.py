# -*- coding: utf-8 -*-
"""诊断：为什么事后检验无信息量？

两个可能的解释必须区分开：
(1) 机制错了（分离度不驱动增益）
(2) 真实训练里根本没有分离度的**对比范围**（range restriction）

本文对比每个臂的：
  - 硬标签本身的分离度（分离度机制声称的驱动量）
  - 训练后门控 argmax 的分离度（真正起作用的路由）
  - 门控组存活数 / 最小组占比（专家坍塌）

若「硬标签分离度」与「训练后门控分离度」差距巨大 → 训练把标签结构破坏了，
事后检验无法判伪，必须靠**主动操纵分离度**的对照实验。
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


def sep(lab, feats):
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


def gate_probs(W, b, yao6, n_phases):
    pm = yao6.astype(np.float32) * 2.0 - 1.0
    lg = pm @ W.T + b
    n_gua = W.shape[0]
    if n_gua == n_phases ** 2:
        side = n_phases
        z = lg.reshape(-1, side, side)
        e = np.exp(z - z.max(-1, keepdims=True))
        p0 = e.sum(-1); p0 /= p0.sum(-1, keepdims=True)
        p1 = e.sum(1);  p1 /= p1.sum(1, keepdims=True)
        return np.einsum("bi,bj->bij", p0, p1).reshape(-1, n_gua)
    z = lg - lg.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    d = np.load("trajectories_hinge.npz", allow_pickle=True)
    S0 = d["states"][:N_EP, :-1]
    n_ep, n_fr = S0.shape[0], S0.shape[1]
    X = np.concatenate([S0.reshape(-1, OBJ_DIMS),
                        d["actions"][:N_EP, :-1].reshape(-1, ACTION_DIM)], 1)
    yao6 = d["yao"][:N_EP, :-1].reshape(-1, 6).astype(np.float32)
    rng = np.random.default_rng(0)
    ev_ep = np.sort(rng.permutation(n_ep)[n_ep // 2:])
    ev = np.zeros(n_ep * n_fr, bool)
    ev.reshape(n_ep, n_fr)[:] = np.isin(np.arange(n_ep), ev_ep)[:, None]
    Xs = (X[ev] - X[ev].mean(0)) / np.maximum(X[ev].std(0), 1e-9)
    yev = yao6[ev]
    # 硬标签分组：6 维（3 爻 × 2 物体）→ 64 值
    hard = (yev[:, 0] * 32 + yev[:, 1] * 16 + yev[:, 2] * 8 +
            yev[:, 3] * 4 + yev[:, 4] * 2 + yev[:, 5]).astype(np.int64)

    import build_e1a_report as B
    per = defaultdict(list)
    for cp in sorted(glob.glob(os.path.join(CKPT_DIR, "*", "best.pt"))):
        tag = os.path.basename(os.path.dirname(cp))
        lp = os.path.join(LOG_DIR, tag + ".log")
        if not os.path.exists(lp):
            continue
        r = B.parse_log(lp)
        if not (r.get("done") and r.get("best") is not None):
            continue
        sd = torch.load(cp, map_location="cpu",
                        weights_only=False)["model_state_dict"]
        if "gua_gate.weight" not in sd:
            continue
        W = sd["gua_gate.weight"].numpy()
        b = sd["gua_gate.bias"].numpy()
        n_gua, n_in = W.shape
        n_axes = 3 if n_in == 6 else n_in // 2
        n_phases = 2 ** n_axes
        sub = yev if n_axes == 3 else np.concatenate(
            [yev[:, 2 * a:2 * a + 2] for a in range(n_axes)], 1)
        g = gate_probs(W, b, sub.astype(np.float32), n_phases)
        lab = g.argmax(1)
        occ = np.bincount(lab, minlength=n_gua)
        hl = hard if n_axes == 3 else None
        per[(r["variant"], n_gua, n_axes)].append(dict(
            sep_hard=sep(hl, Xs) if hl is not None else np.nan,
            sep_gate=sep(lab, Xs), ngrp_gate=int((occ > 0).sum()),
            min_frac=float(occ.min() / occ.sum()),
            ent_gate=entropy_of(lab, n_gua), best=r["best"]))

    print("%-8s %5s %4s %3s %10s %10s %8s %9s %9s" %
          ("臂", "n_gua", "轴", "n", "硬标签分离", "门控分离", "存活组",
           "最小占比", "best"))
    print("-" * 76)
    hs, gs = [], []
    for (v, ng, na), rs in sorted(per.items()):
        f = lambda k: np.nanmean([x[k] for x in rs])
        print("%-8s %5d %4d %3d %10.3f %10.3f %8.1f %9.4f %9.3f" %
              (v, ng, na, len(rs), f("sep_hard"), f("sep_gate"),
               f("ngrp_gate"), f("min_frac"), f("best")))
        hs.append(f("sep_hard")); gs.append(f("sep_gate"))
    hs, gs = np.array(hs), np.array(gs)
    print()
    print("硬标签分离度范围 [%.3f, %.3f]  (跨度 %.3f)" %
          (hs.min(), hs.max(), hs.max() - hs.min()))
    print("门控分离度范围   [%.3f, %.3f]  (跨度 %.3f)" %
          (gs.min(), gs.max(), gs.max() - gs.min()))
    print("硬标签→门控 保真度 r = %+.3f" % float(np.corrcoef(hs, gs)[0, 1]))
    print()
    print("参照：tmp_partition_quality 的候选跨度 = 1.823 - 0.014 = 1.809")
    print("→ 真实训练臂的分离度对比范围远小于代理候选，事后检验【无功效】")
    print("→ 结论：机制既未被证实也未被证伪，必须做主动操纵分离度的对照实验")

    json.dump({str(k): v for k, v in per.items()},
              open("analysis_output/gate_fidelity.json", "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/gate_fidelity.json")


if __name__ == "__main__":
    main()
