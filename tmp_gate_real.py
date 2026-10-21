# -*- coding: utf-8 -*-
"""真实门控分布提取：对每个 checkpoint 在 eval 窗口上前向，得到实际 g。

与 tmp_gate_separation.py（±1 标签代理）的区别：
  这里用训练后模型真正算出的 yao_log = predict_yao(h)，
  再走 _gua_probs(h, yao_log)，即门控在部署时真实看到的输入。

问题：门控分组在 h 空间还是原始状态空间分离？
  机制假设是「标签划分对状态空间的分离度」。因此两个都算：
    sep_state : argmax(g) 在原始 (s_t, a_t) 标准化空间的分离比
    sep_hidden: argmax(g) 在隐藏表示 h 空间的分离比
  若两者都不随 best 变化 → 机制在真实训练中无预测力。
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

import train_e1a as T
from train_e1a import HingeDataE1a, L, K
from model_e1a import E1aModel

CKPT_DIR = "checkpoints-e1a-corrected"
LOG_DIR = "logs-corrected"
N_EP = 400
MAXW = 8000          # 每 run 采样窗口数，控制 CPU 成本


def sep(lab, feats):
    p = np.unique(lab)
    if len(p) < 2:
        return 0.0
    c = np.stack([feats[lab == k].mean(0) for k in p])
    d = np.linalg.norm(feats - c[np.searchsorted(p, lab)], axis=1)
    within = float(d.mean())
    cd = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    between = float(cd[np.triu_indices(len(p), 1)].mean())
    return between / within if within > 1e-12 else 0.0


def ent(lab, n):
    pb = np.bincount(lab, minlength=n) / len(lab)
    pb = pb[pb > 0]
    return float(-(pb * np.log2(pb)).sum())


def main():
    d = np.load("trajectories_hinge.npz")
    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
    targets = d.get("targets", np.zeros((len(states), 2)))
    N = states.shape[0]
    r = np.random.RandomState(42)
    segs = np.arange(N); r.shuffle(segs)
    val_eps = segs[:max(1, int(N * 0.1))]
    tr_eps = segs[int(N * 0.1):]
    of = states[tr_eps].reshape(-1, OBJ_DIMS).astype(np.float64)
    mean_obj = of.mean(0).astype(np.float32)
    std_obj = of.std(0).astype(np.float32)

    ds = HingeDataE1a(states, contact, actions, yao, targets, val_eps,
                      L, K, mean_obj, std_obj, None)
    rng = np.random.RandomState(0)
    idx_all = np.arange(len(ds))
    if len(idx_all) > MAXW:
        idx_all = np.sort(rng.choice(idx_all, size=MAXW, replace=False))

    import build_e1a_report as B
    rows = []
    tags = sorted(os.path.basename(os.path.dirname(p)) for p in
                  glob.glob(os.path.join(CKPT_DIR, "*", "best.pt")))
    print("eval 窗口 %d（每 run）\n" % len(idx_all))

    for ii, tag in enumerate(tags, 1):
        cp = os.path.join(CKPT_DIR, tag, "best.pt")
        lp = os.path.join(LOG_DIR, tag + ".log")
        if not os.path.exists(lp):
            continue
        rec = B.parse_log(lp)
        if not (rec.get("done") and rec.get("best") is not None):
            continue
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        if cfg.get("no_gua"):
            continue
        if cfg.get("free_moe"):
            continue                      # free 门控不由标签驱动，单列
        em = cfg.get("energy_median")
        vc = T.VARIANTS[cfg["variant"]]
        m = E1aModel(axes=tuple(cfg["axes"]), free_moe=cfg["free_moe"],
                     no_gua=False, n_gua_free=vc["n_gua_free"], energy_median=em,
                     input_dim=cfg["input_dim"], obj_dim=cfg["obj_dim"],
                     action_dim=cfg["action_dim"],
                     hidden_dim=cfg["hidden_dim"],
                     num_prototypes=cfg["num_prototypes"],
                     n_gua_override=vc.get("n_gua_override"),
                     gate_axes=vc.get("gate_axes")).eval()
        m.load_state_dict(ck["model_state_dict"])
        m.set_stats(mean_obj, std_obj)

        gs, hs, xs = [], [], []
        with torch.no_grad():
            for j in range(0, len(idx_all), 2048):
                b = ds.gather(idx_all[j:j + 2048])
                xb, ab = b["x"], b["a0"]
                h, c = m.encode_sequence(xb)
                yao_log = m.predict_yao(h)
                g = m._gua_probs(h, yao_log)
                gs.append(g.numpy())
                hs.append(h.numpy())
                xs.append(torch.cat(
                    [xb[:, -1, :OBJ_DIMS], ab], 1).numpy())
        G = np.concatenate(gs); H = np.concatenate(hs); X = np.concatenate(xs)
        Hs = (H - H.mean(0)) / np.maximum(H.std(0), 1e-9)
        Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
        lab = G.argmax(1)
        n_gua = G.shape[1]
        occ = np.bincount(lab, minlength=n_gua)
        row = dict(tag=tag, variant=rec["variant"], seed=rec["seed"],
                   best=rec["best"], collapse=rec["collapse"],
                   n_gua=n_gua, axes=len(cfg["axes"]),
                   sep_state=sep(lab, Xs), sep_hidden=sep(lab, Hs),
                   ent=ent(lab, n_gua), ngrp=int((occ > 0).sum()),
                   min_frac=float(occ.min() / occ.sum()),
                   top_frac=float(occ.max() / occ.sum()),
                   eff_n=float(np.exp(ent(lab, n_gua))))
        rows.append(row)
        if ii % 20 == 0:
            print("  [%d/%d] %s" % (ii, len(tags), tag))

    per = defaultdict(list)
    for r_ in rows:
        per[(r_["variant"], r_["n_gua"])].append(r_)
    print("\n%-9s %5s %3s %10s %10s %8s %8s %9s %9s" %
          ("臂", "n_gua", "n", "sep_state", "sep_hidden", "存活组",
           "eff_n", "min占比", "best"))
    print("-" * 80)
    for (v, ng), rs in sorted(per.items()):
        f = lambda k: float(np.mean([x[k] for x in rs]))
        print("%-9s %5d %3d %10.3f %10.3f %8.1f %8.2f %9.4f %9.3f" %
              (v, ng, len(rs), f("sep_state"), f("sep_hidden"),
               f("ngrp"), f("eff_n"), f("min_frac"), f("best")))
    print()

    from scipy.stats import spearmanr
    for key in ("sep_state", "sep_hidden", "ent", "eff_n", "ngrp"):
        s = np.array([r_[key] for r_ in rows])
        y = np.array([r_["best"] for r_ in rows])
        pe = float(np.corrcoef(s, y)[0, 1])
        sp = float(spearmanr(s, y).statistic)
        print("  %-11s vs best: Pearson %+.3f  Spearman %+.3f" % (key, pe, sp))
    print()
    s = np.array([r_["sep_state"] for r_ in rows])
    h_ = np.array([r_["sep_hidden"] for r_ in rows])
    print("  sep_state 范围 [%.3f, %.3f] 跨度 %.3f" %
          (s.min(), s.max(), s.max() - s.min()))
    print("  代理候选跨度 1.809 → 对比范围比代理小 %.1f 倍" %
          (1.809 / max(s.max() - s.min(), 1e-9)))

    json.dump(rows, open("analysis_output/gate_real.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/gate_real.json")


if __name__ == "__main__":
    main()
