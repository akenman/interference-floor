# -*- coding: utf-8 -*-
"""M4 phase 2：把 E5 与 trunk 消融扩到多 seed。

要坐实的两处**单种子**结论（都已写进论文）：
  (a) 摆世界地板比值：精确档 head/target = 0.11（M3）
  (b) 消融：frozen_match 的闭环 0.365 vs full 的 0.541（g1125，附录 C）

本脚本对每个 seed 做：
  1. matched 数据上训一份 trunk+头（作为"匹配 trunk"的来源）
  2. g1125：full / frozen_match / frozen_mis 三变体
  3. drag ：full / frozen_match 两变体

用法：python -X utf8 tmp_e5_multiseed.py --seeds 45 46 47 48
输出：analysis_output/e5_multiseed.json（逐 seed 累加，可中断续跑）
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import physics_prior_spec as P                                    # noqa: E402
from layout_h import OBJ_DIMS, feat_weights                        # noqa: E402
from model_e0 import INPUT_DIM                                     # noqa: E402
from model_h import WorldModelH                                    # noqa: E402

POS = [1, 2, 12, 13]
W_POS = feat_weights(OBJ_DIMS)[POS]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPEC = "rk4001x5"
L, K_EVAL = 8, 5
EPOCHS, BATCH, LR = 60, 1024, 3e-4
OUT = "analysis_output/e5_multiseed.json"


def prior_raw_g(raw, a_raw, g):
    old = P._G
    try:
        P._G = g
        return P.prior_raw_np(SPEC, raw, a_raw)
    finally:
        P._G = old


def load_world(name):
    f = "trajectories_hinge.npz" if name == "hinge" else f"g4_{name}.npz"
    d = np.load(f)
    return d["states"], d["contact"], d["actions"]


def split(n, seed):
    segs = np.arange(n)
    np.random.RandomState(seed).shuffle(segs)
    nv = max(1, int(n * 0.1))
    return segs[nv:], segs[:nv]


class Win:
    def __init__(self, st, ct, ac, eps, mo, so, g):
        self.mo, self.so, self.g = mo, so, g
        T = st.shape[1]
        n_win = T - L - K_EVAL + 1
        self.ep = np.repeat(eps, n_win)
        self.i0 = np.tile(np.arange(n_win), len(eps))
        off = np.arange(L)
        self.frames = st[self.ep[:, None], self.i0[:, None] + off[None, :]]
        self.cts = ct[self.ep[:, None], self.i0[:, None] + off[None, :]]
        self.a0 = ac[self.ep, self.i0 + L]
        self.y = st[self.ep, self.i0 + L].astype(np.float64)
        self.fut = np.stack([st[self.ep, self.i0 + L + k] for k in range(K_EVAL)],
                            axis=1).astype(np.float64)
        self.af = ac[self.ep[:, None], (self.i0 + L)[:, None]
                     + np.arange(K_EVAL)[None, :]]
        self.n = len(self.ep)

    def x(self):
        return torch.from_numpy(
            ((self.frames - self.mo) / (self.so + 1e-8)).astype(np.float32)).to(DEV)

    def c(self):
        return torch.from_numpy(self.cts.astype(np.float32)).to(DEV)

    def target_norm(self):
        return (self.y - self.mo) / (self.so + 1e-8)

    def prior_norm(self):
        return (prior_raw_g(self.frames[:, -1].astype(np.float64),
                            self.a0.astype(np.float64), self.g)
                - self.mo) / (self.so + 1e-8)

    def step_norm(self, s_last, a_raw):
        return (prior_raw_g(s_last, a_raw, self.g) - self.mo) / (self.so + 1e-8)


def rmse_w(pred_raw, tgt_raw):
    e = ((pred_raw[:, POS] - tgt_raw[:, POS]) ** 2) * W_POS
    return float(np.sqrt(e.mean()))


def make_model(mo, so, seed):
    torch.manual_seed(seed)
    m = WorldModelH(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=1,
                    hidden_dim=256, num_prototypes=32, prior_spec=SPEC).to(DEV)
    m.set_stats(mo, so)
    return m


def train(model, W, d_tr, freeze_trunk, seed, epochs=EPOCHS):
    for n, p in model.named_parameters():
        p.requires_grad_(not freeze_trunk or n.startswith("state_head"))
    groups = [{"params": list(model.state_head.parameters()), "lr": LR}]
    if not freeze_trunk:
        groups.append({"params": [p for n, p in model.named_parameters()
                                  if not n.startswith("state_head")], "lr": LR})
    opt = torch.optim.AdamW(groups, weight_decay=1e-5)
    X = torch.cat([W.x(), W.c()], dim=2)
    A = torch.from_numpy(W.a0.astype(np.float32)).to(DEV)
    D = torch.from_numpy(d_tr).to(DEV)
    n = W.n
    for ep_i in range(1, epochs + 1):
        perm = np.random.RandomState(seed * 1000 + ep_i).permutation(n)
        for j in range(0, n, BATCH):
            b = perm[j:j + BATCH]
            h, _ = model.encode_sequence(X[b])
            loss = F.huber_loss(model.predict_state(h, A[b]), D[b], reduction="mean")
            opt.zero_grad(); loss.backward(); opt.step()


def evaluate(model, W, p):
    X, C = W.x(), W.c()
    A = torch.from_numpy(W.a0.astype(np.float32)).to(DEV)
    with torch.no_grad():
        h, _ = model.encode_sequence(torch.cat([X, C], dim=2))
        dl = model.predict_state(h, A).cpu().numpy()
    one = rmse_w((p + dl) * (W.so + 1e-8) + W.mo, W.y)
    cur = W.frames.copy()
    roll = float("nan")
    for k in range(K_EVAL):
        ak = W.af[:, k].astype(np.float64)
        pn = W.step_norm(cur[:, -1].astype(np.float64), ak)
        Xk = torch.from_numpy(
            ((cur - W.mo) / (W.so + 1e-8)).astype(np.float32)).to(DEV)
        with torch.no_grad():
            hk, _ = model.encode_sequence(torch.cat([Xk, C], dim=2))
            dk = model.predict_state(
                hk, torch.from_numpy(ak.astype(np.float32)).to(DEV)).cpu().numpy()
        nxt_raw = (pn + dk) * (W.so + 1e-8) + W.mo
        if k == K_EVAL - 1:
            roll = rmse_w(nxt_raw, W.fut[:, k])
        cur = np.concatenate([cur[:, 1:], nxt_raw[:, None, :].astype(np.float32)],
                             axis=1)
    return one, roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[45, 46, 47, 48])
    a = ap.parse_args()

    out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for seed in a.seeds:
        print(f"\n########## seed {seed} ##########", flush=True)
        # ---- matched trunk ----
        st, ct, ac = load_world("hinge")
        trk, vak = split(st.shape[0], seed)
        mo = st[trk].reshape(-1, OBJ_DIMS).mean(axis=0)
        so = st[trk].reshape(-1, OBJ_DIMS).std(axis=0)
        Wm = Win(st, ct, ac, trk, mo, so, 900.0)
        pm = Wm.prior_norm().astype(np.float32)
        dm = (Wm.target_norm() - pm).astype(np.float32)
        trunk_m = make_model(mo, so, seed)
        train(trunk_m, Wm, dm, False, seed)
        Wmv = Win(st, ct, ac, vak, mo, so, 900.0)
        pmv = Wmv.prior_norm().astype(np.float32)
        one, roll = evaluate(trunk_m, Wmv, pmv)
        rec = {"matched_full": dict(prior_one=rmse_w(pmv * (so + 1e-8) + mo, Wmv.y),
                                    hybrid_one=one, hybrid_roll_k5=roll)}
        print(f"  matched_full: {rec['matched_full']}", flush=True)

        # ---- g1125 / drag ----
        for cond, data, g in (("g1125", "hinge", 1125.0), ("drag", "drag", 900.0)):
            st2, ct2, ac2 = load_world(data)
            trk2, vak2 = split(st2.shape[0], seed)
            mo2 = st2[trk2].reshape(-1, OBJ_DIMS).mean(axis=0)
            so2 = st2[trk2].reshape(-1, OBJ_DIMS).std(axis=0)
            Wt = Win(st2, ct2, ac2, trk2, mo2, so2, g)
            Wv = Win(st2, ct2, ac2, vak2, mo2, so2, g)
            ptr = Wt.prior_norm().astype(np.float32)
            dtr = (Wt.target_norm() - ptr).astype(np.float32)
            pva = Wv.prior_norm().astype(np.float32)
            prior_one = rmse_w(pva * (so2 + 1e-8) + mo2, Wv.y)
            variants = {}
            # full
            m_full = make_model(mo2, so2, seed)
            train(m_full, Wt, dtr, False, seed)
            o, r = evaluate(m_full, Wv, pva)
            variants["full"] = dict(one=o, roll=r)
            # frozen_match：载入 matched trunk
            m_fm = make_model(mo2, so2, seed)
            m_fm.load_state_dict(trunk_m.state_dict())
            m_fm.set_stats(mo2, so2)
            train(m_fm, Wt, dtr, True, seed)
            o, r = evaluate(m_fm, Wv, pva)
            variants["frozen_match"] = dict(one=o, roll=r)
            # frozen_mis（仅 g1125 做）
            if cond == "g1125":
                m_mis = make_model(mo2, so2, seed)
                train(m_mis, Wt, dtr, False, seed)
                sd = {k: v.clone() for k, v in m_mis.state_dict().items()}
                m_fm2 = make_model(mo2, so2, seed)
                m_fm2.load_state_dict(sd)
                m_fm2.set_stats(mo2, so2)
                train(m_fm2, Wt, dtr, True, seed)
                o, r = evaluate(m_fm2, Wv, pva)
                variants["frozen_mis"] = dict(one=o, roll=r)
            rec[cond] = dict(prior_one=prior_one, variants=variants)
            print(f"  {cond}: prior={prior_one:.4f} " +
                  " ".join(f"{k}={v['one']:.4f}/{v['roll']:.4f}"
                           for k, v in variants.items()), flush=True)
        out[str(seed)] = rec
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"[save] {OUT}  (seeds so far: {sorted(out, key=int)})", flush=True)
    print("\nDONE")


if __name__ == "__main__":
    main()
