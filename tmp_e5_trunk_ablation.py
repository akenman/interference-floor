# -*- coding: utf-8 -*-
"""E5-M2：E5 里那 +90% 的增益，是**头**在起作用还是 **trunk** 在隐式辨识参数？

问题
----
E5 在参数错（g1125）与未建模力（drag）上得到 +75.7% / +92.4% 的单步增益。
但 E5 是"trunk + 头"一起训的，所以增益可能来自两条完全不同的路径：

  (a) **头**在起作用：头学会了一个与状态相关的修正量（隐式辨识）；
  (b) **trunk** 在起作用：隐状态里编码了"这个世界的参数是什么"，
      头只是把这个编码读出来。

审稿人会问这个，而且它决定 v2 里功劳记给谁。

设计（每个 mismatch 条件两次，唯一变量 = 冻结哪一份 trunk）
  full        : trunk+头 一起训（= E5 的结果，作参照）
  frozen_match: 冻结在 **matched 数据**上训出的 trunk，只训头
                -> trunk 里**没有** mismatch 信息，头只能靠输入状态自己推断
  frozen_mis  : 冻结在**同一 mismatch 数据**上训出的 trunk，只训头
                -> 若明显优于 frozen_match，说明参数编码确实在 trunk 里

判读
  * frozen_match ≈ full          -> 头自己能推断，功劳在头（且说明硬编码参数没必要）
  * frozen_match << full         -> 参数辨识发生在 trunk 里，头只是读出器
  * frozen_mis  > frozen_match   -> 直接证明 trunk 里编码了参数信息

用法：python -X utf8 tmp_e5_trunk_ablation.py
输出：analysis_output/e5_trunk_ablation.json
"""
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
SEED, EPOCHS, BATCH, LR = 42, 60, 1024, 3e-4
CKPT_DIR = "checkpoints-E5"


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


def split(n, seed=SEED):
    segs = np.arange(n)
    np.random.RandomState(seed).shuffle(segs)
    nv = max(1, int(n * 0.1))
    return segs[nv:], segs[:nv]


class Win:
    """论文 E0Data 口径：输入 [i0,i0+L)、a0=action[i0+L]、目标帧 i0+L。"""

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

    def prior_step_norm(self, s_last_raw, a_raw):
        return (prior_raw_g(s_last_raw, a_raw, self.g) - self.mo) / (self.so + 1e-8)


def rmse_w(pred_raw, tgt_raw):
    e = ((pred_raw[:, POS] - tgt_raw[:, POS]) ** 2) * W_POS
    return float(np.sqrt(e.mean()))


def make_model(mo, so):
    torch.manual_seed(SEED)
    m = WorldModelH(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=1,
                    hidden_dim=256, num_prototypes=32, prior_spec=SPEC).to(DEV)
    m.set_stats(mo, so)
    return m


def train(model, W, d_tr, freeze_trunk, epochs=EPOCHS, tag=""):
    params = ([p for n, p in model.named_parameters() if not n.startswith("state_head")]
              if not freeze_trunk else [])
    for n, p in model.named_parameters():
        p.requires_grad_(not freeze_trunk or n.startswith("state_head"))
    opt = torch.optim.AdamW(
        [{"params": list(model.state_head.parameters()), "lr": LR}] +
        ([{"params": params, "lr": LR}] if params else []), weight_decay=1e-5)
    X = torch.cat([W.x(), W.c()], dim=2)
    A = torch.from_numpy(W.a0.astype(np.float32)).to(DEV)
    D = torch.from_numpy(d_tr).to(DEV)
    n = W.n
    for ep_i in range(1, epochs + 1):
        perm = np.random.RandomState(SEED + ep_i).permutation(n)
        for j in range(0, n, BATCH):
            b = perm[j:j + BATCH]
            h, _ = model.encode_sequence(X[b])
            loss = F.huber_loss(model.predict_state(h, A[b]), D[b], reduction="mean")
            opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.item())


def evaluate(model, W, p):
    """返回 (单步 px, 闭环 k=5 px)。p = 该条件的先验（归一化，单步）。"""
    X, C = W.x(), W.c()
    A = torch.from_numpy(W.a0.astype(np.float32)).to(DEV)
    with torch.no_grad():
        h, _ = model.encode_sequence(torch.cat([X, C], dim=2))
        dl = model.predict_state(h, A).cpu().numpy()
    one = rmse_w((p + dl) * (W.so + 1e-8) + W.mo, W.y)
    cur = W.frames.copy()
    for k in range(K_EVAL):
        ak = W.af[:, k].astype(np.float64)
        pn = W.prior_step_norm(cur[:, -1].astype(np.float64), ak)
        Xk = torch.from_numpy(
            ((cur - W.mo) / (W.so + 1e-8)).astype(np.float32)).to(DEV)
        with torch.no_grad():
            hk, _ = model.encode_sequence(torch.cat([Xk, C], dim=2))
            dk = model.predict_state(
                hk, torch.from_numpy(ak.astype(np.float32)).to(DEV)).cpu().numpy()
        nxt = pn + dk
        nxt_raw = nxt * (W.so + 1e-8) + W.mo
        if k == K_EVAL - 1:
            roll = rmse_w(nxt_raw, W.fut[:, k])
        cur = np.concatenate([cur[:, 1:], nxt_raw[:, None, :].astype(np.float32)],
                             axis=1)
    return one, roll


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    out = dict(protocol=dict(spec=SPEC, L=L, K=K_EVAL, epochs=EPOCHS,
                             seed=SEED, lr=LR, batch=BATCH,
                             design="full vs trunk-frozen (matched) vs trunk-frozen "
                                    "(mismatched)"),
               conditions={})

    # ---------- 条件 1：matched 上训一份 trunk，供冻结复用 ----------
    st, ct, ac = load_world("hinge")
    trk, vak = split(st.shape[0])
    mo = st[trk].reshape(-1, OBJ_DIMS).mean(axis=0)
    so = st[trk].reshape(-1, OBJ_DIMS).std(axis=0)
    Wm_tr = Win(st, ct, ac, trk, mo, so, 900.0)
    p_m = Wm_tr.prior_step_norm(Wm_tr.frames[:, -1].astype(np.float64),
                                Wm_tr.a0.astype(np.float64)).astype(np.float32)
    d_m = (Wm_tr.target_norm() - p_m).astype(np.float32)

    trunk_m = make_model(mo, so)
    l = train(trunk_m, Wm_tr, d_m, freeze_trunk=False)
    torch.save({"model_state_dict": trunk_m.state_dict(), "mean_obj": mo,
                "std_obj": so, "condition": "matched", "g": 900.0},
               os.path.join(CKPT_DIR, "matched_trunk.pt"))
    print(f"[matched] 训练完成 huber={l:.6f}  -> 已存 {CKPT_DIR}/matched_trunk.pt")

    # 用同一份 matched 模型在 matched 验证集上评估，作为基线行
    Wm_va = Win(st, ct, ac, vak, mo, so, 900.0)
    p_mv = Wm_va.prior_step_norm(Wm_va.frames[:, -1].astype(np.float64),
                                 Wm_va.a0.astype(np.float64)).astype(np.float32)
    one, roll = evaluate(trunk_m, Wm_va, p_mv)
    out["conditions"]["matched_full"] = dict(
        note="matched 数据，trunk+头一起训（E5 的 matched）",
        prior_one=rmse_w(p_mv * (so + 1e-8) + mo, Wm_va.y),
        hybrid_one=one, hybrid_roll_k5=roll, n_train=int(Wm_tr.n))
    print(f"  matched_full: prior={out['conditions']['matched_full']['prior_one']:.4f} "
          f"hybrid={one:.4f}")

    # ---------- 条件 2/3：g1125 与 drag ----------
    for cond, data, g_prior in (("g1125", "hinge", 1125.0),
                                ("drag", "drag", 900.0)):
        st2, ct2, ac2 = load_world(data)
        trk2, vak2 = split(st2.shape[0])
        mo2 = st2[trk2].reshape(-1, OBJ_DIMS).mean(axis=0)
        so2 = st2[trk2].reshape(-1, OBJ_DIMS).std(axis=0)
        Wt = Win(st2, ct2, ac2, trk2, mo2, so2, g_prior)
        Wv = Win(st2, ct2, ac2, vak2, mo2, so2, g_prior)
        p_tr = Wt.prior_step_norm(Wt.frames[:, -1].astype(np.float64),
                                  Wt.a0.astype(np.float64)).astype(np.float32)
        d_tr = (Wt.target_norm() - p_tr).astype(np.float32)
        p_va = Wv.prior_step_norm(Wv.frames[:, -1].astype(np.float64),
                                  Wv.a0.astype(np.float64)).astype(np.float32)
        prior_one = rmse_w(p_va * (so2 + 1e-8) + mo2, Wv.y)
        print(f"\n=== {cond}  (先验单步 {prior_one:.4f} px, train={Wt.n})")

        rows = {}
        # (a) full：trunk+头一起训
        m_full = make_model(mo2, so2)
        train(m_full, Wt, d_tr, freeze_trunk=False)
        one, roll = evaluate(m_full, Wv, p_va)
        rows["full"] = dict(hybrid_one=one, hybrid_roll_k5=roll)
        print(f"  full        : one={one:.4f} roll={roll:.4f}")

        # (b) frozen_match：冻结在 matched 上训的 trunk（用同维统计量）
        m_fm = make_model(mo, so)                       # matched 的统计量 → matched trunk
        m_fm.load_state_dict(trunk_m.state_dict())
        m_fm.set_stats(mo, so)
        # 换到本条件的统计量时，trunk 只吃归一化输入；用本条件统计量重设
        m_fm.set_stats(mo2, so2)
        train(m_fm, Wt, d_tr, freeze_trunk=True)
        one, roll = evaluate(m_fm, Wv, p_va)
        rows["frozen_match"] = dict(hybrid_one=one, hybrid_roll_k5=roll)
        print(f"  frozen_match: one={one:.4f} roll={roll:.4f}")

        # (c) frozen_mis：冻结在本 mismatch 数据上训出的 trunk
        m_mis = make_model(mo2, so2)
        train(m_mis, Wt, d_tr, freeze_trunk=False)      # 先训出带参数的 trunk
        trunk_sd = {k: v.clone() for k, v in m_mis.state_dict().items()}
        m_fm2 = make_model(mo2, so2)                    # 同条件、同初始化
        m_fm2.load_state_dict(trunk_sd)
        m_fm2.set_stats(mo2, so2)
        train(m_fm2, Wt, d_tr, freeze_trunk=True)
        one, roll = evaluate(m_fm2, Wv, p_va)
        rows["frozen_mis"] = dict(hybrid_one=one, hybrid_roll_k5=roll)
        print(f"  frozen_mis  : one={one:.4f} roll={roll:.4f}")

        for k, v in rows.items():
            v["gain_one"] = 1 - v["hybrid_one"] / prior_one
        out["conditions"][cond] = dict(data=data, g_prior=g_prior,
                                       prior_one=prior_one, variants=rows,
                                       n_train=int(Wt.n))
        print(f"  增益：full {100*rows['full']['gain_one']:+.1f}%  "
              f"frozen_match {100*rows['frozen_match']['gain_one']:+.1f}%  "
              f"frozen_mis {100*rows['frozen_mis']['gain_one']:+.1f}%")

    with open("analysis_output/e5_trunk_ablation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/e5_trunk_ablation.json")
    print("\n判读：frozen_match ≈ full -> 功劳在头；frozen_match << full -> 参数在 trunk 里")


if __name__ == "__main__":
    main()
