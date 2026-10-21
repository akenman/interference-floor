# -*- coding: utf-8 -*-
"""E5-C1（修正版）：先验**结构性错**时，残差头能不能赢？

为什么重写
----------
第一版有两个口径错误，导致 matched 条件出现 +56% 的**假增益**：
  1. 动作索引错一位：用了 a0 = action[i0+L] 而非 action[i0+L-1]。
     实测这一位错就让先验单步误差从 0.018 涨到 0.029 px（+60%），
     于是头"学到"的其实是修复我自己的错线；
  2. 位置 RMSE 未加权：论文 validate() 用 feat_weights_h 加权（位置权重 2.0）。
用论文自己的 validate() 复核后：精确档头增益只有 **+0.4%**（0.0378 → 0.0377），
与论文 §4.4 一致；粗档才有 +66~75%。所以第一版的结论必须推翻。

本版口径与论文 train_hinge.validate 一致：
  窗口 [i0, i0+L)、a0 = action[i0+L-1]、目标帧 i0+L、
  位置维 POS=[1,2,12,13]、feat_weights 加权。

判据（预先写死）：
  GO      闭环 k=5 比值 ≤ 0.70（头带来 ≥30% 真实增益）
  PARTIAL 0.70 < 比值 ≤ 0.95
  NO-GO   比值 > 0.95 —— 转写负面论文

用法：python -X utf8 tmp_e5_head_value.py
输出：analysis_output/e5_head_value.json
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


def prior_raw_g(spec, raw, a_raw, g):
    """调用原实现，仅把模块级 _G 换成注入值（力矩/科氏项完全不变）。"""
    old = P._G
    try:
        P._G = g
        return P.prior_raw_np(spec, raw, a_raw)
    finally:
        P._G = old


def selfcheck():
    rng = np.random.RandomState(0)
    raw = rng.randn(64, OBJ_DIMS) * 5 + np.arange(OBJ_DIMS) * 2
    a = rng.randn(64) * 0.3
    err = float(np.max(np.abs(P.prior_raw_np(SPEC, raw, a)
                              - prior_raw_g(SPEC, raw, a, P._G))))
    assert err < 1e-12, err
    print(f"[selfcheck] 重力注入与原实现 max|Δ| = {err:.2e}  OK")


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
    """论文口径窗口：输入 [i0, i0+L)，a0 = action[i0+L-1]，目标帧 i0+L。"""

    def __init__(self, st, ct, ac, eps, mo, so, g_prior):
        self.mo, self.so, self.g = mo, so, g_prior
        T = st.shape[1]
        n_win = T - L - K_EVAL + 1
        self.ep = np.repeat(eps, n_win)
        self.i0 = np.tile(np.arange(n_win), len(eps))
        off = np.arange(L)
        self.frames = st[self.ep[:, None], self.i0[:, None] + off[None, :]]
        self.cts = ct[self.ep[:, None], self.i0[:, None] + off[None, :]]
        # 论文 E0Data 口径：状态 frm[:,L-1]（帧 i0+L-1）、动作 act[i0+L]、
        # 目标帧 frm[:,L]（帧 i0+L）。已验证该口径复现论文的 0.0378 px。
        self.a0 = ac[self.ep, self.i0 + L]
        self.y = st[self.ep, self.i0 + L].astype(np.float64)
        self.fut = np.stack([st[self.ep, self.i0 + L + k] for k in range(K_EVAL)],
                            axis=1).astype(np.float64)
        self.af = ac[self.ep[:, None], (self.i0 + L)[:, None]
                     + np.arange(K_EVAL)[None, :]]
        self.n = len(self.ep)

    def x_norm(self):
        return (self.frames - self.mo) / (self.so + 1e-8)

    def prior_step(self, s_last_raw, a_raw):
        """s_last_raw: 帧 i0+L-1（原始）; a_raw: 帧 i0+L 的动作。"""
        pr = prior_raw_g(SPEC, s_last_raw, a_raw, self.g)
        return (pr - self.mo) / (self.so + 1e-8)

    def target_norm(self):
        """目标帧 = 帧 i0+L 的归一化状态（与 E0Data 的 frm[:,L] 一致）。"""
        return (self.y - self.mo) / (self.so + 1e-8)


def rmse_w(pred_raw, tgt_raw):
    e = ((pred_raw[:, POS] - tgt_raw[:, POS]) ** 2) * W_POS
    return float(np.sqrt(e.mean()))


def main():
    selfcheck()
    out = dict(protocol=dict(spec=SPEC, L=L, K=K_EVAL, epochs=EPOCHS, seed=SEED,
                             lr=LR, batch=BATCH, action_index="a0=action[i0+L] (论文 E0Data 口径)",
                             metric="feat_weights-weighted position RMSE (px)"),
               conditions={})
    PLAN = [
        ("matched", "hinge", 900.0, "数据 g=900，先验 g=900（对照）"),
        ("g1125", "hinge", 1125.0, "数据 g=900，先验 g=1125（参数错 25%）"),
        ("drag", "drag", 900.0, "数据含阻尼+摩擦，先验无（未建模力）"),
    ]

    for cond, data, g_prior, note in PLAN:
        st, ct, ac = load_world(data)
        trk, vak = split(st.shape[0])
        mo = st[trk].reshape(-1, OBJ_DIMS).mean(axis=0)
        so = st[trk].reshape(-1, OBJ_DIMS).std(axis=0)
        Wtr = Win(st, ct, ac, trk, mo, so, g_prior)
        Wva = Win(st, ct, ac, vak, mo, so, g_prior)

        p_tr = Wtr.prior_step(Wtr.frames[:, -1].astype(np.float64),
                              Wtr.a0.astype(np.float64)).astype(np.float32)
        p_va = Wva.prior_step(Wva.frames[:, -1].astype(np.float64),
                              Wva.a0.astype(np.float64)).astype(np.float32)
        d_tr = (Wtr.target_norm() - p_tr).astype(np.float32)
        prior_one = rmse_w(p_va * (so + 1e-8) + mo, Wva.y)
        print(f"\n=== {cond}: {note}")
        print(f"    窗口 train={Wtr.n} val={Wva.n}  "
              f"先验单步(加权)={prior_one:.4f} px  "
              f"残差目标RMS(归一化)={float(np.sqrt((d_tr[:, POS] ** 2).mean())):.5f}")

        torch.manual_seed(SEED)
        model = WorldModelH(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=1,
                            hidden_dim=256, num_prototypes=32,
                            prior_spec=SPEC).to(DEV)
        model.set_stats(mo, so)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

        def assemble(frames, cts):
            seg = torch.from_numpy(
                ((frames - mo) / (so + 1e-8)).astype(np.float32)).to(DEV)
            cg = torch.from_numpy(cts.astype(np.float32)).to(DEV)
            return torch.cat([seg, cg], dim=2)

        Xtr = assemble(Wtr.frames, Wtr.cts)
        Atr = torch.from_numpy(Wtr.a0.astype(np.float32)).to(DEV)
        Dtr = torch.from_numpy(d_tr).to(DEV)
        n = Wtr.n
        for ep_i in range(1, EPOCHS + 1):
            perm = np.random.RandomState(SEED + ep_i).permutation(n)
            for j in range(0, n, BATCH):
                b = perm[j:j + BATCH]
                h, _ = model.encode_sequence(Xtr[b])
                loss = F.huber_loss(model.predict_state(h, Atr[b]), Dtr[b],
                                    reduction="mean")
                opt.zero_grad(); loss.backward(); opt.step()
            if ep_i % 20 == 0 or ep_i == EPOCHS:
                print(f"      ep{ep_i:3d} huber={float(loss.item()):.6f}", flush=True)

        Xva = assemble(Wva.frames, Wva.cts)
        with torch.no_grad():
            h, _ = model.encode_sequence(Xva)
            dl = model.predict_state(
                h, torch.from_numpy(Wva.a0.astype(np.float32)).to(DEV)).cpu().numpy()
        one_h = rmse_w((p_va + dl) * (so + 1e-8) + mo, Wva.y)

        def roll(use_head):
            cur = Wva.frames.copy()
            errs = []
            for k in range(K_EVAL):
                ak = Wva.af[:, k].astype(np.float64)
                pn = Wva.prior_step(cur[:, -1].astype(np.float64), ak)
                if use_head:
                    Xk = assemble(cur, Wva.cts)
                    with torch.no_grad():
                        hk, _ = model.encode_sequence(Xk)
                        dk = model.predict_state(
                            hk, torch.from_numpy(ak.astype(np.float32)).to(DEV)
                        ).cpu().numpy()
                    nxt = pn + dk
                else:
                    nxt = pn
                nxt_raw = nxt * (so + 1e-8) + mo
                errs.append(rmse_w(nxt_raw, Wva.fut[:, k]))
                cur = np.concatenate(
                    [cur[:, 1:], nxt_raw[:, None, :].astype(np.float32)], axis=1)
            return errs

        roll_p, roll_h = roll(False), roll(True)
        ratio = roll_h[-1] / roll_p[-1]
        verdict = "GO" if ratio <= 0.70 else ("PARTIAL" if ratio <= 0.95 else "NO-GO")
        out["conditions"][cond] = dict(
            note=note, data=data, g_prior=g_prior,
            prior_one_step_px=prior_one, hybrid_one_step_px=one_h,
            head_gain_one_step=1 - one_h / prior_one,
            prior_roll=roll_p, hybrid_roll=roll_h,
            ratio_roll_k5=ratio, head_gain_roll_k5=1 - ratio,
            verdict=verdict, n_train=int(n))
        print(f"    -> 单步: 先验 {prior_one:.4f} -> 混合 {one_h:.4f} "
              f"(增益 {100*(1-one_h/prior_one):+.1f}%)")
        print(f"    -> 闭环 k5: 先验 {roll_p[-1]:.4f} -> 混合 {roll_h[-1]:.4f} "
              f"(比值 {ratio:.3f}) == {verdict} ==")

    with open("analysis_output/e5_head_value.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/e5_head_value.json")
    print("\n=== 汇总 ===")
    for k, v in out["conditions"].items():
        print(f"  {k:9s} 闭环比值={v['ratio_roll_k5']:.3f}  "
              f"单步增益={100*v['head_gain_one_step']:+6.1f}%  {v['verdict']}")


if __name__ == "__main__":
    main()
