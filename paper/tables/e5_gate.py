# -*- coding: utf-8 -*-
"""E5-M2c：解码门在 E5 的世界里帮还是伤？（把 E5 与主线 C4 焊起来）

问题
----
论文 §6.5 的解码规则：逐步看接触指示概率，> tau 才用残差头，否则退回解析先验。
它在弹球世界有用，因为头在**自由飞行窗口上制造噪声**（干扰地板）。

E5 的摆世界是同一套门的可迁移场景：接触通道把"手触地/手在靶区"标出来
（`layout_h.yao_labels_h` 把 contact[:,0] 写进 yao 第 0 通道），
而摆世界 96% 是自由飞行。于是有一个**可判定的预测**：

  * 若头在两个 regime 上都有益（E5 在摆世界的 gain 是 +89~97%），
    门只会**伤害**性能——它把 96% 窗口中的头关掉，而那些窗口头本来在帮忙。
  * 若门仍然有益，说明头在自由飞行窗口上净有害，与弹球世界一致。

这直接决定 v2 里该怎么写门的适用范围：它是"针对地板的对策"，
还是"通用改进"。

实现
----
对每个已训模型（E5 的 g1125 与 drag）：
  1. 逐窗口算 单步 与 闭环 k=5 的 :math:`w\\,(pred-tgt)^2`；
  2. 用模型的 yao 头第 0 通道（接触概率）作为门信号；
  3. 在 tau 网格上，按"接触概率 > tau 才用头"的规则得到门控后的 pooled 误差；
  4. 同时报"自由飞行窗口上门的关闭率"。

用法：python -X utf8 tmp_e5_gate.py
输出：analysis_output/e5_gate.json
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
TAUS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95)


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


def main():
    out = dict(protocol=dict(spec=SPEC, L=L, K=K_EVAL, seed=SEED,
                             taus=list(TAUS), rule="use head iff contact_p > tau",
                             gate_signal="model yao head channel 0 (hand-contact)"),
               conditions={})

    for cond, data, g_prior in (("g1125", "hinge", 1125.0),
                                ("drag", "drag", 900.0)):
        st, ct, ac = load_world(data)
        trk, vak = split(st.shape[0])
        mo = st[trk].reshape(-1, OBJ_DIMS).mean(axis=0)
        so = st[trk].reshape(-1, OBJ_DIMS).std(axis=0)
        T = st.shape[1]
        n_win = T - L - K_EVAL + 1
        ep = np.repeat(vak, n_win)
        i0 = np.tile(np.arange(n_win), len(vak))
        off = np.arange(L)
        frames = st[ep[:, None], i0[:, None] + off[None, :]]
        cts = ct[ep[:, None], i0[:, None] + off[None, :]]
        a0 = ac[ep, i0 + L]
        y = st[ep, i0 + L].astype(np.float64)
        fut = np.stack([st[ep, i0 + L + k] for k in range(K_EVAL)], axis=1)
        af = ac[ep[:, None], (i0 + L)[:, None] + np.arange(K_EVAL)[None, :]]
        n = len(ep)
        print(f"\n=== {cond}  val 窗口 {n}  "
              f"接触帧比例 {float((cts.max(axis=2) > 0.5).mean()):.4f}")

        # ---- 训练一个 E5 模型（与本条件的失配一致） ----
        torch.manual_seed(SEED)
        model = WorldModelH(input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=1,
                            hidden_dim=256, num_prototypes=32,
                            prior_spec=SPEC).to(DEV)
        model.set_stats(mo, so)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
        tr_ep = np.repeat(trk, n_win)
        tr_i0 = np.tile(np.arange(n_win), len(trk))
        tr_frames = st[tr_ep[:, None], tr_i0[:, None] + off[None, :]]
        tr_cts = ct[tr_ep[:, None], tr_i0[:, None] + off[None, :]]
        tr_a0 = ac[tr_ep, tr_i0 + L]
        p_tr = (prior_raw_g(tr_frames[:, -1].astype(np.float64),
                            tr_a0.astype(np.float64), g_prior) - mo) / (so + 1e-8)
        d_tr = ((st[tr_ep, tr_i0 + L] - mo) / (so + 1e-8)) - p_tr
        Xtr = torch.cat([torch.from_numpy(
            ((tr_frames - mo) / (so + 1e-8)).astype(np.float32)).to(DEV),
            torch.from_numpy(tr_cts.astype(np.float32)).to(DEV)], dim=2)
        Atr = torch.from_numpy(tr_a0.astype(np.float32)).to(DEV)
        Dtr = torch.from_numpy(d_tr.astype(np.float32)).to(DEV)
        m = len(tr_ep)
        for ep_i in range(1, EPOCHS + 1):
            perm = np.random.RandomState(SEED + ep_i).permutation(m)
            for j in range(0, m, BATCH):
                b = perm[j:j + BATCH]
                h, _ = model.encode_sequence(Xtr[b])
                loss = F.huber_loss(model.predict_state(h, Atr[b]), Dtr[b],
                                    reduction="mean")
                opt.zero_grad(); loss.backward(); opt.step()

        # ---- 逐窗口误差 + 门信号 ----
        p_va = (prior_raw_g(frames[:, -1].astype(np.float64),
                            a0.astype(np.float64), g_prior) - mo) / (so + 1e-8)
        Xva = torch.cat([torch.from_numpy(
            ((frames - mo) / (so + 1e-8)).astype(np.float32)).to(DEV),
            torch.from_numpy(cts.astype(np.float32)).to(DEV)], dim=2)
        with torch.no_grad():
            h, _ = model.encode_sequence(Xva)
            delta = model.predict_state(
                h, torch.from_numpy(a0.astype(np.float32)).to(DEV)).cpu().numpy()
            yao_log = model.predict_yao(h)
            contact_p = torch.sigmoid(yao_log[:, 0]).cpu().numpy()
        hyb = p_va + delta
        err_h = (((hyb * (so + 1e-8) + mo)[:, POS] - y[:, POS]) ** 2) * W_POS
        err_p = (((p_va * (so + 1e-8) + mo)[:, POS] - y[:, POS]) ** 2) * W_POS
        e_h, e_p = err_h.mean(axis=1), err_p.mean(axis=1)     # 逐窗口
        # 闭环 k=5（逐窗口）
        cur = frames.copy()
        roll_h = np.zeros(n); roll_p = np.zeros(n)
        for k in range(K_EVAL):
            ak = af[:, k].astype(np.float64)
            pn = (prior_raw_g(cur[:, -1].astype(np.float64), ak, g_prior)
                  - mo) / (so + 1e-8)
            Xk = torch.cat([torch.from_numpy(
                ((cur - mo) / (so + 1e-8)).astype(np.float32)).to(DEV),
                torch.from_numpy(cts.astype(np.float32)).to(DEV)], dim=2)
            with torch.no_grad():
                hk, _ = model.encode_sequence(Xk)
                dk = model.predict_state(
                    hk, torch.from_numpy(ak.astype(np.float32)).to(DEV)
                ).cpu().numpy()
            nxt_h = pn + dk
            nxt_p = pn
            rh = ((nxt_h * (so + 1e-8) + mo)[:, POS] - fut[:, k][:, POS]) ** 2 * W_POS
            rp = ((nxt_p * (so + 1e-8) + mo)[:, POS] - fut[:, k][:, POS]) ** 2 * W_POS
            if k == K_EVAL - 1:
                roll_h, roll_p = rh.mean(axis=1), rp.mean(axis=1)
            cur = np.concatenate(
                [cur[:, 1:], (nxt_h * (so + 1e-8) + mo)[:, None, :].astype(np.float32)],
                axis=1)

        ff = cts.max(axis=2).max(axis=1) < 0.5      # 窗口级自由飞行
        base_one = float(np.sqrt(e_h.mean()))
        base_p_one = float(np.sqrt(e_p.mean()))
        base_roll = float(np.sqrt(roll_h.mean()))
        base_p_roll = float(np.sqrt(roll_p.mean()))
        rows = []
        for tau in TAUS:
            use = contact_p > tau
            one = float(np.sqrt(np.where(use, e_h, e_p).mean()))
            k5 = float(np.sqrt(np.where(use, roll_h, roll_p).mean()))
            rows.append(dict(tau=tau, one=one, k5=k5,
                             closed_ff_frac=float((~use)[ff].mean()),
                             head_used_frac=float(use.mean())))
            print(f"  tau={tau:<5} one={one:.4f} k5={k5:.4f}  "
                  f"自由飞行窗口被关门 {100*(~use)[ff].mean():5.1f}%")
        out["conditions"][cond] = dict(
            data=data, g_prior=g_prior, n_windows=int(n),
            ff_window_frac=float(ff.mean()),
            ungated=dict(prior_one=base_p_one, hybrid_one=base_one,
                         prior_k5=base_p_roll, hybrid_k5=base_roll),
            grid=rows)

    with open("analysis_output/e5_gate.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/e5_gate.json")
    print("\n判读：若门控后 k5 比 ungated hybrid 更差 -> 门在这个世界是**有害**的")


if __name__ == "__main__":
    main()
