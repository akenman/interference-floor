# -*- coding: utf-8 -*-
"""核对：论文摆世界管线的残差目标对不对齐？

方法：用论文自己的 E0Data.gather 取一个 batch，取出它内部的
  prior_n（先验预测）、d_norm（残差目标）、目标帧 tgt
然后**用同一批窗口**，按物理正确对齐（a0 = action[i0+L-1]）重算先验，
比较两种口径下"先验 vs 目标帧"的误差。

物理正确的对齐：状态在帧 t、动作 u_t 把系统推进到帧 t+1。
论文 gather 的用法：状态取 frm[:, L-1]（帧 i0+L-1），
                   动作取 act_norm[ep, i0+L]（帧 i0+L 的动作），
                   目标取 frm[:, L]（帧 i0+L）。
→ 用"晚一步"的动作去预测"早一步"的帧。

用法：python -X utf8 tmp_align_verify.py
输出：analysis_output/align_verify.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import physics_prior_spec as P                       # noqa: E402
from e0_common import E0Data                         # noqa: E402
from layout_h import OBJ_DIMS, feat_weights          # noqa: E402

POS = [1, 2, 12, 13]
W = feat_weights(OBJ_DIMS)[POS]
L, K, SEED = 8, 5, 42


def rmse_w(pred_raw, tgt_raw):
    e = ((pred_raw[:, POS] - tgt_raw[:, POS]) ** 2) * W
    return float(np.sqrt(e.mean()))


def main():
    d = np.load("trajectories_hinge.npz")
    st, ct, ac, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = st.shape[0]
    segs = np.arange(N)
    np.random.RandomState(SEED).shuffle(segs)
    val = segs[:300]
    tr = segs[300:]
    obj_flat = st[tr].reshape(-1, OBJ_DIMS).astype(np.float64)
    mo, so = obj_flat.mean(0), obj_flat.std(0)

    for spec in ("rk4001x5", "euler005"):
        ds = E0Data(st, ct, ac, yao, val, L, K, mo, so, prior_spec=spec)
        idx = np.arange(0, len(ds), max(1, len(ds) // 20000))    # 抽样
        b = ds.gather(idx)
        ep, i0 = ds.ep_idx[idx], ds.win_start[idx]
        a0_paper = ac[ep, i0 + L]                       # 论文用的动作
        a0_correct = ac[ep, i0 + L - 1]                 # 物理正确的动作
        s_last = st[ep, i0 + L - 1].astype(np.float64)  # 状态（两版相同）
        tgt = st[ep, i0 + L].astype(np.float64)         # 目标帧（两版相同）
        tgt_next = st[ep, i0 + L + 1].astype(np.float64)

        pr_paper = P.prior_raw_np(spec, s_last, a0_paper.astype(np.float64))
        pr_corr = P.prior_raw_np(spec, s_last, a0_correct.astype(np.float64))

        rec = dict(
            spec=spec, n=int(len(idx)),
            # 论文口径
            paper_prior_vs_target=rmse_w(pr_paper, tgt),
            paper_target_is_next_frame=rmse_w(pr_paper, tgt_next),
            # 物理正确口径
            correct_prior_vs_target=rmse_w(pr_corr, tgt),
            # 论文自己算的 one_step_prior（应等于 paper_prior_vs_target 的量级）
            e0data_residual_target_rms_norm=float(
                np.sqrt((b["d_norm"][:, POS].numpy() ** 2).mean())),
        )
        rec["ratio_paper_over_correct"] = (
            rec["paper_prior_vs_target"] / rec["correct_prior_vs_target"])
        print(f"\n--- {spec}")
        print(f"  论文口径（动作取 i0+L）   : 先验 vs 帧 i0+L   = "
              f"{rec['paper_prior_vs_target']:.5f} px")
        print(f"  同一先验 vs 帧 i0+L+1     : {rec['paper_target_is_next_frame']:.5f} px")
        print(f"  物理正确（动作取 i0+L-1） : 先验 vs 帧 i0+L   = "
              f"{rec['correct_prior_vs_target']:.5f} px")
        print(f"  论文口径 / 正确口径       = {rec['ratio_paper_over_correct']:.2f}x")
        print(f"  E0Data 残差目标 RMS(归一化) = "
              f"{rec['e0data_residual_target_rms_norm']:.5f}")
        globals().setdefault("OUT", {})[spec] = rec

    with open("analysis_output/align_verify.json", "w", encoding="utf-8") as f:
        json.dump(OUT, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/align_verify.json")


if __name__ == "__main__":
    main()
