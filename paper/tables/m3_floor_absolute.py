# -*- coding: utf-8 -*-
"""M3（精简版）：干扰地板是**绝对常数**还是**跟着残差目标缩放**？

为什么这个问题是 M3 的核心
--------------------------
弹球世界测到：四个先验档上头在自由飞行窗口的输出都是 0.017~0.027（带内 1.2×），
而先验本身差五个数量级 —— 所以地板看起来是**绝对常数**，与先验无关。

在摆世界可以直接判定这一点，因为已训臂的残差目标相差 **100×**：

  spec         残差目标 RMS(归一化)   先验单步(px)   已训臂头增益
  rk4001x5     0.00040                0.0378        +0.4%   （几乎无可学）
  rk4005       中间                    0.1475        −14.6%
  euler005     0.04005                4.4831        +74.7%

两种可能的规律：
  (A) **绝对地板**：头的自由飞行输出是一个与先验无关的常数（弹球的结论）
      -> 那么它在 rk4001x5 上（目标 0.0004）应当**远大于**目标，比值 ~50×
  (B) **相对地板**：头的输出与残差目标同量级（即它会按需缩放）
      -> 那么 rk4001x5 上应当出现 0.0377−0.0378 ≈ 0 的差值

本脚本用与论文 §6.1 相同的口径测每个已训臂在**自由飞行窗口**上的头输出
（归一化 RMS 与像素 RMS 都给），并直接与目标比较。

用法：python -X utf8 tmp_m3_floor_absolute.py
输出：analysis_output/m3_floor_absolute.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import physics_prior_spec as P                                    # noqa: E402
from e0_common import E0Data                                       # noqa: E402
from layout_h import OBJ_DIMS, feat_weights                        # noqa: E402
from model_e0 import make_model, ARM_SPECS                         # noqa: E402

POS = [1, 2, 12, 13]
W_POS = feat_weights(OBJ_DIMS)[POS]
DEV = torch.device("cpu")
L, K, SEED = 8, 5, 42
MAXW = 30000


def free_flight_windows(ds, idx):
    """窗口级自由飞行：最后输入帧与目标帧的接触通道全 0。"""
    ep, i0 = ds.ep_idx[idx], ds.win_start[idx]
    c0 = ds.contact[ep, i0 + L - 1].max(axis=1) < 0.5
    c1 = ds.contact[ep, i0 + L].max(axis=1) < 0.5
    return c0 & c1


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
    spx = float(np.sqrt((so[POS] ** 2).mean()))     # 每归一化单位的像素尺度

    out = {}
    print(f"每归一化单位的像素尺度 sqrt(mean std^2) = {spx:.1f} px\n")
    print(f"{'arm':28s} {'target_norm':>11s} {'head_norm':>10s} "
          f"{'ratio':>8s} {'head_px':>9s}  n_ff")
    for ckpt in sorted(glob.glob("checkpoints-E0/*_s42/best.pt")):
        tag = os.path.basename(os.path.dirname(ckpt))
        arm, spec = tag.rsplit("_s", 1)[0], "euler005"
        if "@" in arm:
            arm, spec = arm.split("@", 1)
        if arm not in ARM_SPECS or ARM_SPECS[arm].get("cls") is None:
            continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = make_model(arm, prior_spec=spec).to(DEV)
        try:
            model.load_state_dict(ck["model_state_dict"])
        except Exception:
            continue
        if not getattr(model, "physics_prior", True):
            print(f"  skip {tag}: 无物理先验（残差目标不适用）")
            continue
        if not hasattr(model, "predict_state"):
            print(f"  skip {tag}: 无 predict_state")
            continue
        model.eval()
        model.set_stats(mo, so)
        ds = E0Data(st, ct, ac, yao, val, L, K, mo, so, prior_spec=spec)
        idx = np.arange(len(ds))
        if len(idx) > MAXW:
            idx = np.sort(np.random.RandomState(0).choice(idx, size=MAXW,
                                                          replace=False))
        ff = free_flight_windows(ds, idx)
        sub = idx[ff]
        if len(sub) == 0:
            continue
        b = ds.gather(sub)
        xb = b["x"].to(DEV)
        ab = b["a0"].to(DEV)
        with torch.no_grad():
            h, _ = model.encode_sequence(xb)
            delta = model.predict_state(h, ab).float().cpu().numpy()
        tgt = b["d_norm"].numpy()                       # 残差目标（归一化）
        hn = float(np.sqrt((delta[:, POS] ** 2).mean()))
        tn = float(np.sqrt((tgt[:, POS] ** 2).mean()))
        rec = dict(spec=spec, n_ff=int(len(sub)),
                   target_norm=tn, head_norm=hn,
                   ratio_head_over_target=(hn / tn if tn > 0 else float("inf")),
                   head_px=hn * spx, target_px=tn * spx)
        out[tag] = rec
        print(f"{tag:28s} {tn:11.5f} {hn:10.5f} "
              f"{rec['ratio_head_over_target']:8.2f} {rec['head_px']:9.4f}  {len(sub)}")

    with open("analysis_output/m3_floor_absolute.json", "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/m3_floor_absolute.json")
    print("\n判读：(A) 绝对地板 -> rk4001x5 的 ratio 应为 ~50x；"
          "(B) 相对地板 -> ratio 应 ~1")


if __name__ == "__main__":
    main()
