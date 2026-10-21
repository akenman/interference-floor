# -*- coding: utf-8 -*-
"""干扰地板：第二轮诊断（评审动作 5）。

第一轮结果（analysis_output/crosstalk_law.json）：
  P1 = Lip x rep_mismatch x target_contact_rms  →  中位比值 0.53，log-log r=+0.30
  P2 = 接触拟合线性代理外推到自由飞行            →  中位比值 1.96，log-log r=+0.26
两者都系统性偏离约 2 倍、且不能解释跨臂变化。=> 表示重叠/Lipschitz 不是控制量。

第二轮检验替代假说 H-R（损失几何，而非表示几何）：
  自由飞行误差对总损失的贡献极小（残差目标在自由飞行上是 0.011，在接触上是 0.024，
  但接触窗口上"未解释误差"的幅度远大于此），因此梯度几乎不作用于静默；
  静默的程度应由自由飞行损失权重与接触损失权重的比值 r = w_ff 控制。

  预测：log(head_ff_rms) ~ log(w_ff) 呈负斜率。

本脚本对每个 (arm, seed) 计算：
  * 观测到的自由飞行头输出 RMS
  * 损失质量比 r = mean(w_ff * 误差²_ff) / mean(误差²_contact)（用训练臂的残差目标近似）
  * 头的自由飞行方向杠杆 L_ff = ‖J_ff @ (h_ff − h̄_c)‖ / ‖h_ff − h̄_c‖
用法：python -X utf8 tmp_crosstalk_diag.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tmp_s1b_compare import load_model          # noqa: E402
from train_v3 import (load_dataset, split_episodes, norm_stats, V3Data,   # noqa: E402
                      POS_DIMS_OBJ)

SEEDS = (42, 43, 44)
ARMS = {
    "euler001x5": ("checkpoints-S1b/euler001x5_s{s}/best.pt", 1.0),
    "wff10": ("checkpoints-S1c/wff10_s{s}/best.pt", 10.0),
    "wff0": ("checkpoints-S1g/wff0_s{s}/best.pt", 0.0),
    "sgate": ("checkpoints-S1d/sgate_s{s}/best.pt", 1.0),
    "W1": ("checkpoints-S1e/W1_s{s}/best.pt", 1.0),
    "W2": ("checkpoints-S1e/W2_s{s}/best.pt", 1.0),
    "S": ("checkpoints-S1e/S_s{s}/best.pt", 1.0),
}
N_EP = 60
N_WIN = 3000
MAX_BATCH = 512
POS = POS_DIMS_OBJ


def jac_dir_norm(model, h, a, dirv):
    """|J @ d| / |d|：头沿方向 d 的方向导数幅度（逐样本）。"""
    hh = h.detach().clone().requires_grad_(True)
    out = model.predict_state(hh, a)
    tot = torch.zeros(out.shape[0], device=h.device)
    for j in range(out.shape[1]):
        g = torch.autograd.grad(out[:, j].sum(), hh, retain_graph=(j < out.shape[1] - 1))[0]
        tot += (g * dirv).sum(dim=1) ** 2
    return tot.sqrt().detach().cpu().numpy()


def run(model, ds, ff_idx, ct_idx, device):
    Hs_f, Hs_c, Ds_f, Ds_c, Ts_f, Ts_c = [], [], [], [], [], []
    for idx, tag in ((ff_idx, "f"), (ct_idx, "c")):
        for j in range(0, len(idx), MAX_BATCH):
            b = ds.gather(idx[j:j + MAX_BATCH])
            xb, ab = b["x"].to(device), b["a0"].to(device)
            h, _ = model.encode_sequence(xb)
            dn, delta = b["d_norm"].to(device), model.predict_state(h, ab)
            (Hs_f if tag == "f" else Hs_c).append(h.detach().cpu().numpy())
            (Ds_f if tag == "f" else Ds_c).append(delta[:, POS].detach().cpu().numpy())
            (Ts_f if tag == "f" else Ts_c).append(dn[:, POS].detach().cpu().numpy())
    Hf, Hc = np.concatenate(Hs_f), np.concatenate(Hs_c)
    Df, Dc = np.concatenate(Ds_f), np.concatenate(Ds_c)
    Tf, Tc = np.concatenate(Ts_f), np.concatenate(Ts_c)
    mu = Hc.mean(axis=0, keepdims=True)
    # 自由飞行方向（逐样本）
    Dh = Hf - mu
    dn_ = Dh / (np.linalg.norm(Dh, axis=1, keepdims=True) + 1e-12)
    # 只为前 256 个样本算方向杠杆（逐维 backward 很贵）
    k = min(256, len(dn_))
    hb = torch.from_numpy(Hf[:k]).to(device)
    ab = torch.from_numpy(np.zeros((k, 6), dtype=np.float32)).to(device)
    dirv = torch.from_numpy(dn_[:k].astype(np.float32)).to(device)
    lev = jac_dir_norm(model, hb, ab, dirv)
    return dict(
        head_ff_rms=float(np.sqrt((Df ** 2).mean())),
        head_ct_rms=float(np.sqrt((Dc ** 2).mean())),
        target_ff_rms=float(np.sqrt((Tf ** 2).mean())),
        target_ct_rms=float(np.sqrt((Tc ** 2).mean())),
        h_ff_std=float(np.sqrt((Dh ** 2).sum(axis=1).mean())),
        h_ct_std=float(np.sqrt(((Hc - mu) ** 2).sum(axis=1).mean())),
        lev_ff=float(lev.mean()),
        lev_x_hstd=float(lev.mean() * np.sqrt((Dh ** 2).sum(axis=1).mean())),
        ratio_targets=float(np.sqrt((Tf ** 2).mean()) / np.sqrt((Tc ** 2).mean())),
    )


def main():
    device = torch.device("cpu")
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    out = {}
    for seed in SEEDS:
        tr, va = split_episodes(N, seed)
        mo, so, moo, soo = norm_stats(states, obstacles, tr)
        sub = va[:N_EP]
        for arm, (tmpl, wff) in ARMS.items():
            path = tmpl.format(s=seed)
            if not os.path.exists(path):
                continue
            model, ck = load_model(path)
            ds = V3Data(states, obstacles, contact, actions, sub, 8, 3,
                        mo, so, moo, soo, prior_spec=ck["config"].get("prior_spec"))
            rng = np.random.RandomState(seed + 7)
            wf = ds.win_ff.reshape(-1) > 0.5
            ia = np.arange(len(ds))
            ff_idx = rng.choice(ia[wf], size=min(N_WIN, int(wf.sum())), replace=False)
            ct_idx = rng.choice(ia[~wf], size=min(N_WIN, len(ia) - int(wf.sum())),
                                replace=False)
            r = run(model, ds, ff_idx, ct_idx, device)
            r["w_ff"] = wff
            out[f"{arm}_s{seed}"] = r
            print(f"  {arm:10s} s{seed} w_ff={wff:5.1f} head_ff={r['head_ff_rms']:.4f} "
                  f"head_ct={r['head_ct_rms']:.4f} tgt_ff={r['target_ff_rms']:.4f} "
                  f"tgt_ct={r['target_ct_rms']:.4f} h_std_ff={r['h_ff_std']:.2f} "
                  f"lev_ff={r['lev_ff']:.4f} lev*hstd={r['lev_x_hstd']:.4f}")
            del ds, model
    with open("analysis_output/crosstalk_diag.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/crosstalk_diag.json")

    # ---- 假设检验 ----
    print("\n" + "=" * 90)
    keys = list(out)
    y = np.array([out[k]["head_ff_rms"] for k in keys])
    w = np.array([out[k]["w_ff"] for k in keys])
    lev = np.array([out[k]["lev_ff"] for k in keys])
    levh = np.array([out[k]["lev_x_hstd"] for k in keys])
    tgt = np.array([out[k]["target_ff_rms"] for k in keys])
    print("观测：head_ff_rms 与各候选量的关系（log-log）")
    for name, x in (("w_ff (损失权重)", w), ("lip_ff (方向杠杆)", lev),
                    ("lev x h_std", levh), ("target_ff_rms", tgt)):
        m = x > 0
        if m.sum() < 3:
            print(f"  {name:20s}: 样本不足（{m.sum()}）——含 0 的 w_ff=0 无法取对数")
            continue
        r = float(np.corrcoef(np.log(x[m]), np.log(y[m]))[0, 1])
        print(f"  {name:20s}: r={r:+.3f}  (n={m.sum()})")
    # w_ff=0 单独看
    m0 = w == 0
    if m0.any():
        print(f"  w_ff=0 臂（{m0.sum()} 个）：head_ff = "
              f"{[round(v,4) for v in y[m0]]}，w_ff=1 臂均值 "
              f"{y[w == 1].mean():.4f}")
    print(f"  head_ff 全局范围: [{y.min():.4f}, {y.max():.4f}]  "
          f"max/min = {y.max()/y.min():.2f}x")
    print(f"  target_ff 全局范围: [{tgt.min():.5f}, {tgt.max():.5f}]")


if __name__ == "__main__":
    main()
