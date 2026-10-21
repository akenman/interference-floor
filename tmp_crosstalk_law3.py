# -*- coding: utf-8 -*-
"""干扰地板：第三轮——决定性检验（评审动作 5）。

已复现论文口径：clean-FF 掩码 73 窗口上 exact=0.01967 / legacy=0.02026，
与 S1b_probe_cached.json 的 0.01967 / 0.02026 五位一致（tmp_probe_replicate.py）。
采样口径（60 ep / 3000 窗）会给出 0.0166，故以下全部使用论文口径。

待检验的三个说法：

  A. "静默在表示上是可实现的"（论文 §6.1 的机制主张）
     若头在自由飞行输入上的局部增益远小于在接触输入上的增益，说明头已经能把
     两个 regime 分开——那么"不静默"就是损失优先级问题，而不是容量问题。
     量：L_ff = mean‖∂head/∂h‖_F on 73 clean-FF 窗口
         L_ct = mean‖∂head/∂h‖_F on 同规模的接触窗口
     预测：L_ff << L_ct。

  B. "噪声与先验档位无关"（论文 Table 2）
     同一 clean-FF 掩码上比较四个先验档的头输出。

  C. 损失权重的定量影响
     wff0（自由飞行梯度为 0）/ wff1（baseline）/ wff10（×10）在**同一批 73 窗口**
     上的头输出。

用法：python -X utf8 tmp_crosstalk_law3.py
输出：analysis_output/crosstalk_law3.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tmp_s1b_compare import load_model, transfer_clean_mask   # noqa: E402
from train_v3 import load_dataset, split_episodes, norm_stats, V3Data   # noqa: E402

POS = [1, 2, 12, 13, 23, 24]
MAX_BATCH = 1024
VAL_WINDOWS = 20000
SEED = 42
ARMS = {
    # 同一 clean-FF 掩码上的机制对照
    "euler001x5": "checkpoints-S1b/euler001x5_s42/best.pt",
    "wff0": "checkpoints-S1g/wff0_s42/best.pt",
    "wff10": "checkpoints-S1c/wff10_s42/best.pt",
    # 先验档位对照
    "parabola005": "checkpoints-S1b/parabola005_s42/best.pt",
    "euler005": "checkpoints-S1b/euler005_s42/best.pt",
    "rk4005": "checkpoints-S1b/rk4005_s42/best.pt",
}
WFF = {"euler001x5": 1.0, "wff0": 0.0, "wff10": 10.0,
       "parabola005": 1.0, "euler005": 1.0, "rk4005": 1.0}


def jac_frob(model, h, a):
    hh = h.detach().clone().requires_grad_(True)
    out = model.predict_state(hh, a)
    tot = torch.zeros(out.shape[0], device=h.device)
    for j in range(out.shape[1]):
        g = torch.autograd.grad(out[:, j].sum(), hh,
                                retain_graph=(j < out.shape[1] - 1))[0]
        tot += (g ** 2).sum(dim=1)
    return tot.sqrt().detach().cpu().numpy()


def stats_on(model, ds, idx, device, want_jac=False):
    sq_h, sq_t, js, heads = [], [], [], []
    for j in range(0, len(idx), MAX_BATCH):
        b = ds.gather(idx[j:j + MAX_BATCH])
        xb, ab = b["x"].to(device), b["a0"].to(device)
        h, _ = model.encode_sequence(xb)
        delta = model.predict_state(h, ab)
        dn = b["d_norm"].to(device)
        sq_h.append((delta[:, POS] ** 2).mean().item())
        sq_t.append((dn[:, POS] ** 2).mean().item())
        heads.append(delta[:, POS].detach().cpu().numpy())
        if want_jac:
            js.append(jac_frob(model, h, ab))
    out = dict(head_rms=float(np.sqrt(np.mean(sq_h))),
               target_rms=float(np.sqrt(np.mean(sq_t))),
               n=int(len(idx)))
    if want_jac:
        out["jac_frob_mean"] = float(np.concatenate(js).mean())
    return out


def main():
    device = torch.device("cpu")
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    tr, val_eps = split_episodes(N, SEED)
    mo, so, moo, soo = norm_stats(states, obstacles, tr)
    # 固定窗口集合：所有臂共用同一套 idx（保证可比）
    base, _ = load_model(ARMS["euler001x5"])
    ds0 = V3Data(states, obstacles, contact, actions, val_eps, 8, 3,
                 mo, so, moo, soo, prior_spec="euler001x5")
    rng = np.random.RandomState(0)
    idx_all = np.arange(len(ds0))
    idx_all = np.sort(rng.choice(idx_all, size=VAL_WINDOWS, replace=False))
    mask = transfer_clean_mask(ds0, idx_all, 1)
    clean_idx = idx_all[mask]
    # 同规模的接触窗口对照（按窗口级接触旗取，随机抽样到 |clean| 条）
    ep, t0 = ds0.ep_idx[idx_all], ds0.win_start[idx_all] + ds0.L - 1
    ct = ds0.contact[ep, t0].max(axis=1) > 0.5
    ct_idx = np.random.RandomState(1).choice(idx_all[ct], size=len(clean_idx),
                                             replace=False)
    del ds0, base

    report = {"probe": dict(n_clean=int(len(clean_idx)), n_contact_ctrl=int(len(ct_idx)))}
    for name, path in ARMS.items():
        model, ck = load_model(path)
        model = model.to(device)
        ds = V3Data(states, obstacles, contact, actions, val_eps, 8, 3,
                    mo, so, moo, soo, prior_spec=ck["config"]["prior_spec"])
        ff = stats_on(model, ds, clean_idx, device, want_jac=True)
        ct_ = stats_on(model, ds, ct_idx, device, want_jac=True)
        rec = dict(prior_spec=ck["config"]["prior_spec"], w_ff=WFF[name],
                   clean_ff=ff, contact_ctrl=ct_,
                   jac_ratio_ff_over_ct=ff["jac_frob_mean"] / ct_["jac_frob_mean"])
        report[name] = rec
        print(f"{name:12s} prior={rec['prior_spec']:12s} w_ff={WFF[name]:4.1f} | "
              f"cleanFF head={ff['head_rms']:.5f} target={ff['target_rms']:.7f} "
              f"J={ff['jac_frob_mean']:.4f} | contact head={ct_['head_rms']:.4f} "
              f"target={ct_['target_rms']:.4f} J={ct_['jac_frob_mean']:.4f} | "
              f"J_ff/J_ct={rec['jac_ratio_ff_over_ct']:.3f}")
        del ds, model

    with open("analysis_output/crosstalk_law3.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/crosstalk_law3.json")

    # ---- 结论判定 ----
    print("\n" + "=" * 76)
    ff_heads = {k: v["clean_ff"]["head_rms"] for k, v in report.items() if k != "probe"}
    jac_ratios = {k: v["jac_ratio_ff_over_ct"] for k, v in report.items() if k != "probe"}
    print("A. 静默的表示可实现性：J_ff / J_ct")
    for k, v in jac_ratios.items():
        print(f"   {k:12s} {v:.3f}")
    print("B. 先验档位无关性（clean-FF 头输出）")
    for k in ("euler001x5", "parabola005", "euler005", "rk4005"):
        if k in ff_heads:
            print(f"   {k:12s} {ff_heads[k]:.5f}")
    print("C. 损失权重的影响（clean-FF 头输出）")
    for k in ("wff0", "euler001x5", "wff10"):
        if k in ff_heads:
            print(f"   {k:12s} w_ff={WFF[k]:4.1f}  {ff_heads[k]:.5f}")
    if all(k in ff_heads for k in ("wff0", "euler001x5", "wff10")):
        b = ff_heads["euler001x5"]
        print(f"   wff0/wff1 = {ff_heads['wff0']/b:.2f}, "
              f"wff10/wff1 = {ff_heads['wff10']/b:.2f}")


if __name__ == "__main__":
    main()
