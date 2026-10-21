# -*- coding: utf-8 -*-
"""
M1e：rollout 反馈通路消融（零训练成本）
==================================================
假设（来自 M2b 两臂的确诊）：闭环 rollout k≥2 退化的根因是把"三爻头预测的
接触概率软值"反馈回下一帧输入——相态边界上软值误差复合放大。

同一 checkpoint 下对比三种反馈通路（只改推理期反馈，不改权重）：
  soft   = 三爻头接触概率软值（旧行为）
  hard   = 三爻头动爻硬判决（0/1，接触预测准确率 98.9%，理论上接近真值）
  freeze = 冻结最后观测接触（永不反馈自身误差，牺牲相态更新）

若 hard/freeze 显著优于 soft → 假设成立，重训时把训练反馈同步切换；
若三者接近 → 退化根因不在接触通路，回头查状态通道的复合误差。

用法：
  python eval_m1e_feedback.py
"""

import os
import json
import argparse
import numpy as np
import torch

from layout import OBJ_DIMS, OBS_DIMS, ASSEMBLED_DIM, ACTION_DIM, DT_V3, IMPULSE_MAX_V3
from model_v3 import WorldModelV3
from train_v3 import load_dataset, split_episodes, norm_stats, V3Data, pos_rmse


@torch.no_grad()
def eval_mode(model, ds, device, K, mode, max_windows):
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rng = np.random.RandomState(0)
        idx_all = np.sort(rng.choice(idx_all, size=max_windows, replace=False))
    M = len(idx_all)
    bs = 2048
    pred_roll = [np.zeros((M, OBJ_DIMS), dtype=np.float32) for _ in range(K)]
    for j in range(0, M, bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        xb = b["x"].to(device)
        afb = b["af"].to(device)
        obb = xb[:, -1, OBJ_DIMS:OBJ_DIMS + OBS_DIMS].contiguous()
        gua0 = b["gua"].to(device)
        abs_preds, _ = model.rollout(xb, afb, obb, gua0=gua0, contact_mode=mode)
        for k in range(K):
            pred_roll[k][j:j + len(idx)] = (abs_preds[k].float().cpu().numpy()
                                            * (ds.std_obj + 1e-8) + ds.mean_obj)
    out = {}
    for k in range(K):
        fut = ds.obj_norm[ds.ep_idx[idx_all], ds.win_start[idx_all] + ds.L + k] \
            * (ds.std_obj + 1e-8) + ds.mean_obj
        out[f"k={k + 1}"] = pos_rmse(pred_roll[k], fut)
    return out


def main():
    parser = argparse.ArgumentParser(description="M1e 反馈通路消融")
    parser.add_argument("--data_path", type=str, default="trajectories_v3.npz")
    parser.add_argument("--checkpoints", type=str, nargs="+",
                        default=["checkpoints-v3-gua/best.pt", "checkpoints-v3-nogua/best.pt"])
    parser.add_argument("--future_steps", type=int, default=5)
    parser.add_argument("--val_windows", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_report", type=str,
                        default="analysis_output/m1e_feedback_report.json")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 72)
    print("M1e：rollout 反馈通路消融（soft vs hard vs freeze，零训练成本）")
    print("=" * 72)

    states, obstacles, contact, actions = load_dataset(args.data_path)
    _, val_eps = split_episodes(states.shape[0])
    report = {}
    for ckpt_path in args.checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = dict(ckpt["config"])
        # [M2b 教训] config 可能静默失效（gu 臂权重里无 gua_emb）——以权重为准
        if "gua_emb.weight" not in ckpt["model_state_dict"]:
            cfg["use_gua"] = False
        cfg.setdefault("tau", DT_V3)
        cfg.setdefault("gravity", -900.0)
        cfg.setdefault("impulse_max", IMPULSE_MAX_V3)
        model = WorldModelV3(**cfg).to(device)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()
        ns = ckpt["norm_stats"]
        ds = V3Data(states, obstacles, contact, actions, val_eps[:400], 8, args.future_steps,
                    ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"])
        name = os.path.basename(os.path.dirname(ckpt_path))
        report[name] = {}
        print(f"\n[{name}] (epoch {ckpt.get('epoch', '?')}, "
              f"one-step {ckpt.get('val_metrics', {}).get('one_step', {}).get('模型', {}).get('all', '?')}px)")
        print(f"  {'反馈通路':<10}" + "".join(f"{'k=' + str(k + 1):>10}" for k in range(args.future_steps)))
        for mode in ["soft", "hard", "freeze"]:
            res = eval_mode(model, ds, device, args.future_steps, mode, args.val_windows)
            report[name][mode] = res
            print(f"  {mode:<12}" + "".join(f"{res[f'k={k + 1}']:>10.3f}" for k in range(args.future_steps)))

    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 报告已保存: {args.output_report}")
    print("判读：hard/freeze 任一在 k≥2 显著优于 soft → '软接触反馈是根因'假设成立，")
    print("      重训时把训练反馈同步切换为获胜模式；三者接近 → 根因在状态通道复合误差。")


if __name__ == "__main__":
    main()
