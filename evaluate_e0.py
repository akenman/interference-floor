# -*- coding: utf-8 -*-
"""
E0 统一评估 (evaluate_e0.py) — Step 4
==================================================
对 checkpoints-E0/<arm>_s<seed>/best.pt 跑统一协议，产出
analysis_output/E0_raw.json（臂 × 种子 × 指标，含每卦分解）。

协议（与训练期 validate 同一实现，e0_common.validate）：
  one-step RMSE（位置 4 维，feat_weights 加权）
  rollout k=1..5 闭环（接触反馈 = 爻头初爻软概率）
  爻准确率（3 轴分别 + 全对）
  卦覆盖 occupancy
  （A6 另跑 CQR 覆盖）

用法：python evaluate_e0.py [--all] [--arm ours_base] [--seed 42]
"""

import os, json, argparse, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from e0_common import validate, load_checkpoint, pos_rmse_raw, INPUT_DIM
import model_e0

CKPT_ROOT = "checkpoints-E0"
ARMS = list(model_e0.ARM_SPECS)
SEEDS = [42, 43, 44]
POS_IDX = [1, 2, 12, 13]
K = 5


def eval_checkpoint(path, device):
    model, arm, ck = load_checkpoint(path, device)
    # 用训练数据本身的验证集划分（与训练期 validate 一致）
    from e0_common import make_datasets
    data_path = "trajectories_hinge.npz"
    _, val_ds, _, _ = make_datasets(data_path, 8, K, ck["config"]["seed"])
    m = validate(model, val_ds, device, K, 30000)
    return arm, ck["config"]["seed"], m


@torch.no_grad()
def eval_cqr(path, device, nominal=0.80):
    """A6 ours+sigma：CQR 名义 80% 分位数带的实际覆盖率。
    覆盖 = 真值落入 [q0.1, q0.9] 的比例（位置 4 维逐维平均）。"""
    from e0_common import make_datasets
    model, arm, ck = load_checkpoint(path, device)
    assert arm == "ours_sigma", f"仅 ours_sigma 支持 CQR 评估（arm={arm}）"
    _, val_ds, _, _ = make_datasets("trajectories_hinge.npz", 8, K, ck["config"]["seed"])
    model.eval()
    idx = np.arange(len(val_ds))
    if len(idx) > 20000:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(idx, size=20000, replace=False))
    bs = 2048
    cov = 0; n = 0
    for j in range(0, len(idx), bs):
        b = val_ds.gather(idx[j:j + bs])
        xb = b["x"].to(device); ab = b["a0"].to(device)
        h, _ = model.encode_sequence(xb)
        mean = model.predict_state(h, ab)
        sigma = model.predict_sigma(h, ab)
        z = float(torch.erfinv(torch.tensor(nominal, device=device)) * (2 ** 0.5))
        lo, hi = mean - z * sigma, mean + z * sigma
        tgt = (b["x_next"][:, -1, :22] * (val_ds.std_obj + 1e-8) + val_ds.mean_obj).to(device)
        inside = ((tgt >= lo) & (tgt <= hi)).float()[:, POS_IDX].mean()
        cov += float(inside.sum()); n += int(inside.size(0))
    return {"cqr_nominal": nominal, "cqr_coverage": cov / max(n, 1), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    arms = [args.arm] if args.arm else ARMS
    seeds = [args.seed] if args.seed else SEEDS

    results = {}
    for arm in arms:
        spec = model_e0.ARM_SPECS[arm]
        if not spec["trainable"]:
            continue
        for seed in seeds:
            path = os.path.join(CKPT_ROOT, f"{arm}_s{seed}", "best.pt")
            if not os.path.exists(path):
                print(f"[SKIP] 缺失 {path}")
                continue
            a, s, m = eval_checkpoint(path, device)
            rec = dict(arm=a, seed=s, params=spec["hidden"],
                       metrics={k: v for k, v in m.items() if k not in ("gua_rmse", "gua_occupancy")},
                       gua_rmse=m["gua_rmse"], gua_occupancy=m["gua_occupancy"],
                       n_windows=m["n_windows"])
            if arm == "ours_sigma":
                rec["cqr"] = eval_cqr(path, device)
            results[f"{a}_s{s}"] = rec
            print(f"[OK] {a} seed={s}: one-step={m['one_step_model']:.3f} "
                  f"prior={m['one_step_prior']:.3f} k5={m['rollout']['k=5']:.3f} "
                  f"yao={m['yao_acc']:.3f}", flush=True)

    os.makedirs("analysis_output", exist_ok=True)
    with open("analysis_output/E0_raw.json", "w", encoding="utf-8") as f:
        json.dump(dict(protocol=dict(world="hinge", L=8, K=5, data="trajectories_hinge.npz (val split)",
                                     seeds=SEEDS, pos_idx=POS_IDX, weighted=True),
                       results=results), f, indent=2, ensure_ascii=False)
    print(f"[OK] analysis_output/E0_raw.json ({len(results)} records)")


if __name__ == "__main__":
    main()