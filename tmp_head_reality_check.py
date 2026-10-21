# -*- coding: utf-8 -*-
"""校验：论文那批已训练臂的"头"到底有没有真实增益？（E5 校准）

起因：E5 的 matched 条件（先验与数据同参数）显示头把单步误差大幅降低，而论文
§4.4 说"精确先验下头没有可学的东西（0.037 vs 0.038）"。两者矛盾，必须先判定
是 E5 口径错，还是论文那句被自己的数据反驳。

方法：读论文自己的 checkpoint（E0），在**完全相同的窗口**上算三个量的原始像素
位置 RMSE：prior（纯先验）、hybrid（先验+已训练头）、target（真值）。
口径与 train_hinge.evaluate 一致：val 前 10%、L=8、K=5、位置维 POS。

用法：python -X utf8 tmp_head_reality_check.py
输出：analysis_output/head_reality_check.json
"""
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e0_common import E0Data, validate, make_datasets   # noqa: E402
from model_e0 import make_model, ARM_SPECS            # noqa: E402
from layout_h import OBJ_DIMS                        # noqa: E402

POS = [1, 2, 12, 13]
DEV = torch.device("cpu")
L, K, SEED = 8, 5, 42


def main():
    d = np.load("trajectories_hinge.npz")
    states, contact, actions, yao = d["states"], d["contact"], d["actions"], d["yao"]
    N = states.shape[0]
    segs = np.arange(N)
    np.random.RandomState(SEED).shuffle(segs)
    val_eps = segs[:int(N * 0.1)]
    tr_eps = segs[int(N * 0.1):]
    obj_flat = states[tr_eps].reshape(-1, OBJ_DIMS).astype(np.float64)
    mean_obj = obj_flat.mean(0).astype(np.float32)
    std_obj = obj_flat.std(0).astype(np.float32)

    def pos_rmse(p, t):
        return float(np.sqrt(((p[:, POS] - t[:, POS]) ** 2).mean()))

    print(f"val 片段 {len(val_eps)}  位置维 std = "
          f"{[round(float(std_obj[i]), 1) for i in POS]}")

    rows = {}
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
        except Exception as e:
            print(f"  skip {tag}: {e}")
            continue
        model.eval()
        model.set_stats(mean_obj, std_obj)

        ds = E0Data(states, contact, actions, yao, val_eps, L, K,
                    mean_obj, std_obj, prior_spec=spec)
        vm = validate(model, ds, DEV, K, max_windows=30000)
        pr = float(vm["one_step_prior"])
        hy = float(vm["one_step_model"])
        rows[tag] = dict(spec=spec, prior=pr, hybrid=hy,
                         head_gain=1 - hy / pr,
                         roll_k5=float(vm["rollout"]["k=5"]),
                         n=int(len(ds)))
        print(f"  {tag:28s} spec={spec:10s} prior={pr:8.4f}  hybrid={hy:8.4f}  "
              f"head gain={100*(1-hy/pr):+7.1f}%  roll5={vm['rollout']['k=5']:.4f}")

    with open("analysis_output/head_reality_check.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/head_reality_check.json")
    print("论文 §4.4 的说法：精确先验下 头 0.037 vs 先验 0.038（无可学）")


if __name__ == "__main__":
    main()
