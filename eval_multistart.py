# -*- coding: utf-8 -*-
"""多起点辨识：对 §4.3 的单起点 Nelder-Mead 结果做稳健性检验（审稿 W5）。

背景：paper §4.3 报告的 ĝ=705.7（真值 900）只来自单一起点 g=1200（OFF5）。
审稿要求：报告多起点统计，并说明该值是盆地还是全局最优。

本脚本与 eval_nmmulti.py 的 clean 分支使用**完全相同**的数据、辨识窗口
（make_id_tensors，seed=123，前 500 个 episode）、目标函数（nm_multistep，H=5）
与 Nelder-Mead 设置，唯一改变的是初始 g。因此结果可直接与 705.7 比较。

不训练神经网络，只做参数优化，不占用 GPU 训练队列。

用法：python -X utf8 eval_multistart.py
输出：analysis_output/multistart_clean.json
"""
import json
import math
import os

import numpy as np
import torch

from benchmark_surpass import OursParamModel
from layout_h import OBJ_DIMS, ACTION_DIM
from train_staged import P_KEYS, LOG_BOUNDS
from eval_nmmulti import nm_multistep, OFF5

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "trajectories_hinge.npz")
OUT = os.path.join(ROOT, "analysis_output", "multistart_clean.json")
H = 5
TRUE_G = 900.0
# g 的对数边界来自 LOG_BOUNDS（500..1500）；起点必须落在边界内，否则会被截断。
G_LO, G_HI = math.exp(LOG_BOUNDS[0][0]), math.exp(LOG_BOUNDS[0][1])
START_G = [600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0]


def load_clean():
    d = np.load(DATA)
    states = d["states"][:500].astype(np.float64)
    actions = d["actions"][:500]
    flat = states[100:].reshape(-1, OBJ_DIMS)
    mean_obj = flat.mean(0).astype(np.float32)
    std_obj = flat.std(0).astype(np.float32)
    return states, actions, mean_obj, std_obj


def main():
    device = torch.device("cpu")
    states, actions, mean_obj, std_obj = load_clean()
    rows = []
    for g0 in START_G:
        assert G_LO <= g0 <= G_HI, f"start g={g0} outside LOG_BOUNDS [{G_LO:.0f},{G_HI:.0f}]"
        off = dict(OFF5)
        off["g"] = g0
        m = OursParamModel(OBJ_DIMS + 3, OBJ_DIMS, ACTION_DIM, 256).to(device)
        m.set_stats(mean_obj, std_obj)
        m, res, nev = nm_multistep(m, P_KEYS, LOG_BOUNDS, off, states, actions,
                                   mean_obj, std_obj, H, device)
        p = m.get_params()
        row = dict(start_g=g0, end_g=float(p["g"]), rel_err_vs_true=float(p["g"] / TRUE_G - 1),
                   objective=float(res.fun), n_eval=int(nev), success=bool(res.success),
                   message=str(res.message),
                   params={k: float(p[k]) for k in P_KEYS})
        rows.append(row)
        print(f"start g={g0:6.0f} -> end g={p['g']:8.2f} "
              f"(err {row['rel_err_vs_true']:+.1%})  obj={res.fun:.6g}  nev={nev}")

    ends = np.array([r["end_g"] for r in rows])
    objs = np.array([r["objective"] for r in rows])
    best = int(np.argmin(objs))
    summary = dict(
        protocol="same data/windows/objective/NM as eval_nmmulti clean arm; only start g varies",
        H=H, true_g=TRUE_G, bounds_g=[G_LO, G_HI],
        starts=START_G, n_starts=len(rows),
        end_g_min=float(ends.min()), end_g_max=float(ends.max()),
        end_g_median=float(np.median(ends)),
        n_within_20pct_of_true=int(np.sum(np.abs(ends / TRUE_G - 1) <= 0.20)),
        best_objective_index=best, best_start_g=rows[best]["start_g"],
        best_end_g=rows[best]["end_g"],
        distinct_end_basins=int(len({round(x, 0) for x in ends})),
        note=("A single basin would show all starts converging to one end_g. Multiple distinct "
              "end values indicate local minima; the lowest objective is the best candidate found, "
              "not a certificate of global optimality."),
    )
    out = dict(summary=summary, rows=rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n[summary]", json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
