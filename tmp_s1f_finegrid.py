# -*- coding: utf-8 -*-
"""SLOT-1f 补充审计：按预注册分辨率 0.01 复审 τ 规则（W1-ii）。

只读补充：不替换 0.05 网格的 S1f_report.json，另存
analysis_output/S1f_finegrid_audit.json。复用 tmp_s1f_compare.py 的
eval_tau / 数据切分 / 窗口采样（seed 0、20000 窗、前半 A 后半 B），
网格 0.50--0.99 step 0.01，回答：
  - 注册分辨率下规则落点 tau_star(0.01) 是多少；
  - 该落点在 val-B 上是否达到全关（ff_all_silenced）与 FF 误差；
  - 每 seed 在 0.01 网格上最早的“全关”τ（判断边界刀口是否由粗网格造成）。
"""
import io
import json

import numpy as np
import torch

import tmp_s1f_compare as C

OUT = "analysis_output/S1f_finegrid_audit.json"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("载入 trajectories_v3.npz …", flush=True)
    states, obstacles, contact, actions = C.load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    taus = [round(0.50 + 0.01 * i, 2) for i in range(50)]
    out = {
        "note": "W1-ii 复审：预注册分辨率 0.01 的 τ 规则落点（审计补充，不替换 0.05 网格报告）",
        "tau_grid": "0.50--0.99 step 0.01",
        "source": "tmp_s1f_finegrid_audit: 与 tmp_s1f_compare.py 同一切分/采样（RS 0, 20000 窗）",
        "arms": {},
    }
    for arm in ("baseline", "F1", "F2"):
        out["arms"][arm] = {}
        for s in C.SEEDS:
            model, ck = C.load_model(C.ARMS[arm].format(s=s))
            a, ns = ck["args"], ck["norm_stats"]
            _, val_eps = C.split_episodes(N, s)
            ds = C.V3Data(states, obstacles, contact, actions, val_eps,
                          a["seq_len"], a["future_steps"],
                          ns["mean_obj"], ns["std_obj"], ns["mean_obs"],
                          ns["std_obs"], prior_spec=ck["config"]["prior_spec"])
            rng = np.random.RandomState(0)
            idx_all = np.arange(len(ds))
            if len(idx_all) > 20000:
                idx_all = np.sort(rng.choice(idx_all, size=20000, replace=False))
            half = len(idx_all) // 2
            idx_A, idx_B = idx_all[:half], idx_all[half:]
            model = model.to(device)
            row = {"n_A": int(len(idx_A)), "n_B": int(len(idx_B)),
                   "valA_fs": {}, "valB": {}}
            for t in taus:
                r_A = C.eval_tau(model, ds, device, idx_A, a["future_steps"], t)
                r_B = C.eval_tau(model, ds, device, idx_B, a["future_steps"], t)
                row["valA_fs"][str(t)] = float(r_A["false_silence"])
                row["valB"][str(t)] = {
                    "ff_rms": float(r_B["ff_rms"]),
                    "ff_all_silenced": float(r_B["ff_all_silenced"]),
                    "ff_median": float(r_B["ff_median"]),
                    "false_silence": float(r_B["false_silence"]),
                    "contact_rms": float(r_B["contact_rms"]),
                }
            ok = [t for t in taus if row["valA_fs"][str(t)] <= 0.02]
            tstar = max(ok) if ok else taus[0]
            row["tau_star_001"] = tstar
            row["valA_fs_at_tau_star"] = row["valA_fs"][str(tstar)]
            row["valB_at_tau_star"] = row["valB"][str(tstar)]
            floor_taus = [t for t in taus
                          if row["valB"][str(t)]["ff_all_silenced"] >= 1.0]
            row["valB_first_all_silenced_tau"] = floor_taus[0] if floor_taus else None
            out["arms"][arm][str(s)] = row
            print("%-9s s%s | tau*@0.01=%.2f fsA=%.4f | ffB@tau*=%.3e allSil=%s | firstAllSil@0.01=%s"
                  % (arm, s, tstar, row["valA_fs_at_tau_star"],
                     row["valB_at_tau_star"]["ff_rms"],
                     row["valB_at_tau_star"]["ff_all_silenced"],
                     row["valB_first_all_silenced_tau"]), flush=True)
            del model
            torch.cuda.empty_cache()
    io.open(OUT, "w", encoding="utf-8").write(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("[save] %s" % OUT)


if __name__ == "__main__":
    main()
