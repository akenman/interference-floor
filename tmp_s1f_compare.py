# -*- coding: utf-8 -*-
"""SLOT-1f 判决脚本：构造性静默（解码时判决式硬静默 + 指示器训练）。

预注册：experiments/prereg_S1f_constructive_silence.json
方法（解码时，部署可得）：窗口级硬静默 g=1[max_o σ(接触位) ≥ τ]；g=0 的步用纯解析先验。
τ 规则（预注册）：在 **val-A（前半 10000 窗）** 取满足"接触单元误关率 ≤2%"的最大 τ
（网格 0.50→0.99 步长 0.05）；**判决指标只在 val-B（后半）上算**（选择与判决分离）。
无掩码参照：同代码路径的 τ=0（等价自由 rollout，接触反馈同 soft 模式）。

判据（预注册顺序）：
  R1f 方法兑现（主）：val-B、规则 τ 下 FF-clean 闭环 k=3 ≤0.20px 且 3/3 seed
  R2f 不伤接触：接触 regime 闭环 k=3 ≤ 无掩码同 seed ×1.10
  R3f 指示器可训：F1/F2 自由单元保护率 ≥ 参照臂（方向性 3/3）
  R4f 曲线外推：训练臂在同等接触误关率（±0.5pp）下 FF RMS ≤ 参照臂
  R5f mixed 不劣化：mixed 闭环 k=3 ≤ 无掩码 ×1.10

用法：python tmp_s1f_compare.py [--val-windows 20000] [--arms ...] [--no-figure]
输出：analysis_output/S1f_report.json + figures/S1f_constructive_silence.png
"""
import argparse
import json

import numpy as np
import torch

from layout import OBJ_DIMS, OBS_DIMS
from tmp_s1b_compare import load_model, transfer_clean_mask, eval_model
from train_v3 import V3Data, load_dataset, split_episodes

SEEDS = [42, 43, 44]
ARMS = {"baseline": "checkpoints-S1b/euler001x5_s{s}/best.pt",
        "sgate": "checkpoints-S1d/sgate_s{s}/best.pt",
        "W2": "checkpoints-S1e/W2_s{s}/best.pt",
        "F1": "checkpoints-S1f/F1_s{s}/best.pt",
        "F2": "checkpoints-S1f/F2_s{s}/best.pt"}
TRAINED = ["F1", "F2"]
TAU_GRID = [round(0.50 + 0.05 * i, 2) for i in range(10)]     # 0.50 … 0.95
FALSE_SILENCE_CAP = 0.02
R1F_THRESHOLD = 0.20
R2F_MARGIN = 1.10
POS6 = [1, 2, 12, 13, 23, 24]


@torch.no_grad()
def rollout_silence(model, x_seq, actions_fut, obstacles_last, gua0, tau, bs=2048):
    """窗口级硬静默 rollout（τ=0 ⇒ 纯自由 rollout = 无掩码参照）。"""
    B = x_seq.size(0)
    h = torch.zeros(B, model.hidden_dim, device=x_seq.device)
    for t in range(x_seq.size(1)):
        h = model.gru_cell(x_seq[:, t], h)
    h = model.ln_h(h)
    obj_last = x_seq[:, -1, :model.obj_dim]
    K = actions_fut.size(1)
    cur_gua = gua0
    preds = []
    sil = torch.zeros(B, K, device=x_seq.device)
    probs = []
    for k in range(K):
        a_k = actions_fut[:, k]
        prior_k = model.prior_norm(obj_last, a_k)
        delta = model.predict_state(h, a_k, cur_gua)
        yao_log = model.predict_yao(h)
        ct = torch.sigmoid(yao_log[:, 0:3].float())
        silent = (ct.max(dim=1).values < tau).float()          # 1=关门（静默）
        sil[:, k] = silent
        probs.append(ct)
        obj_next = prior_k + (1.0 - silent).unsqueeze(-1) * delta
        preds.append(obj_next)
        x_next = model.assemble_next_input(obj_next, obstacles_last,
                                           ct * (1.0 - silent).unsqueeze(-1))
        h = model.ln_h(model.gru_cell(x_next, h))
        obj_last = obj_next
        if model.use_gua:
            cur_gua = model.gua_from_yao(yao_log.detach())
    return preds, sil, torch.stack(probs, dim=1)


@torch.no_grad()
def eval_tau(model, ds, device, idx, K, tau, bs=2048):
    """返回 val 子集上的 FF / 接触 / 单元级统计（含全关窗率）。"""
    M = len(idx)
    pred = [np.zeros((M, OBJ_DIMS), np.float32) for _ in range(K)]
    sil = np.zeros((M, K), bool)
    for j in range(0, M, bs):
        sub = idx[j:j + bs]
        b = ds.gather(sub)
        sl = slice(j, j + len(sub))
        xb = b["x"].to(device)
        obb = xb[:, -1, OBJ_DIMS:OBJ_DIMS + OBS_DIMS].contiguous()
        ps, s, _ = rollout_silence(model, xb, b["af"].to(device), obb,
                                   b["gua"].to(device), tau)
        for k in range(K):
            pred[k][sl] = ps[k].float().cpu().numpy() * (ds.std_obj + 1e-8) + ds.mean_obj
        sil[sl] = s.cpu().numpy().astype(bool)
    ep, t0 = ds.ep_idx[idx], ds.win_start[idx] + ds.L
    tgt = ds.obj_norm[ep, t0 + K - 1] * (ds.std_obj + 1e-8) + ds.mean_obj
    per_win = np.sqrt(((pred[K - 1][:, POS6] - tgt[:, POS6]) ** 2).mean(axis=1))
    ct_gt = ds.contact[ep[:, None], t0[:, None] + np.arange(K)[None, :]] > 0.5
    ff = transfer_clean_mask(ds, idx, K)
    ct_reg = ds.contact[ep, t0].max(axis=1) > 0.5
    cell_sil = np.repeat(sil[:, None, :], 3, axis=1).transpose(0, 2, 1)   # (M,K,3)
    free, ctc = ~ct_gt, ct_gt
    out = dict(
        n_ff=int(ff.sum()), n_contact=int(ct_reg.sum()),
        mixed_rms=float(np.sqrt((per_win ** 2).mean())),
        ff_rms=float(np.sqrt((per_win[ff] ** 2).mean())) if ff.any() else float("nan"),
        ff_median=float(np.median(per_win[ff])) if ff.any() else float("nan"),
        ff_all_silenced=float(sil[ff].all(axis=1).mean()) if ff.any() else float("nan"),
        contact_rms=float(np.sqrt((per_win[ct_reg] ** 2).mean())) if ct_reg.any()
        else float("nan"),
        protect_free=float(cell_sil[free].mean()) if free.any() else float("nan"),
        false_silence=float(cell_sil[ctc].mean()) if ctc.any() else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-windows", type=int, default=20000)
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("载入 trajectories_v3.npz …", flush=True)
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]

    res = {}
    for arm in arms:
        res[arm] = {}
        for s in SEEDS:
            try:
                model, ck = load_model(ARMS[arm].format(s=s))
            except FileNotFoundError:
                print(f"[缺] {ARMS[arm].format(s=s)} —— ITT 崩溃")
                res[arm][s] = None
                continue
            a, ns = ck["args"], ck["norm_stats"]
            _, val_eps = split_episodes(N, s)
            ds = V3Data(states, obstacles, contact, actions, val_eps,
                        a["seq_len"], a["future_steps"],
                        ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"],
                        prior_spec=ck["config"]["prior_spec"])
            rng = np.random.RandomState(0)
            idx_all = np.arange(len(ds))
            if args.val_windows and len(idx_all) > args.val_windows:
                idx_all = np.sort(rng.choice(idx_all, size=args.val_windows,
                                             replace=False))
            half = len(idx_all) // 2
            idx_A, idx_B = idx_all[:half], idx_all[half:]
            model = model.to(device)
            base = eval_model(model, ds, device, idx_all, a["future_steps"])
            row = dict(n_A=len(idx_A), n_B=len(idx_B),
                       mixed_roll_k3=base["roll_mixed"]["k=3"],
                       mixed_one=base["one_mixed"])
            # 无掩码参照（τ=0）与 τ 网格（选择集 A；曲线集 B 供 R4f）
            row["unmasked_B"] = eval_tau(model, ds, device, idx_B, K=a["future_steps"], tau=0.0)
            grid_A, grid_B = {}, {}
            for tau in TAU_GRID:
                grid_A[str(tau)] = eval_tau(model, ds, device, idx_A, a["future_steps"], tau)
                grid_B[str(tau)] = eval_tau(model, ds, device, idx_B, a["future_steps"], tau)
            row["grid_A"], row["grid_B"] = grid_A, grid_B
            # 预注册 τ 规则（只用 A 的接触单元误关率）
            ok = [t for t in TAU_GRID if grid_A[str(t)]["false_silence"] <= FALSE_SILENCE_CAP]
            tau_star = max(ok) if ok else TAU_GRID[0]
            row["tau_star"] = tau_star
            row["tau_star_rule_ok"] = bool(ok)
            row["verdict_B"] = eval_tau(model, ds, device, idx_B, a["future_steps"], tau_star)
            v, u = row["verdict_B"], row["unmasked_B"]
            row["R1f"] = bool(v["ff_rms"] <= R1F_THRESHOLD)
            row["R2f"] = bool(v["contact_rms"] <= u["contact_rms"] * R2F_MARGIN)
            row["R5f"] = bool(v["mixed_rms"] <= u["mixed_rms"] * R2F_MARGIN)
            res[arm][s] = row
            print(f"[{arm} s{s}] τ*={tau_star:.2f} ({'规则可行' if ok else '规则无解→取最小τ'})"
                  f" | B: FF {u['ff_rms']:.3f}→{v['ff_rms']:.4f}"
                  f"（全关窗 {v['ff_all_silenced']:.0%}）"
                  f" | 接触 {u['contact_rms']:.3f}→{v['contact_rms']:.3f}"
                  f"（×{v['contact_rms'] / max(u['contact_rms'], 1e-9):.2f}）"
                  f" | 单元 保护 {v['protect_free']:.1%} 误关 {v['false_silence']:.1%}"
                  f" | R1f {'[OK]' if row['R1f'] else '[FAIL]'}"
                  f" R2f {'[OK]' if row['R2f'] else '[FAIL]'}", flush=True)
            del model
            torch.cuda.empty_cache()

    def m(arm, key, sub=None):
        vals = []
        for s in SEEDS:
            r = res[arm].get(s) if isinstance(res[arm], dict) else None
            if not r:
                continue
            v = r.get(key, np.nan)
            if sub is not None:
                v = v.get(sub, np.nan) if isinstance(v, dict) else np.nan
            if np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    out = {"prereg": "experiments/prereg_S1f_constructive_silence.json",
           "tau_grid": TAU_GRID, "results": res}
    print("\n=== 判决（预注册顺序）===")
    # R1f：方法兑现（训练臂，3/3 seed）
    out["R1f"] = {a: dict(ff_rms=m(a, "verdict_B", "ff_rms"),
                          per_seed=[bool(res[a][s]["R1f"]) if res[a].get(s) else None
                                    for s in SEEDS],
                          verdict=bool(all(res[a][s] and res[a][s]["R1f"] for s in SEEDS)))
                  for a in arms}
    for a in [x for x in TRAINED if x in out["R1f"]]:
        c = out["R1f"][a]
        print(f"  R1f 方法兑现 [{a}] FF-B(τ*) = {c['ff_rms']:.4f}px | 3/3 seed: "
              f"{'[OK]' if c['verdict'] else '[FAIL]'} {c['per_seed']}")
    # R2f
    print("  R2f 不伤接触（各臂，≤无掩码×1.10）:")
    for a in arms:
        print(f"    [{a}] {m(a, 'verdict_B', 'contact_rms'):.3f} vs 无掩码 "
              f"{m(a, 'unmasked_B', 'contact_rms'):.3f} → "
              f"{'[OK]' if all(res[a][s] and res[a][s]['R2f'] for s in SEEDS) else '[FAIL]'}")
    # R3f 指示器可训
    print("  R3f 指示器可训（自由单元保护率，规则 τ 下）:")
    for a in arms:
        print(f"    [{a}] 保护 {m(a, 'verdict_B', 'protect_free'):.1%} | "
              f"误关 {m(a, 'verdict_B', 'false_silence'):.1%}")
    # R4f ROC 支配（同等误关率 ±0.5pp 下 FF RMS）
    def curve_at(arm, target_fs):
        best = (np.nan, np.nan)
        for s in SEEDS:
            r = res[arm].get(s) if isinstance(res[arm], dict) else None
            if not r:
                continue
            for t in TAU_GRID:
                g = r["grid_B"][str(t)]
                if abs(g["false_silence"] - target_fs) <= 0.005:
                    if not np.isfinite(best[0]) or g["ff_rms"] < best[0]:
                        best = (g["ff_rms"], g["false_silence"])
        return best
    ref_fs = m("baseline", "verdict_B", "false_silence")
    print(f"  R4f 曲线外推（matched 误关率 ≈{ref_fs:.1%}）:")
    out["R4f"] = {}
    for a in [x for x in TRAINED if x in res]:
        ffm, fsm = curve_at(a, ref_fs)
        ffb = curve_at("baseline", ref_fs)[0]
        out["R4f"][a] = dict(matched_false_silence=fsm, ff_rms=ffm,
                             baseline_ff_rms=ffb,
                             verdict=bool(np.isfinite(ffm) and np.isfinite(ffb)
                                          and ffm <= ffb))
        print(f"    [{a}] FF {ffm:.4f} vs baseline {ffb:.4f} → "
              f"{'[OK]' if out['R4f'][a]['verdict'] else '[FAIL]'}")
    # R5f
    print("  R5f mixed 不劣化:",
          {a: bool(all(res[a][s] and res[a][s]["R5f"] for s in SEEDS)) for a in arms})
    crash = sum(res[a][s] is None for a in arms for s in SEEDS) / (len(arms) * len(SEEDS))
    out["crash_rate_ITT"] = crash
    print(f"崩溃率(ITT): {crash:.0%}")

    with open("analysis_output/S1f_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/S1f_report.json")

    if not args.no_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            # 面板 1：各臂 τ 曲线（B 集）——FF RMS vs 误关率（ROC）
            for a in arms:
                xs, ys = [], []
                for s in SEEDS:
                    r = res[a].get(s) if isinstance(res[a], dict) else None
                    if not r:
                        continue
                    for t in TAU_GRID:
                        g = r["grid_B"][str(t)]
                        if np.isfinite(g["false_silence"]) and np.isfinite(g["ff_rms"]):
                            xs.append(g["false_silence"])
                            ys.append(g["ff_rms"])
                if xs:
                    o = np.argsort(xs)
                    axes[0].plot(np.array(xs)[o], np.array(ys)[o], marker=".", label=a)
            axes[0].axhline(R1F_THRESHOLD, ls="--", c="r", lw=0.8)
            axes[0].set_xlabel("contact-cell false-silence rate")
            axes[0].set_ylabel("FF-clean closed-loop k=3 (px)")
            axes[0].set_title("ROC: silence vs contact damage", fontsize=9)
            axes[0].legend(fontsize=7)
            # 面板 2：规则 τ 下的 FF RMS（B 集）+ 中位数
            labs = [a for a in arms]
            x = np.arange(len(labs))
            axes[1].bar(x - 0.2, [m(a, "unmasked_B", "ff_rms") for a in labs],
                        width=0.4, label="unmasked", color="#8C8C8C")
            axes[1].bar(x + 0.2, [m(a, "verdict_B", "ff_rms") for a in labs],
                        width=0.4, label="silenced (τ*)", color="#4C72B0")
            axes[1].axhline(R1F_THRESHOLD, ls="--", c="r", lw=0.8)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(labs, fontsize=8, rotation=20)
            axes[1].set_title("FF-clean closed-loop k=3", fontsize=9)
            axes[1].legend(fontsize=7)
            # 面板 3：接触代价
            axes[2].bar(x - 0.2, [m(a, "unmasked_B", "contact_rms") for a in labs],
                        width=0.4, label="unmasked", color="#8C8C8C")
            axes[2].bar(x + 0.2, [m(a, "verdict_B", "contact_rms") for a in labs],
                        width=0.4, label="silenced (τ*)", color="#C44E52")
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(labs, fontsize=8, rotation=20)
            axes[2].set_title("contact-regime closed-loop k=3", fontsize=9)
            axes[2].legend(fontsize=7)
            fig.suptitle("SLOT-1f: constructive silence "
                         "(τ* = max τ with contact false-silence ≤2% on val-A)")
            fig.tight_layout()
            fig.savefig("figures/S1f_constructive_silence.png", dpi=150)
            print("[save] figures/S1f_constructive_silence.png")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 出图失败（不影响判决）: {e}")


if __name__ == "__main__":
    main()
