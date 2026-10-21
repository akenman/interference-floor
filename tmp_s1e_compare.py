# -*- coding: utf-8 -*-
"""SLOT-1e 判决脚本：闭环损失质量（判据以预注册为准）。

预注册：experiments/prereg_S1e_closed_loop_loss.json
新增臂（各 3 seed）：W1(--w_ff 1/--w_ff_roll 10)、W2(10/10)、S(100/1)；
参照臂零重训复用：baseline = S1b euler001x5、wff10 = S1c、sgate = S1d。
协议沿用 S1c/S1d：同 seed 配对（val 划分随 seed）+ ITT + 崩溃率。

判据（预注册顺序）：
  R1e 单步静默：FF-clean one-step ≤ 0.10px（0.10–0.25 记部分兑现）
  R2e 不伤接触：mixed one-step ≤ 0.628px
  R3e 闭环兑现（主判据）：FF-clean 闭环 k=3 ≤ 0.20px
  R4e 通道归因：W2 的相对改善 ≥ S 臂（方向性）
  R5e mixed 闭环不劣化：≤ baseline ×1.10
上限探针（报告件，非声称方法）：硬静默掩码（爻头预测接触=0 时跳过残差头）。

用法：python tmp_s1e_compare.py [--val-windows 20000] [--no-figure]
输出：控制台判决 + analysis_output/S1e_report.json + figures/S1e_closed_loop_loss.png
"""
import argparse
import json

import numpy as np
import torch

from layout import OBJ_DIMS, OBS_DIMS
from tmp_s1b_compare import load_model, transfer_clean_mask, eval_model
from tmp_s1c_compare import head_rms_on_ff
from train_v3 import V3Data, load_dataset, split_episodes

SEEDS = [42, 43, 44]
ARMS = {"baseline": "checkpoints-S1b/euler001x5_s{s}/best.pt",
        "wff10": "checkpoints-S1c/wff10_s{s}/best.pt",
        "sgate": "checkpoints-S1d/sgate_s{s}/best.pt",
        "W1": "checkpoints-S1e/W1_s{s}/best.pt",
        "W2": "checkpoints-S1e/W2_s{s}/best.pt",
        "S": "checkpoints-S1e/S_s{s}/best.pt"}
REFS = {"W1": "baseline", "W2": "wff10", "S": "wff10"}
# 曾因让位 GPU 中断的臂（2026-10-03）；S 已重训完成，故为空集。
NOT_RUN = set()
POS6 = [1, 2, 12, 13, 23, 24]
R1E, R2E, R3E, R5E_MARGIN = 0.10, 0.628, 0.20, 1.10


@torch.no_grad()
def rollout_hard_silence(model, x_seq, actions_fut, obstacles_last, gua0, bs=2048):
    """上限探针：爻头预测'无接触'的窗口直接跳过残差头（构造性静默）。

    部署可得（只用模型自身的预测接触，不需 oracle）；返回 K 步绝对状态。
    """
    B = x_seq.size(0)
    h = torch.zeros(B, model.hidden_dim, device=x_seq.device)
    for t in range(x_seq.size(1)):
        h = model.gru_cell(x_seq[:, t], h)
    h = model.ln_h(h)
    obj_last = x_seq[:, -1, :model.obj_dim]
    cf_last = x_seq[:, -1, -3:]
    preds = []
    cur_gua = gua0
    for k in range(actions_fut.size(1)):
        a_k = actions_fut[:, k]
        prior_k = model.prior_norm(obj_last, a_k)
        delta = model.predict_state(h, a_k, cur_gua)
        yao_log = model.predict_yao(h)
        ct_prob = torch.sigmoid(yao_log[:, 0:3].float())
        silent = (ct_prob.max(dim=1, keepdim=True).values < 0.5).float()   # 预测无接触
        obj_next = prior_k + (1.0 - silent) * delta
        preds.append(obj_next)
        contact_fb = ct_prob * (1.0 - silent)      # 静默窗口接触通道≈0（ct_prob 本身≈0）
        x_next = model.assemble_next_input(obj_next, obstacles_last, contact_fb)
        h = model.ln_h(model.gru_cell(x_next, h))
        obj_last = obj_next
        if model.use_gua:
            cur_gua = model.gua_from_yao(yao_log.detach())
    return preds


@torch.no_grad()
def eval_hard_silence(model, ds, device, idx_all, K):
    """上限探针的闭环 FF-clean k=3：RMS + 尾部统计（RMS 会被未关门的少数窗主导）。"""
    M = len(idx_all)
    pred = [np.zeros((M, OBJ_DIMS), np.float32) for _ in range(K)]
    for j in range(0, M, 2048):
        idx = idx_all[j:j + 2048]
        b = ds.gather(idx)
        sl = slice(j, j + len(idx))
        xb = b["x"].to(device)
        obb = xb[:, -1, OBJ_DIMS:OBJ_DIMS + OBS_DIMS].contiguous()
        out = rollout_hard_silence(model, xb, b["af"].to(device), obb,
                                   b["gua"].to(device))
        for k in range(K):
            pred[k][sl] = out[k].float().cpu().numpy() * (ds.std_obj + 1e-8) + ds.mean_obj
    ep = ds.ep_idx[idx_all]
    t0 = ds.win_start[idx_all] + ds.L
    cl = transfer_clean_mask(ds, idx_all, K)
    if not cl.any():
        return dict(rms=float("nan"), n=0)
    tgt = ds.obj_norm[ep, t0 + K - 1] * (ds.std_obj + 1e-8) + ds.mean_obj
    per_win = np.sqrt(((pred[K - 1][cl][:, POS6] - tgt[cl][:, POS6]) ** 2).mean(axis=1))
    return dict(rms=float(np.sqrt((per_win ** 2).mean())), n=int(len(per_win)),
                median=float(np.median(per_win)),
                frac_gt_0p5=float((per_win > 0.5).mean()),
                p90=float(np.percentile(per_win, 90)),
                n_unsilenced_tail=int((per_win > 0.5).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-windows", type=int, default=20000)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("载入 trajectories_v3.npz …", flush=True)
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]

    res = {}
    for arm, pat in ARMS.items():
        res[arm] = {}
        for s in SEEDS:
            path = pat.format(s=s)
            try:
                model, ck = load_model(path)
            except FileNotFoundError:
                if arm in NOT_RUN:
                    print(f"[缺] {path} —— 未跑完（让位 GPU，待重跑）")
                else:
                    print(f"[缺] {path} —— ITT 崩溃")
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
            model = model.to(device)
            r = eval_model(model, ds, device, idx_all, a["future_steps"])
            r["head_rms_ff"], r["n_ff"] = head_rms_on_ff(model, ds, device, idx_all)
            r["hard_silence"] = eval_hard_silence(model, ds, device, idx_all,
                                                  a["future_steps"])
            res[arm][s] = r
            hs = r["hard_silence"]
            print(f"[{arm} s{s}] mixed={r['one_mixed']:.3f}  FF={r['one_ff_clean']:.4f}"
                  f"  roll_ff k3={r['roll_ff_clean']['k=3']:.4f}"
                  f"  roll_mixed k3={r['roll_mixed']['k=3']:.3f}"
                  f"  头RMS={r['head_rms_ff']:.4f}"
                  f"  [上限探针 k3={hs['rms']:.4f} 中位={hs.get('median', float('nan')):.4f}"
                  f" 未关门 {hs.get('n_unsilenced_tail', 0)}/{hs.get('n', 0)}]", flush=True)
            del model
            torch.cuda.empty_cache()

    def vals(arm, key, sub=None):
        out = []
        for s in SEEDS:
            r = res[arm][s]
            if not r:
                out.append(np.nan)
                continue
            v = r.get(key, np.nan)
            if isinstance(v, dict):
                v = v.get(sub if sub else "k=3", np.nan)
            out.append(v)
        return np.array(out, dtype=float)

    def m(arm, key, sub=None):
        return float(np.nanmean(vals(arm, key, sub)))
    out = {"prereg": "experiments/prereg_S1e_closed_loop_loss.json", "results": res}
    base_mixed_roll = m("baseline", "roll_mixed")
    out["baseline"] = {"one_mixed": m("baseline", "one_mixed"),
                       "one_ff_clean": m("baseline", "one_ff_clean"),
                       "roll_ff_k3": m("baseline", "roll_ff_clean"),
                       "roll_mixed_k3": base_mixed_roll}
    print(f"\n基线: mixed={out['baseline']['one_mixed']:.3f} "
          f"FF={out['baseline']['one_ff_clean']:.4f} "
          f"roll_ff_k3={out['baseline']['roll_ff_k3']:.4f} "
          f"roll_mixed_k3={base_mixed_roll:.3f}")

    print("\n=== 判决（预注册顺序；各臂对照自身参照）===")
    out["verdicts"] = {}
    for arm in ("W1", "W2", "S"):
        if arm in NOT_RUN and any(res[arm][s] is None for s in SEEDS):
            out["verdicts"][arm] = {"not_run": True, "ref": REFS[arm]}
            print(f"  [{arm}] ⏳ 未跑完（让位 GPU，待重跑）——不判 R1e–R5e")
            continue
        ff, mx = m(arm, "one_ff_clean"), m(arm, "one_mixed")
        k3, k3m = m(arm, "roll_ff_clean"), m(arm, "roll_mixed")
        ref = REFS[arm]
        r1 = ff <= R1E
        partial = bool((not r1) and ff <= 0.25)
        r2 = mx <= R2E
        r3 = k3 <= R3E
        r5 = k3m <= base_mixed_roll * R5E_MARGIN
        d_ff = vals(arm, "one_ff_clean") - vals(ref, "one_ff_clean")
        d_k3 = vals(arm, "roll_ff_clean") - vals(ref, "roll_ff_clean")
        rel_k3 = float((m(ref, "roll_ff_clean") - k3) / m(ref, "roll_ff_clean"))
        out["verdicts"][arm] = dict(
            ref=ref, ff_one=ff, mixed=mx, roll_ff_k3=k3, roll_mixed_k3=k3m,
            ref_roll_ff_k3=m(ref, "roll_ff_clean"), rel_improve_k3=rel_k3,
            hard_silence_k3=m(arm, "hard_silence", "rms"),
            hard_silence_median=m(arm, "hard_silence", "median"),
            hard_silence_frac_gt_0p5=m(arm, "hard_silence", "frac_gt_0p5"),
            R1e=bool(r1), R1e_partial=partial, R2e=bool(r2), R3e=bool(r3), R5e=bool(r5),
            paired_delta_ff=d_ff.tolist(), paired_delta_roll_ff_k3=d_k3.tolist(),
            dir_improved_ff=f"{int(np.nansum(d_ff < 0))}/3",
            dir_improved_roll_ff_k3=f"{int(np.nansum(d_k3 < 0))}/3")
        print(f"  [{arm} vs {ref}] R1e 单步(≤{R1E}): {'[OK]' if r1 else ('◐' if partial else '[FAIL]')}"
              f" ({ff:.4f}) | R2e mixed(≤{R2E}): {'[OK]' if r2 else '[FAIL]'} ({mx:.3f}) | "
              f"R3e 闭环(≤{R3E}): {'[OK]' if r3 else '[FAIL]'} ({k3:.4f}) | "
              f"R5e mixed 闭环: {'[OK]' if r5 else '[FAIL]'} ({k3m:.3f})")
        print(f"        相对改善 {rel_k3:+.1%}（vs {ref} {m(ref, 'roll_ff_clean'):.4f}）"
              f" | 配对 Δff {np.round(d_ff, 4).tolist()} ({int(np.nansum(d_ff < 0))}/3)"
              f" | Δk3 {np.round(d_k3, 4).tolist()} ({int(np.nansum(d_k3 < 0))}/3)"
              f" | 上限探针 k3 {m(arm, 'hard_silence', 'rms'):.4f}"
              f"（中位 {m(arm, 'hard_silence', 'median'):.4f}，"
              f"未关门 {m(arm, 'hard_silence', 'frac_gt_0p5'):.1%}）")

    rel_w2 = out["verdicts"]["W2"]["rel_improve_k3"]
    rel_s = out["verdicts"]["S"].get("rel_improve_k3", float("nan"))
    if any(res["S"][s] is None for s in SEEDS):
        out["R4e"] = dict(rel_W2=rel_w2, rel_S=None, verdict=None,
                          note="S 臂未跑完（让位 GPU 待重跑）⇒ R4e 未判")
        print(f"\n  R4e 通道归因: ⏳ 未判（S 臂待重跑）；W2 相对改善 {rel_w2:+.1%}")
    else:
        out["R4e"] = dict(rel_W2=rel_w2, rel_S=rel_s, verdict=bool(rel_w2 >= rel_s))
        print(f"\n  R4e 通道归因（W2 相对改善 ≥ S）: "
              f"{'[OK]' if rel_w2 >= rel_s else '[FAIL]'}  (W2 {rel_w2:+.1%} vs S {rel_s:+.1%})")

    print("\n=== 上限探针（硬静默掩码，报告件非声称方法）===")
    for arm in ARMS:
        print(f"  [{arm}] 闭环 FF k=3：现状 {m(arm, 'roll_ff_clean'):.4f} → "
              f"硬静默 {m(arm, 'hard_silence', 'rms'):.4f}px"
              f"（中位 {m(arm, 'hard_silence', 'median'):.4f}，"
              f"窗误差>0.5px 占比 {m(arm, 'hard_silence', 'frac_gt_0p5'):.1%}）")

    ran = [a for a in ARMS if a not in NOT_RUN]
    crash = sum(res[a][s] is None for a in ran for s in SEEDS) / (len(ran) * len(SEEDS))
    out["crash_rate_ITT"] = crash
    out["arms_not_run"] = sorted(NOT_RUN)
    print(f"\n崩溃率(ITT，已跑臂): {crash:.0%} | 未跑臂: {', '.join(sorted(NOT_RUN)) or '无'}")
    with open("analysis_output/S1e_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/S1e_report.json")

    if not args.no_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            labels = ["baseline", "wff10", "sgate", "W1", "W2", "S"]
            keys = ["baseline", "wff10", "sgate", "W1", "W2", "S"]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, (lab, fn, thr) in zip(axes, (
                    ("FF-clean one-step (px)", lambda a: m(a, "one_ff_clean"), R1E),
                    ("FF-clean closed-loop k=3 (px)", lambda a: m(a, "roll_ff_clean"), R3E),
                    ("hard-silence probe k=3 (px)",
                     lambda a: m(a, "hard_silence", "rms"), R3E))):
                v = [fn(a) for a in keys]
                ax.bar(np.arange(len(labels)), v,
                       color=["#8C8C8C", "#4C72B0", "#C44E52", "#55A868",
                              "#8172B2", "#CCB974"])
                ax.axhline(thr, ls="--", c="r", lw=0.8)
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels, fontsize=8, rotation=20)
                ax.set_title(lab, fontsize=9)
            fig.suptitle("SLOT-1e: closed-loop loss quality "
                         "(dashed = preregistered R1e/R3e thresholds)")
            fig.tight_layout()
            fig.savefig("figures/S1e_closed_loop_loss.png", dpi=150)
            print("[save] figures/S1e_closed_loop_loss.png")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 出图失败（不影响判决）: {e}")


if __name__ == "__main__":
    main()
