# -*- coding: utf-8 -*-
"""逐窗口误差直方图 + 静默代价分解（回答：2% 静默率能否产生 1.12--1.21× 接触损伤）。

完全复用 S1f 的评估代码路径（`tmp_s1f_compare.rollout_silence` + `train_v3` 的
`V3Data/split_episodes` 窗口构造），把 `eval_tau` 的**逐窗口**数组落盘。

恒等式（kept 窗口在有/无门控下逐位相同 ⇒ 差值只由被静默窗口贡献）：
    pooled_masked^2 - pooled_unmasked^2
        = (n_sil / n_contact) * ( rms_sil_without_head^2 - rms_sil_with_head^2 )

自检：复现的 contact_rms / false_silence 必须与 `S1f_report.json` 逐位一致。

用法：python paper/tables/build_silence_hist.py [--rebuild]
写：analysis_output/silence_hist.json
    analysis_output/_silence_hist_cache.npz（逐窗口数组缓存，避免重复 7 分钟评估）
    paper/figures/fig6_silence_hist.{pdf,png}
"""
import argparse
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "paper", "figures"))

import torch  # noqa: E402
from layout import OBJ_DIMS, OBS_DIMS  # noqa: E402
from tmp_s1f_compare import ARMS, POS6, rollout_silence  # noqa: E402
from tmp_s1b_compare import load_model, transfer_clean_mask  # noqa: E402
from train_v3 import V3Data, load_dataset, split_episodes  # noqa: E402

J = lambda p: json.load(io.open(os.path.join(ROOT, "analysis_output", p), encoding="utf-8"))
CACHE = os.path.join(ROOT, "analysis_output", "_silence_hist_cache.npz")


@torch.no_grad()
def eval_windows(model, ds, device, idx, K, tau, bs=2048):
    """与 tmp_s1f_compare.eval_tau 同一构造，但返回逐窗口数组（不聚合）。"""
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
    return dict(per_win=per_win, sil=sil, ff=ff, ct_reg=ct_reg,
                contact_rms=float(np.sqrt((per_win[ct_reg] ** 2).mean())),
                ff_rms=float(np.sqrt((per_win[ff] ** 2).mean())),
                false_silence=float(cell_sil[~ct_gt].mean() * 0 + cell_sil[ct_gt].mean()))


def evaluate(arm, seed, report, device, val_windows=20000):
    """跑一条 (arm, seed)：返回逐窗口数组 + 汇总。"""
    rep = report[arm][str(seed)]
    tau_star = float(rep["tau_star"])
    ckpt = ARMS[arm].format(s=seed)
    model, ck = load_model(ckpt)
    a, ns = ck["args"], ck["norm_stats"]
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    _, val_eps = split_episodes(N, seed)
    ds = V3Data(states, obstacles, contact, actions, val_eps,
                a["seq_len"], a["future_steps"],
                ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"],
                prior_spec=ck["config"]["prior_spec"])
    rng = np.random.RandomState(0)
    idx_all = np.arange(len(ds))
    if val_windows and len(idx_all) > val_windows:
        idx_all = np.sort(rng.choice(idx_all, size=val_windows, replace=False))
    idx_B = idx_all[len(idx_all) // 2:]          # val-B（判决集）
    model = model.to(device)
    K = a["future_steps"]
    un = eval_windows(model, ds, device, idx_B, K, tau=0.0)
    ma = eval_windows(model, ds, device, idx_B, K, tau=tau_star)

    stored = rep["verdict_B"]["contact_rms"]
    if abs(ma["contact_rms"] - stored) > 1e-9:
        raise SystemExit("MISMATCH %s s%d contact_rms: %.12f vs stored %.12f"
                         % (arm, seed, ma["contact_rms"], stored))
    if abs(ma["false_silence"] - rep["verdict_B"]["false_silence"]) > 1e-9:
        raise SystemExit("MISMATCH %s s%d false_silence" % (arm, seed))

    m = ma["ct_reg"]
    touched = ma["sil"][m].any(axis=1)
    return dict(arm=arm, seed=seed, tau_star=tau_star, checkpoint=ckpt,
                per_win_masked=ma["per_win"][m], per_win_unmasked=un["per_win"][m],
                touched=touched,
                contact_rms_unmasked=un["contact_rms"], contact_rms_masked=ma["contact_rms"],
                ff_rms_unmasked=un["ff_rms"], ff_rms_masked=ma["ff_rms"],
                false_silence=ma["false_silence"])


def summarise(r):
    m, touched = r["per_win_masked"], r["touched"]
    e_ma, e_un = m[touched], r["per_win_unmasked"][touched]
    n_c, n_sil = int(m.size), int(touched.sum())

    def st(x):
        return dict(n=int(x.size), rms=float(np.sqrt((x ** 2).mean())),
                    median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                    max=float(x.max()))

    lhs = r["contact_rms_masked"] ** 2 - r["contact_rms_unmasked"] ** 2
    f = n_sil / n_c
    rhs = f * (float(np.sqrt((e_ma ** 2).mean())) ** 2 - float(np.sqrt((e_un ** 2).mean())) ** 2)
    return dict(arm=r["arm"], seed=r["seed"], tau_star=r["tau_star"],
                checkpoint=r["checkpoint"], n_contact_windows=n_c,
                n_silenced_windows=n_sil, frac_windows_silenced=f,
                false_silence_cells=r["false_silence"],
                contact_rms_unmasked=r["contact_rms_unmasked"],
                contact_rms_masked=r["contact_rms_masked"],
                ratio=r["contact_rms_masked"] / r["contact_rms_unmasked"],
                ff_rms_unmasked=r["ff_rms_unmasked"], ff_rms_masked=r["ff_rms_masked"],
                kept=st(m[~touched]), silenced_with_head=st(e_un),
                silenced_without_head=st(e_ma),
                identity=dict(lhs=float(lhs), rhs=float(rhs),
                              residual=float(abs(lhs - rhs))),
                pool_ratio_silenced_over_pool=float(
                    np.sqrt((e_un ** 2).mean()) / r["contact_rms_unmasked"]))


def figure(rows, outbase):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5,
                         "axes.titlesize": 9, "axes.labelsize": 8.5,
                         "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.dpi": 300})
    C = {"blue": "#4C72B0", "red": "#C44E52", "grey": "#8C8C8C"}
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
    # (a) 逐窗口误差直方图：全体接触窗口 vs 被静默窗口
    allm = np.concatenate([r["per_win_masked"] for r in rows])
    sil = np.concatenate([r["per_win_masked"][r["touched"]] for r in rows])
    bins = np.logspace(np.log10(0.05), np.log10(60), 46)
    pool_rms = float(np.sqrt((allm ** 2).mean()))
    sil_rms = float(np.sqrt((sil ** 2).mean()))
    ax[0].hist(allm, bins=bins, color=C["grey"], alpha=.75,
               label="all contact windows (RMS %.2f px)" % pool_rms)
    ax[0].hist(sil, bins=bins, color=C["red"], alpha=.75,
               label="silenced by the rule (RMS %.2f px)" % sil_rms)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].axvline(pool_rms, ls="--", c="k", lw=.8)
    ax[0].axvline(sil_rms, ls="--", c=C["red"], lw=.8)
    ax[0].set_xlabel("closed-loop $k{=}3$ error per contact window (px, log)")
    ax[0].set_ylabel("windows (log)")
    ax[0].set_title("(a) the silenced windows are not typical")
    ax[0].legend(loc="lower left", framealpha=.9)
    # (b) 被静默窗口：有头 vs 无头
    xs = np.concatenate([r["per_win_unmasked"][r["touched"]] for r in rows])
    ys = np.concatenate([r["per_win_masked"][r["touched"]] for r in rows])
    ax[1].scatter(xs, ys, s=3, alpha=.25, color=C["red"], edgecolors="none")
    lim = [max(1e-2, min(xs.min(), ys.min())), max(xs.max(), ys.max()) * 1.2]
    ax[1].plot(lim, lim, ls="--", c="k", lw=.8)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("error with the head (px)")
    ax[1].set_ylabel("error with the head skipped (px)")
    ax[1].set_title("(b) what skipping the head costs, per window")
    ax[1].text(.03, .95, "n=%d silenced windows\nRMS %.2f$\\to$%.2f px"
               % (xs.size, float(np.sqrt((xs ** 2).mean())), float(np.sqrt((ys ** 2).mean()))),
               transform=ax[1].transAxes, va="top", fontsize=6.5)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(outbase + ext, bbox_inches="tight")
    print("[save]", outbase + ".pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="baseline,F1,F2")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--val-windows", type=int, default=20000)
    ap.add_argument("--rebuild", action="store_true", help="忽略缓存重新评估")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = J("S1f_report.json")["results"]
    combos = [(a, int(s)) for a in args.arms.split(",") for s in args.seeds.split(",")]

    raws = []
    if os.path.exists(CACHE) and not args.rebuild:
        z = np.load(CACHE, allow_pickle=True)
        have = {(str(k), int(s)) for k, s in zip(z["arms"], z["seeds"])}
        if have == set(combos):
            print("[cache]", CACHE)
            for i, (a, s) in enumerate(combos):
                raws.append(dict(arm=a, seed=s, tau_star=float(z["tau_star"][i]),
                                 checkpoint=str(z["ckpt"][i]),
                                 per_win_masked=z["pw_ma_%d" % i],
                                 per_win_unmasked=z["pw_un_%d" % i],
                                 touched=z["touched_%d" % i].astype(bool),
                                 contact_rms_unmasked=float(z["crm_un"][i]),
                                 contact_rms_masked=float(z["crm_ma"][i]),
                                 ff_rms_unmasked=float(z["ff_un"][i]),
                                 ff_rms_masked=float(z["ff_ma"][i]),
                                 false_silence=float(z["fs"][i])))
    if not raws:
        for a, s in combos:
            if str(s) not in report.get(a, {}):
                print("skip (no report):", a, s); continue
            r = evaluate(a, s, report, device, args.val_windows)
            raws.append(r)
            print("[eval] %s s%d done" % (a, s), flush=True)
        np.savez_compressed(CACHE, arms=[r["arm"] for r in raws],
                            seeds=[r["seed"] for r in raws],
                            tau_star=[r["tau_star"] for r in raws],
                            ckpt=[r["checkpoint"] for r in raws],
                            crm_un=[r["contact_rms_unmasked"] for r in raws],
                            crm_ma=[r["contact_rms_masked"] for r in raws],
                            ff_un=[r["ff_rms_unmasked"] for r in raws],
                            ff_ma=[r["ff_rms_masked"] for r in raws],
                            fs=[r["false_silence"] for r in raws],
                            **{"pw_ma_%d" % i: r["per_win_masked"] for i, r in enumerate(raws)},
                            **{"pw_un_%d" % i: r["per_win_unmasked"] for i, r in enumerate(raws)},
                            **{"touched_%d" % i: r["touched"] for i, r in enumerate(raws)})
        print("[save]", CACHE)

    rows = [summarise(r) for r in raws]
    for r in rows:
        print("[%s s%d] tau*=%.2f | contact n=%d sil=%d (%.2f%% of windows, %.2f%% of cells) | "
              "RMS %.4f->%.4f (x%.3f) | silenced %.3f -> %.3f px | identity resid %.2e"
              % (r["arm"], r["seed"], r["tau_star"], r["n_contact_windows"],
                 r["n_silenced_windows"], 100 * r["frac_windows_silenced"],
                 100 * r["false_silence_cells"], r["contact_rms_unmasked"],
                 r["contact_rms_masked"], r["ratio"],
                 r["silenced_with_head"]["rms"], r["silenced_without_head"]["rms"],
                 r["identity"]["residual"]))
    out = dict(
        note=("Per-window closed-loop (k=3) error on contact-regime windows under the "
              "deployed contact-head threshold rule (tau*) and without it (tau=0). Same "
              "code path as S1f_report.json; the reproduction is asserted bit-identical "
              "against the stored contact_rms and false_silence. 'touched' = at least one "
              "rollout step silenced. Identity: pooled_masked^2 - pooled_unmasked^2 = "
              "(n_touched/n_contact) * (rms_without_head^2 - rms_with_head^2)."),
        arms=rows)
    p = os.path.join(ROOT, "analysis_output", "silence_hist.json")
    with io.open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("[save]", p)
    figure(raws, os.path.join(ROOT, "paper", "figures", "fig6_silence_hist"))


if __name__ == "__main__":
    main()
