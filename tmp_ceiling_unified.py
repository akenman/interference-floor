# -*- coding: utf-8 -*-
"""Floor analysis, round 8: unified frozen-trunk experiment (paper numbers).

Two confounds got loose in rounds 6-7 and this script settles them in one place:

  1. Round 6's `build_head` appended an extra GELU, so its head was not the model's
     real head. Corrected: the head is rebuilt by cloning `model.state_head`'s module
     sequence exactly.
  2. Training-set size was not held fixed. With the corrected head, w_ff=100 gave
     0.58x on 400k cached windows (1200 epochs) but 0.278x on 120k (900 epochs). So
     "how silent can the head get" depends on the sample, and the paper must say so.

Design: frozen trunk from checkpoints-S1b/euler001x5_s42, w_ff=100, exact clean-FF
evaluation set of the paper. 2x2 over {training windows 400k, 120k} x {rollout off,
on}, 1200 epochs everywhere, lr 1e-3 cosine, batch 8192, identical init and window
sampling seeds.

Usage: python -X utf8 tmp_ceiling_unified.py
Output: analysis_output/ceiling_unified.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tmp_s1b_compare import load_model, transfer_clean_mask   # noqa: E402
from tmp_silence_ceiling import build_head, collect, infer, rms_pos   # noqa: E402
from train_v3 import load_dataset, split_episodes, norm_stats, V3Data   # noqa: E402

CKPT = "checkpoints-S1b/euler001x5_s42/best.pt"
VAL_WINDOWS = 20000
SEED = 42
W_FF = 100.0
W_ROLL = 0.3
EPOCHS = 1200
LR = 1e-3
BATCH = 8192
EVAL_AT = (200, 600, 1200)
CELLS = [
    ("w400k_no_roll", 400_000, False),
    ("w400k_roll", 400_000, True),
    ("w120k_no_roll", 120_000, False),
    ("w120k_roll", 120_000, True),
]
# --cells 只跑指定格（用于补跑被中断的那两格），--epochs 可覆盖轮数
import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument("--cells", default="")
_p.add_argument("--epochs", type=int, default=EPOCHS)
_a, _ = _p.parse_known_args()
if _a.cells:
    _want = set(_a.cells.split(","))
    CELLS = [c for c in CELLS if c[0] in _want]
EPOCHS = _a.epochs
EVAL_AT = tuple(e for e in EVAL_AT if e <= EPOCHS) or (EPOCHS,)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device, flush=True)
    states, obstacles, contact, actions = load_dataset("trajectories_v3.npz")
    N = states.shape[0]
    tr, val_eps = split_episodes(N, SEED)
    mo, so, moo, soo = norm_stats(states, obstacles, tr)

    model, ck = load_model(CKPT)
    model = model.to(device)
    prior = ck["config"]["prior_spec"]

    dsv = V3Data(states, obstacles, contact, actions, val_eps, 8, 3,
                 mo, so, moo, soo, prior_spec=prior)
    rng = np.random.RandomState(0)
    iv = np.arange(len(dsv))
    iv = np.sort(rng.choice(iv, size=VAL_WINDOWS, replace=False))
    clean_idx = iv[transfer_clean_mask(dsv, iv, 1)]
    ep, t0 = dsv.ep_idx[iv], dsv.win_start[iv] + dsv.L - 1
    ct_idx = np.random.RandomState(1).choice(
        iv[dsv.contact[ep, t0].max(axis=1) > 0.5], size=len(clean_idx), replace=False)
    Hc, Ac, _, _ = collect(model, dsv, clean_idx, device)
    Hk, Ak, _, _ = collect(model, dsv, ct_idx, device)
    base_ff = rms_pos(infer(model.state_head, Hc, Ac))
    base_ct = rms_pos(infer(model.state_head, Hk, Ak))
    print(f"baseline head: FF={base_ff:.5f}  contact={base_ct:.5f}", flush=True)

    dst = V3Data(states, obstacles, contact, actions, tr, 8, 3,
                 mo, so, moo, soo, prior_spec=prior)
    all_idx = np.arange(len(dst))

    def build_cache(n_win, seed_choice):
        it = np.sort(np.random.RandomState(seed_choice).choice(
            all_idx, size=n_win, replace=False))
        H, A, D, W, XS, AF, OB, YF = [], [], [], [], [], [], [], []
        for j in range(0, len(it), 4096):
            b = dst.gather(it[j:j + 4096])
            xb = b["x"].to(device)
            with torch.no_grad():
                h, _ = model.encode_sequence(xb)
            H.append(h.cpu().numpy()); A.append(b["a0"].numpy())
            D.append(b["d_norm"].numpy()); W.append(b["w_ff"].numpy())
            XS.append(xb.cpu().numpy()); AF.append(b["af"].numpy())
            OB.append(xb[:, -1, 33:113].cpu().numpy())
            YF.append(b["y_fut_norm"].numpy())
        return (torch.from_numpy(np.concatenate(H)).to(device),
                torch.from_numpy(np.concatenate(A)).to(device),
                torch.from_numpy(np.concatenate(D)).to(device),
                torch.from_numpy(np.concatenate(W).astype(np.float32)).to(device),
                torch.from_numpy(np.concatenate(XS)).to(device),
                torch.from_numpy(np.concatenate(AF)).to(device),
                torch.from_numpy(np.concatenate(OB)).to(device),
                torch.from_numpy(np.concatenate(YF)).to(device))

    _needed = sorted({c[1] for c in CELLS})
    caches = {n: build_cache(n, 11) for n in _needed}
    for n, c in caches.items():
        print(f"[cache {n//1000}k] FF share {c[3].mean()*100:.2f}%", flush=True)

    out = dict(baseline=dict(head_ff_rms=base_ff, head_ct_rms=base_ct),
               w_ff=W_FF, w_roll=W_ROLL, epochs=EPOCHS, lr=LR, seed=SEED, cells={})

    for tag, n_win, use_roll in CELLS:
        Xt, At, Dt, Wt, XSt, AFt, OBt, YFt = caches[n_win]
        sub = torch.randperm(len(Xt), device=device)[:60000]
        torch.manual_seed(0)
        net = build_head(model).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        sample_w = 1.0 + (W_FF - 1.0) * Wt
        traj = []
        for ep_i in range(1, EPOCHS + 1):
            perm = torch.randperm(len(Xt), device=device)
            for j in range(0, len(perm), BATCH):
                b = perm[j:j + BATCH]
                hu = F.huber_loss(net(torch.cat([Xt[b], At[b]], dim=1)), Dt[b],
                                  reduction="none").mean(dim=1)
                loss = (hu * sample_w[b]).mean()
                if use_roll:
                    preds, _ = model.rollout(XSt[b], AFt[b], OBt[b])
                    rl = sum(((pk[:, :33] - YFt[b, k]) ** 2).mean()
                             for k, pk in enumerate(preds)) / len(preds)
                    loss = loss + W_ROLL * rl
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            if ep_i in EVAL_AT:
                with torch.no_grad():
                    fit = float(F.huber_loss(
                        net(torch.cat([Xt[sub], At[sub]], dim=1)), Dt[sub],
                        reduction="mean").item())
                ff = rms_pos(infer(net, Hc, Ac))
                ct = rms_pos(infer(net, Hk, Ak))
                traj.append(dict(epoch=ep_i, train_huber=fit, head_ff_rms=ff,
                                 head_ct_rms=ct, ff_over_baseline=ff / base_ff,
                                 ct_over_baseline=ct / base_ct))
                print(f"  {tag:16s} ep{ep_i:<5d} huber={fit:.5f} FF={ff:.5f} "
                      f"(x{ff/base_ff:.3f}) ct={ct:.5f} (x{ct/base_ct:.3f})",
                      flush=True)
        out["cells"][tag] = traj
        del net
        torch.cuda.empty_cache()

    with open(os.environ.get("UNIFIED_OUT", "analysis_output/ceiling_unified.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save]", os.environ.get("UNIFIED_OUT", "analysis_output/ceiling_unified.json"))
    print("\n=== final (epoch %d) ===" % EPOCHS)
    print(f"{'cell':18s} {'FF/base':>9s} {'ct/base':>9s}")
    for tag, _, _ in CELLS:
        r = out["cells"][tag][-1]
        print(f"{tag:18s} {r['ff_over_baseline']:9.3f} {r['ct_over_baseline']:9.3f}")


if __name__ == "__main__":
    main()
