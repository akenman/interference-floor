# -*- coding: utf-8 -*-
"""Floor analysis, round 7: does the rollout term pull silence back?

Hypothesis H-roll (the one untested claim left in the paper's floor-analysis
subsection): the frozen-trunk refit only reaches 0.34x of the original free-flight
noise at w_ff=100, and the fully trained arms stop at 0.36x. Our working guess is
that the rollout term in the objective is fit on ALL windows (not the free-flight
subset) and pulls silence back.

Design (single seed 42, w_ff=100, same init / lr / batch / epochs, one switch):
  frozen_no_roll  -- reproduces round 6's w_ff=100 branch (reference)
  frozen_roll     -- adds the rollout term (w_roll=0.3, K=3, closed loop)
  free_no_roll    -- trunk also trains (control: is 0.34x an artifact of cached
                     features?); because the trunk moves, this cell does NOT cache
                     features and re-forwards raw windows each step, so it uses a
                     smaller training set.

The rollout loss mirrors train_v3: model.rollout(x_seq, af, obs_last) gives K
autoregressive predictions, MSE against y_fut_norm over the 33 object dims.

Usage: python -X utf8 tmp_trunk_isolation.py
Output: analysis_output/trunk_isolation.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tmp_s1b_compare import load_model, transfer_clean_mask   # noqa: E402
from tmp_silence_ceiling import collect, infer, rms_pos   # noqa: E402
from train_v3 import load_dataset, split_episodes, norm_stats, V3Data   # noqa: E402

CKPT = "checkpoints-S1b/euler001x5_s42/best.pt"
POS = [1, 2, 12, 13, 23, 24]
VAL_WINDOWS = 20000
SEED = 42
W_FF = 100.0
W_ROLL = 0.3

F_N_TRAIN = 120_000      # frozen-trunk cell: cached features
F_EPOCHS = 900
F_LR = 1e-3
F_BATCH = 8192
F_EVAL = (200, 500, 900)

G_N_TRAIN = 60_000       # free-trunk cell: raw windows, no caching
G_EPOCHS = 500
G_LR_HEAD = 1e-3
G_LR_TRUNK = 1e-4
G_BATCH = 2048
G_EVAL = (100, 300, 500)
G_ROLL_SUB = 0.25        # only a quarter of batches carry the rollout term


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

    # evaluation windows: identical set to round 6
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

    out = dict(baseline=dict(head_ff_rms=base_ff, head_ct_rms=base_ct),
               w_ff=W_FF, w_roll=W_ROLL, seed=SEED, cells={})

    # ---------------- frozen trunk: cache features, two cells ----------------
    dst = V3Data(states, obstacles, contact, actions, tr, 8, 3,
                 mo, so, moo, soo, prior_spec=prior)
    it = np.arange(len(dst))
    if len(it) > F_N_TRAIN:
        it = np.sort(np.random.RandomState(11).choice(it, size=F_N_TRAIN, replace=False))
    H, A, D, W, XS, AF, OB, YF = [], [], [], [], [], [], [], []
    for j in range(0, len(it), 4096):
        b = dst.gather(it[j:j + 4096])
        xb = b["x"].to(device)
        with torch.no_grad():
            h, _ = model.encode_sequence(xb)
        H.append(h.cpu().numpy())
        A.append(b["a0"].numpy())
        D.append(b["d_norm"].numpy())
        W.append(b["w_ff"].numpy())
        XS.append(xb.cpu().numpy())
        AF.append(b["af"].numpy())
        OB.append(xb[:, -1, 33:113].cpu().numpy())
        YF.append(b["y_fut_norm"].numpy())
    Xt = torch.from_numpy(np.concatenate(H)).to(device)
    At = torch.from_numpy(np.concatenate(A)).to(device)
    Dt = torch.from_numpy(np.concatenate(D)).to(device)
    Wt = torch.from_numpy(np.concatenate(W).astype(np.float32)).to(device)
    XSt = torch.from_numpy(np.concatenate(XS)).to(device)
    AFt = torch.from_numpy(np.concatenate(AF)).to(device)
    OBt = torch.from_numpy(np.concatenate(OB)).to(device)
    YFt = torch.from_numpy(np.concatenate(YF)).to(device)
    print(f"[frozen] windows {len(Xt)}  FF share {Wt.mean()*100:.2f}%", flush=True)
    sub = torch.randperm(len(Xt), device=device)[:60000]

    def huber_of(m):
        with torch.no_grad():
            return float(F.huber_loss(m.predict_state(Xt[sub], At[sub]), Dt[sub],
                                      reduction="mean").item())

    for tag, use_roll in (("frozen_no_roll", False), ("frozen_roll", True)):
        torch.manual_seed(0)
        m2, _ = load_model(CKPT)
        m2 = m2.to(device)
        for p_ in m2.parameters():
            p_.requires_grad_(False)
        for p_ in m2.state_head.parameters():
            p_.requires_grad_(True)
        opt = torch.optim.AdamW(m2.state_head.parameters(), lr=F_LR,
                                weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=F_EPOCHS)
        sample_w = 1.0 + (W_FF - 1.0) * Wt
        traj = []
        for ep_i in range(1, F_EPOCHS + 1):
            perm = torch.randperm(len(Xt), device=device)
            for j in range(0, len(perm), F_BATCH):
                b = perm[j:j + F_BATCH]
                hu = F.huber_loss(m2.predict_state(Xt[b], At[b]), Dt[b],
                                  reduction="none").mean(dim=1)
                loss = (hu * sample_w[b]).mean()
                if use_roll:
                    preds, _ = m2.rollout(XSt[b], AFt[b], OBt[b])
                    rl = sum(((pk[:, :33] - YFt[b, k]) ** 2).mean()
                             for k, pk in enumerate(preds)) / len(preds)
                    loss = loss + W_ROLL * rl
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            if ep_i in F_EVAL:
                ff, ct = (rms_pos(infer(m2.state_head, Hc, Ac)),
                          rms_pos(infer(m2.state_head, Hk, Ak)))
                traj.append(dict(epoch=ep_i, train_huber=huber_of(m2),
                                 head_ff_rms=ff, head_ct_rms=ct,
                                 ff_over_baseline=ff / base_ff,
                                 ct_over_baseline=ct / base_ct))
                print(f"  {tag:16s} ep{ep_i:<5d} huber={traj[-1]['train_huber']:.5f} "
                      f"FF={ff:.5f} (x{ff/base_ff:.3f}) ct={ct:.5f} "
                      f"(x{ct/base_ct:.3f})", flush=True)
        out["cells"][tag] = traj
        del m2

    # ---------------- free trunk: no caching, one cell ----------------
    # evaluation must use the trunk's CURRENT hidden states, not the frozen cache
    live = []
    for idx in (clean_idx, ct_idx):
        store = []
        for j in range(0, len(idx), 4096):
            b = dsv.gather(idx[j:j + 4096])
            xb = b["x"].to(device)
            with torch.no_grad():
                h, _ = model.encode_sequence(xb)
            store.append((h.cpu(), b["a0"].cpu()))
        live.append(store)

    def eval_live(m):
        res = []
        for store in live:
            sq = []
            for h_, a_ in store:
                with torch.no_grad():
                    o = m.predict_state(h_.to(device), a_.to(device))
                sq.append((o[:, POS] ** 2).mean().item())
            res.append(float(np.sqrt(np.mean(sq))))
        return res[0], res[1]

    itg = np.arange(len(dst))
    if len(itg) > G_N_TRAIN:
        itg = np.sort(np.random.RandomState(12).choice(itg, size=G_N_TRAIN,
                                                       replace=False))
    grng = np.random.RandomState(13)
    torch.manual_seed(0)
    m3, _ = load_model(CKPT)
    m3 = m3.to(device)
    trunk_p = [p_ for n_, p_ in m3.named_parameters()
               if not n_.startswith("state_head")]
    opt = torch.optim.AdamW([{"params": trunk_p, "lr": G_LR_TRUNK},
                             {"params": list(m3.state_head.parameters()),
                              "lr": G_LR_HEAD}], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=G_EPOCHS)
    traj = []
    for ep_i in range(1, G_EPOCHS + 1):
        perm = torch.randperm(len(itg), device=device)
        for j in range(0, len(perm), G_BATCH):
            b = itg[perm[j:j + G_BATCH].cpu().numpy()]
            bb = dst.gather(b)
            xb = bb["x"].to(device)
            ab = bb["a0"].to(device)
            h, _ = m3.encode_sequence(xb)
            sw = 1.0 + (W_FF - 1.0) * bb["w_ff"].to(device)
            hu = F.huber_loss(m3.predict_state(h, ab), bb["d_norm"].to(device),
                              reduction="none").mean(dim=1)
            loss = (hu * sw).mean()
            if grng.rand() < G_ROLL_SUB:
                preds, _ = m3.rollout(xb, bb["af"].to(device), xb[:, -1, 33:113])
                yf = bb["y_fut_norm"].to(device)
                rl = sum(((pk[:, :33] - yf[:, k]) ** 2).mean()
                         for k, pk in enumerate(preds)) / len(preds)
                loss = loss + W_ROLL * rl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m3.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep_i in G_EVAL:
            ff, ct = eval_live(m3)
            traj.append(dict(epoch=ep_i, head_ff_rms=ff, head_ct_rms=ct,
                             ff_over_baseline=ff / base_ff,
                             ct_over_baseline=ct / base_ct))
            print(f"  {'free_no_roll':16s} ep{ep_i:<5d} FF={ff:.5f} "
                  f"(x{ff/base_ff:.3f}) ct={ct:.5f} (x{ct/base_ct:.3f})", flush=True)
    out["cells"]["free_no_roll"] = traj
    del m3

    with open("analysis_output/trunk_isolation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/trunk_isolation.json")


if __name__ == "__main__":
    main()
