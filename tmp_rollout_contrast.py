# -*- coding: utf-8 -*-
"""Floor analysis, round 7b: clean rollout-on/off contrast on the TRUE head.

Why this exists: round 6's `build_head` appended an extra GELU, so its head was not
the model's real head. Round 7 (tmp_trunk_isolation.py) trains the real head in place
and got 0.278x for the w_ff=100 / no-rollout cell, versus round 6's 0.338x. This
script re-runs the decisive pair (rollout off vs on) with the true architecture, so
the number that goes into the paper matches the model.

Design: frozen trunk from checkpoints-S1b/euler001x5_s42, w_ff=100, 120k cached
training windows, 900 epochs, lr 1e-3 cosine, batch 8192. The ONLY difference between
the two cells is whether the rollout term (w_roll=0.3, K=3, pure closed loop) is
added. Evaluation windows are the paper's clean-FF mask plus a matched contact set,
identical to rounds 6 and 7.

Usage: python -X utf8 tmp_rollout_contrast.py
Output: analysis_output/rollout_contrast.json
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
N_TRAIN = 120_000
EPOCHS = 900
LR = 1e-3
BATCH = 8192
EVAL_AT = (200, 500, 900)


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
    it = np.arange(len(dst))
    it = np.sort(np.random.RandomState(11).choice(it, size=N_TRAIN, replace=False))
    H, A, D, W, XS, AF, OB, YF = [], [], [], [], [], [], [], []
    for j in range(0, len(it), 4096):
        b = dst.gather(it[j:j + 4096])
        xb = b["x"].to(device)
        with torch.no_grad():
            h, _ = model.encode_sequence(xb)
        H.append(h.cpu().numpy()); A.append(b["a0"].numpy())
        D.append(b["d_norm"].numpy()); W.append(b["w_ff"].numpy())
        XS.append(xb.cpu().numpy()); AF.append(b["af"].numpy())
        OB.append(xb[:, -1, 33:113].cpu().numpy()); YF.append(b["y_fut_norm"].numpy())
    Xt = torch.from_numpy(np.concatenate(H)).to(device)
    At = torch.from_numpy(np.concatenate(A)).to(device)
    Dt = torch.from_numpy(np.concatenate(D)).to(device)
    Wt = torch.from_numpy(np.concatenate(W).astype(np.float32)).to(device)
    XSt = torch.from_numpy(np.concatenate(XS)).to(device)
    AFt = torch.from_numpy(np.concatenate(AF)).to(device)
    OBt = torch.from_numpy(np.concatenate(OB)).to(device)
    YFt = torch.from_numpy(np.concatenate(YF)).to(device)
    print(f"windows {len(Xt)}  FF share {Wt.mean()*100:.2f}%", flush=True)
    sub = torch.randperm(len(Xt), device=device)[:60000]

    out = dict(baseline=dict(head_ff_rms=base_ff, head_ct_rms=base_ct),
               w_ff=W_FF, w_roll=W_ROLL, n_train=N_TRAIN, epochs=EPOCHS,
               lr=LR, seed=SEED, cells={})

    for tag, use_roll in (("no_roll", False), ("roll", True)):
        torch.manual_seed(0)
        m2, _ = load_model(CKPT)
        m2 = m2.to(device)
        for p_ in m2.parameters():
            p_.requires_grad_(False)
        for p_ in m2.state_head.parameters():
            p_.requires_grad_(True)
        opt = torch.optim.AdamW(m2.state_head.parameters(), lr=LR, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        sample_w = 1.0 + (W_FF - 1.0) * Wt
        traj = []
        for ep_i in range(1, EPOCHS + 1):
            perm = torch.randperm(len(Xt), device=device)
            for j in range(0, len(perm), BATCH):
                b = perm[j:j + BATCH]
                hu = F.huber_loss(m2.predict_state(Xt[b], At[b]), Dt[b],
                                  reduction="none").mean(dim=1)
                loss = (hu * sample_w[b]).mean()
                if use_roll:
                    preds, _ = m2.rollout(XSt[b], AFt[b], OBt[b])
                    rl = sum(((pk[:, :33] - YFt[b, k]) ** 2).mean()
                             for k, pk in enumerate(preds)) / len(preds)
                    loss = loss + W_ROLL * rl
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            if ep_i in EVAL_AT:
                with torch.no_grad():
                    fit = float(F.huber_loss(m2.predict_state(Xt[sub], At[sub]),
                                             Dt[sub], reduction="mean").item())
                ff, ct = (rms_pos(infer(m2.state_head, Hc, Ac)),
                          rms_pos(infer(m2.state_head, Hk, Ak)))
                traj.append(dict(epoch=ep_i, train_huber=fit, head_ff_rms=ff,
                                 head_ct_rms=ct, ff_over_baseline=ff / base_ff,
                                 ct_over_baseline=ct / base_ct))
                print(f"  {tag:8s} ep{ep_i:<5d} huber={fit:.5f} FF={ff:.5f} "
                      f"(x{ff/base_ff:.3f}) ct={ct:.5f} (x{ct/base_ct:.3f})",
                      flush=True)
        out["cells"][tag] = traj
        del m2

    with open("analysis_output/rollout_contrast.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[save] analysis_output/rollout_contrast.json")
    a = out["cells"]["no_roll"][-1]
    b = out["cells"]["roll"][-1]
    print(f"\nVERDICT: rollout raises FF from {a['head_ff_rms']:.5f} to "
          f"{b['head_ff_rms']:.5f}  =  x{b['head_ff_rms']/a['head_ff_rms']:.2f}")
    print(f"         contact {a['head_ct_rms']:.5f} -> {b['head_ct_rms']:.5f}  "
          f"=  x{b['head_ct_rms']/a['head_ct_rms']:.2f}")


if __name__ == "__main__":
    main()
