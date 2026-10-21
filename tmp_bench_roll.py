# -*- coding: utf-8 -*-
"""定位 w_roll 的每批开销（单批计时，不训练）。"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

from e0_common import make_datasets, RESIDUAL_ARMS, INPUT_DIM, OBJ_DIMS
from layout_h import feat_weights
import model_e0

dev = torch.device("cuda")
train_ds, val_ds, mu, sd = make_datasets("trajectories_hinge.npz", 8, 5, 42,
                                         prior_spec="rk4001x5")
model = model_e0.make_model("ours_base", prior_spec="rk4001x5").to(dev)
model.set_stats(mu, sd)
fw = torch.from_numpy(feat_weights(OBJ_DIMS)).to(dev)
from train_e0 import ResidualLoss
crit = ResidualLoss("ours_base")
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

BS = 256
batch = train_ds.gather(np.arange(BS))
xb = batch["x"].to(dev); ab = batch["a0"].to(dev); afb = batch["af"].to(dev)
db = batch["d_norm"].to(dev); yaob = batch["yao"].to(dev); xf = batch["x_fut"].to(dev)


def step(w_roll):
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, c = model.encode_sequence(xb)
        loss, parts = crit(model, h, c, ab, db, yaob, fw)
        if w_roll > 0:
            abs_preds, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=0.0)
            roll = sum((0.7 ** k) * F.huber_loss(
                abs_preds[k].float().clamp(-50, 50),
                xf[:, k, :OBJ_DIMS].float(), reduction="mean")
                for k in range(5)) / 5
            loss = loss + w_roll * roll
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # 复刻 train_e0 的逐参数 NaN 检查（怀疑的隐藏开销）
    bad = any(not torch.isfinite(p.grad.sum()) for p in model.parameters()
              if p.grad is not None)
    opt.step()
    return bool(bad)


def bench(tag, w_roll, n=8, nan_check=True):
    step(w_roll)                                    # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        if nan_check:
            step(w_roll)
        else:
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h, c = model.encode_sequence(xb)
                loss, parts = crit(model, h, c, ab, db, yaob, fw)
                if w_roll > 0:
                    ap, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=0.0)
                    loss = loss + w_roll * sum(
                        (0.7 ** k) * F.huber_loss(ap[k].float().clamp(-50, 50),
                                                  xf[:, k, :OBJ_DIMS].float(),
                                                  reduction="mean")
                        for k in range(5)) / 5
            opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n * 1000
    print(f"  {tag:<44} {ms:8.1f} ms/batch   预计每 epoch(1983 批) = {ms*1983/1000:6.0f}s")


print(f"参数张量数 = {sum(1 for p in model.parameters() if p.requires_grad)}")
bench("w_roll=0（含 NaN 检查）", 0.0)
bench("w_roll=0.05（含 NaN 检查）", 0.05)
bench("w_roll=0.05（去掉 NaN 检查）", 0.05, nan_check=False)
