# -*- coding: utf-8 -*-
"""测 w_roll 的三档成本：原样 / 先验 detach / 先验 detach + rk4005 基座。

先验 detach 的含义：物理先验的**数值不变**，只是不向它的输入反传梯度
（残差路径的跨步梯度仍然完整）。这是把"物理基座"当作固定转移函数的标准近似。
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

from e0_common import make_datasets, OBJ_DIMS
from layout_h import feat_weights
from physics_prior_spec import prior_norm_torch
import model_e0
from train_e0 import ResidualLoss

dev = torch.device("cuda")
train_ds, val_ds, mu, sd = make_datasets("trajectories_hinge.npz", 8, 5, 42,
                                         prior_spec="rk4001x5")
model = model_e0.make_model("ours_base", prior_spec="rk4001x5").to(dev)
model.set_stats(mu, sd)
fw = torch.from_numpy(feat_weights(OBJ_DIMS)).to(dev)
crit = ResidualLoss("ours_base")
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

BS = 256
b = train_ds.gather(np.arange(BS))
xb = b["x"].to(dev); ab = b["a0"].to(dev); afb = b["af"].to(dev)
db = b["d_norm"].to(dev); yaob = b["yao"].to(dev); xf = b["x_fut"].to(dev)


def rollout_detached(model, x_seq, actions_fut, prior_spec):
    """WorldModelH.rollout 的等价实现，但先验在 no_grad 下算并 detach。"""
    B = x_seq.size(0)
    h = torch.zeros(B, model.hidden_dim, device=x_seq.device)
    for t in range(x_seq.size(1)):
        h = model.gru_cell(x_seq[:, t], h)
    h = model.ln_h(h)
    obj_last = x_seq[:, -1, :model.obj_dim]
    ct_frozen = x_seq[:, -1, -3:]
    out = []
    for k in range(actions_fut.size(1)):
        a_k = actions_fut[:, k]
        with torch.no_grad():
            prior_k = prior_norm_torch(prior_spec, obj_last.detach(), a_k,
                                       model.mean_obj, model.std_obj)
        yao_log = model.predict_yao(h)
        delta = model.predict_state(h, a_k, yao_log)
        obj_next = prior_k + delta
        out.append(obj_next)
        ct_fb = torch.sigmoid(yao_log[:, 0:2].float())
        contact_fb = torch.cat([ct_fb, ct_frozen[:, 2:3]], dim=-1)
        h = model.ln_h(model.gru_cell(torch.cat([obj_next, contact_fb], dim=-1), h))
        obj_last = obj_next
    return out


def one_step(w_roll, mode="plain"):
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, c = model.encode_sequence(xb)
        loss, _ = crit(model, h, c, ab, db, yaob, fw)
        if w_roll > 0:
            if mode == "plain":
                ap, _ = model.rollout(xb, afb, x_fut_gt=xf, ss_prob=0.0)
            else:
                ap = rollout_detached(model, xb, afb, "rk4001x5")
            loss = loss + w_roll * sum(
                (0.7 ** k) * F.huber_loss(ap[k].float().clamp(-50, 50),
                                          xf[:, k, :OBJ_DIMS].float(),
                                          reduction="mean") for k in range(5)) / 5
    opt.zero_grad(); loss.backward(); opt.step()


def bench(tag, w_roll, mode="plain", n=8):
    one_step(w_roll, mode); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        one_step(w_roll, mode)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / n * 1000
    print(f"  {tag:<40} {ms:8.1f} ms/batch  → epoch(1983批) {ms*1983/1000:6.0f}s  "
          f"→ 60 epoch {ms*1983*60/3600/1000:5.1f}h/run")


bench("w_roll=0.05  原样（先验带梯度）", 0.05, "plain")
bench("w_roll=0.05  先验 detach", 0.05, "detach")
