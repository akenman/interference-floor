# -*- coding: utf-8 -*-
"""SLOT-1c 机制守卫（regime 重加权 + 静默门控头）。"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from layout import ASSEMBLED_DIM, OBJ_DIMS, ACTION_DIM  # noqa: E402
from model_v3 import WorldModelV3, GatedWorldModelV3, V3Loss  # noqa: E402


def test_v3data_win_ff_flag():
    """win_ff = (末输入帧无接触) & (目标帧无接触)，逐窗对齐 ep_idx/win_start。"""
    from train_v3 import V3Data
    E, T, L, K = 2, 20, 8, 3
    rng = np.random.RandomState(0)
    states = rng.uniform(-50, 50, (E, T, OBJ_DIMS)).astype(np.float32)
    states[:, :, 0] = 1.0
    states[:, :, 8] = 1.0
    states[:, :, 9] = 20.0
    states[:, :, 10] = 2.0
    obstacles = np.zeros((E, 80), dtype=np.float32)
    contact = np.zeros((E, T, 3), dtype=np.float32)
    contact[:, 8, 0] = 1.0          # 帧 8 接触：i0=0 的目标帧（L=8）→ 窗 0 非_ff
    contact[:, 16, 1] = 1.0         # 帧 16：i0=8 的末输入帧（8+8-1=15? 8+8-1=15, 16=目标）→ 窗 8 非_ff
    actions = np.zeros((E, T, ACTION_DIM), dtype=np.float32)
    ds = V3Data(states, obstacles, contact, actions, np.arange(E), L, K,
                np.zeros(OBJ_DIMS, np.float32), np.ones(OBJ_DIMS, np.float32),
                np.zeros(80, np.float32), np.ones(80, np.float32))
    n_win = T - L - K + 1
    cf = ds.contact.max(axis=2) > 0.5
    expect = np.zeros((E, n_win), dtype=np.float32)
    for e in range(E):
        for i0 in range(n_win):
            expect[e, i0] = float(not cf[e, i0 + L - 1] and not cf[e, i0 + L])
    got = ds.win_ff[ds.ep_idx, ds.win_start]
    assert np.array_equal(got, expect.flatten())
    assert got[0 * n_win + 0] == 0.0 and got[0 * n_win + 8] == 0.0


def test_gated_model_wiring_and_init():
    torch.manual_seed(0)
    m = GatedWorldModelV3(input_dim=ASSEMBLED_DIM, obj_dim=OBJ_DIMS,
                          action_dim=ACTION_DIM, hidden_dim=32, prior_spec="euler001x5")
    assert "gate_head.0.weight" in m.state_dict()
    assert m.prior_spec == "euler001x5"
    B = 8
    h = torch.randn(B, 32)
    a = torch.randn(B, ACTION_DIM)
    with torch.no_grad():
        d_gate = m.predict_state(h, a)
        d_base = WorldModelV3.predict_state(m, h, a)
        g = torch.sigmoid(m.gate_head(torch.cat([h, a], dim=-1)))
    assert d_gate.shape == d_base.shape
    assert torch.allclose(d_gate, g * d_base, atol=1e-6)
    assert (g > 0.9).all()                      # 偏置 +4 ⇒ 起步近 1（≈基线行为）


def test_v3loss_sample_w():
    torch.manual_seed(0)
    m = WorldModelV3(input_dim=ASSEMBLED_DIM, obj_dim=OBJ_DIMS,
                     action_dim=ACTION_DIM, hidden_dim=32)
    B = 6
    h = torch.randn(B, 32)
    c = torch.nn.functional.normalize(torch.randn(B, 32), dim=-1)
    fine = torch.randn(B, m.num_prototypes)
    coarse = torch.randn(B, m.num_coarse)
    a = torch.randn(B, ACTION_DIM)
    tgt = torch.randn(B, OBJ_DIMS)
    yao = torch.zeros(B, 9)
    fine_next = torch.randn(B, m.num_prototypes)
    fw = torch.ones(OBJ_DIMS)
    loss_none, _, _ = V3Loss()(m, h, c, fine, coarse, a, tgt, yao, fine_next, fw)
    loss_ones, _, _ = V3Loss()(m, h, c, fine, coarse, a, tgt, yao, fine_next, fw,
                               sample_w=torch.ones(B))
    assert torch.isclose(loss_none, loss_ones, atol=1e-6)
    w = torch.zeros(B)
    w[0] = 1.0
    loss_one, _, _ = V3Loss()(m, h, c, fine, coarse, a, tgt, yao, fine_next, fw,
                              sample_w=w)
    # 只计第 0 样本的状态项 + 均匀的其余项：状态项应变小但不为 0
    assert loss_one < loss_none and loss_one > 0


def test_gate_supervision_semantics():
    """SLOT-1d：gate_logits 可访问；门目标语义 = 1−win_ff（FF→关，接触→开）。"""
    torch.manual_seed(0)
    m = GatedWorldModelV3(input_dim=ASSEMBLED_DIM, obj_dim=OBJ_DIMS,
                          action_dim=ACTION_DIM, hidden_dim=32,
                          prior_spec="euler001x5")
    B = 8
    h = torch.randn(B, 32)
    a = torch.randn(B, ACTION_DIM)
    logits = m.gate_logits(h, a)
    assert logits.shape == (B,)
    g = torch.sigmoid(logits)
    with torch.no_grad():
        d_gate = m.predict_state(h, a)
        d_base = WorldModelV3.predict_state(m, h, a)
    assert torch.allclose(d_gate, g.unsqueeze(-1) * d_base, atol=1e-6)
    # BCE 语义：win_ff=1(FF)→目标 0（关门）；win_ff=0(接触)→目标 1（开门）
    win_ff = torch.tensor([1.0, 0.0])
    tgt = 1.0 - win_ff
    assert tgt.tolist() == [0.0, 1.0]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits[:2], tgt)
    assert torch.isfinite(bce)
