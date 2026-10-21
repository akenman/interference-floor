# -*- coding: utf-8 -*-
"""SLOT-1e 机制守卫：闭环 regime 重加权（--w_ff_roll）。

判据语义三条：
  ① w_ff_roll=1.0 → 权重为 None（历史行为逐位不变，计算图不变）；
  ② 否则 win_ff=1 的窗口权重 = w_ff_roll，win_ff=0 的窗口权重 = 1；
  ③ 重构后的逐样本归约在无权时与原来的整体均值等价（数值级）。
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_v3 import rollout_ff_weight, parse_args  # noqa: E402


def test_default_is_off():
    """默认 1.0 → None（调用方不加权 ⇒ 历史行为不变）。"""
    assert rollout_ff_weight(1.0, torch.tensor([0.0, 1.0, 1.0, 0.0])) is None


def test_weight_algebra_matches_win_ff():
    win = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0])
    for w_ff in (10.0, 100.0):
        got = rollout_ff_weight(w_ff, win)
        expect = torch.tensor([1.0, w_ff, w_ff, 1.0, 1.0])
        assert torch.allclose(got, expect)
    w = rollout_ff_weight(10.0, win)
    assert torch.allclose(w[win == 1], torch.full((2,), 10.0))
    assert torch.allclose(w[win == 0], torch.ones(3))


def test_reduction_equivalence_after_refactor():
    """逐样本归约 → 加权 → 均值，在无权时与原整体均值等价。"""
    torch.manual_seed(0)
    hub = torch.rand(8, 33)
    feat_w = torch.rand(33)
    flat = (hub * feat_w.view(1, -1)).mean()
    per = (hub * feat_w.view(1, -1)).mean(dim=1)
    assert torch.allclose(per.mean(), flat, atol=1e-7)
    sw = rollout_ff_weight(1.0, torch.ones(8))
    assert sw is None


def test_weighting_shifts_loss_mass_to_ff_windows():
    """上权确实把损失质量移向 win_ff=1 的样本（方向核验）。"""
    per = torch.tensor([1.0, 1.0, 0.0, 0.0])     # 前两窗有误差
    win = torch.tensor([1.0, 1.0, 0.0, 0.0])     # 前两窗是 FF
    w = rollout_ff_weight(10.0, win)
    assert (per * w).mean() > per.mean()
    win2 = torch.tensor([0.0, 0.0, 1.0, 1.0])    # FF 窗恰好无误差
    w2 = rollout_ff_weight(10.0, win2)
    assert torch.allclose((per * w2).mean(), per.mean())


def test_argparse_registers_w_ff_roll_independently():
    """--w_ff_roll 独立登记、默认 1.0，且与 --w_ff 解耦（配方项分别登记）。"""
    old = sys.argv
    try:
        sys.argv = ["train_v3.py"]
        a = parse_args()
        assert a.w_ff == 1.0 and a.w_ff_roll == 1.0
        sys.argv = ["train_v3.py", "--w_ff", "10", "--w_ff_roll", "10"]
        b = parse_args()
        assert b.w_ff == 10.0 and b.w_ff_roll == 10.0
        sys.argv = ["train_v3.py", "--w_ff_roll", "100"]
        c = parse_args()
        assert c.w_ff == 1.0 and c.w_ff_roll == 100.0
    finally:
        sys.argv = old
