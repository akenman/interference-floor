# -*- coding: utf-8 -*-
"""SLOT-1f 机制守卫：闭环接触位监督（--w_ct_roll / --ct_pos_weight）。

今日三次度量 bug（轴序错位、反掩码、标签错）全部靠对拍抓出，故本槽位的守卫以
**接线正确性**为第一优先：
  ① x_fut 的最后 3 维必须就是未来帧真值接触旗（组装约定），且与数据集对得上；
  ② ct_roll_loss 与手工 BCE 逐值一致（取 yao 前 3 位 = 接触位）；
  ③ pos_weight<1 时正类（接触）样本损失按比例缩小；
  ④ argparse 默认关闭（历史行为不变）。
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from layout import OBJ_DIMS, OBS_DIMS, ACTION_DIM  # noqa: E402
from train_v3 import ct_roll_loss, parse_args, V3Data  # noqa: E402


def _mk_ds(contact_flags):
    """构造小数据集：contact_flags (E,T,3) 已知，返回 V3Data。"""
    E, T = contact_flags.shape[0], contact_flags.shape[1]
    L, K = 8, 3
    rng = np.random.RandomState(0)
    states = rng.uniform(-50, 50, (E, T, OBJ_DIMS)).astype(np.float32)
    states[:, :, 0] = 1.0
    states[:, :, 8] = 1.0
    states[:, :, 9] = 20.0
    states[:, :, 10] = 2.0
    obstacles = np.zeros((E, 80), dtype=np.float32)
    actions = np.zeros((E, T, ACTION_DIM), dtype=np.float32)
    return V3Data(states, obstacles, contact_flags.astype(np.float32), actions,
                  np.arange(E), L, K, np.zeros(OBJ_DIMS, np.float32),
                  np.ones(OBJ_DIMS, np.float32), np.zeros(80, np.float32),
                  np.ones(80, np.float32)), L, K


def test_x_fut_contact_bits_are_ground_truth():
    """① 组装约定：x_fut[:, k, -3:] == contact[ep, win_start+L+k]（逐值对拍）。"""
    E, T = 3, 20
    ct = np.zeros((E, T, 3), np.float32)
    ct[0, 8] = [1, 0, 0]
    ct[0, 9] = [0, 1, 1]
    ct[1, 10] = [1, 1, 0]
    ds, L, K = _mk_ds(ct)
    idx = np.array([0, 1, 2])
    b = ds.gather(idx)
    xf = b["x_fut"].numpy()                        # (B, K, input_dim)
    assert xf.shape[2] == OBJ_DIMS + OBS_DIMS + 3
    for k in range(K):
        ep = ds.ep_idx[idx]
        t = ds.win_start[idx] + L + k
        assert np.array_equal(xf[:, k, -3:], ct[ep, t]), f"k={k} 接触位不对齐"
    # 障碍段也必须落在 [OBJ_DIMS, OBJ_DIMS+OBS_DIMS)
    assert np.array_equal(xf[:, 0, OBJ_DIMS:OBJ_DIMS + OBS_DIMS],
                          b["x"].numpy()[:, -1, OBJ_DIMS:OBJ_DIMS + OBS_DIMS])


def test_ct_roll_loss_matches_manual_bce():
    """② 与手工 BCE 逐值一致（只取 yao 前 3 位接触位）。"""
    torch.manual_seed(0)
    B, K = 6, 3
    yao = [torch.randn(B, 9) for _ in range(K)]
    xf = torch.zeros(B, K, OBJ_DIMS + OBS_DIMS + 3)
    xf[:, :, -3:] = (torch.rand(B, K, 3) > 0.5).float()
    got = ct_roll_loss(yao, xf).item()
    manual = float(np.mean([
        F.binary_cross_entropy_with_logits(yao[k][:, 0:3].float(),
                                           xf[:, k, -3:].float()).item()
        for k in range(K)]))
    assert abs(got - manual) < 1e-6
    # 尾部 3 维之外的通道不得进入损失：改动物体段不应改变结果
    xf2 = xf.clone()
    xf2[:, :, :OBJ_DIMS] = torch.randn(B, K, OBJ_DIMS)
    assert abs(ct_roll_loss(yao, xf2).item() - got) < 1e-9


def test_pos_weight_scales_positive_class():
    """③ pos_weight：正类（接触=1）样本损失按权重缩放，负类不受影响。"""
    lg = torch.zeros(1, 3)
    pos = torch.ones(1, 3)
    neg = torch.zeros(1, 3)
    base_pos = F.binary_cross_entropy_with_logits(lg, pos).item()
    w_pos = F.binary_cross_entropy_with_logits(
        lg, pos, pos_weight=torch.tensor([0.5])).item()
    assert abs(w_pos - 0.5 * base_pos) < 1e-6
    base_neg = F.binary_cross_entropy_with_logits(lg, neg).item()
    w_neg = F.binary_cross_entropy_with_logits(
        lg, neg, pos_weight=torch.tensor([0.5])).item()
    assert abs(w_neg - base_neg) < 1e-6
    # 端到端：ct_roll_loss 传 pos_weight 后应与手工加权一致
    B, K = 4, 3
    torch.manual_seed(1)
    yao = [torch.randn(B, 9) for _ in range(K)]
    xf = torch.zeros(B, K, OBJ_DIMS + OBS_DIMS + 3)
    xf[:, :, -3:] = torch.ones(B, K, 3)
    got = ct_roll_loss(yao, xf, pos_weight=0.5).item()
    manual = float(np.mean([
        F.binary_cross_entropy_with_logits(
            yao[k][:, 0:3].float(), xf[:, k, -3:].float(),
            pos_weight=torch.tensor([0.5])).item() for k in range(K)]))
    assert abs(got - manual) < 1e-6


def test_argparse_defaults_off():
    """④ 默认关闭（w_ct_roll=0.0 / pos_weight=1.0）⇒ 历史行为不变。"""
    old = sys.argv
    try:
        sys.argv = ["train_v3.py"]
        a = parse_args()
        assert a.w_ct_roll == 0.0 and a.ct_pos_weight == 1.0
        sys.argv = ["train_v3.py", "--w_ct_roll", "1.0", "--ct_pos_weight", "0.67"]
        b = parse_args()
        assert b.w_ct_roll == 1.0 and abs(b.ct_pos_weight - 0.67) < 1e-9
    finally:
        sys.argv = old
