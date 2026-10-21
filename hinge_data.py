# -*- coding: utf-8 -*-
"""
铰链世界数据集 (hinge_data.py) — 统一实现
==========================================
合并 train_hinge.py / train_e0.py / train_e1a.py / train_e1b.py 中的
HingeData / HingeDataE1a / E0Data 为一个参数化类。

唯一差异：energy_median 参数（V4 第4轴能量标签）。
"""

import numpy as np
import torch

from physics_prior import prior_full_np, OBJ_DIMS


class HingeWindowDataset:
    """惰性窗口数据源：从铰链轨迹生成训练窗口。

    支持：
      - 标准 5 项输出（x, x_next, a0, af, d_norm, yao）
      - 可选第4轴能量标签（energy_median 不为 None 时）
      - 残差目标（d_norm = truth - prior）
    """

    def __init__(self, states, contact, actions, yao, targets, episodes,
                 L, K, mean_obj, std_obj, energy_median=None):
        """
        states:   (N, T, 22) 物体状态
        contact:  (N, T, 3)  接触通道
        actions:  (N, T, 1)  力矩
        yao:      (N, T, 3, 2) 爻标签
        targets:  (N, 2)     附加目标（可选）
        episodes: 训练/验证片段索引
        L:        输入序列长度
        K:        rollout 步数
        """
        self.L, self.K = L, K
        self.energy_median = energy_median
        self.mean_obj, self.std_obj = mean_obj, std_obj
        episodes = np.asarray(episodes)
        T = states.shape[1]
        E = len(episodes)

        self.obj_norm = ((states[episodes] - mean_obj) /
                         (std_obj + 1e-8)).astype(np.float32)
        self.act_norm = actions[episodes].astype(np.float32)
        self.contact = contact[episodes].astype(np.float32)
        self.yaos = yao[episodes]
        self.targets = targets[episodes]
        self.states = states[episodes]

        n_win = T - L - K + 1
        self.ep_idx = np.repeat(np.arange(E), n_win)
        self.win_start = np.tile(np.arange(n_win), E)

    def __len__(self):
        return len(self.ep_idx)

    def gather(self, idx):
        """按索引获取训练窗口，返回 dict of torch tensors。"""
        ep = self.ep_idx[idx]
        i0 = self.win_start[idx]
        L, K = self.L, self.K
        B = len(idx)
        off = np.arange(L + K)
        frm = self.obj_norm[ep[:, None], i0[:, None] + off[None, :]]
        ct = self.contact[ep[:, None], i0[:, None] + off[None, :]]

        def assemble(obj_seg, ct_seg, n):
            return np.concatenate([obj_seg, ct_seg], axis=2)

        x = assemble(frm[:, :L], ct[:, :L], L)
        x_next = assemble(frm[:, 1:L + 1], ct[:, 1:L + 1], L)
        x_fut = assemble(frm[:, L:L + K], ct[:, L:L + K], K)
        a0 = self.act_norm[ep, i0 + L]
        af = self.act_norm[ep[:, None], i0[:, None] + L + np.arange(K)[None, :]]
        prior_n = prior_full_np(frm[:, L - 1], a0, self.mean_obj, self.std_obj)
        d_norm = frm[:, L] - prior_n
        y_raw = frm[:, L] * (self.std_obj + 1e-8) + self.mean_obj
        s_last = frm[:, L - 1] * (self.std_obj + 1e-8) + self.mean_obj
        yao_t = self.yaos[ep, i0 + L].reshape(B, 6)

        e4 = np.zeros(B, dtype=np.int8)
        if self.energy_median is not None:
            from model_e1a import energy_level_labels
            e4 = energy_level_labels(self.states, ep, i0, L, self.energy_median)
            e4 = np.asarray(e4).reshape(-1).astype(np.int8)

        to = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return dict(x=to(x), x_next=to(x_next), a0=to(a0), af=to(af),
                    d_norm=to(d_norm), y_raw=to(y_raw), yao=to(yao_t),
                    e4=to(e4), s_last=to(s_last), x_fut=to(x_fut))


def pos_rmse(pred_raw, tgt_raw, pos_idx=None):
    """位置 RMSE（归一化到物理空间后的4维加权）。"""
    if pos_idx is None:
        pos_idx = [1, 2, 12, 13]
    err = (pred_raw[:, pos_idx] - tgt_raw[:, pos_idx]) ** 2
    return float(np.sqrt(err.mean()))
