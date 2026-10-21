# -*- coding: utf-8 -*-
"""
E1a 通用模型 (model_e1a.py) — 爻轴数 / 卦门控基数 的消融参数化
==========================================================
支持 E1a "为什么是 8 相" 的全部 7 个变体 + 无门控基线。

变体映射（仅改爻头输出维 + 卦门控基数，训练配置与 E0 完全一致）：
  V1   axes=(0,)                 n_phases=2   最简相态
  V2a  axes=(0,1)                n_phases=4   去掉意图
  V2b  axes=(0,2)                n_phases=4   去掉能量趋势
  V2c  axes=(1,2)                n_phases=4   去掉接触
  V3   axes=(0,1,2)              n_phases=8   完整架构（当前）
  V4   axes=(0,1,2,3)            n_phases=16  + 能量高低轴（第4轴）
  Vfree free_moe=True, n_gua=8   n_phases=8   自由 8 聚类（无三爻结构）

门控基数规则（"相态 → 卦" 的外积构造，与 GuaMoEWorldModelH 同构）：
  结构化：每物体 n_phases 卦 → 外积 n_gua = n_phases²
    V1: 2²=4   V2: 4²=16   V3: 8²=64   V4: 16²=256
  自由型（Vfree）：h 直连 n_gua=8，无外积结构（free_moe=True）

第4轴语义（V4）：系统总动能 高于/低于 训练分布中位数 —— 与中爻"趋势"互补的"水平"轴。
                标签在数据层按固定阈值生成，不进梯度。
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_h import WorldModelH, HLoss


# --------------------------------------------------------------------------
# 第4轴：能量高低（训练集全局中位数阈值，评估时冻结）
# --------------------------------------------------------------------------
def compute_energy_median(states):
    """全数据集逐帧总动能（端点质量：m1=2 @ obj1, m2=1 @ obj2）的中位数。"""
    v2_1 = states[:, :, 5] ** 2 + states[:, :, 6] ** 2
    v2_2 = states[:, :, 16] ** 2 + states[:, :, 17] ** 2
    ke = 0.5 * 2.0 * v2_1 + 0.5 * 1.0 * v2_2          # (N, T)
    return float(np.median(ke))


def energy_level_labels(states, episodes, win_start, L, energy_median):
    """(B,) 第4轴标签：t = i0+L-1 帧的总动能 > median → 1。"""
    v2_1 = states[:, :, 5] ** 2 + states[:, :, 6] ** 2
    v2_2 = states[:, :, 16] ** 2 + states[:, :, 17] ** 2
    ke = 0.5 * 2.0 * v2_1 + 0.5 * 1.0 * v2_2
    ep = np.asarray(episodes)
    i0 = np.asarray(win_start)
    return (ke[ep, i0 + L - 1] > energy_median).astype(np.int8)


# --------------------------------------------------------------------------
# E1a 通用模型
# --------------------------------------------------------------------------
class E1aModel(WorldModelH):
    """参数化爻轴数 + 卦门控基数的铰链世界模型。

    axes           : 保留的爻轴元组，如 (0,1,2)
    free_moe       : True → 门控由 h 直接生成（无三爻结构），n_gua 为自由专家数
    no_gua         : True → 无门控（M3 的"无门控基线"）
    n_gua_free     : free_moe 时的专家数（Vfree=8）
    energy_median  : V4 第4轴的全局阈值（None 则第4轴不可用）
    """

    def __init__(self, axes=(0, 1, 2), free_moe=False, no_gua=False,
                 n_gua_free=8, energy_median=None,
                 n_gua_override=None, gate_axes=None, gate_outer=None,
                 input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256,
                 num_prototypes=32):
        """axes           : 保留的爻轴元组，如 (0,1,2)
        free_moe       : True → 门控由 h 直接生成（无三爻结构），n_gua 为自由专家数
        no_gua         : True → 无门控（M3 的"无门控基线"）
        n_gua_free     : free_moe 时的专家数（Vfree=8）
        energy_median  : V4 第4轴的全局阈值（None 则第4轴不可用）

        —— 以下为「解耦消融」参数，默认值保持历史行为不变（可与既有 run 对比）——
        n_gua_override : 显式指定门控专家数，解除 n_gua = n_phases² 与轴数的共变
        gate_axes      : 门控读取哪几根爻的 logits；None = 历史行为（读全部 6 维）
        gate_outer     : 门控是否用 (n_phases×n_phases) 外积结构；
                         None = 自动（n_gua == n_phases² 时用外积）
        """
        self.axes = tuple(axes)
        self.free_moe = free_moe
        self.no_gua = no_gua
        self.n_axes = len(self.axes)
        self.n_phases = 2 ** self.n_axes
        if n_gua_override is not None:
            self.n_gua = int(n_gua_override)
        else:
            self.n_gua = (self.n_phases ** 2) if not free_moe else n_gua_free
        self.energy_median = energy_median
        self.has_gate = (not no_gua)
        self.learned_axes = [a for a in self.axes if a < 3]
        # 门控输入轴：None 保持历史行为（完整 6 维 logits）
        self.gate_axes = None if gate_axes is None else tuple(gate_axes)
        self.gate_outer = (self.n_gua == self.n_phases ** 2) \
            if gate_outer is None else bool(gate_outer)

        # 爻头固定 6 维（3 轴×2 物体）；_select_axes 按变体取子集。
        # V4 的第4轴是训练集阈值定义（不进梯度），不算学习轴。
        super().__init__(input_dim=input_dim, obj_dim=obj_dim,
                         action_dim=action_dim, hidden_dim=hidden_dim,
                         num_prototypes=num_prototypes, yao_dim=6)

        if self.has_gate:
            self.film = nn.Linear(self.n_gua, hidden_dim * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)     # 起步无调制 = 父类行为
            if not free_moe:
                # 历史行为：读全部 6 维；解耦臂：只读 gate_axes 指定轴
                n_in = 6 if self.gate_axes is None else 2 * len(self.gate_axes)
                self.gua_gate = nn.Linear(n_in, self.n_gua)   # 由爻 logits 生成
            else:
                self.free_gate = nn.Linear(hidden_dim, self.n_gua)

            self.head_mid = nn.Sequential(
                nn.Linear(hidden_dim + action_dim, hidden_dim),
                nn.GELU(),
            )
            self.head_out = nn.Sequential(
                nn.GELU(),
                nn.Linear(hidden_dim, obj_dim),
            )
            nn.init.normal_(self.head_out[-1].weight, std=0.01)
            nn.init.zeros_(self.head_out[-1].bias)
            del self.state_head   # 父类一体化头由 FiLM 版替代

    # ---------------- 卦门控 ----------------
    def _gua_probs(self, h, yao_log):
        """(B, n_gua) 门控概率。yao_log 为 6 维爻 logits。

        gate_axes=None  → 历史行为：门控读全部 6 维（3 轴全部信息）。
        gate_axes=(a,)  → 解耦臂：门控只读指定轴，屏蔽其余轴的信息通路。
        gate_outer=False→ 专家数与 n_phases² 不匹配时不做外积，直接 softmax。
        """
        if self.free_moe:
            return F.softmax(self.free_gate(h), dim=-1)
        g_in = yao_log if self.gate_axes is None \
            else self._select_axes(yao_log, self.gate_axes)
        logits_obj = self.gua_gate(g_in)            # (B, n_gua)
        if not self.gate_outer:
            return F.softmax(logits_obj, dim=-1)
        side = self.n_phases
        log8 = logits_obj.view(-1, side, side)
        p0 = F.softmax(log8.sum(dim=2) * 2.0, dim=-1)
        p1 = F.softmax(log8.sum(dim=1) * 2.0, dim=-1)
        return torch.einsum("bi,bj->bij", p0, p1).reshape(-1, self.n_gua)

    def predict_state(self, h, action, yao_log=None):
        if not self.has_gate:
            return self.state_head(torch.cat([h, action], dim=-1))
        x = self.head_mid(torch.cat([h, action], dim=-1))
        if yao_log is not None:
            gate = self._gua_probs(h, yao_log)
            film = self.film(gate)
            gamma, beta = film.chunk(2, dim=-1)
            x = (1.0 + gamma) * x + beta
        return self.head_out(x)

    def forward_state(self, h, action):
        if not self.has_gate:
            return self.predict_state(h, action), None
        yao_log = self.predict_yao(h)
        return self.predict_state(h, action, yao_log), yao_log

    # ---------------- 爻轴选择 ----------------
    def _select_axes(self, yao_log_full, axes):
        """从固定 6 维 logits 取学习轴（每轴 2 物体，连续存放）。"""
        if len(axes) == 3:
            return yao_log_full
        return torch.cat([yao_log_full[:, 2 * a:2 * a + 2] for a in axes], dim=-1)

    def predict_yao_selected(self, h):
        return self._select_axes(self.predict_yao(h), self.learned_axes)

    # ---------------- 第4轴标签（V4）----------------
    def energy_axis_label(self, states, episodes, win_start, L):
        if self.energy_median is None:
            raise RuntimeError("V4 requires energy_median")
        return energy_level_labels(states, episodes, win_start, L,
                                   self.energy_median)


# --------------------------------------------------------------------------
# E1a 损失（与 HLoss 同构，爻头按变体轴数对齐）
# --------------------------------------------------------------------------
class E1aLoss(HLoss):
    """与 HLoss 同构；爻头由模型按变体轴数自行投影（predict_yao_selected）。"""

    @staticmethod
    def _select_target(yao_target_6, axes):
        """从 6 维完整标签取保留轴（顺序与 E1aModel._select_axes 一致）。"""
        if len(axes) == 3:
            return yao_target_6
        return torch.cat([yao_target_6[:, 2 * a:2 * a + 2] for a in axes], dim=-1)

    def forward(self, model, h, c_norm, fine_logits, action0, target_delta,
                yao_target, fine_logits_next, feat_w):
        K = model.num_prototypes
        yao_log_full = model.predict_yao(h)
        huber = F.huber_loss(model.predict_state(h, action0, yao_log_full),
                             target_delta, reduction="none")
        state_loss = (huber * feat_w.view(1, -1)).mean()

        yao_log = model.predict_yao_selected(h)
        yao_target_sel = self._select_target(yao_target, model.learned_axes)
        yao_loss = F.binary_cross_entropy_with_logits(yao_log, yao_target_sel.float(),
                                                      reduction="mean")

        prior_p = model.predict_proto(h, action0)
        target_dist = F.softmax(fine_logits_next, dim=-1)
        proto_loss = F.kl_div(F.log_softmax(prior_p, dim=-1), target_dist,
                              reduction="batchmean")

        assigned = model.prototypes.detach()[fine_logits.argmax(dim=-1)]
        commit_loss = (1.0 - F.cosine_similarity(assigned, c_norm, dim=-1)).mean()

        assignment = F.softmax(fine_logits.float(), dim=-1)
        batch_avg = assignment.mean(dim=0).clamp(min=1e-8)
        usage_loss = (math.log(K) - (-(batch_avg * batch_avg.clamp(min=1e-8).log()).sum())) / math.log(K)

        total = (self.w_state * state_loss + self.w_yao * yao_loss
                 + self.w_proto * proto_loss + self.w_commit * commit_loss
                 + self.w_usage * usage_loss)
        parts = dict(state=state_loss.item(), yao=yao_loss.item(),
                     proto=proto_loss.item(), commit=commit_loss.item(),
                     usage=usage_loss.item())
        return total, parts, yao_log
