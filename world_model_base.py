# -*- coding: utf-8 -*-
"""
世界模型基类 (world_model_base.py)
===================================
所有铰链世界模型的统一接口定义。
新增模型变体时只需继承此基类并覆写差异方法。

接口契约：
  encode_sequence(x_seq) -> (h, c_norm)
  evidence(c_norm) -> logits
  prior_norm(obj_norm, action) -> obj_pred
  predict_state(h, action, **cond) -> delta
  predict_yao(h) -> logits
  predict_proto(h, action) -> logits
  assemble_next_input(obj_pred, contact_fb) -> x_next
  ema_prototypes(c_norm, det_assign, gamma=0.99)
  rollout(x_seq, actions_fut, x_fut_gt=None, ss_prob=0.0) -> (abs_preds, yao_preds)
"""

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

from physics_prior import prior_full_torch, OBJ_DIMS


class WorldModelBase(nn.Module, abc.ABC):
    """铰链世界模型抽象基类。

    子类必须实现：
      - prior_norm(obj_norm, action)
      - predict_state(h, action, **cond)

    子类可选覆写：
      - predict_yao(h)          （默认返回零）
      - predict_proto(h, action)（默认返回零）
      - rollout()               （默认先验+残差闭环）
      - assemble_next_input()   （默认 obj+contact 拼接）
    """

    def __init__(self, input_dim=25, obj_dim=OBJ_DIMS, action_dim=1,
                 hidden_dim=256, num_prototypes=32, yao_dim=6):
        super().__init__()
        self.input_dim = input_dim
        self.obj_dim = obj_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.yao_dim = yao_dim

        # ── 共享主干：GRU + LayerNorm ──
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.ln_h = nn.LayerNorm(hidden_dim)

        # ── 原型辅助层 ──
        protos = torch.nn.init.orthogonal_(torch.empty(num_prototypes, hidden_dim))
        self.prototypes = nn.Parameter(F.normalize(protos, p=2, dim=-1))
        self.proto_pred = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_prototypes),
        )

        # ── 归一化统计量（由 set_stats 注入）──
        self.register_buffer("mean_obj", torch.zeros(obj_dim))
        self.register_buffer("std_obj", torch.ones(obj_dim))

    # ═══════════════════════════════════════════
    # 生命周期
    # ═══════════════════════════════════════════

    def set_stats(self, mean_obj, std_obj):
        with torch.no_grad():
            self.mean_obj.copy_(torch.as_tensor(mean_obj, dtype=self.mean_obj.dtype))
            self.std_obj.copy_(torch.as_tensor(std_obj, dtype=self.std_obj.dtype))

    # ═══════════════════════════════════════════
    # 编码器
    # ═══════════════════════════════════════════

    def encode_sequence(self, x_seq):
        """GRU 编码序列，返回 (hidden, l2_normalized_hidden)。"""
        h = torch.zeros(x_seq.size(0), self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.gru_cell(x_seq[:, t], h)
        h = self.ln_h(h)
        return h, F.normalize(h, p=2, dim=-1)

    def evidence(self, c_norm):
        """原型证据：余弦相似度。"""
        return torch.mm(c_norm, self.prototypes.detach().t())

    # ═══════════════════════════════════════════
    # 预测接口（子类必须/可选覆写）
    # ═══════════════════════════════════════════

    @abc.abstractmethod
    def prior_norm(self, obj_norm, action):
        """物理先验：从归一化状态预测下一帧。"""
        ...

    @abc.abstractmethod
    def predict_state(self, h, action, **cond):
        """状态残差预测。子类定义 cond 的含义。"""
        ...

    def predict_yao(self, h):
        """爻相位 logits。默认返回零（无爻模型）。"""
        return torch.zeros(h.size(0), self.yao_dim, device=h.device)

    def predict_proto(self, h, action):
        """原型预测头。"""
        return self.proto_pred(torch.cat([h, action], dim=-1))

    def assemble_next_input(self, obj_norm, contact_fb):
        """组装下一帧 GRU 输入。"""
        return torch.cat([obj_norm, contact_fb], dim=-1)

    # ═══════════════════════════════════════════
    # 原型 EMA
    # ═══════════════════════════════════════════

    @torch.no_grad()
    def ema_prototypes(self, c_norm, det_assign, gamma=0.99):
        protos, assign = self.prototypes, det_assign
        K, D = protos.size(0), protos.size(1)
        sums = torch.zeros(K, D, device=protos.device).index_add_(0, assign, c_norm)
        counts = torch.bincount(assign, minlength=K).clamp(min=1).float()
        means = sums / counts.unsqueeze(1)
        new = gamma * protos.data + (1 - gamma) * means
        protos.data.copy_(F.normalize(new, p=2, dim=-1))

    # ═══════════════════════════════════════════
    # 闭环 Rollout
    # ═══════════════════════════════════════════

    def rollout(self, x_seq, actions_fut, x_fut_gt=None, ss_prob=0.0):
        """闭环 rollout：先验+残差，接触通道用爻头初爻软概率反馈。"""
        B = x_seq.size(0)
        h = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.gru_cell(x_seq[:, t], h)
        h = self.ln_h(h)

        obj_last = x_seq[:, -1, :self.obj_dim]
        ct_frozen = x_seq[:, -1, -3:]
        abs_preds, yao_preds = [], []
        K = actions_fut.size(1)
        for k in range(K):
            a_k = actions_fut[:, k]
            prior_k = self.prior_norm(obj_last, a_k)
            yao_log = self.predict_yao(h)
            delta = self.predict_state(h, a_k, yao_log=yao_log)
            obj_next = prior_k + delta
            abs_preds.append(obj_next)
            yao_preds.append(yao_log)
            ct_fb = torch.sigmoid(yao_log[:, 0:2].float())
            contact_fb = torch.cat([ct_fb, ct_frozen[:, 2:3]], dim=-1)
            if x_fut_gt is not None and ss_prob > 0:
                mask = (torch.rand(B, device=x_seq.device) < ss_prob).float().unsqueeze(1)
                x_next = mask * x_fut_gt[:, k] + (1 - mask) * \
                    self.assemble_next_input(obj_next, contact_fb)
            else:
                x_next = self.assemble_next_input(obj_next, contact_fb)
            h = self.ln_h(self.gru_cell(x_next, h))
            obj_last = obj_next
        return abs_preds, yao_preds
