# -*- coding: utf-8 -*-
"""
统一核心模型 (core_model.py) — 三臂对比的公平基础
==================================================
三个架构共享同一个 GRU 主干，差异仅在预测路径：
  Ours      = 物理先验基座 + 残差头 + 三爻头 + 原型辅助层
  NoPrior   = 同 GRU + 直接状态头（无物理先验、无爻、无原型）
  JEPALike  = 同 GRU + latent 空间预测 + EMA target encoder（表征范式）

设计原则：
  所有臂同参数量级（±10%），同训练预算，同评估协议。
  差异仅在于"预测路径"——隔离架构选择的因果效应。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layout_h import L1, L2, M1, M2, TAU_MAX

G_PRIOR = 900.0
TAU_DT = 0.05


def _euler_step_torch(sin1, cos1, sin2, cos2, w1, w2, a_norm):
    """物理先验的一步（0.05s 半隐式欧拉，torch 批量版）。"""
    th1 = torch.atan2(sin1, cos1)
    th2 = torch.atan2(sin2, cos2)
    tau = a_norm.squeeze(-1) * TAU_MAX
    c2 = torch.cos(th2)
    s2 = torch.sin(th2)
    M11 = (M1 + M2) * L1**2 + M2 * L2**2 + 2 * M2 * L1 * L2 * c2
    M12 = M2 * L2**2 + M2 * L1 * L2 * c2
    M22 = M2 * L2**2
    h1 = (-M2 * L1 * L2 * (2 * w1 + w2) * w2 * s2
          + (M1 + M2) * G_PRIOR * L1 * torch.sin(th1)
          + M2 * G_PRIOR * L2 * torch.sin(th1 + th2))
    h2 = (M2 * L1 * L2 * w1**2 * s2
          + M2 * G_PRIOR * L2 * torch.sin(th1 + th2)
          - tau)
    det = M11 * M22 - M12**2
    det = torch.where(det.abs() < 1e-9,
                       det.sign() * 1e-9, det)
    a1 = (M22 * (-h1) - M12 * (-h2)) / det
    a2 = (-M12 * (-h1) + M11 * (-h2)) / det
    w1n = w1 + a1 * TAU_DT
    w2n = w2 + a2 * TAU_DT
    th1n = th1 + w1n * TAU_DT
    th2n = th2 + w2n * TAU_DT
    return torch.sin(th1n), torch.cos(th1n), torch.sin(th2n), torch.cos(th2n), w1n, w2n


# ============================================================
# 共享 GRU 主干
# ============================================================

class SharedTrunk(nn.Module):
    """GRU 编码器 + LayerNorm（三臂完全相同）。"""

    def __init__(self, input_dim=25, hidden_dim=256):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.ln_h = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x_seq):
        """(B, L, input_dim) → (B, hidden_dim)。"""
        h = torch.zeros(x_seq.size(0), self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.gru_cell(x_seq[:, t], h)
        return self.ln_h(h)


# ============================================================
# 臂 1: Ours（物理先验 + 残差 + 三爻）
# ============================================================

class OursModel(nn.Module):
    """完整理念架构。"""

    physics_prior = True    # 供 e0_common.validate 判定先验列（显式声明，勿靠 hasattr）

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256):
        super().__init__()
        self.obj_dim = obj_dim
        self.trunk = SharedTrunk(input_dim, hidden_dim)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, obj_dim))
        nn.init.normal_(self.state_head[-1].weight, std=0.01)
        nn.init.zeros_(self.state_head[-1].bias)
        self.yao_head = nn.Linear(hidden_dim, 6)
        self.register_buffer("mean_obj", torch.zeros(obj_dim))
        self.register_buffer("std_obj", torch.ones(obj_dim))

    def set_stats(self, mean_obj, std_obj):
        with torch.no_grad():
            self.mean_obj.copy_(torch.as_tensor(mean_obj, dtype=self.mean_obj.dtype))
            self.std_obj.copy_(torch.as_tensor(std_obj, dtype=self.std_obj.dtype))

    def prior_norm(self, obj_norm, action):
        # 对称往返（与 HingeData 归一化互逆）：×std 会破坏 std≈0 的常数维
        raw = obj_norm * (self.std_obj + 1e-8) + self.mean_obj
        b1, b2 = 0, 11
        s1n, c1n, s2n, c2n, w1n, w2n = _euler_step_torch(
            raw[:, b1+3], raw[:, b1+4], raw[:, b2+3], raw[:, b2+4],
            raw[:, b1+7], raw[:, b2+7], action)
        out = raw.clone()
        th1n = torch.atan2(s1n, c1n); th2n = torch.atan2(s2n, c2n)
        x1 = L1 * torch.sin(th1n); y1 = -L1 * torch.cos(th1n)
        x2 = x1 + L2 * torch.sin(th1n + th2n); y2 = y1 - L2 * torch.cos(th1n + th2n)
        vx1 = L1 * w1n * torch.cos(th1n); vy1 = L1 * w1n * torch.sin(th1n)
        vx2 = vx1 + L2 * (w1n + w2n) * torch.cos(th1n + th2n)
        vy2 = vy1 + L2 * (w1n + w2n) * torch.sin(th1n + th2n)
        out[:, b1+1], out[:, b1+2] = x1, y1
        out[:, b2+1], out[:, b2+2] = x2, y2
        out[:, b1+3], out[:, b1+4] = s1n, c1n
        out[:, b2+3], out[:, b2+4] = s2n, c2n
        out[:, b1+5], out[:, b1+6] = vx1, vy1
        out[:, b2+5], out[:, b2+6] = vx2, vy2
        out[:, b1+7], out[:, b2+7] = w1n, w2n
        return (out - self.mean_obj) / (self.std_obj + 1e-8)

    def predict(self, h, action):
        """残差：先验在外部组合（pred = prior_norm(last_obj, action) + predict(h, action)）。"""
        return self.state_head(torch.cat([h, action], dim=-1))


# ============================================================
# 臂 2: NoPrior（同 GRU + 直接状态头，无物理先验）
# ============================================================

class NoPriorModel(nn.Module):
    """纯学习：无物理先验，GRU 直接预测下一状态。"""

    physics_prior = False

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256):
        super().__init__()
        self.obj_dim = obj_dim
        self.trunk = SharedTrunk(input_dim, hidden_dim)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, obj_dim))
        nn.init.normal_(self.state_head[-1].weight, std=0.01)
        nn.init.zeros_(self.state_head[-1].bias)

    def set_stats(self, mean_obj, std_obj):
        pass  # 无需统计量（纯学习，不使用物理先验）

    def predict(self, h, action):
        return self.state_head(torch.cat([h, action], dim=-1))

    def predict_yao(self, h):
        """noprior 臂无相位头 → 占位零输出（评估协议要求统一接口）。"""
        return torch.zeros(h.size(0), 6, device=h.device)

    def encode(self, x_seq):
        h = torch.zeros(x_seq.size(0), self.trunk.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.trunk.gru_cell(x_seq[:, t], h)
        return self.trunk.ln_h(h)

    def encode_sequence(self, x_seq):
        """与 WorldModelH 同接口：返回 (h, c) 供统一管线消费。"""
        h = self.encode(x_seq)
        return h, F.normalize(h, p=2, dim=-1)

    def rollout(self, x_seq, actions_fut, x_fut_gt=None, ss_prob=0.0):
        """闭环滚动：纯学习直接输出绝对状态；接触通道沿用末帧（无爻头）。"""
        B = x_seq.size(0)
        h = torch.zeros(B, self.trunk.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.trunk.gru_cell(x_seq[:, t], h)
        h = self.trunk.ln_h(h)
        ct_frozen = x_seq[:, -1, -3:]
        abs_preds = []
        K = actions_fut.size(1)
        for k in range(K):
            a_k = actions_fut[:, k]
            obj_next = self.predict(h, a_k)          # 直接输出绝对状态（无先验）
            abs_preds.append(obj_next)
            x_next = torch.cat([obj_next, ct_frozen], dim=-1)
            if x_fut_gt is not None and ss_prob > 0:
                mask = (torch.rand(B, device=x_seq.device) < ss_prob).float().unsqueeze(1)
                x_next = mask * x_fut_gt[:, k] + (1 - mask) * x_next
            h = self.trunk.ln_h(self.trunk.gru_cell(x_next, h))
        return abs_preds, None


# ============================================================
# 臂 3: JEPA-style（同 GRU + latent 空间预测 + EMA target）
# ============================================================

class JEPALikeModel(nn.Module):
    """JEPA 范式：在表征空间预测，不直接输出状态。
    context encoder (在线) + target encoder (EMA) + predictor。
    评估时通过 linear probe 从 latent 解码状态。"""

    physics_prior = False

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256,
                 latent_dim=128):
        super().__init__()
        self.obj_dim = obj_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # 用 GRUCell（而非 nn.GRU）——与 SharedTrunk / NoPriorModel 的 trunk.gru_cell
        # 完全同构，这样"同一 GRU 主干"的公平性承诺在字面上成立；
        # 且 nn.GRUCell 支持单步步进，闭环 rollout 无需手工拆权重。
        self.context_encoder = nn.GRUCell(input_dim, hidden_dim)
        self.target_encoder = nn.GRUCell(input_dim, hidden_dim)
        self.ln_h = nn.LayerNorm(hidden_dim)          # trunk 有 ln_h，JEPA 臂不能缺
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim))
        # 目标投影器：把 EMA 编码器的 hidden 投影到与 predictor 输出同维的 latent 空间
        # （无它则 latent_loss 比较 128 vs hidden，是维度不匹配而非架构差异）
        self.target_projector = nn.Linear(hidden_dim, latent_dim)
        nn.init.normal_(self.target_projector.weight, std=0.01)
        nn.init.zeros_(self.target_projector.bias)
        self.probe = nn.Linear(latent_dim + action_dim, obj_dim)   # probe: latent+action → state
        nn.init.normal_(self.probe.weight, std=0.01)
        nn.init.zeros_(self.probe.bias)
        self.register_buffer("mean_obj", torch.zeros(obj_dim))
        self.register_buffer("std_obj", torch.ones(obj_dim))

    def set_stats(self, mean_obj, std_obj):
        with torch.no_grad():
            self.mean_obj.copy_(torch.as_tensor(mean_obj, dtype=self.mean_obj.dtype))
            self.std_obj.copy_(torch.as_tensor(std_obj, dtype=self.std_obj.dtype))

    @torch.no_grad()
    def _ema_update(self):
        for p_online, p_target in zip(self.context_encoder.parameters(),
                                       self.target_encoder.parameters()):
            p_target.data.mul_(0.99).add_(p_online.data, alpha=0.01)
        # 目标投影器 = predictor 最后一层的 EMA 副本（与 iBOT 目标头同理）
        w, b = self.predictor[-1].weight, self.predictor[-1].bias
        self.target_projector.weight.data.mul_(0.99).add_(w.data, alpha=0.01)
        if b is not None:
            self.target_projector.bias.data.mul_(0.99).add_(b.data, alpha=0.01)

    def _step(self, x, h):
        """单步编码：GRUCell → LayerNorm，与 trunk 的推进方式一致。"""
        return self.ln_h(self.context_encoder(x, h))

    @torch.no_grad()
    def _target_step(self, x, h):
        """目标编码器的单步（无梯度，EMA 权重）。"""
        return self.ln_h(self.target_encoder(x, h))

    def forward_latent(self, x_seq, action):
        """在线编码整段历史 → 预测下一 latent。"""
        h = torch.zeros(x_seq.size(0), self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self._step(x_seq[:, t], h)
        return self.predictor(torch.cat([h, action], dim=-1))

    def forward_target(self, x_seq):
        """EMA target 编码整段 → 投影到 latent 空间。"""
        h = torch.zeros(x_seq.size(0), self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self._target_step(x_seq[:, t], h)
        return self.target_projector(h)

    def predict_latent(self, h, action):
        """由编码后的隐状态预测下一步 latent（JEPA 的核心前向）。"""
        return self.predictor(torch.cat([h, action], dim=-1))

    def predict_state(self, h, action):
        """Linear probe：隐状态 → 状态。
        调用约定：h 为 GRU 隐状态（评估/rollout）或已预测的 latent（训练时的 jepa_loss）；
        按维度自动区分，避免两条路径写出两套解码。"""
        if h.size(-1) == self.latent_dim:
            latent = h
        else:
            latent = self.predictor(torch.cat([h, action], dim=-1))
        return self.probe(torch.cat([latent, action], dim=-1))

    def predict_yao(self, h):
        """JEPA 臂无相位头 → 占位零输出（评估协议要求统一接口）。"""
        return torch.zeros(h.size(0), 6, device=h.device)

    def encode(self, x_seq):
        h = torch.zeros(x_seq.size(0), self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self._step(x_seq[:, t], h)
        return h

    def encode_sequence(self, x_seq):
        """与 WorldModelH 同接口：h = 上下文隐状态，c = 归一化版本。"""
        h = self.encode(x_seq)
        return h, F.normalize(h, p=2, dim=-1)

    def rollout(self, x_seq, actions_fut, x_fut_gt=None, ss_prob=0.0):
        """闭环滚动：在线 GRU 演化 → 预测 latent → probe 解码状态。"""
        B = x_seq.size(0)
        h = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self._step(x_seq[:, t], h)
        ct_frozen = x_seq[:, -1, -3:]
        abs_preds = []
        K = actions_fut.size(1)
        for k in range(K):
            a_k = actions_fut[:, k]
            latent = self.predictor(torch.cat([h, a_k], dim=-1))
            obj_next = self.predict_state(latent, a_k)     # linear probe 解码
            abs_preds.append(obj_next)
            x_next = torch.cat([obj_next, ct_frozen], dim=-1)
            if x_fut_gt is not None and ss_prob > 0:
                mask = (torch.rand(B, device=x_seq.device) < ss_prob).float().unsqueeze(1)
                x_next = mask * x_fut_gt[:, k] + (1 - mask) * x_next
            h = self._step(x_next, h)
        return abs_preds, None
