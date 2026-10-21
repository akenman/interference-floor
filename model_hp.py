# -*- coding: utf-8 -*-
"""
铰链世界部分可观测模型 (model_hp.py) — G3 + 周易认识论融合
==================================================
《易传》："幾者，动之微，吉凶之先见者也。"——从可见之象推不可见之机。
六爻的"内卦/外卦"结构 = POMDP 信念状态的天然架构：
  外卦（可观测）：位置/动作/历史 —— 传感器直接可见之象
  内卦（推断）：  接触状态/能量趋势/角速度 —— 隐藏之幾，须由观象推知
  六十四卦 = 外卦 ⊕ 内卦 = 可观测境况 × 信念隐藏态 —— 动力学预测以两者为条件。

三个新增组件（相对 WorldModelH）：
  1. 角速度估计头（ω̂）：遮蔽速度后，从位置历史推断 ω1/ω2（T7 可控域的前提），
     ω̂ 直接喂入物理先验（残差-先验基座在有损观测下的存续方式）。
  2. 信念头（内卦）：从可观测历史推断隐藏三爻 [接触, 能量趋势×2, 趋靶×2, ...]。
     信念准确率 = "知幾"的可测量形式（P3.2）。
  3. 内卦条件化（消融开关）：动力学预测以信念内卦为条件——T10 消融在
     POMDP 场景的重赛（P3.4）。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_h import WorldModelH
from layout_h import L1, L2, M1, M2, TAU_MAX

G_PRIOR = 900.0
TAU_DT = 0.05

# 遮蔽维（归一化 obj 段内）：vx,vy,ω × 2 物体 = [5,6,7,16,17,18]
VEL_DIMS = [5, 6, 7, 16, 17, 18]


class HingePOMDPModel(WorldModelH):
    """部分可观测铰链模型：速度估计 + 内卦信念 + 可选内卦条件化。"""

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256,
                 num_prototypes=32, yao_dim=6, use_belief_film=True):
        super().__init__(input_dim, obj_dim, action_dim, hidden_dim,
                         num_prototypes, yao_dim)
        self.use_belief_film = use_belief_film
        # 1. 角速度估计头：h → (ω̂1, ω̂2)（原始单位 rad/s）
        self.vel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 2))
        # 2. 信念头（内卦）：h → 隐藏三爻 logits（接触1 + 能量趋势2 + 趋靶2 = 5）
        #    （上爻-趋靶在仅位置观测下部分可推——信念头统一学习，让数据决定）
        self.belief_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, yao_dim))
        # 3. 内卦 FiLM：信念 → (γ, β) 调制状态主头
        if use_belief_film:
            self.belief_film = nn.Linear(yao_dim, hidden_dim * 2)
            nn.init.zeros_(self.belief_film.weight)
            nn.init.zeros_(self.belief_film.bias)
        # 状态主头拆两段以支持 FiLM
        self.head_mid = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.GELU())
        self.head_out = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, obj_dim))
        nn.init.normal_(self.head_out[-1].weight, std=0.01)
        nn.init.zeros_(self.head_out[-1].bias)
        del self.state_head

    def estimate_omega(self, h):
        return self.vel_head(h)     # (B, 2) raw rad/s

    def predict_belief(self, h):
        return self.belief_head(h)  # (B, 6) 内卦 logits

    def prior_norm(self, obj_norm, action, w_hat=None):
        """物理先验（归一化空间）。ω 被遮蔽时用估计值 ω̂ 注入。

        obj_norm: (B,22) 可能含零速度（遮蔽）；w_hat: (B,2) raw rad/s 或 None。
        """
        raw = obj_norm * self.std_obj + self.mean_obj
        b1, b2 = 0, 11
        s1n_in = raw[:, b1 + 3].clone()
        c1n_in = raw[:, b1 + 4].clone()
        w1_in = raw[:, b1 + 7].clone()
        w2_in = raw[:, b2 + 7].clone()
        if w_hat is not None:
            w1_in = w_hat[:, 0]
            w2_in = w_hat[:, 1]
        s1n, c1n, s2n, c2n, w1n, w2n = self._euler_step(
            s1n_in, c1n_in, raw[:, b2 + 3], raw[:, b2 + 4], w1_in, w2_in, action)
        out = raw.clone()
        th1n = torch.atan2(s1n, c1n)
        th2n = torch.atan2(s2n, c2n)
        x1 = L1 * torch.sin(th1n); y1 = -L1 * torch.cos(th1n)
        x2 = x1 + L2 * torch.sin(th1n + th2n); y2 = y1 - L2 * torch.cos(th1n + th2n)
        vx1 = L1 * w1n * torch.cos(th1n); vy1 = L1 * w1n * torch.sin(th1n)
        vx2 = vx1 + L2 * (w1n + w2n) * torch.cos(th1n + th2n)
        vy2 = vy1 + L2 * (w1n + w2n) * torch.sin(th1n + th2n)
        out[:, b1 + 1], out[:, b1 + 2] = x1, y1
        out[:, b2 + 1], out[:, b2 + 2] = x2, y2
        out[:, b1 + 3], out[:, b1 + 4] = s1n, c1n
        out[:, b2 + 3], out[:, b2 + 4] = s2n, c2n
        out[:, b1 + 5], out[:, b1 + 6] = vx1, vy1
        out[:, b2 + 5], out[:, b2 + 6] = vx2, vy2
        out[:, b1 + 7], out[:, b2 + 7] = w1n, w2n
        return (out - self.mean_obj) / (self.std_obj + 1e-8)

    @staticmethod
    def _euler_step(s1, c1, s2, c2, w1, w2, a_norm):
        th1 = torch.atan2(s1, c1)
        th2 = torch.atan2(s2, c2)
        tau = a_norm.squeeze(-1) * TAU_MAX
        c2c = torch.cos(th2)
        s2s = torch.sin(th2)
        M11 = (M1 + M2) * L1**2 + M2 * L2**2 + 2 * M2 * L1 * L2 * c2c
        M12 = M2 * L2**2 + M2 * L1 * L2 * c2c
        M22 = M2 * L2**2
        h1 = (-M2 * L1 * L2 * (2 * w1 + w2) * w2 * s2s
              + (M1 + M2) * G_PRIOR * L1 * torch.sin(th1)
              + M2 * G_PRIOR * L2 * torch.sin(th1 + th2))
        h2 = (M2 * L1 * L2 * w1**2 * s2s
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

    def predict_state(self, h, action, belief_log=None):
        """FiLM(内卦信念) 调制的残差头。"""
        x = self.head_mid(torch.cat([h, action], dim=-1))
        if self.use_belief_film and belief_log is not None:
            gamma, beta = self.belief_film(belief_log).chunk(2, dim=-1)
            x = (1.0 + gamma) * x + beta
        return self.head_out(x)

    def forward_state(self, h, action):
        belief = self.predict_belief(h)
        return self.predict_state(h, action, belief), belief
