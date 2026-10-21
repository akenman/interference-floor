# -*- coding: utf-8 -*-
"""
model_h_e1b.py — E1b 相态容量分配的模型变体（独立于 model_h.py）

独立成模块的原因：model_h.py 被其他实验流程并发编辑，把 E1b 变体放在这里
可避免被覆盖。依赖仅来自 model_h.py 的稳定基类（WorldModelH / HLoss）。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_h import WorldModelH, HLoss


class PhaseCapacityWorldModelH(WorldModelH):
    """B-phase：状态主头拆为 8 个相态专用子头，输入经卦门控路由。

    与父类的差异仅在主头：8 个并行的子头各输出完整 obj_dim，
    输出 = Σ_g w_g(卦) · head_g([h,a])，w_g 由预测爻 logits 的 8 卦软概率给出。
    子头末层零初始化 ⇒ 起步输出 0 ⇒ 起步=物理先验（公平起步）。

    参数量（hidden=256, obj_dim=22, action=1, sub_head_dim=60）：
      8 × [(257×60+60) + (60×22+22)] = 399 318  vs 基线 402 236（−0.7%，在 ±1% 内）
    """

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256,
                 num_prototypes=32, yao_dim=6, sub_head_dim=60):
        super().__init__(input_dim, obj_dim, action_dim, hidden_dim,
                         num_prototypes, yao_dim)
        self.sub_head_dim = sub_head_dim
        self.n_gua = 8
        del self.state_head

        self.sub_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + action_dim, sub_head_dim),
                nn.GELU(),
                nn.Linear(sub_head_dim, obj_dim),
            ) for _ in range(self.n_gua)])
        for head in self.sub_heads:
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)

    def _gua_probs(self, yao_log):
        """由爻 logits (B,6) → 手端(物体1) 8 卦概率 (B,8)。

        yao_log 布局（原数组 (3 轴, 2 物体) 行主序）:
            [轴0_物0, 轴0_物1, 轴1_物0, 轴1_物1, 轴2_物0, 轴2_物1]
        手端（物体1）三爻 = 下标 1, 3, 5。
        """
        bits = torch.sigmoid(yao_log.float()[:, [1, 3, 5]])          # (B,3)
        p = torch.stack([bits, 1.0 - bits], dim=-1)                  # (B,3,2)
        probs = torch.empty(yao_log.size(0), 8, device=yao_log.device, dtype=yao_log.dtype)
        for g in range(8):
            b0, b1, b2 = (g >> 2) & 1, (g >> 1) & 1, g & 1
            probs[:, g] = p[:, 0, b0] * p[:, 1, b1] * p[:, 2, b2]
        return probs

    def predict_state(self, h, action, yao_log=None):
        """卦门控路由的残差主头。"""
        x = torch.cat([h, action], dim=-1)
        out = torch.zeros(h.size(0), self.obj_dim, device=h.device, dtype=h.dtype)
        if yao_log is not None:
            w = self._gua_probs(yao_log)                             # (B, 8)
            for g in range(self.n_gua):
                out = out + w[:, g:g + 1] * self.sub_heads[g](x)
        else:
            for g in range(self.n_gua):
                out = out + self.sub_heads[g](x) / self.n_gua
        return out

    def forward_state(self, h, action):
        """供 loss 调用（内部先算爻 logits 再门控）。"""
        yao_log = self.predict_yao(h)
        return self.predict_state(h, action, yao_log), yao_log


class PriorityHLoss(HLoss):
    """B-priority 损失：状态 Huber 项按卦难度加权。

    难卦权重 w_g = max(1, err_g/mean_err)，易卦 = 1；
    最后按数据集均值归一化，保持总损失量级不变（避免有效 LR 漂移）。
    """

    def __init__(self, gua_rel_err, w_state=1.0, w_yao=0.1, w_proto=0.05,
                 w_commit=0.01, w_usage=0.02):
        super().__init__(w_state, w_yao, w_proto, w_commit, w_usage)
        rel = torch.as_tensor(gua_rel_err, dtype=torch.float32)
        w = torch.where(rel > 1.0, rel, torch.ones_like(rel))
        self.register_buffer("_gua_weight", w / w.mean())

    def forward(self, model, h, c_norm, fine_logits, action0, target_delta,
                yao_target, fine_logits_next, feat_w, gua_idx):
        K = model.num_prototypes
        huber = F.huber_loss(model.predict_state(h, action0), target_delta,
                             reduction="none")                  # (B, obj_dim)
        per_sample = (huber * feat_w.view(1, -1)).mean(dim=-1)   # (B,)
        w = self._gua_weight[gua_idx]                             # (B,)
        state_loss = (per_sample * w).mean()

        yao_log = model.predict_yao(h)
        yao_loss = F.binary_cross_entropy_with_logits(
            yao_log, yao_target.float(), reduction="mean")

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
