# -*- coding: utf-8 -*-
"""
铰链世界模型 (model_h.py) — 与 model_v3 同设计语言、按铰链世界维度参数化
==================================================
独立文件的原因：model_v3 的 prior_norm/爻头硬编码 3 物体弹球世界语义，
且该文件存在并行外部编辑；铰链世界（2 物体、1 维力矩动作、6 维爻头）
单独建文件避免接线事故重演（M2b 教训）。

结构（与 v3 相同的哲学，T1/T3 定理直接复用）：
  GRU 主干 →
    ├─ 状态主头【残差参数化】 pred = 先验(惯性双摆 0.05s 半隐式欧拉) + head([h,a])
    ├─ 三爻头 6 logits = [初(2物体) | 中(2) | 上(2)]（构造性相位变量）
    └─ 原型辅助层（意象流转，可解释副产物）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 物理常量（与 layout_h 一致）
L1, L2 = 80.0, 80.0
M1, M2 = 2.0, 1.0
G_PRIOR = 900.0
TAU_MAX = 28000.0
TAU_DT = 0.05

# 先验积分器已收敛到 physics_prior_spec（唯一实现来源，2026-10-01 审计）。
# prior_step_torch 保留为兼容包装：默认 euler@0.05×1，与旧行为逐位一致。
from physics_prior_spec import (DEFAULT_SPEC as PRIOR_DEFAULT,   # noqa: E402
                                prior_step_torch_spec, prior_norm_torch)


def prior_step_torch(sin1, cos1, sin2, cos2, w1, w2, a_norm,
                     spec=PRIOR_DEFAULT):
    """无接触双摆先验（归一化空间，torch 批量版）。

    签名与旧版兼容；实现委托给 physics_prior_spec（spec 默认 euler@0.05×1）。
    """
    return prior_step_torch_spec(spec, sin1, cos1, sin2, cos2, w1, w2, a_norm)


class WorldModelH(nn.Module):
    """铰链世界模型：GRU 记忆 + 残差状态主头 + 三爻头 + 原型辅助层。"""

    # 供评估管线判定「该模型是否真的带物理先验」（e0_common.validate）。
    # 默认 False：只有显式声明 True 的模型才会被当成先验臂，避免把
    # 「零 prior_norm」的纯学习臂误报成先验误差（strong_mlp 的 87.9px 伪影）。
    physics_prior = True
    # 是否具备爻头。无爻头的消融臂在闭环里必须冻结接触通道，
    # 否则 sigmoid(0)=0.5 会被喂进 GRU（训练时只见过 0/1），污染 rollout 指标。
    has_yao_head = True
    # 闭环训练时是否把物理先验的输入 detach（数值不变、省时间）。默认关闭，
    # 由 train_e0 --detach_prior 设置；见 rollout 内的注释与 P0-b 预注册。
    rollout_detach_prior = False

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256,
                 num_prototypes=32, yao_dim=6, prior_spec=PRIOR_DEFAULT):
        super().__init__()
        self.input_dim = input_dim
        self.obj_dim = obj_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.yao_dim = yao_dim
        self.prior_spec = prior_spec      # 先验档位（physics_prior_spec.SPECS）

        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.ln_h = nn.LayerNorm(hidden_dim)

        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, obj_dim),
        )
        nn.init.normal_(self.state_head[-1].weight, std=0.01)
        nn.init.zeros_(self.state_head[-1].bias)   # 零初始化 → 起步=物理先验

        self.yao_head = nn.Linear(hidden_dim, yao_dim)

        protos = torch.nn.init.orthogonal_(torch.empty(num_prototypes, hidden_dim))
        self.prototypes = nn.Parameter(F.normalize(protos, p=2, dim=-1))
        self.proto_pred = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_prototypes),
        )

        self.register_buffer("mean_obj", torch.zeros(obj_dim))
        self.register_buffer("std_obj", torch.ones(obj_dim))

    def set_stats(self, mean_obj, std_obj):
        with torch.no_grad():
            self.mean_obj.copy_(torch.as_tensor(mean_obj, dtype=self.mean_obj.dtype))
            self.std_obj.copy_(torch.as_tensor(std_obj, dtype=self.std_obj.dtype))

    def encode_sequence(self, x_seq):
        h = torch.zeros(x_seq.size(0), self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.gru_cell(x_seq[:, t], h)
        h = self.ln_h(h)
        return h, F.normalize(h, p=2, dim=-1)

    def evidence(self, c_norm):
        return torch.mm(c_norm, self.prototypes.detach().t())

    def prior_norm(self, obj_norm, action):
        """无接触双摆先验（归一化空间）。obj_norm: (B, 22)，action: (B,1)。

        实现委托 physics_prior_spec.prior_norm_torch（spec = self.prior_spec）；
        默认 euler005 与旧实现逐位一致（tests/test_prior_spec.py 断言）。
        """
        return prior_norm_torch(self.prior_spec, obj_norm, action,
                                self.mean_obj, self.std_obj)

    def predict_state(self, h, action, yao_log=None):
        """残差主头：pred = 先验 + head([h,a])。返回残差。"""
        return self.state_head(torch.cat([h, action], dim=-1))

    def predict_yao(self, h):
        return self.yao_head(h)

    def predict_proto(self, h, action):
        return self.proto_pred(torch.cat([h, action], dim=-1))

    def assemble_next_input(self, obj_norm, contact_fb):
        return torch.cat([obj_norm, contact_fb], dim=-1)

    @torch.no_grad()
    def ema_prototypes(self, c_norm, det_assign, gamma=0.99):
        protos, assign = self.prototypes, det_assign
        K, D = protos.size(0), protos.size(1)
        sums = torch.zeros(K, D, device=protos.device).index_add_(0, assign, c_norm)
        counts = torch.bincount(assign, minlength=K).clamp(min=1).float()
        means = sums / counts.unsqueeze(1)
        new = gamma * protos.data + (1 - gamma) * means
        protos.data.copy_(F.normalize(new, p=2, dim=-1))

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
            if getattr(self, "rollout_detach_prior", False):
                # 闭环训练的成本控制（P0-b）：物理先验的**数值不变**，只是不在
                # 它的输入上反传梯度（残差路径的跨步梯度仍完整）。精确先验
                # （rk4@0.01×5）每步要做 20 次动力学求值，带梯度时占 rollout 成本
                # 的 ~2/3（实测 309→152 ms/batch）。默认关闭，仅训练时按 flag 打开。
                with torch.no_grad():
                    prior_k = self.prior_norm(obj_last.detach(), a_k)
            else:
                prior_k = self.prior_norm(obj_last, a_k)
            yao_log = self.predict_yao(h)
            delta = self.predict_state(h, a_k, yao_log)
            obj_next = prior_k + delta
            abs_preds.append(obj_next)
            yao_preds.append(yao_log)
            if self.has_yao_head:
                ct_fb = torch.sigmoid(yao_log[:, 0:2].float())          # 初爻 2 物体软概率
                contact_fb = torch.cat([ct_fb, ct_frozen[:, 2:3]], dim=-1)   # + 冻结警戒通道 = 3 维
            else:
                # 无爻头消融臂：predict_yao 是接口占位零输出，sigmoid(0)=0.5 属
                # 训练分布外的输入 → 冻结末帧接触（与 NoPriorModel.rollout 同口径）
                contact_fb = ct_frozen
            if x_fut_gt is not None and ss_prob > 0:
                mask = (torch.rand(B, device=x_seq.device) < ss_prob).float().unsqueeze(1)
                x_next = mask * x_fut_gt[:, k] + (1 - mask) * \
                    self.assemble_next_input(obj_next, contact_fb)
            else:
                x_next = self.assemble_next_input(obj_next, contact_fb)
            h = self.ln_h(self.gru_cell(x_next, h))
            obj_last = obj_next
        return abs_preds, yao_preds


class HLoss(nn.Module):
    """铰链世界损失（与 v3 同构 5 项 + 可选校准）。"""

    def __init__(self, w_state=1.0, w_yao=0.1, w_proto=0.05, w_commit=0.01, w_usage=0.02):
        super().__init__()
        self.w_state, self.w_yao = w_state, w_yao
        self.w_proto, self.w_commit, self.w_usage = w_proto, w_commit, w_usage

    def forward(self, model, h, c_norm, fine_logits, action0, target_delta,
                yao_target, fine_logits_next, feat_w):
        K = model.num_prototypes
        yao_log = model.predict_yao(h)
        huber = F.huber_loss(model.predict_state(h, action0, yao_log), target_delta, reduction="none")
        state_loss = (huber * feat_w.view(1, -1)).mean()
        yao_loss = F.binary_cross_entropy_with_logits(yao_log, yao_target.float(), reduction="mean")

        prior_p = model.predict_proto(h, action0)
        target_dist = F.softmax(fine_logits_next, dim=-1)
        proto_loss = F.kl_div(F.log_softmax(prior_p, dim=-1), target_dist, reduction="batchmean")

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


# ==================== E0 消融臂（与 WorldModelH 同主干，仅改头） ====================

class NoYaoWorldModelH(WorldModelH):
    """A5 ours-yao：去掉三爻头 → 无相位信号、无卦门控、无原型辅助。
    用于检验相位结构是否是门控增益的因果来源（而非容量副产品）。"""

    has_yao_head = False        # 闭环里冻结接触通道（见 WorldModelH.rollout）

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256,
                 num_prototypes=32, prior_spec=PRIOR_DEFAULT):
        super().__init__(input_dim, obj_dim, action_dim, hidden_dim, num_prototypes,
                         yao_dim=6, prior_spec=prior_spec)
        del self.yao_head
        del self.prototypes
        del self.proto_pred

    def predict_yao(self, h):
        return torch.zeros(h.size(0), 6, device=h.device)


class SigmaWorldModelH(WorldModelH):
    """A6 ours+sigma：在 A3 上加 σ 头（异方差）+ CQR 覆盖保证。
    校准损失 NLL 均值解耦：predict_state detach，只有 σ 接收梯度。"""

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256,
                 num_prototypes=32, prior_spec=PRIOR_DEFAULT):
        super().__init__(input_dim, obj_dim, action_dim, hidden_dim, num_prototypes,
                         prior_spec=prior_spec)
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, obj_dim))
        nn.init.normal_(self.sigma_head[-1].weight, std=0.01)
        nn.init.zeros_(self.sigma_head[-1].bias)

    def predict_sigma(self, h, action):
        """逐维度 log-σ（softplus 保证正）。"""
        return F.softplus(self.sigma_head(torch.cat([h, action], dim=-1)))

    @torch.no_grad()
    def quantile_band(self, h, action, nominal=0.80):
        """名义 nominal 的中心带：mean ± z·σ，z 为标准正态分位点。"""
        mean = self.predict_state(h, action)
        sigma = self.predict_sigma(h, action)
        z = float(torch.erfinv(torch.tensor(nominal, device=mean.device)) * math.sqrt(2))
        return mean - z * sigma, mean + z * sigma


# ============================================================
# E1b 变体：相态容量分配（Phase-Aware Capacity Allocation）
# ============================================================

class PhaseCapacityWorldModelH(WorldModelH):
    """B-phase：状态主头拆为 8 个相态专用子头，输入经卦门控路由。

    与父类的差异仅在主头：8 个并行的子头各输出完整 obj_dim，
    输出 = Σ_g w_g(卦) · head_g([h,a])，w_g 由预测爻 logits 经 gua_from_yao
    得到的 8 卦概率给出（软路由，可微）。
    子头末层零初始化 ⇒ 起步输出 0 ⇒ 与父类一样起步=物理先验（公平起步）。

    参数量核对（hidden=256, obj_dim=22, action=1, sub_head_dim=64）：
      8 × [(257×64+64) + (64×22+22)] = 143 536
      父类 state_head                   = 137 238   （+4.6%，在 ±10% 内）
    """

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256,
                 num_prototypes=32, yao_dim=6, sub_head_dim=64):
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

        yao_log 布局（与 train_hinge 一致，原数组 (3 轴, 2 物体) 行主序）:
            [轴0_物0, 轴0_物1, 轴1_物0, 轴1_物1, 轴2_物0, 轴2_物1]
        手端（物体1）三爻 = 下标 1, 3, 5；sigmoid 后作为各爻伯努利概率，
        8 卦概率 = 三爻独立外积（软版本，可微）。
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
            w = self._gua_probs(yao_log)                         # (B, 8)
            for g in range(self.n_gua):
                out = out + w[:, g:g + 1] * self.sub_heads[g](x)
        else:
            # 无门控退化：8 头等权平均（保持零初始化下的先验起步）
            for g in range(self.n_gua):
                out = out + self.sub_heads[g](x) / self.n_gua
        return out

    def forward_state(self, h, action):
        """供 loss 调用（内部先算爻 logits 再门控）。"""
        yao_log = self.predict_yao(h)
        return self.predict_state(h, action, yao_log), yao_log


class PriorityHLoss(HLoss):
    """B-priority 损失：状态项按卦难度加权。

    难卦权重 w_g = max(1, error_g / mean_error) ，易卦权重 = 1；
    最后按数据集平均权重归一化，使整体损失量级与均匀训练一致
    （避免有效学习率漂移）。
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
        yao_log = model.predict_yao(h)
        huber = F.huber_loss(model.predict_state(h, action0, yao_log), target_delta,
                             reduction="none")                  # (B, obj_dim)
        per_sample = (huber * feat_w.view(1, -1)).mean(dim=-1)   # (B,)
        w = self._gua_weight[gua_idx]                                     # (B,)
        state_loss = (per_sample * w).mean()
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


# ============================================================
# 八卦混合专家动力学（理念 → 计算本体）
# "易有太极，是生两仪，两仪生四象，四象生八卦" 的架构化：
#   爻头 6 logits → 2 物体各 8 卦概率 → 64 卦象门控 → FiLM 调制状态主头
# 相态不再是预测标签，而是驱动状态转移的控制信号——卦变即计算变。
# 预注册消融对手：同参数量的自由 64 专家 MoE（无三爻结构先验）。
# ============================================================

class GuaMoEWorldModelH(WorldModelH):
    """八卦门控专家版铰链世界模型。

    与父类的差异仅在状态主头：head 的隐层被 FiLM(卦象) 调制——
      x' = γ(卦) ⊙ x + β(卦)
    卦象索引由两物体卦概率的外积构造（64 选 1，可微 soft 组合）。
    """

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256,
                 num_prototypes=32, yao_dim=6, free_moe=False,
                 prior_spec=PRIOR_DEFAULT):
        super().__init__(input_dim, obj_dim, action_dim, hidden_dim,
                         num_prototypes, yao_dim, prior_spec=prior_spec)
        self.free_moe = free_moe   # 消融开关：True = 自由门控（无三爻结构）
        self.n_gua = 64
        # 每卦 FiLM 参数（γ, β 各 hidden_dim）
        self.film = nn.Linear(self.n_gua, hidden_dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)     # 零初始化 → 起步无调制（=父类行为）

        # 结构门控（八卦）：由爻 logits 的物体级组合生成
        self.gua_gate = nn.Linear(6, self.n_gua)
        # 自由门控（消融）：直接由 h 生成 64 专家权重
        self.free_gate = nn.Linear(hidden_dim, self.n_gua)

        # 把 state_head 的隐层拆成可调制形式（覆盖父类：两段式）
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
        del self.state_head   # 移除父类一体化头（fi lm 版替代）

    def _gua_probs(self, h, yao_log):
        """返回 (B, 64) 卦象门控概率。
        结构模式：两物体卦概率外积（爻头 logits 经温度化 softmax）。
        自由模式：h 直接过 free_gate。"""
        if self.free_moe:
            return F.softmax(self.free_gate(h), dim=-1)
        # 每物体 8 卦分布（爻 3 位 → 8 类，用可学习线性从爻 logits 得到）
        logits_obj = self.gua_gate(yao_log)                     # (B, 64)
        # 结构先验：64 维 logits 重排为 8×8 外积形式（物体0卦 × 物体1卦）
        log8 = logits_obj.view(-1, 8, 8)
        p0 = F.softmax(log8.sum(dim=2) * 2.0, dim=-1)           # 物体0 的 8 卦（行）
        p1 = F.softmax(log8.sum(dim=1) * 2.0, dim=-1)           # 物体1 的 8 卦（列）
        outer = torch.einsum("bi,bj->bij", p0, p1)              # (B, 8, 8)
        return outer.reshape(-1, 64)

    def predict_state(self, h, action, yao_log=None):
        """FiLM 调制的残差头：mid → FiLM(卦象) → out。"""
        x = self.head_mid(torch.cat([h, action], dim=-1))
        if yao_log is not None:
            gate = self._gua_probs(h, yao_log)                   # (B, 64)
            film = self.film(gate)                               # (B, 2*hidden)
            gamma, beta = film.chunk(2, dim=-1)
            x = (1.0 + gamma) * x + beta
        return self.head_out(x)

    def forward_state(self, h, action):
        """供 loss 调用（内部先算爻 logits 再 FiLM）。"""
        yao_log = self.predict_yao(h)
        return self.predict_state(h, action, yao_log), yao_log


# ============================================================
# 多假设预测（T11 出路 + "生生之谓易"的计算化）
# 单点预测在混沌世界的时程上限（T11）只有分布预测能突破：
# K 个假设 = 摆动弹道的 K 条可能分支（"太极分两仪"的可能性树）。
# 训练：winner 软分配（防模式塌缩）；评估：oracle 曲线 + 分叉度。
# ============================================================

class MultiHypWorldModelH(WorldModelH):
    """多假设铰链世界模型：每步预测 K 条候选未来状态 + 置信度。"""

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256,
                 num_prototypes=32, yao_dim=6, n_hyp=8):
        super().__init__(input_dim, obj_dim, action_dim, hidden_dim,
                         num_prototypes, yao_dim)
        self.n_hyp = n_hyp
        self.hyp_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                          nn.GELU(),
                          nn.Linear(hidden_dim, obj_dim))
            for _ in range(n_hyp)])
        for head in self.hyp_heads:   # 零初始化 → 全假设起步=先验
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)
        self.conf_head = nn.Linear(hidden_dim + action_dim, n_hyp)

    def predict_hyps(self, h, action):
        """返回 (B, K, obj_dim) 假设（相对先验的残差经各自头输出）与 (B, K) 置信度。"""
        x = torch.cat([h, action], dim=-1)
        hyps = torch.stack([head(x) for head in self.hyp_heads], dim=1)
        conf = self.conf_head(x)
        return hyps, conf

    def rollout_hyp(self, x_seq, actions_fut, ss_prob=0.0):
        """beam 滚动：每条 beam 每步用自身 top-置信假设演化。
        返回 abs_preds: K 步 × (B, n_hyp, obj_dim)——每步 K 个假设的绝对状态。"""
        B = x_seq.size(0)
        h = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.gru_cell(x_seq[:, t], h)
        h = self.ln_h(h)
        obj_last = x_seq[:, -1, :self.obj_dim].unsqueeze(1).expand(-1, self.n_hyp, -1)
        ct_frozen = x_seq[:, -1, -3:]   # 冻结接触通道（GRU 输入 25 维需要）
        out_per_step = []
        K = actions_fut.size(1)
        for k in range(K):
            a_k = actions_fut[:, k]
            # 每条 beam 独立滚动：先验用各自 beam 状态，残差由各自假设头给出
            h_rep = h.unsqueeze(1).expand(-1, self.n_hyp, -1)              # (B, n_hyp, hidden)
            a_rep = a_k.unsqueeze(1).expand(-1, self.n_hyp, -1)            # (B, n_hyp, 1)
            prior_flat = self.prior_norm(
                obj_last.reshape(B * self.n_hyp, -1),
                a_rep.reshape(B * self.n_hyp, -1)).reshape(B, self.n_hyp, -1)
            x_in = torch.cat([h, a_k], dim=-1)                             # (B, hidden+1) 共享
            beam_delta = torch.stack([head(x_in) for head in self.hyp_heads], dim=1)
            obj_next = prior_flat + beam_delta                  # (B, n_hyp, 22)
            out_per_step.append(obj_next)
            # GRU 输入：beam 均值 + 冻结接触通道（25 维，与输入帧一致）
            x_next = torch.cat([obj_next.mean(dim=1), ct_frozen], dim=-1)
            h = self.ln_h(self.gru_cell(x_next, h))
            obj_last = obj_next
        return out_per_step, None

    def _head_input(self, h, a_k, beam_i):
        """假设头 i 的输入 [h, a]（h 共享——beam 间差异由先验状态与头 ID 承载）。"""
        return torch.cat([h, a_k], dim=-1)


# ============================================================
# 分位数分布预测 (G2v3) — T11 出路 + G2 合流解
# 单点预测在混沌世界的时程上限（T11）由分布预测突破：模型直接输出
# 未来状态的 K 个分位数（pinball 训练），发散由分位数带自然变宽表达。
# 高斯 NLL σ 路线已判负（G2 v1/v2）；分位数回归免高斯假设、免模式塌缩
# （分位数由 τ 唯一定义，不存在 winner 分配）。
# ============================================================

class QuantileWorldModelH(WorldModelH):
    """分位数铰链世界模型：每步预测未来状态的 K 个分位数（残差空间）。"""

    def __init__(self, input_dim=25, obj_dim=22, action_dim=1, hidden_dim=256,
                 num_prototypes=32, yao_dim=6,
                 quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
        super().__init__(input_dim, obj_dim, action_dim, hidden_dim,
                         num_prototypes, yao_dim)
        self.quantiles = tuple(quantiles)
        self.n_q = len(self.quantiles)
        self.q_idx_med = self.quantiles.index(0.5) if 0.5 in self.quantiles else self.n_q // 2
        self.q_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                          nn.GELU(),
                          nn.Linear(hidden_dim, obj_dim))
            for _ in range(self.n_q)])
        for head in self.q_heads:   # 零初始化 → 起步 = 物理先验（T1 基座）
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)

    def predict_q(self, h, action):
        """(B, n_q, obj_dim) 残差分位数。"""
        x = torch.cat([h, action], dim=-1)
        return torch.stack([head(x) for head in self.q_heads], dim=1)

    def rollout_q(self, x_seq, actions_fut, x_fut_gt=None, ss_prob=0.0):
        """闭环比滚动：每个分位数一条 beam（先验+各自分位数残差），
        GRU 用中位数 beam 的状态反馈。返回每步 (B, n_q, obj_dim) 绝对分位数。

        T17 修补（2026-09-03）：接触通道反馈从"冻结观测帧"改为爻头初爻软概率
        （与 WorldModelH.rollout 语义对齐——base 的闭环优势之一是 GRU 在
        scheduled-sampling 训练中学过消费自生成输入；冻结通道在相态转移后
        与预测状态矛盾）。
        """
        B = x_seq.size(0)
        h = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.gru_cell(x_seq[:, t], h)
        h = self.ln_h(h)
        obj_last = x_seq[:, -1, :self.obj_dim].unsqueeze(1).expand(-1, self.n_q, -1)
        ct_frozen = x_seq[:, -1, -3:]
        out_per_step = []
        K = actions_fut.size(1)
        for k in range(K):
            a_k = actions_fut[:, k]
            h_rep = h.unsqueeze(1).expand(-1, self.n_q, -1).reshape(B * self.n_q, -1)
            a_rep = a_k.unsqueeze(1).expand(-1, self.n_q, -1).reshape(B * self.n_q, -1)
            prior_flat = self.prior_norm(
                obj_last.reshape(B * self.n_q, -1),
                a_rep.reshape(B * self.n_q, -1)).reshape(B, self.n_q, -1)
            x_in = torch.cat([h, a_k], dim=-1)
            q_delta = torch.stack([head(x_in) for head in self.q_heads], dim=1)
            obj_next = prior_flat + q_delta                     # (B, n_q, 22)
            out_per_step.append(obj_next)
            med_state = obj_next[:, self.q_idx_med]             # 中位数 beam 反馈
            yao_log = self.predict_yao(h)                       # 爻反馈（T17 修补）
            ct_fb = torch.sigmoid(yao_log[:, 0:2].float())      # 初爻 2 物体软概率
            contact_fb = torch.cat([ct_fb, ct_frozen[:, 2:3]], dim=-1)
            x_next = torch.cat([med_state, contact_fb], dim=-1)
            if x_fut_gt is not None and ss_prob > 0:
                mask = (torch.rand(B, device=x_seq.device) < ss_prob).float().unsqueeze(1)
                x_next = mask * x_fut_gt[:, k] + (1 - mask) * x_next
            h = self.ln_h(self.gru_cell(x_next, h))
            obj_last = obj_next
        return out_per_step, None


def pinball_loss(y, q, taus):
    """分位数损失。y: (B, D) 真值；q: (B, n_q, D) 分位数预测；taus: (n_q,)。"""
    e = y.unsqueeze(1) - q
    t = taus.view(1, -1, 1)
    return torch.maximum(t * e, (t - 1.0) * e).mean()
