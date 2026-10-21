# -*- coding: utf-8 -*-
"""
观象世界模型 v3 (WorldModelV3)
==================================================
设计原则（对应 ROADMAP M1b / R8 / R9 / R10）：
  - 世界模型的本体职责交给【连续状态预测主头】：GRU 记忆 + 动作 → 下一帧物体状态
    （symlog 空间回归，Huber 损失，Dreamer 式稳健化）。v2 的教训：把预测塞进
    32 维离散瓶颈、且从不评估状态空间，导致"预测通道从未被训练"。
  - 离散原型层【降级为可解释副产物】：原型分配 + 意象流转预测保留为辅助损失，
    供 M2 的卦象分析，不再充当信息瓶颈。
  - 三爻头：初爻（接触，引擎真值）/ 中爻（能量趋势）/ 上爻（趋离意图）作为
    下一帧二元预测——"动爻即惊讶"的雏形，也是创新点 1/3 的落地。
  - 损失从 12+ 项减到 5 项；事件分类头、PAST 解码器、free_energy/void 全部移除
    （事件 ≈ 初爻翻转；重构由状态主头取代）。

输入组装（layout.assemble_frame）：[物体 33 | 障碍 80 | 接触 3] = 116 维/帧
状态主头预测目标：下一帧的物体段（33 维，归一化后取 symlog）。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import physics_prior_spec_v3


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symlog_inv(y):
    return torch.sign(y) * torch.expm1(torch.abs(y))


class WorldModelV3(nn.Module):
    """GRU 主干 + 连续状态主头 + 三爻头 + 原型辅助层。"""

    def __init__(self, input_dim=116, obj_dim=33, action_dim=6, hidden_dim=256,
                 num_prototypes=32, num_coarse=8, tau=0.05, gravity=-900.0,
                 impulse_max=25.0, use_gua=False, gua_emb_dim=16, use_innov=True,
                 innov_dim=None, prior_spec=None):
        super().__init__()
        self.input_dim = input_dim
        self.obj_dim = obj_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        self.num_coarse = num_coarse
        # 物理先验常数（M1d：惯性残差基座）
        self.tau = float(tau)
        self.gravity = float(gravity)
        self.impulse_max = float(impulse_max)
        # SLOT-1b：先验档位（physics_prior_spec_v3.SPECS）；None/默认 = legacy 闭式，
        # 行为与历史 checkpoint 逐位一致
        self.prior_spec = prior_spec or "parabola005"
        # M2b：构造性相位条件化（卦 = 三爻位模式，by-construction）
        self.use_gua = bool(use_gua)
        self.gua_emb_dim = int(gua_emb_dim)
        # G2-v2：创新量特征（逐帧物理先验-观测失配）——σ 头的"惊讶"输入
        self.use_innov = bool(use_innov)
        self.innov_dim = int(innov_dim) if innov_dim is not None else 3 * obj_dim
        # 归一化统计注册为 buffer → 随 checkpoint 保存，prior_norm 自包含
        self.register_buffer("mean_obj", torch.zeros(obj_dim))
        self.register_buffer("std_obj", torch.ones(obj_dim))

        # 主干：GRUCell 逐步演化（训练用序列一次喂入，rollout 用单步演化，共享权重）
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.ln_h = nn.LayerNorm(hidden_dim)
        if self.use_gua:
            self.gua_emb = nn.Embedding(8, self.gua_emb_dim)

        # 主头：状态预测（残差空间）——世界模型本体
        head_in = hidden_dim + action_dim + (self.gua_emb_dim if self.use_gua else 0)
        self.state_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, obj_dim),
        )
        # 输出层小初始化：起步接近"预测=0"（symlog 空间的均值附近），避免开局爆炸
        nn.init.normal_(self.state_head[-1].weight, std=0.01)
        nn.init.zeros_(self.state_head[-1].bias)

        # G2：异方差不确定度头（"术"的 v3 实体化，精度 = 1/σ²）
        # G2-v2：输入加入逐帧创新量特征（prior(x_t) 与 x_t 的失配）——
        # 池化 h 无法做逐帧一致性推断（G2 v1 判决 P2.1/P2.3 失败的根因假设）
        sig_in = hidden_dim + action_dim
        if self.use_innov:
            sig_in += self.innov_dim
        if self.use_gua:
            sig_in += self.gua_emb_dim
        self.sigma_head = nn.Sequential(
            nn.Linear(sig_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, obj_dim),
        )
        nn.init.normal_(self.sigma_head[-1].weight, std=0.01)
        nn.init.zeros_(self.sigma_head[-1].bias)   # 初始 σ=1（归一化空间），NLL 自动收缩

        # 三爻头：9 个 logits = [初爻接触(3) | 中爻能量趋势(3) | 上爻趋离(3)]
        self.yao_head = nn.Linear(hidden_dim, 9)

        # 意象层（辅助，可解释）：原型库 + 流转预测
        protos = torch.nn.init.orthogonal_(torch.empty(num_prototypes, hidden_dim))
        self.prototypes = nn.Parameter(F.normalize(protos, p=2, dim=-1))
        coarse = torch.nn.init.orthogonal_(torch.empty(num_coarse, hidden_dim))
        self.coarse_prototypes = nn.Parameter(F.normalize(coarse, p=2, dim=-1))
        self.proto_pred = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_prototypes),
        )
        self.coarse_pred = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_coarse),
        )

    def set_stats(self, mean_obj, std_obj):
        """训练侧注入归一化统计（随 state_dict 保存/恢复）。"""
        with torch.no_grad():
            self.mean_obj.copy_(torch.as_tensor(mean_obj, dtype=self.mean_obj.dtype))
            self.std_obj.copy_(torch.as_tensor(std_obj, dtype=self.std_obj.dtype))

    # ---------------- 编码 ----------------

    def encode_step(self, x_t, h):
        """单步演化：x_t (B, input_dim), h (B, hidden) → h_next。"""
        return self.gru_cell(x_t, h)

    def encode_sequence(self, x_seq):
        """(B, L, input_dim) → (h_last, c_norm)。"""
        h = torch.zeros(x_seq.size(0), self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.gru_cell(x_seq[:, t], h)
        h = self.ln_h(h)
        c_norm = F.normalize(h, p=2, dim=-1)
        return h, c_norm

    def evidence(self, c_norm, detached_protos=True):
        """原型证据 logits（余弦相似度）。detached_protos: 证据分配不经原型梯度
        （EMA 负责原型移动，梯度只推 c——与 v2 的 B4 结论一致）。"""
        p = self.prototypes.detach() if detached_protos else self.prototypes
        pc = self.coarse_prototypes.detach() if detached_protos else self.coarse_prototypes
        fine_logits = torch.mm(c_norm, p.t())
        coarse_logits = torch.mm(c_norm, pc.t())
        return fine_logits, coarse_logits

    # ---------------- 预测 ----------------

    def prior_norm(self, obj_norm, action_norm):
        """无接触物理先验（归一化空间）。

        obj_norm: (B, 33) 上帧物体段（归一化）；action_norm: (B, 6) 归一化净冲量。
        返回 (B, 33) 先验的归一化物体段。自由飞行段残差应为 0。

        prior_spec="parabola005"（默认）走 legacy 闭式抛物线（逐位兼容历史）；
        其余档位委托 physics_prior_spec_v3（SLOT-1b 阶梯，显式欧拉引擎类）。
        """
        raw = obj_norm * self.std_obj + self.mean_obj
        imp = action_norm * self.impulse_max
        out = raw.clone()
        if self.prior_spec != "parabola005":
            return physics_prior_spec_v3.prior_norm_torch(
                self.prior_spec, obj_norm, action_norm,
                self.mean_obj, self.std_obj, self.impulse_max)
        g, tau = self.gravity, self.tau
        for o in range(3):
            b = o * 11
            x, y = raw[:, b + 1], raw[:, b + 2]
            st, ct_ = raw[:, b + 3], raw[:, b + 4]
            vx, vy, w = raw[:, b + 5], raw[:, b + 6], raw[:, b + 7]
            mass = torch.where(raw[:, b + 10] > 1e-3, raw[:, b + 10],
                               torch.ones_like(raw[:, b + 10]))
            theta2 = torch.atan2(st, ct_) + w * tau
            out[:, b + 1] = x + vx * tau
            out[:, b + 2] = y + vy * tau + 0.5 * g * tau * tau
            out[:, b + 3] = torch.sin(theta2)
            out[:, b + 4] = torch.cos(theta2)
            out[:, b + 5] = vx + imp[:, 2 * o] / mass
            out[:, b + 6] = vy + imp[:, 2 * o + 1] / mass + g * tau
        return (out - self.mean_obj) / (self.std_obj + 1e-8)

    def predict_state(self, h, action, gua_idx=None):
        """主头：[h, action(, 卦嵌入)] → 下一帧物体段的【残差】（归一化空间）。

        [残差参数化] 首版让头直接输出绝对状态（symlog 空间），实测卡在 25px
        系统性外推偏差。改为残差：pred = prior + head([h,a])，输出层零初始化 ⇒
        模型起步即物理先验，网络只学修正（重力/冲量/碰撞/积分格式差）。
        [M2b 相位条件化] use_gua 时把当前三爻卦象（8 选 1 的构造性相位变量）
        经嵌入拼入——检验"显式相位变量"是否优于让 GRU 自己隐式学相位。
        """
        xs = [h, action]
        if self.use_gua:
            if gua_idx is not None:
                xs.append(self.gua_emb(gua_idx))
            else:
                xs.append(torch.zeros(h.size(0), self.gua_emb_dim, device=h.device))
        x = torch.cat(xs, dim=-1)
        return self.state_head(x)

    @staticmethod
    def gua_from_yao(yao_log):
        """由爻 logits (B,9) 得窗口级卦索引（三轴各自对物体取均值后 0.5 阈值）。"""
        p = torch.sigmoid(yao_log.float())
        rates = p.view(p.size(0), 3, 3).mean(dim=2)          # (B, 3)
        bits = (rates > 0.5).long()
        return bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]

    def predict_yao(self, h):
        """三爻头：h → 9 logits [初(3)|中(3)|上(3)]。"""
        return self.yao_head(h)

    def innovation_feats(self, x_seq, a_win):
        """G2-v2：逐帧物理先验-观测失配特征（σ 头的"惊讶"输入）。

        x_seq: (B, L, input_dim)；a_win: (B, L-1, action_dim) 窗口内各转移动作（归一化）。
        innov_t = |prior_norm(obj_{t-1}, a_t) − obj_t|，汇总为 [逐维均值 | 逐维最大 | 最后帧]。
        """
        B, L = x_seq.shape[0], x_seq.shape[1]
        obj = x_seq[:, :, :self.obj_dim]
        N = B * (L - 1)
        prev = obj[:, :-1, :].reshape(N, self.obj_dim)
        cur = obj[:, 1:, :].reshape(N, self.obj_dim)
        a = a_win.reshape(N, a_win.size(-1))
        pr = self.prior_norm(prev, a)                      # (N, 33)
        innov = (pr - cur).abs().reshape(B, L - 1, self.obj_dim)
        return torch.cat([innov.mean(dim=1), innov.max(dim=1).values,
                          innov[:, -1]], dim=-1)

    def predict_sigma(self, h, action, innov=None, gua_idx=None):
        """G2 不确定度头：[h, action(, 创新量)(, 卦嵌入)] → 逐维 log σ²（归一化空间）。"""
        xs = [h, action]
        if self.use_innov:
            if innov is not None:
                xs.append(innov)
            else:
                xs.append(torch.zeros(h.size(0), self.innov_dim, device=h.device))
        if self.use_gua:
            if gua_idx is not None:
                xs.append(self.gua_emb(gua_idx))
            else:
                xs.append(torch.zeros(h.size(0), self.gua_emb_dim, device=h.device))
        return self.sigma_head(torch.cat(xs, dim=-1))

    def predict_proto(self, h, action):
        """意象流转预测（辅助）：[h, action] → 下一帧原型/粗原型 logits。"""
        x = torch.cat([h, action], dim=-1)
        return self.proto_pred(x), self.coarse_pred(x)

    def assemble_next_input(self, obj_pred_norm, obstacles_last, contact_fb):
        """rollout 用：组装下一帧输入。接触通道由调用方按反馈模式给出（见 rollout）。"""
        return torch.cat([obj_pred_norm, obstacles_last, contact_fb], dim=-1)

    @torch.no_grad()
    def ema_prototypes(self, c_norm, det_assign, det_coarse, gamma=0.99):
        """VQ 风格 EMA 原型更新（无梯度）。"""
        for protos, assign in ((self.prototypes, det_assign),
                               (self.coarse_prototypes, det_coarse)):
            K, D = protos.size(0), protos.size(1)
            sums = torch.zeros(K, D, device=protos.device).index_add_(0, assign, c_norm)
            counts = torch.bincount(assign, minlength=K).clamp(min=1).float()
            means = sums / counts.unsqueeze(1)
            new = gamma * protos.data + (1 - gamma) * means
            protos.data.copy_(F.normalize(new, p=2, dim=-1))

    # ---------------- 多步 rollout ----------------

    def rollout(self, x_seq, actions_fut, obstacles_last, x_fut_gt=None, ss_prob=0.0,
                gua0=None, contact_mode="soft"):
        """闭环 rollout：h 从 x_seq 演化，之后每步 物理先验 + 残差 组装下一输入。

        参数:
            x_seq:         (B, L, input_dim) 历史窗口
            actions_fut:   (B, K, action_dim) 未来各步动作（归一化后）
            obstacles_last: (B, OBS_DIM) 障碍槽位（归一化，片段常量）
            x_fut_gt:      (B, K, input_dim) 未来真值帧（scheduled sampling 用，可选）
            ss_prob:       每步每样本用真值帧替代自身预测的概率（0=纯闭环）
            gua0:          (B,) 上帧（最后输入帧）的卦索引；use_gua 时第 0 步条件化，
                           之后各步用模型自身动爻预测推相（闭环可部署路径）
            contact_mode:  闭环反馈通路（M1e 消融）：
                           "soft"  = 三爻头接触概率软值（旧行为，假设的退化根因）
                           "hard"  = 三爻头动爻硬判决（0/1）
                           "freeze"= 冻结最后观测接触（永不反馈自身误差）
        返回:
            abs_preds:     K 个 (B, obj_dim) —— 每步绝对状态（归一化）= 先验 + 残差
            yao_preds:     K 个 (B, 9) 三爻 logits
        """
        B = x_seq.size(0)
        h = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.gru_cell(x_seq[:, t], h)
        h = self.ln_h(h)

        obj_last = x_seq[:, -1, :self.obj_dim]   # 上帧物体段（归一化）
        ct_frozen = x_seq[:, -1, -3:]            # 最后观测接触（freeze 模式用）
        abs_preds, yao_preds = [], []
        K = actions_fut.size(1)
        cur_gua = gua0
        for k in range(K):
            a_k = actions_fut[:, k]
            prior_k = self.prior_norm(obj_last, a_k)
            delta = self.predict_state(h, a_k, cur_gua)
            yao_log = self.predict_yao(h)
            obj_next = prior_k + delta
            abs_preds.append(obj_next)
            yao_preds.append(yao_log)
            if contact_mode == "hard":
                contact_fb = (torch.sigmoid(yao_log[:, 0:3].float()) > 0.5).float()
            elif contact_mode == "freeze":
                contact_fb = ct_frozen
            else:
                contact_fb = torch.sigmoid(yao_log[:, 0:3].float())
            if x_fut_gt is not None and ss_prob > 0:
                mask = (torch.rand(B, device=x_seq.device) < ss_prob).float().unsqueeze(1)
                x_next = mask * x_fut_gt[:, k] + (1 - mask) * \
                    self.assemble_next_input(obj_next, obstacles_last, contact_fb)
            else:
                x_next = self.assemble_next_input(obj_next, obstacles_last, contact_fb)
            h = self.ln_h(self.gru_cell(x_next, h))
            obj_last = obj_next
            if self.use_gua:
                cur_gua = self.gua_from_yao(yao_log.detach())
        return abs_preds, yao_preds


class GatedWorldModelV3(WorldModelV3):
    """SLOT-1c：静默门控头——g=σ(MLP([h,a])) 逐样本乘在残差输出上。

    动机（SLOT-1b 头干涉噪声地板）：接触主导的训练混合使头在先验已精确的
    FF regime 输出 ~0.85px 噪声。门给"静默"一条专用通路：接触帧必须开门
    （否则 mixed 恶化自罚），FF 帧可关门。门偏置初始化 +4 ⇒ 起步 g≈0.982
    ≈ 基线行为（公平起步）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, 4.0)

    def gate_logits(self, h, action):
        return self.gate_head(torch.cat([h, action], dim=-1)).squeeze(-1)

    def predict_state(self, h, action, gua_idx=None):
        base = super().predict_state(h, action, gua_idx)
        g = torch.sigmoid(self.gate_logits(h, action)).unsqueeze(-1)
        return g * base


class V3Loss(nn.Module):
    """v3 精简损失（5 项，全部可单独消融）。

    L = w_state · Huber(状态主头残差, 残差目标)
      + w_yao  · BCE(三爻头)
      + w_proto· KL(意象流转 || 下一帧证据)
      + w_commit·(1 - cos(h, 分配原型))
      + w_usage · (log K - 批次分配熵)   [防原型坍缩]
    """

    def __init__(self, w_state=1.0, w_yao=0.1, w_proto=0.05, w_commit=0.01, w_usage=0.02,
                 w_cal=0.2):
        super().__init__()
        self.w_state = w_state
        self.w_yao = w_yao
        self.w_proto = w_proto
        self.w_commit = w_commit
        self.w_usage = w_usage
        self.w_cal = w_cal

    def forward(self, model, h, c_norm, fine_logits, coarse_logits,
                action0, target_delta, yao_target, fine_logits_next, feat_w, gua=None,
                innov=None, sample_w=None):
        device = h.device
        K = model.num_prototypes

        # 1) 状态主损失（残差参数化：预测"相对物理先验的修正量"，维度加权 Huber）
        #    SLOT-1c：sample_w 为逐样本 regime 权重（None=均匀，历史行为不变）
        huber = F.huber_loss(model.predict_state(h, action0, gua), target_delta,
                             reduction="none")
        per_sample = (huber * feat_w.view(1, -1)).mean(dim=1)
        state_loss = (per_sample * sample_w).mean() if sample_w is not None \
            else per_sample.mean()

        # 2) 三爻 BCE
        yao_log = model.predict_yao(h)
        yao_loss = F.binary_cross_entropy_with_logits(
            yao_log, yao_target.float(), reduction="mean")

        # 3) 意象流转 KL（目标 = 下一帧证据分布）
        prior_p, _ = model.predict_proto(h, action0)
        target_dist = F.softmax(fine_logits_next, dim=-1)
        proto_loss = F.kl_div(F.log_softmax(prior_p, dim=-1), target_dist, reduction="batchmean")

        # 4) commit：c 朝分配原型（单方向；原型由 EMA 移动）
        assigned = model.prototypes.detach()[fine_logits.argmax(dim=-1)]
        commit_loss = (1.0 - F.cosine_similarity(assigned, c_norm, dim=-1)).mean()

        # 5) 使用率：批次分配分布的熵越接近 log K 越好
        assignment = F.softmax(fine_logits.float(), dim=-1)
        batch_avg = assignment.mean(dim=0).clamp(min=1e-8)
        batch_entropy = -(batch_avg * batch_avg.log()).sum()
        usage_loss = (math.log(K) - batch_entropy) / math.log(K)

        # 6) G2 校准损失：高斯 NLL（异方差 σ 头，"术"的实体化）
        #    G2-v2：均值解耦（NLL 不回传均值——均值由 Huber 单独负责）
        calib_loss = torch.tensor(0.0, device=device)
        if self.w_cal > 0 and hasattr(model, "sigma_head"):
            logvar = model.predict_sigma(h, action0, innov, gua).clamp(-8.0, 8.0)
            mean_det = model.predict_state(h, action0, gua).detach()
            nll = 0.5 * (torch.exp(-logvar) * (target_delta - mean_det) ** 2 + logvar)
            calib_loss = (nll * feat_w.view(1, -1)).mean()

        total = (self.w_state * state_loss + self.w_yao * yao_loss
                 + self.w_proto * proto_loss + self.w_commit * commit_loss
                 + self.w_usage * usage_loss + self.w_cal * calib_loss)
        parts = dict(state=state_loss.item(), yao=yao_loss.item(),
                     proto=proto_loss.item(), commit=commit_loss.item(),
                     usage=usage_loss.item(), calib=calib_loss.item())
        return total, parts, yao_log


if __name__ == "__main__":
    # 维度自检
    m = WorldModelV3()
    x = torch.randn(4, 8, 116)
    h, c = m.encode_sequence(x)
    fl, cl = m.evidence(c)
    a = torch.randn(4, 6)
    delta = m.predict_state(h, a)
    pn = m.prior_norm(x[:, -1, :33], a)
    yl = m.predict_yao(h)
    aps, yps = m.rollout(x, torch.randn(4, 3, 6), torch.randn(4, 80))
    print(f"h {h.shape} | fine {fl.shape} | delta {delta.shape} | prior {pn.shape} | yao {yl.shape} "
          f"| rollout abs {[tuple(s.shape) for s in aps]}")
    print("零初始化下残差应为 0:", float(delta.abs().max()))
    print("模型参数量:", sum(p.numel() for p in m.parameters()))
