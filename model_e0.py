# -*- coding: utf-8 -*-
"""
E0 统一主表 — 臂注册表 (model_e0.py)
==================================================
9 个臂的公平性底座：
  (1) 全部共享 GRU 主干语义，参数量落在 ours-base (402,236) 的 ±10% 内；
  (2) 所有状态输出层零初始化（std=0.01, bias=0），起步 = 先验/零预测；
  (3) 由 state_dict 反推结构的接线断言（tests/test_wiring_e0.py），
      不以 config 为准（M2b 教训）。

参数量调平（实测，hidden 已调）：
  A0 zero_vel   0          (无训练参数)
  A1 inertia    0          (解析先验)
  A2 noprior    402,151    hidden=273   (-0.02%)
  A3 ours_base  402,236    hidden=256   (基准)
  A4 ours_gua   386,620    hidden=256   (-3.9%)
  A5 ours_yao   402,151    hidden=273   (-0.02%)
  A6 ours_sigma 402,702    hidden=234   (+0.12%)
  A7 jepa       430,823    hidden=219   (+7.1%，含目标投影器)
  A8 strong_mlp 401,276    hidden=208   (-0.24%，动作拼接到状态头)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core_model import NoPriorModel, JEPALikeModel
from layout_h import (OBJ_DIMS, CONTACT_DIMS, ACTION_DIM, L1, L2, M1, M2,
                      TAU_MAX, DT as DT_H, obj_base, F as LF)
from model_h import (WorldModelH, GuaMoEWorldModelH, NoYaoWorldModelH,
                     SigmaWorldModelH, prior_step_torch)
from physics_prior_spec import DEFAULT_SPEC, SPECS, prior_norm_torch

INPUT_DIM = OBJ_DIMS + CONTACT_DIMS   # 22 + 3 = 25

# ---------------------------------------------------------------- 臂规格
ARM_SPECS = {
    "zero_vel":   dict(cls=None,           hidden=256, trainable=False,
                       tag="A0 zero-vel",  note="上帧状态静止（下界）"),
    "inertia":    dict(cls=None,           hidden=256, trainable=False,
                       tag="A1 inertia",   note="layout_h.acrobot_step 解析先验",
                       prior_spec="euler005"),
    # ---- 先验质量阶梯的解析档位（0 参数，无需训练；P0-a 的"地板"参照）----
    "inertia_euler001x5": dict(cls=None,   hidden=256, trainable=False,
                       tag="A1b inertia·euler@0.01×5", note="细化积分器（T19 M-F）",
                       prior_spec="euler001x5"),
    "inertia_rk4005": dict(cls=None,       hidden=256, trainable=False,
                       tag="A1c inertia·rk4@0.05", note="RK4 粗步",
                       prior_spec="rk4005"),
    "inertia_rk4001x5": dict(cls=None,     hidden=256, trainable=False,
                       tag="A1d inertia·rk4@0.01×5", note="与数据生成器同积分器（精确）",
                       prior_spec="rk4001x5"),
    "noprior":    dict(cls=NoPriorModel,   hidden=273, trainable=True,
                       tag="A2 noprior",   note="GRU + 直接状态头，无先验/爻/原型"),
    "ours_base":  dict(cls=WorldModelH,    hidden=256, trainable=True,
                       tag="A3 ours-base", note="prior + 残差头 + 三爻头 + 原型辅助"),
    "ours_gua":   dict(cls=GuaMoEWorldModelH, hidden=256, trainable=True,
                       tag="A4 ours+gua",  note="A3 + GuaMoE（FiLM 64 卦）"),
    "ours_yao":   dict(cls=NoYaoWorldModelH, hidden=273, trainable=True,
                       tag="A5 ours-yao",  note="A3 去掉 yao 头（消融相位信号）"),
    "ours_sigma": dict(cls=SigmaWorldModelH, hidden=234, trainable=True,
                       tag="A6 ours+sigma",note="A3 + σ 头 + CQR"),
    "jepa":       dict(cls=JEPALikeModel,  hidden=219, trainable=True,
                       tag="A7 JEPA-like", note="GRU + latent 预测 + EMA target + linear probe"),
    "strong_mlp": dict(cls=None,           hidden=208, trainable=True,
                       tag="A8 strong-MLP",note="2 层 MLP(512→hidden) + LSTM，无先验"),
}


class ZeroVelModel(nn.Module):
    """A0：上帧状态静止（零速度）。无训练参数；先验 = 恒等（返回归一化末帧）。"""

    # A0 的"先验"就是零速度预测器本身（上帧=下一帧），故计入先验臂：
    # 这样 one_step_model / 先验列 都等于零速度基线 20.6px，语义自洽。
    physics_prior = True

    def __init__(self, input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=ACTION_DIM):
        super().__init__()
        self.obj_dim, self.action_dim = obj_dim, action_dim
        self.register_buffer("mean_obj", torch.zeros(obj_dim))
        self.register_buffer("std_obj", torch.ones(obj_dim))

    def set_stats(self, mean_obj, std_obj):
        with torch.no_grad():
            self.mean_obj.copy_(torch.as_tensor(mean_obj, dtype=self.mean_obj.dtype))
            self.std_obj.copy_(torch.as_tensor(std_obj, dtype=self.std_obj.dtype))

    def encode_sequence(self, x_seq):
        h = torch.zeros(x_seq.size(0), 1, device=x_seq.device)
        return h, h

    def predict_state(self, h, action):
        return torch.zeros(h.size(0), self.obj_dim, device=h.device)

    def prior_norm(self, obj_norm, action):
        """恒等：预测 = 上一帧（零速度假设）"""
        return obj_norm

    def predict_yao(self, h):
        return torch.zeros(h.size(0), 6, device=h.device)

    def rollout(self, x_seq, actions_fut, x_fut_gt=None, ss_prob=0.0):
        B = x_seq.size(0)
        obj_last = x_seq[:, -1, :self.obj_dim]
        return [obj_last.clone() for _ in range(actions_fut.size(1))], None


class InertiaModel(nn.Module):
    """A1：纯解析先验（0 参数）。state head 零初始化 → 预测 = 先验。"""

    physics_prior = True

    def __init__(self, input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=ACTION_DIM,
                 prior_spec=DEFAULT_SPEC):
        super().__init__()
        self.obj_dim, self.action_dim = obj_dim, action_dim
        self.prior_spec = prior_spec
        self.register_buffer("mean_obj", torch.zeros(obj_dim))
        self.register_buffer("std_obj", torch.ones(obj_dim))

    def set_stats(self, mean_obj, std_obj):
        with torch.no_grad():
            self.mean_obj.copy_(torch.as_tensor(mean_obj, dtype=self.mean_obj.dtype))
            self.std_obj.copy_(torch.as_tensor(std_obj, dtype=self.std_obj.dtype))

    def encode_sequence(self, x_seq):
        h = torch.zeros(x_seq.size(0), 1, device=x_seq.device)
        return h, h

    def predict_state(self, h, action):
        return torch.zeros(h.size(0), self.obj_dim, device=h.device)

    def prior_norm(self, obj_norm, action):
        """解析先验；实现委托 physics_prior_spec（spec = self.prior_spec）。"""
        return prior_norm_torch(self.prior_spec, obj_norm, action,
                                self.mean_obj, self.std_obj)

    def predict_yao(self, h):
        return torch.zeros(h.size(0), 6, device=h.device)

    def rollout(self, x_seq, actions_fut, x_fut_gt=None, ss_prob=0.0):
        B = x_seq.size(0)
        obj_last = x_seq[:, -1, :self.obj_dim]
        ct_frozen = x_seq[:, -1, -3:]
        abs_preds = []
        for k in range(actions_fut.size(1)):
            a_k = actions_fut[:, k]
            obj_next = self.prior_norm(obj_last, a_k)
            abs_preds.append(obj_next)
            obj_last = obj_next
        return abs_preds, None


class StrongMLP(nn.Module):
    """A8：强学习基线 — 2 层 MLP + LSTM，无物理先验、无爻、无原型。
    hidden 经调平至 ≈401k（-0.24% vs ours-base）。动作拼接到状态头输入。"""

    physics_prior = False       # prior_norm 返回全 0（纯学习，直接输出绝对状态）

    def __init__(self, input_dim=INPUT_DIM, obj_dim=OBJ_DIMS, action_dim=ACTION_DIM,
                 hidden_dim=208):
        super().__init__()
        self.obj_dim, self.action_dim, self.hidden_dim = obj_dim, action_dim, hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # 头接受 [h, action]（与 ours-base 的残差头一致）
        self.head = nn.Linear(hidden_dim + action_dim, obj_dim)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("mean_obj", torch.zeros(obj_dim))
        self.register_buffer("std_obj", torch.ones(obj_dim))

    def set_stats(self, mean_obj, std_obj):
        with torch.no_grad():
            self.mean_obj.copy_(torch.as_tensor(mean_obj, dtype=self.mean_obj.dtype))
            self.std_obj.copy_(torch.as_tensor(std_obj, dtype=self.std_obj.dtype))

    def forward(self, x_seq, action):
        B, L, _ = x_seq.shape
        h = self.mlp(x_seq.reshape(B * L, -1)).reshape(B, L, -1)
        h, _ = self.lstm(h)
        return self.head(h[:, -1])

    def encode_sequence(self, x_seq):
        B, L, _ = x_seq.shape
        h = self.mlp(x_seq.reshape(B * L, -1)).reshape(B, L, -1)
        h, _ = self.lstm(h)
        h = h[:, -1]
        return h, F.normalize(h, p=2, dim=-1)

    def predict_state(self, h, action):
        return self.head(torch.cat([h, action], dim=-1))

    def prior_norm(self, obj_norm, action):
        return torch.zeros_like(obj_norm)

    def predict_yao(self, h):
        return torch.zeros(h.size(0), 6, device=h.device)

    def rollout(self, x_seq, actions_fut, x_fut_gt=None, ss_prob=0.0):
        B, L, _ = x_seq.shape
        h = self.mlp(x_seq.reshape(B * L, -1)).reshape(B, L, -1)
        _, (h, c) = self.lstm(h)          # (1, B, D)
        h, c = h[0], c[0]
        ct_frozen = x_seq[:, -1, -3:]
        abs_preds = []
        for k in range(actions_fut.size(1)):
            a_k = actions_fut[:, k]
            obj_next = self.head(torch.cat([h, a_k], dim=-1))
            abs_preds.append(obj_next)
            x_next = torch.cat([obj_next, ct_frozen], dim=-1)
            if x_fut_gt is not None and ss_prob > 0:
                mask = (torch.rand(B, device=x_seq.device) < ss_prob).float().unsqueeze(1)
                x_next = mask * x_fut_gt[:, k] + (1 - mask) * x_next
            h2 = self.mlp(x_next).unsqueeze(1)
            _, (h, c) = self.lstm(h2, (h.unsqueeze(0), c.unsqueeze(0)))
            h, c = h[0], c[0]
        return abs_preds, None


# ---------------------------------------------------------------- 工厂
def make_model(arm, prior_spec=None):
    """按臂名构造模型（含调平后的 hidden_dim）。

    prior_spec：先验积分器档位（physics_prior_spec.SPECS）。
      · None → 用臂自带的 spec（解析臂），否则用 DEFAULT_SPEC（= 旧行为 euler005）
      · 训练时由 train_e0.py --prior_spec 传入，并写入 checkpoint args
    """
    spec = ARM_SPECS[arm]
    cls = spec["cls"]
    h = spec["hidden"]
    spec_name = prior_spec or spec.get("prior_spec") or DEFAULT_SPEC
    if arm == "zero_vel":
        return ZeroVelModel()
    if arm == "inertia" or arm.startswith("inertia_"):
        return InertiaModel(prior_spec=spec_name)
    if arm == "strong_mlp":
        return StrongMLP(hidden_dim=h)
    if arm == "noprior":
        return cls(hidden_dim=h)          # NoPriorModel 无先验，不收 prior_spec
    if arm == "jepa":
        return cls(hidden_dim=h, latent_dim=128)
    if arm == "ours_gua":
        return cls(hidden_dim=h, free_moe=False, prior_spec=spec_name)
    if arm == "ours_yao":
        return cls(hidden_dim=h, prior_spec=spec_name)
    if arm == "ours_sigma":
        return cls(hidden_dim=h, prior_spec=spec_name)
    # ours_base / noprior
    return cls(hidden_dim=h, prior_spec=spec_name)


# ---------------------------------------------------------------- 接线断言
EXPECTED = {
    "noprior":     ["trunk.gru_cell.weight_ih", "trunk.ln_h.weight", "state_head.0.weight",
                    "state_head.2.weight", "state_head.4.weight"],
    "ours_base":   ["gru_cell.weight_ih", "ln_h.weight", "state_head.0.weight",
                    "state_head.2.weight", "state_head.4.weight",
                    "yao_head.weight", "prototypes", "proto_pred.0.weight"],
    "ours_gua":    ["gru_cell.weight_ih", "ln_h.weight", "head_mid.0.weight",
                    "head_out.1.weight", "film.weight", "gua_gate.weight",
                    "prototypes", "proto_pred.0.weight"],
    "ours_yao":    ["gru_cell.weight_ih", "ln_h.weight", "state_head.0.weight",
                    "state_head.2.weight", "state_head.4.weight"],
    "ours_sigma":  ["gru_cell.weight_ih", "ln_h.weight", "state_head.0.weight",
                    "yao_head.weight", "prototypes", "proto_pred.0.weight",
                    "sigma_head.0.weight"],
    "jepa":        ["context_encoder.weight_ih", "target_encoder.weight_ih",
                    "predictor.0.weight", "predictor.2.weight",
                    "target_projector.weight", "probe.weight", "ln_h.weight"],
    "strong_mlp":  ["mlp.0.weight", "mlp.2.weight", "lstm.weight_ih_l0", "head.weight"],
}
ABSENT = {
    "ours_gua":    ["state_head.0.weight"],        # 被 head_mid / head_out 替代
    "ours_yao":    ["yao_head.weight"],
    "ours_base":   ["sigma_head.0.weight"],
    "ours_sigma":  [],
}


def assert_wiring(arm, sd):
    """由 state_dict 反推结构，断言预期模块存在 / 不应存在的模块缺失。
    返回 (ok, problems)。"""
    if arm in ("zero_vel", "inertia"):
        return True, []
    problems = []
    for key in EXPECTED.get(arm, []):
        if key not in sd:
            problems.append(f"预期模块缺失: {key}")
    for key in ABSENT.get(arm, []):
        if key in sd:
            problems.append(f"不应存在的模块（消融未生效）: {key}")
    # 状态输出层零初始化检查
    last = [k for k in sd if k.endswith("weight") and
            ("state_head.4" in k or "head_out.1" in k or "head.weight" in k
             or "probe.weight" in k)]
    for k in last:
        std = float(sd[k].std().item())
        if std > 0.05:
            problems.append(f"状态输出层未零初始化: {k} std={std:.4f} (应 ≈0.01)")
    return len(problems) == 0, problems


def param_table():
    """打印所有臂的实测参数量表（用于报告）。"""
    rows = []
    base = sum(p.numel() for p in make_model("ours_base").parameters())
    for arm, spec in ARM_SPECS.items():
        if spec["trainable"]:
            n = sum(p.numel() for p in make_model(arm).parameters())
            dev = (n - base) / base * 100
        else:
            n, dev = 0, 0.0
        rows.append((spec["tag"], arm, n, dev))
    print(f"\n{'臂':<14}{'参数量':>10}{'vs ours-base':>14}")
    for tag, arm, n, dev in rows:
        print(f"{tag:<14}{n:>10,}{dev:>+13.1f}%")
    print(f"\n基准 ours-base = {base:,}（±10% 窗口: [{int(base*0.9):,}, {int(base*1.1):,}]）")
    return rows


if __name__ == "__main__":
    param_table()
    print("\n接线断言自检:")
    for arm in ARM_SPECS:
        ok, probs = assert_wiring(arm, make_model(arm).state_dict())
        print(f"  [{'PASS' if ok else 'FAIL'}] {arm}" + (f"  {probs}" if probs else ""))