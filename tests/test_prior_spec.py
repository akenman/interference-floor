# -*- coding: utf-8 -*-
"""先验颗粒度对齐守卫（tests/test_prior_spec.py）。

背景（2026-10-01 审计）：先验曾有**三处**独立实现，且训练期构造残差目标用的
先验与评估期 model.prior_norm 用的是两条代码路径 —— 一旦只改一处，训练目标与
评估先验会**静默错配**（损失降、指标怪，且没有任何断言会报警）。

本测试把以下四件事变成可执行约束：
  1. 三处实现的等价性（numpy spec ↔ torch spec ↔ layout_h 生成器）；
  2. rk4001x5 档位 == 数据生成器（acrobot_rk4 @0.01 ×5），即"精确先验"名副其实；
  3. **训练/评估先验同源**：E0Data._prior_np 与 model.prior_norm 在同一 spec 下相等；
  4. 先验阶梯的单调性（口径回归守卫，数字来自 tmp_prior_ladder_pilot.py）。

运行：python -m pytest tests/test_prior_spec.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from layout_h import (OBJ_DIMS, ACTION_DIM, PHYSICS_DT, DT, TAU_MAX,  # noqa: E402
                      obj_base, F as LF, acrobot_rk4, feat_weights,
                      L1, L2, M1, M2, G)
from physics_prior_spec import (SPECS, DEFAULT_SPEC, prior_raw_np,  # noqa: E402
                                prior_norm_torch, prior_step_torch_spec,
                                rebuild_np, integrate_np)
from e0_common import E0Data  # noqa: E402
import model_e0  # noqa: E402

DATA = os.path.join(ROOT, "trajectories_hinge.npz")
B1, B2 = obj_base(0), obj_base(1)
POS_IDX = [1, 2, 12, 13]
W = feat_weights(OBJ_DIMS)[POS_IDX]
N_WIN = 4000


@pytest.fixture(scope="module")
def data():
    d = np.load(DATA)
    return dict(states=d["states"], contact=d["contact"], actions=d["actions"],
                yao=d["yao"])


@pytest.fixture(scope="module")
def batch(data):
    """固定的一小批窗口（原始单位 + 归一化），供各断言共用。"""
    S, A = data["states"], data["actions"]
    n_ep, T = S.shape[0], S.shape[1]
    L, K = 8, 5
    n_win = T - L - K + 1
    ep = np.repeat(np.arange(6), n_win)
    w0 = np.tile(np.arange(n_win), 6)
    idx = np.arange(len(ep))[:N_WIN]
    ep, w0 = ep[idx], w0[idx]
    raw_last = S[ep, w0 + L - 1].astype(np.float64)
    a0 = A[ep, w0 + L].astype(np.float64).reshape(-1)
    tgt = S[ep, w0 + L][:, :OBJ_DIMS].astype(np.float64)
    return dict(ep=ep, w0=w0, raw_last=raw_last, a0=a0, tgt=tgt,
                true_next=S[ep, w0 + L][:, :OBJ_DIMS].astype(np.float64))


# ---------------------------------------------------------------- 1. 实现等价性
def test_numpy_spec_matches_generator(batch):
    """rk4001x5（精确档）必须逐位等于数据生成器 acrobot_rk4 @0.01 跑 5 次。"""
    raw, a = batch["raw_last"], batch["a0"]
    out = prior_raw_np("rk4001x5", raw, a)
    ref = []
    for i in range(len(raw)):
        t1 = np.arctan2(raw[i, B1 + LF["sin_t"]], raw[i, B1 + LF["cos_t"]])
        t2 = np.arctan2(raw[i, B2 + LF["sin_t"]], raw[i, B2 + LF["cos_t"]])
        w1, w2 = raw[i, B1 + LF["w"]], raw[i, B2 + LF["w"]]
        tau = a[i] * TAU_MAX
        for _ in range(5):
            t1, t2, w1, w2 = acrobot_rk4(t1, t2, w1, w2, tau, dt=PHYSICS_DT)
        ref.append(rebuild_np(raw[i:i + 1], np.array([t1]), np.array([t2]),
                              np.array([w1]), np.array([w2]))[0])
    assert np.abs(out - np.array(ref)).max() < 1e-8


def test_torch_and_numpy_specs_agree(batch):
    """torch 与 numpy 两条实现路径在所有档位上一致（float64 对拍）。"""
    raw, a = batch["raw_last"][:256], batch["a0"][:256]
    for spec in SPECS:
        np_out = prior_raw_np(spec, raw, a)
        t = torch.tensor(raw)
        an = torch.tensor(a).reshape(-1, 1)
        th1 = torch.atan2(t[:, B1 + LF["sin_t"]], t[:, B1 + LF["cos_t"]])
        th2 = torch.atan2(t[:, B2 + LF["sin_t"]], t[:, B2 + LF["cos_t"]])
        s1, c1, s2, c2, w1, w2 = prior_step_torch_spec(
            spec, th1.sin(), th1.cos(), th2.sin(), th2.cos(),
            t[:, B1 + LF["w"]], t[:, B2 + LF["w"]], an)
        th1n = torch.atan2(s1, c1).numpy()
        th2n = torch.atan2(s2, c2).numpy()
        t_out = rebuild_np(raw, th1n, th2n, w1.numpy(), w2.numpy())
        assert np.abs(t_out - np_out).max() < 1e-9, f"{spec} numpy/torch 不一致"


# ---------------------------------------------------------------- 2. 训练/评估同源
def test_rollout_detach_prior_is_value_identical(data):
    """P0-b 的成本控制开关：detach 先验输入只切断梯度，**数值必须逐位相同**。"""
    ds, mean, std = _make_small(data, "rk4001x5")
    b = ds.gather(np.arange(32))
    model = model_e0.make_model("ours_base", prior_spec="rk4001x5")
    model.set_stats(mean, std)
    model.eval()
    with torch.no_grad():
        model.rollout_detach_prior = False
        a1, _ = model.rollout(b["x"], b["af"], x_fut_gt=b["x_fut"], ss_prob=0.0)
        model.rollout_detach_prior = True
        a2, _ = model.rollout(b["x"], b["af"], x_fut_gt=b["x_fut"], ss_prob=0.0)
    for k in range(len(a1)):
        assert torch.equal(a1[k], a2[k]), f"detach 改变了 k={k} 的数值"
    assert model.rollout_detach_prior is True          # 类属性存在，可被 train_e0 设置


def _make_small(data, spec):
    S, C, A, Y = data["states"], data["contact"], data["actions"], data["yao"]
    eps = np.arange(6)
    flat = S[eps].reshape(-1, OBJ_DIMS).astype(np.float64)
    mean = flat.mean(0).astype(np.float32)
    std = flat.std(0).astype(np.float32)
    ds = E0Data(S, C, A, Y, eps, L=8, K=5, mean_obj=mean, std_obj=std,
                prior_spec=spec)
    return ds, mean, std


def test_train_eval_prior_same_source(data):
    """核心不变量：残差目标隐含的先验 == 模型 prior_norm 的先验（同 spec）。"""
    for spec in SPECS:
        ds, mean, std = _make_small(data, spec)
        b = ds.gather(np.arange(64))
        model = model_e0.make_model("ours_base", prior_spec=spec)
        model.set_stats(mean, std)
        prior_model = model.prior_norm(b["x"][:, -1, :OBJ_DIMS], b["a0"]).numpy()
        # d_norm = x_next(L) − prior_np  ⇒  prior_np = x_next(L) − d_norm
        prior_from_ds = (b["x_next"][:, -1, :OBJ_DIMS] - b["d_norm"]).numpy()
        assert np.abs(prior_model - prior_from_ds).max() < 1e-5, \
            f"{spec}: 训练期残差目标与评估期先验不同源"


def test_default_spec_is_legacy_euler005():
    """不传 spec 时必须等于旧行为（euler@0.05×1），保证历史数字可复现。"""
    assert DEFAULT_SPEC == "euler005"
    for arm in ("ours_base", "ours_yao", "ours_sigma", "ours_gua", "inertia"):
        m = model_e0.make_model(arm)
        assert getattr(m, "prior_spec", DEFAULT_SPEC) == "euler005", arm


# ---------------------------------------------------------------- 3. 阶梯单调性
def _one_step_rmse(spec, batch):
    pred = prior_raw_np(spec, batch["raw_last"], batch["a0"])
    e = (pred[:, POS_IDX] - batch["tgt"][:, POS_IDX]) ** 2
    return float(np.sqrt((e * W).mean()))


def test_ladder_ordering_and_magnitudes(batch):
    """阶梯必须单调，且精确档的单步误差比现行档小一个数量级以上。
    数字锚定 tmp_prior_ladder_pilot.py 的 30k 窗口实测（3.630 / 0.532 / 0.167 / 0.038）。"""
    vals = {s: _one_step_rmse(s, batch) for s in SPECS}
    assert vals["rk4001x5"] < vals["rk4005"] < vals["euler001x5"] < vals["euler005"], vals
    assert vals["rk4001x5"] < 0.10, vals          # "精确先验"名副其实
    assert vals["euler005"] > 3.0, vals           # E0 现行档确实是粗积分器
    assert vals["euler005"] / vals["rk4001x5"] > 20, vals
