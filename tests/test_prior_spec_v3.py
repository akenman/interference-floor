# -*- coding: utf-8 -*-
"""SLOT-1b 先验注册表守卫（tests/test_prior_spec_v3.py）。

把三件事变成可执行约束：
  1. 默认档 parabola005 与 legacy layout.physics_prior **逐位一致**（历史行为不变）；
  2. 解析锚：euler001x5 在常加速度合成场景满足引擎关系（显式欧拉，位置增益
     v·τ + 10·a·dt²），euler005/rk4 与解析闭式一致——SLOT-1 的判决数字可复算；
  3. numpy 与 torch 实现一致（训练期残差目标与评估期 model.prior_norm 同源）；
  4. v3 覆盖保护拦截异 prior_spec/异 seed（E0 覆盖事故教训在 v3 的最小落地）。
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from layout import physics_prior, DT_V3  # noqa: E402
import physics_prior_spec_v3 as PV  # noqa: E402


def _synthetic(M=64, seed=0):
    """合成 v3 状态：3 物体随机位置/速度/角度 + 常量属性；零冲量。"""
    rng = np.random.RandomState(seed)
    x = np.zeros((M, 33), dtype=np.float32)
    for o in range(3):
        b = o * 11
        x[:, b + 0] = 1.0
        x[:, b + 1] = rng.uniform(-200, 200, M)
        x[:, b + 2] = rng.uniform(-200, 200, M)
        th = rng.uniform(-np.pi, np.pi, M)
        x[:, b + 3] = np.sin(th)
        x[:, b + 4] = np.cos(th)
        x[:, b + 5] = rng.uniform(-100, 100, M)
        x[:, b + 6] = rng.uniform(-100, 100, M)
        x[:, b + 7] = rng.uniform(-1, 1, M)
        x[:, b + 8] = 1.0
        x[:, b + 9] = 20.0
        x[:, b + 10] = rng.choice([1.0, 2.0], M)
    return x, np.zeros((M, 6), dtype=np.float32)


def test_default_spec_bitmatches_legacy():
    x, i = _synthetic()
    a = PV.prior_raw_np("parabola005", x, i)
    b = physics_prior(x, i, tau=DT_V3, g=-900.0)
    assert np.array_equal(a, b)          # 逐位一致（委托实现 + 相同 dtype 路径）


def test_engine_class_analytic_anchor():
    """显式欧拉引擎关系（SLOT-1 实测）：零冲量下
    dy = vy·τ + 10·g·dt²（g=-900 ⇒ -0.9px），dvx = 0，dvy = g·τ。"""
    x, i = _synthetic()
    out = PV.prior_raw_np("euler001x5", x, i)
    tau, dt, g = 0.05, 0.01, -900.0
    for o in range(3):
        b = o * 11
        dy = out[:, b + 2] - x[:, b + 2]
        expect = x[:, b + 6] * tau + 10 * g * dt * dt
        assert np.allclose(dy, expect, atol=1e-3)
        assert np.allclose(out[:, b + 5] - x[:, b + 5], 0.0, atol=1e-4)
        assert np.allclose(out[:, b + 6] - x[:, b + 6], g * tau, atol=1e-3)
        # euler005（显式 @0.05×1）：位置无加速度项 → dy = vy·τ
        out5 = PV.prior_raw_np("euler005", x, i)
        assert np.allclose(out5[:, b + 2] - x[:, b + 2], x[:, b + 6] * tau, atol=1e-3)
        # rk4 = 连续精确：dy = vy·τ + ½gτ²
        out4 = PV.prior_raw_np("rk4005", x, i)
        assert np.allclose(out4[:, b + 2] - x[:, b + 2],
                           x[:, b + 6] * tau + 0.5 * g * tau * tau, atol=1e-3)


def test_numpy_torch_agree():
    for spec in PV.SPECS:
        x, i = _synthetic(32, seed=1)
        np_out = PV.prior_raw_np(spec, x, i)
        t_out = PV.prior_raw_torch(spec, torch.from_numpy(x), torch.from_numpy(i))
        assert np.abs(np_out - t_out.numpy()).max() < 1e-4, spec


def test_overwrite_guard_v3(tmp_path):
    d = str(tmp_path / "euler001x5_s42")
    os.makedirs(d, exist_ok=True)
    torch.save(dict(model_state_dict={}, epoch=1,
                    args=dict(prior_spec="euler001x5", seed=42)),
               os.path.join(d, "best.pt"))
    PV.assert_dir_not_foreign_v3(d, "euler001x5", 42)          # 同配方放行
    import pytest
    with pytest.raises(SystemExit):
        PV.assert_dir_not_foreign_v3(d, "parabola005", 42)      # 换档 → 拦
    with pytest.raises(SystemExit):
        PV.assert_dir_not_foreign_v3(d, "euler001x5", 43)       # 换 seed → 拦
