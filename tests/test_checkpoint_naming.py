# -*- coding: utf-8 -*-
"""检查点路径与覆盖保护守卫（tests/test_checkpoint_naming.py）。

事故（2026-10-01）：`train_e0.py` 内联 `args.checkpoint_dir = f"checkpoints-E0/{arm}_s{seed}"`
**无条件覆盖** `run_e0.py` 传来的 `--checkpoint_dir`，导致 P0-a 的 3 个先验档位 × 3 seed
全部写进同一批目录：互相覆盖，并覆盖掉 E0 原始 `ours_base@euler005` 检查点。

本测试把三件事变成可执行约束：
  1. `train_e0.resolve_checkpoint_dir` 与 `run_e0.task_name` **命名约定一致**（跨模块不变量）；
  2. 显式 `--checkpoint_dir` 必须被尊重（事故根因）；
  3. `assert_dir_not_foreign` 必须在「目录里已有别的 arm/seed/prior_spec」时拒绝训练。
"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from physics_prior_spec import DEFAULT_SPEC, SPECS  # noqa: E402
import run_e0  # noqa: E402
from train_e0 import resolve_checkpoint_dir, assert_dir_not_foreign  # noqa: E402


def test_default_spec_dir_matches_legacy_name():
    assert resolve_checkpoint_dir("ours_base", 42) == "checkpoints-E0/ours_base_s42"
    assert resolve_checkpoint_dir("ours_base", 42, DEFAULT_SPEC) \
        == "checkpoints-E0/ours_base_s42"


def test_non_default_spec_dir_encodes_spec():
    for spec in SPECS:
        if spec == DEFAULT_SPEC:
            continue
        got = resolve_checkpoint_dir("ours_base", 43, spec)
        assert got == f"checkpoints-E0/ours_base@{spec}_s43"


def test_explicit_dir_is_honored():
    """事故根因：显式路径曾被无条件覆盖。"""
    explicit = "checkpoints-E0/EXP-先验阶梯/rk4001x5_s42"
    assert resolve_checkpoint_dir("ours_base", 42, "rk4001x5", explicit) == explicit


def test_naming_agrees_with_runner():
    """train_e0 与 run_e0 必须给出同一个目录（事故就是两者不一致）。"""
    for spec in SPECS:
        for seed in (42, 43, 44):
            assert resolve_checkpoint_dir("ours_base", seed, spec) \
                == f"checkpoints-E0/{run_e0.task_name('ours_base', seed, spec)}"


def _write_ckpt(d, arm, seed, prior_spec):
    os.makedirs(d, exist_ok=True)
    torch.save(dict(model_state_dict={}, epoch=1,
                    args=dict(arm=arm, seed=seed, prior_spec=prior_spec)),
               os.path.join(d, "best.pt"))


def test_overwrite_guard_blocks_foreign_spec(tmp_path):
    d = str(tmp_path / "ours_base_s42")
    _write_ckpt(d, "ours_base", 42, "euler005")
    with pytest.raises(SystemExit):
        assert_dir_not_foreign(d, "ours_base", 42, "euler001x5")   # 事故场景
    with pytest.raises(SystemExit):
        assert_dir_not_foreign(d, "ours_base", 43, "euler005")     # seed 不符
    with pytest.raises(SystemExit):
        assert_dir_not_foreign(d, "strong_mlp", 42, "euler005")    # arm 不符


def test_overwrite_guard_allows_same_job_and_missing(tmp_path):
    d = str(tmp_path / "ours_base_s42")
    assert_dir_not_foreign(d, "ours_base", 42, "euler005")         # 目录不存在 → 放行
    _write_ckpt(d, "ours_base", 42, "euler005")
    assert_dir_not_foreign(d, "ours_base", 42, "euler005")         # 同 job（续训/重跑）→ 放行


def test_overwrite_guard_tolerates_legacy_checkpoint_without_args(tmp_path):
    """E0 早期 checkpoint 的 args 里没有 prior_spec，不应因此拒训。"""
    d = str(tmp_path / "legacy")
    os.makedirs(d, exist_ok=True)
    torch.save(dict(model_state_dict={}, epoch=1), os.path.join(d, "best.pt"))
    assert_dir_not_foreign(d, "ours_base", 42, "rk4001x5")


# ---------------------------------------------------------------- 配方与根目录（P0-b）
def _write_ckpt_recipe(d, arm, seed, prior_spec, w_roll, lr):
    os.makedirs(d, exist_ok=True)
    torch.save(dict(model_state_dict={}, epoch=1,
                    args=dict(arm=arm, seed=seed, prior_spec=prior_spec,
                              w_roll=w_roll, lr=lr)),
               os.path.join(d, "best.pt"))


def test_cmd_encodes_checkpoint_root_and_recipe():
    """P0-b：不同训练配方必须落进不同根目录，且配方值要传给 train_e0。"""
    c = run_e0.cmd("ours_base", 42, "rk4001x5", "checkpoints-P0b", 0.05, 5e-4)
    joined = " ".join(c)
    assert "checkpoints-P0b/ours_base@rk4001x5_s42" in joined
    assert "--w_roll 0.05" in joined and "--lr 0.0005" in joined
    # 默认（单步配方 + E0 根）保持历史行为
    d = " ".join(run_e0.cmd("ours_base", 42, "euler005"))
    assert "checkpoints-E0/ours_base_s42" in d
    assert "--w_roll 0.0" in d and "--lr 0.001" in d


def test_overwrite_guard_blocks_different_recipe(tmp_path):
    """换训练配方（w_roll/lr）写进同一目录 → 必须拒训（P0-b 与单步配方不得混放）。"""
    d = str(tmp_path / "ours_base_s42")
    _write_ckpt_recipe(d, "ours_base", 42, "euler005", 0.0, 1e-3)
    assert_dir_not_foreign(d, "ours_base", 42, "euler005", 0.0, 1e-3)      # 同配方放行
    with pytest.raises(SystemExit):
        assert_dir_not_foreign(d, "ours_base", 42, "euler005", 0.05, 5e-4)  # 换配方 → 拦
    with pytest.raises(SystemExit):
        assert_dir_not_foreign(d, "ours_base", 42, "euler005", 0.0, 5e-4)   # 只换 lr → 拦


# ---------------------------------------------------------------- 日志名隔离（T1，P0-c）
def test_log_name_isolates_p0c_cells():
    """T1 修复点：P0-c 四格（C00/C01/C10/C11）只差 --checkpoint_root，
    w_roll/lr/wpe/epochs 全同。日志名若不含 root 末级名，四格会写进
    同一条日志互相覆盖、事后无法归因（run_e0 此前只带 wr/lr/wpe/ep）。"""
    kw = dict(w_roll=0.0, lr=1e-3, windows_per_epoch=150000,
              detach_prior=False, epochs=60)
    n00 = run_e0.log_name("ours_base", 42, "euler005",
                          checkpoint_root="checkpoints-P0c/C00", **kw)
    n11 = run_e0.log_name("ours_base", 42, "euler005",
                          checkpoint_root="checkpoints-P0c/C11", **kw)
    n10 = run_e0.log_name("ours_base", 42, "euler005",
                          checkpoint_root="checkpoints-P0c/C10", **kw)
    assert len({n00, n10, n11}) == 3
    assert "C00" in n00 and "C10" in n10 and "C11" in n11
    assert n00 == "ours_base_s42~C00_wr0_lr0.001_wpe150000_ep60"
    # 检查点目录同样按 root 隔离（铁律 4：配方隔离靠"每格一个根目录"）
    c00 = run_e0.cmd("ours_base", 42, "euler005", "checkpoints-P0c/C00", 0.0, 1e-3,
                     windows_per_epoch=150000, epochs=60,
                     extra_train_args=["--w_yao", "0", "--w_proto", "0",
                                       "--w_commit", "0", "--w_usage", "0"])
    c11 = run_e0.cmd("ours_base", 42, "euler005", "checkpoints-P0c/C11", 0.0, 1e-3,
                     windows_per_epoch=150000, epochs=60)
    d00 = c00[c00.index("--checkpoint_dir") + 1]
    d11 = c11[c11.index("--checkpoint_dir") + 1]
    assert d00 != d11
    assert d00.startswith("checkpoints-P0c/C00/")
    assert d11.startswith("checkpoints-P0c/C11/")
    # --extra_train_args 必须原样透传给 train_e0（P0-c 的格子定义）
    for flag in ("--w_yao", "--w_proto", "--w_commit", "--w_usage"):
        assert c00[c00.index(flag) + 1] == "0"
    assert "--w_yao" not in c11          # C11 全默认：无附加参数


def test_log_name_default_recipe_unchanged():
    """默认 E0 配方不得改历史日志名（追加模式靠同名续写既有日志）。"""
    assert run_e0.log_name("ours_base", 42) == "ours_base_s42"
    assert run_e0.log_name("ours_base", 42, "rk4001x5") == "ours_base@rk4001x5_s42"


def test_normalize_extra_args_both_forms():
    """实测（2026-10-02）：分 token 形式被未知旗标截断报错，整串形式收成单个
    含空格元素。两种形态必须规整成同一 token 列表（P0-c 四格命令写法不敏感）。"""
    token_form = ["--w_yao", "0", "--w_proto", "0"]
    string_form = ["--w_yao 0 --w_proto 0 --w_commit 0 --w_usage 0"]
    assert run_e0.normalize_extra_args(token_form) == token_form
    assert run_e0.normalize_extra_args(string_form) == \
        ["--w_yao", "0", "--w_proto", "0", "--w_commit", "0", "--w_usage", "0"]
    assert run_e0.normalize_extra_args([]) == []


def test_log_name_recipe_changes_still_encoded():
    """root 之外的配方变化（w_roll/lr/wpe/detach/epochs）仍要进日志名（P0-b 行为回归）。
    T1 起后缀同时含 root 末级名，故格式为 ~<root>_wr.._lr.._wpe..[_det]_ep..。"""
    n = run_e0.log_name("ours_base", 42, "euler005", checkpoint_root="checkpoints-P0bB",
                        w_roll=0.05, lr=5e-4, windows_per_epoch=150000,
                        detach_prior=True, epochs=60)
    assert n == ("ours_base_s42~checkpoints-P0bB_wr0.05_lr0.0005"
                 "_wpe150000_det_ep60")
