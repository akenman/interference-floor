# -*- coding: utf-8 -*-
"""
接线自检测试 (tests/test_wiring.py) — 工程红线
==================================================
gua 臂静默未接线事故（M2b）的制度化防御：从 checkpoint **权重**反推模型结构，
与 config 断言一致，并验证构造后可无损加载。任何新训练产出必须通过本测试。

检查项：
  1. use_gua flag 与 gua_emb 权重存在性一致
  2. state_head 输入维 = hidden + action + (gua?16:0)
  3. sigma_head 输入维 = state_head 输入维 + (innov?99:0)
  4. 由权重反推的 flags 修正 config 后，state_dict 可 strict 加载（零缺失/零多余）

用法：
  python tests/test_wiring.py [checkpoint ...]    # 缺省扫描 checkpoints-v3*/best.pt
"""

import sys
import glob
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model_v3 import WorldModelV3
from layout import DT_V3, IMPULSE_MAX_V3


def infer_from_weights(sd, cfg, legacy_no_sigma=False):
    """从权重反推结构 flags 与期望输入维度，返回 (修正后的 cfg, 问题列表)。"""
    problems = []
    has_gua = "gua_emb.weight" in sd
    if bool(cfg.get("use_gua", False)) != has_gua:
        problems.append(f"use_gua config={cfg.get('use_gua', False)} 但权重中 gua_emb "
                        f"{'存在' if has_gua else '缺失'}（静默接线事故特征）")
    cfg["use_gua"] = has_gua

    hidden = cfg.get("hidden_dim", 256)
    action = cfg.get("action_dim", 6)
    st_in = sd["state_head.0.weight"].shape[1]
    expected_st = hidden + action + (cfg.get("gua_emb_dim", 16) if has_gua else 0)
    if st_in != expected_st:
        problems.append(f"state_head 输入维 {st_in} ≠ 期望 {expected_st}")

    if not legacy_no_sigma:
        sh_in = sd["sigma_head.0.weight"].shape[1]
        innov_dim = sh_in - expected_st
        if innov_dim not in (0, 3 * cfg.get("obj_dim", 33)):
            problems.append(f"sigma_head 创新量维 {innov_dim} 非法（应为 0 或 {3 * cfg.get('obj_dim', 33)}）")
        cfg["use_innov"] = innov_dim > 0
        if innov_dim > 0:
            cfg["innov_dim"] = innov_dim
    return cfg, problems


def check_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    cfg = dict(ck["config"])
    cfg.setdefault("tau", DT_V3)
    cfg.setdefault("gravity", -900.0)
    cfg.setdefault("impulse_max", IMPULSE_MAX_V3)
    tag = ""
    # 遗留兼容：σ 头与归一化 buffer 引入之前训练的 checkpoint 缺这些键属预期
    legacy_no_sigma = "sigma_head.0.weight" not in sd
    if legacy_no_sigma:
        tag += "（遗留：无σ头）"
    cfg, problems = infer_from_weights(sd, cfg, legacy_no_sigma)
    model = WorldModelV3(**cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    legacy_prefixes = ("sigma_head.",)
    legacy_keys = {"mean_obj", "std_obj"}
    legacy_missing = [k for k in missing
                      if k.startswith(legacy_prefixes) or k in legacy_keys]
    missing = [k for k in missing if k not in legacy_missing]
    if legacy_missing and "遗留" not in tag:
        tag += f"（遗留：缺 {len(legacy_missing)} 个后增键）"
    if missing:
        problems.append(f"缺失键 {len(missing)}: {list(missing)[:3]}")
    if unexpected:
        problems.append(f"多余键 {len(unexpected)}: {list(unexpected)[:3]}")
    return problems, tag


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob(os.path.join("checkpoints-v3*", "best.pt")))
    if not paths:
        print("[SKIP] 未找到任何 checkpoint")
        return
    n_fail = 0
    for p in paths:
        try:
            problems, tag = check_checkpoint(p)
        except Exception as e:
            problems, tag = [f"加载失败: {e}"], ""
        verdict = "PASS" if not problems else "FAIL"
        n_fail += problems and 1 or 0
        print(f"[{verdict}] {p} {tag}")
        for q in problems:
            print(f"    - {q}")
    print(f"\n{'全部通过' if n_fail == 0 else f'{n_fail} 个 checkpoint 未通过接线自检'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
