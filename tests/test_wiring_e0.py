# -*- coding: utf-8 -*-
"""
E0 接线自检 (tests/test_wiring_e0.py)
==================================================
E0 主表 Step 2：对每个臂的 state_dict 做"由权重反推结构"断言，
并验证构造后可 strict 加载。捕获 gua 臂静默未接线类事故（M2b 教训）。

用法：python tests/test_wiring_e0.py [--ckpt checkpoints-E0/xxx.pt ...]
      不给路径则对所有 9 臂构造随机权重的模型做断言。
"""

import sys, os, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import model_e0


def check_arm(arm):
    model = model_e0.make_model(arm)
    sd = model.state_dict()
    ok, problems = model_e0.assert_wiring(arm, sd)
    # strict 加载自检（无缺失/无多余）
    fresh = model_e0.make_model(arm)
    missing, unexpected = fresh.load_state_dict(sd, strict=True)
    if missing:
        problems.append(f"strict 加载缺失 {len(missing)}: {list(missing)[:3]}")
    if unexpected:
        problems.append(f"strict 加载多余 {len(unexpected)}: {list(unexpected)[:3]}")
    return ok and not missing and not unexpected, problems


def check_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    arm = ck["config"].get("arm")
    if not arm:
        return False, [f"checkpoint 未记录 arm 字段 (config keys={list(ck.get('config',{}).keys())})"]
    if arm not in model_e0.ARM_SPECS:
        return False, [f"未知 arm={arm}"]
    ok, problems = model_e0.assert_wiring(arm, sd)
    fresh = model_e0.make_model(arm)
    missing, unexpected = fresh.load_state_dict(sd, strict=True)
    if missing:
        problems.append(f"strict 加载缺失 {len(missing)}: {list(missing)[:3]}")
    if unexpected:
        problems.append(f"strict 加载多余 {len(unexpected)}: {list(unexpected)[:3]}")
    # 归一化统计量随盘
    for key in ("mean_obj", "std_obj"):
        if key not in ck.get("norm_stats", {}):
            problems.append(f"norm_stats 缺失 {key}")
    return ok and not missing and not unexpected, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[])
    args = ap.parse_args()

    paths = args.ckpt or []
    n_fail = 0

    if paths:
        for p in paths:
            try:
                ok, problems = check_checkpoint(p)
            except Exception as e:
                ok, problems = False, [f"加载失败: {e}"]
            print(f"[{'PASS' if ok else 'FAIL'}] {p}")
            for q in problems:
                print(f"    - {q}")
            n_fail += not ok
    else:
        print("对所有 9 臂做构造期接线断言（无训练权重）:")
        for arm in model_e0.ARM_SPECS:
            try:
                ok, problems = check_arm(arm)
            except Exception as e:
                ok, problems = False, [str(e)]
            tag = model_e0.ARM_SPECS[arm]["tag"]
            print(f"[{'PASS' if ok else 'FAIL'}] {tag} ({arm})" + (f"  {problems}" if problems else ""))
            n_fail += not ok

    print(f"\n{'全部通过' if n_fail == 0 else f'{n_fail} 个臂/checkpoint 未通过接线自检'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()