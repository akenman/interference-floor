# -*- coding: utf-8 -*-
"""
E1b 驱动脚本 (run_e1b.py)
==================================================
按序执行 E1b 全部 9 次训练 + 统一评估。
  Step 1: B-uniform seed 42 → 确定卦级难度排序 → 计算 B-priority 权重
  Step 2: 剩余 8 次训练（uniform s43/s44, phase s42/s43/s44, priority s42/s43/s44）
  Step 3: eval_e1b.py + plot_e1b.py

所有训练使用完全相同的 windows_per_epoch=120000（文档化偏差：完整数据需 507,600，
此处取 23.6% 以保证三臂配置完全一致、相对比较有效；总步数仍足以收敛）。
用时估算：~50 s/epoch × 60 ≈ 50 min/次，9 次 ≈ 7.5 h。

用法: python run_e1b.py
"""

import os
import sys
import json
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

WINDOWS_PER_EPOCH = 120000
EPOCHS = 60
SEEDS = [42, 43, 44]
VARIANTS = ["uniform", "phase", "priority"]
CK_ROOT = "checkpoints-e1b"
GUA_NAMES = ["坤", "艮", "坎", "巽", "震", "离", "兑", "乾"]


def run(cmd, log):
    print(f"\n=== {cmd}")
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            fh.write(line)
            sys.stdout.flush()
    rc = p.wait()
    print(f"=== 退出码 {rc} ({log})")
    return rc == 0


def main():
    os.makedirs(CK_ROOT, exist_ok=True)

    # ---------- Step 1: B-uniform seed 42（定卦级难度）-----------------
    print("\n" + "=" * 72)
    print("Step 1: B-uniform seed 42 → 卦级难度排序")
    print("=" * 72)
    ok = run([sys.executable, "train_e1b.py", "--variant", "uniform", "--seed", "42",
              "--epochs", str(EPOCHS), "--windows_per_epoch", str(WINDOWS_PER_EPOCH),
              "--checkpoint_dir", f"{CK_ROOT}/uniform-s42"], "logs/e1b-uniform-s42.log")
    if not ok:
        sys.exit("Step 1 失败，中止")

    ok = run([sys.executable, "eval_e1b.py", "--pattern", f"{CK_ROOT}/uniform-s42/best.pt",
              "--output", "analysis_output/E1b_step1.json"], "logs/e1b-eval-step1.log")
    if not ok:
        sys.exit("Step 1 评估失败，中止")

    with open("analysis_output/E1b_step1.json", encoding="utf-8") as f:
        s1 = json.load(f)
    ge = s1["runs"][0]["gua_err"]
    order = sorted(range(8), key=lambda g: -ge[f"g{g} ({GUA_NAMES[g]})"])
    print("\n卦级难度排序（由难到易，基于 B-uniform seed 42）:")
    for g in order:
        print(f"  {GUA_NAMES[g]} (g{g}): {ge[f'g{g} ({GUA_NAMES[g]})']:.3f} px")
    hardest, easiest = order[0], order[-1]
    print(f"\n→ 最难 = {GUA_NAMES[hardest]} (g{hardest}), 最易 = {GUA_NAMES[easiest]} (g{easiest})")

    # B-priority 权重: w_g = max(1, err_g / mean)，归一化
    vals = [ge[f"g{g} ({GUA_NAMES[g]})"] for g in range(8)]
    mean = sum(vals) / len(vals)
    w = [max(1.0, v / mean) for v in vals]
    sw = sum(w)
    w = [x * len(w) / sw for x in w]          # 归一化使均值 = 1
    gw = ",".join(f"{x:.4f}" for x in w)
    print(f"B-priority 卦权重: {gw}")

    # ---------- Step 2: 剩余 8 次训练 -----------------
    jobs = []
    for v in VARIANTS:
        for s in SEEDS:
            if v == "uniform" and s == 42:
                continue
            jobs.append((v, s))

    for v, s in jobs:
        extra = []
        if v == "priority":
            extra = ["--gua_weights", gw]
        label = f"{v}-s{s}"
        print(f"\n{'=' * 72}")
        print(f"Step 2: {label}")
        print("=" * 72)
        ok = run([sys.executable, "train_e1b.py", "--variant", v, "--seed", str(s),
                  "--epochs", str(EPOCHS), "--windows_per_epoch", str(WINDOWS_PER_EPOCH),
                  "--checkpoint_dir", f"{CK_ROOT}/{label}"] + extra,
                 f"logs/e1b-{label}.log")
        if not ok:
            print(f"  !! {label} 失败，继续后续任务")

    # ---------- Step 3: 统一评估 + 作图 -----------------
    print("\n" + "=" * 72)
    print("Step 3: 统一评估")
    print("=" * 72)
    run([sys.executable, "eval_e1b.py", "--pattern", f"{CK_ROOT}/*/best.pt",
         "--output", "analysis_output/E1b_raw.json"], "logs/e1b-eval.log")
    print("\n作图...")
    run([sys.executable, "plot_e1b.py"], "logs/e1b-plot.log")
    print("\n[E1b 全部完成]")


if __name__ == "__main__":
    main()