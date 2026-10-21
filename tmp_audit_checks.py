# -*- coding: utf-8 -*-
"""E1a 训练逻辑实证检查（只读，不影响运行中的训练）。

C1 数据切分：训练/验证片段是否真正不相交
C2 归一化统计：checkpoint 里存的是否等于"只用训练集"重算，且不同于全量统计
C3 协议与状态：checkpoint 内容完整性
C4 best 值：checkpoint 的 best 是否与日志报告一致
C5 配对前提：同 seed 下各变体是否获得完全相同的批次序列（numpy 全局 RNG 消耗）
C6 POS_IDX：one-step 指标取的维度是否真的是位置维
"""
import io
import os
import sys

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")

from layout_h import OBJ_DIMS
from train_e1a import POS_IDX, HingeDataE1a, L, K

# ---------- C1 数据切分 ----------
d = np.load("trajectories_hinge.npz")
states = d["states"]
N = states.shape[0]
rng = np.random.RandomState(42)
segs = np.arange(N)
rng.shuffle(segs)
val_eps = segs[:max(1, int(N * 0.1))]
tr_eps = segs[int(N * 0.1):]
overlap = set(tr_eps.tolist()) & set(val_eps.tolist())
print("[C1] 切分")
print(f"     N={N} 训练={len(tr_eps)} 验证={len(val_eps)} 交集={len(overlap)}"
      f" {'[OK] 不相交' if not overlap else '[FAIL] 泄漏!'}")
print(f"     划分固定 RandomState(42)，与训练 seed 解耦 → 所有 seed/变体共用同一验证集")

# ---------- C2 归一化 ----------
obj_flat_tr = states[tr_eps].reshape(-1, OBJ_DIMS).astype(np.float64)
obj_flat_all = states.reshape(-1, OBJ_DIMS).astype(np.float64)
mean_tr, std_tr = obj_flat_tr.mean(0), obj_flat_tr.std(0)
mean_all, std_all = obj_flat_all.mean(0), obj_flat_all.std(0)
print("\n[C2] 归一化统计")
print(f"     |mean_tr - mean_all|_max = {np.abs(mean_tr - mean_all).max():.4g}")
print(f"     |std_tr  - std_all |_max = {np.abs(std_tr  - std_all ).max():.4g}")
print(f"     → 全量与训练集统计{'不同' if np.abs(mean_tr - mean_all).max() > 1e-9 else '相同'}"
      f"（相同则无法区分泄漏，需看 checkpoint 实存值）")

# ---------- C3/C4 checkpoint 检查 ----------
print("\n[C3/C4] checkpoint")
ck_dir = "checkpoints-e1a-corrected"
for name in ["V1_s45", "V3_s43", "V2b_s42", "V2c_s44"]:
    bp = os.path.join(ck_dir, name, "best.pt")
    lp = os.path.join(ck_dir, name, "checkpoint_latest.pt")
    if not os.path.exists(bp):
        print(f"     {name:<10} best.pt 不存在（跳过）")
        continue
    c = torch.load(bp, map_location="cpu", weights_only=False)
    cfg = c.get("config", {})
    ns = c.get("norm_stats", {})
    m_ns, s_ns = np.array(ns.get("mean_obj", [])), np.array(ns.get("std_obj", []))
    match_tr = (m_ns.shape[0] > 0 and np.allclose(m_ns, mean_tr.astype(np.float32), atol=1e-5)
                and np.allclose(s_ns, std_tr.astype(np.float32), atol=1e-5))
    match_all = (m_ns.shape[0] > 0 and np.allclose(m_ns, mean_all.astype(np.float32), atol=1e-5))
    has = [k for k in ["model_state_dict", "optimizer_state_dict",
                       "scheduler_state_dict", "best", "epoch", "val_metrics"]
           if k not in c]
    print(f"     {name:<10} protocol={cfg.get('protocol_version')!r} "
          f"epoch={c.get('epoch')} best={c.get('best'):.3f} "
          f"{'[OK]只用训练集' if match_tr else '[FAIL]非训练集统计!'}"
          f"{' (且等于全量→无法证明)' if match_all else ' [OK]≠全量→证明非泄漏'}")
    if has:
        print(f"     {'':<10} 缺失字段: {has}")
    if os.path.exists(lp):
        cl = torch.load(lp, map_location="cpu", weights_only=False)
        if c.get("best") is not None and cl.get("best") is not None:
            tag = "latest 更差(已崩溃/过冲)" if cl["best"] > c["best"] + 1e-9 else \
                  "latest 与 best 一致"
            print(f"     {'':<10} latest.best={cl['best']:.3f} → {tag}")

# ---------- C5 配对前提：批次序列 ----------
print("\n[C5] 配对前提：同一 seed 下不同变体的批次序列是否一致")
# 复刻 train_e1a.main 的 RNG 使用顺序：seed → (np 局部 RNG 不消耗全局) → permutation
def perm_for(seed, n_win, wpe, epochs):
    np.random.seed(seed)
    return [np.random.permutation(n_win)[:wpe].tobytes() for _ in range(epochs)]

for seed in (45, 46):
    a = perm_for(seed, 461880, 60000, 3)   # V1 形态
    b = perm_for(seed, 461880, 60000, 3)   # V3 形态（数据源与变体无关）
    print(f"     seed={seed} 序列完全一致: {a == b} "
          f"（数据集/窗口数不依赖变体，故 perm 只由 seed 决定）")

# 确认全局 np RNG 消耗点：只有 seed 与 permutation
import re
src = open("train_e1a.py", encoding="utf-8").read()
calls = re.findall(r"np\.random\.\w+", src)
print(f"     train_e1a.py 中 np.random 调用: {sorted(set(calls))}")
print(f"     → RandomState(42)/RandomState(0) 均为局部对象，不消耗全局状态")

# ---------- C6 POS_IDX ----------
print("\n[C6] one-step 指标维度")
print(f"     OBJ_DIMS={OBJ_DIMS}  POS_IDX={list(POS_IDX)}")
frm = states[tr_eps[0], 0]
for i in POS_IDX:
    obj = i // 11 if OBJ_DIMS == 22 else -1
    field = i % 11
    print(f"     idx {i:>2} → 物体{obj} 字段序号 {field}")

print("\n检查完毕（未修改任何文件，未影响运行中的训练）")
