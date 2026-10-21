# -*- coding: utf-8 -*-
"""续训前置检查：断点可加载性、字段完整性、变体合法性。"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch

CKPT = "checkpoints-e1a-corrected/Vfree_s49/checkpoint_latest.pt"
BEST = "checkpoints-e1a-corrected/Vfree_s49/best.pt"

print("=== 断点检查 ===")
if not os.path.exists(CKPT):
    print(f"[FAIL] 缺 {CKPT}")
    sys.exit(1)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"  keys        : {sorted(ck.keys())}")
for k in ("epoch", "best"):
    if k in ck:
        print(f"  {k:<12}: {ck[k]}")
print(f"  config      : {ck.get('config')}")

opt = ck.get("optimizer_state_dict") or {}
sch = ck.get("scheduler_state_dict") or {}
st = opt.get("state", {}) if isinstance(opt, dict) else {}
n_par = len(st)
print(f"  optimizer_state_dict: 参数字典 {n_par} 项")
if n_par:
    e0 = next(iter(st.values())).get("step")
    print(f"    某参数 step: {e0}")
    print("    -> 优化器状态完整，续训不会退化为热启动 [OK]"
          if e0 is not None else "    -> step 缺失 [FAIL]")
else:
    print("    -> [警告] optimizer 状态为空！续训等价于热启动，结果不可比")
print(f"  scheduler_state_dict: {sch if not sch else list(sch)[:4]}")
if isinstance(sch, dict) and "last_epoch" in sch:
    print(f"    last_epoch={sch['last_epoch']}")

print("\n=== best.pt ===")
if os.path.exists(BEST):
    b = torch.load(BEST, map_location="cpu", weights_only=False)
    print(f"  可加载 [OK]  keys={sorted(b.keys())[:8]}")
    vm = b.get("val_metrics", {})
    print(f"  val_metrics.one_step_model = {vm.get('one_step_model')}")
else:
    print("  [缺失] best.pt 不存在")

print("\n=== 变体合法性 ===")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_e1a  # noqa: E402

for name, tbl in (("VARIANTS", getattr(train_e1a, "VARIANTS", {})),
                  ("VARIANTS_DECOUPLED", getattr(train_e1a, "VARIANTS_DECOUPLED", {}))):
    keys = list(tbl.keys()) if isinstance(tbl, dict) else list(tbl)
    print(f"  {name:<18} 含 Vfree: {'Vfree' in keys}")
print(f"  断点 config  : {ck.get('config')}")
print("\n前置检查完成")
