# -*- coding: utf-8 -*-
"""针对崩溃 run：验证 best.pt 是否真的是崩溃前的权重（评测可用性）。"""
import io
import os
import re
import sys

import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LOGS = "logs-corrected"
CK = "checkpoints-e1a-corrected"
RE_COL = re.compile(r"\[EARLY-STOP\]\s+collapse=1\s+epoch=(\d+)\s+val=([0-9.]+)px\s+best=([0-9.]+)px")
RE_VAL = re.compile(r"\[val\]\s*one-step[^0-9]*([0-9]+\.[0-9]+)")

collapsed, allruns = [], []
for f in sorted(os.listdir(LOGS)):
    if not f.startswith("V") or not f.endswith(".log") or "_err" in f:
        continue
    p = os.path.join(LOGS, f)
    raw = open(p, "rb").read()
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    m = RE_COL.search(text)
    vals = [float(x) for x in RE_VAL.findall(text)]
    rec = dict(name=f[:-4], vals=vals, best=min(vals) if vals else None,
               collapse_epoch=int(m.group(1)) if m else None,
               collapse_val=float(m.group(2)) if m else None)
    allruns.append(rec)
    if m:
        collapsed.append(rec)

print(f"日志总数 {len(allruns)}，崩溃(collapse=1) {len(collapsed)}\n")
print(f"{'run':<16}{'崩溃epoch':>10}{'崩溃时val':>10}{'best':>8}"
      f"{'best在崩溃前?':>16}{'best.pt权重val':>16}{'可用?':>8}")
print("-" * 84)
for r in collapsed:
    name = r["name"]
    bp = os.path.join(CK, name, "best.pt")
    if not os.path.exists(bp):
        print(f"{name:<16}{r['collapse_epoch']:>10}{r['collapse_val']:>10.1f}"
              f"{r['best']:>8.3f}  best.pt 缺失")
        continue
    c = torch.load(bp, map_location="cpu", weights_only=False)
    w_val = c.get("val_metrics", {}).get("one_step_model")
    w_best = c.get("best")
    # best 是否取自崩溃前的验证值
    pre = r["best"] < r["collapse_val"]
    usable = (w_val is not None and w_val <= 5.0)
    print(f"{name:<16}{r['collapse_epoch']:>10}{r['collapse_val']:>10.1f}"
          f"{r['best']:>8.3f}{str(pre):>16}{w_val if w_val else float('nan'):>16.3f}"
          f"{('[OK]' if usable else '[FAIL]'):>8}")

print("\n关键：best.pt 中 val_metrics.one_step_model 应等于 r['best']"
      "（即崩溃前的最优值）")
print("结论：崩溃 run 的 best.pt 若 val ≤5px 则该 run 仍可用于稳定均值；"
      "若 latest 已发散则不可用 latest 评测。")

print("\n对照：全部 run 的 best.pt 当前 val（发散的即 latest 已不可用）:")
bad = []
for r in allruns:
    bp = os.path.join(CK, r["name"], "best.pt")
    lp = os.path.join(CK, r["name"], "checkpoint_latest.pt")
    if not (os.path.exists(bp) and os.path.exists(lp)):
        continue
    cb = torch.load(bp, map_location="cpu", weights_only=False)
    cl = torch.load(lp, map_location="cpu", weights_only=False)
    vb = cb.get("val_metrics", {}).get("one_step_model")
    vl = cl.get("val_metrics", {}).get("one_step_model")
    if vb is None or vl is None:
        continue
    if vl - vb > 1.0:
        bad.append((r["name"], vb, vl))
if bad:
    print("  latest 相对 best 已劣化 >1px 的 run（必须用 best.pt 评测）:")
    for n, vb, vl in bad:
        print(f"    {n:<16} best.pt={vb:.3f}  latest={vl:.3f}")
else:
    print("  无（所有 latest 与 best 一致，未出现发散存档）")
