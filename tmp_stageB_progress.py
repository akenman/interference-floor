# -*- coding: utf-8 -*-
"""Stage B（no_gua 臂 × seeds 45-51）进度监控。

刻意复用 build_e1a_report.parse_log 作为 done / collapse / best 的唯一判定口径，
避免监控脚本与最终报告对「是否完成」「是否崩溃」产生分歧。
"""
import glob
import io
import json
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_e1a_report as B

LOGS = "logs-corrected"
ARMS = ["V1", "V2a", "V2b", "V2c", "V3", "V4"]
SEEDS = list(range(45, 52))
PLANNED = [(a, s) for a in ARMS for s in SEEDS]
IDLE_MIN = 60      # 相邻启动间隔超过此值视为暂停空窗，剔除出 per-run 统计

rows = {}
for p in glob.glob(os.path.join(LOGS, "*_nogua.log")):
    if "_err" in p:
        continue
    r = B.parse_log(p)
    if r["no_gua"] and r["variant"] in ARMS and r["seed"] in SEEDS:
        rows[(r["variant"], r["seed"])] = r

done = [k for k, r in rows.items() if r["done"]]
print("=== Stage B 进度 %s ===" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("完成 %d / %d" % (len(done), len(PLANNED)))

# 逐臂
print("\n%-7s %s" % ("arm", "  ".join("s%-2d" % s for s in SEEDS)))
for a in ARMS:
    cells = []
    for s in SEEDS:
        r = rows.get((a, s))
        if r is None:
            cells.append(" -- ")
        elif r["done"]:
            mark = "C" if r["collapse"] else ("E" if r.get("early_stop") else "ok")
            cells.append(" %s " % mark)
        else:
            cells.append(" .. ")
    print("%-7s %s" % (a, "".join(cells)))
print("\n图例: ok=正常  C=崩溃(best>=5)  E=早停  ..=进行中  --=未启动")

# 当前 run
print("\n=== 当前 run ===")
cur = None
for (a, s), r in rows.items():
    if not r["done"]:
        cur = (a, s, r)
if cur:
    a, s, r = cur
    log = os.path.join(LOGS, "%s_s%d_nogua.log" % (a, s))
    ep = None
    try:
        with open(log, encoding="utf-8", errors="ignore") as fh:
            m = re.findall(r"Epoch (\d+)/60", fh.read())
            if m:
                ep = int(m[-1])
    except OSError:
        pass
    print("  %s_s%d  epoch %s/60  best(中途)=%.3f" % (a, s, ep, r["best"]))
else:
    print("  无进行中任务")
# ETA —— 数据源：各 run 日志的 mtime（完成时刻）
# 不用 scheduler_stageB.log：它只记 HH:MM 无日期，且 --skip-done 恢复时被
# 截断/追加混写（本次实测只剩 7 条、日期推断失效、中位数算出 0.0）。
# 各 run 的 *.log 独立、mtime 自带完整日期，跨午夜与多次恢复都无歧义。
if done:
    remain = len(PLANNED) - len(done)
    try:
        stamps = []
        for a, s in PLANNED:
            p = os.path.join(LOGS, "%s_s%d%s.log" % (a, s, "_nogua"))
            if not os.path.exists(p):
                continue
            if not B.parse_log(p).get("done"):
                continue
            stamps.append((datetime.fromtimestamp(os.path.getmtime(p)), a, s))
        stamps.sort()
        if remain > 0 and len(stamps) > 1:
            # 只统计**有实际训练**的间隔：相邻 run 完成时刻之差。
            # > IDLE_MIN 视为暂停/挂起空窗并剔除，避免把停机时间算进 per-run。
            gaps = [(stamps[i + 1][0] - stamps[i][0]).total_seconds() / 60.0
                    for i in range(len(stamps) - 1)]
            active = [g for g in gaps if g <= IDLE_MIN]
            idle = [g for g in gaps if g > IDLE_MIN]
            per = float(np.median(active)) if active else 0.0
            eta = datetime.now() + timedelta(seconds=per * 60 * remain)
            print("\n=== ETA ===")
            print("  依据 %d 个已完成 run 的日志 mtime" % len(stamps))
            print("  活跃间隔 %d 个，中位数 %.1f min/run（剔除 %d 个 >%dmin 空窗）"
                  % (len(active), per, len(idle), IDLE_MIN))
            for g in sorted(idle, reverse=True)[:3]:
                print("     剔除空窗 %.1f min" % g)
            print("  剩 %d run" % remain)
            print("  预计完成 %s" % eta.strftime("%Y-%m-%d %H:%M"))
    except OSError:
        pass

# 错误
bad = []
for p in glob.glob(os.path.join(LOGS, "*_nogua.log")):
    try:
        with open(p, encoding="utf-8", errors="ignore") as fh:
            t = fh.read()
        for pat in ("Traceback", "out of memory", "RuntimeError"):
            if pat in t:
                bad.append((os.path.basename(p), pat))
                break
    except OSError:
        pass
print("\n=== 错误 ===")
if bad:
    for n, pat in bad:
        print("  %s : %s" % (n, pat))
else:
    print("  无 [OK]")
