# -*- coding: utf-8 -*-
"""Stage A 进度统计（与 run_e1a.py 的 --skip-done 判定一致：看尾部 [OK]）。"""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

VARIANTS = ["V1", "V2a", "V2b", "V3", "Vfree"]
SEEDS = list(range(45, 52))
LOGS = "logs-corrected"

RE_OK = re.compile(r"\[OK\]")
RE_EARLY = re.compile(r"\[EARLY-STOP\][^\n]*")
RE_VAL = re.compile(r"\[val\]\s*one-step[^0-9]*([0-9]+\.[0-9]+)")


def tail(p, n=6144):
    with open(p, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - n))
        raw = fh.read()
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "ignore")


done, running, pending = [], [], []
for s in SEEDS:
    for v in VARIANTS:
        name = f"{v}_s{s}"
        p = os.path.join(LOGS, name + ".log")
        if not os.path.exists(p):
            pending.append(name)
            continue
        t = tail(p)
        if RE_OK.search(t):
            vals = [float(x) for x in RE_VAL.findall(t)]
            ep = RE_EARLY.search(t)
            best = min(vals) if vals else float("nan")
            tag = ""
            if ep:
                tag = " [早停]" if "collapse=1" not in ep.group(0) else " [崩溃]"
            done.append((name, best, tag))
        else:
            running.append(name)

print(f"Stage A 进度: 完成 {len(done)} / 35")
print(f"  进行中: {running if running else '—'}")
print(f"  待跑  : {len(pending)} 个")
print()
print(f"{'run':<12}{'best px':>10}{'标记':>8}")
print("-" * 30)
for n, b, t in sorted(done):
    print(f"{n:<12}{b:>10.3f}{t:>8}")

# GPU 侧
print()
import subprocess
r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,"
                    "utilization.gpu,temperature.gpu", "--format=csv,noheader"],
                   capture_output=True, text=True)
print("GPU:", (r.stdout or "").strip())
