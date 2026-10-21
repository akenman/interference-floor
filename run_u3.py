# -*- coding: utf-8 -*-
"""
U3 运行器 (run_u3.py) — T8/M-D v2 target-bug 复核：两臂顺序执行
================================================================
臂1 legacy：eval_planning --legacy_target（复现 M-D v2 原始条件）
臂2 fixed ：eval_planning（env.target = select_scene 靶点）
报告：analysis_output/md_planning_{legacy,fixed}_report.json
"""

import subprocess
import sys

PY = sys.executable
BASE = ["eval_planning.py", "--episodes", "100",
        "--model_ckpt", "checkpoints-v3-calib2/best.pt"]

ARMS = [
    ("legacy", BASE + ["--legacy_target",
                       "--output_report", "analysis_output/md_planning_legacy_report.json"]),
    ("fixed", BASE + ["--output_report", "analysis_output/md_planning_fixed_report.json"]),
]

for i, (name, cmd) in enumerate(ARMS, 1):
    log = open(f"logs/u3_{name}.log", "w", encoding="utf-8", buffering=1)
    print(f"[U3] 臂{i}/2 启动: {name}", flush=True)
    rc = subprocess.call([PY, "-u"] + cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"[U3] 臂{i}/2 完成 rc={rc}", flush=True)
print("[U3] DONE", flush=True)
