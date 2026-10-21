# -*- coding: utf-8 -*-
"""S1f 阈值扫描汇总：四臂 × 3 seed 的两模式曲线（预注册标定依据）。"""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1] if len(sys.argv) > 1
                   else "analysis_output/S1f_threshold_final.json", encoding="utf-8"))
print(f"{'臂':9s} {'模式/τ':>10s} {'FF全关窗':>9s} {'FF RMS':>8s} {'FF中位':>8s} "
      f"{'自由单元保护':>12s} {'接触误关':>9s} {'接触RMS':>8s}")
for arm, seeds in d["arms"].items():
    keys = list(seeds["42"].keys())
    for key in keys:
        rows = [seeds[s][key] for s in seeds]
        f = rows[0].get("ff", {})
        full = np.mean([r["ff"]["frac_all_silenced"] for r in rows])
        rms = np.mean([r["ff"]["rms"] for r in rows])
        med = np.mean([r["ff"]["median"] for r in rows])
        pr = np.mean([r["cells"]["protect_rate_free"] for r in rows])
        fs = np.mean([r["cells"]["false_silence_contact"] for r in rows])
        cr = np.mean([r["contact"]["rms"] for r in rows])
        print(f"{arm:9s} {key:>10s} {full:9.1%} {rms:8.4f} {med:8.4f} "
              f"{pr:12.1%} {fs:9.1%} {cr:8.4f}")
    print()
