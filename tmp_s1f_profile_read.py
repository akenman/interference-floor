# -*- coding: utf-8 -*-
"""读 S1f 指示器剖面：逐步（k）单元级判别质量（p̄_free vs p̄_contact + 保护/误关）。"""
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "analysis_output/S1f_indicator_profile.json"
d = json.load(open(path, encoding="utf-8"))
for arm, seeds in d["arms"].items():
    print(f"\n=== {arm} ===")
    for k in ("k=1", "k=2", "k=3"):
        rows = [seeds[s]["win|0.5"]["per_k"][k] for s in seeds]
        pf = np.mean([r["prob_free"] for r in rows])
        pc = np.mean([r["prob_contact"] for r in rows])
        pr = np.mean([r["protect_rate_free"] for r in rows])
        fs = np.mean([r["false_silence_contact"] for r in rows])
        nf = int(np.mean([r["n_free"] for r in rows]))
        nc = int(np.mean([r["n_contact"] for r in rows]))
        print(f"  {k}: 自由单元 n={nf:6d} p̄={pf:.3f} 保护={pr:6.1%} | "
              f"接触单元 n={nc:6d} p̄={pc:.3f} 误关={fs:6.1%} | 分离度 p̄差={pc - pf:+.3f}")
