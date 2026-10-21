# -*- coding: utf-8 -*-
"""对拍：同一 (臂, seed, k) 下 win 与 obj 两种模式的概率/保护率是否自洽。"""
import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1
                   else "analysis_output/S1f_indicator_profile.json", encoding="utf-8"))
for arm, seeds in d["arms"].items():
    s = "42"
    w = seeds[s]["win|0.5"]
    o = seeds[s]["obj|0.5"]
    print(f"\n[{arm} s{s}]")
    for k in ("k=1", "k=2", "k=3"):
        a, b = w["per_k"][k], o["per_k"][k]
        print(f"  {k}: win p̄_free={a['prob_free']:.4f} 保护={a['protect_rate_free']:.3f} | "
              f"obj p̄_free={b['prob_free']:.4f} 保护={b['protect_rate_free']:.3f}")
        print(f"       win p̄_ct={a['prob_contact']:.4f} 误关={a['false_silence_contact']:.3f} | "
              f"obj p̄_ct={b['prob_contact']:.4f} 误关={b['false_silence_contact']:.3f}")
