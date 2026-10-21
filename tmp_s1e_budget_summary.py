# -*- coding: utf-8 -*-
"""SLOT-1e 预算汇总：把 S1e_error_budget*.json 折成三项归因表。

用法：python tmp_s1e_budget_summary.py [json路径,逗号分隔可合并] [臂名,逗号分隔]
"""
import json
import sys

import numpy as np

SEEDS = [42, 43, 44]
ARMS = ["baseline", "wff10", "sgate", "W1", "W2", "S"]
# (条件 key, 中文标签)
CONDS = [
    ("head-gt-gt", "头噪声（真值反馈=无复合）"),
    ("head-self-soft", "现状协议"),
    ("head-self-gt", "接触反馈=真值"),
    ("head-self-freeze", "接触反馈=冻结"),
    ("zero-self-soft", "纯先验+自预测+爻头反馈"),
    ("zero-gt-gt", "纯先验+真值反馈（解析恒等）"),
]


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "analysis_output/S1e_error_budget.json"
    sel = sys.argv[2].split(",") if len(sys.argv) > 2 else ARMS
    paths = [p for p in spec.split(",") if p]
    merged = {"arms": {}}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        merged["arms"].update(d["arms"])
    d = merged
    arms = [a for a in sel if a in d["arms"]]
    print(f"[源] {', '.join(paths)}\n[臂] {','.join(arms)}")
    print(f"{'臂':10s} {'条件':26s} {'k=1(one)':>9s} {'闭环k=1':>8s} "
          f"{'k=2':>8s} {'k=3':>8s}")
    tab = {}
    for arm in arms:
        tab[arm] = {}
        for key, lab in CONDS:
            v1, v2, v3, vo = [], [], [], []
            for s in SEEDS:
                r = d["arms"][arm][str(s)][key]
                v1.append(r["k=1_oneclean"])
                vo.append(r["k=1_ffclean"])
                v2.append(r["k=2_ffclean"])
                v3.append(r["k=3_ffclean"])
            m = lambda x: float(np.mean(x))
            tab[arm][key] = dict(one=m(v1), k1=m(vo), k2=m(v2), k3=m(v3))
            print(f"{arm:10s} {lab:26s} {m(v1):9.4f} {m(vo):8.4f} "
                  f"{m(v2):8.4f} {m(v3):8.4f}")
        print()

    print("=== 三项归因（闭环 FF k=3，px，3 seed 均值）===")
    out = {}
    for arm in arms:
        t = tab[arm]
        head, free = t["head-gt-gt"]["k3"], t["head-self-soft"]["k3"]
        fb, prior = t["head-self-gt"]["k3"], t["zero-self-soft"]["k3"]
        ident = t["zero-gt-gt"]["k3"]
        comp = free / head if head else float("nan")
        one = t["head-self-soft"]["one"]
        out[arm] = dict(one=one, free=free, head=head, fb=fb, prior=prior,
                        ident=ident, compound=comp, feedback=free - fb,
                        tight=free - head)
        print(f"[{arm}] 单步 {one:.4f} | 闭环 k=3 {free:.4f} = 解析恒等 {ident:.5f}"
              f" + 头噪声 {head:.4f} + 自回归复合 {free - head:.4f}")
        print(f"        接触反馈通道贡献 {free - fb:.4f} | 纯先验自回归 {prior:.5f}"
              f" | 复合因子 {comp:.2f}×")
    print("\n=== 静默上限外推（同一复合因子下按头噪声线性缩放）===")
    for arm in arms:
        o = out[arm]
        print(f"[{arm}] 头噪声×0.5 ⇒ 闭环 ≈ {o['head'] * 0.5 * o['compound']:.3f}px"
              f" | ×0.1 ⇒ {o['head'] * 0.1 * o['compound']:.3f}px"
              f" | ×0.01 ⇒ {o['head'] * 0.01 * o['compound']:.4f}px"
              f"（R3 门槛 0.20 / S1c 门槛 1.02）")
    print("\n=== 完美静默的直接参照 ===")
    for arm in arms:
        print(f"[{arm}] 零头噪声 + 自预测 + 爻头反馈闭环 k=3 = "
              f"{tab[arm]['zero-self-soft']['k3']:.5f}px"
              f"（= 硬静默掩码的可达上限）")

    outp = "analysis_output/S1e_error_budget_all_summary.json"
    json.dump({a: out[a] for a in arms},
              open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n[save] {outp}")


if __name__ == "__main__":
    main()
