# -*- coding: utf-8 -*-
"""SLOT-2 汇总判定：T18/T19 机制数字的多种子复现（判据以预注册为准）。

预注册：experiments/prereg_S2_multiseed.json
输入：analysis_output/S2_g4_s{42,43,44}.json（崩塌，eval_g4）
      analysis_output/S2_nmmulti_s{42,43,44}.json（辨识，eval_nmmulti）

判据（预注册顺序）：
  R2-a 崩塌复现：3/3 seed，g125 与 drag 上 noprior k5 ≥ 3× hybrid_fixed k5
  R2-b 混叠复现：3/3 seed，clean nm5_H5 ĝ∈[600,850]（补偿盆地）；
       g125 nm5sub_H5 |ĝ/1125−1| ≤ 0.20
  R2-c 时程因果：3/3 seed，ĝ(H1) < ĝ(H5) < ĝ(H10) 且 k5(H10) ≤ k5(H1)

用法：python tmp_s2_multiseed.py
输出：控制台判定表 + analysis_output/S2_multiseed_verdict.json
"""
import io
import json
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SEEDS = [42, 43, 44]
K5 = lambda r: r[-1]


def load(path):
    return json.load(open(path, encoding="utf-8"))


def main():
    g4 = {s: load(f"analysis_output/S2_g4_s{s}.json") for s in SEEDS}
    nm = {s: load(f"analysis_output/S2_nmmulti_s{s}.json") for s in SEEDS}

    out = {"prereg": "experiments/prereg_S2_multiseed.json", "per_seed": {}}

    # ---------- R2-a 崩塌 ----------
    print("=== R2-a 崩塌复现（noprior k5 ≥ 3× hybrid_fixed k5）===")
    ra = {}
    for s in SEEDS:
        w = g4[s]["worlds"]
        row = {}
        for world in ("g125", "drag"):
            no = K5(w[world]["noprior"])
            hy = K5(w[world]["hybrid_fixed"])
            row[world] = dict(noprior=no, hybrid=hy, ratio=no / hy, ok=bool(no >= 3 * hy))
        ra[s] = row
        print(f"  s{s}: g125 {row['g125']['noprior']:.1f}/{row['g125']['hybrid']:.1f}"
              f"={row['g125']['ratio']:.1f}x {'[OK]' if row['g125']['ok'] else '[FAIL]'} | "
              f"drag {row['drag']['noprior']:.1f}/{row['drag']['hybrid']:.1f}"
              f"={row['drag']['ratio']:.1f}x {'[OK]' if row['drag']['ok'] else '[FAIL]'}")
    ra_ok = all(ra[s]["g125"]["ok"] and ra[s]["drag"]["ok"] for s in SEEDS)
    out["R2-a"] = {"per_seed": ra, "verdict": bool(ra_ok)}
    print(f"  ⇒ R2-a {'[OK] 3/3' if ra_ok else '[FAIL]'}")

    # ---------- R2-b 混叠 ----------
    print("\n=== R2-b 混叠复现（clean 补偿盆地 + g125 细化恢复）===")
    rb = {}
    for s in SEEDS:
        cl = nm[s]["worlds"]["clean"]["params"]["g"]   # clean 段的键是 "params"
        sub = nm[s]["worlds"]["g125"]["nm5sub_H5_params"]["g"]
        rb[s] = dict(g_clean=cl, basin_ok=bool(600 <= cl <= 850),
                     g_sub=sub, recover_ok=bool(abs(sub / 1125.0 - 1) <= 0.20))
        print(f"  s{s}: clean ĝ={cl:.0f} (盆地[600,850] {'[OK]' if rb[s]['basin_ok'] else '[FAIL]'}) | "
              f"g125 nm5sub ĝ={sub:.0f} (|err|={abs(sub/1125-1):.1%} "
              f"{'[OK]' if rb[s]['recover_ok'] else '[FAIL]'})")
    rb_ok = all(rb[s]["basin_ok"] and rb[s]["recover_ok"] for s in SEEDS)
    out["R2-b"] = {"per_seed": rb, "verdict": bool(rb_ok)}
    print(f"  ⇒ R2-b {'[OK] 3/3' if rb_ok else '[FAIL]'}")

    # ---------- R2-c 时程因果 ----------
    print("\n=== R2-c 时程因果复现（ĝ 单调 + k5(H10) ≤ k5(H1)，g125）===")
    rc = {}
    for s in SEEDS:
        w = nm[s]["worlds"]["g125"]
        g1 = w["nm5_H1_params"]["g"]
        g5 = w["nm5_H5_params"]["g"]
        g10 = w["nm5_H10_params"]["g"]
        mono = bool(g1 < g5 < g10)
        k_ok = bool(K5(w["nm5_H10"]) <= K5(w["nm5_H1"]))
        rc[s] = dict(g_H1=g1, g_H5=g5, g_H10=g10, mono=mono, k5_improved=k_ok)
        print(f"  s{s}: ĝ {g1:.0f} → {g5:.0f} → {g10:.0f} "
              f"{'[OK]' if mono else '[FAIL]'} | k5 H10={K5(w['nm5_H10']):.2f} "
              f"vs H1={K5(w['nm5_H1']):.2f} {'[OK]' if k_ok else '[FAIL]'}")
    rc_ok = all(rc[s]["mono"] and rc[s]["k5_improved"] for s in SEEDS)
    out["R2-c"] = {"per_seed": rc, "verdict": bool(rc_ok)}
    print(f"  ⇒ R2-c {'[OK] 3/3' if rc_ok else '[FAIL]'}")

    n_pass = sum(bool(x) for x in (ra_ok, rb_ok, rc_ok))
    out["summary"] = f"{n_pass}/3 主张 3/3 seed 复现"
    print(f"\n=== 总判决：{out['summary']} ===")

    with open("analysis_output/S2_multiseed_verdict.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/S2_multiseed_verdict.json")


if __name__ == "__main__":
    main()
