# -*- coding: utf-8 -*-
"""复现核查：用一个新训练的 checkpoint 重建主表数字，并与存储的报告逐项比对。

用法：
    python paper/repro_check.py --checkpoint_root checkpoints-REPRO \
        --arm ours_base --seed 42 [--report analysis_output/E0_report.json] [--rtol 0.05]

步骤：
  1) 调 build_e0_report.py（与原表**同一个**评估函数 e0_common.validate）从
     --checkpoint_root 重建一份临时报告；
  2) 取该臂该 seed 的单步 / 先验 / 闭环 k=2 / k=5，与存储报告里的对应条目比对；
  3) 打印差值、相对偏差与判定（默认相对容差 5%，即论文自报的噪声底量级）。

意义：把"论文数字来自该流程"从假设升级为可执行的核查。
"""
import argparse
import io
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = [("one_step_model", "one-step"), ("one_step_prior", "prior one-step"),
          ("rollout k=2", "closed-loop k=2"), ("rollout k=5", "closed-loop k=5")]


def get(rec, path):
    if path.startswith("rollout "):
        return rec.get("rollout", {}).get(path.split(" ", 1)[1])
    return rec.get(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_root", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--report", default="analysis_output/E0_report.json")
    ap.add_argument("--rtol", type=float, default=0.05)
    ap.add_argument("--tmp_out", default="analysis_output/_repro_tmp.json")
    args = ap.parse_args()

    # 1) 重建
    out = os.path.join(ROOT, args.tmp_out)
    cmd = [sys.executable, "build_e0_report.py",
           "--checkpoint_root", args.checkpoint_root,
           "--skip_analytic", "--out", args.tmp_out,
           "--title", "REPRO(%s)" % args.checkpoint_root]
    print("running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        print("FAILED: build_e0_report.py exit=%d" % rc)
        return 1
    if not os.path.exists(out):
        print("FAILED: %s not written" % out)
        return 1

    new = json.load(io.open(out, encoding="utf-8"))
    old = json.load(io.open(os.path.join(ROOT, args.report), encoding="utf-8"))

    def find(report, arm, seed):
        runs = report.get("arms", {}).get(arm)
        if not runs:
            return None
        for r in runs:
            if int(r.get("seed", -1)) == seed:
                return r
        return None

    a = find(old, args.arm, args.seed)
    b = find(new, args.arm, args.seed)
    if a is None or b is None:
        print("FAILED: entry not found (stored=%s reproduced=%s)" % (a is not None, b is not None))
        print("  stored arms:", sorted(old.get("arms", {})))
        print("  repro  arms:", sorted(new.get("arms", {})))
        return 1

    print("\n=== reproduction check: %s seed %d ===" % (args.arm, args.seed))
    print("  stored     : epoch=%s task=%s" % (a.get("epoch"), a.get("task")))
    print("  reproduced : epoch=%s task=%s" % (b.get("epoch"), b.get("task")))
    print("\n%-18s %14s %14s %10s %10s  %s" %
          ("quantity", "stored", "reproduced", "abs diff", "rel diff", "verdict"))
    print("-" * 84)
    worst, n_fail = 0.0, 0
    for path, label in FIELDS:
        va, vb = get(a, path), get(b, path)
        if va is None or vb is None:
            print("%-18s %14s %14s %10s %10s  %s" % (label, va, vb, "-", "-", "N/A"))
            continue
        d = abs(vb - va)
        rel = d / abs(va) if va else float("inf")
        ok = rel <= args.rtol
        worst = max(worst, rel)
        n_fail += 0 if ok else 1
        print("%-18s %14.6f %14.6f %10.6f %9.2f%%  %s" %
              (label, va, vb, d, rel * 100, "OK" if ok else "DIFF"))
    print("-" * 84)
    print("worst relative deviation: %.2f%%  (tolerance %.0f%%)" % (worst * 100, args.rtol * 100))
    if n_fail:
        print("VERDICT: %d of %d quantities outside tolerance -> investigate" % (n_fail, len(FIELDS)))
    else:
        print("VERDICT: reproduced within tolerance")

    # 记录到 JSON（可溯源；GPU 非确定性导致的差异同样入库）
    rec = {"arm": args.arm, "seed": args.seed, "checkpoint_root": args.checkpoint_root,
           "source_report": args.report, "rtol": args.rtol,
           "stored_epoch": a.get("epoch"), "reproduced_epoch": b.get("epoch"),
           "quantities": {}, "worst_rel_diff": worst, "n_outside_tolerance": n_fail}
    for path, label in FIELDS:
        va, vb = get(a, path), get(b, path)
        if va is None or vb is None:
            continue
        rec["quantities"][label] = {"stored": va, "reproduced": vb,
                                    "rel_diff": abs(vb - va) / abs(va) if va else None}
    try:
        import torch
        pt = os.path.join(ROOT, args.checkpoint_root, "%s_s%d" % (args.arm, args.seed), "best.pt")
        if os.path.exists(pt):
            ck = torch.load(pt, map_location="cpu", weights_only=False)
            rec["n_params"] = int(sum(v.numel() for v in ck["model_state_dict"].values()
                                      if hasattr(v, "numel")))
    except Exception:
        pass
    rec_path = os.path.join(ROOT, "analysis_output",
                            "repro_check_%s_s%d.json" % (args.arm, args.seed))
    with io.open(rec_path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    print("[save] %s" % rec_path)

    return 0 if not n_fail else 2


if __name__ == "__main__":
    sys.exit(main())
