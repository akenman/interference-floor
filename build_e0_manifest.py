# -*- coding: utf-8 -*-
"""E0/E1a checkpoint 清单 — 入库可核验的「清单」，而非 65MB 权重本体。

背景：checkpoints-E0/ 共 21 个 best.pt（约 65MB），未入 git（仓库只跟踪少量 best.pt）。
为保留可复现性与取证链，本脚本把每个 checkpoint 的关键元信息 + 权重哈希写成 JSON 入库。

用法：python build_e0_manifest.py [目录 ...]   默认 checkpoints-E0
输出：analysis_output/checkpoints_manifest.json
"""
import glob
import hashlib
import json
import os
import sys
from datetime import datetime

import torch

OUT = "analysis_output/checkpoints_manifest.json"


def sha256_16(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def describe(pt):
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    args = ck.get("args", {}) or {}
    cfg = ck.get("config", {}) or {}
    sd = ck.get("model_state_dict", {}) or {}
    vm = ck.get("val_metrics", {}) or {}
    n_params = int(sum(v.numel() for v in sd.values()
                       if hasattr(v, "numel") and v.dtype.is_floating_point))
    return dict(
        path=pt.replace("\\", "/"),
        arm=args.get("arm", cfg.get("arm")),
        seed=args.get("seed", cfg.get("seed")),
        epoch=ck.get("epoch"),
        epochs_arg=args.get("epochs"),
        w_roll=args.get("w_roll"),
        lr=args.get("lr"),
        n_params=n_params,
        val_one_step=vm.get("one_step_model"),
        val_prior=vm.get("one_step_prior"),
        size_MB=round(os.path.getsize(pt) / 1048576, 2),
        sha256_16=sha256_16(pt),
        mtime=datetime.fromtimestamp(os.path.getmtime(pt)).isoformat(timespec="seconds"),
    )


def main():
    roots = sys.argv[1:] or ["checkpoints-E0"]
    rows, failures = [], []
    for root in roots:
        for dd in sorted(glob.glob(os.path.join(root, "*"))):
            pt = os.path.join(dd, "best.pt")
            if not os.path.isfile(pt):
                continue
            try:
                rows.append(describe(pt))
            except Exception as e:                      # noqa: BLE001
                failures.append(dict(path=pt, error=str(e)))

    d = dict(generated_at=datetime.now().isoformat(timespec="seconds"),
             roots=roots, n=len(rows), checkpoints=rows, failures=failures)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(d, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"清单 {OUT}: {len(rows)} 个 checkpoint，失败 {len(failures)}")
    for r in rows:
        print(f"  {r['path']:<42} seed={r['seed']} epoch={r['epoch']} "
              f"params={r['n_params']:,} one_step={r['val_one_step']} "
              f"{r['size_MB']}MB sha={r['sha256_16']}")
    for f in failures:
        print(f"  [FAIL] {f['path']}: {f['error']}")


if __name__ == "__main__":
    main()
