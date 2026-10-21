# -*- coding: utf-8 -*-
"""把已完成的统一 2x2 运行（tmp_ceiling_unified.py）结果落盘。

该运行在最后一格（w120k_no_roll / w120k_roll）中途被手动暂停，因此
`ceiling_unified.json` 没有被写出。但前三格中已完成的两格
（w400k_no_roll 全程、w400k_roll 到 epoch 600）已经打印到 stdout，
这里把它们转成结构化 JSON，避免结果只存在于终端日志里。

用法：python -X utf8 tmp_persist_unified_partial.py
输出：analysis_output/ceiling_unified_partial.json
"""
import json
import re

LOG = "tmp_unified_log.txt"
BASE_FF = 0.01967342384159565
BASE_CT = 0.014915704727172852
PAT = re.compile(
    r"(?P<cell>w\d+k_\w+)\s+ep(?P<ep>\d+)\s+huber=(?P<h>[\d.]+)\s+FF=(?P<ff>[\d.]+)\s+"
    r"\(x(?P<ffr>[\d.]+)\)\s+ct=(?P<ct>[\d.]+)\s+\(x(?P<ctr>[\d.]+)\)")

cells = {}
# PowerShell 的 Tee-Object 写成 UTF-16LE；两种编码都试，避免日志解析因编码静默失败
for enc in ("utf-16", "utf-8"):
    try:
        text = open(LOG, encoding=enc).read()
    except (UnicodeError, UnicodeDecodeError):
        continue
    hits = 0
    tmp = {}
    for line in text.splitlines():
        m = PAT.search(line)
        if not m:
            continue
        hits += 1
        tmp.setdefault(m.group("cell"), []).append(dict(
            epoch=int(m.group("ep")),
            train_huber=float(m.group("h")),
            head_ff_rms=float(m.group("ff")),
            head_ct_rms=float(m.group("ct")),
            ff_over_baseline=float(m.group("ffr")),
            ct_over_baseline=float(m.group("ctr")),
        ))
    if hits:
        cells, used_enc = tmp, enc
        break
if not cells:
    raise SystemExit(f"no result lines parsed from {LOG} (encoding?)")

out = dict(
    source_log=LOG, source_encoding=used_enc,
    status="partial: w120k cells never ran (run stopped by hand)",
    baseline=dict(head_ff_rms=BASE_FF, head_ct_rms=BASE_CT),
    w_ff=100.0, w_roll=0.3, epochs=1200, lr=1e-3, seed=42,
    head_architecture="cloned from model.state_head (no extra GELU)",
    cells=cells,
    matched_fit=None,
)
if "w400k_no_roll" in cells and "w400k_roll" in cells:
    nr = cells["w400k_no_roll"][-1]
    rr = min(cells["w400k_roll"],
             key=lambda r: abs(r["train_huber"] - nr["train_huber"]))
    out["matched_fit"] = dict(
        no_roll_epoch=nr["epoch"], roll_epoch=rr["epoch"],
        no_roll_huber=nr["train_huber"], roll_huber=rr["train_huber"],
        no_roll_ff_over=nr["ff_over_baseline"], roll_ff_over=rr["ff_over_baseline"],
        ratio=rr["head_ff_rms"] / nr["head_ff_rms"],
        no_roll_ct_over=nr["ct_over_baseline"], roll_ct_over=rr["ct_over_baseline"])

with open("analysis_output/ceiling_unified_partial.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("[save] analysis_output/ceiling_unified_partial.json")
for k, v in cells.items():
    print(f"  {k:16s} epochs={[r['epoch'] for r in v]}  "
          f"last FF/ct = {v[-1]['ff_over_baseline']:.3f} / {v[-1]['ct_over_baseline']:.3f}")
if out["matched_fit"]:
    mf = out["matched_fit"]
    print(f"  matched fit: {mf['no_roll_huber']:.5f} vs {mf['roll_huber']:.5f} -> "
          f"FF x{mf['ratio']:.2f}, contact x{mf['roll_ct_over']/mf['no_roll_ct_over']:.2f}")
