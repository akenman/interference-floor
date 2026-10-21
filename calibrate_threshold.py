# -*- coding: utf-8 -*-
"""
阈值标定器 (calibrate_threshold.py) — "阈值必须先在试点数据上标定"的制度化
======================================================================
来源：DEEP_DIVE_REPORT.md §6 P0-4 · LONG_TASK_PLAN.md P2-4 · 2026-09-21
用途：任何"报警/门控/判决"阈值（如 E4 的卦置信度 α、安全门控阈值）在使用前，
      必须先在本工具上对**试点数据**标定，产出标定卡（JSON）。禁止口头拍阈值。

协议（对应执行守则第 1 条的阈值版）：
  1. 标定数据必须与使用数据分离（本工具只认 pilot/held-out，参数里必须声明来源）
  2. 产出标定卡：{写入时间, 数据哈希, 目标约束, 阈值, bootstrap 95% CI, 样本量}
  3. 使用方引用标定卡路径；无卡 = 未标定 = 不可用

支持的目标约束：
  --target fpr       假阳性率 ≤ value      （报警类：误报控制）
  --target precision 精确率 ≥ value        （报警类：报就要准）
  --target recall    召回率 ≥ value        （安全类：漏报控制）
  --target coverage  覆盖率 ≥ value        （区间类：|y-ŷ|≤q 的经验覆盖）

用法：
  python calibrate_threshold.py --scores pilot_scores.npy --labels pilot_labels.npy \
      --target precision --value 0.5 --data_source experiments/pilot.npz \
      --output analysis_output/calib_card_e4.json
  python calibrate_threshold.py --selftest   # 自检（合成数据，不产出卡片）
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np


def _hash(arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def bootstrap_ci(stat_fn, scores, labels, n_boot=1000, seed=0):
    """对给定统计量做 bootstrap 95% CI。"""
    rng = np.random.RandomState(seed)
    n = len(scores)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        v = stat_fn(scores[idx], labels[idx])
        if np.isfinite(v):
            stats.append(v)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def calibrate(scores, labels, target, value):
    """返回 (threshold, metric_at_threshold, mode)。
    scores 大 = 更倾向正类（报警）；labels ∈ {0,1}（1=真实不良事件）。
    coverage 模式：scores 为"是否被覆盖"的 0/1，直接返回经验覆盖率。
    """
    scores = np.asarray(scores, dtype=np.float64)
    if target == "coverage":
        return None, float(np.mean(scores)), "coverage"
    labels = np.asarray(labels, dtype=np.int64)
    assert labels.size == scores.size and set(np.unique(labels)) <= {0, 1}

    cands = np.unique(np.quantile(scores, np.linspace(0, 1, 501)))

    def metrics(t):
        alarm = scores >= t
        tp = float((alarm & (labels == 1)).sum())
        fp = float((alarm & (labels == 0)).sum())
        fn = float((~alarm & (labels == 1)).sum())
        tn = float((~alarm & (labels == 0)).sum())
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        fpr = fp / max(fp + tn, 1.0)
        return prec, rec, fpr

    ok = []
    for t in cands:
        prec, rec, fpr = metrics(t)
        if target == "fpr" and fpr <= value:
            ok.append((t, fpr))
        elif target == "precision" and prec >= value:
            ok.append((t, prec))
        elif target == "recall" and rec >= value:
            ok.append((t, rec))
    if not ok:
        return None, None, "infeasible"
    # 满足约束中取最松可用阈值（fpr/precision 取最小 t；recall 取最大 t）
    t_sel, m_sel = (ok[0] if target in ("fpr", "precision") else ok[-1])
    return float(t_sel), float(m_sel), target


def main():
    ap = argparse.ArgumentParser(description="阈值标定器（先标定后使用，产出标定卡）")
    ap.add_argument("--scores", type=str, help="试点数据分数 .npy（大=倾向报警）")
    ap.add_argument("--labels", type=str, help="试点数据标签 .npy（0/1）")
    ap.add_argument("--target", type=str, required=False,
                    choices=["fpr", "precision", "recall", "coverage"])
    ap.add_argument("--value", type=float, help="约束值（如 precision >= 0.5）")
    ap.add_argument("--data_source", type=str, default="",
                    help="试点数据来源声明（必填，写入标定卡）")
    ap.add_argument("--output", type=str, default="",
                    help="标定卡输出路径（默认 analysis_output/calib_card_<时间戳>.json）")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 20000)
        s = y * 1.5 + rng.randn(20000)          # AUC ~= 0.86 的合成信号
        t, m, mode = calibrate(s, y, "precision", 0.8)
        assert t is not None and m >= 0.8, f"precision 标定失败: {t}, {m}"
        alarm = s >= t
        tp = (alarm & (y == 1)).sum()
        fp = (alarm & (y == 0)).sum()
        held = tp / max(tp + fp, 1)
        assert held >= 0.78, f"留出复算精度 {held:.3f} 偏离标定 {m:.3f}"
        lo, hi = bootstrap_ci(lambda ss, ll: float((ss >= t).mean()), s, y, 200)
        assert lo <= (s >= t).mean() <= hi
        print(f"[selftest OK] precision>=0.8 → 阈值 {t:.3f}（留出复算 {held:.3f}），"
              f"报警率 CI [{lo:.3f}, {hi:.3f}]")
        t2, m2, _ = calibrate(s, y, "fpr", 0.05)
        assert t2 is not None and m2 <= 0.05
        print(f"[selftest OK] fpr<=0.05 → 阈值 {t2:.3f}（实测 {m2:.3f}）")
        print("[selftest OK] 全部通过")
        return

    if not (args.scores and args.target and args.value is not None):
        ap.error("非自检模式必须提供 --scores/--target/--value（coverage 可免 --labels）")
    if not args.data_source:
        ap.error("必须声明 --data_source（纪律：标定数据来源可追溯）")

    scores = np.load(args.scores)
    labels = np.load(args.labels) if args.labels else None
    t, m, mode = calibrate(scores, labels if labels is not None else scores,
                           args.target, args.value)
    if mode == "infeasible":
        print(f"[不可行] 试点数据上不存在满足 {args.target}={args.value} 的阈值。"
              f"如实记录，不得放宽门槛硬过。")
        sys.exit(2)

    card = {
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "先标定后使用（DEEP_DIVE P0-4 制度化）；标定数据与使用数据必须分离",
        "data_source": args.data_source,
        "data_hash": _hash([scores] + ([labels] if labels is not None else [])),
        "n_samples": int(len(scores)),
        "target": args.target, "constraint": args.value,
        "threshold": t, "metric_at_threshold": m,
    }
    if t is not None and labels is not None:
        lo, hi = bootstrap_ci(
            lambda ss, ll: float((ss >= t).mean()), scores, labels, 1000)
        card["alarm_rate_bootstrap95"] = [lo, hi]
    out = args.output or os.path.join(
        "analysis_output", f"calib_card_{int(time.time())}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)
    print(f"[OK] 阈值 = {t}（{args.target}={m:.4f} @ 约束 {args.value}）")
    print(f"[OK] 标定卡已保存: {out}——使用方须引用此卡，无卡 = 未标定")


if __name__ == "__main__":
    main()
