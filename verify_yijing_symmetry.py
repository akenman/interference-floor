# -*- coding: utf-8 -*-
"""
错综对称性检验 (verify_yijing_symmetry.py) — 理念层可证伪实验 (M2)
==================================================
回答一个问题：八卦结构是"真实存在于学到的动力学里"，还是"事后贴的命名"？

方法（三步，全部可证伪）：
  1. 三爻锚定：对每个粗原型 k，统计被分配到它的窗口在目标帧的三爻物理签名
     [初爻接触率, 中爻能量升率, 上爻趋近率]（引擎真值），按 0.5 阈值二值化为
     3-bit 卦象 → 卦名。若多个原型落入同一卦象，锚定退化，如实报告。
  2. 64 卦转移矩阵：相邻窗口的粗原型转移，经卦名聚合为 8×8 卦变矩阵 T。
  3. 对称性检验：
     - 综卦（爻序颠倒，初↔上，反相运动）: corr(T[g,g'], T[R(g'),R(g)])，R=位序反转
     - 错卦（三爻全变，对立相态）:       corr(T[g,g'], T[C(g),C(g')])，C=按位取反
     零分布：随机置换 卦↔原型 标签 10,000 次重算得分，p = P(S_null ≥ S_obs)。
     若 p 不显著，说明对称性不依赖于语义锚定——八卦只是命名，如实报告。

物理含义（本项目的爻定义）：初爻=接触/腾空，中爻=动能升/降，上爻=趋近/远离。
  111 乾 = 接触+能量升+趋近（撞击相）  000 坤 = 腾空+能量降+远离（飞散相）

用法：
  python verify_yijing_symmetry.py --checkpoint checkpoints-v3/best.pt
"""

import os
import json
import math
import argparse
import numpy as np
import torch

from layout import OBJ_DIMS, OBS_DIMS, ASSEMBLED_DIM, ACTION_DIM, NUM_OBJ, DT_V3, IMPULSE_MAX_V3
from model_v3 import WorldModelV3
from train_v3 import load_dataset, split_episodes, norm_stats, V3Data
from layout import yao_labels


def idx_train_safe(arr):
    return np.asarray(arr, dtype=np.int64)

# 卦名表：bits=(初爻,中爻,上爻)，1=阳
TRIGRAM_NAMES = {
    (0, 0, 0): "坤", (1, 0, 0): "震", (0, 1, 0): "坎", (0, 0, 1): "艮",
    (1, 1, 0): "兑", (1, 0, 1): "离", (0, 1, 1): "巽", (1, 1, 1): "乾",
}
ALL_PATTERNS = [(i >> 2 & 1, i >> 1 & 1, i & 1) for i in range(8)]  # 初中上


def trigram_name(bits):
    return TRIGRAM_NAMES[tuple(bits)]


def reverse_bits(bits):    # 综卦：爻序颠倒（初↔上，中不变）
    return (bits[2], bits[1], bits[0])


def complement_bits(bits):  # 错卦：三爻全变
    return (1 - bits[0], 1 - bits[1], 1 - bits[2])


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def collect_assignments(model, ds, device, max_windows):
    """返回 (采样索引, 窗口级粗原型分配)。三爻签名由调用方从 ds.yaos 取。"""
    idx_all = np.arange(len(ds))
    if max_windows and len(idx_all) > max_windows:
        rng = np.random.RandomState(0)
        idx_all = np.sort(rng.choice(idx_all, size=max_windows, replace=False))
    coarse = np.zeros(len(idx_all), dtype=np.int64)
    bs = 2048
    for j in range(0, len(idx_all), bs):
        idx = idx_all[j:j + bs]
        b = ds.gather(idx)
        xb = b["x"].to(device)
        _, c = model.encode_sequence(xb)
        _, coarse_l = model.evidence(c)
        coarse[j:j + len(idx)] = coarse_l.argmax(-1).cpu().numpy()
    return idx_all, coarse


def main():
    parser = argparse.ArgumentParser(description="错综对称性检验（M2）")
    parser.add_argument("--checkpoint", type=str, default="checkpoints-v3/best.pt")
    parser.add_argument("--data_path", type=str, default="trajectories_v3.npz")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max_windows", type=int, default=60000)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_report", type=str,
                        default="analysis_output/yijing_symmetry_report.json")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 68)
    print("错综对称性检验 — 八卦是结构还是命名？（M2 可证伪实验）")
    print("=" * 68)

    # ---- 模型与数据 ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    cfg = ckpt["config"]
    cfg = {**cfg, "tau": cfg.get("tau", DT_V3), "gravity": cfg.get("gravity", -900.0),
           "impulse_max": cfg.get("impulse_max", IMPULSE_MAX_V3)}
    model = WorldModelV3(**cfg).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    ns = ckpt["norm_stats"]

    states, obstacles, contact, actions = load_dataset(args.data_path)
    _, val_eps = split_episodes(states.shape[0])
    val_eps = val_eps[:args.episodes]
    ds = V3Data(states, obstacles, contact, actions, val_eps, 8, 3,
                ns["mean_obj"], ns["std_obj"], ns["mean_obs"], ns["std_obs"])

    idx_all, coarse = collect_assignments(model, ds, device, args.max_windows)
    # 目标帧三爻（窗口级签名 = 三物体该爻均值的占用率 ∈ [0,1]）
    ep_local = ds.ep_idx[idx_all]
    win = ds.win_start[idx_all]
    yao_full = ds.yaos[ep_local, win + ds.L]          # (N, 3轴, 3物体)
    yao_win = yao_full.mean(axis=2)                    # (N, 3)
    n_win_per_ep = ds.yaos.shape[1] - ds.L - ds.K + 1

    # ---- 步骤 1：三爻锚定（原型 → 卦象） ----
    K = model.num_coarse
    sig = np.zeros((K, 3))
    for k in range(K):
        m = coarse == k
        if m.sum() > 0:
            sig[k] = yao_win[m].mean(axis=0)
    bits = [(tuple((s > 0.5).astype(int))) for s in sig]
    names = [trigram_name(b) for b in bits]
    distinct = len(set(names))
    print(f"\n[锚定] 原型→卦象（按三爻签名 0.5 阈值）：")
    for k in range(K):
        print(f"  原型{k}: 签名=[接触{sig[k,0]:.2f} 能量升{sig[k,1]:.2f} 趋近{sig[k,2]:.2f}]"
              f" → {bits[k]} {names[k]}")
    print(f"  覆盖卦象数: {distinct}/8" + ("（锚定退化：有卦象未被占用/重复占用）" if distinct < 8 else ""))

    # ---- 步骤 2：64 卦转移矩阵（相邻窗口） ----
    # 相邻窗口对：(ep, i) 与 (ep, i+1)，i ≤ n_win-2
    pos = {}
    for j, (e, w) in enumerate(zip(ep_local, win)):
        pos[(e, w)] = j
    pairs = []
    for j in range(len(idx_all)):
        if win[j] < n_win_per_ep - 1:
            j2 = pos.get((ep_local[j], win[j] + 1))
            if j2 is not None:
                pairs.append((j, j2))
    pairs = np.asarray(pairs)
    print(f"\n[转移对] 相邻窗口对: {len(pairs):,}")

    T_tri = np.zeros((8, 8))
    for (ja, jb) in pairs:
        ga = ALL_PATTERNS.index(bits[coarse[ja]]) if bits[coarse[ja]] in ALL_PATTERNS else None
        gb = ALL_PATTERNS.index(bits[coarse[jb]]) if bits[coarse[jb]] in ALL_PATTERNS else None
        T_tri[ga, gb] += 1
    Tn = T_tri / max(T_tri.sum(), 1)
    active = np.where(Tn.sum(axis=1) > 0)[0]

    # ---- 步骤 3：对称性检验（标签置换零分布） ----
    def symmetry_scores(mat, transform):
        pairs_a, pairs_b, w = [], [], []
        for ga in range(8):
            for gb in range(8):
                ra, rb = transform(ALL_PATTERNS[ga]), transform(ALL_PATTERNS[gb])
                ia, ib = ALL_PATTERNS.index(ra), ALL_PATTERNS.index(rb)
                if mat[ga, gb] > 0 and mat[ia, ib] > 0 and (ga, gb) != (ia, ib):
                    pairs_a.append(mat[ga, gb])
                    pairs_b.append(mat[ia, ib])
                    w.append(1.0)
        if len(pairs_a) < 4:
            return 0.0, 0
        return pearson(pairs_a, pairs_b), len(pairs_a)

    results = {}
    for test_name, transform in [("zonggua_reversed", reverse_bits), ("cuagua_complement", complement_bits)]:
        s_obs, n_pairs = symmetry_scores(Tn, transform)
        null = []
        rng = np.random.RandomState(42)
        for _ in range(args.n_perm):
            perm = rng.permutation(8)
            Tp = np.zeros_like(Tn)
            for ga in range(8):
                for gb in range(8):
                    Tp[perm[ga], perm[gb]] = Tn[ga, gb]
            s_p, _ = symmetry_scores(Tp, transform)
            null.append(s_p)
        null = np.asarray(null)
        p_val = float((null >= s_obs).mean())
        results[test_name] = {
            "score": s_obs, "n_pairs": n_pairs, "p_value": p_val,
            "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "passed": bool(p_val < 0.05 and s_obs > 0.2),
        }
        cn = "综卦(反相)" if test_name == "zonggua_reversed" else "错卦(对立相态)"
        verdict = "成立" if results[test_name]["passed"] else "不成立"
        print(f"\n[{cn}] 对称性得分 = {s_obs:.3f}（{n_pairs} 对） | "
              f"零分布均值 {null.mean():.3f} / 95分位 {np.percentile(null, 95):.3f} | "
              f"p = {p_val:.4f} → {verdict}")

    # ---- 步骤 2b：数据层检验（不依赖模型） ----
    # 物体级三爻是天然二值（接触0/1、能量趋势符号、趋离符号），直接检验
    # 世界自身的物体级相态转移是否具有错综对称性——这是"世界性质"而非"模型性质"。
    print("\n[数据层检验] 物体级三爻相态转移（与模型无关）...")
    T_data = np.zeros((8, 8))
    n_trans = 0
    for ep in val_eps:
        yao_ep = yao_labels(states[ep], contact[ep].astype(np.int8), obstacles[ep])  # (T,3,3)
        idx_tri = (yao_ep[:, 0, :] * 4 + yao_ep[:, 1, :] * 2 + yao_ep[:, 2, :]).astype(int)  # (T,3)
        for o in range(NUM_OBJ):
            a = idx_train_safe(idx_tri[:-1, o])
            bidx = idx_tri[1:, o]
            np.add.at(T_data, (a, bidx), 1)
            n_trans += len(a)
    Td = T_data / max(T_data.sum(), 1)
    print(f"  物体级转移样本: {n_trans:,} | 覆盖卦象: 行{int((Td.sum(1) > 0).sum())}/8 列{int((Td.sum(0) > 0).sum())}/8")
    data_level_result = {}
    for test_name, transform in [("zonggua_reversed", reverse_bits),
                                 ("cuagua_complement", complement_bits)]:
        s_obs, n_p = symmetry_scores(Td, transform)
        null = []
        rng = np.random.RandomState(42)
        for _ in range(args.n_perm):
            perm = rng.permutation(8)
            Tp = np.zeros_like(Td)
            for ga in range(8):
                for gb in range(8):
                    Tp[perm[ga], perm[gb]] = Td[ga, gb]
            s_p, _ = symmetry_scores(Tp, transform)
            null.append(s_p)
        null = np.asarray(null)
        p_val = float((null >= s_obs).mean())
        passed = bool(p_val < 0.05 and s_obs > 0.2)
        data_level_result[test_name] = {
            "score": s_obs, "n_pairs": n_p, "p_value": p_val,
            "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "passed": passed,
        }
        cn = "综卦(反相)" if test_name == "zonggua_reversed" else "错卦(对立相态)"
        print(f"  [{cn}] 得分 = {s_obs:.3f}（{n_p} 对） | 零分布均值 {null.mean():.3f} | "
              f"p = {p_val:.4f} → {'成立' if passed else '不成立'}")
    data_level_result["any_passed"] = any(v.get("passed") for v in data_level_result.values()
                                          if isinstance(v, dict))
    data_level_result["n_transitions"] = int(n_trans)

    # ---- 描述统计：最频繁卦变 ----
    flat = [(ga, gb, Tn[ga, gb]) for ga in range(8) for gb in range(8) if Tn[ga, gb] > 0]
    flat.sort(key=lambda x: -x[2])
    top = [{"from": trigram_name(ALL_PATTERNS[ga]), "to": trigram_name(ALL_PATTERNS[gb]),
            "prob": float(p)} for ga, gb, p in flat[:10]]
    print("\n[最频繁卦变 Top-10]")
    for t in top:
        print(f"  {t['from']} → {t['to']}: {t['prob']:.3f}")

    # 静止分布熵
    stat = Tn.sum(axis=1)
    stat = stat / max(stat.sum(), 1e-9)
    stat = stat[stat > 0]
    ent = float(-(stat * np.log(stat)).sum() / math.log(8))
    print(f"\n[静止分布归一化熵] {ent:.3f}（1=八个相态均匀使用）")

    report = {
        "checkpoint": args.checkpoint, "data_path": args.data_path,
        "n_pairs": int(len(pairs)), "anchoring": {
            "signatures": [[float(x) for x in row] for row in sig],
            "bits": [[int(x) for x in b] for b in bits],
            "names": names, "distinct_trigrams": int(distinct)},
        "transition_top": top, "stationary_entropy": ent,
        "tests": results,
        "data_level": data_level_result,
        "overall": "PASSED" if any(r["passed"] for r in results.values())
                   or data_level_result.get("any_passed") else "NOT_ESTABLISHED",
    }
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 报告已保存: {args.output_report}")
    print("判读：任一对称性 p<0.05 且得分>0.2 → 八卦结构在转移图中有统计学支撑；")
    print("      两者皆不成立 → 如实记录'未建立'，理念层退回解释性框架。")


if __name__ == "__main__":
    main()
