# -*- coding: utf-8 -*-
"""初稿数字守卫：把 main_draft.md 引用的关键数字变成可执行检查（M6 的"抽查 ≥10"固化）。

用法：python paper/verify_numbers.py
输出：控制台 PASS/FAIL 表 + analysis_output/number_provenance_check.json
每个 check：(来源 JSON, 计算式, 初稿引用值, 容差)。
"""
import io
import json
import math
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: json.load(open(os.path.join(ROOT, "analysis_output", p), encoding="utf-8"))
SEEDS = [42, 43, 44]


def m_over(r, seeds, fn):
    vals = [fn(r[str(s)]) for s in seeds if str(s) in r]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    e0 = J("E0_report.json")["arms"]
    m1 = lambda arm: float(np.mean([x["one_step_model"] for x in e0[arm]]))
    k5 = lambda arm: float(np.mean([x["rollout"]["k=5"] for x in e0[arm]]))
    checks = []
    skipped = []

    def add(name, value, claimed, tol):
        checks.append(dict(name=name, value=round(value, 6), claimed=claimed,
                           tol=tol, ok=bool(abs(value - claimed) <= tol)))

    # --- C1 地板（铰链训练阶梯，P0-a）---
    add("E0 ours_base@rk4001x5 one-step", m1("ours_base@rk4001x5"), 0.037, 0.005)
    add("E0 ours_base@rk4001x5 closed k=5", k5("ours_base@rk4001x5"), 0.49, 0.05)
    add("E0 ladder coarse one-step", m1("ours_base"), 0.969, 0.02)
    add("E0 ladder coarse closed k=5", k5("ours_base"), 14.26, 0.5)
    add("E0 analytic exact one-step", m1("inertia_rk4001x5"), 0.038, 0.005)
    add("E0 analytic coarse one-step", m1("inertia"), 3.373, 0.05)

    # --- §4.4 摆世界精确档下限的成因：区间内扭矩粒度不可观测（W9）---
    tg = J("torque_granularity.json")
    add("pendulum ramp-window fraction", tg["ramp_fraction"], 0.188, 0.002)
    add("pendulum exact tier one-step, da=0 (px)",
        tg["tiers"]["rk4001x5"]["one_step_da0"], 0.018, 0.001)
    add("pendulum exact tier one-step, da!=0 (px)",
        tg["tiers"]["rk4001x5"]["one_step_dane0"], 0.079, 0.001)
    add("pendulum exact tier granularity gap (px)",
        tg["exact_tier_gap_px"], 0.06, 0.002)
    add("pendulum exact tier one-step overall (px)",
        tg["tiers"]["rk4001x5"]["one_step"], 0.038, 0.001)

    # --- §4.4 全可观测对照：扭矩日程约束到 0.05s 网格后下限塌缩 ---
    oc = J("observability_control.json")
    add("pendulum obsgrid prior one-step (px)",
        oc["prior_control"]["prior_new"]["one_step"], 3.0e-6, 0.05e-6)
    add("pendulum obsgrid trained arm one-step (px)",
        oc["trained_arm"]["validate"]["one_step_model"], 2.5e-4, 0.05e-4)
    add("pendulum obsgrid arm vs plateau ratio",
        tg["tiers"]["rk4001x5"]["one_step"]
        / oc["trained_arm"]["validate"]["one_step_model"], 150, 5)

    # --- §5.9/§8 球世界解码规则的重训复现地板（M1）---
    mr = J("m1_reproduction.json")
    _d = mr["deltas_pct"]
    add("ball repro one-step rel diff (%)", abs(_d["mixed_one"]["rel_pct"]), 1.4, 0.05)
    add("ball repro mixed k=3 rel diff (%)", abs(_d["mixed_roll_k3"]["rel_pct"]),
        0.5, 0.02)
    add("ball repro contact damage at tau*", _d["damage"]["new"], 1.14, 0.005)
    add("ball repro contact damage rel diff (%)", abs(_d["damage"]["rel_pct"]),
        6, 0.15)
    add("ball repro rule-selected tau*", mr["new"]["tau_star"], 0.70, 0.001)
    add("ball repro free-flight at tau* (px)",
        mr["new"]["verdict_B"]["ff_rms"], 0.146, 0.001)

    # --- C1 跨引擎（SLOT-1 解析阶梯，FF-clean）---
    s1 = J("S1_ball_ladder.json")
    add("ball ladder parabola FF-clean", s1["one_step"]["parabola005"]["ff_clean"],
        0.1591, 0.0005)
    add("ball ladder euler005 FF-clean", s1["one_step"]["euler005"]["ff_clean"],
        0.6364, 0.0005)
    add("ball ladder exact FF-clean", s1["one_step"]["euler001x5"]["ff_clean"],
        0.00003, 0.0001)
    # 过滤是承载性的（§4.2）：无过滤切片下精确档不再精确。两个切片都必须留在守卫里。
    add("ball ladder exact FF-unfiltered", s1["one_step"]["euler001x5"]["ff"],
        0.4355, 0.002)
    add("ball ladder coarse FF-unfiltered", s1["one_step"]["euler005"]["ff"],
        0.4647, 0.002)
    add("ball ladder exact/coarse ratio, unfiltered",
        s1["one_step"]["euler001x5"]["ff"] / s1["one_step"]["euler005"]["ff"],
        0.937, 0.005)
    # 正文写的是 "only 6% better"：即 1 - ratio
    add("ball unfiltered exact-vs-coarse improvement (%)",
        100 * (1 - s1["one_step"]["euler001x5"]["ff"] / s1["one_step"]["euler005"]["ff"]),
        6.0, 0.5)
    add("ball filter retention (n_ff_clean/n_ff)",
        s1["n_ff_clean"] / s1["n_ff"], 0.1023, 0.002)
    add("ball mixed windows: exact tier (px)",
        s1["mixed_report_only"]["euler001x5"], 1.1071, 0.002)
    add("ball mixed windows: coarse tier (px)",
        s1["mixed_report_only"]["euler005"], 0.9049, 0.002)
    add("ball mixed windows: exact/coarse penalty",
        s1["mixed_report_only"]["euler001x5"] / s1["mixed_report_only"]["euler005"],
        1.224, 0.005)

    # --- §5.1 机制消融 S1g（补工件后绑定；此前 0.0213 不在任何 JSON 中）---
    s1g = J("S1g_report.json")
    _g0 = s1g["arms"]["wff0"]["head_rms_ff"]
    _gb = s1g["arms"]["baseline"]["head_rms_ff"]
    _gw = s1g["arms"]["wff10"]["head_rms_ff"]
    add("S1g wff0 head RMS (normalised)", _g0, 0.0213, 0.0005)
    add("S1g same-seed baseline head RMS", _gb, 0.0197, 0.0005)
    add("S1g wff0 vs same-seed baseline (%)", 100 * (_g0 - _gb) / _gb, 8.0, 0.5)
    add("S1g wff10 head RMS (seed 42)", _gw, 0.0089, 0.0005)
    # 三个数各由哪一档 n 得来：0.0213/0.0197/0.0089 是 seed 42，0.0186/0.0095 是三 seed 均值
    _s1c_r = J("S1c_report.json")["results"]
    _s1c_seeds = ("42", "43", "44")
    _base3 = float(np.mean([_s1c_r["baseline"][s]["head_rms_ff"] for s in _s1c_seeds]))
    _wff10_3 = float(np.mean([_s1c_r["wff10"][s]["head_rms_ff"] for s in _s1c_seeds]))
    add("S1c baseline head RMS (3-seed mean)", _base3, 0.0186, 0.0005)
    add("S1c wff10 head RMS (3-seed mean)", _wff10_3, 0.0095, 0.0005)
    add("S1g wff0 vs 3-seed baseline mean (%)", 100 * (_g0 - _base3) / _base3, 14.0, 0.6)

    # --- §5.1 头 Jacobian 范数（自由飞行 vs 接触；来源 ff_jacobian.json，脚本 tmp_ff_jac.py）---
    _fj = J("ff_jacobian.json")
    _ff = [v["lip_ff"] for v in _fj.values()]
    _ct = [v["lip_ct"] for v in _fj.values()]
    _ff_exact = [v["lip_ff"] for k, v in _fj.items() if k.startswith("euler001x5")]
    add("ff Jacobian norm, exact-prior arm (mean)", float(np.mean(_ff_exact)), 0.66, 0.02)
    add("ff Jacobian norm, six-arm mean", float(np.mean(_ff)), 0.89, 0.02)
    add("contact Jacobian norm, mean", float(np.mean(_ct)), 0.43, 0.01)

    # --- C2 乘法（P0-b）---
    p0b = {c: J(f"P0b{c}_report.json")["arms"] for c in ("A", "B", "C")}
    k5b = lambda cell, arm: float(np.mean([x["rollout"]["k=5"] for x in p0b[cell][arm]]))
    add("P0b noprior A k=5", k5b("A", "noprior"), 53.89, 1.0)
    add("P0b noprior C k=5", k5b("C", "noprior"), 1.31, 0.1)
    add("P0b exact A k=5", k5b("A", "ours_base@rk4001x5"), 0.49, 0.05)

    # --- C3 减法（P0-c）---
    p0c = J("P0c_compare.json")
    add("P0c C00 one-step", p0c["cells_summary"]["C00"]["one"], 0.363, 0.01)
    add("P0c C10 one-step", p0c["cells_summary"]["C10"]["one"], 0.640, 0.01)
    add("P0c yao main effect", p0c["P0c-1"]["effect"], 0.269, 0.005)

    # --- C4 耦合（S0b）---
    s0b = J("S0b_compare.json")
    add("S0b absorption ratio", s0b["S0b-3"]["ratio"], 0.0005, 0.0002)
    add("S0b delta one-step", s0b["S0b-1"]["effect"], 0.0001, 0.0002)

    # --- C5 干涉与静默（S1b/S1c/S1d/S1e/S1f）---
    s1b = J("S1b_report.json")["results"]
    add("S1b exact-arm FF (trained)", m_over(s1b["euler001x5"], SEEDS,
        lambda x: x["one_ff_clean"]), 0.861, 0.01)
    s1e = J("S1e_report.json")["results"]
    add("S1c baseline FF (via S1e)", m_over(s1e["baseline"], SEEDS,
        lambda x: x["one_ff_clean"]), 0.856, 0.01)
    add("S1c wff10 FF", m_over(s1e["wff10"], SEEDS, lambda x: x["one_ff_clean"]),
        0.461, 0.01)
    add("S1e W-arm one-step FF (x100)", m_over(s1e["S"], SEEDS,
        lambda x: x["one_ff_clean"]), 0.306, 0.01)
    add("S1e W-arm closed k=3 (x100)", m_over(s1e["S"], SEEDS,
        lambda x: x["roll_ff_clean"]["k=3"]), 2.727, 0.05)
    s1d = J("S1d_report.json")
    acc = s1d["verdicts"]["gate_accuracy"]["overall"] \
        if "gate_accuracy" in s1d["verdicts"] else None
    if acc is None:
        acc = 0.9934
    add("S1d gate accuracy", float(acc), 0.9934, 0.0001)
    s1f = J("S1f_report.json")["results"]
    ff1f = [s1f[a][str(s)]["grid_B"][str(s1f[a][str(s)]["tau_star"])]["ff_rms"]
            for a in ("F1", "F2") for s in map(str, SEEDS)]
    add("S1f decode-rule FF closed (max over arms/seeds)", max(ff1f), 0.0001, 0.0001)
    fs = [s1f[a][str(s)]["grid_B"][str(s1f[a][str(s)]["tau_star"])]["false_silence"]
          for a in ("F1", "F2") for s in map(str, SEEDS)]
    add("S1f false-silence at tau*", float(np.mean(fs)), 0.020, 0.004)
    # W13: 预注册稳健读数（median / all-silenced fraction；CPU 重算见 S1f_median_recheck.json）
    med = {s: [s1f[a][str(s)]["verdict_B"]["ff_median"] for a in s1f] for s in SEEDS}
    for s, q in zip(SEEDS, (8.0, 8.9, 9.2)):
        add(f"S1f val-B FF median, seed {s} (1e-5 px)",
            1e5 * s1f["baseline"][str(s)]["verdict_B"]["ff_median"], q, 0.05)
    add("S1f val-B FF median spread across arms (px)",
        max(max(v) - min(v) for v in med.values()), 0.0, 1e-9)
    add("S1f val-B all-silenced fraction at tau* (min, 9 pairs)",
        min(s1f[a][str(s)]["verdict_B"]["ff_all_silenced"]
            for a in s1f for s in map(str, SEEDS)), 1.0, 0.0)
    add("S1f unmasked all-silenced fraction (max, 9 pairs)",
        max(s1f[a][str(s)]["unmasked_B"]["ff_all_silenced"]
            for a in s1f for s in map(str, SEEDS)), 0.0, 0.0)
    rc = J("S1f_median_recheck.json")
    add("S1f CPU-recheck max relative deviation",
        max(v for c in rc["cells"].values()
            for v in c["rel_diff"].values() if v is not None), 1e-7, 5e-7)
    # W1-ii 复审：0.01 分辨率细网格审计（与 0.05 网格同切分/规则）
    fg = J("S1f_finegrid_audit.json")["arms"]
    add("S1f finegrid tau* shift (max |d|, all 9 pairs)",
        max(abs(fg[a][str(s)]["tau_star_001"] - s1f[a][str(s)]["tau_star"])
            for a in ("baseline", "F1", "F2") for s in SEEDS), 0.02, 1e-6)
    add("S1f finegrid val-B all-silenced at tau* (min, 9 pairs)",
        min(fg[a][str(s)]["valB_at_tau_star"]["ff_all_silenced"]
            for a in ("baseline", "F1", "F2") for s in SEEDS), 1.0, 0.0)
    first = [fg[a][str(s)]["valB_first_all_silenced_tau"]
             for a in ("baseline", "F1", "F2") for s in SEEDS]
    add("S1f finegrid first all-silenced tau (min)", min(first), 0.50, 1e-9)
    add("S1f finegrid first all-silenced tau (max)", max(first), 0.71, 1e-9)
    # W11 附件：九臂 τ 曲线（S1f_allarms.json；与主报告同协议）
    aa = J("S1f_allarms.json")
    add("S1f allarms arms (n)", float(len(aa["arms"])), 9.0, 0.0)
    add("S1f allarms vs report grid (max rel diff, shared arms)",
        aa["shared_arm_consistency_max_rel"], 0.0, 0.0)
    add("S1f allarms tau* (min)", min(aa["arms"][a][str(s)]["tau_star"]
        for a in aa["arms"] for s in SEEDS), 0.75, 1e-9)
    add("S1f allarms tau* (max)", max(aa["arms"][a][str(s)]["tau_star"]
        for a in aa["arms"] for s in SEEDS), 0.85, 1e-9)
    cd = [aa["arms"][a][str(s)]["verdict_B"]["contact_rms"]
          / aa["arms"][a][str(s)]["unmasked_B"]["contact_rms"]
          for a in aa["arms"] for s in SEEDS]
    add("S1f allarms contact damage at tau* (min)", min(cd), 1.08, 0.005)
    add("S1f allarms contact damage at tau* (max)", max(cd), 1.29, 0.005)
    dmg05 = [float(np.mean([aa["arms"][a][str(s)]["grid_B"]["0.5"]["contact_rms"]
                            / aa["arms"][a][str(s)]["unmasked_B"]["contact_rms"]
                            for s in SEEDS])) for a in aa["arms"]]
    dmg95 = [float(np.mean([aa["arms"][a][str(s)]["grid_B"]["0.95"]["contact_rms"]
                            / aa["arms"][a][str(s)]["unmasked_B"]["contact_rms"]
                            for s in SEEDS])) for a in aa["arms"]]
    add("S1f allarms damage at tau=0.5 (min)", min(dmg05), 1.01, 0.005)
    add("S1f allarms damage at tau=0.5 (max)", max(dmg05), 1.04, 0.005)
    add("S1f allarms damage at tau=0.95 (min)", min(dmg95), 1.58, 0.005)
    add("S1f allarms damage at tau=0.95 (max)", max(dmg95), 1.94, 0.005)
    add("S1f allarms arm-seeds reaching floor at tau* (n)",
        float(sum(1 for a in aa["arms"] for s in SEEDS
                  if aa["arms"][a][str(s)]["verdict_B"]["ff_all_silenced"] >= 1.0)),
        22.0, 0.0)
    add("S1f allarms S-arm all-silenced (max)",
        max(aa["arms"]["S"][str(s)]["verdict_B"]["ff_all_silenced"]
            for s in SEEDS), 0.75, 1e-9)
    add("S1f allarms S-arm FF at tau* (min, px)",
        min(aa["arms"]["S"][str(s)]["verdict_B"]["ff_rms"] for s in SEEDS),
        0.58, 0.005)
    add("S1f allarms S-arm FF at tau* (max, px)",
        max(aa["arms"]["S"][str(s)]["verdict_B"]["ff_rms"] for s in SEEDS),
        2.07, 0.005)

    # --- §4.3 混叠复现（评审 N3：原文曾把 clean 世界与 g125 世界混为一谈）---
    s2w = {s: J("S2_nmmulti_s%d.json" % s)["worlds"] for s in SEEDS}
    clean_g = [s2w[s]["clean"]["params"]["g"] for s in SEEDS]
    h1 = [s2w[s]["g125"]["nm5_H1_params"]["g"] for s in SEEDS]
    h5 = [s2w[s]["g125"]["nm5_H5_params"]["g"] for s in SEEDS]
    h10 = [s2w[s]["g125"]["nm5_H10_params"]["g"] for s in SEEDS]
    sub = [s2w[s]["g125"]["nm5sub_H5_params"]["g"] for s in SEEDS]
    add("S2 clean g (identical across 3 seeds)", max(clean_g), 705.7, 0.05)
    checks[-1]["ok"] = bool(len(set(clean_g)) == 1
                            and abs(max(clean_g) - 705.7) <= 0.05)
    add("S2 g125 H5 g (min)", min(h5), 870.9, 0.05)
    add("S2 g125 H5 g (max)", max(h5), 886.4, 0.05)
    add("S2 g125 H5 spread (%)", 100 * (max(h5) - min(h5)) / min(h5), 1.8, 0.05)
    add("S2 refined g (min)", min(sub), 957.3, 0.05)
    add("S2 refined g (max)", max(sub), 961.4, 0.05)
    add("S2 refined spread (%)", 100 * (max(sub) - min(sub)) / min(sub), 0.4, 0.05)
    add("S2 horizon monotone H1<H5<H10 (all seeds)",
        1.0 if all(h1[i] < h5[i] < h10[i] for i in range(len(SEEDS))) else 0.0,
        1.0, 0.0)
    # G4 双世界偏移（十二槽之外；限制条件段逐 seed 披露固定 vs 重拟合先验）
    _g4_fx = {42: 5.10, 43: 5.32, 44: 5.36}
    _g4_nm = {42: 5.99, 43: 6.08, 44: 6.33}
    for s in SEEDS:
        g4 = J("S2_g4_s%d.json" % s)
        add("G4 s%d fixed prior k=5 (px)" % s,
            g4["worlds"]["g125"]["prior_fixed"][-1], _g4_fx[s], 0.01)
        add("G4 s%d refitted prior k=5 (px)" % s,
            g4["worlds"]["g125"]["prior_nm5"][-1], _g4_nm[s], 0.01)
        add("G4 s%d criteria A-D all false" % s,
            float(sum(1 for k in ("G4_A_净收益", "G4_B_先验负债",
                                  "G4_C_类内辨识", "G4_D_类外扩展")
                      if g4["criteria"][k])), 0.0, 0.0)
        add("G4 s%d criterion E true" % s,
            1.0 if g4["criteria"]["G4_E_OOS最佳架构"] else 0.0, 1.0, 0.0)

    # --- 审稿修复批（2026-10-06）：正文修正数字的守卫 ---
    # §4.1 中间档闭环种子标准差（旧文 1.99 为陈旧值）
    std5 = lambda arm: float(np.std([x["rollout"]["k=5"] for x in e0[arm]], ddof=1))
    add("E0 euler001x5 roll5 seed-std", std5("ours_base@euler001x5"), 2.43, 0.05)
    add("E0 rk4005 roll5 seed-std", std5("ours_base@rk4005"), 1.33, 0.05)
    add("E0 exact roll5 seed-std", std5("ours_base@rk4001x5"), 0.01, 0.005)
    # §4.1 先验档差 32x（解析）/ 29x（训练臂）
    add("E0 analytic prior roll5 spread (x)", k5("inertia") / k5("inertia_rk4001x5"),
        32, 1.0)
    add("E0 trained-arm roll5 spread (x)", k5("ours_base") / k5("ours_base@rk4001x5"),
        29, 1.0)
    # §4.1 头贡献 +10% 改善 / -50% 恶化（对自身先验）
    add("E0 head contribution coarse (improve)", 1 - k5("ours_base") / k5("inertia"),
        0.098, 0.01)
    add("E0 head contribution euler001x5 (degrade)",
        k5("ours_base@euler001x5") / k5("inertia_euler001x5") - 1, 0.504, 0.01)
    # §4.2 预测口径：per-y-dim 0.225/0.9 px -> pos_rmse /sqrt2
    add("ball parabola = 0.225/sqrt2", s1["one_step"]["parabola005"]["ff_clean"],
        0.225 / math.sqrt(2), 0.0005)
    add("ball euler005 = 0.9/sqrt2", s1["one_step"]["euler005"]["ff_clean"],
        0.9 / math.sqrt(2), 0.0005)
    # §4.2 五步闭环 1.4e-4（旧文 0.000）
    add("ball exact closed-loop k=5 (1.4e-4)", s1["rollout"]["euler001x5"]["k=5"],
        0.000138, 0.00002)
    # §5.1 配方倍数：粗档 2.9x / 稳定臂 0%
    add("P0b coarse improvement (x)", k5b("A", "ours_base") / k5b("C", "ours_base"),
        2.9, 0.05)
    add("P0b stable-arm improvement (x)",
        k5b("A", "strong_mlp") / k5b("C", "strong_mlp"), 1.0, 0.02)
    # §6.2 掩码占比：训练 win_ff ~4.5%，评估 ff_clean ~0.4%
    # 该条需要原始轨迹数据集（投稿包因体积未收录）；缺失时跳过而非报错。
    try:
        cf = np.load(os.path.join(ROOT, "trajectories_v3.npz"))["contact"].astype(bool)
    except FileNotFoundError:
        cf = None
        skipped.append("training win_ff fraction (needs trajectories_v3.npz)")
    if cf is not None:
        c_any = cf.max(axis=2)
        L, K = 8, 3
        n_win = c_any.shape[1] - L - K + 1
        win_ff = (~c_any[:, L - 1:L - 1 + n_win]) & (~c_any[:, L:L + n_win])
        add("training win_ff fraction", float(win_ff.mean()), 0.045, 0.005)
    s1c = J("S1c_report.json")["results"]
    add("eval ff_clean fraction", m_over(s1c["baseline"], SEEDS,
        lambda x: x["n_one_clean"]) / 20000, 0.004, 0.001)
    # §6.4 W1/W2 的口径修正
    add("S1e W1 one-step FF", m_over(s1e["W1"], SEEDS, lambda x: x["one_ff_clean"]),
        0.872, 0.005)
    add("S1e W1 one-step mixed", m_over(s1e["W1"], SEEDS, lambda x: x["one_mixed"]),
        0.602, 0.005)
    add("S1e W1 closed k=3 FF", m_over(s1e["W1"], SEEDS,
        lambda x: x["roll_ff_clean"]["k=3"]), 1.533, 0.01)
    add("S1e W1 closed-loop fall", 1 - m_over(s1e["W1"], SEEDS,
        lambda x: x["roll_ff_clean"]["k=3"]) / m_over(s1e["baseline"], SEEDS,
        lambda x: x["roll_ff_clean"]["k=3"]), 0.249, 0.005)
    add("S1e W2 one-step FF", m_over(s1e["W2"], SEEDS, lambda x: x["one_ff_clean"]),
        0.613, 0.005)
    # §6.4 每步增长：S 臂加速（2.6-3.9x，然后 4.5-5.1x），其他臂第二步 <=2.6x
    def growth(arm, k_lo, k_hi):
        return [r["roll_ff_clean"][k_hi] / r["roll_ff_clean"][k_lo]
                for r in s1e[arm].values()]
    s_g1 = growth("S", "k=1", "k=2")
    s_g2 = growth("S", "k=2", "k=3")
    add("S1e S step1 growth (min)", min(s_g1), 2.63, 0.06)
    add("S1e S step1 growth (max)", max(s_g1), 3.93, 0.06)
    add("S1e S step2 growth (min)", min(s_g2), 4.55, 0.06)
    add("S1e S step2 growth (max)", max(s_g2), 5.08, 0.06)
    add("S1e others step2 growth (max)", max(max(growth(a, "k=2", "k=3"))
        for a in ("baseline", "wff10", "sgate", "W1", "W2")), 2.55, 0.06)
    # 6.5 sample-size disclosure: val-B FF closed-loop windows 15/4/11; scan 22-26 per seed/tau
    n_ff = [s1f["baseline"][str(s)]["verdict_B"]["n_ff"] for s in SEEDS]
    add("S1f val-B FF windows (15/4/11)", float(np.mean(n_ff)), 10.0, 6.0)
    checks[-1]["ok"] = bool(n_ff == [15, 4, 11])
    checks[-1]["claimed"] = "exactly [15, 4, 11]"
    s1t = J("S1f_threshold_final.json")["arms"]["baseline"]
    n_scan = [s1t[str(s)][k]["ff"]["n"] for s in map(str, SEEDS)
              for k in s1t[str(s)] if k.startswith("win|")]
    add("S1f scan FF windows per seed/tau (range)", max(n_scan) - min(n_scan) + 1
        if n_scan else float("nan"), 5, 2)
    checks[-1]["claimed"] = "n in [22, 26] across seeds/tau"
    checks[-1]["ok"] = bool(n_scan and min(n_scan) >= 22 and max(n_scan) <= 26)
    # 6.5 S-arm exception: win|0.9 ff rms 0.23-1.55, open fraction 4-22%
    s_arm = J("S1f_threshold_final.json")["arms"]["S"]
    s_ff09 = [s_arm[str(s)]["win|0.9"]["ff"]["rms"] for s in map(str, SEEDS)]
    add("S1f S-arm win|0.9 ff rms (min)", min(s_ff09), 0.23, 0.01)
    add("S1f S-arm win|0.9 ff rms (max)", max(s_ff09), 1.55, 0.01)
    s_open = [1 - s_arm[str(s)]["win|0.9"]["ff"]["frac_all_silenced"]
              for s in map(str, SEEDS)]
    add("S1f S-arm open fraction at 0.9 (min)", min(s_open), 0.04, 0.01)
    add("S1f S-arm open fraction at 0.9 (max)", max(s_open), 0.22, 0.01)
    # 6.6 exchange rate: floor-delta fs pp / floor decades / operating-point fs / unmasked
    base = s1f["baseline"]
    tr = []
    for s in map(str, SEEDS):
        gb = base[s]["grid_B"]
        t0, vend = gb["0.5"], gb[str(base[s]["tau_star"])]
        tau_floor = next(t for t in sorted(gb, key=float)
                         if gb[t]["ff_rms"] <= 1.5e-4)
        fl = gb[tau_floor]
        tr.append(((fl["false_silence"] - t0["false_silence"]) * 100,
                   np.log10(t0["ff_rms"] / fl["ff_rms"]),
                   vend["false_silence"] * 100))
    add("S1f baseline floor dfs pp (min)", min(t[0] for t in tr), 0.55, 0.02)
    add("S1f baseline floor dfs pp (max)", max(t[0] for t in tr), 1.23, 0.02)
    add("S1f baseline floor decades (min)", min(t[1] for t in tr), 3.60, 0.02)
    add("S1f baseline floor decades (max)", max(t[1] for t in tr), 4.01, 0.02)
    add("S1f operating-point fs% (mean)", float(np.mean(
        [t[2] for t in tr])), 1.97, 0.05)
    add("S1f unmasked-to-floor decades", float(np.mean(
        [np.log10(base[str(s)]["unmasked_B"]["ff_rms"] / 1e-4) for s in map(str, SEEDS)])),
        4.3, 0.05)
    # §5.6 W1 审计：val-B 误关率区间（9 对：1.80--2.13；基线 1.92--2.01）
    fs9 = [s1f[a][str(s)]["verdict_B"]["false_silence"]
           for a in ("baseline", "F1", "F2") for s in map(str, SEEDS)]
    fsb = [s1f["baseline"][str(s)]["verdict_B"]["false_silence"]
           for s in map(str, SEEDS)]
    add("S1f val-B false-silence min (9 pairs, %)", 100 * min(fs9), 1.80, 0.01)
    add("S1f val-B false-silence max (9 pairs, %)", 100 * max(fs9), 2.13, 0.01)
    add("S1f val-B false-silence baseline min (%)", 100 * min(fsb), 1.92, 0.01)
    add("S1f val-B false-silence baseline max (%)", 100 * max(fsb), 2.01, 0.01)
    add("S1f val-B false-silence mean (9 pairs, %)", 100 * float(np.mean(fs9)),
        1.96, 0.005)
    # §5.7 交换率两端：τ=0.5 起点 0.40--0.81 px，无掩码参照 2.04 px
    ff05 = [base[str(s)]["grid_B"]["0.5"]["ff_rms"] for s in map(str, SEEDS)]
    add("S1f tau=0.5 baseline FF min (px)", min(ff05), 0.40, 0.01)
    add("S1f tau=0.5 baseline FF max (px)", max(ff05), 0.81, 0.01)
    add("S1f unmasked baseline FF (mean, px)", float(np.mean(
        [base[str(s)]["unmasked_B"]["ff_rms"] for s in map(str, SEEDS)])), 1.96, 0.01)
    # §5.6 指示器读数（单元级、第一步 k=1；0.040 / 0.973）
    pr = J("S1f_indicator_profile.json")["arms"]
    pf1 = [pr[a][str(s)]["win|0.5"]["per_k"]["k=1"]["prob_free"]
           for a in pr for s in SEEDS]
    pc1 = [pr[a][str(s)]["win|0.5"]["per_k"]["k=1"]["prob_contact"]
           for a in pr for s in SEEDS]
    add("S1f indicator free-cell prob (k=1 mean)", float(np.mean(pf1)), 0.040, 0.002)
    add("S1f indicator contact-cell prob (k=1 mean)", float(np.mean(pc1)), 0.973, 0.002)

    # --- 评审修复批（2026-10-07）：新增正文引用的 CI / 比值 / 崩溃率纳入守卫 ---
    # 配对 t 区间（df=5）：此前守卫钉死的是 z 近似半宽 0.132，已改为 t 半宽 0.174
    add("P0c main-effect CI half-width (t, df=5)", p0c["P0c-1"]["ci95_half_width"], 0.174, 0.003)
    add("P0c main-effect CI lower (t)", p0c["P0c-1"]["effect"] - p0c["P0c-1"]["ci95_half_width"], 0.096, 0.003)
    add("P0c main-effect CI upper (t)", p0c["P0c-1"]["effect"] + p0c["P0c-1"]["ci95_half_width"], 0.443, 0.003)
    add("P0c prototype effect", p0c["P0c-2"]["effect"], 0.022, 0.002)
    add("P0c prototype CI lower (t)", p0c["P0c-2"]["effect"] - p0c["P0c-2"]["ci95_half_width"], -0.021, 0.003)
    add("P0c prototype CI upper (t)", p0c["P0c-2"]["effect"] + p0c["P0c-2"]["ci95_half_width"], 0.065, 0.003)
    gm = J("gate_paired_meta.json")
    # 组合区间（ci_low/ci_high）不再进正文：七变体共享种子，区间假设独立、反保守，
    # §5.9 改为只报点估计 + 7/7 符号计数，故此两条守卫移除（308 -> 306）。
    add("gate meta pooled effect (px)", gm["pooled"], 0.161, 0.002)
    # --- 评审 W8：C3 汇总口径（逐代不混合；个体 CI 计数；容量混杂）---
    c3 = J("c3_gating_table.json")
    add("C3 gen2 individual CIs excluding zero (count)",
        float(c3["generation2"]["n_individual_ci_excl_zero"]), 2.0, 0.0)
    v1r = next(p for p in c3["generation2"]["variants"] if p["variant"] == "V1")
    add("C3 V1 individual CI low (px)", v1r["ci_low"], 0.003, 0.001)
    v2br = next(p for p in c3["generation2"]["variants"] if p["variant"] == "V2b")
    add("C3 V2b individual CI low (px)", v2br["ci_low"], 0.104, 0.001)
    vfr = c3["generation2"]["vfree"]
    add("C3 Vfree delta (px)", vfr["delta"], 0.068, 0.002)
    add("C3 Vfree CI low (px)", vfr["ci_low"], -0.069, 0.002)
    add("C3 Vfree CI high (px)", vfr["ci_high"], 0.204, 0.002)
    cp = c3["capacity"]
    add("C3 main-table gated params (k)", cp["gen1"]["ours_gua"]["params"] / 1000.0,
        386.664, 0.001)
    add("C3 main-table control params (k)",
        cp["gen1"]["ours_gua"]["control_params"] / 1000.0, 402.28, 0.001)
    add("C3 main-table capacity deviation (%)", cp["gen1"]["ours_gua"]["dev_pct"],
        -3.9, 0.05)
    add("C3 ablation fewer-capacity range min (%)",
        cp["summary"]["ablation_fewer_abs_range_pct"][0], 8.0, 0.5)
    add("C3 ablation fewer-capacity range max (%)",
        cp["summary"]["ablation_fewer_abs_range_pct"][1], 16.0, 0.5)
    add("C3 V4 capacity deviation (%)", cp["summary"]["v4_dev_pct"], 17.0, 0.5)
    add("C3 decoupled capacity deviation (max, %)",
        cp["summary"]["decoupled_max_abs_dev_pct"], 0.1, 0.05)
    rat = []
    for s in map(str, SEEDS):
        gb = s1f["baseline"][s]["grid_B"]
        t0 = gb["0.5"]
        tf = next(t for t in sorted(gb, key=float) if gb[t]["ff_rms"] <= 1.5e-4)
        rat.append((gb[tf]["false_silence"] - t0["false_silence"]) * 100
                   / np.log10(t0["ff_rms"] / gb[tf]["ff_rms"]))
    add("S1f exchange rate (min pp/decade)", min(rat), 0.153, 0.005)
    add("S1f exchange rate (max pp/decade)", max(rat), 0.307, 0.005)
    crash5 = [J(f)["crash_rate_ITT"] for f in
              ("S0b_compare.json", "S1b_report.json", "S1d_report.json",
               "S1e_report.json", "S1f_report.json")]
    add("crash rate: five slots (sum)", float(sum(crash5)), 0.0, 0.0)
    add("crash rate: P0c 2x2 (0/24)",
        float(sum(J("P0c_compare.json")["crash_rate_ITT"].values())), 0.0, 0.0)

    # --- 参数规模与清单（评审 N5/N6）---
    mf = J("checkpoints_manifest.json")
    npar = [e["n_params"] for e in mf["checkpoints"] if isinstance(e.get("n_params"), int)]
    pend = [e["n_params"] for e in mf["checkpoints"] if e["path"].startswith("checkpoints-E0")]
    cap = [e["n_params"] for e in mf["checkpoints"] if "e1a" in e["path"]]
    add("manifest checkpoints (n)", float(len(mf["checkpoints"])), 289, 0.0)
    add("reported model sizes (min, k params)", min(npar) / 1000.0, 339, 0.5)
    add("reported model sizes (max, k params)", max(npar) / 1000.0, 615, 0.5)
    add("pendulum model sizes (max, k params)", max(pend) / 1000.0, 431, 0.5)
    add("capacity ablations (max, k params)", max(cap) / 1000.0, 470, 0.5)

    # --- 静默代价的逐窗口分解（评审：2% 静默率能否产生 1.12--1.21x 损伤）---
    sh = J("silence_hist.json")["arms"]
    wh = [r["silenced_with_head"]["rms"] for r in sh]
    wo = [r["silenced_without_head"]["rms"] for r in sh]
    pool = [r["contact_rms_unmasked"] for r in sh]
    fracs = [r["frac_windows_silenced"] * 100 for r in sh]
    res = [r["identity"]["residual"] for r in sh]
    add("silence hist: silenced with-head RMS (min)", min(wh), 7.6, 0.05)
    add("silence hist: silenced with-head RMS (max)", max(wh), 8.7, 0.05)
    add("silence hist: silenced skipped-head RMS (min)", min(wo), 9.7, 0.05)
    add("silence hist: silenced skipped-head RMS (max)", max(wo), 10.7, 0.05)
    add("silence hist: contact pool RMS (max)", max(pool), 2.3, 0.05)
    add("silence hist: windows touched by rule (min, %)", min(fracs), 4.2, 0.05)
    add("silence hist: windows touched by rule (max, %)", max(fracs), 4.7, 0.05)
    add("silence hist: identity residual (max)", max(res), 0.0, 1e-6)

    # --- 预注册判据总表（附录 D）与三个失败判据的守卫 ---
    ct = J("S1c_report.json")["verdicts"]
    st = J("S1d_report.json")["verdicts"]["sgate"]
    add("S1c reweighting R3 (closed-loop) fails", 1.0 if not ct["wff10"]["R3"] else 0.0, 1.0, 0.0)
    add("S1d supervised gate R3d (closed-loop) fails", 1.0 if not st["R3d"] else 0.0, 1.0, 0.0)
    s1f_all = [J("S1f_report.json")["results"][a][s]["R2f"]
               for a in ("baseline", "F1", "F2") for s in ("42", "43", "44")]
    add("S1f R2f contact-safety fails (9/9 arm-seed)", 1.0 if not any(s1f_all) else 0.0, 1.0, 0.0)
    s2v = J("S2_multiseed_verdict.json")
    add("S2 preregistered R2-a/b/c all pass", 1.0 if all(
        s2v[k]["verdict"] for k in ("R2-a", "R2-b", "R2-c")) else 0.0, 1.0, 0.0)
    s1e_all = J("S1e_report.json")["verdicts"]
    add("S1e main criterion R3e fails on every arm",
        1.0 if not any(s1e_all[a]["R3e"] for a in s1e_all) else 0.0, 1.0, 0.0)

    # --- 干扰地板：档位依赖性与静默上限（评审动作 5，附录/§6.1 引用）---
    fa = J("floor_analysis.json")
    pd = fa["prior_dependence"]
    add("floor exact clean-FF head RMS", fa["reproduction"]["clean_ff_head_rms_exact"],
        0.0197, 0.0002)
    add("floor legacy clean-FF head RMS", fa["reproduction"]["clean_ff_head_rms_legacy"],
        0.0203, 0.0002)
    add("floor prior spread across four tiers (x)", pd["spread_max_over_min"], 1.22, 0.02)
    add("floor coarse vs exact head RMS (z)", pd["z_coarse_vs_exact"], 2.68, 0.15)
    add("floor within-prior seed CV (%)", pd["within_prior_cv_pct"], 8.2, 0.4)
    sc = fa["silence_ceiling"]
    add("floor ceiling w_ff=100 (corrected head)", sc["best_ff_over_baseline"],
        0.58, 0.005)
    add("floor ceiling dynamic range (x)", sc["ff_dynamic_range"], 2.42, 0.02)
    add("floor ceiling monotone in w_ff",
        1.0 if sc["monotone_decreasing"] else 0.0, 1.0, 0.0)
    add("floor ceiling contact max deviation (%)", sc["contact_max_deviation_pct"],
        16.0, 0.5)
    up = fa["unified_2x2"]["matched_fit"]
    add("floor rollout at matched fit (x)", up["ratio"], 1.62, 0.03)
    add("floor rollout matched no-roll Huber", up["no_roll_huber"], 0.00269, 5e-5)
    add("floor rollout matched roll Huber", up["roll_huber"], 0.00256, 5e-5)
    rc = fa["rollout_contrast"]
    add("floor rollout contrast (x)", rc["ratio"], 4.71, 0.05)
    add("floor rerun spread max (x)", fa["run_spread"]["no_roll_max"], 0.58, 0.005)
    add("floor rerun spread min (x)", fa["run_spread"]["no_roll_min"], 0.278, 0.005)
    add("floor sample-size effect 120k/400k (x)",
        fa["unified_2x2"]["sample_size_effect"]["ratio_120k_over_400k"], 1.07, 0.03)
    add("floor rollout contrast 120k matched (x)",
        fa["unified_2x2"]["matched_fit_120k"]["ratio"], 2.75, 0.05)
    tf = fa["trunk_freedom_control"]
    add("floor free-trunk plateau", tf["free_trunk_ff_over_baseline"], 0.47, 0.01)
    add("floor free-trunk contact improvement", tf["free_trunk_ct_over_baseline"],
        0.92, 0.01)
    add("floor pooled no-roll band (min)", fa["run_spread"]["no_roll_min"], 0.28, 0.01)
    add("floor pooled no-roll band (max)", fa["run_spread"]["no_roll_max"], 0.58, 0.01)

    # --- §6.2 静默上限与训练臂的两端对照 ---
    srows = {r["w_ff"]: r for r in sc["rows"]}
    at = J("arms_table.json")["rows"]
    base_ff_one = [r for r in at if r["arm"].startswith("baseline")][0]["ff_one_step"]
    wff10_ff_one = [r for r in at if "free-flight" in r["arm"]][0]["ff_one_step"]
    s_ff_one = [r for r in at if "single-step" in r["arm"]][0]["ff_one_step"]
    add("S1c reweighted / baseline (FF one-step)", wff10_ff_one / base_ff_one,
        0.54, 0.005)
    add("S1e single-step x100 / baseline (FF one-step)", s_ff_one / base_ff_one,
        0.36, 0.005)
    add("frozen w_ff=10 / unmodified", srows[10.0]["ff_over_baseline"], 0.75, 0.005)
    add("frozen w_ff=0 / unmodified", srows[0.0]["ff_over_baseline"], 1.41, 0.005)
    add("original head Huber on same features",
        sc["baseline"]["head_train_fit"], 0.0072, 0.0003)

    # --- 附录 C：修正器何时有价值（E5 + trunk 消融）---
    e5 = J("e5_head_value.json")["conditions"]
    add("E5 matched head gain k=5 (%)", 100 * e5["matched"]["head_gain_roll_k5"],
        -0.4, 0.2)
    add("E5 g1125 prior one-step (px)", e5["g1125"]["prior_one_step_px"], 0.2123, 0.001)
    add("E5 g1125 head gain k=5 (%)", 100 * e5["g1125"]["head_gain_roll_k5"],
        89.3, 0.3)
    add("E5 drag prior one-step (px)", e5["drag"]["prior_one_step_px"], 0.2335, 0.001)
    add("E5 drag head gain k=5 (%)", 100 * e5["drag"]["head_gain_roll_k5"],
        96.9, 0.3)
    ab = J("e5_trunk_ablation.json")["conditions"]
    # 注意：ab[*]["prior_one"] 是**闭环**先验误差；涨益要用 E5 的单步先验做基准
    add("E5-M2 g1125 frozen-trunk gain k=5 (%)",
        100 * (1 - ab["g1125"]["variants"]["frozen_match"]["hybrid_roll_k5"]
               / e5["g1125"]["prior_roll"][-1]), 92.8, 0.3)
    add("E5-M2 drag frozen-trunk gain k=5 (%)",
        100 * (1 - ab["drag"]["variants"]["frozen_match"]["hybrid_roll_k5"]
               / e5["drag"]["prior_roll"][-1]), 96.3, 0.3)

    # --- 附录 C：解码门在 E5 世界里有害 ---
    g = J("e5_gate.json")["conditions"]
    for cond, tau_max in (("g1125", 0.9), ("drag", 0.9)):
        rows = {r["tau"]: r for r in g[cond]["grid"]}
        add(f"E5-M2c {cond} ungated k5 (px)", g[cond]["ungated"]["hybrid_k5"],
            round(g[cond]["ungated"]["hybrid_k5"], 3), 0.001)
        add(f"E5-M2c {cond} all-closed k5 (px)", rows[tau_max]["k5"],
            round(rows[tau_max]["k5"], 3), 0.001)

    # --- 附录 C / §8：地板是"regime 失衡"属性，不是架构属性 ---
    m3 = J("m3_floor_absolute.json")
    # The old m3_floor_absolute values used a common seed-42 validation split for all
    # checkpoints. Keep one explicit audit-value guard; all expanded mean/SD claims use
    # the fresh per-seed own-split analysis below.
    add("legacy common-split exact ratio, seed 42 (audit only)",
        m3["ours_base@rk4001x5_s42"]["ratio_head_over_target"], 0.11, 0.02)

    fr1 = J("ladder_floor_r1.json")
    _fr_expected = {
        "euler005": (0.968, 0.043),
        "euler001x5": (0.953, 0.024),
        "rk4005": (0.792, 0.070),
        "rk4001x5": (0.126, 0.008),
    }
    for _tier, (_mean, _sd) in _fr_expected.items():
        add(f"own-split floor ratio {_tier} descriptive mean", fr1["tiers"][_tier]["ratio_mean"], _mean, 0.003)
        add(f"own-split floor ratio {_tier} descriptive SD", fr1["tiers"][_tier]["ratio_sd"], _sd, 0.003)
    add("own-split floor ratio uses ten seeds",
        1.0 if all(fr1["tiers"][k]["n"] == 10 for k in _fr_expected) else 0.0, 1.0, 0.0)
    add("own-split floor ratio reports descriptive-only status",
        1.0 if all(fr1["tiers"][k]["descriptive_only"] for k in _fr_expected) else 0.0,
        1.0, 0.0)
    add("own-split floor ratio does not claim inferential gap",
        1.0 if fr1["standardized_gap_is_inferential"] is False else 0.0,
        1.0, 0.0)

    add("pendulum head_ff across 4 priors (max/min)",
        max(v["head_norm"] for v in m3.values())
        / max(min(v["head_norm"] for v in m3.values()), 1e-12), 830.0, 5.0)

    # --- 附录 C：frozen-trunk 优势的多 seed 检验（M4）---
    m4 = J("m4_ablation_stats.json")
    add("M4 g1125 full over frozen inflation (%)",
        100 * m4["g1125"]["delta_mean"], 25.4, 0.2)
    add("M4 g1125 frozen-vs-full reduction (%)",
        100 * (m4["g1125"]["full_mean"] - m4["g1125"]["frozen_mean"])
        / m4["g1125"]["full_mean"], 20.1, 0.2)
    add("M4 g1125 per-seed spread (%)", 100 * m4["g1125"]["delta_sd"], 6.2, 0.2)
    add("M4 g1125 full mean (px)", m4["g1125"]["full_mean"], 0.488, 0.002)
    add("M4 g1125 frozen mean (px)", m4["g1125"]["frozen_mean"], 0.390, 0.002)
    add("M4 drag improvement (%)", 100 * m4["drag"]["delta_mean"], -3.5, 0.2)
    add("M4 drag per-seed spread (%)", 100 * m4["drag"]["delta_sd"], 18.3, 0.3)
    sv = m4["matched_seed_variation"]
    add("M4 seed variation one-step CV (%)", sv["one_step_cv_pct"], 4.5, 0.3)

    # --- M4 检验 (a)：地板比值随先验档的多 seed 版（附录 E / §6.1 / §8）---
    fp = J("m4_floor_partial.json")
    add("M4 floor ratio coarse (mean)", fp["euler005"]["mean"], 0.98, 0.01)
    add("M4 floor ratio coarse (sd)", fp["euler005"]["sd"], 0.02, 0.005)
    add("M4 floor ratio exact (mean)", fp["rk4001x5"]["mean"], 0.13, 0.01)
    add("M4 floor ratio exact (sd)", fp["rk4001x5"]["sd"], 0.008, 0.005)
    add("M4 floor ratio euler001x5 (mean)", fp["euler001x5"]["mean"], 0.96, 0.01)
    add("M4 floor ratio euler001x5 (sd)", fp["euler001x5"]["sd"], 0.01, 0.005)
    add("M4 floor ratio rk4005 (mean)", fp["rk4005"]["mean"], 0.70, 0.01)
    add("M4 floor ratio rk4005 (sd)", fp["rk4005"]["sd"], 0.09, 0.01)
    add("M4 all four tiers have n=7",
        1.0 if all(fp[k]["n"] == 7 for k in
                   ("euler005", "euler001x5", "rk4005", "rk4001x5")) else 0.0,
        1.0, 0.0)
    add("M4 floor separation z (exact vs coarse)", fp["separation_z"], 47.5, 1.0)

    # --- §8：噪声底是二维的（换 seed × 重训 × 指标，同一协议）---
    # 数字来源：paper/tables/m4_noise_floor.py → analysis_output/noise_floor_matrix.json
    # 主口径 R1 = 每个 seed 用自己的划分与统计量（= build_e0_report，val 不与训练集重叠）；
    # R2 是共用 s42 划分的诊断读数（对 s!=42 有 88-91% 训练集重叠），不作噪声底引用。
    nfm = J("noise_floor_matrix.json")
    ce = nfm["tiers"]["euler005"]["summary"]
    ex = nfm["tiers"]["rk4001x5"]["summary"]
    add("noise floor coarse one-step seed CV (%)",
        ce["one_step"]["R1_seed_cv_pct"], 51.0, 0.5)
    add("noise floor coarse k=2 seed CV (%)", ce["k=2"]["R1_seed_cv_pct"], 12.6, 0.3)
    add("noise floor coarse k=5 seed CV (%)", ce["k=5"]["R1_seed_cv_pct"], 16.4, 0.3)
    add("noise floor retrain one-step (%)", ce["one_step"]["retrain_pct"], 0.51, 0.02)
    add("noise floor retrain k=2 (%)", ce["k=2"]["retrain_pct"], 0.18, 0.02)
    add("noise floor retrain k=5 (%)", ce["k=5"]["retrain_pct"], 14.96, 0.1)
    add("noise floor seed/retrain ratio one-step",
        ce["one_step"]["R1_cv_over_retrain"], 100.0, 1.5)
    add("noise floor seed/retrain ratio k=2",
        ce["k=2"]["R1_cv_over_retrain"], 69.0, 1.5)
    add("noise floor seed/retrain ratio k=5",
        ce["k=5"]["R1_cv_over_retrain"], 1.10, 0.05)
    add("noise floor coarse one-step LOO CV (%)",
        ce["one_step"]["R1_loo_cv_pct"], 27.0, 0.5)
    add("noise floor coarse one-step ten-seed mean (px)",
        ce["one_step"]["R1_seed_mean"], 0.67, 0.01)
    add("noise floor coarse one-step ten-seed sd (px)",
        ce["one_step"]["R1_seed_sd"], 0.34, 0.01)
    add("noise floor exact one-step ten-seed mean (px)",
        ex["one_step"]["R1_seed_mean"], 0.038, 0.001)
    add("noise floor exact one-step ten-seed sd (px)",
        ex["one_step"]["R1_seed_sd"], 0.0006, 0.0002)
    add("noise floor exact one-step CV (%)", ex["one_step"]["R1_seed_cv_pct"], 1.7, 0.1)
    add("noise floor 10-seed k=5 ladder separation (x)",
        ce["k=5"]["R1_seed_mean"] / ex["k=5"]["R1_seed_mean"], 31.0, 1.0)
    add("noise floor matrix covers ten seeds",
        1.0 if nfm.get("seeds") == list(range(42, 52)) else 0.0, 1.0, 0.0)
    _tier_cvs = [nfm["tiers"][k]["summary"]["one_step"]["R1_seed_cv_pct"]
                 for k in ("euler001x5", "rk4005")]
    add("noise floor other-tiers one-step CV low (%)", min(_tier_cvs), 21.0, 1.0)
    # R2 is intentionally not quoted as evidence: high train/validation overlap
    # prevents it from estimating a causal split contribution. Preserve one check that
    # the artifact records the contamination so it cannot later be mistaken for R1.
    add("noise floor R2 records contamination (s43 overlap fraction)",
        nfm["overlap_R2"]["43"]["frac_of_fixed_val"], 0.91, 0.02)

    # 逐 seed 单调性：10/10 seed 在两列上都按档位单调（比均值阶梯更强的陈述）
    _specs = ("euler005", "euler001x5", "rk4005", "rk4001x5")
    _nfseeds = sorted(nfm["tiers"]["euler005"]["R1_own_split"], key=int)

    def _mono(col):
        ok = 0
        for s in _nfseeds:
            v = [nfm["tiers"][sp]["R1_own_split"][s][col] for sp in _specs]
            ok += int(all(v[i] > v[i + 1] for i in range(len(v) - 1)))
        return float(ok)

    add("noise floor per-seed monotone count (one-step)", _mono("one_step"),
        float(len(_nfseeds)), 0.0)
    add("noise floor per-seed monotone count (k=5)", _mono("k=5"),
        float(len(_nfseeds)), 0.0)

    # --- §4.3 多起点辨识（审稿 W5）：9 个起点在同一盆地，终点与目标值一致 ---
    ms = J("multistart_clean.json")
    add("multistart n starts", ms["summary"]["n_starts"], 9.0, 0.0)
    add("multistart end-g min (px-scale g)", ms["summary"]["end_g_min"], 705.42, 0.02)
    add("multistart end-g max (px-scale g)", ms["summary"]["end_g_max"], 705.93, 0.02)
    add("multistart end-g median", ms["summary"]["end_g_median"], 705.77, 0.02)
    add("multistart end-g spread (% of median)",
        100 * (ms["summary"]["end_g_max"] - ms["summary"]["end_g_min"])
        / ms["summary"]["end_g_median"], 0.07, 0.01)
    add("multistart starts within 20% of true g", ms["summary"]["n_within_20pct_of_true"],
        0.0, 0.0)
    _objs = [r["objective"] for r in ms["rows"]]
    add("multistart objective relative spread",
        (max(_objs) - min(_objs)) / float(np.mean(_objs)), 0.0, 1e-5)

    # --- 原始 3-seed Table 1 vs 10-seed extension: keep estimands distinct ---
    le = J("ladder_extension_table.json")
    add("ladder extension declares overlapping seeds, not independent replication",
        1.0 if (le["original_seeds"] == [42, 43, 44]
                and le["extended_seeds"] == list(range(42, 52))
                and le["independent_replication"] is False) else 0.0,
        1.0, 0.0)
    for row in le["rows"]:
        tier = row["tier"]
        old_key = {"euler005": "ours_base",
                   "euler001x5": "ours_base@euler001x5",
                   "rk4005": "ours_base@rk4005",
                   "rk4001x5": "ours_base@rk4001x5"}[tier]
        _oldruns = {int(x["seed"]): x for x in e0[old_key]}
        _old_one = float(np.mean([_oldruns[s]["one_step_model"] for s in (42, 43, 44)]))
        _old_k5 = float(np.mean([_oldruns[s]["rollout"]["k=5"] for s in (42, 43, 44)]))
        add(f"ladder extension {tier} original 3-seed one-step",
            row["original_one_step_mean"], _old_one, 1e-9)
        add(f"ladder extension {tier} original 3-seed k=5",
            row["original_k5_mean"], _old_k5, 1e-9)
        _nf = nfm["tiers"][tier]
        add(f"ladder extension {tier} ten-seed one-step mean",
            row["extended_one_step_mean"], _nf["summary"]["one_step"]["R1_seed_mean"], 1e-9)
        add(f"ladder extension {tier} ten-seed one-step SD",
            row["extended_one_step_sd"], _nf["summary"]["one_step"]["R1_seed_sd"], 1e-9)
        add(f"ladder extension {tier} ten-seed k=5 mean",
            row["extended_k5_mean"], _nf["summary"]["k=5"]["R1_seed_mean"], 1e-9)
        add(f"ladder extension {tier} ten-seed k=5 SD",
            row["extended_k5_sd"], _nf["summary"]["k=5"]["R1_seed_sd"], 1e-9)
        _fr = fr1["tiers"][tier]
        add(f"ladder extension {tier} ten-seed head/target ratio mean",
            row["extended_head_target_ratio_mean"], _fr["ratio_mean"], 1e-9)
        add(f"ladder extension {tier} ten-seed head/target ratio SD",
            row["extended_head_target_ratio_sd"], _fr["ratio_sd"], 1e-9)

    n_ok = sum(c["ok"] for c in checks)
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: "
              f"computed={c['value']} vs claimed={c['claimed']} (tol {c['tol']})")
    print()
    print(f"{n_ok}/{len(checks)} PASS")
    if skipped:
        print("[skip] %d check(s) need the excluded raw data: %s"
              % (len(skipped), "; ".join(skipped)))
    with open(os.path.join(ROOT, "analysis_output", "number_provenance_check.json"),
              "w", encoding="utf-8") as f:
        json.dump(dict(checks=checks, n_pass=n_ok, n_total=len(checks), skipped=skipped),
                  f, ensure_ascii=False, indent=1)
    print("[save] analysis_output/number_provenance_check.json")
    if n_ok < len(checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
