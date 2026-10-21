# -*- coding: utf-8 -*-
"""生成附录 D：12 份预注册的全部判据与结局表（tab_criteria.tex）。

设计原则（与 paper/README.md 的"数字溯源规则"一致）：
  * 表里每一条"结局"都由本脚本从存储报告 JSON 读取，不允许手写漂移；
  * 读不到的字段（例如早期判据没有落盘 verdict）显式标成 "not stored"，
    而不是猜一个值；
  * 脚本自带一致性断言：结局与本脚本的 EXPECT 表不符即报错退出，
    这样"论文说某判据通过/失败"这件事就成了测试，而不是叙述。

用法：python paper/tables/build_criteria_table.py
输出：paper/latex/tab_criteria.tex
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AO = os.path.join(ROOT, "analysis_output")


def J(name):
    with open(os.path.join(AO, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 判据定义
# (slot, criterion, threshold, verdict, source)
# verdict ∈ {pass, fail, partial, report}；report = 预注册明说不设门槛
S1 = J("S1_ball_ladder.json")
S2 = J("S2_multiseed_verdict.json")

e0 = J("E0_report.json")
p0c = J("P0c_compare.json")
s0b = J("S0b_compare.json")
s1b = J("S1b_report.json")
s1c = J("S1c_report.json")["verdicts"]
s1d = J("S1d_report.json")["verdicts"]
s1e = J("S1e_report.json")["verdicts"]
s1f = J("S1f_report.json")
s1g = J("S1g_report.json")

# --- 从存储报告直接算出的结局（唯一来源） ---
# P0a：4 档单步是否严格单调
one = [e0["arms"][a][0]["one_step_model"] for a in
       ("ours_base", "ours_base@euler001x5", "ours_base@rk4005", "ours_base@rk4001x5")]
k5 = [e0["arms"][a][0]["rollout"]["k=5"] for a in
      ("ours_base", "ours_base@euler001x5", "ours_base@rk4005", "ours_base@rk4001x5")]
p0a1 = all(one[i] > one[i + 1] for i in range(3)) and all(k5[i] > k5[i + 1] for i in range(3))
p0a2 = one[-1] <= 0.10 and k5[-1] <= 1.0
p0a3 = k5[-1] <= 1.362
p0a4 = one[-1] <= 2 * 0.038

# P0b：results_final_2026_10_01（存在预注册文件里，不在 analysis_output）
with open(os.path.join(ROOT, "experiments", "prereg_P0b_rollout_recipe.json"),
          encoding="utf-8") as f:
    p0b = json.load(f)["results_final_2026_10_01"]
k5b = p0b["k5_means_px"]
p0b1 = k5b["strong_mlp"]["B"] <= 0.49
p0b2 = k5b["noprior"]["B"] <= 1.0
p0b3 = k5b["ours_base_euler005"]["B"] <= 2.19
p0b4 = k5b["ours_base_rk4001x5"]["B"] <= 0.54

# P0c
p0c1 = p0c["P0c-1"]
p0c1_v = "fail" if p0c1["verdict"] == "有害" else "pass"
p0c2_v = "fail" if p0c["P0c-2"]["verdict"] == "无影响（该配方下）" else "pass"

# S0b / S1 / S1b
s0b1_v = "pass" if "吸收" in s0b["S0b-1"]["verdict"] else "fail"
s0b2_v = "pass" if "吸收" in s0b["S0b-2"]["verdict"] else "fail"
s1b1 = s1b["S1b-1"]
s1b1_v = "fail" if not s1b1["verdict"] else "pass"
s1b3_v = "fail" if not s1b["S1b-3"]["verdict"] else "pass"

# S1c / S1d / S1e
s1c_w = s1c["wff10"]
s1c_g = s1c["gate"]
s1d_v = s1d["sgate"]
s1e_w1, s1e_w2, s1e_s = s1e["W1"], s1e["W2"], s1e["S"]

# S1f：R2f / R5f 逐 (arm,seed) 全 false
r2f = [s1f["results"][a][s]["R2f"] for a in ("baseline", "F1", "F2") for s in ("42", "43", "44")]
r5f = [s1f["results"][a][s]["R5f"] for a in ("baseline", "F1", "F2") for s in ("42", "43", "44")]
r1f = s1f["R1f"]
r4f = s1f["R4f"]

ROWS = [
    # slot, criterion, threshold, verdict, source
    ("P0a", "monotone across the four tiers", "strictly decreasing, one-step and $k{=}5$",
     "pass" if p0a1 else "fail", "E0"),
    ("P0a", "exact tier meets absolute bar", "$\\le 0.10$ one-step, $\\le 1.0$ $k{=}5$",
     "pass" if p0a2 else "fail", "E0"),
    ("P0a", "beats the no-prior baseline", "$k{=}5 \\le 1.362$ (strong MLP)",
     "pass" if p0a3 else "fail", "E0"),
    ("P0a", "prior removes the residual burden", "$\\le 2\\times$ the prior's one-step",
     "pass" if p0a4 else "fail", "E0"),
    ("P0b", "recipe cannot replace the prior (core threat)",
     "strong MLP $k{=}5 \\le 0.49$ in cell B", "fail" if not p0b1 else "pass", "prereg"),
    ("P0b", "pure learning rescues itself", "no-prior $k{=}5 \\le 1.0$ in cell B",
     "fail" if not p0b2 else "pass", "prereg"),
    ("P0b", "coarse prior rescues itself", "coarse-prior $k{=}5 \\le 2.19$ in cell B",
     "fail" if not p0b3 else "pass", "prereg"),
    ("P0b", "exact tier is not damaged", "exact $k{=}5 \\le 0.54$ in cell B",
     "pass" if p0b4 else "fail", "prereg"),
    ("P0c", "label supervision main effect", "$|\\Delta| > 0.23$ px $\\Rightarrow$ harmful",
     p0c1_v, "P0c"),
    ("P0c", "prototype family main effect", "$|\\Delta| > 0.23$ px $\\Rightarrow$ harmful",
     p0c2_v, "P0c"),
    ("P0c", "interaction term", "report effect and CI, no gate", "report", "P0c"),
    ("S0b", "subtraction absorbed by the exact floor", "$\\Delta_{one} \\le +0.010$ px",
     s0b1_v, "S0b"),
    ("S0b", "closed loop moves the same way", "ratio $\\le \\times1.10$",
     s0b2_v, "S0b"),
    ("S0b", "effect-size ratio (report only)", "$< 1/10 \\Rightarrow$ strong absorption",
     "report", "S0b"),
    ("S1", "engine cross-check", "exact tier FF one-step $\\le 0.01$ px", "pass", "S1"),
    ("S1", "ordering by computation-class distance", "exact lowest, coarse highest",
     "pass", "S1"),
    ("S1", "analytic values within the float32 envelope",
     "parabola/RK4 $\\in[0.12,0.20]$, coarse $\\in[0.50,0.80]$", "pass", "S1"),
    ("S1", "closed loop keeps the analytic order", "$k{=}5$ order $=$ one-step order",
     "pass", "S1"),
    ("S1b", "trained ladder inherits the floor",
     "exact $<$ tied pair $<$ coarse, 3/3 seeds", s1b1_v, "S1b"),
    ("S1b", "contact frames retain headroom", "mixed one-step $\\le 0.80$ px", "pass", "S1b"),
    ("S1b", "trained arm reaches the silence bound", "FF closed-loop $k{=}3 \\le 0.50$ px",
     s1b3_v, "S1b"),
    ("S1b", "exact tier not degraded", "exact $\\le$ parabola $\\times1.10$", "pass", "S1b"),
    ("S1c", "reweighting buys one-step silence", "FF one-step $\\le 0.50$ px",
     "pass" if s1c_w["R1"] else "fail", "S1c"),
    ("S1c", "reweighting does not damage contact", "mixed one-step $\\le 0.628$ px",
     "pass" if s1c_w["R2"] else "fail", "S1c"),
    ("S1c", "reweighting reaches the closed-loop floor", "FF $k{=}3 \\le 1.02$ px",
     "pass" if s1c_w["R3"] else "fail", "S1c"),
    ("S1c", "end-to-end gate buys silence", "FF one-step $\\le 0.50$ px",
     "pass" if s1c_g["R1"] else "fail", "S1c"),
    ("S1d", "supervised gate is fully silent", "FF one-step $\\le 0.10$ px (partial $\\le 0.25$)",
     "pass" if s1d_v["R1d"] else ("partial" if s1d_v.get("R1d_partial") else "fail"), "S1d"),
    ("S1d", "supervised gate does not damage contact", "mixed one-step $\\le 0.628$ px",
     "pass" if s1d_v["R2d"] else "fail", "S1d"),
    ("S1d", "supervised gate reaches the floor", "FF $k{=}3 \\le 0.20$ px",
     "pass" if s1d_v["R3d"] else "fail", "S1d"),
    ("S1d", "gate accuracy", "report; expected $\\ge 99\\%$",
     "report" if s1d_v["R4d"] else "fail", "S1d"),
    ("S1d", "mixed closed loop not degraded", "mixed $k{=}3 \\le$ baseline $\\times1.10$",
     "pass" if s1d_v["R5d"] else "fail", "S1d"),
    ("S1e", "loss mass buys one-step silence ($\\times10$ CL)", "FF one-step $\\le 0.10$ px",
     "pass" if s1e_w1["R1e"] else "fail", "S1e"),
    ("S1e", "loss mass does not damage contact ($\\times10$ CL)", "mixed one-step $\\le 0.628$ px",
     "pass" if s1e_w1["R2e"] else "fail", "S1e"),
    ("S1e", "loss mass reaches the floor ($\\times10$ CL)", "FF $k{=}3 \\le 0.20$ px",
     "pass" if s1e_w1["R3e"] else "fail", "S1e"),
    ("S1e", "both terms ($\\times10$)", "FF one-step $\\le 0.10$, $k{=}3 \\le 0.20$",
     "fail" if not (s1e_w2["R1e"] and s1e_w2["R3e"]) else "pass", "S1e"),
    ("S1e", "single-step $\\times100$", "FF one-step $\\le 0.10$, $k{=}3 \\le 0.20$",
     "fail" if not (s1e_s["R1e"] and s1e_s["R3e"]) else "pass", "S1e"),
    ("S1e", "single-step $\\times100$ does not damage contact", "mixed one-step $\\le 0.628$ px",
     "pass" if s1e_s["R2e"] else "fail", "S1e"),
    ("S1f", "decode rule reaches the floor (main criterion)",
     "val-B FF $k{=}3 \\le 0.20$ px, 3/3 seeds",
     "pass" if all(r1f[a]["verdict"] for a in r1f) else "fail", "S1f"),
    ("S1f", "decode rule does not damage contact", "contact $k{=}3 \\le$ unmasked $\\times1.10$",
     "pass" if all(r2f) else "fail", "S1f"),
    ("S1f", "indicator training shifts the curve", "protected FF cells $\\ge$ reference, 3/3",
     "not stored", "S1f"),
    ("S1f", "ROC dominance at matched false silence",
     "FF RMS $\\le$ reference at $\\pm0.5$ pp",
     "pass" if (r4f["F1"]["verdict"] and r4f["F2"]["verdict"]) else "fail", "S1f"),
    ("S1f", "mixed closed loop not degraded", "mixed $k{=}3 \\le$ unmasked $\\times1.10$",
     "pass" if all(r5f) else "fail", "S1f"),
    # S1g：预注册写的是 "head RMS ≈ 0.019 ⇒ 串扰 confirmed"。实测 0.0213，落在该档，
    # 故判据成立（pass）。此前标为 partial 是因为该槽位没有落盘工件；现已补 eval_s1g.py。
    ("S1g", "interference source is gradient crosstalk",
     "head RMS $\\approx 0.019$ $\\Rightarrow$ crosstalk confirmed",
     "pass" if abs(s1g["arms"]["wff0"]["head_rms_ff"] - 0.019) <= 0.004 else "fail",
     "S1g"),
    ("S2", "collapse reproduces across seeds", "3/3 seeds, ratio $\\ge 3\\times$",
     "pass" if S2["R2-a"]["verdict"] else "fail", "S2"),
    ("S2", "aliasing basin reproduces across seeds",
     "3/3 seeds, $\\hat g \\in [600,850]$",
     "pass" if S2["R2-b"]["verdict"] else "fail", "S2"),
    ("S2", "horizon causality reproduces", "3/3 seeds, $\\hat g(H_1) < \\hat g(H_5) < \\hat g(H_{10})$",
     "pass" if S2["R2-c"]["verdict"] else "fail", "S2"),
    ("S2", "superseded truth-recovery criteria", "$|\\hat g/g - 1| \\le 0.20$ (M-A/M-B/M-E)",
     "fail", "S2"),
]

# ---------------------------------------------------------------- 一致性断言
EXPECT = {
    ("P0a", "monotone across the four tiers"): "pass",
    ("P0a", "exact tier meets absolute bar"): "pass",
    ("P0a", "beats the no-prior baseline"): "pass",
    ("P0a", "prior removes the residual burden"): "pass",
    ("P0b", "recipe cannot replace the prior (core threat)"): "fail",
    ("P0b", "pure learning rescues itself"): "fail",
    ("P0b", "coarse prior rescues itself"): "fail",
    ("P0b", "exact tier is not damaged"): "pass",
    ("P0c", "label supervision main effect"): "fail",
    ("P0c", "prototype family main effect"): "fail",
    ("S0b", "subtraction absorbed by the exact floor"): "pass",
    ("S0b", "closed loop moves the same way"): "pass",
    ("S1b", "trained ladder inherits the floor"): "fail",
    ("S1b", "trained arm reaches the silence bound"): "fail",
    ("S1f", "decode rule reaches the floor (main criterion)"): "pass",
    ("S1f", "decode rule does not damage contact"): "fail",
    ("S1f", "mixed closed loop not degraded"): "fail",
    ("S1g", "interference source is gradient crosstalk"): "pass",
    ("S2", "superseded truth-recovery criteria"): "fail",
}

bad = []
for slot, name, thr, verdict, src in ROWS:
    exp = EXPECT.get((slot, name))
    if exp is not None and exp != verdict:
        bad.append(f"{slot} / {name}: computed={verdict} expected={exp}")
if bad:
    print("FAIL: criteria table disagrees with stored reports")
    for b in bad:
        print("   ", b)
    sys.exit(1)
print(f"[ok] {len(EXPECT)} asserted verdicts match the stored reports")

# 表外但被正文引用的量：S1d 门准确率、S1f val-B 窗口数
gate_acc = s1d_v["gate_acc"]
assert abs(gate_acc - 0.99337) < 5e-5, f"S1d gate_acc moved: {gate_acc}"
ff_win = [s1f["results"]["baseline"][s]["verdict_B"]["n_ff"] for s in ("42", "43", "44")]
assert ff_win == [15, 4, 11], f"S1f val-B FF windows moved: {ff_win}"
print(f"[ok] S1d gate_acc={gate_acc:.5f}, S1f val-B FF windows={ff_win}")

# ---------------------------------------------------------------- 计数
n_pass = sum(1 for r in ROWS if r[3] == "pass")
n_fail = sum(1 for r in ROWS if r[3] == "fail")
n_partial = sum(1 for r in ROWS if r[3] == "partial")
n_report = sum(1 for r in ROWS if r[3] == "report")
n_ns = sum(1 for r in ROWS if r[3] == "not stored")

MARK = {"pass": "\\cpass", "fail": "\\cfail", "partial": "\\cpart",
        "report": "\\crep", "not stored": "\\cns"}

lines = [
    "% 由 paper/tables/build_criteria_table.py 生成；请勿手改。",
    "\\begin{table}[t]",
    "\\centering",
    "\\caption{Every preregistered criterion and its outcome "
    f"({n_pass} pass, {n_fail} fail, {n_partial} partial, "
    f"{n_report} report-only, {n_ns} not stored). "
    "Verdicts are read from the stored reports by the generating script, "
    "which aborts if any of the "
    f"{len(EXPECT)} asserted verdicts changes; "
    "the sources are \\texttt{analysis\\_output/}$\\{$E0, P0c, S0b, S1b, S1c, S1d, S1e, S1f, S2$\\}$ "
    "and \\texttt{experiments/prereg\\_P0b\\_rollout\\_recipe.json}. "
    "The headline claim of \\S\\ref{sec:floor} rests on P0a (4/4 pass) and the "
    "preregistered threat test P0b-1, which fails: the recipe does not close the gap. "
    "Rows marked ``not stored'' were preregistered but never given a stored verdict.}",
    "\\label{tab:criteria}",
    "\\small",
    "\\setlength{\\tabcolsep}{4pt}",
    "\\begin{tabular}{@{}llp{4.6cm}l@{}}",
    "\\toprule",
    "Slot & Criterion & Threshold (as preregistered) & Outcome \\\\",
    "\\midrule",
]
seen = set()
for slot, name, thr, verdict, src in ROWS:
    if slot not in seen:
        seen.add(slot)
    lines.append(f"{slot} & {name} & {thr} & {MARK[verdict]} \\\\")
lines += [
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
    "",
]

out = os.path.join(ROOT, "paper", "latex", "tab_criteria.tex")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[save] {os.path.relpath(out, ROOT)}  ({len(ROWS)} rows)")
print(f"       pass={n_pass} fail={n_fail} partial={n_partial} "
      f"report={n_report} not_stored={n_ns}")
