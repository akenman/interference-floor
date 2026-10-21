# -*- coding: utf-8 -*-
"""E0 数字单一来源守卫（防止再次出现「文档与 JSON 各持一套 Table 1」）。

背景：2026-10-01 审计发现 FINAL_REPORT.md §4.1 与 analysis_output/E0_report.json
各持一套主表（noprior 0.252/0.607 vs 0.265/0.978），且 rollout 图曾硬编码 3 臂，
恰好排除闭环最好的 strong_mlp/jepa。本测试把以下三件事变成可执行约束：

  1. E0_report.json 的结构完整性（臂数、rollout 列、无先验臂的 prior 必为空）；
  2. FINAL_REPORT.md §4.1 表里的每个数字必须等于 JSON 的均值（3 位小数容差）；
  3. rollout 图必须覆盖 JSON 中全部有 rollout 数据的臂（防选择性出图）。

运行：python -m pytest tests/test_e0_single_source.py -q
"""
import json
import math
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, "analysis_output", "E0_report.json")
FINAL = os.path.join(ROOT, "FINAL_REPORT.md")
ROLLOUT_FIG = os.path.join(ROOT, "figures", "E0_rollout_curves.png")

ANALYTIC_ARMS = {"zero_vel", "inertia", "inertia_euler001x5",
                 "inertia_rk4005", "inertia_rk4001x5"}   # 0 参数，n=1
NO_PRIOR_ARMS = {"noprior", "jepa", "strong_mlp"}  # physics_prior=False
TOL = 0.0015                                      # 文档按 3 位小数取整


@pytest.fixture(scope="module")
def report():
    assert os.path.exists(REPORT), f"缺少主表：{REPORT}（先跑 build_e0_report.py）"
    with open(REPORT, encoding="utf-8") as f:
        return json.load(f)


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _mean(vals):
    vals = [v for v in vals if _finite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


# ---------------------------------------------------------------- 1. 结构完整性
def test_report_has_all_arms(report):
    arms = report["arms"]
    expected = {"noprior", "strong_mlp", "ours_base", "ours_gua", "ours_sigma",
                "ours_yao", "jepa"} | ANALYTIC_ARMS
    assert expected <= set(arms), f"缺臂：{expected - set(arms)}"
    for arm, runs in arms.items():
        n = 1 if arm in ANALYTIC_ARMS else 3
        assert len(runs) == n, f"{arm} 应有 {n} 个 seed，实得 {len(runs)}"
        for r in runs:
            assert _finite(r["one_step_model"]), f"{arm} one_step_model 非有限"
            for k in (2, 5):
                assert _finite(r["rollout"][f"k={k}"]), f"{arm} rollout k={k} 非有限"


def test_no_prior_arms_have_empty_prior(report):
    """StrongMLP 的零占位 prior_norm 曾被当成物理先验 → 先验列写成 87.9px 伪影。"""
    for arm, runs in report["arms"].items():
        for r in runs:
            p = r.get("one_step_prior")
            if arm in NO_PRIOR_ARMS:
                assert not _finite(p), f"{arm} 无物理先验，prior 却为 {p}"
            else:
                assert _finite(p), f"{arm} 有物理先验，prior 却为 {p}"


# ---------------------------------------------------------------- 2. 文档 = JSON
def _final_report_rows():
    """解析 FINAL_REPORT §4.1 表，返回 [(arm, {列名: 单元格})]。
    按表头定位列（而非写死列号）——本次新增 spec / rollout 列即靠此稳健解析。"""
    txt = open(FINAL, encoding="utf-8").read()
    m = re.search(r"### 4\.1 主表.*?\n(.*?)\n### 4\.2", txt, re.S)
    assert m, "FINAL_REPORT.md 里找不到 §4.1 主表（标题或分隔被改动了？）"
    keys = None
    rows = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("**", "") for c in line.strip("|").split("|")]
        if keys is None:                       # 表头行
            keys = cells
            continue
        if set("".join(cells)) <= set("-: "):  # 分隔行
            continue
        if len(cells) != len(keys):
            continue
        d = dict(zip(keys, cells))
        d["_arm_name"] = re.split(r"[（(]", d.get("臂", ""))[0].strip()
        rows.append(d)
    assert rows, "§4.1 表体未解析到任何数据行"
    return rows


def _col(d, *names):
    for n in names:
        for k in d:
            if k.startswith(n):
                return d[k]
    raise KeyError(f"表里找不到列 {names}（现有列：{list(d)}）")


def _num(cell):
    cell = cell.strip()
    if cell in ("—", "-", ""):
        return float("nan")
    return float(cell)


def _num_tol(cell):
    """按文档里显示的小数位数决定容差（写 0.12 就该按 ±0.005 判，而不是 ±0.0015）。"""
    cell = cell.strip().replace("**", "")
    if cell in ("—", "-", ""):
        return float("nan"), 1e-9
    val = float(cell)
    dec = len(cell.split(".")[1]) if "." in cell else 0
    return val, 0.5 * 10 ** (-dec) + 1e-9


def test_final_report_table_matches_json(report):
    arms = report["arms"]
    seen = set()
    for d in _final_report_rows():
        name = d["_arm_name"]
        assert name in arms, f"FINAL_REPORT 出现未知臂 {name}"
        runs = arms[name]
        n = int(_col(d, "n"))
        one, tol_one = _num_tol(_col(d, "one-step"))
        std, tol_std = _num_tol(_col(d, "std"))
        r2, tol_r2 = _num_tol(_col(d, "roll k=2", "rollout\nk=2", "rollout k=2"))
        r5, tol_r5 = _num_tol(_col(d, "roll k=5", "rollout\nk=5", "rollout k=5"))
        seen.add(name)
        assert n == len(runs), f"{name}: 文档 n={n}，JSON n={len(runs)}"
        # spec 列（若表里有）必须与 JSON 一致；无物理先验的臂两边都记空
        spec_cells = [k for k in d if k.startswith("spec")]
        if spec_cells:
            from physics_prior_spec import SPEC_LABELS
            specs = {r.get("prior_spec") for r in runs}
            assert len(specs) == 1, f"{name}: JSON 里 spec 不一致 {specs}"
            js = specs.pop()
            doc_spec = d[spec_cells[0]]
            if js is None:
                assert doc_spec in ("—", "-", ""), \
                    f"{name}: JSON 无 spec，文档却写 {doc_spec}"
            else:
                assert doc_spec in (js, SPEC_LABELS.get(js)), \
                    f"{name}: 文档 spec={doc_spec} ∉ {{{js}, {SPEC_LABELS.get(js)}}}"
        checks = [("one-step", one, tol_one, None),
                  ("roll k=2", r2, tol_r2, "k=2"),
                  ("roll k=5", r5, tol_r5, "k=5")]
        for label, doc, tol, rk in checks:
            got = _mean([r["one_step_model"] for r in runs]) if rk is None \
                else _mean([r["rollout"][rk] for r in runs])
            assert abs(doc - got) <= tol, \
                f"{name} {label}: 文档 {doc} ≠ JSON {got:.4f}（容差 {tol}）"
        got_std = float(__import__("numpy").std([r["one_step_model"] for r in runs]))
        assert abs(std - got_std) <= tol_std, \
            f"{name} std: 文档 {std} ≠ JSON {got_std:.4f}（容差 {tol_std}）"
        # 先验列
        doc_prior, tol_prior = _num_tol(_col(d, "先验"))
        got_prior = _mean([r.get("one_step_prior") for r in runs])
        if name in NO_PRIOR_ARMS:
            assert not _finite(doc_prior), f"{name} 无先验，文档却写 {doc_prior}"
        else:
            assert abs(doc_prior - got_prior) <= tol_prior, \
                f"{name} 先验: 文档 {doc_prior} ≠ JSON {got_prior:.4f}（容差 {tol_prior}）"
    missing = set(arms) - seen
    assert not missing, f"FINAL_REPORT §4.1 漏了臂：{missing}"


# ---------------------------------------------------------------- 3. 全臂出图
def test_rollout_figure_not_cherry_picked():
    """rollout 图曾硬编码 (noprior, ours_base, ours_gua)，排除 strong_mlp/jepa。"""
    assert os.path.exists(ROLLOUT_FIG), "缺少 rollout 图（先跑 build_e0_figures.py）"
    src = open(os.path.join(ROOT, "build_e0_figures.py"), encoding="utf-8").read()
    m = re.search(r"def fig_rollout\(arms\):(.*?)\n# -", src, re.S)
    assert m, "找不到 fig_rollout 实现"
    body = re.sub(r'""".*?"""', "", m.group(1), flags=re.S)   # 去掉 docstring 的叙述
    for arm in ("noprior", "ours_base", "ours_gua", "strong_mlp", "jepa"):
        assert not re.search(r'\(\s*"%s"' % arm, body), \
            f"fig_rollout 仍在硬编码臂列表（出现 {arm}）—— 必须遍历全部臂"
    assert "for a in sorted(arms)" in body or "for a in arms" in body, \
        "fig_rollout 应遍历全部臂"
