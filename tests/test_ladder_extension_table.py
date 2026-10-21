# -*- coding: utf-8 -*-
"""Tests for the provenance-bound original-vs-extended prior ladder table."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ladder_extension_sources_and_values():
    p = os.path.join(ROOT, "analysis_output", "ladder_extension_table.json")
    assert os.path.exists(p), "run paper/tables/build_ladder_extension_table.py first"
    d = json.load(open(p, encoding="utf-8"))
    assert d["original_seeds"] == [42, 43, 44]
    assert d["extended_seeds"] == list(range(42, 52))
    assert d["independent_replication"] is False
    assert d["common_split_ratio_used_for_inference"] is False
    expected = {
        "euler005": (0.969, 0.665, 0.339, 14.26, 15.40, 2.53, 0.968, 0.043),
        "euler001x5": (0.196, 0.172, 0.035, 4.87, 4.98, 3.18, 0.953, 0.024),
        "rk4005": (0.136, 0.097, 0.043, 2.19, 1.68, 0.77, 0.792, 0.070),
        "rk4001x5": (0.037, 0.038, 0.001, 0.49, 0.50, 0.01, 0.126, 0.008),
    }
    for r in d["rows"]:
        one_old, one_ext, one_sd, k_old, k_ext, k_sd, ratio, ratio_sd = expected[r["tier"]]
        assert abs(r["original_one_step_mean"] - one_old) < 0.0006
        assert abs(r["extended_one_step_mean"] - one_ext) < 0.001
        assert abs(r["extended_one_step_sd"] - one_sd) < 0.0007
        assert abs(r["original_k5_mean"] - k_old) < 0.006
        assert abs(r["extended_k5_mean"] - k_ext) < 0.006
        assert abs(r["extended_k5_sd"] - k_sd) < 0.006
        assert abs(r["extended_head_target_ratio_mean"] - ratio) < 0.001
        assert abs(r["extended_head_target_ratio_sd"] - ratio_sd) < 0.001
    assert os.path.exists(os.path.join(ROOT, "paper", "latex", "tab_ladder_extension.tex"))


def test_ladder_extension_table_rebuild_is_deterministic():
    script = os.path.join(ROOT, "paper", "tables", "build_ladder_extension_table.py")
    assert os.path.exists(script)
    before_json = open(os.path.join(ROOT, "analysis_output", "ladder_extension_table.json"), "rb").read()
    before_tex = open(os.path.join(ROOT, "paper", "latex", "tab_ladder_extension.tex"), "rb").read()
    rc = subprocess.call([sys.executable, "-X", "utf8", script], cwd=ROOT)
    assert rc == 0
    after_json = open(os.path.join(ROOT, "analysis_output", "ladder_extension_table.json"), "rb").read()
    after_tex = open(os.path.join(ROOT, "paper", "latex", "tab_ladder_extension.tex"), "rb").read()
    assert before_json == after_json
    assert before_tex == after_tex
