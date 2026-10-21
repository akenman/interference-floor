# -*- coding: utf-8 -*-
"""Tests for the per-seed own-held-out-split floor-ratio sensitivity analysis."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_own_split_floor_ratios_are_descriptive_and_source_bound():
    p = os.path.join(ROOT, "analysis_output", "ladder_floor_r1.json")
    d = json.load(open(p, encoding="utf-8"))
    assert "descriptive_only" not in d or d["descriptive_only"] is True
    assert d["standardized_gap_is_inferential"] is False
    assert "not independent replications" in d["split_warning"]
    expected = {
        "euler005": (0.968, 0.043),
        "euler001x5": (0.953, 0.024),
        "rk4005": (0.792, 0.070),
        "rk4001x5": (0.126, 0.008),
    }
    for tier, (mean, sd) in expected.items():
        x = d["tiers"][tier]
        assert x["n"] == 10
        assert x["descriptive_only"] is True
        assert abs(x["ratio_mean"] - mean) < 0.002
        assert abs(x["ratio_sd"] - sd) < 0.002
        assert set(x["per_seed"]) == {str(s) for s in range(42, 52)}
