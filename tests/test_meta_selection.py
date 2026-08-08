from unittest.mock import patch

from core.analysis.momentum import (
    _score_meta,
    _select_meta_candidate,
    rebuild_entry_plan_from_snapshot,
)


def test_meta_selection_prefers_true_confluence_over_single_indicator_zone():
    single = {
        "low": 99.8,
        "high": 100.2,
        "family": "moving_average",
        "source": "SMA20",
    }
    confluence = {
        "low": 99.4,
        "high": 100.4,
        "is_confluence": True,
        "families": ["moving_average", "support_zone"],
        "source": "SMA20 + support",
    }

    assert _select_meta_candidate([single, confluence], 100.0) is confluence


def test_meta_selection_prefers_narrower_true_confluence():
    broad = {
        "low": 98.8,
        "high": 100.8,
        "is_confluence": True,
        "families": ["moving_average", "support_zone"],
        "source": "broad",
    }
    narrow = {
        "low": 99.6,
        "high": 100.1,
        "is_confluence": True,
        "families": ["moving_average", "trendline"],
        "source": "narrow",
    }

    assert _select_meta_candidate([broad, narrow], 100.0) is narrow


def test_meta_selection_does_not_turn_a_single_indicator_zone_into_meta():
    nearby = {"low": 99.5, "high": 100.0, "family": "moving_average"}
    farther = {"low": 98.0, "high": 98.2, "family": "support_zone"}

    assert _select_meta_candidate([farther, nearby], 100.2) is None


def test_meta_selection_rejects_an_overly_wide_confluence():
    broad = {
        "low": 96.0,
        "high": 104.0,
        "is_confluence": True,
        "families": ["moving_average", "support_zone"],
    }

    assert _select_meta_candidate([broad], 100.0) is None


def test_score_meta_clears_stale_selected_zone_when_no_valid_confluence():
    broad = {
        "low": 96.0,
        "high": 104.0,
        "family": "compact_base",
        "families": ["compact_base", "moving_average"],
        "is_confluence": True,
    }
    evidence = {"meta_candidates": [broad], "selected_meta_zone": broad.copy()}
    latest = {
        "close": 100.0,
        "open": 99.0,
        "MACDh": 0.2,
        "dry_up_ratio": 0.8,
        "range_contraction": True,
    }
    previous = {"MACDh": 0.1}

    _score_meta(latest, previous, 95.0, evidence, 0.04)

    assert "selected_meta_zone" not in evidence


def test_snapshot_rebuild_drops_stale_broad_meta_zone():
    row = {
        "setup_phase": "pullback_forming",
        "setup_evidence": {
            "chart_evidence": {
                "selected_meta_zone": {"low": 96.0, "high": 104.0},
            },
        },
        "open": [99.0, 100.0],
        "high": [101.0, 101.5],
        "low": [98.0, 99.5],
        "close": [100.0, 101.0],
        "volume": [1_000_000, 900_000],
        "ATR": [2.0, 2.0],
        "SMA_10": [99.0, 99.5],
        "SMA_20": [98.0, 98.5],
        "SMA_50": [95.0, 95.5],
        "pivot_high_20": [102.0, 102.0],
        "pivot_high_50": [103.0, 103.0],
        "range_contraction": [False, False],
    }
    broad = {
        "low": 96.0,
        "high": 104.0,
        "is_confluence": True,
        "families": ["moving_average", "support_zone"],
    }

    with patch(
        "core.analysis.momentum.build_chart_evidence",
        return_value={"meta_candidates": [broad]},
    ):
        rebuilt = rebuild_entry_plan_from_snapshot(row)

    assert rebuilt["setup_phase"] == "unclear_structure"
    assert rebuilt["entry_zone_low"] is None
    assert rebuilt["entry_zone_high"] is None
    assert "selected_meta_zone" not in rebuilt["setup_evidence"]["chart_evidence"]


def test_unclear_snapshot_rebuild_refreshes_stale_meta_score_and_flags():
    row = {
        "setup_phase": "unclear_structure",
        "meta_score": 58,
        "setup_evidence": {
            "raw_risk_pct": 0.04,
            "meta_score": 58,
            "candidate_flags": {
                "pullback_entry": False,
                "pullback_forming": True,
            },
        },
        "open": [99.0, 100.0],
        "high": [101.0, 101.5],
        "low": [98.0, 99.5],
        "close": [100.0, 101.0],
        "volume": [1_000_000, 900_000],
        "ATR": [2.0, 2.0],
        "SMA_10": [99.0, 99.5],
        "SMA_20": [98.0, 98.5],
        "SMA_50": [95.0, 95.5],
        "pivot_high_20": [102.0, 102.0],
        "pivot_high_50": [103.0, 103.0],
        "range_contraction": [False, False],
        "dry_up_ratio": [0.8, 0.8],
        "MACDh": [0.1, 0.2],
        "bullish_engulfing": [False, False],
        "reversal_180": [False, False],
    }
    broad = {
        "low": 96.0,
        "high": 104.0,
        "is_confluence": True,
        "families": ["moving_average", "support_zone"],
    }

    with patch(
        "core.analysis.momentum.build_chart_evidence",
        return_value={"meta_candidates": [broad]},
    ):
        rebuilt = rebuild_entry_plan_from_snapshot(row)

    evidence = rebuilt["setup_evidence"]
    assert rebuilt["setup_phase"] == "unclear_structure"
    assert rebuilt["meta_score"] != 58
    assert evidence["meta_score_detail"]["distance_to_meta"][0]["actual"] is None
    assert evidence["candidate_flags"]["pullback_entry"] is False
    assert evidence["candidate_flags"]["pullback_forming"] is False
    assert evidence["candidate_flag_details"]["pullback_entry"]["has_meta_zone"] is False
    assert evidence["candidate_flag_details"]["pullback_forming"]["has_meta_zone"] is False
