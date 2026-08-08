from __future__ import annotations

import unittest

from core.analysis.contracts import ScreeningSettings
from core.analysis.explainability import (
    build_decision_tree,
    build_rule_checklist,
    build_score_checklist,
    deterministic_summary,
    earnings_context,
    phase_score,
    rules_passed_text,
    score_breakdown,
)


def fixture_stock(phase: str, **overrides):
    stock = {
        "symbol": "TEST",
        "setup_phase": phase,
        "setup_status": "breakout_buy",
        "price": 120.0,
        "ma50": 110.0,
        "ma200": 95.0,
        "market_cap": 4_000_000_000,
        "avgvol3m": 800_000,
        "pct_from_high": -0.08,
        "SMA_10": [116.0],
        "SMA_20": [112.0],
        "SMA_50": [110.0],
        "SMA_200": [95.0],
        "volume_ratio": [1.4],
        "dry_up_ratio": [0.7],
        "dist_pivot_20": [-0.01],
        "RSI": [62.0],
        "ATR_pct": [0.03],
        "risk_pct": 0.06,
        "vs_market_1m": 0.05,
        "vs_market_3m": 0.11,
        "vs_market_6m": 0.15,
        "ma200_is_trending_up": True,
        "breakout_score": 78,
        "meta_score": 64,
        "setup_evidence": {"failure_flags": {}, "extension_flags": {}},
        "data_date": "2026-07-10",
        "earnings_date": "2026-07-24",
    }
    stock.update(overrides)
    return stock


class ExplainabilityTests(unittest.TestCase):
    def setUp(self):
        self.settings = ScreeningSettings()

    def test_breakout_uses_breakout_score_and_summary_numbers(self):
        stock = fixture_stock(
            "fresh_breakout",
            confirmation_price=120.0,
            entry_price=120.0,
        )
        rules = build_rule_checklist(stock, self.settings)
        self.assertEqual(phase_score(stock), (78, "突破評分"))
        summary = deterministic_summary(stock)
        self.assertIn("已收市突破", summary)
        self.assertIn("門檻 1.10x", summary)
        self.assertIn("1.4x", summary)
        self.assertIn("Risk", {rule["group"] for rule in rules})
        self.assertRegex(rules_passed_text(rules), r"^\d+/\d+$")

    def test_breakout_summary_explains_confirmation_and_chase_ceiling(self):
        stock = fixture_stock(
            "fresh_breakout",
            confirmation_price=100.0,
            entry_price=100.0,
        )
        summary = deterministic_summary(stock)
        self.assertIn("突破 $100.00", summary)
        self.assertIn("追價上限 $101.00", summary)

    def test_near_breakout_summary_explains_pending_volume_confirmation(self):
        stock = fixture_stock(
            "near_breakout",
            confirmation_price=100.0,
            entry_price=100.0,
        )
        summary = deterministic_summary(stock)
        self.assertIn("等待收市突破 $100.00", summary)
        self.assertIn("20日均量 1.10x", summary)

    def test_breakout_proximity_uses_buffered_entry_distance(self):
        stock = fixture_stock(
            "fresh_breakout",
            dist_pivot_20=[0.0128],
            setup_evidence={
                "dist_to_entry": 0.0078,
                "failure_flags": {},
                "extension_flags": {},
            },
        )

        rule = next(
            item for item in build_rule_checklist(stock, self.settings)
            if item["rule"] == "距離突破入場參考"
        )

        self.assertEqual(rule["status"], "Pass")
        self.assertEqual(rule["actual"], 0.0078)

    def test_52_week_high_rule_uses_positive_drawdown_and_exposes_high(self):
        stock = fixture_stock("fresh_breakout", price=120.0, pct_from_high=-0.08)
        rule = next(
            item for item in build_rule_checklist(stock, self.settings)
            if "距 52 週高位不超過" in item["rule"]
        )

        self.assertEqual(rule["status"], "Pass")
        self.assertEqual(rule["comparison"], "<=")
        self.assertEqual(rule["threshold"], 0.25)
        self.assertEqual(rule["actual"]["距高位"], 0.08)
        self.assertAlmostEqual(rule["actual"]["52 週高位"], 130.43, places=2)

    def test_weighted_score_is_not_the_rule_count(self):
        stock = fixture_stock(
            "fresh_breakout",
            setup_evidence={
                "failure_flags": {},
                "extension_flags": {},
                "breakout_score_parts": {"trend_structure": 25, "volume_context": 10},
            },
        )
        breakdown = score_breakdown(stock)
        self.assertEqual(breakdown["score"], 78)
        self.assertIn("趨勢結構 25/25", breakdown["basis"])
        score_rules = build_score_checklist(stock)
        trend_rule = next(rule for rule in score_rules if rule["rule"] == "趨勢結構")
        self.assertEqual(trend_rule["actual"], "25/25")
        self.assertEqual(trend_rule["status"], "Pass")

    def test_score_checklist_uses_each_saved_subitem_score(self):
        stock = fixture_stock(
            "fresh_breakout",
            setup_evidence={
                "failure_flags": {},
                "extension_flags": {},
                "breakout_score_parts": {"trend_structure": 25},
                "breakout_score_detail": {
                    "trend_structure": [
                        {
                            "label": "收市價 > SMA20 > SMA50",
                            "points": 10,
                            "maximum": 10,
                            "actual": {"收市價": 120.0},
                            "detail": "短中期均線呈多頭排列。",
                            "status": "Pass",
                        },
                        {
                            "label": "SMA10 > SMA20",
                            "points": 5,
                            "maximum": 5,
                            "actual": {"SMA10": 116.0},
                            "detail": "短期動能維持向上。",
                            "status": "Pass",
                        },
                    ],
                },
            },
        )

        score_rules = build_score_checklist(stock)
        short_stack = next(rule for rule in score_rules if rule["rule"] == "收市價 > SMA20 > SMA50")
        short_momentum = next(rule for rule in score_rules if rule["rule"] == "SMA10 > SMA20")

        self.assertEqual(short_stack["actual"], "10/10")
        self.assertEqual(short_momentum["actual"], "5/5")
        self.assertNotEqual(short_stack["actual"], "25/25")

    def test_score_denominators_are_explicit(self):
        breakout = score_breakdown(fixture_stock("fresh_breakout"))
        self.assertEqual(breakout["possible"], 90)
        self.assertEqual(breakout["score_display"], "78/90")

        meta = score_breakdown(fixture_stock("pullback_entry"))
        self.assertEqual(meta["possible"], 80)
        self.assertEqual(meta["score_display"], "64/80")

    def test_both_score_templates_can_be_audited_before_classification(self):
        stock = fixture_stock("fresh_breakout")

        breakout = score_breakdown(stock, source="breakout")
        meta = score_breakdown(stock, source="meta")

        self.assertEqual(breakout["display_label"], "突破評分")
        self.assertEqual(breakout["score_display"], "78/90")
        self.assertEqual(meta["display_label"], "META 評分")
        self.assertEqual(meta["score_display"], "64/80")

    def test_decision_tree_exposes_meta_review_gate(self):
        tree = build_decision_tree(fixture_stock("pullback_forming"), self.settings)
        meta_gate = next(rule for rule in tree if rule["rule"] == "META 入場條件")
        self.assertEqual(meta_gate["status"], "Needs review")
        self.assertIn("確認突破條件", {rule["rule"] for rule in tree})

    def test_meta_phase_with_saved_chart_evidence_but_no_zone_fails_meta_gate(self):
        tree = build_decision_tree(
            fixture_stock(
                "pullback_entry",
                setup_evidence={"chart_evidence": {}, "failure_flags": {}, "extension_flags": {}},
            ),
            self.settings,
        )
        meta_gate = next(rule for rule in tree if rule["rule"] == "META 入場條件")
        self.assertEqual(meta_gate["status"], "Fail")

    def test_near_earnings_is_a_separate_review_warning(self):
        stock = fixture_stock("fresh_breakout", earnings_date="2026-07-15")
        context = earnings_context(stock)
        self.assertEqual(context["status"], "Needs review")
        rule = next(
            item for item in build_rule_checklist(stock, self.settings)
            if item["group"] == "Earnings Risk"
        )
        self.assertEqual(rule["status"], "Needs review")

    def test_earnings_context_accepts_yahoo_unix_timestamp(self):
        stock = fixture_stock("fresh_breakout", data_date="2026-07-13", earnings_date=1785414600)
        context = earnings_context(stock)
        self.assertGreater(context["days"], 0)

    def test_pullback_uses_meta_score(self):
        stock = fixture_stock("pullback_entry")
        self.assertEqual(phase_score(stock), (64, "META 評分"))
        self.assertIn("0.7x", deterministic_summary(stock))

    def test_extended_requires_review(self):
        stock = fixture_stock(
            "extended_breakout",
            setup_evidence={
                "failure_flags": {},
                "extension_flags": {"within_chase_limit": False},
            },
        )
        risk_rule = next(
            rule for rule in build_rule_checklist(stock, self.settings)
            if rule["rule"] == "仍在合理追價範圍"
        )
        self.assertEqual(risk_rule["status"], "Needs review")

    def test_failed_breakout_exposes_structure_failure(self):
        stock = fixture_stock(
            "failed_breakout",
            setup_evidence={"failure_flags": {"close_back_below_pivot": True}, "extension_flags": {}},
        )
        structure = next(
            rule for rule in build_rule_checklist(stock, self.settings)
            if rule["rule"] == "突破結構仍然有效"
        )
        self.assertEqual(structure["status"], "Fail")
        self.assertIn("結構失敗", deterministic_summary(stock))

    def test_signal_details_are_rebuilt_for_old_snapshots(self):
        stock = fixture_stock(
            "fresh_breakout",
            close=[105.2, 103.8],
            low=[102.2, 101.5],
            pivot_high_20=[104.0, 104.0],
            entry_price=104.52,
            volume_ratio=[1.3],
            setup_evidence={
                "failure_flags": {
                    "failed_breakout": False,
                    "breaks_support": False,
                    "risk_too_wide": False,
                    "long_term_trend_broken": False,
                },
                "extension_flags": {
                    "above_entry_threshold": False,
                    "extended_from_sma10": False,
                    "extended_by_atr": False,
                },
            },
        )

        structure = next(
            rule for rule in build_rule_checklist(stock, self.settings)
            if rule["rule"] == "突破結構仍然有效"
        )
        self.assertEqual(
            set(structure["actual"]),
            {"holds_breakout_pivot", "holds_short_term_support"},
        )
        detail = structure["detail"]["holds_breakout_pivot"]
        self.assertEqual(detail["prev_close"], 105.2)
        self.assertEqual(detail["latest_close"], 103.8)
        self.assertEqual(detail["pivot_high_20"], 104.0)
        self.assertAlmostEqual(detail["entry_price"], 104.52, places=2)

    def test_not_recommended_and_incomplete_data_are_explainable(self):
        rejected = fixture_stock("not_recommended", price=8.0)
        price_rule = next(
            rule for rule in build_rule_checklist(rejected, self.settings)
            if rule["rule"] == "股價達標"
        )
        self.assertEqual(price_rule["status"], "Fail")
        self.assertIn("必要條件", deterministic_summary(rejected))

        incomplete = fixture_stock("unclear_structure", ma50=None, ma200=None)
        ma_rule = next(
            rule for rule in build_rule_checklist(incomplete, self.settings)
            if rule["rule"] == "股價 >= MA50 >= MA200"
        )
        self.assertEqual(ma_rule["status"], "Fail")

    def test_settings_fingerprint_and_disabled_rule_do_not_mix(self):
        disabled = ScreeningSettings(require_vs_market_6m=False)
        self.assertNotEqual(self.settings.fingerprint, disabled.fingerprint)
        stock = fixture_stock("near_breakout", vs_market_6m=-0.2)
        rule = next(
            item for item in build_rule_checklist(stock, disabled)
            if item["rule"] == "跑贏大市（6 個月）"
        )
        self.assertEqual(rule["status"], "Not applicable")

    def test_beta_is_not_a_detail_rule_or_score_input(self):
        rules = build_rule_checklist(fixture_stock("fresh_breakout"), self.settings)
        score_rules = build_score_checklist(fixture_stock("fresh_breakout"))
        names = [rule["rule"] for rule in [*rules, *score_rules]]
        self.assertFalse(any("beta" in name.lower() for name in names))


if __name__ == "__main__":
    unittest.main()
