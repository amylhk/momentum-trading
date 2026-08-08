from __future__ import annotations

import unittest

import pandas as pd

from ui.rule_display import display_score_actual, display_score_value, expand_signal_dict_rows


class RuleDisplayTests(unittest.TestCase):
    def test_score_evidence_hides_internal_variable_names(self):
        rendered = display_score_value({
            "價格條件類別": ["gap_fill", "pivot_zone", "short_ma_cluster"],
            "來源": ["MA 10 / MA 20", "Gap Fill"],
        })

        self.assertEqual(
            rendered,
            "價格條件類別：裂口回補、樞紐區、短期均線匯聚\n"
            "來源：10MA / 20MA 匯聚區、未完全回補裂口",
        )
        self.assertNotIn("gap_fill", rendered)
        self.assertNotIn("pivot_zone", rendered)

    def test_score_meta_zone_evidence_is_readable(self):
        rendered = display_score_value({
            "source": "MA 10 / MA 20 + Gap Fill",
            "family": "confluence",
            "low": 327.14,
            "high": 327.35,
            "is_confluence": True,
        })

        self.assertIn("來源：10MA / 20MA 匯聚區 + 未完全回補裂口", rendered)
        self.assertIn("技術條件：多重條件匯聚", rendered)
        self.assertIn("區域下限：$327.14", rendered)
        self.assertIn("是否為匯聚區：是", rendered)

    def test_score_scalar_evidence_uses_friendly_units(self):
        self.assertEqual(display_score_value({"3日/20日均量": 0.7}), "3日/20日均量：0.70x")
        self.assertEqual(display_score_value({"risk_pct": 0.06}), "風險百分比：+6.0%")
        self.assertEqual(display_score_value({"收市價": 327.14}), "收市價：$327.14")

    def test_score_actual_formats_breakout_distance_and_rsi(self):
        self.assertEqual(display_score_actual(0.012345, "距離突破樞紐"), "+1.2%")
        self.assertEqual(display_score_actual(62.345, "RSI 動能區間"), "62.3")

    def test_score_actual_is_one_concise_value_per_condition(self):
        self.assertEqual(
            display_score_actual(
                {"價格條件類別": ["gap_fill", "pivot_zone", "short_ma_cluster"]},
                "META 匯聚度",
            ),
            "✅ 裂口回補：計入 8 分\n"
            "✅ 樞紐區：計入 8 分\n"
            "✅ 短期均線匯聚：計入 8 分\n"
            "合計：3 種符合條件，計入 24 分／30 分",
        )
        self.assertEqual(
            display_score_actual(
                {
                    "source": "MA 10 / MA 20 + Gap Fill",
                    "low": 327.14,
                    "high": 327.35,
                    "現時股價": 327.20,
                    "距離區頂": -0.00046,
                    "是否在區內": True,
                },
                "距離選定 META 區",
            ),
            "META 區 $327.14–$327.35\n現價 $327.20\n距離區頂 0.0%\n位於 META 區內\n"
            "來源：10MA / 20MA 匯聚區 + 未完全回補裂口",
        )

    def test_meta_confluence_lists_missing_conditions_and_caps_extra_points(self):
        rendered = display_score_actual(
            {
                "所有現存條件": ["gap_fill", "pivot_zone", "sma50"],
                "符合條件": ["gap_fill", "pivot_zone"],
            },
            "META 匯聚度",
        )

        self.assertIn("✅ 裂口回補：計入 8 分", rendered)
        self.assertIn("✅ 樞紐區：計入 8 分", rendered)
        self.assertIn("❌ 50MA：0 分", rendered)
        self.assertIn("合計：2 種符合條件，計入 16 分／30 分", rendered)

    def test_validity_flags_expand_to_readable_rows(self):
        rules = pd.DataFrame([
            {
                "group": "Setup",
                "rule": "突破結構仍然有效",
                "status": "Pass",
                "actual": {
                    "holds_breakout_pivot": True,
                    "holds_short_term_support": True,
                },
                "threshold": False,
                "comparison": "無失敗訊號",
                "detail": {
                    "holds_breakout_pivot": {
                        "breakout_context": True,
                        "latest_close": 105.0,
                        "pivot_high_20": 104.0,
                        "latest_below_pivot": False,
                    },
                    "holds_short_term_support": {
                        "breakout_context": True,
                        "latest_close": 105.0,
                        "sma20": 100.0,
                        "sma50": 95.0,
                        "below_sma20": False,
                        "below_sma50": False,
                    },
                },
            }
        ])

        expanded = expand_signal_dict_rows(rules)
        signal_rows = expanded[expanded["rule"].str.contains(r"\(", regex=True)]

        self.assertEqual(
            signal_rows["rule"].tolist(),
            [
                "守在原突破樞紐上 (holds_breakout_pivot)",
                "守在短中期支持上 (holds_short_term_support)",
            ],
        )
        self.assertTrue(all(signal_rows["status"] == "✅"))

    def test_extension_flags_expand_to_readable_rows(self):
        rules = pd.DataFrame([
            {
                "group": "Risk",
                "rule": "仍在合理追價範圍",
                "status": "Needs review",
                "actual": {"within_chase_limit": False},
                "threshold": False,
                "comparison": "未延伸",
                "detail": {
                    "within_chase_limit": {
                        "latest_close": 106.0,
                        "entry_price": 104.0,
                        "entry_ceiling": 105.04,
                        "ceiling_pct": 0.01,
                    },
                },
            },
        ])

        expanded = expand_signal_dict_rows(rules)
        signal_rows = expanded[expanded["rule"].str.contains(r"\(", regex=True)]

        self.assertEqual(len(expanded), 1)
        self.assertEqual(
            signal_rows["rule"].tolist(),
            ["仍在追價上限內 (within_chase_limit)"],
        )
        self.assertEqual(signal_rows.iloc[0]["status"], "❌")
        self.assertEqual(
            signal_rows.iloc[0]["actual"],
            "最新收市 $106.00 | 入場參考 $104.00 | 追價上限 $105.04",
        )
        self.assertEqual(
            signal_rows.iloc[0]["threshold"],
            "入場參考 $104.00 + 1.0% = $105.04",
        )
        self.assertEqual(signal_rows.iloc[0]["comparison"], "<=")

    def test_signal_detail_uses_saved_price_values(self):
        rules = pd.DataFrame([
            {
                "group": "Setup",
                "rule": "突破結構仍然有效",
                "status": "Fail",
                "actual": {"holds_breakout_pivot": False},
                "threshold": True,
                "comparison": "兩項均成立",
                "detail": {
                    "holds_breakout_pivot": {
                        "prev_close": 105.2,
                        "latest_close": 103.8,
                        "pivot_high_20": 104.0,
                        "entry_price": 104.52,
                        "volume_ratio": 1.3,
                        "volume_threshold": 1.1,
                        "breakout_context": True,
                        "latest_below_pivot": True,
                        "prev_above_entry": True,
                        "volume_confirm": True,
                    }
                },
            }
        ])

        expanded = expand_signal_dict_rows(rules)

        self.assertEqual(expanded.iloc[0]["rule"], "守在原突破樞紐上 (holds_breakout_pivot)")
        self.assertEqual(expanded.iloc[0]["status"], "❌")
        self.assertEqual(
            expanded.iloc[0]["actual"],
            "最新收市 $103.80 | 原 20 日樞紐 $104.00",
        )
        self.assertEqual(expanded.iloc[0]["threshold"], "原 20 日樞紐 $104.00")
        self.assertEqual(expanded.iloc[0]["comparison"], ">=")

    def test_failed_breakout_hides_transition_check_without_prior_breakout(self):
        rules = pd.DataFrame([{
            "group": "Setup",
            "rule": "突破結構仍然有效",
            "status": "Pass",
            "actual": {"holds_breakout_pivot": True},
            "threshold": True,
            "comparison": "兩項均成立",
            "detail": {
                "holds_breakout_pivot": {
                    "breakout_context": False,
                    "latest_below_pivot": False,
                    "latest_close": 105.73,
                    "pivot_high_20": 103.51,
                }
            },
        }])

        expanded = expand_signal_dict_rows(rules)

        self.assertEqual(expanded.iloc[0]["status"], "—")
        self.assertEqual(
            expanded.iloc[0]["actual"],
            "最新收市 $105.73 | 原 20 日樞紐 $103.51",
        )
        self.assertEqual(expanded.iloc[0]["comparison"], "確認突破後才檢查")


if __name__ == "__main__":
    unittest.main()
