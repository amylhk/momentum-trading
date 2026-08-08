from __future__ import annotations

import unittest

import pandas as pd

from core.analysis.momentum import (
    _failed_breakout_evidence,
    _recent_breakout_evidence,
)


class FailedBreakoutTests(unittest.TestCase):
    def test_requires_prior_volume_backed_breakout_then_loss_of_raw_pivot(self):
        prev = pd.Series({
            "close": 104.60,
            "pivot_high_20": 104.00,
            "volume_ratio": 1.30,
        })
        latest = pd.Series({"close": 103.90})

        detail = _failed_breakout_evidence(prev, latest)

        self.assertTrue(detail["failed"])
        self.assertTrue(detail["confirmed_prior_breakout"])
        self.assertTrue(detail["latest_below_pivot"])

    def test_dipping_below_buffered_entry_but_holding_pivot_is_not_failure(self):
        prev = pd.Series({
            "close": 104.60,
            "pivot_high_20": 104.00,
            "volume_ratio": 1.30,
        })
        latest = pd.Series({"close": 104.20})

        detail = _failed_breakout_evidence(prev, latest)

        self.assertFalse(detail["failed"])
        self.assertFalse(detail["latest_below_pivot"])

    def test_prior_breakout_must_have_volume_confirmation(self):
        prev = pd.Series({
            "close": 104.60,
            "pivot_high_20": 104.00,
            "volume_ratio": 1.05,
        })
        latest = pd.Series({"close": 103.90})

        detail = _failed_breakout_evidence(prev, latest)

        self.assertFalse(detail["failed"])
        self.assertFalse(detail["volume_confirm"])

    def test_latest_day_can_establish_the_breakout_context(self):
        history = pd.DataFrame([
            {"close": 103.80, "pivot_high_20": 103.50, "volume_ratio": 0.70},
            {"close": 105.73, "pivot_high_20": 103.51, "volume_ratio": 1.80},
        ])

        detail = _failed_breakout_evidence(history, history.iloc[-1])

        self.assertTrue(detail["breakout_context"])
        self.assertFalse(detail["failed"])
        self.assertEqual(detail["breakout_close"], 105.73)

    def test_breakout_remains_traceable_beyond_the_previous_day(self):
        history = pd.DataFrame([
            {"close": 104.60, "pivot_high_20": 104.00, "volume_ratio": 1.30},
            {"close": 105.20, "pivot_high_20": 104.60, "volume_ratio": 0.80},
            {"close": 103.90, "pivot_high_20": 105.20, "volume_ratio": 0.90},
        ])

        detail = _failed_breakout_evidence(history, history.iloc[-1])

        self.assertTrue(detail["breakout_context"])
        self.assertTrue(detail["failed"])
        self.assertEqual(detail["pivot_high_20"], 104.00)


class RecentBreakoutTests(unittest.TestCase):
    def test_price_without_buffer_or_volume_is_not_a_recent_breakout(self):
        history = pd.DataFrame([
            {"close": 100.30, "pivot_high_20": 100.00, "volume_ratio": 1.30},
            {"close": 100.60, "pivot_high_20": 100.00, "volume_ratio": 0.90},
            {"close": 99.80, "pivot_high_20": 100.00, "volume_ratio": 1.20},
            {"close": 99.90, "pivot_high_20": 100.00, "volume_ratio": 0.80},
        ], index=pd.to_datetime(["2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30"]))

        detail = _recent_breakout_evidence(history)

        self.assertFalse(detail["confirmed"])

    def test_recent_breakout_requires_buffer_and_volume_on_same_session(self):
        history = pd.DataFrame([
            {"close": 99.80, "pivot_high_20": 100.00, "volume_ratio": 0.80},
            {"close": 100.60, "pivot_high_20": 100.00, "volume_ratio": 1.20},
            {"close": 100.20, "pivot_high_20": 100.00, "volume_ratio": 0.80},
            {"close": 100.10, "pivot_high_20": 100.00, "volume_ratio": 0.90},
            {"close": 99.90, "pivot_high_20": 100.00, "volume_ratio": 0.80},
        ], index=pd.to_datetime([
            "2026-07-26", "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30",
        ]))

        detail = _recent_breakout_evidence(history)

        self.assertTrue(detail["confirmed"])
        self.assertEqual(detail["date"], "2026-07-27")
        self.assertAlmostEqual(detail["confirmation_price"], 100.50)
        self.assertAlmostEqual(detail["volume_ratio"], 1.20)

    def test_remaining_above_confirmation_price_is_not_a_new_breakout(self):
        history = pd.DataFrame([
            {"close": 100.70, "pivot_high_20": 100.00, "volume_ratio": 1.00},
            {"close": 100.80, "pivot_high_20": 100.00, "volume_ratio": 1.30},
            {"close": 100.90, "pivot_high_20": 100.00, "volume_ratio": 1.20},
            {"close": 100.60, "pivot_high_20": 100.00, "volume_ratio": 0.80},
        ], index=pd.to_datetime([
            "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30",
        ]))

        detail = _recent_breakout_evidence(history)

        self.assertFalse(detail["confirmed"])


if __name__ == "__main__":
    unittest.main()
