from __future__ import annotations

import unittest

import pandas as pd

from core.analysis.chart_evidence import build_chart_evidence
from core.analysis.momentum import _entry_plan, rebuild_entry_plan_from_snapshot


def _pivot_frame() -> pd.DataFrame:
    rows = 90
    frame = pd.DataFrame({
        "open": [60.0] * 64 + [71.0] * 26,
        "high": [65.0] * 64 + [72.37] * 26,
        "low": [55.0] * 64 + [70.08] * 26,
        "close": [60.0] * 64 + [72.09] * 26,
        "volume": [1_000_000] * rows,
        "ATR": [0.808] * rows,
        "SMA_10": [71.83] * rows,
        "SMA_20": [71.50] * rows,
        "SMA_50": [69.0] * rows,
        "pivot_high_20": [72.37] * rows,
        "pivot_high_50": [72.37] * rows,
        "range_contraction": [True] * rows,
    })
    return frame


class BreakoutStopPlanTests(unittest.TestCase):
    def test_stop_uses_same_primary_pivot_zone_as_chart(self):
        frame = _pivot_frame()
        evidence = build_chart_evidence(frame)

        plan = _entry_plan(
            frame.iloc[-1],
            "near_breakout",
            frame["low"].tail(10).min(),
            evidence,
        )

        self.assertAlmostEqual(plan["stop_pivot_zone_low"], 70.08, places=2)
        self.assertAlmostEqual(plan["stop_pivot_zone_high"], 72.37, places=2)
        self.assertAlmostEqual(plan["stop_price"], 70.08 - 0.25 * 0.808, places=3)
        self.assertLess(plan["stop_price"], plan["stop_pivot_zone_low"])

    def test_saved_indicator_snapshot_can_rebuild_structural_stop(self):
        frame = _pivot_frame()
        snapshot = frame.to_dict(orient="list")
        snapshot.update({
            "setup_phase": "near_breakout",
            "stop_price": 71.62,
            "setup_evidence": {},
        })

        rebuilt = rebuild_entry_plan_from_snapshot(snapshot)

        self.assertAlmostEqual(rebuilt["stop_price"], 69.878, places=3)
        self.assertAlmostEqual(rebuilt["risk_pct"], 0.0392, places=3)


if __name__ == "__main__":
    unittest.main()
