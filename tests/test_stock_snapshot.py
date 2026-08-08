import unittest

import pandas as pd

from ui.stock_snapshot import comparable_timestamp, row_freshness, row_ohlcv_frame


class StockSnapshotTests(unittest.TestCase):
    def test_freshness_uses_actual_series_end_when_metadata_dates_match(self):
        stale = {
            "data_date": "2026-07-20",
            "updated_at": "2026-07-20 18:00:00",
            "date": ["2026-07-12", "2026-07-13"],
        }
        complete = {
            "data_date": "2026-07-20",
            "updated_at": "2026-07-20 18:00:00",
            "date": ["2026-07-13", "2026-07-20"],
        }

        self.assertIs(max((stale, complete), key=row_freshness), complete)

    def test_saved_ohlcv_frame_preserves_complete_snapshot(self):
        snapshot = {
            "date": ["2026-07-13", "2026-07-20"],
            "open": [112.0, 113.0],
            "high": [116.0, 115.0],
            "low": [111.0, 112.0],
            "close": [115.09, 114.19],
            "volume": [1_000_000, 1_200_000],
        }

        frame = row_ohlcv_frame(snapshot)

        self.assertEqual(list(frame.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(frame.index[-1], pd.Timestamp("2026-07-20"))
        self.assertAlmostEqual(frame.iloc[-1]["Close"], 114.19)

    def test_comparable_timestamp_removes_timezone(self):
        timestamp = comparable_timestamp("2026-07-20T16:00:00-04:00")

        self.assertIsNone(timestamp.tzinfo)
        self.assertEqual(timestamp, pd.Timestamp("2026-07-20 20:00:00"))


if __name__ == "__main__":
    unittest.main()
