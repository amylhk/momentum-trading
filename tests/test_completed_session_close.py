import unittest

import pandas as pd

from services.stock_service import (
    completed_session_closes_from_intraday,
    repair_completed_session_close,
)


class CompletedSessionCloseTests(unittest.TestCase):
    def _frame(self, close=float('nan')):
        return pd.DataFrame(
            {
                'Open': [72.14],
                'High': [72.20],
                'Low': [72.07],
                'Close': [close],
                'Volume': [3_264_664],
            },
            index=pd.DatetimeIndex(['2026-08-03'], name='Date'),
        )

    def test_repairs_missing_close_for_expected_completed_session(self):
        repaired = repair_completed_session_close(self._frame(), '2026-08-03', 72.14)

        self.assertEqual(repaired.iloc[-1]['Close'], 72.14)

    def test_does_not_use_quote_outside_daily_range(self):
        repaired = repair_completed_session_close(self._frame(), '2026-08-03', 72.31)

        self.assertTrue(pd.isna(repaired.iloc[-1]['Close']))

    def test_does_not_repair_a_different_session(self):
        repaired = repair_completed_session_close(self._frame(), '2026-08-04', 72.14)

        self.assertTrue(pd.isna(repaired.iloc[-1]['Close']))

    def test_extracts_last_close_from_expected_intraday_session(self):
        columns = pd.MultiIndex.from_product([['Close'], ['TECH', 'VIK']])
        intraday = pd.DataFrame(
            [[72.12, 106.20], [72.14, 106.42], [72.30, 107.00]],
            index=pd.DatetimeIndex(
                [
                    '2026-08-03 19:58:00+00:00',
                    '2026-08-03 19:59:00+00:00',
                    '2026-08-04 13:30:00+00:00',
                ]
            ),
            columns=columns,
        )

        closes = completed_session_closes_from_intraday(intraday, '2026-08-03')

        self.assertEqual(closes, {'TECH': 72.14, 'VIK': 106.42})


if __name__ == '__main__':
    unittest.main()
