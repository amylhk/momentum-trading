import unittest

from utils import _dedupe_overlapping_trendline_spans


class TrendlineDedupeTests(unittest.TestCase):
    def test_keeps_only_best_ranked_line_for_same_display_span(self):
        candidates = [
            {'latest_level': 98.3, '_display_start_index': 6, '_display_end_index': 89},
            {'latest_level': 92.0, '_display_start_index': 6, '_display_end_index': 89},
            {'latest_level': 105.9, '_display_start_index': 0, '_display_end_index': 72},
        ]

        result = _dedupe_overlapping_trendline_spans(candidates)

        self.assertEqual(result, [candidates[0]])

    def test_keeps_lines_from_distinct_time_spans(self):
        candidates = [
            {'latest_level': 98.3, '_display_start_index': 0, '_display_end_index': 30},
            {'latest_level': 92.0, '_display_start_index': 45, '_display_end_index': 89},
        ]

        self.assertEqual(_dedupe_overlapping_trendline_spans(candidates), candidates)


if __name__ == '__main__':
    unittest.main()
