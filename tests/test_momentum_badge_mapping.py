import unittest

from core.analysis.momentum import _holding_action_for_phase, _setup_status_for_phase
from utils import normalize_holding_action


class MomentumBadgeMappingTests(unittest.TestCase):
    def test_trend_family_maps_to_empty_position_category(self):
        self.assertEqual("breakout_buy", _setup_status_for_phase("fresh_breakout"))
        self.assertEqual("breakout_buy", _setup_status_for_phase("near_breakout"))
        self.assertEqual("pullback_wait", _setup_status_for_phase("pullback_entry"))
        self.assertEqual("pullback_wait", _setup_status_for_phase("pullback_forming"))
        self.assertEqual("watchlist", _setup_status_for_phase("unclear_structure"))
        self.assertEqual("watchlist", _setup_status_for_phase("failed_breakout"))
        self.assertEqual("watchlist", _setup_status_for_phase("extended_breakout"))

    def test_only_confirmed_entry_phases_can_add(self):
        self.assertEqual(
            "add",
            _holding_action_for_phase("fresh_breakout", breaks_support=False, extended=False),
        )
        self.assertEqual(
            "add",
            _holding_action_for_phase("pullback_entry", breaks_support=False, extended=False),
        )
        self.assertEqual(
            "hold",
            _holding_action_for_phase("near_breakout", breaks_support=False, extended=False),
        )
        self.assertEqual(
            "hold",
            _holding_action_for_phase("pullback_forming", breaks_support=False, extended=False),
        )

    def test_risk_overrides_entry_phase(self):
        self.assertEqual(
            "sell_weakness",
            _holding_action_for_phase("fresh_breakout", breaks_support=True, extended=False),
        )
        self.assertEqual(
            "add",
            _holding_action_for_phase("fresh_breakout", breaks_support=False, extended=True),
        )
        self.assertEqual(
            "hold",
            _holding_action_for_phase("near_breakout", breaks_support=False, extended=True),
        )

    def test_legacy_sell_strength_normalizes_to_hold(self):
        self.assertEqual("hold", normalize_holding_action("sell_strength", "extended_breakout"))


if __name__ == "__main__":
    unittest.main()
