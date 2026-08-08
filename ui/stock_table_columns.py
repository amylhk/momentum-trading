SCREENER_REVIEW_COLUMNS = [
    'symbol_link', 'name', 'price', 'market_cap_b', 'sector_display', 'industry_display',
    'classification', 'setup_phase_display', 'holding_action_display', 'score_equivalent',
    'vs_market_1m', 'vs_market_3m', 'vs_market_6m', 'days_to_earnings_display', 'data_date',
]

MY_STOCKS_REVIEW_COLUMNS = [
    'symbol_link', 'name', 'price', 'market_cap_b', 'sector_display', 'industry_display',
    'classification', 'setup_phase_display', 'holding_action_display', 'score_equivalent',
    'days_to_earnings_display', 'data_date',
]

# Kept as a compatibility alias for callers outside the two primary tables.
STOCK_REVIEW_COLUMNS = [
    'symbol_link', 'name', 'price', 'market_cap_b', 'sector_display', 'industry_display',
    'classification', 'setup_status_display', 'setup_phase_display', 'holding_action_display',
    'score_equivalent', 'score_label', 'score_basis', 'rules_passed',
    'avgvol3m', 'vs_market_1m', 'vs_market_3m', 'vs_market_6m',
    'volume_ratio', 'dry_up_ratio', 'dist_pivot_20', 'ATR_pct', 'RSI',
    'entry_price', 'stop_price', 'risk_pct',
    'breakout_score', 'meta_score', 'trendline_dist', 'trendline_slope', 'trendline_breakout',
    'days_to_earnings_display', 'earnings_date', 'earnings_warning',
    'data_date', 'notes',
]


def holding_action_display_text(action):
    display = {
        'add': '➕ 考慮加倉',
        'sell_weakness': '⚠️ 趁弱賣出',
        'hold': '✅ 繼續持有',
    }
    return display.get(action, display['hold'])
