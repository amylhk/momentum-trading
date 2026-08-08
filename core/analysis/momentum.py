import pandas as pd
import pandas_ta as ta

from core.analysis.chart_evidence import build_chart_evidence
from utils import clean_yf_df


BREAKOUT_BUFFER_PCT = 0.005
NEAR_BREAKOUT_MIN_DIST = -0.03
NEAR_BREAKOUT_MAX_DIST = 0.0
BREAKOUT_VOLUME_CONFIRM_RATIO = 1.10
FRESH_BREAKOUT_MAX_ABOVE_ENTRY_PCT = 0.01
# Temporary no-chase boundary. Review this value before the next ruleset revision.
BREAKOUT_CHASE_CEILING_PCT = 0.01
MAX_RISK_PCT = 0.08
# A META candidate can be a nearby support area before price reaches it. The
# distance component still decides how many points that proximity earns.
META_PROXIMITY_PCT = 0.05
META_DISTANCE_PARTIAL_PCT = 0.03
META_MAX_ZONE_WIDTH_PCT = 0.03
PULLBACK_DRY_UP_RATIO = 0.90
FAILURE_VOLUME_RATIO = 1.20
EXTENDED_FROM_SMA10_PCT = 0.08
EXTENDED_FROM_SMA20_ATR = 3
RECENT_BREAKOUT_LOOKBACK = 3
FLOAT_EPSILON = 1e-9


def _safe_div(numerator, denominator, default=0.0):
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return default
    return numerator / denominator


def _safe_float(value, default=None):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _failed_breakout_evidence(history, latest):
    """Find the latest confirmed breakout and test whether its pivot was lost."""
    if isinstance(history, pd.Series):
        candidates = history.to_frame().T
    else:
        candidates = history

    confirmed = candidates[
        candidates['close'].notna()
        & candidates['pivot_high_20'].notna()
        & candidates['volume_ratio'].notna()
        & (
            candidates['close']
            >= candidates['pivot_high_20'] * (1 + BREAKOUT_BUFFER_PCT)
        )
        & (candidates['volume_ratio'] >= BREAKOUT_VOLUME_CONFIRM_RATIO)
    ]
    breakout = confirmed.iloc[-1] if not confirmed.empty else None
    breakout_close = _safe_float(breakout.get('close')) if breakout is not None else None
    latest_close = _safe_float(latest.get('close'))
    breakout_pivot = _safe_float(breakout.get('pivot_high_20')) if breakout is not None else None
    breakout_entry = (
        breakout_pivot * (1 + BREAKOUT_BUFFER_PCT)
        if breakout_pivot is not None
        else None
    )
    breakout_volume_ratio = _safe_float(breakout.get('volume_ratio')) if breakout is not None else None
    breakout_confirmed = breakout is not None
    pivot_window_start = None
    pivot_window_end = None
    pivot_source_date = None
    if breakout is not None and 'high' in candidates.columns:
        breakout_position = next(
            (
                index
                for index in range(len(candidates) - 1, -1, -1)
                if candidates.index[index] == breakout.name
            ),
            None,
        )
        if breakout_position is not None:
            pivot_window = candidates.iloc[max(0, breakout_position - 20):breakout_position]
            valid_highs = pd.to_numeric(pivot_window['high'], errors='coerce').dropna()
            if not pivot_window.empty:
                pivot_window_start = pd.Timestamp(pivot_window.index[0]).date().isoformat()
                pivot_window_end = pd.Timestamp(pivot_window.index[-1]).date().isoformat()
            if not valid_highs.empty:
                pivot_source_date = pd.Timestamp(valid_highs.idxmax()).date().isoformat()
    latest_below_pivot = bool(
        latest_close is not None
        and breakout_pivot is not None
        and latest_close < breakout_pivot
    )
    return {
        'failed': bool(breakout_confirmed and latest_below_pivot),
        'breakout_context': breakout_confirmed,
        'breakout_close': breakout_close,
        'breakout_date': breakout.name if breakout is not None else None,
        'latest_close': latest_close,
        'pivot_high_20': breakout_pivot,
        'pivot_window_start': pivot_window_start,
        'pivot_window_end': pivot_window_end,
        'pivot_source_date': pivot_source_date,
        'entry_price': breakout_entry,
        'volume_ratio': breakout_volume_ratio,
        'volume_threshold': BREAKOUT_VOLUME_CONFIRM_RATIO,
        'latest_below_pivot': latest_below_pivot,
        # Legacy aliases keep older display helpers and saved snapshots readable.
        'prev_close': breakout_close,
        'prev_above_entry': breakout_confirmed,
        'volume_confirm': breakout_confirmed,
        'confirmed_prior_breakout': breakout_confirmed,
    }


def _nearest_support_below(latest, recent_low, chart_evidence=None, upper_bound=None):
    close = _safe_float(latest.get('close'))
    if close is None:
        return None

    candidates = [
        _safe_float(latest.get('SMA_10')),
        _safe_float(latest.get('SMA_20')),
        _safe_float(recent_low),
    ]
    ceiling = close if upper_bound is None else min(close, upper_bound)
    candidates = [value for value in candidates if value is not None and value > 0 and value < ceiling]
    for zone in (chart_evidence or {}).get('meta_candidates', []):
        upper = _safe_float(zone.get('high'))
        if upper is not None and 0 < upper < ceiling:
            candidates.append(upper)
    if not candidates:
        fallback = _safe_float(latest.get('SMA_20')) or _safe_float(latest.get('SMA_10'))
        return fallback if fallback is not None and fallback < ceiling else None
    return max(candidates)


def _nearest_resistance_or_pivot(latest):
    for key in ['pivot_high_20', 'pivot_high_50']:
        value = _safe_float(latest.get(key))
        if value is not None and value > 0:
            return value
    return None


def _pivot_zone_stop_base(latest, chart_evidence=None, upper_bound=None):
    """Return the lower edge of the most relevant pivot zone for a stop.

    A pivot zone is useful only when it is below the planned entry and either
    contains the current price or sits closest below it.  This keeps a broad,
    distant historical range from becoming an arbitrary stop level.
    """
    close = _safe_float(latest.get('close'))
    reference = close if close is not None else upper_bound
    if upper_bound is not None and reference is not None:
        reference = min(reference, upper_bound)
    breakout_pivot = _nearest_resistance_or_pivot(latest)
    atr = _safe_float(latest.get('ATR'), 0.0) or 0.0
    pivot_tolerance = max((breakout_pivot or 0) * 0.015, atr)
    candidates = []
    for zone in (chart_evidence or {}).get('meta_candidates', []):
        if zone.get('family') != 'pivot_zone' or zone.get('stop_eligible', True) is False:
            continue
        low = _safe_float(zone.get('low'))
        high = _safe_float(zone.get('high'))
        if low is None or high is None or low <= 0 or high <= 0:
            continue
        if reference is None:
            continue
        contains_reference = low <= reference <= high
        below_reference = high < reference
        if not (contains_reference or below_reference):
            continue
        matches_breakout_pivot = bool(
            breakout_pivot is not None
            and abs(high - breakout_pivot) <= pivot_tolerance
        )
        candidates.append((
            matches_breakout_pivot,
            bool(zone.get('is_primary_pivot')),
            contains_reference,
            high,
            low,
            zone,
        ))
    if not candidates:
        return None, None
    matching = [candidate for candidate in candidates if candidate[0]]
    pool = matching or candidates
    _, _, _, _, low, zone = max(
        pool,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    return low, zone


def _cap_stop_to_max_risk(entry_price, stop_price):
    risk_pct = _safe_div(entry_price - stop_price, entry_price, default=None)
    if risk_pct is not None and risk_pct > MAX_RISK_PCT + FLOAT_EPSILON:
        return entry_price * (1 - MAX_RISK_PCT), True
    return stop_price, False


def _entry_plan(latest, setup_phase, recent_low, chart_evidence=None):
    close = _safe_float(latest.get('close'))
    pivot = _nearest_resistance_or_pivot(latest)
    atr = _safe_float(latest.get('ATR'), 0.0) or 0.0
    support = _nearest_support_below(latest, recent_low, chart_evidence)

    if close is None:
        return {
            'entry_price': None,
            'entry_zone_low': None,
            'entry_zone_high': None,
            'confirmation_price': None,
            'stop_price': None,
            'risk_pct': None,
            'entry_note': '價格資料不足，暫時未能計算進場與止損參考。',
        }

    risk_capped = False
    stop_base = None
    pivot_zone = None
    pivot_zone_source = None
    breakout_phases = {'near_breakout', 'fresh_breakout', 'extended_breakout', 'failed_breakout'}
    if setup_phase in breakout_phases and pivot:
        entry_zone_low = pivot
        entry_zone_high = pivot * (1 + BREAKOUT_BUFFER_PCT)
        confirmation_price = entry_zone_high
        entry_price = confirmation_price
        pivot_zone_base, pivot_zone = _pivot_zone_stop_base(
            latest, chart_evidence, upper_bound=entry_price
        )
        stop_base = pivot_zone_base or support or pivot * 0.95
        stop_price = stop_base - (0.25 * atr if atr else 0)
        pivot_zone_source = str((pivot_zone or {}).get('source') or 'Pivot Zone')
        stop_note = (
            f"止損以 {pivot_zone_source} 下界 {_safe_float(pivot_zone_base):.2f} 再減 0.25 ATR 緩衝"
            if pivot_zone_base is not None
            else "止損以最近支持位再減 0.25 ATR 緩衝"
        )
        note = f"突破入場區：樞紐 {_safe_float(pivot):.2f} 至確認價 {_safe_float(confirmation_price):.2f}（+{BREAKOUT_BUFFER_PCT:.1%}）；{stop_note}。"
    elif setup_phase in {'pullback_forming', 'pullback_entry'}:
        selected_meta = (chart_evidence or {}).get('selected_meta_zone') or {}
        entry_zone_low = _safe_float(selected_meta.get('low'))
        entry_zone_high = _safe_float(selected_meta.get('high'))
        if entry_zone_low is None or entry_zone_high is None:
            # Do not manufacture a META range ending exactly at the current
            # price. Without a selected confluence zone there is no META
            # entry area to present; the stock remains unconfirmed.
            entry_zone_low = None
            entry_zone_high = None
        confirmation_price = close if setup_phase == 'pullback_entry' else None
        entry_price = confirmation_price or entry_zone_high or close
        stop_support = _nearest_support_below(
            latest,
            recent_low,
            chart_evidence,
            upper_bound=entry_zone_low,
        )
        stop_base = stop_support or entry_zone_low or support or close * 0.95
        stop_price = stop_base - (0.25 * atr if atr else 0)
        note = "META 入場區以多重支持匯聚判定；等反轉確認價出現後才入場，止損參考設於 META 下方結構位。"
    else:
        entry_price = close
        entry_zone_low = None
        entry_zone_high = None
        confirmation_price = None
        stop_price = close * (1 - MAX_RISK_PCT)
        note = "暫未有可行動的進場結構；只顯示 8% 風險上限作參考。"

    raw_risk_pct = _safe_div(entry_price - stop_price, entry_price, default=None)
    if stop_price is None or stop_price <= 0 or stop_price >= entry_price:
        stop_price = entry_price * (1 - MAX_RISK_PCT)
        note = f"{note} 技術止損位不清晰，改用 8% 風險上限參考。"
    else:
        stop_price, risk_capped = _cap_stop_to_max_risk(entry_price, stop_price)

    risk_pct = _safe_div(entry_price - stop_price, entry_price, default=None)
    if risk_capped:
        note = f"{note} 原始止損風險超過 8%，已限制為 8% 風險上限參考。"
    return {
        'entry_price': entry_price,
        'entry_zone_low': entry_zone_low,
        'entry_zone_high': entry_zone_high,
        'confirmation_price': confirmation_price,
        'stop_price': stop_price,
        'risk_pct': risk_pct,
        'raw_risk_pct': raw_risk_pct,
        'risk_capped': risk_capped,
        'stop_base': stop_base,
        'stop_source': pivot_zone_source if setup_phase in breakout_phases and pivot else None,
        'stop_buffer_atr': 0.25 if atr else 0.0,
        'stop_pivot_zone_low': _safe_float((pivot_zone or {}).get('low')) if setup_phase in breakout_phases and pivot else None,
        'stop_pivot_zone_high': _safe_float((pivot_zone or {}).get('high')) if setup_phase in breakout_phases and pivot else None,
        'entry_note': note,
    }


def rebuild_entry_plan_from_snapshot(row_data: dict) -> dict:
    """Rebuild entry/stop fields from saved indicator lists without rerunning MAs.

    Saved screener rows begin only after long-window indicators become valid, so
    they may not contain 200 raw bars.  Their existing indicator lists are still
    sufficient to rebuild chart evidence and the structural entry plan.
    """

    phase = str(row_data.get('setup_phase') or '')
    breakout_phases = {'near_breakout', 'fresh_breakout', 'extended_breakout', 'failed_breakout'}
    meta_phases = {'pullback_forming', 'pullback_entry'}
    rebuildable_phases = breakout_phases | meta_phases | {'unclear_structure'}
    if phase not in rebuildable_phases:
        return row_data

    field_names = (
        'open', 'high', 'low', 'close', 'volume', 'ATR',
        'SMA_10', 'SMA_20', 'SMA_50', 'pivot_high_20', 'pivot_high_50',
        'range_contraction', 'dry_up_ratio', 'MACDh',
        'bullish_engulfing', 'reversal_180',
    )
    available = {
        field: row_data.get(field)
        for field in field_names
        if isinstance(row_data.get(field), list)
    }
    lengths = {len(values) for values in available.values()}
    if not available or len(lengths) != 1 or next(iter(lengths), 0) == 0:
        return row_data

    frame = pd.DataFrame(available)
    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else latest
    recent_low_10 = _safe_float(frame['low'].tail(10).min())
    chart_evidence = build_chart_evidence(frame)

    evidence = dict(row_data.get('setup_evidence') or {})
    provisional_risk = _safe_float(
        evidence.get('raw_risk_pct'),
        _safe_float(row_data.get('raw_risk_pct'), _safe_float(row_data.get('risk_pct'))),
    )
    meta_score, meta_score_parts, meta_score_detail = _score_meta(
        latest,
        previous,
        recent_low_10,
        chart_evidence,
        provisional_risk,
    )
    selected_meta = chart_evidence.get('selected_meta_zone')
    validity_flags = evidence.get('validity_flags') or {}
    major_failure_clear = bool(validity_flags.get('holds_short_term_support', True))

    candidate_flags = dict(evidence.get('candidate_flags') or {})
    candidate_flags['pullback_entry'] = bool(
        selected_meta is not None
        and major_failure_clear
        and meta_score >= 55
        and meta_score_parts['meta_confluence'] >= 16
        and meta_score_parts['reversal_signal'] >= 6
    )
    candidate_flags['pullback_forming'] = bool(
        selected_meta is not None
        and major_failure_clear
        and meta_score >= 45
        and meta_score_parts['meta_confluence'] >= 16
        and meta_score_parts['reversal_signal'] < 6
    )

    candidate_details = dict(evidence.get('candidate_flag_details') or {})
    common_meta_detail = {
        'score': meta_score,
        'meta_confluence': meta_score_parts['meta_confluence'],
        'meta_confluence_threshold': 16,
        'reversal_signal': meta_score_parts['reversal_signal'],
        'has_meta_zone': selected_meta is not None,
        'major_failure_clear': major_failure_clear,
    }
    candidate_details['pullback_entry'] = {
        **common_meta_detail,
        'score_threshold': 55,
        'reversal_threshold': 6,
    }
    candidate_details['pullback_forming'] = {
        **common_meta_detail,
        'score_threshold': 45,
        'reversal_threshold': 6,
    }

    if phase in meta_phases:
        if candidate_flags['pullback_entry']:
            phase = 'pullback_entry'
        elif candidate_flags['pullback_forming']:
            phase = 'pullback_forming'
        else:
            # Older snapshots could promote a broad compact base to META even
            # without overlapping independent price evidence. Do not carry
            # that stale range into the current explainability view.
            phase = 'unclear_structure'
            chart_evidence.pop('selected_meta_zone', None)

    plan = _entry_plan(latest, phase, recent_low_10, chart_evidence)

    evidence['chart_evidence'] = chart_evidence
    evidence['meta_score'] = meta_score
    evidence['meta_score_parts'] = meta_score_parts
    evidence['meta_score_detail'] = meta_score_detail
    evidence['candidate_flags'] = candidate_flags
    evidence['candidate_flag_details'] = candidate_details
    return {
        **row_data,
        **plan,
        'meta_score': meta_score,
        'setup_phase': phase,
        'setup_status': _setup_status_for_phase(phase),
        'setup_caption': _phase_caption(phase),
        'status': _setup_status_for_phase(phase),
        'setup_evidence': evidence,
    }


def _is_bullish_engulfing(df):
    prev = df.shift(1)
    return (
        (df['close'] > df['open'])
        & (prev['close'] < prev['open'])
        & (df['close'] > prev['open'])
        & (df['open'] < prev['close'])
    )


def _is_180_reversal(df):
    prev = df.shift(1)
    candle_range = (df['high'] - df['low']).replace(0, pd.NA)
    close_position = (df['close'] - df['low']) / candle_range
    return (
        (prev['close'] < prev['open'])
        & (df['low'] < prev['low'])
        & (df['close'] > prev[['open', 'close']].mean(axis=1))
        & (close_position >= 0.7)
    )


def _recent_breakout_evidence(df, lookback=RECENT_BREAKOUT_LOOKBACK):
    """Return a recent breakout only for a price crossover confirmed by volume."""
    recent = df.tail(lookback + 1).iloc[:-1]
    if recent.empty:
        return {'confirmed': False}

    previous_close = df['close'].shift(1).reindex(recent.index)
    confirmation_price = recent['pivot_high_20'] * (1 + BREAKOUT_BUFFER_PCT)
    confirmed = recent[
        recent['close'].notna()
        & recent['pivot_high_20'].notna()
        & recent['volume_ratio'].notna()
        & previous_close.notna()
        & (previous_close < confirmation_price)
        & (recent['close'] >= confirmation_price)
        & (recent['volume_ratio'] >= BREAKOUT_VOLUME_CONFIRM_RATIO)
    ]
    if confirmed.empty:
        return {'confirmed': False}

    breakout = confirmed.iloc[-1]
    pivot = _safe_float(breakout.get('pivot_high_20'))
    breakout_date = breakout.name
    if hasattr(breakout_date, 'strftime'):
        breakout_date = breakout_date.strftime('%Y-%m-%d')
    return {
        'confirmed': True,
        'date': breakout_date,
        'previous_close': _safe_float(previous_close.loc[breakout.name]),
        'close': _safe_float(breakout.get('close')),
        'pivot': pivot,
        'confirmation_price': (
            pivot * (1 + BREAKOUT_BUFFER_PCT) if pivot is not None else None
        ),
        'volume_ratio': _safe_float(breakout.get('volume_ratio')),
        'volume_threshold': BREAKOUT_VOLUME_CONFIRM_RATIO,
    }


def _has_recent_breakout(df, lookback=RECENT_BREAKOUT_LOOKBACK):
    return bool(_recent_breakout_evidence(df, lookback).get('confirmed'))


def _score_item(label, points, maximum, actual, detail):
    return {
        'label': label, 'points': points, 'maximum': maximum,
        'actual': actual, 'detail': detail,
        'status': 'Pass' if points == maximum else 'Needs review' if points else 'Fail',
    }


def _score_breakout(latest, entry_price, risk_pct, chart_evidence):
    close = _safe_float(latest.get('close'))
    volume_ratio = _safe_float(latest.get('volume_ratio'))
    dry_up_ratio = _safe_float(latest.get('dry_up_ratio'))
    score = {
        'trend_structure': 0,
        'pivot_proximity': 0,
        'base_quality': 0,
        'volume_context': 0,
        'market_leadership': 0,
        'range_contraction': 0,
        'risk_quality': 0,
    }

    sma10 = _safe_float(latest.get('SMA_10'))
    sma20 = _safe_float(latest.get('SMA_20'))
    sma50 = _safe_float(latest.get('SMA_50'))
    sma200 = _safe_float(latest.get('SMA_200'))
    rsi = _safe_float(latest.get('RSI'))

    detail = {key: [] for key in score}
    short_stack = close is not None and sma20 is not None and sma50 is not None and close > sma20 > sma50
    above_200 = close is not None and sma200 is not None and close > sma200
    short_momentum = sma10 is not None and sma20 is not None and sma10 > sma20
    rsi_points = 5 if rsi is not None and 50 <= rsi <= 75 else 2 if rsi is not None and 75 < rsi <= 80 else 0
    score['trend_structure'] = (10 if short_stack else 0) + (5 if above_200 else 0) + (5 if short_momentum else 0) + rsi_points
    detail['trend_structure'] = [
        _score_item('收市價 > MA20 > MA50', 10 if short_stack else 0, 10, {'收市價': close, 'MA20': sma20, 'MA50': sma50}, '短中期均線呈多頭排列。'),
        _score_item('收市價 > MA200', 5 if above_200 else 0, 5, {'收市價': close, 'MA200': sma200}, '長期趨勢站在 200 日均線之上。'),
        _score_item('MA10 > MA20', 5 if short_momentum else 0, 5, {'MA10': sma10, 'MA20': sma20}, '短期動能維持向上。'),
        _score_item('RSI 動能區間', rsi_points, 5, rsi, 'RSI 50-75 得滿分；75-80 得部分分數；高於 80 不加分，避免追逐過度延伸。'),
    ]

    if close is not None and entry_price:
        dist_to_entry = _safe_div(close - entry_price, entry_price, default=None)
        if dist_to_entry is not None:
            if -0.01 <= dist_to_entry <= BREAKOUT_CHASE_CEILING_PCT:
                score['pivot_proximity'] = 20
            elif -0.03 <= dist_to_entry < -0.01:
                score['pivot_proximity'] = 15
            elif -0.05 <= dist_to_entry < -0.03:
                score['pivot_proximity'] = 5
            elif BREAKOUT_CHASE_CEILING_PCT < dist_to_entry <= 0.03:
                score['pivot_proximity'] = 10
    detail['pivot_proximity'] = [_score_item('距離突破樞紐', score['pivot_proximity'], 20, _safe_div(close - entry_price, entry_price, default=None) if close is not None and entry_price else None, f'入場價下方 1% 至上方 {BREAKOUT_CHASE_CEILING_PCT:.0%} 得滿分；此追高上限為可調整變數，日後需再覆核。')]

    base = chart_evidence.get('base') or {}
    duration = int(base.get('duration_bars') or 0)
    base_range = _safe_float(base.get('range_pct'))
    base_duration_points = 5 if duration >= 15 and base_range is not None and base_range <= 0.15 else 3 if duration >= 10 and base_range is not None and base_range <= 0.20 else 0
    base_contraction_points = 3 if base.get('range_contracted') else 0
    pivot_zone_points = 2 if any(item.get('family') == 'pivot_zone' for item in chart_evidence.get('meta_candidates', [])) else 0
    score['base_quality'] = min(base_duration_points + base_contraction_points + pivot_zone_points, 10)
    detail['base_quality'] = [
        _score_item('底部持續時間與寬度', base_duration_points, 5, {'持續交易日': duration, '區間幅度': base_range}, '至少 15 日且區間不超過 15% 得 5 分；至少 10 日且不超過 20% 得 3 分。'),
        _score_item('底部區間收窄', base_contraction_points, 3, bool(base.get('range_contracted')), '近期區間收窄，反映波動收斂。'),
        _score_item('樞紐區確認', pivot_zone_points, 2, any(item.get('family') == 'pivot_zone' for item in chart_evidence.get('meta_candidates', [])), '偵測到樞紐區作為價格結構確認。'),
    ]

    breakout_context = close is not None and entry_price is not None and close >= entry_price
    if breakout_context and volume_ratio is not None and volume_ratio >= BREAKOUT_VOLUME_CONFIRM_RATIO:
        score['volume_context'] = 10
    elif not breakout_context and dry_up_ratio is not None and dry_up_ratio <= PULLBACK_DRY_UP_RATIO:
        score['volume_context'] = 10
    elif not breakout_context and dry_up_ratio is not None and dry_up_ratio <= 1.0:
        score['volume_context'] = 5
    detail['volume_context'] = [_score_item('階段性成交量背景', score['volume_context'], 10, {'階段': '突破' if breakout_context else '建構中', '當日/20日均量': volume_ratio, '3日/20日均量': dry_up_ratio}, '突破時以放量為佳；建構期以量能收縮為佳，因此同一成交量數值會按階段使用不同準則。')]

    for key, points in [('vs_market_1m', 3), ('vs_market_3m', 3), ('vs_market_6m', 4)]:
        value = _safe_float(latest.get(key))
        if value is not None and value > 0:
            score['market_leadership'] += points
        period = {'1m': '1 個月', '3m': '3 個月', '6m': '6 個月'}[key[-2:]]
        detail['market_leadership'].append(_score_item(f'跑贏大市（{period}）', points if value is not None and value > 0 else 0, points, value, '個股回報減去 S&P 500 同期回報。'))

    if bool(latest.get('range_contraction', False)):
        score['range_contraction'] = 5
    detail['range_contraction'] = [_score_item('10 日區間較 20 日收窄', score['range_contraction'], 5, bool(latest.get('range_contraction', False)), '近期波幅收窄，有助形成較緊密的底部結構。')]

    if risk_pct is not None:
        if risk_pct <= 0.05:
            score['risk_quality'] = 10
        elif risk_pct <= MAX_RISK_PCT:
            score['risk_quality'] = 5
    detail['risk_quality'] = [_score_item('技術止損風險', score['risk_quality'], 10, risk_pct, '以未套用 8% 顯示上限前的技術止損距離計算：不超過 5% 得 10 分；不超過 8% 得 5 分；其餘 0 分。')]

    return sum(score.values()), score, detail


def _score_meta(latest, prev, recent_low, chart_evidence, risk_pct):
    close = _safe_float(latest.get('close'))
    score = {
        'meta_confluence': 0,
        'distance_to_meta': 0,
        'volume_dry_up': 0,
        'range_contraction': 0,
        'reversal_signal': 0,
        'risk_quality': 0,
    }
    detail = {key: [] for key in score}
    candidates = chart_evidence.get('meta_candidates') or []
    nearby = []
    if close:
        for candidate in candidates:
            midpoint = (_safe_float(candidate.get('low'), close) + _safe_float(candidate.get('high'), close)) / 2
            if abs(_safe_div(close - midpoint, midpoint, default=1)) <= META_PROXIMITY_PCT:
                nearby.append(candidate)
    families = {
        family
        for item in nearby
        for family in (
            item.get('families', [])
            if item.get('is_confluence')
            else [item.get('family')]
        )
        if family
    }
    all_families = {
        family
        for item in candidates
        for family in (
            item.get('families', [])
            if item.get('is_confluence')
            else [item.get('family')]
        )
        if family and family != 'confluence'
    }
    score['meta_confluence'] = 30 if len(families) >= 4 else 24 if len(families) == 3 else 16 if len(families) == 2 else 8 if families else 0
    selected = _select_meta_candidate(candidates, close)
    chart_evidence.pop('selected_meta_zone', None)
    if selected:
        chart_evidence['selected_meta_zone'] = selected
        top = _safe_float(selected.get('high'), close)
        bottom = _safe_float(selected.get('low'), close)
        distance = _safe_div(close - top, top, default=None)
        score['distance_to_meta'] = (
            15
            if bottom <= close <= top * 1.01
            else 10
            if distance is not None and 0 <= distance <= META_DISTANCE_PARTIAL_PCT
            else 5
            if distance is not None and 0 <= distance <= META_PROXIMITY_PCT
            else 0
        )
    detail['meta_confluence'] = [_score_item('META 匯聚度', score['meta_confluence'], 30, {
        '價格條件類別': sorted(families),
        '所有現存條件': sorted(all_families),
        '符合條件': sorted(families),
        '來源': [item.get('source') for item in nearby],
    }, f'在共同 {META_PROXIMITY_PCT:.0%} META 候選範圍內，計算彼此獨立的技術價格條件類別；每種條件計入 8 分，最多 30 分（3 種 = 24 分；4 種或以上 = 30 分）。')]
    detail['distance_to_meta'] = [_score_item('距離選定 META 區', score['distance_to_meta'], 15, selected, '先判斷現價是否在 META 區內，再計算現價相對區頂的距離百分比；區內或高於區頂不超過 1% 得 15 分；不超過 3% 得 10 分；不超過 5% 得 5 分。')]

    dry_up_ratio = _safe_float(latest.get('dry_up_ratio'))
    if dry_up_ratio is not None:
        if dry_up_ratio <= PULLBACK_DRY_UP_RATIO:
            score['volume_dry_up'] = 10
        elif dry_up_ratio <= 1.0:
            score['volume_dry_up'] = 5
    detail['volume_dry_up'] = [_score_item('三日成交量收縮', score['volume_dry_up'], 10, dry_up_ratio, '回調情境：三日均量不超過 20 日均量的 0.90 倍得 10 分；不超過 1.00 倍得 5 分。')]

    if bool(latest.get('range_contraction', False)):
        score['range_contraction'] = 5
    detail['range_contraction'] = [_score_item('區間收窄', score['range_contraction'], 5, bool(latest.get('range_contraction', False)), '近期 10 日區間較 20 日區間收窄。')]

    if close is not None and _safe_float(latest.get('open')) is not None and close > latest['open']:
        score['reversal_signal'] += 3
    if _safe_float(latest.get('MACDh')) is not None and _safe_float(prev.get('MACDh')) is not None and latest['MACDh'] > prev['MACDh']:
        score['reversal_signal'] += 3
    if bool(latest.get('bullish_engulfing', False)) or bool(latest.get('reversal_180', False)):
        score['reversal_signal'] += 4
    green_close = bool(close is not None and _safe_float(latest.get('open')) is not None and close > latest['open'])
    macd_hist_rising = bool(_safe_float(latest.get('MACDh')) is not None and _safe_float(prev.get('MACDh')) is not None and latest['MACDh'] > prev['MACDh'])
    bullish_pattern = bool(latest.get('bullish_engulfing', False) or latest.get('reversal_180', False))
    detail['reversal_signal'] = [
        _score_item('陽燭收市', 3 if green_close else 0, 3, green_close, '收市價高於開市價。'),
        _score_item('MACD 柱狀體回升', 3 if macd_hist_rising else 0, 3, macd_hist_rising, 'MACD 柱狀體高於前一日。'),
        _score_item('看漲反轉陰陽燭', 4 if bullish_pattern else 0, 4, bullish_pattern, '偵測到吞沒或 180 度反轉陰陽燭。'),
    ]
    score['risk_quality'] = 10 if risk_pct is not None and risk_pct <= 0.05 else 5 if risk_pct is not None and risk_pct <= MAX_RISK_PCT else 0
    detail['risk_quality'] = [_score_item('技術止損風險', score['risk_quality'], 10, risk_pct, '以 META 下方技術止損距離計算，未套用顯示上限。')]

    return sum(score.values()), score, detail


def _select_meta_candidate(candidates, close):
    """Select a compact overlap between at least two independent price edges."""
    if not candidates or not close:
        return None

    nearby = []
    for candidate in candidates:
        low = _safe_float(candidate.get('low'), close)
        high = _safe_float(candidate.get('high'), close)
        midpoint = (low + high) / 2
        width_pct = _safe_div(max(0.0, high - low), midpoint, default=1)
        families = {family for family in candidate.get('families', []) if family}
        if (
            bool(candidate.get('is_confluence'))
            and len(families) >= 2
            and width_pct <= META_MAX_ZONE_WIDTH_PCT
            and abs(_safe_div(close - midpoint, midpoint, default=1)) <= META_PROXIMITY_PCT
        ):
            nearby.append(candidate)
    if not nearby:
        return None

    def rank(item):
        low = _safe_float(item.get('low'), close)
        high = _safe_float(item.get('high'), close)
        midpoint = (low + high) / 2
        width_pct = _safe_div(max(0.0, high - low), midpoint, default=1)
        distance_pct = abs(_safe_div(close - midpoint, midpoint, default=1))
        families = {family for family in item.get('families', []) if family}
        return (width_pct, -len(families), distance_pct)

    return min(nearby, key=rank)


def _phase_caption(setup_phase):
    return {
        'near_breakout': '接近突破，尚未觸發；等待股價突破入場參考位，並最好有成交量確認。',
        'fresh_breakout': '已帶量突破入場參考位；檢查是否未延伸及止損距離是否合理。',
        'extended_breakout': '突破後已延伸；不追高，等待新的底部結構或低風險回調。',
        'pullback_forming': '股價回到潛在 META，但未有反轉確認；等待停止下跌訊號。',
        'pullback_entry': '回調買點確認；以 META 下方支持作止損參考。',
        'failed_breakout': '突破失敗；暫不作新買入，持倉者檢查止損或減持。',
        'unclear_structure': '通過基本篩選，但技術結構未清晰；等待進場結構成形。',
        'not_recommended': '未符合基本篩選或技術條件，不宜買入。',
    }.get(setup_phase, '通過基本篩選，但技術結構未清晰；等待進場結構成形。')


def _setup_status_for_phase(setup_phase):
    if setup_phase in {'near_breakout', 'fresh_breakout'}:
        return 'breakout_buy'
    if setup_phase in {'pullback_forming', 'pullback_entry'}:
        return 'pullback_wait'
    return 'watchlist'


def _holding_action_for_phase(setup_phase, *, breaks_support, extended):
    if breaks_support:
        return 'sell_weakness'
    if setup_phase in {'fresh_breakout', 'pullback_entry'}:
        return 'add'
    return 'hold'


def run_tech_analysis(ohlcv: pd.DataFrame, market_metrics: dict | None = None) -> dict:
    ohlcv_indicators = clean_yf_df(ohlcv)

    # Moving averages used by the deterministic MVP rules and chart display.
    ohlcv_indicators['SMA_10'] = ta.sma(ohlcv_indicators['close'], length=10)
    ohlcv_indicators['SMA_20'] = ta.sma(ohlcv_indicators['close'], length=20)
    ohlcv_indicators['SMA_50'] = ta.sma(ohlcv_indicators['close'], length=50)
    ohlcv_indicators['SMA_200'] = ta.sma(ohlcv_indicators['close'], length=200)

    ohlcv_indicators['ATR'] = ta.atr(
        high=ohlcv_indicators['high'],
        low=ohlcv_indicators['low'],
        close=ohlcv_indicators['close'],
        length=14,
    )
    ohlcv_indicators['ATR_pct'] = ohlcv_indicators['ATR'] / ohlcv_indicators['close']

    macd_df = ta.macd(ohlcv_indicators['close'], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        ohlcv_indicators['MACD'] = macd_df.iloc[:, 0]
        ohlcv_indicators['MACDh'] = macd_df.iloc[:, 1]
        ohlcv_indicators['MACD_signal'] = macd_df.iloc[:, 2]

    ohlcv_indicators['RSI'] = ta.rsi(ohlcv_indicators['close'], length=14)
    ohlcv_indicators['Vol_MA20'] = ta.sma(ohlcv_indicators['volume'], length=20)

    ohlcv_indicators['pivot_high_20'] = ohlcv_indicators['high'].shift(1).rolling(window=20).max()
    ohlcv_indicators['pivot_high_50'] = ohlcv_indicators['high'].shift(1).rolling(window=50).max()
    ohlcv_indicators['dist_pivot_20'] = (
        ohlcv_indicators['close'] / ohlcv_indicators['pivot_high_20'] - 1
    )
    ohlcv_indicators['dist_pivot_50'] = (
        ohlcv_indicators['close'] / ohlcv_indicators['pivot_high_50'] - 1
    )

    ohlcv_indicators['volume_ratio'] = ohlcv_indicators['volume'] / ohlcv_indicators['Vol_MA20']
    ohlcv_indicators['dry_up_ratio'] = (
        ohlcv_indicators['volume'].rolling(window=3).mean() / ohlcv_indicators['Vol_MA20']
    )

    high_10 = ohlcv_indicators['high'].rolling(window=10).max()
    low_10 = ohlcv_indicators['low'].rolling(window=10).min()
    high_20 = ohlcv_indicators['high'].rolling(window=20).max()
    low_20 = ohlcv_indicators['low'].rolling(window=20).min()
    ohlcv_indicators['range_10_pct'] = (high_10 - low_10) / ohlcv_indicators['close']
    ohlcv_indicators['range_20_pct'] = (high_20 - low_20) / ohlcv_indicators['close']
    ohlcv_indicators['range_contraction'] = (
        ohlcv_indicators['range_10_pct'] < ohlcv_indicators['range_20_pct']
    )

    ohlcv_indicators['bullish_engulfing'] = _is_bullish_engulfing(ohlcv_indicators)
    ohlcv_indicators['reversal_180'] = _is_180_reversal(ohlcv_indicators)

    # S&P-relative returns are calculated by the service layer rather than the
    # OHLCV frame, so attach them before calculating the deterministic score.
    latest = ohlcv_indicators.iloc[-1].copy()
    prev = ohlcv_indicators.iloc[-2].copy()
    for metric in ("vs_market_1m", "vs_market_3m", "vs_market_6m"):
        latest[metric] = (market_metrics or {}).get(metric)
        prev[metric] = (market_metrics or {}).get(metric)
    recent_low_10 = ohlcv_indicators['low'].tail(10).min()
    recent_breakout_detail = _recent_breakout_evidence(ohlcv_indicators)
    recent_breakout = bool(recent_breakout_detail.get('confirmed'))
    chart_evidence = build_chart_evidence(ohlcv_indicators)
    base_entry_plan = _entry_plan(latest, 'near_breakout', recent_low_10, chart_evidence)
    entry_price = base_entry_plan.get('entry_price')
    risk_pct = base_entry_plan.get('raw_risk_pct')

    sma200 = _safe_float(latest.get('SMA_200'))
    ma_structure_bullish = (
        latest['close'] > latest['SMA_20'] > latest['SMA_50']
        and (sma200 is None or latest['close'] > sma200)
        and latest['SMA_10'] > latest['SMA_20']
    )
    healthy_rsi = 50 <= latest['RSI'] < 78
    pivot_breakout = (
        latest['close'] > latest['pivot_high_20']
        or latest['close'] > latest['pivot_high_50']
    )
    volume_confirm = latest['volume_ratio'] >= BREAKOUT_VOLUME_CONFIRM_RATIO
    stopping_signal = (
        latest['close'] > latest['open']
        or latest.get('MACDh', 0) > prev.get('MACDh', 0)
        or bool(latest.get('bullish_engulfing', False))
        or bool(latest.get('reversal_180', False))
    )
    extended_from_sma10 = (
        _safe_div(latest['close'] - latest['SMA_10'], latest['SMA_10'])
        >= EXTENDED_FROM_SMA10_PCT
    )
    extended_by_atr = (
        _safe_div(latest['close'] - latest['SMA_20'], latest['ATR'])
        >= EXTENDED_FROM_SMA20_ATR
    )
    close = _safe_float(latest.get('close'))
    dist_to_entry = _safe_div(close - entry_price, entry_price, default=None) if close is not None and entry_price else None
    breakout_score, breakout_score_parts, breakout_score_detail = _score_breakout(
        latest, entry_price, risk_pct, chart_evidence,
    )
    meta_score, meta_score_parts, meta_score_detail = _score_meta(
        latest, prev, recent_low_10, chart_evidence, risk_pct,
    )
    fresh_breakout_not_extended = (
        close is not None
        and entry_price is not None
        and close >= entry_price
        and close <= entry_price * (1 + BREAKOUT_CHASE_CEILING_PCT)
    )
    near_breakout = (
        breakout_score >= 65
        and close is not None
        and entry_price is not None
        and close < entry_price
        and dist_to_entry is not None
        and NEAR_BREAKOUT_MIN_DIST <= dist_to_entry < NEAR_BREAKOUT_MAX_DIST
        and not recent_breakout
    )
    extended_breakout = (
        close is not None
        and entry_price is not None
        and close > entry_price * (1 + BREAKOUT_CHASE_CEILING_PCT)
    )
    failed_breakout_detail = _failed_breakout_evidence(ohlcv_indicators, latest)
    failed_breakout = failed_breakout_detail['failed']
    fresh_breakout = (
        breakout_score >= 70
        and fresh_breakout_not_extended
        and volume_confirm
    )
    pullback_forming = (
        bool(chart_evidence.get('selected_meta_zone'))
        and meta_score >= 45
        and meta_score_parts['meta_confluence'] >= 16
        and meta_score_parts['reversal_signal'] < 6
    )
    pullback_entry = (
        bool(chart_evidence.get('selected_meta_zone'))
        and meta_score >= 55
        and meta_score_parts['meta_confluence'] >= 16
        and meta_score_parts['reversal_signal'] >= 6
    )
    breaks_support = (
        latest['close'] < latest['SMA_20']
        or latest['close'] < latest['SMA_50']
    )
    breakout_context = bool(failed_breakout_detail['breakout_context'])
    breakout_invalid = bool(breakout_context and (failed_breakout or breaks_support))
    validity_flags = {
        'holds_breakout_pivot': not bool(failed_breakout),
        'holds_short_term_support': not bool(breaks_support),
    }
    failure_flags = {
        'risk_too_wide': bool(risk_pct is not None and risk_pct > MAX_RISK_PCT + FLOAT_EPSILON),
        'long_term_trend_broken': bool(
            sma200 is not None and latest['close'] < sma200
        ),
    }
    validity_flag_details = {
        'holds_breakout_pivot': failed_breakout_detail,
        'holds_short_term_support': {
            'breakout_context': breakout_context,
            'latest_close': close,
            'sma20': _safe_float(latest.get('SMA_20')),
            'sma50': _safe_float(latest.get('SMA_50')),
            'prev_low': _safe_float(prev.get('low')),
            'volume_ratio': _safe_float(latest.get('volume_ratio')),
            'volume_threshold': FAILURE_VOLUME_RATIO,
            'below_sma20': bool(latest['close'] < latest['SMA_20']),
            'below_sma50': bool(latest['close'] < latest['SMA_50']),
        },
    }
    failure_flag_details = {
        'risk_too_wide': {
            'risk_pct': _safe_float(risk_pct),
            'risk_threshold': MAX_RISK_PCT,
        },
        'long_term_trend_broken': {
            'latest_close': close,
            'sma200': _safe_float(latest.get('SMA_200')),
        },
    }
    extension_flags = {
        'within_chase_limit': bool(
            close is not None
            and entry_price is not None
            and close <= entry_price * (1 + BREAKOUT_CHASE_CEILING_PCT)
        ),
        'extended_from_sma10': bool(extended_from_sma10),
        'extended_by_atr': bool(extended_by_atr),
    }
    extension_flag_details = {
        'within_chase_limit': {
            'latest_close': close,
            'entry_price': _safe_float(entry_price),
            'entry_ceiling': _safe_float(entry_price * (1 + BREAKOUT_CHASE_CEILING_PCT)) if entry_price else None,
            'ceiling_pct': BREAKOUT_CHASE_CEILING_PCT,
        },
        'extended_from_sma10': {
            'latest_close': close,
            'sma10': _safe_float(latest.get('SMA_10')),
            'distance_pct': _safe_div(latest['close'] - latest['SMA_10'], latest['SMA_10'], default=None),
            'threshold_pct': EXTENDED_FROM_SMA10_PCT,
        },
        'extended_by_atr': {
            'latest_close': close,
            'sma20': _safe_float(latest.get('SMA_20')),
            'atr': _safe_float(latest.get('ATR')),
            'distance_atr': _safe_div(latest['close'] - latest['SMA_20'], latest['ATR'], default=None),
            'threshold_atr': EXTENDED_FROM_SMA20_ATR,
        },
    }
    structural_failure = not all(validity_flags.values())
    major_failure = not validity_flags['holds_short_term_support']
    candidate_flags = {
        'fresh_breakout': bool(fresh_breakout),
        'near_breakout': bool(near_breakout and not structural_failure),
        'pullback_entry': bool(pullback_entry and not major_failure),
        'pullback_forming': bool(pullback_forming and not major_failure),
    }
    candidate_flag_details = {
        'fresh_breakout': {
            'score': breakout_score,
            'score_threshold': 70,
            'dist_to_entry': dist_to_entry,
            'distance_min': 0.0,
            'distance_max': BREAKOUT_CHASE_CEILING_PCT,
            'volume_ratio': _safe_float(latest.get('volume_ratio')),
            'volume_threshold': BREAKOUT_VOLUME_CONFIRM_RATIO,
            'extension_clear': extension_flags['within_chase_limit'],
        },
        'near_breakout': {
            'score': breakout_score,
            'score_threshold': 65,
            'dist_to_entry': dist_to_entry,
            'distance_min': NEAR_BREAKOUT_MIN_DIST,
            'distance_max': NEAR_BREAKOUT_MAX_DIST,
            'recent_breakout': bool(recent_breakout),
            'recent_breakout_detail': recent_breakout_detail,
            'failure_clear': not structural_failure,
        },
        'pullback_entry': {
            'score': meta_score,
            'score_threshold': 55,
            'meta_confluence': meta_score_parts['meta_confluence'],
            'meta_confluence_threshold': 16,
            'reversal_signal': meta_score_parts['reversal_signal'],
            'reversal_threshold': 6,
            'has_meta_zone': bool(chart_evidence.get('selected_meta_zone')),
            'major_failure_clear': not major_failure,
        },
        'pullback_forming': {
            'score': meta_score,
            'score_threshold': 45,
            'meta_confluence': meta_score_parts['meta_confluence'],
            'meta_confluence_threshold': 16,
            'reversal_signal': meta_score_parts['reversal_signal'],
            'reversal_threshold': 6,
            'has_meta_zone': bool(chart_evidence.get('selected_meta_zone')),
            'major_failure_clear': not major_failure,
        },
    }

    if breakout_invalid:
        setup_phase = 'failed_breakout'
    elif extended_breakout:
        setup_phase = 'extended_breakout'
    elif fresh_breakout:
        setup_phase = 'fresh_breakout'
    elif near_breakout and not structural_failure:
        setup_phase = 'near_breakout'
    elif pullback_entry and not major_failure:
        setup_phase = 'pullback_entry'
    elif pullback_forming and not major_failure:
        setup_phase = 'pullback_forming'
    else:
        setup_phase = 'unclear_structure'

    setup_status = _setup_status_for_phase(setup_phase)
    holding_action = _holding_action_for_phase(
        setup_phase,
        breaks_support=breaks_support,
        extended=extended_from_sma10 or extended_by_atr,
    )

    ohlcv_indicators = ohlcv_indicators.dropna().reset_index(drop=True)
    ohlcv_series = ohlcv_indicators.to_dict(orient='list')
    ohlcv_series['setup_phase'] = setup_phase
    ohlcv_series.update(_entry_plan(latest, setup_phase, recent_low_10, chart_evidence))
    ohlcv_series['setup_status'] = setup_status
    ohlcv_series['breakout_score'] = breakout_score
    ohlcv_series['meta_score'] = meta_score
    ohlcv_series['setup_evidence'] = {
        'breakout_score': breakout_score,
        'breakout_score_parts': breakout_score_parts,
        'breakout_score_detail': breakout_score_detail,
        'meta_score': meta_score,
        'meta_score_parts': meta_score_parts,
        'meta_score_detail': meta_score_detail,
        'chart_evidence': chart_evidence,
        'market_metrics': {
            metric: _safe_float(latest.get(metric))
            for metric in ("vs_market_1m", "vs_market_3m", "vs_market_6m")
        },
        'dist_to_entry': dist_to_entry,
        'raw_risk_pct': risk_pct,
        'failure_flags': failure_flags,
        'failure_flag_details': failure_flag_details,
        'validity_flags': validity_flags,
        'validity_flag_details': validity_flag_details,
        'breakout_context': breakout_context,
        'breakout_invalid': breakout_invalid,
        'extension_flags': extension_flags,
        'extension_flag_details': extension_flag_details,
        'candidate_flags': candidate_flags,
        'candidate_flag_details': candidate_flag_details,
    }
    ohlcv_series['setup_caption'] = _phase_caption(setup_phase)
    ohlcv_series['holding_action'] = holding_action
    ohlcv_series['status'] = setup_status

    return ohlcv_series
