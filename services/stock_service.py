import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from datetime import datetime, timedelta, timezone

from core.analysis.contracts import AnalysisSnapshot, RULESET_VERSION, ScreeningSettings
from core.analysis.explainability import build_decision_tree, build_rule_checklist, build_score_checklist, deterministic_summary, earnings_context, phase_score, rules_passed_text, score_breakdown
from core.analysis.momentum import run_tech_analysis
from repositories.storage import (
    load_ohlcv_cache,
    load_stock_cache,
    record_analysis_snapshot,
    save_ohlcv_cache,
    save_stock_cache,
)


SP500 = '^GSPC'
EXCHANGES = {'NMS', 'NGM', 'NYQ', 'ASE'}
OHLCV_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']


def _safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_earnings_timestamp(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (int, float)):
        return int(value) if value > 0 else None

    ts = pd.to_datetime(value, errors='coerce')
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return int(ts.timestamp())


def _to_utc_timestamp(value):
    ts = pd.to_datetime(value, errors='coerce')
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize('UTC')
    return ts.tz_convert('UTC')


def _earnings_timestamp_from_payload(payload):
    if not payload:
        return None

    for key in (
        'earningsTimestamp',
        'earningsTimestampStart',
        'earningsTimestampEnd',
        'earningsDate',
    ):
        timestamp = _coerce_earnings_timestamp(payload.get(key))
        if timestamp:
            return timestamp
    return None


def _earnings_timestamp_from_calendar(calendar):
    if calendar is None:
        return None

    candidates = []
    if isinstance(calendar, pd.DataFrame):
        for column in ('Earnings Date', 'Earnings High', 'Earnings Low'):
            if column in calendar.columns:
                candidates.extend(calendar[column].dropna().tolist())
    elif isinstance(calendar, dict):
        for key in ('Earnings Date', 'Earnings High', 'Earnings Low', 'earningsDate'):
            value = calendar.get(key)
            if isinstance(value, (list, tuple, pd.Series)):
                candidates.extend(value)
            else:
                candidates.append(value)
    else:
        return _coerce_earnings_timestamp(calendar)

    now = pd.Timestamp.now(tz='UTC').normalize()
    parsed = []
    for candidate in candidates:
        ts = _to_utc_timestamp(candidate)
        if ts is not None:
            parsed.append(ts)

    if not parsed:
        return None

    future = [ts for ts in parsed if ts.normalize() >= now]
    selected = min(future) if future else max(parsed)
    return int(selected.timestamp())


def _fetch_earnings_timestamp(ticker, info):
    timestamp = _earnings_timestamp_from_payload(info)
    if timestamp:
        return timestamp

    try:
        timestamp = _earnings_timestamp_from_calendar(ticker.calendar)
        if timestamp:
            return timestamp
    except Exception:
        pass

    try:
        earnings_dates = ticker.get_earnings_dates(limit=8)
        if earnings_dates is not None and not earnings_dates.empty:
            index = pd.to_datetime(earnings_dates.index, errors='coerce')
            index = index[~pd.isna(index)]
            if len(index):
                now = pd.Timestamp.now(tz='UTC').normalize()
                parsed = [_to_utc_timestamp(ts) for ts in index]
                parsed = [ts for ts in parsed if ts is not None]
                if not parsed:
                    return None
                future = [ts for ts in parsed if ts.normalize() >= now]
                selected = min(future) if future else max(parsed)
                return int(selected.timestamp())
    except Exception:
        pass

    return None


def _market_profile(symbol, exchange=None):
    symbol = symbol.upper()
    exchange = (exchange or '').upper()

    if symbol.endswith('.HK') or exchange in {'HKG', 'HKEX', 'XHKG'}:
        return {
            'calendar': 'HKEX',
            'timezone': 'Asia/Hong_Kong',
            'market': 'HK',
        }

    return {
        'calendar': 'NYSE',
        'timezone': 'America/New_York',
        'market': 'US',
    }


def _market_clock(symbol, exchange=None, now_utc=None):
    profile = _market_profile(symbol, exchange=exchange)
    now_utc = now_utc or datetime.now(timezone.utc)
    calendar = mcal.get_calendar(profile['calendar'])
    start_date = (now_utc - timedelta(days=14)).date()
    end_date = (now_utc + timedelta(days=2)).date()
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)

    if schedule.empty:
        return {
            **profile,
            'is_open': False,
            'expected_trading_date': None,
            'now_utc': now_utc,
        }

    now_ts = pd.Timestamp(now_utc)
    opened_sessions = schedule[schedule['market_open'] <= now_ts]
    if opened_sessions.empty:
        expected_trading_date = schedule.index[0].date()
    else:
        expected_trading_date = opened_sessions.index[-1].date()

    current_session = schedule[
        (schedule['market_open'] <= now_ts)
        & (schedule['market_close'] > now_ts)
    ]

    return {
        **profile,
        'is_open': not current_session.empty,
        'expected_trading_date': expected_trading_date,
        'now_utc': now_utc,
    }


def _normalize_ohlcv(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = df.copy()
    df = df[[col for col in OHLCV_COLUMNS if col in df.columns]]
    df = df.dropna(how='all', subset=[col for col in OHLCV_COLUMNS if col in df.columns])
    df.index.name = 'Date'
    return df


def _close_by_trading_date(ohlcv):
    if ohlcv is None or ohlcv.empty or 'Close' not in ohlcv.columns:
        return pd.Series(dtype='float64')

    close = ohlcv['Close'].dropna().copy()
    close.index = pd.Index([idx.date() for idx in close.index], name='Date')
    close = close[~close.index.duplicated(keep='last')]
    return close


def _fetch_history(symbol, period='18mo'):
    history = yf.Ticker(symbol).history(period=period, auto_adjust=False)
    return _normalize_ohlcv(history)


def _latest_complete_trading_date(ohlcv):
    if ohlcv is None or ohlcv.empty:
        return None

    complete = ohlcv.dropna(subset=OHLCV_COLUMNS)
    if complete.empty:
        return None

    return complete.index[-1].date()


def _merge_ohlcv(existing, recent):
    frames = [df for df in [existing, recent] if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep='last')]
    return merged


def _fast_info_latest_bar(symbol, trading_date, profile):
    fast_info = yf.Ticker(symbol).fast_info
    open_price = _safe_float(fast_info.get('open') if hasattr(fast_info, 'get') else fast_info['open'])
    high = _safe_float(fast_info.get('dayHigh') if hasattr(fast_info, 'get') else fast_info['dayHigh'])
    low = _safe_float(fast_info.get('dayLow') if hasattr(fast_info, 'get') else fast_info['dayLow'])
    close = _safe_float(fast_info.get('lastPrice') if hasattr(fast_info, 'get') else fast_info['lastPrice'])
    volume = _safe_float(fast_info.get('lastVolume') if hasattr(fast_info, 'get') else fast_info['lastVolume'])

    if min(open_price, high, low, close) <= 0:
        return None

    index = pd.Timestamp(trading_date).tz_localize(profile['timezone'])
    return pd.DataFrame(
        [{
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume,
        }],
        index=pd.DatetimeIndex([index], name='Date'),
    )


def _patch_latest_bar(symbol, ohlcv, clock):
    expected_date = clock.get('expected_trading_date')
    if expected_date is None:
        return ohlcv

    latest_complete_date = _latest_complete_trading_date(ohlcv)
    expected_rows = ohlcv[ohlcv.index.date == expected_date] if not ohlcv.empty else pd.DataFrame()
    expected_row_incomplete = (
        not expected_rows.empty
        and expected_rows.tail(1)[OHLCV_COLUMNS].isna().any(axis=None)
    )

    if latest_complete_date is not None and latest_complete_date >= expected_date and not expected_row_incomplete:
        return ohlcv

    latest_bar = _fast_info_latest_bar(symbol, expected_date, clock)
    if latest_bar is None:
        return ohlcv

    return _merge_ohlcv(ohlcv, latest_bar)


def get_ohlcv(symbol, exchange=None, force_refresh=False):
    symbol = symbol.upper().strip()
    clock = _market_clock(symbol, exchange=exchange)
    cached = load_ohlcv_cache(symbol)
    cached_ohlcv = cached.get('data') if cached else None
    cached_at = cached.get('updated_at') if cached else None

    refresh_recent = force_refresh or cached_ohlcv is None or cached_ohlcv.empty
    if cached_at and not refresh_recent:
        try:
            cached_dt = datetime.fromisoformat(cached_at)
            age_minutes = (datetime.now() - cached_dt).total_seconds() / 60
            max_age_minutes = 5 if clock['is_open'] else 360
            refresh_recent = age_minutes > max_age_minutes
        except (TypeError, ValueError):
            refresh_recent = True

    try:
        if cached_ohlcv is None or cached_ohlcv.empty:
            ohlcv = _fetch_history(symbol, period='18mo')
        elif refresh_recent:
            recent = _fetch_history(symbol, period='1mo')
            ohlcv = _merge_ohlcv(cached_ohlcv, recent)
        else:
            ohlcv = cached_ohlcv
    except Exception:
        if cached_ohlcv is None or cached_ohlcv.empty:
            raise
        ohlcv = cached_ohlcv

    try:
        ohlcv = _patch_latest_bar(symbol, ohlcv, clock)
    except Exception:
        # A failed live quote must not discard a usable completed-session cache.
        ohlcv = cached_ohlcv if cached_ohlcv is not None and not cached_ohlcv.empty else ohlcv
    save_ohlcv_cache(symbol, ohlcv, _market_profile(symbol, exchange=exchange))
    return ohlcv


def _get_symbol_metadata(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    # Some Yahoo quote payloads omit sector/industry even though the asset
    # profile still contains them. Keep this fallback narrow and metadata-only.
    try:
        profile = getattr(ticker, 'asset_profile', None) or {}
    except Exception:
        profile = {}

    price = _safe_float(
        info.get('currentPrice')
        or info.get('regularMarketPrice')
        or info.get('previousClose')
    )
    ma50 = _safe_float(info.get('fiftyDayAverage'))
    ma200 = _safe_float(info.get('twoHundredDayAverage'))
    avgvol3m = _safe_float(
        info.get('averageVolume')
        or info.get('averageDailyVolume3Month')
        or info.get('averageVolume10days')
    )

    return {
        'symbol': symbol,
        'name': info.get('shortName') or info.get('longName') or symbol,
        'exchange': info.get('exchange', '未知'),
        'financial_currency': info.get('financialCurrency', '未知'),
        'price': price,
        'ma50': ma50,
        'ma200': ma200,
        'market_cap': _safe_float(info.get('marketCap')),
        '1y_pct_change': _safe_float(info.get('52WeekChange')),
        'pct_from_high': _safe_float(info.get('fiftyTwoWeekHighChangePercent')),
        'earnings_date': _fetch_earnings_timestamp(ticker, info),
        'is_earnings_estimate': info.get('isEarningsDateEstimate', True),
        'sector_name': info.get('sector') or info.get('sectorDisp') or profile.get('sector') or '未知',
        'industry_name': info.get('industry') or info.get('industryDisp') or profile.get('industry') or '未知',
        'avgvol3m': avgvol3m,
    }


def _cached_symbol_metadata(symbol):
    """Return the last known metadata without pretending it came from Yahoo now."""
    cached = load_stock_cache(symbol)
    if not cached:
        return None
    keys = (
        'symbol', 'name', 'exchange', 'financial_currency', 'price', 'ma50', 'ma200',
        'market_cap', '1y_pct_change', 'pct_from_high', 'earnings_date',
        'is_earnings_estimate', 'sector_name', 'industry_name', 'avgvol3m',
    )
    metadata = {key: cached.get(key) for key in keys}
    metadata['symbol'] = symbol
    required = ('exchange', 'price', 'ma50', 'ma200', 'market_cap', 'avgvol3m')
    return metadata if all(metadata.get(key) is not None for key in required) else None


def _passes_quote_gate(meta, settings=None):
    settings = settings or ScreeningSettings()
    price = _safe_float(meta.get('price'))
    market_cap = _safe_float(meta.get('market_cap'))
    avgvol3m = _safe_float(meta.get('avgvol3m'))
    ma50 = _safe_float(meta.get('ma50'))
    ma200 = _safe_float(meta.get('ma200'))
    pct_from_high = _safe_float(meta.get('pct_from_high'))
    numeric_values = (price, market_cap, avgvol3m, ma50, ma200, pct_from_high)
    if any(value is None for value in numeric_values):
        return False
    return (
        meta.get('exchange') in EXCHANGES
        and market_cap >= settings.market_cap_min
        and avgvol3m >= settings.avg_volume_3m_min
        and price >= settings.price_min
        and (not settings.require_price_above_ma_stack or price >= ma50 >= ma200)
        and avgvol3m * price >= settings.dollar_volume_min
        and pct_from_high >= -settings.max_distance_52w_high
    )


def _relative_strength_metrics(close_prices, symbol, settings=None):
    settings = settings or ScreeningSettings()
    sp500_close = close_prices[SP500].dropna()
    symbol_close = close_prices[symbol].dropna()

    if len(sp500_close) < 200 or len(symbol_close) < 200:
        return {
            'passes_stage2': False,
            'vs_market_1m': None,
            'vs_market_3m': None,
            'vs_market_6m': None,
            'stock_pct_1m': None,
            'stock_pct_3m': None,
            'stock_pct_6m': None,
            'sp500_pct_1m': None,
            'sp500_pct_3m': None,
            'sp500_pct_6m': None,
            'ma200_is_trending_up': None,
        }

    sp500_pct_1m = (sp500_close.iloc[-1] / sp500_close.iloc[-21]) - 1
    sp500_pct_3m = (sp500_close.iloc[-1] / sp500_close.iloc[-63]) - 1
    sp500_pct_6m = (sp500_close.iloc[-1] / sp500_close.iloc[-126]) - 1

    ma200_history = symbol_close.rolling(window=200).mean()
    ma200_is_trending_up = ma200_history.iloc[-1] > ma200_history.iloc[-6]

    ma10 = symbol_close.rolling(window=10).mean().iloc[-1]
    ma20 = symbol_close.rolling(window=20).mean().iloc[-1]

    pct_1m = (symbol_close.iloc[-1] / symbol_close.iloc[-21]) - 1
    pct_3m = (symbol_close.iloc[-1] / symbol_close.iloc[-63]) - 1
    pct_6m = (symbol_close.iloc[-1] / symbol_close.iloc[-126]) - 1

    passes_stage2 = (
        (not settings.require_sma10_above_sma20 or ma10 > ma20)
        and (not settings.require_ma200_rising or ma200_is_trending_up)
        and (not settings.require_vs_market_1m or pct_1m > sp500_pct_1m)
        and (not settings.require_vs_market_3m or pct_3m > sp500_pct_3m)
        and (not settings.require_vs_market_6m or pct_6m > sp500_pct_6m)
    )
    return {
        'passes_stage2': passes_stage2,
        'vs_market_1m': pct_1m - sp500_pct_1m,
        'vs_market_3m': pct_3m - sp500_pct_3m,
        'vs_market_6m': pct_6m - sp500_pct_6m,
        'stock_pct_1m': pct_1m,
        'stock_pct_3m': pct_3m,
        'stock_pct_6m': pct_6m,
        'sp500_pct_1m': sp500_pct_1m,
        'sp500_pct_3m': sp500_pct_3m,
        'sp500_pct_6m': sp500_pct_6m,
        'ma200_is_trending_up': bool(ma200_is_trending_up),
    }


def _passes_stage2_gate(close_prices, symbol, settings=None):
    return _relative_strength_metrics(close_prices, symbol, settings)['passes_stage2']


def completed_daily_ohlcv(ohlcv, symbol, exchange=None):
    """Exclude an in-progress session so rules never depend on live patches."""
    daily = _normalize_ohlcv(ohlcv)
    if daily.empty:
        return daily
    clock = _market_clock(symbol, exchange=exchange)
    expected = clock.get('expected_trading_date')
    if clock.get('is_open') and expected and daily.index[-1].date() == expected:
        daily = daily.iloc[:-1].copy()
    return daily.dropna(subset=OHLCV_COLUMNS)


def repair_completed_session_close(ohlcv, expected_date, fallback_close):
    """Fill a missing completed-session close only from a plausible quote price."""
    daily = _normalize_ohlcv(ohlcv)
    if daily.empty or expected_date is None:
        return daily

    try:
        expected = pd.Timestamp(expected_date).date()
    except (TypeError, ValueError):
        return daily

    matching_index = [index for index in daily.index if index.date() == expected]
    if not matching_index:
        return daily

    index = matching_index[-1]
    row = daily.loc[index]
    if pd.notna(row.get('Close')):
        return daily
    if any(pd.isna(row.get(column)) for column in ('Open', 'High', 'Low', 'Volume')):
        return daily

    close = _safe_float(fallback_close)
    low = _safe_float(row.get('Low'))
    high = _safe_float(row.get('High'))
    if close <= 0 or low <= 0 or high <= 0 or not low <= close <= high:
        return daily

    daily.loc[index, 'Close'] = close
    return daily


def completed_session_closes_from_intraday(intraday, expected_date):
    """Return the final regular-session minute close for each symbol."""
    if intraday is None or intraday.empty or expected_date is None:
        return {}

    try:
        expected = pd.Timestamp(expected_date).date()
    except (TypeError, ValueError):
        return {}

    if isinstance(intraday.columns, pd.MultiIndex):
        if 'Close' not in intraday.columns.get_level_values(0):
            return {}
        closes = intraday['Close']
    elif 'Close' in intraday.columns:
        closes = intraday[['Close']]
    else:
        return {}

    timestamps = pd.DatetimeIndex(closes.index)
    if timestamps.tz is not None:
        trading_dates = timestamps.tz_convert('America/New_York').date
    else:
        trading_dates = timestamps.date
    session_closes = closes.loc[trading_dates == expected]
    if session_closes.empty:
        return {}

    results = {}
    for symbol in session_closes.columns:
        values = pd.to_numeric(session_closes[symbol], errors='coerce').dropna()
        if not values.empty and float(values.iloc[-1]) > 0:
            results[str(symbol)] = float(values.iloc[-1])
    return results


def build_analyzed_stock(meta, ohlcv, setup_status_override=None, status_override=None, updated_at=None, settings=None, data_date=None, record_history=False):
    settings = settings or ScreeningSettings()
    analysis = run_tech_analysis(ohlcv, market_metrics=meta)
    override = setup_status_override or status_override
    if override is not None:
        analysis['setup_status'] = override
        analysis['status'] = override
        if override == 'not_recommended':
            analysis['setup_phase'] = 'not_recommended'
            analysis['setup_caption'] = '未符合基本篩選或技術條件，不宜買入。'
            analysis['entry_note'] = '未通過報價或 Stage 2 條件；暫時沒有可行動的進場參考。'

    stock = {**dict(meta), **analysis}
    if analysis.get('close'):
        stock['price'] = analysis['close'][-1]
    stock['updated_at'] = updated_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    stock['data_date'] = str(data_date or (ohlcv.index[-1].date() if not ohlcv.empty else 'unknown'))
    stock['data_source'] = f"Yahoo Finance 日線資料，截至 {stock['data_date']} 收市"
    stock['ruleset_version'] = RULESET_VERSION
    stock['settings_fingerprint'] = settings.fingerprint
    stock['settings'] = settings.to_dict()
    stock['deterministic_summary'] = deterministic_summary(stock)
    score, score_label = phase_score(stock)
    stock['mvp_score'] = score
    stock['score_label'] = score_label
    stock['score_breakdown'] = score_breakdown(stock)
    stock['score_basis'] = stock['score_breakdown']['basis']
    stock['mvp_score_possible'] = stock['score_breakdown']['possible']
    stock['score_display'] = stock['score_breakdown']['score_display']
    base_rules = build_rule_checklist(stock, settings)
    stock['rule_checklist'] = [
        *base_rules,
        *build_decision_tree(stock, settings),
        *build_score_checklist(stock),
    ]
    stock['earnings_context'] = earnings_context(stock)
    stock['earnings_warning'] = stock['earnings_context']['warning']
    stock['days_to_earnings'] = stock['earnings_context']['days']
    stock['rules_passed'] = rules_passed_text(stock['rule_checklist'])
    if record_history:
        snapshot = AnalysisSnapshot(
            symbol=stock['symbol'], classification=stock['setup_phase'], setup_status=stock['setup_status'],
            score=score, score_label=score_label, data_date=stock['data_date'],
            settings_fingerprint=settings.fingerprint, ruleset_version=RULESET_VERSION,
            summary=stock['deterministic_summary'], rules=stock['rule_checklist'],
        ).to_dict()
        record_analysis_snapshot(snapshot)
    return stock


def analyze_single_stock(symbol, settings=None, cached_metadata=None):
    settings = settings or ScreeningSettings()
    symbol = symbol.upper().strip()
    metadata_stale = False
    try:
        meta = _get_symbol_metadata(symbol)
    except Exception:
        meta = cached_metadata or _cached_symbol_metadata(symbol)
        if meta is None:
            raise
        meta = dict(meta)
        meta['symbol'] = symbol
        metadata_stale = True

    ohlcv = get_ohlcv(symbol, exchange=meta.get('exchange'))
    daily_ohlcv = completed_daily_ohlcv(ohlcv, symbol, meta.get('exchange'))
    sp500_ohlcv = completed_daily_ohlcv(get_ohlcv(SP500), SP500)
    close_prices = pd.DataFrame({
        symbol: _close_by_trading_date(daily_ohlcv),
        SP500: _close_by_trading_date(sp500_ohlcv),
    }).dropna()

    if daily_ohlcv.empty:
        raise ValueError(f'No price history found for {symbol}.')

    vs_market_metrics = _relative_strength_metrics(close_prices, symbol, settings)
    meta.update(vs_market_metrics)
    meta.pop('passes_stage2', None)

    setup_status_override = None
    if not (_passes_quote_gate(meta, settings) and vs_market_metrics['passes_stage2']):
        setup_status_override = 'not_recommended'

    stock = build_analyzed_stock(
        meta, daily_ohlcv, setup_status_override=setup_status_override, settings=settings,
        # Preserve the last known classification when Yahoo is unavailable rather
        # than appending a newly inferred state from stale metadata.
        data_date=daily_ohlcv.index[-1].date(), record_history=not metadata_stale,
    )
    if metadata_stale:
        stock['data_stale'] = True
        stock['refresh_error'] = 'Yahoo metadata refresh failed; showing the last successful cached daily snapshot.'
    return stock


def refresh_bookmarked_stocks(bookmarks, force_refresh=False, batch_updated_at=None, settings=None):
    settings = settings or ScreeningSettings()
    if bookmarks is None or bookmarks.empty:
        return pd.DataFrame()

    refreshed = []
    for _, row in bookmarks.iterrows():
        symbol = str(row.get('symbol', '')).upper().strip()
        if not symbol:
            continue

        meta = {
            key: row.get(key)
            for key in [
                'symbol', 'name', 'exchange', 'financial_currency', 'price',
                'ma50', 'ma200', 'market_cap', '1y_pct_change', 'pct_from_high',
                'earnings_date', 'is_earnings_estimate', 'sector_name', 'industry_name',
                'vs_market_1m', 'vs_market_3m', 'vs_market_6m',
                'stock_pct_1m', 'stock_pct_3m', 'stock_pct_6m',
                'sp500_pct_1m', 'sp500_pct_3m', 'sp500_pct_6m',
                'avgvol3m',
            ]
            if key in row
        }
        meta['symbol'] = symbol
        exchange = meta.get('exchange')
        try:
            stock = analyze_single_stock(symbol, settings=settings, cached_metadata=meta)
            if stock.get('data_stale'):
                # A failed refresh must not silently create a new setup state or signal.
                stock = row.to_dict()
                stock['data_stale'] = True
                stock['refresh_error'] = 'Yahoo refresh failed; showing the last successful cached daily snapshot.'
            else:
                stock['updated_at'] = batch_updated_at or stock['updated_at']
                stock['refresh_error'] = None
                stock['data_stale'] = False
                save_stock_cache(symbol, stock, source='bookmark_refresh')
        except Exception as e:
            stock = row.to_dict()
            cached_closes = stock.get('close')
            has_cached_prices = isinstance(cached_closes, (list, tuple)) and len(cached_closes) > 0
            if has_cached_prices or not pd.isna(stock.get('price')):
                stock['refresh_error'] = str(e)
                stock['data_stale'] = True
            else:
                stock['refresh_error'] = str(e)
        refreshed.append(stock)

    return pd.DataFrame(refreshed)
