import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import warnings
import html
import hashlib
import json
import os
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from native_stock_chart import render_native_stock_chart
from core.analysis.chart_evidence import detect_priority_pivot_ranges
from core.analysis.explainability import deterministic_summary
from core.i18n.sector_industry_zh import translate_industry, translate_sector

COLOR_BULL = 'rgba(38,166,154,0.9)'  #26a69a
COLOR_BEAR = 'rgba(239,83,80,0.9)'  #ef5350
CHART_RANGES = {
    '1 個月': 18,
    '3 個月': 8,
    '6 個月': 4,
    '全部': None,
}
CHART_VISIBLE_BARS = {
    '1 個月': 21,
    '3 個月': 63,
    '6 個月': 126,
    '全部': None,
}
TRENDLINE_COLORS = ['#ff6b6b', '#50e3c2', '#b8e986']
TRENDLN_METHOD_OPTIONS = {
    'Fast scan (NSQUREDLOGN)': 'METHOD_NSQUREDLOGN',
    'Exhaustive (NCUBED)': 'METHOD_NCUBED',
    'Hough points': 'METHOD_HOUGHPOINTS',
    'Hough lines': 'METHOD_HOUGHLINES',
    'Probabilistic Hough': 'METHOD_PROBHOUGH',
}
DEFAULT_TRENDLN_METHOD_LABEL = 'Fast scan (NSQUREDLOGN)'
YOLO_PATTERN_MODEL = 'foduucom/stockmarket-pattern-detection-yolov8'
YOLO_CONFIDENCE_THRESHOLD = 0.30
YOLO_PATTERN_LOOKBACK_BARS = 260
YOLO_PATTERN_HELP = (
    '目前形態模型只作提示，可辨識部分常見形態，例如雙底、雙頂、頭肩底、三角形、楔形等；'
    '暫未可靠辨識 VCP 或 Cup With Handle。'
)
YOLO_PATTERN_LABELS = {
    'w_bottom': '雙底',
    'w bottom': '雙底',
    'w-bottom': '雙底',
    'm_head': '雙頂',
    'm head': '雙頂',
    'm-head': '雙頂',
    'head and shoulders bottom': '頭肩底',
    'head_shoulders_bottom': '頭肩底',
    'head-and-shoulders bottom': '頭肩底',
    'head and shoulders top': '頭肩頂',
    'head_shoulders_top': '頭肩頂',
    'head-and-shoulders top': '頭肩頂',
    'head and shoulder top': '頭肩頂',
    'head_shoulder_top': '頭肩頂',
    'head-shoulder top': '頭肩頂',
    'h&s top': '頭肩頂',
    'hs top': '頭肩頂',
    'm_top': '雙頂',
    'm top': '雙頂',
    'm-top': '雙頂',
    'triangle': '三角形',
    'ascending triangle': '上升三角形',
    'ascending_triangle': '上升三角形',
    'descending triangle': '下降三角形',
    'descending_triangle': '下降三角形',
    'wedge': '楔形',
    'bull flag': '看漲旗形',
    'bear flag': '看跌旗形',
}
EXPERIMENT_OUTPUT_ROOT = Path(__file__).resolve().parent / 'data' / 'experiments'
os.environ.setdefault('MPLCONFIGDIR', str(EXPERIMENT_OUTPUT_ROOT / '.matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(EXPERIMENT_OUTPUT_ROOT / '.cache'))
os.environ.setdefault('YOLO_CONFIG_DIR', str(EXPERIMENT_OUTPUT_ROOT / '.ultralytics'))
YFINANCE_CACHE_DIR = Path(__file__).resolve().parent / 'data' / 'yfinance_cache'
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))
    yf.cache.set_cache_location(str(YFINANCE_CACHE_DIR))
except Exception:
    pass

def clean_yf_df(df):
    df = df.reset_index()
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = df.dropna()
    return df

@st.cache_data(ttl=60, show_spinner=False)
def get_live_price(ticker):
    return yf.Ticker(ticker).fast_info['lastPrice']


def is_us_market_open(now=None):
    now = now or datetime.now(ZoneInfo("America/New_York"))
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return now.weekday() < 5 and market_open <= now < market_close


def get_live_card_data(row, ticker):
    closes = list(row.get("close", []))
    dates = list(row.get("date", []))

    if len(closes) < 2:
        raise ValueError(f"Not enough price history for {ticker}.")

    latest_close = float(closes[-1])
    previous_close = float(closes[-2])

    if is_us_market_open():
        try:
            live_price = float(get_live_price(ticker))
        except Exception:
            live_price = latest_close
        today_str = datetime.now(ZoneInfo("America/New_York")).strftime('%Y-%m-%d')
        latest_historical_date = dates[-1] if dates else None
        chart_data = list(closes[-20:-1]) + [live_price] if latest_historical_date == today_str else list(closes[-19:]) + [live_price]
        metric_price = live_price
    else:
        metric_price = latest_close
        chart_data = list(closes[-20:])

    chart_data = [round(float(value), 2) for value in chart_data]
    pct_change = (metric_price - previous_close) / previous_close * 100 if previous_close else 0

    return metric_price, pct_change, chart_data, "vs 上一收市價"


def _label_setup_status(status):
    return {
        'breakout_buy': ('空倉：留意突破', '🚀', 'green'),
        'breakout': ('空倉：留意突破', '🚀', 'green'),
        'pullback_wait': ('空倉：留意META', '🛡️', 'violet'),
        'pullback_buy': ('空倉：留意META', '🛡️', 'violet'),
        'pullback': ('空倉：留意META', '🛡️', 'violet'),
        'watchlist': ('空倉：觀望後續', '👀', 'blue'),
        'not_recommended': ('空倉：不宜買入', '❌', 'grey'),
    }.get(status, ('空倉：觀望後續', '👀', 'blue'))


def normalize_setup_status(status=None, phase=None):
    if status == 'not_recommended':
        return status
    if phase in {'near_breakout', 'fresh_breakout'}:
        return 'breakout_buy'
    if phase in {'pullback_forming', 'pullback_entry'}:
        return 'pullback_wait'
    if status in {None, '', 'None'}:
        return 'watchlist'
    return status


def normalize_holding_action(action=None, phase=None):
    if action == 'sell_strength':
        return 'hold'
    if action == 'add' and phase in {'near_breakout', 'pullback_forming'}:
        return 'hold'
    return action if action in {'add', 'sell_weakness', 'hold'} else 'hold'


def setup_phase_display_text(phase):
    phase_badge = _label_setup_phase(phase)
    if not phase_badge:
        return '❔ 趨勢：未形成趨勢'
    label, icon, _ = phase_badge
    return f"{icon} 趨勢：{label}"


def setup_phase_badge(phase):
    return _label_setup_phase(phase) or ('未形成趨勢', '❔', 'grey')


def get_vs_market_value(row, period):
    return row.get(f'vs_market_{period}', row.get(f'rs_{period}'))


def _label_holding_action(action):
    return {
        'add': ('持倉：考慮加倉', '➕', 'green'),
        'sell_weakness': ('持倉：趁弱賣出', '⚠️', 'red'),
        'hold': ('持倉：繼續持有', '✅', 'blue'),
    }.get(action, ('持倉：繼續持有', '✅', 'blue'))


def _label_setup_phase(phase):
    return {
        'near_breakout': ('接近突破', '🟡', 'orange'),
        'fresh_breakout': ('確認突破', '🟢', 'green'),
        'extended_breakout': ('突破已延伸', '🔼', 'orange'),
        'pullback_forming': ('等待回調', '⌛', 'violet'),
        'pullback_entry': ('確認回調', '🎯', 'green'),
        'failed_breakout': ('突破失敗', '😭', 'red'),
        'unclear_structure': ('結構未清晰', '❔', 'grey'),
        'not_recommended': ('不宜買入', '❌', 'grey'),
        'watch': ('未形成趨勢', '❔', 'grey'),
    }.get(phase)


def _format_price(value):
    if value is None or pd.isna(value):
        return "未有"
    return f"${float(value):,.2f}"


def _format_percent(value):
    if value is None or pd.isna(value):
        return "未有"
    return f"{float(value):.1%}"


def _safe_float_value(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_numeric_value(row, key):
    value = row.get(key)
    if isinstance(value, (list, tuple, pd.Series)):
        value = value[-1] if len(value) else None
    return _safe_float_value(value)


def _trade_plan_reference_values(row, current_price):
    pivot = None
    pivot_source = None
    for key, label in (("pivot_high_20", "前 20 日樞紐"), ("pivot_high_50", "前 50 日樞紐")):
        value = _latest_numeric_value(row, key)
        if value is not None and value > 0:
            pivot = value
            pivot_source = label
            break

    close = _safe_float_value(current_price) or _latest_numeric_value(row, "close")
    support_candidates = []
    for key, label in (("SMA_10", "10MA"), ("SMA_20", "20MA")):
        value = _latest_numeric_value(row, key)
        if value is not None:
            support_candidates.append((label, value))

    lows = row.get("low")
    if isinstance(lows, (list, tuple, pd.Series)) and len(lows):
        recent_lows = [_safe_float_value(value) for value in list(lows)[-10:]]
        recent_lows = [value for value in recent_lows if value is not None]
        if recent_lows:
            support_candidates.append(("近 10 日低位", min(recent_lows)))

    source_labels = {
        "MA 10 / MA 20": "10MA／20MA 匯聚區",
        "MA 50": "50MA 區",
        "20d swing low": "前 20 日波段低位",
        "Prior 20d pivot held": "前 20 日樞紐轉支持",
        "Gap Fill": "裂口支持區",
        "20d compact base": "20 日緊密底部",
    }

    def _display_source(source):
        parts = [source_labels.get(part.strip(), part.strip()) for part in str(source or "").split("+")]
        return " + ".join(part for part in parts if part) or "價格支持區"

    setup_evidence = row.get("setup_evidence")
    chart_evidence = setup_evidence.get("chart_evidence", {}) if isinstance(setup_evidence, dict) else {}
    for zone in chart_evidence.get("meta_candidates", []):
        value = _safe_float_value(zone.get("high"))
        if value is not None:
            source = str(zone.get("source") or "價格支持區")
            support_candidates.append((_display_source(source), value))

    valid_supports = [
        candidate for candidate in support_candidates
        if close is not None and 0 < candidate[1] < close
    ]
    support_source, support = max(valid_supports, key=lambda candidate: candidate[1]) if valid_supports else (None, None)
    return {
        "pivot": pivot,
        "pivot_source": pivot_source,
        "support": support,
        "support_source": support_source,
        "atr": _latest_numeric_value(row, "ATR"),
    }


def _metric_help_context(row, current_price, entry_price, stop_price, risk_pct):
    phase = row.get('setup_phase')
    risk_capped = bool(row.get('risk_capped', False))
    references = _trade_plan_reference_values(row, current_price)

    if phase in {'near_breakout', 'fresh_breakout'}:
        pivot = references["pivot"]
        entry_source = (
            f"{references['pivot_source']} {_format_price(pivot)} × 1.005（+0.5%）"
            if pivot is not None else "前期樞紐／阻力位 + 0.5% 緩衝"
        )
        setup_label = "突破區"
    elif phase in {'pullback_forming', 'pullback_entry'}:
        entry_source = "META／短線支持區附近"
        setup_label = "META 區"
    elif phase == 'extended_breakout':
        entry_source = "突破已過度延伸；現價只作觀察參考，不代表可追價入場。"
        setup_label = "等待區"
    elif phase == 'failed_breakout':
        entry_source = "突破結構已失效；現價只作觀察參考，暫不提供新入場設定。"
        setup_label = "失效區"
    else:
        entry_source = "未有可行動進場結構"
        setup_label = "觀察區"

    support = references["support"]
    atr = references["atr"]
    if phase in {'near_breakout', 'fresh_breakout', 'pullback_forming', 'pullback_entry'}:
        stop_source = (
            f"最近支持 {_format_price(support)}（{references['support_source']}）"
            f" - 0.25 × ATR {_format_price(atr)}"
            if support is not None and atr is not None
            else "最近支持位 - 0.25 ATR 緩衝"
        )
    else:
        stop_source = "非可行動設定；只顯示相對參考價低 8% 的風險下限。"

    if risk_capped:
        stop_source = "原始風險超過 8%，已套用上限"

    if current_price is not None and entry_price is not None and not pd.isna(current_price) and not pd.isna(entry_price):
        distance = (float(current_price) - float(entry_price)) / float(entry_price)
        if phase in {'extended_breakout', 'failed_breakout'}:
            current_detail = "現價只作狀態觀察，不代表可行動入場價。"
        elif distance >= 0:
            current_detail = f"現價高於入場參考 {_format_percent(distance)}"
        else:
            current_detail = f"現價距入場參考 {_format_percent(abs(distance))}"
    else:
        current_detail = "價格資料不足"

    if entry_price is not None and stop_price is not None and not pd.isna(entry_price) and not pd.isna(stop_price):
        risk_formula = f"({_format_price(entry_price)} - {_format_price(stop_price)}) / {_format_price(entry_price)}"
    else:
        risk_formula = "等待入場及止損參考"

    return setup_label, [
        ("入場參考", _format_price(entry_price), entry_source),
        ("止損參考", _format_price(stop_price), stop_source),
        ("風險距離", _format_percent(risk_pct), risk_formula),
        ("目前位置", _format_price(current_price), current_detail),
    ]


def _trade_plan_price_lines(row, dates, current_price=None):
    phase = row.get('setup_phase')
    # Breakout confirmation is already the upper edge of the breakout box.
    confirmation = row.get('confirmation_price') if phase == 'pullback_entry' else None
    confirmation_label = (
        "確認突破價"
        if row.get('setup_phase') in {'near_breakout', 'fresh_breakout'}
        else "反轉確認價"
        if row.get('setup_phase') in {'pullback_entry'}
        else "觀察"
    )
    entries = [
        ("confirmation_price", confirmation_label, "#2dd4bf", 2),
    ]
    if current_price is not None:
        entries.append(("_current_price", "現價", "#22d3ee", 1))

    lines = []
    price_lines = []
    for key, label, color, width in entries:
        value = (
            current_price
            if key == "_current_price"
            else confirmation
            if key == "confirmation_price"
            else row.get(key)
        )
        numeric = _safe_float_value(value)
        if numeric is None or numeric <= 0:
            continue
        lines.append(
            _level_line(
                dates,
                numeric,
                label,
                color,
                line_style=2,
                line_width=width,
                show_last_value=False,
                alpha=0.92,
            )
        )
        price_lines.append({
            "price": numeric,
            "title": label,
            "color": color,
            "lineWidth": width,
            "lineStyle": 2,
            "lineVisible": True,
            "axisLabelVisible": True,
        })
    return lines, price_lines


def _axis_label_price_line(price, title, color):
    numeric = _safe_float_value(price)
    if numeric is None or numeric <= 0:
        return None
    return {
        "price": numeric,
        "title": title,
        "color": color,
        "lineWidth": 1,
        "lineStyle": 2,
        "lineVisible": False,
        "axisLabelVisible": True,
    }


def setup_phase_caption(phase, fallback=None):
    captions = {
        'near_breakout': '尚未觸發突破；等待收市站上入場參考並配合成交量確認。',
        'fresh_breakout': '已帶量突破入場參考，目前未觸發過度延伸訊號。',
        'extended_breakout': '突破結構成立，但現價已超出追價範圍；等待回調或新底部。',
        'pullback_forming': '價格已進入潛在 META 回調區，但反轉訊號尚未確認。',
        'pullback_entry': 'META 回調入場條件已確認，止損參考設於支持位下方。',
        'failed_breakout': '突破結構已失效，暫不提供新買入設定。',
        'unclear_structure': '通過基本篩選，但技術結構未清晰；等待進場結構成形。',
        'not_recommended': '未符合基本篩選或技術條件，不宜買入。',
    }
    return captions.get(phase, fallback)


def _parse_earnings_date(value):
    numeric_value = pd.to_numeric(value, errors='coerce')
    if not pd.isna(numeric_value):
        if numeric_value <= 1_000_000_000:
            return pd.NaT
        unit = 'ms' if numeric_value >= 1_000_000_000_000 else 's'
        return pd.to_datetime(numeric_value, unit=unit, errors='coerce')
    return pd.to_datetime(value, errors='coerce')


def render_stock_detail(row, ticker):
    raw_name = row.get('name')
    if pd.isna(raw_name) or str(raw_name).strip().lower() == 'nan':
        display_name = ticker
    else:
        display_name = str(raw_name).strip()
    title = display_name if display_name == ticker else f"{display_name} ({ticker})"
    st.subheader(title)
    sector = translate_sector(row.get('sector_name'))
    industry = translate_industry(row.get('industry_name'))
    sector_industry = "｜".join(part for part in [sector, industry] if part)
    caption_parts = [
        f"行業：{sector_industry or '未知'}",
        f"市值：${row['market_cap'] / 1e9:.0f}B",
    ]
    earnings_date = _parse_earnings_date(row.get('earnings_date'))
    if not pd.isna(earnings_date) and earnings_date.year >= 2000:
        # Keep the header on the same completed-session basis as the
        # deterministic analysis card. Using today's wall-clock date here
        # made the header and the auditable earnings metric disagree by one.
        reference_date = _parse_earnings_date(row.get('data_date'))
        if pd.isna(reference_date):
            reference_date = pd.Timestamp.today().normalize()
        days_to_earnings = (earnings_date.normalize() - reference_date.normalize()).days
    else:
        days_to_earnings = None
    st.caption("　　".join(caption_parts))
    setup_status = normalize_setup_status(
        row.get('setup_status', row.get('status', 'watchlist')),
        row.get('setup_phase'),
    )
    setup_label, setup_icon, setup_color = _label_setup_status(setup_status)
    holding_action = normalize_holding_action(row.get('holding_action', 'hold'), row.get('setup_phase'))
    hold_label, hold_icon, hold_color = _label_holding_action(holding_action)
    with st.container(horizontal=True, vertical_alignment='center', gap='xsmall'):
        st.badge(setup_label, icon=setup_icon, color=setup_color)
        st.badge(hold_label, icon=hold_icon, color=hold_color)
        phase_badge = _label_setup_phase(row.get('setup_phase'))
        if phase_badge:
            phase_label, phase_icon, phase_color = phase_badge
            st.badge(f"趨勢：{phase_label}", icon=phase_icon, color=phase_color)

    # Render as plain text so currency symbols are never interpreted as inline math.
    st.markdown(
        (
            '<div class="stock-summary-caption">'
            f'{html.escape(deterministic_summary(row))}'
            '</div>'
            '<style>'
            '.stock-summary-caption {'
            'color: rgba(250, 250, 250, 0.6);'
            'font-size: 0.875rem;'
            'line-height: 1.5;'
            'margin: 0.35rem 0 1rem;'
            '}'
            '</style>'
        ),
        unsafe_allow_html=True,
    )

    phase = row.get('setup_phase')
    breakout_phase = phase in {'near_breakout', 'fresh_breakout'}
    current_price = row.get('price')
    closes = row.get('close')
    if isinstance(closes, list) and closes:
        current_price = closes[-1]
    entry_price = row.get('entry_price')
    entry_zone_low = _safe_float_value(row.get('entry_zone_low'))
    entry_zone_high = _safe_float_value(row.get('entry_zone_high'))
    confirmation_price = _safe_float_value(row.get('confirmation_price'))
    stop_price = row.get('stop_price')
    risk_pct = row.get('risk_pct')
    actionable_plan = row.get('setup_phase') in {
        'near_breakout', 'fresh_breakout', 'pullback_forming', 'pullback_entry',
    }
    is_meta = phase in {'pullback_forming', 'pullback_entry'}
    # Older snapshots may only have selected META evidence while their entry
    # fields still contain the old current-price fallback.
    if is_meta:
        evidence = row.get('setup_evidence') or {}
        chart_evidence = evidence.get('chart_evidence') or {}
        selected_meta = chart_evidence.get('selected_meta_zone')
        if isinstance(selected_meta, dict):
            entry_zone_low = entry_zone_low or _safe_float_value(selected_meta.get('low'))
            entry_zone_high = entry_zone_high or _safe_float_value(selected_meta.get('high'))
        if phase == 'pullback_forming':
            confirmation_price = None
    entry_label = 'META 入場區' if is_meta else '突破入場價'
    _, metric_help_cards = _metric_help_context(row, current_price, entry_price, stop_price, risk_pct)
    metric_help = {label: detail for label, _, detail in metric_help_cards}
    # Keep every headline metric at the same visual scale as the custom META
    # range metric below.  Streamlit's default metric value size is smaller,
    # which made the neighbouring prices and percentages look inconsistent.
    st.markdown(
        """
        <style>
            div[data-testid="stMetricValue"] {
                font-size: clamp(1.2rem, 1.75vw, 1.75rem) !important;
                line-height: 1.25;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Non-actionable classifications do not have a usable entry/stop plan.
    # Showing fallback prices beside the current price made observation stocks
    # look tradable and also left a missing confirmation column in the middle.
    entry_metrics = st.container(key=f"stock_primary_metrics_{ticker}")
    entry_cols = entry_metrics.columns(4 if breakout_phase or not actionable_plan else 5)
    if current_price is not None and not pd.isna(current_price):
        entry_cols[0].metric('現價', f"${current_price:,.2f}", help=metric_help.get('目前位置'))
    if actionable_plan and not breakout_phase and entry_zone_low is not None and entry_zone_high is not None:
        zone_text = f"${entry_zone_low:,.2f}–${entry_zone_high:,.2f}"
        # st.metric auto-shrinks and ellipsizes long ranges. Render the range
        # with the same visual scale as the neighbouring headline metrics so
        # META values remain readable instead of becoming `113.67–11...`.
        entry_cols[1].markdown(
            f"""
            <div class="primary-range-metric" title="META 為多重支持匯聚區；突破區為前期樞紐至確認價之間。">
                <div class="primary-range-metric-label">{html.escape(entry_label)} <span aria-hidden="true">ⓘ</span></div>
                <div class="primary-range-metric-value">{html.escape(zone_text)}</div>
            </div>
            <style>
                .primary-range-metric {{ min-height: 96px; min-width: 0; padding-top: 0.25rem; overflow: hidden; }}
                .primary-range-metric-label {{ color: rgba(250, 250, 250, 0.8); font-size: 1rem; font-weight: 600; line-height: 1.35; }}
                .primary-range-metric-label span {{ color: rgba(250, 250, 250, 0.62); font-size: 0.9rem; }}
                .primary-range-metric-value {{ color: rgb(250, 250, 250); font-size: clamp(1.2rem, 1.75vw, 1.75rem); line-height: 1.25; white-space: normal; overflow-wrap: anywhere; word-break: break-word; letter-spacing: 0; }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif actionable_plan and not breakout_phase and entry_price is not None and not pd.isna(entry_price):
        entry_cols[1].metric(entry_label, f"${entry_price:,.2f}", help=metric_help.get('入場參考'))
    if actionable_plan and confirmation_price is not None:
        confirmation_col = entry_cols[1] if breakout_phase else entry_cols[2]
        confirmation_col.metric(
            '確認突破價' if breakout_phase else '確認價',
            f"${confirmation_price:,.2f}",
            help="收市價站上確認突破價，並通過突破成交量確認，才算確認突破。" if breakout_phase else "價格出現相應反轉或突破確認後，才視為可執行入場參考。",
        )
    if actionable_plan and stop_price is not None and not pd.isna(stop_price):
        stop_col = entry_cols[2] if breakout_phase else entry_cols[3]
        stop_col.metric('止損參考', f"${stop_price:,.2f}", help=metric_help.get('止損參考'))
    if actionable_plan and risk_pct is not None and not pd.isna(risk_pct):
        risk_col = entry_cols[3] if breakout_phase else entry_cols[4]
        risk_col.metric('風險', f"{risk_pct:.1%}", help=metric_help.get('風險距離'))

    vs_market_metrics = st.container(key=f"stock_vs_market_metrics_{ticker}")
    vs_market_cols = vs_market_metrics.columns(4)
    for col, period, label in zip(vs_market_cols[:3], ['1m', '3m', '6m'], ['相對大市 1 個月', '相對大市 3 個月', '相對大市 6 個月']):
        value = get_vs_market_value(row, period)
        if value is not None and not pd.isna(value):
            sp500_return = _safe_float_value(row.get(f'sp500_pct_{period}'))
            market_detail = (
                f"同期標普 500 指數回報為 {sp500_return:+.1%}。"
                if sp500_return is not None
                else "同期標普 500 指數回報資料不足。"
            )
            col.metric(
                label,
                f"{value:+.1%}",
                help=f"股票回報減去標普 500 指數回報。{market_detail}",
            )
    if not pd.isna(earnings_date) and earnings_date.year >= 2000 and days_to_earnings is not None:
        if days_to_earnings >= 0:
            estimate_text = "預估日期。" if bool(row.get('is_earnings_estimate', True)) else "已公布日期。"
            timing_text = "今天" if days_to_earnings == 0 else f"距離分析資料日期 {days_to_earnings} 日"
            vs_market_cols[3].metric(
                "下次業績",
                earnings_date.strftime('%Y-%m-%d'),
                help=f"{timing_text}；{estimate_text}",
            )
        else:
            vs_market_cols[3].metric("下次業績", "尚未公布", help="現有業績日期早於分析資料日期。")
    else:
        vs_market_cols[3].metric("下次業績", "尚未公布", help="Yahoo Finance 暫未提供下一次業績日期。")

    chart_range = st.segmented_control(
        '圖表範圍',
        options=list(CHART_RANGES.keys()),
        default='3 個月',
        selection_mode='single',
    )
    chart_range = chart_range or '3 個月'
    range_state_key = f'{ticker}_chart_range'
    previous_chart_range = st.session_state.get(range_state_key)
    reset_visible_range = previous_chart_range is not None and previous_chart_range != chart_range
    st.session_state[range_state_key] = chart_range

    automatic_trendlines, trendline_metadata, trendline_errors = _automatic_diagonal_trendlines(row)

    indicator_cols = st.columns([1.35, 1.05, 1.0])
    with indicator_cols[0]:
        ma_labels = ['🟨 10MA', '🟧 20MA', '🟦 50MA', '🟥 200MA']
        selected_ma_labels = st.segmented_control(
            '移動平均線',
            options=ma_labels,
            default=ma_labels[:3],
            selection_mode='multi',
            key=f'{ticker}_ma_indicators',
        )
        # Streamlit can briefly return None on the first render of a new
        # ticker before the multi-select widget has materialised. Keep the
        # intended 10/20/50MA defaults visible during that first load.
        selected_ma_labels = ma_labels[:3] if selected_ma_labels is None else selected_ma_labels
        selected_ma = set(selected_ma_labels)

    with indicator_cols[1]:
        zone_labels = ['🟩 支持區', '🟥 阻力區', '🟧 樞紐區']
        selected_zone_labels = st.segmented_control(
            '價格區域',
            options=zone_labels,
            default=[],
            selection_mode='multi',
            key=f'{ticker}_price_zones_v2',
        ) or []
        selected_zones = set(selected_zone_labels)

    trendline_name_by_id = {
        line['trendline_id']: line['display_name']
        for line in automatic_trendlines
    }
    with indicator_cols[2]:
        if trendline_name_by_id:
            selected_trendline_ids = st.segmented_control(
                '趨勢線',
                options=list(trendline_name_by_id),
                default=list(trendline_name_by_id),
                selection_mode='multi',
                format_func=trendline_name_by_id.get,
                key=f'{ticker}_trendline_indicators',
            ) or []
        else:
            selected_trendline_ids = []

    is_meta_setup = row.get('setup_phase') in {'pullback_forming', 'pullback_entry'}
    is_breakout_setup = row.get('setup_phase') in {'near_breakout', 'fresh_breakout'}
    entry_range_label = '🟦 META 區間' if is_meta_setup else '🟦 突破區間'
    has_entry_range = (
        (is_meta_setup or is_breakout_setup)
        and entry_zone_low is not None
        and entry_zone_high is not None
        and entry_zone_high > entry_zone_low
    )
    stop_value = _safe_float_value(stop_price)
    stop_reference = _safe_float_value(entry_price)
    has_stop_plan = (
        actionable_plan
        and stop_value is not None
        and stop_reference is not None
        and stop_value < stop_reference
    )
    standalone_labels = ['🟪 形態提示', '📅 業績日期']
    if has_entry_range:
        standalone_labels.append(entry_range_label)
    if has_stop_plan:
        standalone_labels.append('🟥 止損建議')
    selected_standalone_labels = st.segmented_control(
        '其他指標',
        options=standalone_labels,
        default=standalone_labels,
        selection_mode='multi',
        key=f'{ticker}_standalone_indicators',
        help=YOLO_PATTERN_HELP,
    ) or []
    selected_standalone = set(selected_standalone_labels)
    indicators = {
        'SMA_10': '🟨 10MA' in selected_ma,
        'SMA_20': '🟧 20MA' in selected_ma,
        'SMA_50': '🟦 50MA' in selected_ma,
        'SMA_200': '🟥 200MA' in selected_ma,
        # Kept in the internal indicator contract for older callers. The
        # 20-day high remains analysis data, but is no longer a chart toggle.
        'pivot_high_20': False,
        'Support_Zone': '🟩 支持區' in selected_zones,
        'Resistance_Zone': '🟥 阻力區' in selected_zones,
        'Pivot_Zone': '🟧 樞紐區' in selected_zones,
        'YOLO_Pattern': '🟪 形態提示' in selected_standalone,
        'Vol_MA20': True,
        'Earnings_Date': '📅 業績日期' in selected_standalone,
        # This flag controls the setup-specific actionable range. Its label is
        # META 區間 for pullbacks and 突破區間 for breakout setups.
        'Breakout_Range': entry_range_label in selected_standalone,
        'Stop_Risk': '🟥 止損建議' in selected_standalone,
    }
    trendline_visibility = {
        line_id: line_id in selected_trendline_ids
        for line_id in trendline_name_by_id
    }
    visible_trendline_errors = [
        error for error in trendline_errors
        if '未找到符合條件' not in error
        and 'K 線數量不足' not in error
        and '陰陽燭數量不足' not in error
    ]
    if visible_trendline_errors:
        st.caption(' | '.join(visible_trendline_errors))
    recorded_trendlines = load_historical_chart(
        row,
        ticker,
        chart_range=chart_range,
        indicators=indicators,
        automatic_trendlines=automatic_trendlines,
        trendline_visibility=trendline_visibility,
        reset_visible_range=reset_visible_range,
    )
    return recorded_trendlines or trendline_metadata


def _trendln_method_caption(config):
    method_key = config.get('method', TRENDLN_METHOD_OPTIONS[DEFAULT_TRENDLN_METHOD_LABEL])
    for label, key in TRENDLN_METHOD_OPTIONS.items():
        if key == method_key:
            return label
    return DEFAULT_TRENDLN_METHOD_LABEL


def _trendln_method_constant(trendln, method_key):
    return getattr(trendln, method_key, trendln.METHOD_NSQUREDLOGN)


AUTOMATIC_TRENDLINE_CONFIG = {
    'lookback': 90,
    'window': 90,
    'error_tolerance': 0.5,
    'method': TRENDLN_METHOD_OPTIONS[DEFAULT_TRENDLN_METHOD_LABEL],
    'include_edge': False,
    'display_count': 3,
}


def _trendline_classification(line_type, slope):
    role = 'resistance' if line_type == 'Resistance' else 'support'
    slope_direction = 'rising' if float(slope) >= 0 else 'falling'
    line_class = {
        ('resistance', 'rising'): 'TRL',
        ('resistance', 'falling'): 'DTL',
        ('support', 'rising'): 'UTL',
        ('support', 'falling'): 'TSL',
    }[(role, slope_direction)]
    return role, slope_direction, line_class


def _dedupe_overlapping_trendline_spans(candidates, overlap_threshold=0.8):
    """Keep one ranked line when same-direction structures cover the same span."""
    kept = []
    for candidate in candidates:
        start = int(candidate.get('_display_start_index', 0))
        end = int(candidate.get('_display_end_index', start))
        span = max(1, end - start + 1)
        duplicate = False
        for existing in kept:
            existing_start = int(existing.get('_display_start_index', 0))
            existing_end = int(existing.get('_display_end_index', existing_start))
            existing_span = max(1, existing_end - existing_start + 1)
            overlap = max(0, min(end, existing_end) - max(start, existing_start) + 1)
            if overlap / min(span, existing_span) >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


_TRENDLINE_CLASS_DISPLAY = {
    'TRL': '上升趨勢阻力線',
    'DTL': '下降趨勢阻力線',
    'UTL': '上升趨勢支持線',
    'TSL': '下降趨勢支持線',
}


def _automatic_diagonal_trendlines(row):
    required = ('date', 'high', 'low', 'close')
    if not all(isinstance(row.get(key), list) and row.get(key) for key in required):
        return [], [], []

    config = AUTOMATIC_TRENDLINE_CONFIG
    lines = []
    errors = []
    class_counts = {}
    for line_type in ('Resistance', 'Support'):
        side_lines, side_error = _cached_trendline_overlays(
            tuple(row['date']),
            tuple(row['high']),
            tuple(row['low']),
            tuple(row['close']),
            line_type,
            config['lookback'],
            config['window'],
            config['error_tolerance'],
            1,
            config['display_count'],
            'Any slope',
            0,
            config['method'],
            config['include_edge'],
        )
        if side_error:
            line_label = "阻力趨勢線" if line_type == "Resistance" else "支持趨勢線"
            errors.append(f'{line_label}：{side_error}')
        for line in side_lines:
            start_date = str(line['data'][0]['time'])
            end_date = str(line['data'][-1]['time'])
            identity = f"{line_type}:{start_date}:{end_date}:{line['slope']:.10f}"
            trendline_id = hashlib.sha1(identity.encode('utf-8')).hexdigest()[:12]
            role, slope_direction, line_class = _trendline_classification(line_type, line['slope'])
            class_counts[line_class] = class_counts.get(line_class, 0) + 1
            display_name = f"{_TRENDLINE_CLASS_DISPLAY[line_class]} #{class_counts[line_class]}"
            line.update({
                'trendline_id': trendline_id,
                'display_name': display_name,
                'color': '#ff6b6b' if line_type == 'Resistance' else '#50e3c2',
                'role': role,
                'slope_direction': slope_direction,
                'line_class': line_class,
            })
            lines.append(line)
    metadata = _trendline_metadata_from_lines(lines)
    return lines, metadata, errors


def _automatic_yolo_pattern_result(ticker, df):
    required = {'date', 'open', 'high', 'low', 'close'}
    if df.empty or not required.issubset(df.columns):
        return None

    dates = tuple(df['date'].astype(str))
    opens = tuple(df['open'].astype(float))
    highs = tuple(df['high'].astype(float))
    lows = tuple(df['low'].astype(float))
    closes = tuple(df['close'].astype(float))
    result = _cached_yolo_pattern_scan(
        ticker,
        dates,
        opens,
        highs,
        lows,
        closes,
        YOLO_CONFIDENCE_THRESHOLD,
        YOLO_PATTERN_LOOKBACK_BARS,
    )

    if result.get('status') == 'ok':
        signature = hashlib.md5(
            json.dumps(
                {
                    'dates': dates,
                    'closes': closes,
                    'threshold': YOLO_CONFIDENCE_THRESHOLD,
                    'lookback': YOLO_PATTERN_LOOKBACK_BARS,
                },
                default=str,
            ).encode('utf-8')
        ).hexdigest()
        stat_key = f'yolo_pattern_stat_{ticker}'
        if st.session_state.get(stat_key) != signature:
            _record_yolo_pattern_stat(ticker, YOLO_CONFIDENCE_THRESHOLD, result)
            st.session_state[stat_key] = signature
    return result


@st.cache_resource(show_spinner=False)
def _load_yolo_pattern_model():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    model_path = hf_hub_download(
        repo_id=YOLO_PATTERN_MODEL,
        filename='model.pt',
        cache_dir=str(EXPERIMENT_OUTPUT_ROOT / 'trendline_engine_trial' / '_models'),
    )
    return YOLO(model_path)


@st.cache_data(show_spinner=False)
def _cached_yolo_pattern_scan(ticker, dates, opens, highs, lows, closes, confidence_threshold, lookback_bars):
    started = time.perf_counter()
    try:
        image_path, chart_meta = _render_yolo_static_chart(ticker, dates, opens, highs, lows, closes, lookback_bars)
        model = _load_yolo_pattern_model()
        load_ms = (time.perf_counter() - started) * 1000
        infer_started = time.perf_counter()
        results = model.predict(source=str(image_path), conf=float(confidence_threshold), verbose=False)
        inference_ms = (time.perf_counter() - infer_started) * 1000
        detections = []
        annotated_png = None
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    'label': names.get(cls_id, str(cls_id)),
                    'confidence': float(box.conf[0]),
                    'xyxy': [float(value) for value in box.xyxy[0].tolist()],
                })
            annotated = result.plot()
            try:
                import cv2

                ok, encoded = cv2.imencode('.png', annotated)
                if ok:
                    annotated_png = encoded.tobytes()
            except Exception:
                annotated_png = None
        detections = sorted(detections, key=lambda item: item['confidence'], reverse=True)
        return {
            'status': 'ok',
            'load_ms': load_ms,
            'inference_ms': inference_ms,
            'detections': detections,
            'annotated_png': annotated_png,
            'image_path': str(image_path),
            'chart_meta': chart_meta,
            'lookback_bars': int(lookback_bars),
        }
    except Exception as exc:
        return {'status': 'error', 'error': str(exc)}


def _record_yolo_pattern_stat(ticker, threshold, result):
    if result.get('status') != 'ok':
        return
    output_dir = EXPERIMENT_OUTPUT_ROOT / 'yolo_pattern_hints'
    output_dir.mkdir(parents=True, exist_ok=True)
    stat_path = output_dir / 'detections.jsonl'
    payload = {
        'created_at': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(timespec='seconds'),
        'ticker': ticker,
        'threshold': float(threshold),
        'lookback_bars': result.get('lookback_bars'),
        'load_ms': result.get('load_ms'),
        'inference_ms': result.get('inference_ms'),
        'detections': result.get('detections', []),
        'image_path': result.get('image_path'),
    }
    with stat_path.open('a') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


def _render_yolo_static_chart(ticker, dates, opens, highs, lows, closes, lookback_bars):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    df = pd.DataFrame({
        'date': pd.to_datetime(list(dates), errors='coerce'),
        'open': list(opens),
        'high': list(highs),
        'low': list(lows),
        'close': list(closes),
    }).dropna().tail(int(lookback_bars))
    output_dir = EXPERIMENT_OUTPUT_ROOT / 'yolo_pattern_hints'
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f'{ticker}_clean.png'
    chart_dates = mdates.date2num(df['date'].to_numpy(dtype='datetime64[ms]'))

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.set_facecolor('#101418')
    fig.patch.set_facecolor('#101418')
    candle_width = 0.65
    for idx, row in enumerate(df.itertuples(index=False)):
        color = '#26a69a' if row.close >= row.open else '#ef5350'
        ax.vlines(chart_dates[idx], row.low, row.high, color=color, linewidth=1)
        lower = min(row.open, row.close)
        height = abs(row.close - row.open) or 0.01
        ax.add_patch(
            plt.Rectangle(
                (chart_dates[idx] - candle_width / 2, lower),
                candle_width,
                height,
                color=color,
                alpha=0.95,
            )
        )
    ax.grid(color='#2f3540', linewidth=0.6)
    ax.tick_params(colors='#9aa0a6')
    ax.set_title(f'{ticker} clean candles', color='#d8dee9')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.tight_layout()
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    fig_width, fig_height = fig.canvas.get_width_height()
    chart_meta = {
        'xlim': [float(value) for value in ax.get_xlim()],
        'ylim': [float(value) for value in ax.get_ylim()],
        'axis_bbox': [float(bbox.x0), float(bbox.y0), float(bbox.width), float(bbox.height)],
        'image_size': [float(fig_width), float(fig_height)],
        'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
        'date_nums': [float(value) for value in chart_dates],
    }
    fig.savefig(image_path)
    plt.close(fig)
    return image_path, chart_meta


def _yolo_detection_rectangles(result):
    if not result or result.get('status') != 'ok':
        return []
    meta = result.get('chart_meta') or {}
    xlim = meta.get('xlim') or []
    ylim = meta.get('ylim') or []
    axis_bbox = meta.get('axis_bbox') or []
    image_size = meta.get('image_size') or []
    dates = meta.get('dates') or []
    date_nums = meta.get('date_nums') or []
    if len(xlim) != 2 or len(ylim) != 2 or len(axis_bbox) != 4 or len(image_size) != 2:
        return []
    if not dates or len(dates) != len(date_nums):
        return []

    axis_x, axis_y, axis_w, axis_h = axis_bbox
    image_h = image_size[1]
    if axis_w <= 0 or axis_h <= 0:
        return []

    def pixel_to_data(px, py):
        x_ratio = (px - axis_x) / axis_w
        y_from_bottom = image_h - py
        y_ratio = (y_from_bottom - axis_y) / axis_h
        x_value = xlim[0] + x_ratio * (xlim[1] - xlim[0])
        y_value = ylim[0] + y_ratio * (ylim[1] - ylim[0])
        return x_value, y_value

    def nearest_date_index(x_value):
        idx = min(range(len(date_nums)), key=lambda item: abs(date_nums[item] - x_value))
        return idx

    rectangles = []
    for detection in result.get('detections', []):
        xyxy = detection.get('xyxy') or []
        if len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = xyxy
        left_x, high_y = pixel_to_data(min(x1, x2), min(y1, y2))
        right_x, low_y = pixel_to_data(max(x1, x2), max(y1, y2))
        low = min(low_y, high_y)
        high = max(low_y, high_y)
        if low <= 0 or high <= 0:
            continue
        left_idx = nearest_date_index(left_x)
        right_idx = nearest_date_index(right_x)
        if right_idx < left_idx:
            left_idx, right_idx = right_idx, left_idx
        rectangles.append({
            'leftTime': dates[left_idx],
            'leftPrevTime': dates[left_idx - 1] if left_idx > 0 else None,
            'leftLogical': float(left_idx) - 0.5,
            'rightTime': dates[right_idx],
            'rightNextTime': dates[right_idx + 1] if right_idx + 1 < len(dates) else None,
            'rightLogical': float(right_idx) + 0.5,
            'low': float(low),
            'high': float(high),
            'color': '#a855f7',
            'fillColor': _rgba_from_hex('#a855f7', 0.18),
            'label': _display_yolo_pattern_label(detection.get('label')),
            'zOrder': 'top',
        })
    return rectangles


def _display_yolo_pattern_label(label):
    raw = str(label or '').strip()
    if not raw:
        return '形態'
    normalized = raw.replace('_', ' ').replace('-', ' ').lower()
    return YOLO_PATTERN_LABELS.get(raw.lower()) or YOLO_PATTERN_LABELS.get(normalized) or raw.replace('_', ' ')


def _mid_line_marker(data, label, color):
    if not label or not data:
        return []
    midpoint = data[len(data) // 2]
    return [{
        'time': midpoint['time'],
        'position': 'aboveBar',
        'color': color,
        'shape': 'circle',
        'text': label,
        'size': 0,
    }]


def _line_series(
    df,
    column,
    color,
    title=None,
    line_width=1,
    show_last_value=True,
    show_price_line=True,
):
    data = (
        df[['date', column]]
        .dropna()
        .rename(columns={'date': 'time', column: 'value'})
        .to_dict(orient='records')
    )

    return {
        'type': 'Line',
        'data': data,
        # Retained for pane hover readouts. The series itself now renders its
        # last value natively on the Y axis.
        'axisLabel': title if show_last_value and title else None,
        'options': {
            'color': color,
            'lineWidth': line_width,
            'title': title or '',
            'lastValueVisible': show_last_value,
            'priceLineVisible': show_price_line,
            'crosshairMarkerVisible': False,
        },
    }


def _horizontal_line_series(df, column, color, title=None, line_width=1, show_last_value=False):
    value = df[column].dropna().iloc[-1] if column in df.columns and not df[column].dropna().empty else None
    if value is None:
        return None

    data = [{'time': date, 'value': value} for date in df['date']]
    return {
        'type': 'Line',
        'data': data,
        'options': {
            'color': color,
            'lineWidth': line_width,
            'lineStyle': 2,
            'title': '',
            'lastValueVisible': show_last_value,
            'priceLineVisible': False,
            'crosshairMarkerVisible': False,
        },
    }


def _trendline_series(line):
    trendline_markers = _mid_line_marker(
        line.get('data', []),
        line.get('line_class'),
        line.get('color', '#9ca3af'),
    )
    if line.get('type'):
        return {
            **line,
            'markers': [*line.get('markers', []), *trendline_markers],
            'options': {
                **line.get('options', {}),
                'crosshairMarkerVisible': False,
            },
        }
    return {
        'type': 'Line',
        'data': line['data'],
        'options': {
            'color': line['color'],
            'lineWidth': line.get('lineWidth', 1),
            'lineStyle': line.get('lineStyle', 1),
            'title': line.get('title', ''),
            'lastValueVisible': line.get('lastValueVisible', False),
            'priceLineVisible': False,
            'crosshairMarkerVisible': False,
        },
        'markers': trendline_markers,
    }


def _level_line(dates, level, label, color, line_style=2, line_width=1, show_last_value=False, alpha=1.0):
    data = [{'time': date, 'value': float(level)} for date in dates]
    return {
        'data': data,
        'label': label,
        'color': _rgba_from_hex(color, alpha) if alpha < 1 else color,
        'lineStyle': line_style,
        'lineWidth': line_width,
        'lastValueVisible': show_last_value,
    }


def _zone_area(dates, low, high, color, label='', alpha=0.30):
    data = [{'time': date, 'value': float(high)} for date in dates]
    return {
        'type': 'Baseline',
        'data': data,
        'excludeFromAutoscale': True,
        'options': {
            'baseValue': {
                'type': 'price',
                'price': float(low),
            },
            'topFillColor1': _rgba_from_hex(color, alpha),
            'topFillColor2': _rgba_from_hex(color, alpha),
            'bottomFillColor1': 'rgba(0,0,0,0)',
            'bottomFillColor2': 'rgba(0,0,0,0)',
            'topLineColor': 'rgba(0,0,0,0)',
            'bottomLineColor': 'rgba(0,0,0,0)',
            'lineVisible': False,
            'lastValueVisible': False,
            'priceLineVisible': False,
        },
    }


def _zone_overlay(dates, low, high, color, label='', line_alpha=1.0, show_boundaries=True):
    zone_dates = list(dates)
    overlays = [_zone_area(zone_dates, low, high, color, label=label)]
    if show_boundaries:
        overlays.extend([
            _level_line(zone_dates, high, label, color, line_style=0, line_width=1, alpha=line_alpha),
            _level_line(zone_dates, low, '', color, line_style=0, line_width=1, alpha=line_alpha),
        ])
    return overlays


def _native_pivot_line(time_values, value_values, color, line_width=1):
    return {
        'type': 'Line',
        'data': [
            {'time': time_value, 'value': float(value)}
            for time_value, value in zip(time_values, value_values)
        ],
        'options': {
            'color': color,
            'lineWidth': line_width,
            'title': '',
            'lastValueVisible': False,
            'priceLineVisible': False,
            'crosshairMarkerVisible': False,
        },
    }


def _pivot_rectangle_overlay(
    dates,
    first_idx,
    last_idx,
    low,
    high,
    color,
    label='',
    label_offset_y=None,
):
    if not dates:
        return []
    rectangle = {
        'leftTime': dates[first_idx],
        'leftPrevTime': dates[first_idx - 1] if first_idx > 0 else None,
        'rightTime': dates[last_idx],
        'rightNextTime': dates[last_idx + 1] if last_idx + 1 < len(dates) else None,
        'low': float(low),
        'high': float(high),
        'color': color,
        'fillColor': _rgba_from_hex(color, 0.30),
        'label': label,
        'zOrder': 'top',
    }
    if label_offset_y is not None:
        rectangle['labelOffsetY'] = float(label_offset_y)
    return [{
        'type': 'PivotRectangle',
        'rectangle': rectangle,
    }]


def _risk_arrow_overlay(dates, index, low, high, label):
    if not dates:
        return None
    index = max(0, min(int(index), len(dates) - 1))
    return {
        'time': dates[index],
        'previousTime': dates[index - 1] if index > 0 else None,
        'low': float(low),
        'high': float(high),
        'color': '#fb7185',
        'label': label,
        'zOrder': 'normal',
    }


def _zones_overlap(first, second, tolerance_pct=0.0):
    first_low = float(first.get('low', 0))
    first_high = float(first.get('high', 0))
    second_low = float(second.get('low', 0))
    second_high = float(second.get('high', 0))
    if first_low <= 0 or first_high <= 0 or second_low <= 0 or second_high <= 0:
        return False
    tolerance = max(first_high, second_high) * tolerance_pct / 100
    return max(first_low, second_low) <= min(first_high, second_high) + tolerance


def _zone_mid_gap_pct(first, second):
    first_mid = float(first.get('mid', 0))
    second_mid = float(second.get('mid', 0))
    base = min(first_mid, second_mid)
    if base <= 0:
        return 999.0
    return abs(first_mid / second_mid - 1) * 100


def _zone_edge_gap_value(first, second):
    first_low = float(first.get('low', 0))
    first_high = float(first.get('high', 0))
    second_low = float(second.get('low', 0))
    second_high = float(second.get('high', 0))
    if first_low <= 0 or first_high <= 0 or second_low <= 0 or second_high <= 0:
        return 999.0
    if _zones_overlap(first, second):
        return 0.0
    return max(first_low, second_low) - min(first_high, second_high)


def _normalize_zone_width_pct(zone, min_width_pct=0.0, max_width_pct=0.0):
    if min_width_pct <= 0 and max_width_pct <= 0:
        return zone
    mid = float(zone.get('mid', 0))
    low = float(zone.get('low', 0))
    high = float(zone.get('high', 0))
    if mid <= 0 or low <= 0 or high <= 0:
        return zone
    width = high - low
    min_width = mid * min_width_pct / 100 if min_width_pct > 0 else 0
    max_width = mid * max_width_pct / 100 if max_width_pct > 0 else 0
    target_width = max(width, min_width)
    if max_width > 0:
        target_width = min(target_width, max_width)
    if abs(target_width - width) < 0.000001:
        return zone
    normalized = dict(zone)
    normalized['low'] = max(mid - target_width / 2, 0.01)
    normalized['high'] = mid + target_width / 2
    return normalized


def _select_spaced_zone_candidates(candidates, display_count, latest_close, atr, exclude_zones=()):
    filtered = []
    for candidate in candidates:
        if any(_zones_overlap(candidate, excluded, tolerance_pct=0.1) for excluded in exclude_zones):
            continue
        filtered.append(candidate)

    atr_pct = atr / latest_close * 100 if latest_close else 0
    gap_values = [
        max(latest_close * 0.025, atr * 0.8),
        max(latest_close * 0.018, atr * 0.55),
        max(latest_close * 0.012, atr * 0.35),
    ]
    best_selection = []
    minimum_useful_count = min(display_count, len(filtered), 3)
    for gap_value in gap_values:
        selected = []
        for candidate in filtered:
            if any(_zones_overlap(candidate, existing, tolerance_pct=0.2) for existing in selected):
                continue
            if any(_zone_edge_gap_value(candidate, existing) < gap_value for existing in selected):
                continue
            selected.append(candidate)
            if len(selected) >= display_count:
                break
        if len(selected) > len(best_selection):
            best_selection = selected
        if len(selected) >= minimum_useful_count:
            return selected[:display_count]
    return (best_selection or filtered[:1])[:display_count]


def _select_nearest_non_overlapping_zones(candidates, display_count, exclude_zones=(), tolerance_pct=0.1):
    selected = []
    for candidate in candidates:
        if any(_zones_overlap(candidate, excluded, tolerance_pct=tolerance_pct) for excluded in exclude_zones):
            continue
        if any(_zones_overlap(candidate, existing, tolerance_pct=tolerance_pct) for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= display_count:
            break
    return selected


def _detect_pivot_ranges(highs, lows, start_idx, end_idx, min_days, max_range_pct):
    ranges = []
    idx = start_idx
    while idx <= end_idx - min_days:
        window_high = float(np.max(highs[idx:idx + min_days]))
        window_low = float(np.min(lows[idx:idx + min_days]))
        range_pct = (window_high / window_low - 1) * 100 if window_low else 999
        if range_pct <= max_range_pct:
            pivot_start = idx
            pivot_end = idx + min_days - 1
            while pivot_end + 1 < end_idx:
                next_high = float(np.max(highs[pivot_start:pivot_end + 2]))
                next_low = float(np.min(lows[pivot_start:pivot_end + 2]))
                next_range_pct = (next_high / next_low - 1) * 100 if next_low else 999
                if next_range_pct > max_range_pct:
                    break
                pivot_end += 1
                window_high, window_low, range_pct = next_high, next_low, next_range_pct
            mid = (window_high + window_low) / 2
            ranges.append({
                'low': window_low,
                'mid': mid,
                'high': window_high,
                'first_touch': pivot_start,
                'last_touch': pivot_end,
                'days': pivot_end - pivot_start + 1,
                'range_pct': range_pct,
            })
            idx = pivot_end + 1
        else:
            idx += 1
    return ranges


def _pivot_range_schedule(max_range_pct):
    range_ceiling = max(5.0, float(max_range_pct))
    wider_ranges = [round(value, 1) for value in np.arange(5.5, range_ceiling + 0.001, 0.5)]
    schedule = [(5.0, 10)]
    schedule.extend((5.0, days) for days in range(9, 2, -1))
    for range_pct in wider_ranges:
        schedule.append((range_pct, 10))
        schedule.extend((range_pct, days) for days in range(9, 2, -1))
    return schedule


def _select_priority_pivot_ranges(highs, lows, closes, display_count, max_range_pct):
    return detect_priority_pivot_ranges(
        highs,
        lows,
        closes,
        display_count=display_count,
        max_range_pct=max_range_pct,
    )


def _cluster_price_levels(prices, indexes, latest_close, min_zone_width, min_touches):
    if len(prices) < min_touches:
        return []

    tolerance = max(min_zone_width, 0.01)
    try:
        from sklearn.cluster import AgglomerativeClustering
        clusters = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=tolerance,
            linkage='single',
        ).fit_predict(np.array(prices).reshape(-1, 1))
    except Exception:
        clusters = np.arange(len(prices))

    grouped = []
    for cluster_id in sorted(set(clusters)):
        mask = clusters == cluster_id
        cluster_prices = np.array(prices)[mask]
        cluster_indexes = np.array(indexes)[mask]
        if len(cluster_prices) < min_touches:
            continue
        low = float(cluster_prices.min())
        high = float(cluster_prices.max())
        midpoint = float(cluster_prices.mean())
        width = max(high - low, min_zone_width)
        grouped.append({
            'low': midpoint - width / 2,
            'mid': midpoint,
            'high': midpoint + width / 2,
            'touches': int(len(cluster_prices)),
            'first_touch': int(cluster_indexes.min()),
            'last_touch': int(cluster_indexes.max()),
        })
    return grouped


def _overlay_score(touches, last_touch, series_length, distance):
    touch_score = min(float(touches) / 5, 1.0) * 45
    recency_score = max(0.0, 1 - ((series_length - 1 - last_touch) / max(series_length, 1))) * 30
    distance_score = max(0.0, 1 - min(abs(float(distance)), 0.2) / 0.2) * 25
    return int(round(touch_score + recency_score + distance_score))


def _line_usefulness(score):
    return f"U{score}"


def _mean_true_range(highs, lows, closes):
    if len(closes) < 2:
        return 0.0
    prev_closes = closes[:-1]
    ranges = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - prev_closes),
        np.abs(lows[1:] - prev_closes),
    ])
    return float(np.nanmean(ranges[-14:])) if len(ranges) else 0.0


def _volatility_pct(highs, lows, closes):
    latest_close = float(closes[-1]) if len(closes) else 0.0
    if latest_close <= 0:
        return 1.0
    return max(_mean_true_range(highs, lows, closes) / latest_close * 100, 0.1)


def _trendline_clarity_score(idxs, values, slope, intercept, latest_close):
    idxs = np.array(list(idxs), dtype='float64')
    if len(idxs) < 3 or latest_close <= 0:
        return 0
    fitted = slope * idxs + intercept
    residual_pct = np.abs(values[idxs.astype(int)] - fitted) / np.maximum(latest_close, 0.01)
    fit_score = max(0.0, 1 - min(float(np.mean(residual_pct)), 0.08) / 0.08) * 45
    span_score = min((idxs.max() - idxs.min()) / max(len(values) - 1, 1), 1.0) * 30
    touch_score = min(len(idxs) / 5, 1.0) * 25
    return int(round(fit_score + span_score + touch_score))


def _first_trendline_structure_break(highs, lows, closes, slope, intercept, line_type, last_touch_index):
    """Return a confirmed post-touch break, ignoring a single shallow pierce."""
    highs = np.array(highs, dtype='float64')
    lows = np.array(lows, dtype='float64')
    closes = np.array(closes, dtype='float64')
    true_ranges = np.empty(len(closes), dtype='float64')
    true_ranges[0] = 0.0
    if len(closes) > 1:
        true_ranges[1:] = np.maximum.reduce([
            highs[1:] - lows[1:],
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ])
    for index in range(max(0, int(last_touch_index) + 1), len(closes)):
        level = slope * index + intercept
        if level <= 0:
            continue
        close = closes[index]
        is_beyond = close > level if line_type == 'Resistance' else close < level
        if not is_beyond:
            continue
        atr = float(np.mean(true_ranges[max(1, index - 13):index + 1])) if index else 0.0
        beyond_by_atr = abs(close - level) >= atr > 0
        if index + 1 < len(closes):
            next_level = slope * (index + 1) + intercept
            next_close = closes[index + 1]
            confirmed_next_close = next_close > next_level if line_type == 'Resistance' else next_close < next_level
        else:
            confirmed_next_close = False
        if beyond_by_atr or confirmed_next_close:
            return index
    return None


def _trendline_inter_touch_violation(highs, lows, closes, slope, intercept, line_type, idxs):
    """Reject lines whose confirming touches are separated by a real close-through."""
    highs = np.array(highs, dtype='float64')
    lows = np.array(lows, dtype='float64')
    closes = np.array(closes, dtype='float64')
    idxs = sorted({int(index) for index in idxs if 0 <= int(index) < len(closes)})
    if len(idxs) < 2 or len(closes) < 2:
        return None

    true_ranges = np.empty(len(closes), dtype='float64')
    true_ranges[0] = 0.0
    true_ranges[1:] = np.maximum.reduce([
        highs[1:] - lows[1:],
        np.abs(highs[1:] - closes[:-1]),
        np.abs(lows[1:] - closes[:-1]),
    ])
    touch_indexes = set(idxs)
    first_touch = idxs[0]
    last_touch = idxs[-1]
    for index in range(first_touch + 1, last_touch):
        if index in touch_indexes:
            continue
        level = slope * index + intercept
        if level <= 0:
            continue
        close = closes[index]
        atr = float(np.mean(true_ranges[max(1, index - 13):index + 1])) if index else 0.0
        tolerance = max(atr * 0.5, abs(level) * 0.01)
        if line_type == 'Resistance' and close > level + tolerance:
            return index
        if line_type == 'Support' and close < level - tolerance:
            return index
    return None


def _rgba_from_hex(color, alpha):
    color = str(color).lstrip('#')
    if len(color) != 6:
        return color
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha:.2f})"


@st.cache_data(show_spinner=False)
def _cached_sr_zone_overlays(
    dates,
    opens,
    highs,
    lows,
    closes,
    line_type,
    lookback,
    candidate_rank,
    display_count,
    min_touches,
    prominence_pct,
    zone_width_pct,
    gap_min_pct,
    gap_mode,
    exclude_zones=(),
    role_filter=(),
):
    line_type = {
        '支持區': 'Support Zone',
        '阻力區': 'Resistance Zone',
        '樞紐區間': 'Pivot Range',
        '樞紐區': 'Pivot Range',
        '缺口回補': 'Gap Fill',
        'Support': 'Support Zone',
        'Resistance': 'Resistance Zone',
        'Pivot': 'Pivot Range',
    }.get(str(line_type), line_type)
    dates = list(dates)[-lookback:]
    opens = np.array(list(opens)[-lookback:], dtype='float64')
    highs = np.array(list(highs)[-lookback:], dtype='float64')
    lows = np.array(list(lows)[-lookback:], dtype='float64')
    closes = np.array(list(closes)[-lookback:], dtype='float64')
    if len(closes) < 40 or len(dates) != len(closes):
        return [], "Not enough bars for S/R detection.", []

    latest_close = closes[-1]
    if latest_close <= 0:
        return [], "Latest close is invalid.", []
    atr = _mean_true_range(highs, lows, closes)
    min_zone_width = max(latest_close * zone_width_pct / 100, atr * 0.35, 0.01)

    if line_type == 'Gap Fill':
        gaps = []
        for idx in range(1, len(closes)):
            prev_high = highs[idx - 1]
            prev_low = lows[idx - 1]
            prev_close = closes[idx - 1]
            open_price = opens[idx]
            low = lows[idx]
            high = highs[idx]
            gap_candidates = []
            if open_price > prev_close:
                gap_low, gap_high = sorted((float(prev_close), float(open_price)))
                gap_candidates.append(('up', gap_low, gap_high))
            elif open_price < prev_close:
                gap_low, gap_high = sorted((float(prev_close), float(open_price)))
                gap_candidates.append(('down', gap_low, gap_high))

            if gap_mode == 'All notable gaps':
                if low > prev_high:
                    gap_candidates.append(('up', float(prev_high), float(low)))
                elif high < prev_low:
                    gap_candidates.append(('down', float(high), float(prev_low)))

            for direction, gap_low, gap_high in gap_candidates:
                role = 'support' if direction == 'up' else 'resistance'
                if role_filter and role not in role_filter:
                    continue
                gap_pct = gap_high / gap_low - 1 if gap_low else 0
                if gap_pct < gap_min_pct / 100:
                    continue
                filled = bool(np.any(lows[idx:] <= gap_low)) if direction == 'up' else bool(np.any(highs[idx:] >= gap_high))
                is_relevant = latest_close >= gap_low if direction == 'up' else latest_close <= gap_high
                if (gap_mode == 'All notable gaps' or not filled) and is_relevant:
                    gaps.append({
                        'low': gap_low,
                        'mid': (gap_low + gap_high) / 2,
                        'high': gap_high,
                        'direction': direction,
                        'touches': 1,
                        'first_touch': max(0, idx - 1),
                        'last_touch': idx,
                        'distance': abs(latest_close / ((gap_low + gap_high) / 2) - 1),
                        'range_pct': (gap_high / gap_low - 1) * 100 if gap_low else 0,
                    })
        candidates = sorted(gaps, key=lambda item: (item['distance'], -item['last_touch']))
        if not candidates:
            return [], "No open gap-fill level found.", []
        start_index = max(0, min(candidate_rank - 1, len(candidates) - 1))
        selected = _select_nearest_non_overlapping_zones(
            candidates[start_index:],
            display_count,
            tolerance_pct=0.1,
        )
        overlays = []
        metadata = []
        for item in selected:
            is_support = item['direction'] == 'up'
            label = '支持區' if is_support else '阻力區'
            color = '#50e3c2' if is_support else '#ff6b6b'
            overlays.extend(
                _zone_overlay(
                    dates[item['first_touch']:],
                    item['low'],
                    item['high'],
                    color,
                    label,
                    show_boundaries=False,
                )
            )
            metadata.append({
                'type': 'gap_fill',
                'role': 'support' if is_support else 'resistance',
                'low': float(item['low']),
                'high': float(item['high']),
                'mid': float(item['mid']),
                'range_pct': float(item['range_pct']),
                'start_date': dates[item['first_touch']],
                'gap_date': dates[item['last_touch']],
            })
        return overlays, None, metadata

    try:
        from scipy.signal import find_peaks
    except ImportError:
        return [], "scipy is not installed.", []

    if line_type == 'Pivot Range':
        max_range_pct = float(prominence_pct)
        selected_ranges = _select_priority_pivot_ranges(highs, lows, closes, min(display_count, 5), max_range_pct)
        if not selected_ranges:
            return [], f"No pivot range found under {max_range_pct:.1f}%.", []
        overlays = []
        metadata = []
        for selected in selected_ranges:
            color = '#f5a623'
            label = f"{int(selected['days'])}d {float(selected['range_pct']):.1f}%"
            first_touch = int(selected['first_touch'])
            last_touch = int(selected['last_touch'])
            overlays.extend(_pivot_rectangle_overlay(
                dates,
                first_touch,
                last_touch,
                selected['low'],
                selected['high'],
                color,
                label,
            ))
            metadata.append({
                'type': 'pivot_range',
                'role': 'pivot',
                'low': float(selected['low']),
                'high': float(selected['high']),
                'mid': float(selected['mid']),
                'range_pct': float(selected['range_pct']),
                'days': int(selected['days']),
                'start_date': dates[first_touch],
                'end_date': dates[last_touch],
            })
        return overlays, None, metadata

    prominence = max(latest_close * prominence_pct / 100, 0.01)
    peak_indexes, _ = find_peaks(highs, prominence=prominence, distance=3)
    valley_indexes, _ = find_peaks(-lows, prominence=prominence, distance=3)
    if line_type == 'Resistance Zone':
        resistance_prices = list(highs[peak_indexes])
        resistance_indexes = list(peak_indexes)
        overhead_valleys = valley_indexes[lows[valley_indexes] > latest_close]
        resistance_prices.extend(lows[overhead_valleys])
        resistance_indexes.extend(overhead_valleys)
        levels = _cluster_price_levels(resistance_prices, resistance_indexes, latest_close, min_zone_width, min_touches)
        levels = [_normalize_zone_width_pct(level, 1.5, 2.5) for level in levels]
        candidates = [
            dict(level, distance=level['mid'] / latest_close - 1, score=_overlay_score(level['touches'], level['last_touch'], len(closes), level['mid'] / latest_close - 1))
            for level in levels
            if level['mid'] >= latest_close
        ]
        if len(candidates) < display_count:
            fallback_prominence = max(latest_close * prominence_pct / 200, atr * 0.35, 0.01)
            fallback_peaks, _ = find_peaks(highs, prominence=fallback_prominence, distance=2)
            fallback_valleys, _ = find_peaks(-lows, prominence=fallback_prominence, distance=2)
            fallback_prices = list(highs[fallback_peaks])
            fallback_indexes = list(fallback_peaks)
            fallback_overhead_valleys = fallback_valleys[lows[fallback_valleys] > latest_close]
            fallback_prices.extend(lows[fallback_overhead_valleys])
            fallback_indexes.extend(fallback_overhead_valleys)
            fallback_levels = _cluster_price_levels(fallback_prices, fallback_indexes, latest_close, min_zone_width, 1)
            fallback_levels = [_normalize_zone_width_pct(level, 1.5, 2.5) for level in fallback_levels]
            for level in fallback_levels:
                if level['mid'] < latest_close:
                    continue
                if any(_zones_overlap(level, existing, tolerance_pct=0.5) for existing in candidates):
                    continue
                distance = level['mid'] / latest_close - 1
                candidates.append(dict(
                    level,
                    distance=distance,
                    score=_overlay_score(level['touches'], level['last_touch'], len(closes), distance),
                    fallback=True,
                ))
        color = '#ff6b6b'
        label = 'R'
    else:
        support_prices = list(lows[valley_indexes])
        support_indexes = list(valley_indexes)
        broken_peaks = peak_indexes[highs[peak_indexes] < latest_close]
        support_prices.extend(highs[broken_peaks])
        support_indexes.extend(broken_peaks)
        levels = _cluster_price_levels(support_prices, support_indexes, latest_close, min_zone_width, min_touches)
        levels = [_normalize_zone_width_pct(level, 1.5, 2.5) for level in levels]
        candidates = [
            dict(level, distance=latest_close / level['mid'] - 1, score=_overlay_score(level['touches'], level['last_touch'], len(closes), latest_close / level['mid'] - 1))
            for level in levels
            if level['mid'] <= latest_close
        ]
        if len(candidates) < display_count:
            fallback_prominence = max(latest_close * prominence_pct / 200, atr * 0.35, 0.01)
            fallback_peaks, _ = find_peaks(highs, prominence=fallback_prominence, distance=2)
            fallback_valleys, _ = find_peaks(-lows, prominence=fallback_prominence, distance=2)
            fallback_prices = list(lows[fallback_valleys])
            fallback_indexes = list(fallback_valleys)
            fallback_broken_peaks = fallback_peaks[highs[fallback_peaks] < latest_close]
            fallback_prices.extend(highs[fallback_broken_peaks])
            fallback_indexes.extend(fallback_broken_peaks)
            fallback_levels = _cluster_price_levels(fallback_prices, fallback_indexes, latest_close, min_zone_width, 1)
            fallback_levels = [_normalize_zone_width_pct(level, 1.5, 2.5) for level in fallback_levels]
            for level in fallback_levels:
                if level['mid'] > latest_close:
                    continue
                if any(_zones_overlap(level, existing, tolerance_pct=0.5) for existing in candidates):
                    continue
                distance = latest_close / level['mid'] - 1
                candidates.append(dict(
                    level,
                    distance=distance,
                    score=_overlay_score(level['touches'], level['last_touch'], len(closes), distance),
                    fallback=True,
                ))
        color = '#50e3c2'
        label = 'S'

    candidates = sorted(
        candidates,
        key=lambda item: (
            abs(item['distance']),
            -item['touches'],
            -(item['last_touch']),
        ),
    )
    if not candidates:
        return [], f"No matching {line_type.lower()} found.", []

    start_index = max(0, min(candidate_rank - 1, len(candidates) - 1))
    if line_type in {'Resistance Zone', 'Support Zone'}:
        selected_zones = _select_spaced_zone_candidates(
            candidates[start_index:],
            display_count,
            latest_close,
            atr,
            exclude_zones,
        )
    else:
        selected_zones = candidates[start_index:start_index + display_count]
    overlays = []
    metadata = []
    for selected in selected_zones:
        zone_metadata = {
            'type': 'sr_zone',
            'role': 'resistance' if line_type == 'Resistance Zone' else 'support',
            'low': float(selected['low']),
            'high': float(selected['high']),
            'mid': float(selected['mid']),
            'range_pct': float((selected['high'] / selected['low'] - 1) * 100) if selected['low'] else 0.0,
            'touches': int(selected['touches']),
            'start_date': dates[selected['first_touch']],
            'last_touch_date': dates[selected['last_touch']],
        }
        if any(_zones_overlap(zone_metadata, excluded, tolerance_pct=0.1) for excluded in exclude_zones):
            continue
        zone_label = '阻力區' if line_type == 'Resistance Zone' else '支持區'
        overlays.extend(
            _zone_overlay(
                dates[selected['first_touch']:],
                selected['low'],
                selected['high'],
                color,
                zone_label,
                line_alpha=0.9,
                # Keep both support and resistance zones as soft fills only.
                # Their upper/lower borders compete with candles and moving
                # averages, especially when several zones overlap.
                show_boundaries=False,
            )
        )
        metadata.append(zone_metadata)
    return overlays, None, metadata


@st.cache_data(show_spinner=False)
def _cached_trendline_overlays(
    dates,
    highs,
    lows,
    closes,
    line_type,
    lookback,
    window,
    error_tolerance,
    candidate_rank,
    display_count,
    slope_filter,
    end_offset,
    method_key,
    include_edge,
):
    try:
        import trendln
    except ImportError:
        return [], "trendln is not installed."

    all_dates = list(dates)
    all_highs = np.array(list(highs), dtype='float64')
    all_lows = np.array(list(lows), dtype='float64')
    all_closes = np.array(list(closes), dtype='float64')
    end_index = max(0, len(all_closes) - max(0, int(end_offset or 0)))
    start_index = max(0, end_index - lookback)
    dates = all_dates[start_index:end_index]
    highs = all_highs[start_index:end_index]
    lows = all_lows[start_index:end_index]
    closes = all_closes[start_index:end_index]
    display_closes = all_closes[start_index:]
    display_dates = all_dates[start_index:]
    if len(highs) < 60 or len(dates) != len(highs):
        return [], "陰陽燭數量不足，未能偵測趨勢線。"

    source = (None, highs) if line_type == 'Resistance' else (lows, None)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = trendln.calc_support_resistance(
                source,
                method=_trendln_method_constant(trendln, method_key),
                window=min(window, len(highs)),
                errpct=error_tolerance / 100,
                sortError=True,
                include_edge=bool(include_edge),
            )
    except Exception as exc:
        return [], f"趨勢線計算失敗：{exc}"

    if len(result) < 3 or not result[2]:
        label = "阻力" if line_type == "Resistance" else "支持"
        return [], f"未找到符合條件的{label}趨勢線。"

    latest_index = len(highs) - 1
    latest_close = closes[-1]
    line_values = highs if line_type == 'Resistance' else lows
    candidates = []
    for idxs, params in result[2]:
        if len(idxs) < 3:
            continue
        slope, intercept = float(params[0]), float(params[1])
        if slope_filter == 'Downward only' and slope >= 0:
            continue
        if slope_filter == 'Upward only' and slope <= 0:
            continue

        latest_level = slope * latest_index + intercept
        if latest_level <= 0:
            continue

        distance = latest_close / latest_level - 1
        if abs(distance) > 0.5:
            continue

        inter_touch_violation_index = _trendline_inter_touch_violation(
            highs,
            lows,
            closes,
            slope,
            intercept,
            line_type,
            idxs,
        )
        if inter_touch_violation_index is not None:
            continue

        structure_break_index = _first_trendline_structure_break(
            all_highs[start_index:],
            all_lows[start_index:],
            display_closes,
            slope,
            intercept,
            line_type,
            max(idxs),
        )
        first_touch_index = min(idxs)
        line_end_index = structure_break_index if structure_break_index is not None else len(display_dates) - 1
        if line_end_index < first_touch_index:
            continue
        line_data = [
            {'time': date, 'value': float(slope * idx + intercept)}
            for idx, date in enumerate(display_dates[:line_end_index + 1])
            if idx >= first_touch_index and slope * idx + intercept > 0
        ]
        if len(line_data) < 2:
            continue
        usefulness = _overlay_score(len(idxs), max(idxs), len(highs), distance)
        clarity = _trendline_clarity_score(idxs, line_values, slope, intercept, latest_close)

        candidates.append({
            'data': line_data,
            'label': f"TL C{clarity}/U{usefulness}",
            'distance': float(distance),
            'slope': float(slope / latest_close) if latest_close else 0,
            'points': len(idxs),
            'clarity': clarity,
            'is_downtrend': slope < 0,
            'line_type': line_type,
            'latest_level': float(latest_level),
            'structure_status': 'broken' if structure_break_index is not None else 'active',
            'structure_break_date': display_dates[structure_break_index] if structure_break_index is not None else None,
            '_display_start_index': first_touch_index,
            '_display_end_index': line_end_index,
            'sort_key': (
                0 if structure_break_index is None else 1,
                -max(idxs),
                -clarity,
                abs(distance),
                -len(idxs),
            ),
        })

    candidates = sorted(candidates, key=lambda item: item['sort_key'])
    minimum_line_gap = max(latest_close * 0.015, _mean_true_range(highs, lows, closes))

    # Treat nearby lines as one structure only when they have the same role and
    # slope direction. Rising resistance and falling resistance are different
    # TTT concepts, so they should not dedupe each other.
    candidates_by_slope = {'rising': [], 'falling': []}
    for candidate in candidates:
        _, slope_direction, _ = _trendline_classification(line_type, candidate['slope'])
        candidates_by_slope[slope_direction].append(candidate)

    deduped_candidates = []
    for slope_group in candidates_by_slope.values():
        slope_group = _dedupe_overlapping_trendline_spans(slope_group)
        by_level = sorted(slope_group, key=lambda item: item['latest_level'])
        level_groups = []
        for candidate in by_level:
            if not level_groups or candidate['latest_level'] - level_groups[-1][-1]['latest_level'] >= minimum_line_gap:
                level_groups.append([candidate])
            else:
                level_groups[-1].append(candidate)

        if line_type == 'Resistance':
            deduped_candidates.extend(max(group, key=lambda item: item['latest_level']) for group in level_groups)
        else:
            deduped_candidates.extend(min(group, key=lambda item: item['latest_level']) for group in level_groups)

    candidates = deduped_candidates
    candidates.sort(key=lambda item: item['sort_key'])
    rank_index = max(0, min(candidate_rank - 1, len(candidates) - 1))
    candidates = candidates[rank_index:rank_index + display_count]
    for candidate in candidates:
        candidate.pop('sort_key', None)
        candidate.pop('_display_start_index', None)
        candidate.pop('_display_end_index', None)

    if not candidates:
        label = "阻力" if line_type == "Resistance" else "支持"
        return [], f"以目前設定未找到符合條件的{label}趨勢線。"
    return candidates, None


def _trendline_metadata_from_lines(lines):
    records = []
    for line in lines:
        if not line.get('data'):
            continue
        role, slope_direction, line_class = _trendline_classification(
            line['line_type'],
            line['slope'],
        )
        records.append({
            'id': line['trendline_id'],
            'name': line['display_name'],
            'side': line['line_type'],
            'role': line.get('role', role),
            'slope_direction': line.get('slope_direction', slope_direction),
            'line_class': line.get('line_class', line_class),
            'start_date': str(line['data'][0]['time']),
            'end_date': str(line['data'][-1]['time']),
            'latest_level': round(float(line['latest_level']), 6),
            'distance_pct': round(float(line['distance']) * 100, 4),
            'slope_pct': round(float(line['slope']) * 100, 6),
            'touches': int(line['points']),
            'clarity': int(line['clarity']),
            'structure_status': line['structure_status'],
            'structure_break_date': (
                str(line['structure_break_date']) if line['structure_break_date'] else None
            ),
            'config': dict(AUTOMATIC_TRENDLINE_CONFIG),
        })
    return records


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _fetch_earnings_marker_dates(ticker):
    try:
        earnings = yf.Ticker(ticker).get_earnings_dates(limit=24)
    except Exception:
        return []
    if earnings is None or earnings.empty:
        return []

    normalized = earnings.reset_index()
    date_column = normalized.columns[0]
    normalized[date_column] = pd.to_datetime(
        normalized[date_column],
        errors='coerce',
        utc=True,
    ).dt.tz_convert(None)
    normalized = normalized.dropna(subset=[date_column])

    rows = []
    today = pd.Timestamp.today().normalize()
    historical_start = today - pd.Timedelta(days=180)
    for _, earnings_row in normalized.iterrows():
        earnings_day = earnings_row[date_column].normalize()
        if earnings_day < historical_start:
            continue
        is_future = earnings_day > today
        rows.append({
            'date': earnings_day.strftime('%Y-%m-%d'),
            'is_estimate': is_future,
        })
    return rows


def _earnings_marker(date_value, is_estimate):
    return {
        'time': date_value,
        'position': 'belowBar',
        'color': '#f5a623' if is_estimate else '#4a90e2',
        'shape': 'circle',
        'text': 'e' if is_estimate else 'E',
        'size': 0.5,
    }


def _earnings_markers(row, df, ticker):
    chart_dates = set(df['date'])
    markers = []
    seen_dates = set()

    for item in _fetch_earnings_marker_dates(ticker):
        if item['date'] in chart_dates and item['date'] not in seen_dates:
            markers.append(_earnings_marker(item['date'], item['is_estimate']))
            seen_dates.add(item['date'])

    earnings_date = row.get('earnings_date')
    if earnings_date is None or pd.isna(earnings_date):
        return markers

    earnings_ts = pd.to_datetime(earnings_date, errors='coerce')
    if pd.isna(earnings_ts):
        return markers

    earnings_day = earnings_ts.strftime('%Y-%m-%d')
    if earnings_day not in chart_dates or earnings_day in seen_dates:
        return markers

    today = pd.Timestamp.today().normalize()
    is_future = earnings_ts.normalize() > today
    is_estimate = is_future and bool(row.get('is_earnings_estimate', True))
    markers.append(_earnings_marker(earnings_day, is_estimate))
    return markers


def _histogram_series(df, column, color, title=None):
    data = (
        df[['date', column]]
        .dropna()
        .rename(columns={'date': 'time', column: 'value'})
        .to_dict(orient='records')
    )

    return {
        'type': 'Histogram',
        'data': data,
        'options': {
            'color': color,
            'title': '',
        },
    }


def _fallback_ohlcv_chart_frame(ticker, exchange=None):
    """Load chart-only OHLCV when an analysis cache omitted its price arrays."""
    try:
        from services.stock_service import get_ohlcv

        history = get_ohlcv(ticker, exchange=exchange)
    except Exception:
        return pd.DataFrame()

    if history is None or history.empty:
        return pd.DataFrame()

    frame = history.reset_index()
    date_column = frame.columns[0]
    return frame.rename(columns={
        date_column: 'date',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
    })


def load_historical_chart(
    row,
    ticker,
    live_price=None,
    chart_range='3 個月',
    indicators=None,
    automatic_trendlines=None,
    trendline_visibility=None,
    reset_visible_range=False,
):

    list_items = {idx: x for idx, x in row.items() if isinstance(x, list)}
    if list_items.get('date'):
        series_length = len(list_items['date'])
        temp_dict = {
            idx: values
            for idx, values in list_items.items()
            if len(values) == series_length
        }
        df = pd.DataFrame(temp_dict)
    else:
        df = pd.DataFrame()

    indicators = indicators or {}
    required_chart_columns = {'date', 'open', 'high', 'low', 'close', 'volume'}
    missing_columns = required_chart_columns - set(df.columns)
    if missing_columns:
        df = _fallback_ohlcv_chart_frame(ticker, exchange=row.get('exchange'))
        missing_columns = required_chart_columns - set(df.columns)
        if missing_columns:
            st.warning('暫時未有可用的歷史價格資料，請更新股票資料後再試。')
            return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for column in ('open', 'high', 'low', 'close', 'volume'):
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = (
        df.dropna(subset=['date', 'open', 'high', 'low', 'close'])
        .loc[lambda frame: frame['high'] >= frame['low']]
        .sort_values('date')
        .drop_duplicates(subset=['date'], keep='last')
        .reset_index(drop=True)
    )
    if df.empty:
        df = _fallback_ohlcv_chart_frame(ticker, exchange=row.get('exchange'))
        if df.empty:
            st.warning('暫時未有可用的歷史價格資料，請更新股票資料後再試。')
            return
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for column in ('open', 'high', 'low', 'close', 'volume'):
            df[column] = pd.to_numeric(df[column], errors='coerce')
        df = (
            df.dropna(subset=['date', 'open', 'high', 'low', 'close'])
            .loc[lambda frame: frame['high'] >= frame['low']]
            .sort_values('date')
            .drop_duplicates(subset=['date'], keep='last')
            .reset_index(drop=True)
        )
        if df.empty:
            st.warning('暫時未有可用的歷史價格資料，請更新股票資料後再試。')
            return
    df['volume'] = df['volume'].fillna(0).clip(lower=0)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    try:
        # 靜態注入：將 Live Price 當作歷史數據嘅一部分處理
        # if live_price and live_price > 0:
        #     today_str = datetime.date.today().strftime('%Y-%m-%d')
        #     # 確保日期列格式一致
        #     # df['date'] = df['date'].astype(str)
        #
        #     if df.iloc[-1]['date'] == today_str:
        #         # 更新今日最後一行
        #         df.loc[df.index[-1], 'close'] = float(live_price)
        #         df.loc[df.index[-1], 'high'] = max(float(df.iloc[-1]['high']), float(live_price))
        #         df.loc[df.index[-1], 'low'] = min(float(df.iloc[-1]['low']), float(live_price))
        #     else:
        #         # 新增今日行
        #         new_row = pd.DataFrame([{
        #             'date': today_str,
        #             'open': float(df.iloc[-1]['close']),
        #             'high': float(max(df.iloc[-1]['close'], live_price)),
        #             'low': float(min(df.iloc[-1]['close'], live_price)),
        #             'close': float(live_price),
        #             'volume': float(df.iloc[-1]['volume'])
        #         }])
        #         df = pd.concat([df, new_row], ignore_index=True)

        #  【關鍵修正】強制排序並重置 Index，保證數據節點連續
        # df = df.sort_values('date').reset_index(drop=True)

        df['color'] = np.where(df['open'] > df['close'], COLOR_BEAR, COLOR_BULL)
        yolo_result = None
        if indicators.get('YOLO_Pattern'):
            # Analyse the same window that the chart currently shows. Running
            # YOLO over a much longer history can produce valid rectangles
            # outside the visible range, making the toggle look ineffective.
            yolo_frame = df
            visible_yolo_bars = CHART_VISIBLE_BARS.get(chart_range)
            if visible_yolo_bars is not None:
                yolo_frame = df.tail(int(visible_yolo_bars)).reset_index(drop=True)
            yolo_result = _automatic_yolo_pattern_result(ticker, yolo_frame)
            if yolo_result and yolo_result.get('status') == 'error':
                error_text = str(yolo_result.get('error') or '未知錯誤').splitlines()[0]
                st.warning(f'形態提示暫時無法顯示：{error_text}')
            elif yolo_result and yolo_result.get('status') == 'ok':
                detection_count = len(yolo_result.get('detections') or [])
                if detection_count:
                    st.caption(f'形態提示：目前圖表範圍偵測到 {detection_count} 個形態框。')
                else:
                    st.caption('形態提示：目前圖表範圍未偵測到符合信心度門檻的形態。')

        df1 = df[['date', 'open', 'high', 'low', 'close']]
        df1.columns = ['time', 'open', 'high', 'low', 'close']
        ohlc_series = df1.to_dict(orient='records')

        df2 = df[['date', 'volume', 'color']]
        df2.columns = ['time', 'value', 'color']
        vol_series = df2.to_dict(orient='records')

        aligned_settings = {
            'layout': {
                'background': {
                    'color': 'transparent'
                },
                'textColor': 'grey',
            },
            'grid': {
                'vertLines': {
                    'color': 'rgba(42, 46, 57, 0.8)'
                },
                'horzLines': {
                    'color': 'rgba(42, 46, 57, 0.8)'
                }
            },
            'crosshair': {
                'mode': 0,
                'horzLine': {
                    'visible': True,
                    'labelVisible': False,
                },
                'vertLine': {
                    'visible': False,
                    'labelVisible': False,
                },
            },
            'timeScale': {
                'fixLeftEdge': True,
                'fixRightEdge': True,
            },
            # v5 invalidates older persisted ranges that can leave otherwise
            # valid charts positioned outside their available data.
            'zoomStorageKey': f'{ticker}_{chart_range}_visible_logical_range_v5',
            'resetVisibleLogicalRange': bool(reset_visible_range),
            'rightPriceScale': {
                'alignLabels': True,
                'minimumWidth': 64,
            },
            'priceScale':{
                'alignLabels': True
            },
            'localization': {
                'dateFormat': 'yyyy-MM-dd'
            }
        }
        bar_spacing = CHART_RANGES.get(chart_range)
        if bar_spacing is not None:
            aligned_settings['timeScale']['barSpacing'] = bar_spacing
            visible_bars = CHART_VISIBLE_BARS.get(chart_range) or len(df)
            aligned_settings['visibleLogicalRange'] = {
                'from': max(0, len(df) - visible_bars),
                'to': len(df) - 1,
            }

        my_chart_options = [
            {
                **aligned_settings,
                'height': 300,
                'watermark': {
                    'visible': True,
                    'text': '價格',
                    'fontSize': 16,
                    'color': 'rgba(210, 210, 210, 0.55)',
                    'horzAlign': 'left',
                    'vertAlign': 'top',
                },
            },
            {
                **aligned_settings,
                'height': 150,
                'watermark': {
                    'visible': True,
                    'text': '成交量',
                    'fontSize': 16,
                    'color': 'rgba(210, 210, 210, 0.55)',
                    'horzAlign': 'left',
                    'vertAlign': 'top',
                },
            },
            {
                **aligned_settings,
                'height': 120,
                'watermark': {
                    'visible': True,
                    'text': 'RSI',
                    'fontSize': 16,
                    'color': 'rgba(210, 210, 210, 0.55)',
                    'horzAlign': 'left',
                    'vertAlign': 'top',
                },
            },
            {
                **aligned_settings,
                'height': 150,
                'watermark': {
                    'visible': True,
                    'text': 'MACD',
                    'fontSize': 16,
                    'color': 'rgba(210, 210, 210, 0.55)',
                    'horzAlign': 'left',
                    'vertAlign': 'top',
                },
            }]

        price_series = [
            {
                'type': 'Candlestick',
                'data': ohlc_series,
                'markers': _earnings_markers(row, df, ticker) if indicators.get('Earnings_Date', True) else [],
                # 'options': {
                #     'color': 'blue',
                #     'lineWidth': 2
                # }
            },
        ]
        price_axis_lines = []

        indicator_styles = {
            'SMA_10': ('#f8e71c', '10MA', True),
            'SMA_20': ('#f5a623', '20MA', True),
            'SMA_50': ('#4a90e2', '50MA', True),
            'SMA_200': ('#d0021b', '200MA', True),
        }
        for column, (color, title, show_last_value) in indicator_styles.items():
            if indicators.get(column) and column in df.columns:
                price_series.append(
                    _line_series(
                        df,
                        column,
                        color,
                        title=title,
                        show_last_value=show_last_value,
                        show_price_line=False,
                    )
                )

        pivot_rectangles = []
        risk_arrows = []
        setup_evidence = row.get('setup_evidence')
        chart_evidence = (
            setup_evidence.get('chart_evidence', {})
            if isinstance(setup_evidence, dict)
            else {}
        )
        selected_meta_zone = chart_evidence.get('selected_meta_zone')
        if (
            row.get('setup_phase') in {'pullback_forming', 'pullback_entry'}
            and indicators.get('Breakout_Range', True)
            and isinstance(selected_meta_zone, dict)
            and len(df)
        ):
            meta_low = _safe_float_value(selected_meta_zone.get('low'))
            meta_high = _safe_float_value(selected_meta_zone.get('high'))
            if meta_low is not None and meta_high is not None:
                pivot_rectangles.extend(
                    overlay['rectangle']
                    for overlay in _pivot_rectangle_overlay(
                        list(df['date']),
                        max(0, len(df) - 20),
                        len(df) - 1,
                        meta_low,
                        meta_high,
                        '#7dd3fc',
                        f"META區間 ${meta_low:,.2f} - ${meta_high:,.2f}",
                    )
                )
        # The 20-day pivot remains in the analysis data, but is deliberately
        # not drawn as a separate chart line. For breakout candidates, show
        # the actionable entry band instead: pivot -> confirmation price.
        breakout_phases = {
            'near_breakout', 'fresh_breakout', 'extended_breakout', 'failed_breakout',
        }
        if row.get('setup_phase') in breakout_phases and indicators.get('Breakout_Range', True):
            breakout_low = _safe_float_value(row.get('entry_zone_low'))
            breakout_high = _safe_float_value(row.get('entry_zone_high'))
            if breakout_low is not None and breakout_high is not None and breakout_high > breakout_low:
                pivot_rectangles.extend(
                    overlay['rectangle']
                    for overlay in _pivot_rectangle_overlay(
                        list(df['date']),
                        max(0, len(df) - 20),
                        len(df) - 1,
                        breakout_low,
                        breakout_high,
                        '#7dd3fc',
                        f"突破區間 ${breakout_low:,.2f}–${breakout_high:,.2f}",
                    )
                )
        risk_setup_phases = {
            'near_breakout', 'fresh_breakout', 'pullback_forming', 'pullback_entry',
        }
        if row.get('setup_phase') in risk_setup_phases and indicators.get('Stop_Risk', True):
            risk_reference_price = _safe_float_value(row.get('entry_price'))
            stop_price = _safe_float_value(row.get('stop_price'))
            if (
                risk_reference_price is not None
                and stop_price is not None
                and stop_price < risk_reference_price
            ):
                risk_pct = (risk_reference_price - stop_price) / risk_reference_price
                risk_arrows.append(
                    _risk_arrow_overlay(
                        list(df['date']),
                        len(df) - 1,
                        stop_price,
                        risk_reference_price,
                        (
                            f"${risk_reference_price:,.2f}買入，\n"
                            f"${stop_price:,.2f}止損 ({risk_pct:.1%})"
                        ),
                    )
                )

        breakout_level_series = None
        auto_gap_min_pct = _volatility_pct(
            df['high'].to_numpy(dtype='float64'),
            df['low'].to_numpy(dtype='float64'),
            df['close'].to_numpy(dtype='float64'),
        )
        zone_role_filter = tuple(
            role
            for role, enabled in (
                ('support', indicators.get('Support_Zone', True)),
                ('resistance', indicators.get('Resistance_Zone', True)),
            )
            if enabled
        )
        auto_gap_lines, auto_gap_error, auto_gap_zones = ([], None, [])
        if zone_role_filter:
            auto_gap_lines, auto_gap_error, auto_gap_zones = _cached_sr_zone_overlays(
                tuple(df['date']),
                tuple(df['open']),
                tuple(df['high']),
                tuple(df['low']),
                tuple(df['close']),
                'Gap Fill',
                len(df),
                1,
                999,
                1,
                1.0,
                0.1,
                auto_gap_min_pct,
                'Open gaps only',
                (),
                zone_role_filter,
            )
        st.session_state[f'{ticker}_gap_fill_zones'] = auto_gap_zones
        for gap_line in auto_gap_lines:
            price_series.append(_trendline_series(gap_line))

        auto_support_lines, auto_support_error, auto_support_zones = ([], None, [])
        if indicators.get('Support_Zone', True):
            support_gap_zones = tuple(
                zone for zone in auto_gap_zones if zone.get('role') == 'support'
            )
            auto_support_lines, auto_support_error, auto_support_zones = _cached_sr_zone_overlays(
                tuple(df['date']),
                tuple(df['open']),
                tuple(df['high']),
                tuple(df['low']),
                tuple(df['close']),
                '支持區',
                len(df),
                1,
                5,
                2,
                2.0,
                0.1,
                1.0,
                'Open gaps only',
                support_gap_zones,
            )
        st.session_state[f'{ticker}_support_zones'] = auto_support_zones
        for support_line in auto_support_lines:
            price_series.append(_trendline_series(support_line))

        auto_resistance_lines, auto_resistance_error, auto_resistance_zones = ([], None, [])
        if indicators.get('Resistance_Zone', True):
            resistance_gap_zones = tuple(
                zone for zone in auto_gap_zones if zone.get('role') == 'resistance'
            )
            auto_resistance_lines, auto_resistance_error, auto_resistance_zones = _cached_sr_zone_overlays(
                tuple(df['date']),
                tuple(df['open']),
                tuple(df['high']),
                tuple(df['low']),
                tuple(df['close']),
                '阻力區',
                len(df),
                1,
                5,
                2,
                2.0,
                0.1,
                1.0,
                'Open gaps only',
                resistance_gap_zones,
            )
        st.session_state[f'{ticker}_resistance_zones'] = auto_resistance_zones
        for resistance_line in auto_resistance_lines:
            price_series.append(_trendline_series(resistance_line))

        auto_pivot_lines, auto_pivot_error, auto_pivot_zones = ([], None, [])
        if indicators.get('Pivot_Zone', True):
            auto_pivot_lines, auto_pivot_error, auto_pivot_zones = _cached_sr_zone_overlays(
                tuple(df['date']),
                tuple(df['open']),
                tuple(df['high']),
                tuple(df['low']),
                tuple(df['close']),
                '樞紐區間',
                min(90, len(df)),
                1,
                5,
                10,
                10.0,
                0.1,
                1.0,
                'Open gaps only',
            )
        st.session_state[f'{ticker}_pivot_zones'] = auto_pivot_zones
        for pivot_line in auto_pivot_lines:
            if pivot_line.get('type') == 'PivotRectangle':
                pivot_rectangles.append(pivot_line['rectangle'])
            else:
                price_series.append(_trendline_series(pivot_line))

        if indicators.get('YOLO_Pattern'):
            pivot_rectangles.extend(_yolo_detection_rectangles(yolo_result))

        for trendline in automatic_trendlines or []:
            if trendline_visibility is not None and not trendline_visibility.get(trendline['trendline_id'], True):
                continue
            price_series.append(_trendline_series(trendline))

        # Add the selected breakout level after generic zones and trendlines so
        # it remains readable when several overlays share a similar price.
        if breakout_level_series:
            price_series.append(breakout_level_series)

        current_reference_price = live_price
        if current_reference_price is None:
            current_reference_price = row.get('price')
        if current_reference_price is None or pd.isna(current_reference_price):
            current_reference_price = df['close'].dropna().iloc[-1] if not df['close'].dropna().empty else None
        trade_level_lines, trade_price_lines = _trade_plan_price_lines(
            row,
            list(df['date']),
            current_price=current_reference_price,
        )
        for trade_line in trade_level_lines:
            price_series.append(_trendline_series(trade_line))

        volume_series = [
            {
                'type': 'Histogram',
                'data': vol_series,
                'options': {
                    'priceFormat': {
                        'type': 'volume'
                    },
                    'scaleMargins': {
                        'bottom': 0
                    },
                    'lastValueVisible': True,
                    'priceLineVisible': False,
                },
                'axisLabel': '成交量',
            }
        ]

        if indicators.get('Vol_MA20') and 'Vol_MA20' in df.columns:
            volume_series.append(
                _line_series(
                    df,
                    'Vol_MA20',
                    '#f8e71c',
                    title='20 日均量',
                    show_last_value=True,
                    show_price_line=False,
                )
            )

        chart_payload = [
            {
                'chart': {
                    **my_chart_options[0],
                },
                'series': price_series,
                'pivotRectangles': pivot_rectangles,
                'riskArrows': [arrow for arrow in risk_arrows if arrow],
                'priceLines': [*price_axis_lines, *trade_price_lines],
            },
            {
                'chart': my_chart_options[1],
                'series': volume_series
            }
        ]
        if 'RSI' in df.columns:
            chart_payload.append({
                'chart': my_chart_options[2],
                'series': [
                    _line_series(
                        df,
                        'RSI',
                        '#50e3c2',
                        title='RSI',
                        line_width=1,
                        show_last_value=True,
                        show_price_line=False,
                    )
                ]
            })

        if 'MACDh' in df.columns:
            macd_df = df.copy()
            macd_df['MACD_color'] = np.where(macd_df['MACDh'] >= 0, COLOR_BULL, COLOR_BEAR)
            macd_data = (
                macd_df[['date', 'MACDh', 'MACD_color']]
                .dropna()
                .rename(columns={'date': 'time', 'MACDh': 'value', 'MACD_color': 'color'})
                .to_dict(orient='records')
            )
            macd_series = [{
                'type': 'Histogram',
                'data': macd_data,
                'options': {
                    'title': '',
                    'lastValueVisible': False,
                    'priceLineVisible': False,
                }
            }]
            if 'MACD' in df.columns:
                macd_series.append(
                    _line_series(
                        df,
                        'MACD',
                        '#4a90e2',
                        title='快線',
                        line_width=1,
                        show_last_value=True,
                        show_price_line=False,
                    )
                )
            if 'MACD_signal' in df.columns:
                macd_series.append(
                    _line_series(
                        df,
                        'MACD_signal',
                        '#f5a623',
                        title='慢線',
                        line_width=1,
                        show_last_value=True,
                        show_price_line=False,
                    )
                )
            chart_payload.append({
                'chart': my_chart_options[3],
                'series': macd_series
            })

        render_native_stock_chart(chart_payload, key=f"stock_chart_{ticker}_native")
        return _trendline_metadata_from_lines(automatic_trendlines or [])


    except Exception as e:
        st.error(f"圖表加載失敗: {str(e)}")
        return []
