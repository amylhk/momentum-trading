import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from datetime import datetime
import html

try:
    from streamlit_searchbox import st_searchbox
except ImportError:  # Keep local/dev fallback when optional dependency is absent.
    st_searchbox = None

from utils import (
    get_live_card_data,
    _parse_earnings_date,
    get_vs_market_value,
    normalize_setup_status,
    normalize_holding_action,
    setup_phase_badge,
    setup_phase_display_text,
)
from services.stock_service import analyze_single_stock, refresh_bookmarked_stocks
from services.universe_service import format_symbol_option, load_symbol_universe, search_symbol_universe, selected_symbol
from repositories.storage import load_stock_cache, save_stock_cache
from core.i18n.sector_industry_zh import sector_industry_zh_text, translate_industry, translate_sector
from ui.browser_bookmarks import bookmark_symbols, hydrate_browser_bookmarks
from ui.stock_table_columns import MY_STOCKS_REVIEW_COLUMNS, holding_action_display_text
from native_stock_chart import browser_bookmark_storage


_STOCK_OPTION_SPLIT = "\u2063"


DEFAULT_STOCK_SEARCH_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "META",
    "GOOGL",
    "AMZN",
    "MU",
    "PLTR",
    "H",
    "DOCN",
    "SNDK",
]


def _setup_status_display_text(status, phase=None):
    normalized = normalize_setup_status(status, phase)
    display = {
        'breakout_buy': ('🚀', '空倉：留意突破'),
        'pullback_wait': ('🛡️', '空倉：留意META'),
        'pullback_buy': ('🛡️', '空倉：留意META'),
        'watchlist': ('👀', '空倉：觀望後續'),
        'not_recommended': ('❌', '空倉：不宜買入'),
    }
    icon, label = display.get(normalized, display['watchlist'])
    return f"{icon} {label}"


def _fallback_symbol_options(limit: int = 12) -> list[str]:
    return DEFAULT_STOCK_SEARCH_SYMBOLS[:limit]


def _format_stock_search_option(payload: dict[str, str]) -> str:
    symbol = str(payload.get("symbol", "")).upper().strip()
    name = str(payload.get("name", "")).strip()
    sector = translate_sector(payload.get("sector_name", ""))
    industry = translate_industry(payload.get("industry_name", ""))

    left = symbol if not name else f"{symbol} ({name})"
    right = " | ".join(value for value in [sector, industry] if value)
    return f"{left}{_STOCK_OPTION_SPLIT}{right}"


def _stock_option_from_row(row) -> tuple[str, dict[str, str]]:
    _, payload = format_symbol_option(row)
    return _format_stock_search_option(payload), payload


@st.cache_data(show_spinner=False)
def _stock_search_default_options(limit: int = 12) -> list[str]:
    universe = load_symbol_universe()
    if universe.empty:
        return _fallback_symbol_options(limit)

    options: list[str] = []
    seen: set[str] = set()
    if "symbol" in universe.columns:
        preferred = universe[
            universe["symbol"].astype(str).str.upper().isin(DEFAULT_STOCK_SEARCH_SYMBOLS)
        ].copy()
        preferred_order = {symbol: idx for idx, symbol in enumerate(DEFAULT_STOCK_SEARCH_SYMBOLS)}
        preferred["_sort"] = preferred["symbol"].astype(str).str.upper().map(preferred_order)
        for _, row in preferred.sort_values("_sort").iterrows():
            label, payload = _stock_option_from_row(row)
            symbol = str(payload.get("symbol", "")).upper()
            if symbol and symbol not in seen:
                options.append(label)
                seen.add(symbol)

    for _, row in universe.iterrows():
        label, payload = _stock_option_from_row(row)
        symbol = str(payload.get("symbol", "")).upper()
        if symbol and symbol not in seen:
            options.append(label)
            seen.add(symbol)
        if len(options) >= limit:
            break
    return options[:limit] or _fallback_symbol_options(limit)


def _stock_search_options(searchterm: str) -> list[tuple[str, dict[str, str]]]:
    if not (searchterm or "").strip():
        return []
    return [
        (_format_stock_search_option(payload), payload)
        for _, payload in search_symbol_universe(searchterm)
    ]


_STOCK_SEARCH_STYLE_OVERRIDES = {
    "wrapper": {
        "position": "relative",
        "zIndex": 100000,
    },
    "searchbox": {
        "menuList": {
            "zIndex": 100002,
            "maxHeight": "320px",
        },
        "option": {
            "fontFamily": "inherit",
        },
        "optionEmpty": "hidden",
    },
}


@st.cache_data(show_spinner=False)
def _stock_search_fallback_options() -> list[str]:
    universe = load_symbol_universe()
    if universe.empty:
        return _fallback_symbol_options(50)
    return [_stock_option_from_row(row)[0] for _, row in universe.iterrows()]

# Bookmark Functions

def _bump_bookmark_view_version():
    st.session_state.bookmark_view_version = st.session_state.get('bookmark_view_version', 0) + 1


def _persist_bookmark_order():
    st.session_state.browser_bookmarks_loaded = True
    st.session_state.browser_bookmark_symbols = bookmark_symbols(
        st.session_state.bookmark_stocks
    )


def add_stock(ticker):
    new_ticker = ticker.upper().strip()
    if not new_ticker:
        return False

    saved_symbols = set()
    if 'symbol' in st.session_state.bookmark_stocks.columns:
        saved_symbols = set(st.session_state.bookmark_stocks['symbol'])

    if new_ticker in saved_symbols:
        st.toast("此股票已在清單中。")
        return True
    else:
        try:
            stock = load_stock_cache(new_ticker, max_age_hours=24)
            if stock is None:
                stock = analyze_single_stock(new_ticker)
                save_stock_cache(new_ticker, stock, source='manual')
            new_stock_df = pd.DataFrame([stock])
        except Exception as e:
            st.toast(f"未能分析 {new_ticker}：{e}")
            return False

        st.session_state.bookmark_stocks = (
            pd.concat([st.session_state.bookmark_stocks, new_stock_df], ignore_index=True)
            .drop_duplicates(subset=['symbol'], keep='last')
            .reset_index(drop=True)
        )
        _persist_bookmark_order()
        _bump_bookmark_view_version()
        st.toast(f"已加入 {new_ticker}")
        return True


def _reset_stock_search_inputs():
    st.session_state.stock_searchbox_clear_token = (
        st.session_state.get('stock_searchbox_clear_token', 0) + 1
    )
    st.session_state.stock_searchbox_nonce = st.session_state.get('stock_searchbox_nonce', 0) + 1
    for key in list(st.session_state.keys()):
        key_text = str(key)
        if (
            key_text.startswith("stock_searchbox_")
            and "_react_" not in key_text
            and isinstance(st.session_state.get(key), dict)
        ):
            st.session_state.pop(key, None)
        elif key_text.startswith("stock_selectbox_"):
            st.session_state.pop(key, None)
    st.session_state.pop('pending_stock_search_add', None)


def _resolve_stock_search_symbol(value):
    if isinstance(value, dict):
        ticker = selected_symbol(value)
        if ticker:
            return ticker
    elif isinstance(value, str):
        ticker = value.split(_STOCK_OPTION_SPLIT, 1)[0].strip().split(None, 1)[0].upper()
        if ticker:
            return ticker
    else:
        return ""
    query = str(value or "").strip()
    if not query:
        return ""
    matches = search_symbol_universe(query)
    if matches:
        label, payload = matches[0]
        return selected_symbol(label) or str(payload.get("symbol", "")).upper().strip()
    return query.upper()


def _queue_stock_add(value):
    ticker = _resolve_stock_search_symbol(value)
    if ticker:
        st.session_state.pending_stock_search_add = ticker


def _repair_stock_search_state():
    for key in list(st.session_state.keys()):
        key_text = str(key)
        state = st.session_state.get(key)
        if (
            key_text.startswith("stock_searchbox_")
            and "_react_" not in key_text
            and isinstance(state, dict)
            and state.get("options_py") == []
            and state.get("result") is None
        ):
            st.session_state.pop(key, None)


def _queue_selectbox_stock_add(input_key):
    _queue_stock_add(st.session_state.get(input_key, ""))


def _analyze_and_bookmark_stock(value):
    ticker_to_analyze = _resolve_stock_search_symbol(value)
    if not ticker_to_analyze:
        st.toast("請先輸入股票代號。")
        return

    with st.spinner(f"正在分析 {ticker_to_analyze}..."):
        added = add_stock(ticker_to_analyze)
    if added:
        _reset_stock_search_inputs()
        st.rerun()

def delete_stock(ticker):
    # TODO: Add a warning popup first
    st.session_state.bookmark_stocks = st.session_state.bookmark_stocks[st.session_state.bookmark_stocks['symbol'] != ticker]
    _persist_bookmark_order()
    _bump_bookmark_view_version()
    st.rerun()


def bookmark_row_is_stale(row):
    if 'updated_at' not in row or pd.isna(row.get('updated_at')):
        return True

    updated_at = pd.to_datetime(row.get('updated_at'), errors='coerce')
    if pd.isna(updated_at):
        return True

    return updated_at.date() < datetime.now().date()


def bookmarks_need_refresh(bookmarks):
    if bookmarks.empty:
        return False
    return bookmarks.apply(bookmark_row_is_stale, axis=1).any()


def _row_updated_at(row):
    updated_at = pd.to_datetime(row.get('updated_at'), errors='coerce')
    if pd.isna(updated_at):
        return pd.Timestamp.min
    return updated_at


def _latest_bookmark_row(row):
    symbol = str(row.get('symbol', '')).upper().strip()
    candidates = [row.to_dict()]

    analyzed = st.session_state.get('analyzed_stocks', [])
    if analyzed:
        analyzed_df = analyzed if isinstance(analyzed, pd.DataFrame) else pd.DataFrame(analyzed)
        if not analyzed_df.empty and 'symbol' in analyzed_df.columns:
            matches = analyzed_df[analyzed_df['symbol'] == symbol]
            candidates.extend(matches.to_dict(orient='records'))

    cached = load_stock_cache(symbol)
    if cached:
        candidates.append(cached)

    return max(candidates, key=_row_updated_at)


def sync_bookmarks_with_latest_sources():
    bookmarks = st.session_state.bookmark_stocks
    if bookmarks.empty or 'symbol' not in bookmarks.columns:
        return bookmarks

    synced_rows = [
        _latest_bookmark_row(row)
        for _, row in bookmarks.reset_index(drop=True).iterrows()
    ]
    synced = pd.DataFrame(synced_rows).drop_duplicates(subset=['symbol'], keep='last').reset_index(drop=True)

    old_compare = bookmarks.reset_index(drop=True).astype(str)
    new_compare = synced.reset_index(drop=True).reindex(columns=bookmarks.columns.union(synced.columns)).astype(str)
    old_compare = old_compare.reindex(columns=new_compare.columns)
    if not old_compare.equals(new_compare):
        st.session_state.bookmark_stocks = synced
        _persist_bookmark_order()
        _bump_bookmark_view_version()

    return st.session_state.bookmark_stocks


def _latest_value(row, key, default=None):
    values = row.get(key)
    if isinstance(values, list) and values:
        return values[-1]
    return row.get(key, default)


def _prepare_bookmark_list_view(df, host):
    display_df = df.copy()
    display_df['symbol_link'] = [f'http://{host}/view_stock?symbol={x}' for x in display_df['symbol']]
    display_df['sector_display'] = display_df.apply(
        lambda row: translate_sector(row.get('sector_name')),
        axis=1,
    )
    display_df['industry_display'] = display_df.apply(
        lambda row: translate_industry(row.get('industry_name')),
        axis=1,
    )

    latest_list_columns = [
        'volume_ratio', 'dry_up_ratio', 'dist_pivot_20', 'ATR_pct', 'RSI',
    ]
    for column in latest_list_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(
                lambda value: value[-1] if isinstance(value, list) and value else value
            )

    if 'market_cap' in display_df.columns:
        display_df['market_cap_b'] = display_df['market_cap'] / 1e9

    for period in ['1m', '3m', '6m']:
        old_column = f'rs_{period}'
        new_column = f'vs_market_{period}'
        if new_column not in display_df.columns and old_column in display_df.columns:
            display_df[new_column] = display_df[old_column]

    for column in ['vs_market_1m', 'vs_market_3m', 'vs_market_6m', 'dist_pivot_20', 'ATR_pct', 'risk_pct']:
        if column in display_df.columns:
            display_df[column] = display_df[column] * 100
    if 'setup_phase' in display_df.columns:
        display_df['setup_phase_display'] = display_df['setup_phase'].apply(setup_phase_display_text)
    if 'setup_status' in display_df.columns:
        display_df['setup_status_display'] = display_df.apply(
            lambda row: _setup_status_display_text(row.get('setup_status'), row.get('setup_phase')),
            axis=1,
        )
    if 'holding_action' in display_df.columns:
        display_df['holding_action_display'] = display_df.apply(
            lambda row: holding_action_display_text(
                normalize_holding_action(row.get('holding_action'), row.get('setup_phase'))
            ),
            axis=1,
        )

    if 'earnings_date' in display_df.columns:
        earnings = pd.to_datetime(display_df['earnings_date'].apply(_parse_earnings_date), errors='coerce')
        earnings = earnings.mask(earnings.dt.year < 2000)
        display_df['earnings_date'] = earnings
        # Match the deterministic rule engine: event risk is measured from
        # the last completed daily data session, not the wall-clock date.
        reference_dates = pd.to_datetime(
            display_df.get('data_date', pd.Series(pd.NaT, index=display_df.index)),
            errors='coerce',
        ).dt.normalize()
        reference_dates = reference_dates.fillna(pd.Timestamp.today().normalize())
        display_df['days_to_earnings'] = (earnings.dt.normalize() - reference_dates).dt.days
        display_df['days_to_earnings_display'] = display_df['days_to_earnings'].apply(
            lambda value: (
                ""
                if pd.isna(value)
                else f"⚠ {int(value)} 日"
                if -1 <= value <= 7
                else f"{int(value)} 日"
            )
        )

    if 'close' in display_df.columns:
        display_df['close'] = display_df['close'].apply(
            lambda value: [round(float(item), 2) for item in value[-20:]]
            if isinstance(value, list)
            else value
        )

    return display_df


def refresh_bookmarks(force_refresh=False):
    bookmarks = st.session_state.bookmark_stocks.copy().reset_index(drop=True)
    if bookmarks.empty:
        return pd.DataFrame()

    batch_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    if force_refresh:
        refresh_targets = bookmarks
        fresh_rows = pd.DataFrame()
    else:
        stale_mask = bookmarks.apply(bookmark_row_is_stale, axis=1)
        refresh_targets = bookmarks[stale_mask]
        fresh_rows = bookmarks[~stale_mask]

    if refresh_targets.empty:
        return bookmarks

    try:
        refreshed = refresh_bookmarked_stocks(
            refresh_targets,
            force_refresh=force_refresh,
            batch_updated_at=batch_updated_at,
        )
    except Exception as e:
        st.toast(f"未能更新收藏股票：{e}")
        return pd.DataFrame()

    if not refreshed.empty:
        st.session_state.bookmark_stocks = (
            pd.concat([fresh_rows, refreshed], ignore_index=True)
            .drop_duplicates(subset=['symbol'], keep='last')
            .reset_index(drop=True)
        )
        _persist_bookmark_order()
        _bump_bookmark_view_version()
    return st.session_state.bookmark_stocks

# UI Starts here
legacy_symbols = (
    bookmark_symbols(st.session_state.get('bookmark_stocks'))
    if str(st.context.headers.get('host', '')).split(':', 1)[0] in {'localhost', '127.0.0.1'}
    else []
)
browser_storage_state = browser_bookmark_storage(
    storage_key='momentum-trading:bookmarks:v1',
    symbols=(
        st.session_state.get('browser_bookmark_symbols', bookmark_symbols(st.session_state.bookmark_stocks))
        if st.session_state.get('browser_bookmarks_loaded')
        else None
    ),
    initial_symbols=legacy_symbols,
    key='browser_bookmark_storage',
)
if not st.session_state.get('browser_bookmarks_loaded'):
    if not isinstance(browser_storage_state, dict) or not browser_storage_state.get('ready'):
        st.stop()
    st.session_state.bookmark_stocks = hydrate_browser_bookmarks(
        browser_storage_state.get('symbols') or [],
        existing=st.session_state.get('bookmark_stocks'),
        analyzed=st.session_state.get('analyzed_stocks'),
    )
    _persist_bookmark_order()

components.html(
    """
    <script>
    const title = '我的股票 | Momentum Trading';
    const parentWindow = window.parent;
    parentWindow.__momentumPageTitle = title;
    const syncTitle = () => {
        if (parentWindow.__momentumPageTitle !== title) {
            observer.disconnect();
            window.clearInterval(timer);
            return;
        }
        if (parentWindow.document.title !== title) parentWindow.document.title = title;
    };
    const observer = new MutationObserver(syncTitle);
    observer.observe(parentWindow.document.head, {childList: true, subtree: true, characterData: true});
    const timer = window.setInterval(syncTitle, 500);
    syncTitle();
    </script>
    """,
    height=0,
)

st.header('我的股票')

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlock"]:has(iframe[title*="streamlit_searchbox"]),
    div[data-testid="stVerticalBlock"]:has(iframe[title*="st_searchbox"]),
    div[data-testid="stVerticalBlock"]:has(iframe[src*="streamlit_searchbox"]) {
        position: relative !important;
        z-index: 100000 !important;
        overflow: visible !important;
    }
    div[data-testid="stElementContainer"]:has(iframe[title*="streamlit_searchbox"]),
    div[data-testid="stElementContainer"]:has(iframe[title*="st_searchbox"]),
    div[data-testid="stElementContainer"]:has(iframe[src*="streamlit_searchbox"]) {
        position: relative !important;
        z-index: 100001 !important;
        min-height: 58px !important;
        overflow: visible !important;
    }
    iframe[title*="streamlit_searchbox"],
    iframe[title*="st_searchbox"],
    iframe[src*="streamlit_searchbox"] {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        width: 100% !important;
        z-index: 100002 !important;
    }
    div[data-testid="stVerticalBlock"]:has(iframe[title*="streamlit_searchbox"]) + div,
    div[data-testid="stVerticalBlock"]:has(iframe[title*="st_searchbox"]) + div,
    div[data-testid="stVerticalBlock"]:has(iframe[src*="streamlit_searchbox"]) + div {
        margin-top: -0.75rem !important;
    }
    div[data-testid="stElementContainer"]:has(iframe[title*="streamlit_searchbox"]) + div[data-testid="stElementContainer"],
    div[data-testid="stElementContainer"]:has(iframe[title*="st_searchbox"]) + div[data-testid="stElementContainer"],
    div[data-testid="stElementContainer"]:has(iframe[src*="streamlit_searchbox"]) + div[data-testid="stElementContainer"],
    div[data-testid="stElementContainer"]:has(iframe[title*="streamlit_searchbox"]) + div[data-testid="stElementContainer"] + div[data-testid="stElementContainer"],
    div[data-testid="stElementContainer"]:has(iframe[title*="st_searchbox"]) + div[data-testid="stElementContainer"] + div[data-testid="stElementContainer"],
    div[data-testid="stElementContainer"]:has(iframe[src*="streamlit_searchbox"]) + div[data-testid="stElementContainer"] + div[data-testid="stElementContainer"],
    div[data-testid="stElementContainer"]:has(iframe[title*="streamlit_searchbox"]) + div[data-testid="stElementContainer"] + div[data-testid="stElementContainer"] + div[data-testid="stElementContainer"],
    div[data-testid="stElementContainer"]:has(iframe[title*="st_searchbox"]) + div[data-testid="stElementContainer"] + div[data-testid="stElementContainer"] + div[data-testid="stElementContainer"],
    div[data-testid="stElementContainer"]:has(iframe[src*="streamlit_searchbox"]) + div[data-testid="stElementContainer"] + div[data-testid="stElementContainer"] + div[data-testid="stElementContainer"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    div[data-testid="stMainBlockContainer"] > div:first-child {
        gap: 0.8rem;
    }
    div.st-key-bookmark_add_stock_help {
        margin-top: -1.6rem;
    }
    div.st-key-bookmark_add_stock_help + div[data-testid="stElementContainer"],
    div.st-key-bookmark_add_stock_help + div[data-testid="stElementContainer"] + div[data-testid="stElementContainer"] {
        margin-top: -0.25rem;
    }
    div.st-key-bookmark_refresh_row {
        margin-top: -1.05rem;
    }
    div.st-key-bookmark_refresh_row > div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    div.st-key-bookmark_refresh_button button {
        min-width: 8.8rem;
        white-space: nowrap;
    }
    div.st-key-bookmark_sort_direction button {
        height: 2.25rem;
        min-height: 2.25rem;
        min-width: 3.15rem;
        padding: 0 0.75rem;
        line-height: 1;
    }
    div.st-key-bookmark_sort_direction button p {
        line-height: 1;
    }
    div.st-key-bookmark_sort_row {
        margin-top: 0.25rem;
    }
    div.st-key-bookmark_card_grid {
        margin-top: 0.2rem;
    }
    div.st-key-bookmark_card_grid .stock-card-heading {
        margin-bottom: 0.9rem;
        min-width: 0;
    }
    div.st-key-bookmark_card_grid .stock-card-ticker {
        color: #fafafa;
        font-size: 1.45rem;
        font-weight: 400;
        line-height: 1.15;
    }
    div.st-key-bookmark_card_grid .stock-card-sector {
        color: rgba(250, 250, 250, 0.66);
        font-size: 0.82rem;
        line-height: 1.25;
        margin-top: 0.2rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    div.st-key-bookmark_card_grid [data-testid="stMetric"] {
        margin-top: 0.15rem;
    }
    @media (max-width: 640px) {
        div.st-key-bookmark_refresh_row > div[data-testid="stHorizontalBlock"],
        div.st-key-bookmark_sort_row > div[data-testid="stHorizontalBlock"] {
            flex-wrap: nowrap !important;
            gap: 0.55rem !important;
        }
        div.st-key-bookmark_refresh_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
        div.st-key-bookmark_sort_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            min-width: 0 !important;
            width: auto !important;
        }
        div.st-key-bookmark_refresh_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) {
            flex: 0 0 auto !important;
        }
        div.st-key-bookmark_refresh_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) {
            flex: 1 1 auto !important;
        }
        div.st-key-bookmark_refresh_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) {
            display: none !important;
        }
        div.st-key-bookmark_sort_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) {
            flex: 0 0 2.2rem !important;
        }
        div.st-key-bookmark_sort_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) {
            flex: 1 1 auto !important;
        }
        div.st-key-bookmark_sort_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) {
            flex: 0 0 3.15rem !important;
        }
        div.st-key-bookmark_sort_row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(4) {
            display: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

components.html(
    f"""
    <script>
    const splitToken = {repr(_STOCK_OPTION_SPLIT)};
    const clearToken = {int(st.session_state.get('stock_searchbox_clear_token', 0))};
    function formatSearchOptions() {{
        const frames = window.parent.document.querySelectorAll(
            'iframe[title*="streamlit_searchbox"], iframe[title*="st_searchbox"], iframe[src*="streamlit_searchbox"]'
        );
        frames.forEach((frame) => {{
            let doc;
            try {{
                doc = frame.contentDocument;
            }} catch (error) {{
                return;
            }}
            if (!doc) return;
            if (clearToken && doc.body.dataset.stockClearToken !== String(clearToken)) {{
                doc.body.dataset.stockClearToken = String(clearToken);
                const clearSelectedValue = () => {{
                    doc.querySelectorAll('[class*="-singleValue"]').forEach((element) => {{
                        if ((element.textContent || '').includes(splitToken)) {{
                            element.remove();
                        }}
                    }});
                    doc.querySelectorAll('#aria-selection').forEach((element) => {{
                        if ((element.textContent || '').includes(splitToken)) {{
                            element.textContent = '';
                        }}
                    }});
                    const input = doc.querySelector('input[id^="react-select"]');
                    if (input && input.value) {{
                        const setter = Object.getOwnPropertyDescriptor(
                            window.HTMLInputElement.prototype,
                            'value'
                        ).set;
                        setter.call(input, '');
                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }}
                }};
                [80, 250, 700].forEach((delay) => {{
                    window.setTimeout(clearSelectedValue, delay);
                }});
                clearSelectedValue();
            }}
            if (!doc.getElementById('stock-option-flex-style')) {{
                const style = doc.createElement('style');
                style.id = 'stock-option-flex-style';
                style.textContent = `
                    [role="option"] .stock-option-row {{
                        align-items: center;
                        display: flex;
                        gap: 1rem;
                        justify-content: space-between;
                        width: 100%;
                    }}
                    [role="option"] .stock-option-left {{
                        min-width: 0;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                    }}
                    [role="option"] .stock-option-right {{
                        align-items: center;
                        color: rgba(250, 250, 250, 0.78);
                        display: inline-flex;
                        flex: 0 0 auto;
                        gap: 0.55rem;
                        margin-left: auto;
                        text-align: right;
                        white-space: nowrap;
                    }}
                    [role="option"] .stock-option-add-icon {{
                        border: 1.5px solid currentColor;
                        border-radius: 999px;
                        display: inline-block;
                        height: 1.05rem;
                        position: relative;
                        width: 1.05rem;
                    }}
                    [role="option"] .stock-option-add-icon::before,
                    [role="option"] .stock-option-add-icon::after {{
                        background: currentColor;
                        content: "";
                        left: 50%;
                        position: absolute;
                        top: 50%;
                        transform: translate(-50%, -50%);
                    }}
                    [role="option"] .stock-option-add-icon::before {{
                        height: 1.5px;
                        width: 0.55rem;
                    }}
                    [role="option"] .stock-option-add-icon::after {{
                        height: 0.55rem;
                        width: 1.5px;
                    }}
                `;
                doc.head.appendChild(style);
            }}
            doc.querySelectorAll('svg path[d="m6 9 6 6 6-6"]').forEach((path) => {{
                const svg = path.closest('svg');
                if (!svg || svg.dataset.stockSearchIcon === '1') return;
                svg.setAttribute('viewBox', '0 0 24 24');
                svg.setAttribute('width', '18');
                svg.setAttribute('height', '18');
                svg.setAttribute('fill', 'none');
                svg.setAttribute('stroke', '#fafafa');
                svg.setAttribute('stroke-width', '2.5');
                svg.setAttribute('stroke-linecap', 'round');
                svg.setAttribute('stroke-linejoin', 'round');
                svg.innerHTML = '<circle cx="11" cy="11" r="7"></circle><path d="m20 20-3.5-3.5"></path>';
                svg.dataset.stockSearchIcon = '1';
            }});
            doc.querySelectorAll('[role="option"]').forEach((option) => {{
                if (option.dataset.stockFormatted === '1') return;
                const text = option.textContent || '';
                if (!text.includes(splitToken)) return;
                const [left, right] = text.split(splitToken);
                option.textContent = '';
                const row = doc.createElement('span');
                row.className = 'stock-option-row';
                const leftSpan = doc.createElement('span');
                leftSpan.className = 'stock-option-left';
                leftSpan.textContent = left.trim();
                const rightSpan = doc.createElement('span');
                rightSpan.className = 'stock-option-right';
                rightSpan.textContent = right.trim();
                const addIcon = doc.createElement('span');
                addIcon.className = 'stock-option-add-icon';
                addIcon.setAttribute('aria-hidden', 'true');
                rightSpan.appendChild(addIcon);
                row.append(leftSpan, rightSpan);
                option.appendChild(row);
                option.dataset.stockFormatted = '1';
            }});
        }});
    }}
    formatSearchOptions();
    window.setInterval(formatSearchOptions, 150);
    </script>
    """,
    height=0,
)

st.session_state.setdefault('stock_searchbox_nonce', 0)
_repair_stock_search_state()
with st.container(key="bookmark_add_stock_help"):
    st.write('輸入股票代號或公司名稱，加入至「我的股票」收藏列表。')
st.caption('收藏清單只會儲存在此瀏覽器；請使用同一個瀏覽器，並避免清除本網站的資料。')
if st_searchbox is not None:
    st_searchbox(
        _stock_search_options,
        label="",
        placeholder="搜尋股票代號或公司名稱",
        default_options=[],
        clear_on_submit=False,
        edit_after_submit="disabled",
        style_absolute=True,
        style_overrides=_STOCK_SEARCH_STYLE_OVERRIDES,
        submit_function=_queue_stock_add,
        key=f"stock_searchbox_{st.session_state.stock_searchbox_nonce}",
    )
else:
    fallback_options = _stock_search_fallback_options()
    fallback_key = f"stock_selectbox_{st.session_state.stock_searchbox_nonce}"
    st.selectbox(
        "搜尋股票",
        options=fallback_options,
        index=None,
        placeholder="搜尋股票代號或公司名稱",
        label_visibility="collapsed",
        key=fallback_key,
        on_change=_queue_selectbox_stock_add,
        args=(fallback_key,),
    )

pending_symbol = st.session_state.pop('pending_stock_search_add', "")
if pending_symbol:
    _analyze_and_bookmark_stock(pending_symbol)

update_max = ""
if 'updated_at' in st.session_state.bookmark_stocks.columns:
    updates = pd.to_datetime(st.session_state.bookmark_stocks['updated_at'], errors='coerce').dropna()
    if not updates.empty:
        update_max = updates.max().strftime('%Y-%m-%d %H:%M')

with st.container(key="bookmark_refresh_row"):
    refresh_col, updated_col, _ = st.columns([1.5, 1.8, 5.2], vertical_alignment="center")
    with refresh_col:
        if st.button('更新股價及訊號', icon=':material/refresh:', key="bookmark_refresh_button"):
            with st.spinner("正在更新已收藏股票..."):
                refresh_bookmarks(force_refresh=True)
            st.session_state.bookmarks_refreshed_this_session = True
            st.session_state.bookmarks_batch_refreshed_this_session = True
            st.rerun()
    with updated_col:
        if update_max:
            st.caption(f"最後更新：{update_max}")

if not st.session_state.get('bookmarks_refreshed_this_session'):
    with st.spinner("正在更新已收藏股票..."):
        refresh_bookmarks(force_refresh=True)
    st.session_state.bookmarks_refreshed_this_session = True
    st.session_state.bookmarks_batch_refreshed_this_session = True

if not st.session_state.get('bookmarks_batch_refreshed_this_session'):
    sync_bookmarks_with_latest_sources()
df_stocks = st.session_state.bookmark_stocks.copy().reset_index(drop=True)
if 'refresh_error' in df_stocks.columns:
    has_cached_data = df_stocks.apply(
        lambda row: (
            isinstance(row.get('close'), (list, tuple)) and len(row.get('close')) > 0
        ) or not pd.isna(row.get('price')),
        axis=1,
    )
    stale_error_mask = df_stocks['refresh_error'].notna() & has_cached_data
    if stale_error_mask.any():
        stale_symbols = df_stocks.loc[stale_error_mask, 'symbol'].tolist()
        st.warning(
            "以下股票暫時顯示上次成功的日線快照："
            f"{', '.join(stale_symbols)}。由於更新失敗，資料可能不是最新。"
        )

if df_stocks.empty:
    st.info("尚未加入任何股票。請先使用股票篩選器，然後加入至「我的股票」。")
    st.stop()

if 'updated_at' in df_stocks.columns:
    updates = pd.to_datetime(df_stocks['updated_at'], errors='coerce').dropna()
    if not updates.empty:
        update_max = updates.max().strftime('%Y-%m-%d %H:%M')
if 'refresh_error' in df_stocks.columns and df_stocks['refresh_error'].notna().any():
    failed_rows = df_stocks[df_stocks['refresh_error'].notna()].copy()
    if 'data_stale' in failed_rows.columns:
        failed_rows = failed_rows[~failed_rows['data_stale'].fillna(False)]
    failed_symbols = failed_rows['symbol'].tolist()
    if failed_symbols:
        st.warning(f"部分股票未能更新：{', '.join(failed_symbols)}")

# @st.fragment(run_every=10) #TODO: Only turn on the live when ready
def _sort_bookmarks(by, ascending=True):
    if st.session_state.bookmark_stocks.empty:
        return

    sorted_df = st.session_state.bookmark_stocks.copy()
    if by == "ticker":
        sorted_df["_sort_symbol"] = sorted_df["symbol"].astype(str).str.upper()
        sorted_df = sorted_df.sort_values("_sort_symbol", ascending=ascending, kind="mergesort")
        sorted_df = sorted_df.drop(columns=["_sort_symbol"])
    elif by == "sector":
        sorted_df["_sort_sector"] = sorted_df.apply(
            lambda row: translate_sector(row.get("sector_name")),
            axis=1,
        )
        sorted_df["_sort_industry"] = sorted_df.apply(
            lambda row: translate_industry(row.get("industry_name")),
            axis=1,
        )
        sorted_df["_sort_symbol"] = sorted_df["symbol"].astype(str).str.upper()
        sorted_df = sorted_df.sort_values(
            ["_sort_sector", "_sort_industry", "_sort_symbol"],
            ascending=[ascending, ascending, True],
            kind="mergesort",
        )
        sorted_df = sorted_df.drop(columns=["_sort_sector", "_sort_industry", "_sort_symbol"])
    else:
        return

    st.session_state.bookmark_stocks = sorted_df.reset_index(drop=True)
    _persist_bookmark_order()
    _bump_bookmark_view_version()
    st.rerun()


def _apply_bookmark_sort():
    sort_state = st.session_state.get("bookmark_sort", {"field": "ticker", "ascending": True})
    _sort_bookmarks(sort_state["field"], ascending=sort_state["ascending"])


def _set_bookmark_sort_field():
    selected_label = st.session_state.get("bookmark_sort_field", "股票代號")
    field = "sector" if selected_label == "行業" else "ticker"
    current = st.session_state.get("bookmark_sort", {"field": "ticker", "ascending": True})
    st.session_state.bookmark_sort = {
        "field": field,
        "ascending": current.get("ascending", True),
    }
    _apply_bookmark_sort()


def _toggle_bookmark_sort_direction():
    current = st.session_state.get("bookmark_sort", {"field": "ticker", "ascending": True})
    st.session_state.bookmark_sort = {
        "field": current.get("field", "ticker"),
        "ascending": not current.get("ascending", True),
    }
    _apply_bookmark_sort()


def render_stock_card_heading(row, ticker):
    sector_industry = sector_industry_zh_text(row)
    sector_markup = (
        f'<div class="stock-card-sector">{html.escape(sector_industry)}</div>'
        if sector_industry
        else ""
    )
    st.markdown(
        (
            '<div class="stock-card-heading">'
            f'<div class="stock-card-ticker">{html.escape(str(ticker))}</div>'
            f"{sector_markup}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_live_card(row, ticker):
    metric_price, pct_change, mini_chart_price, delta_description = get_live_card_data(row, ticker)

    return st.metric(
        label="現價",
        value=f'${metric_price:.2f}',
        delta=f'{pct_change:+.2f}%',
        delta_description=delta_description,
        chart_data=mini_chart_price,
        chart_type='line',
        label_visibility="collapsed",
    )


st.session_state.setdefault("bookmark_sort", {"field": "ticker", "ascending": True})
sort_state = st.session_state.bookmark_sort
sort_field_label = "行業" if sort_state.get("field") == "sector" else "股票代號"
sort_direction_icon = "↑" if sort_state.get("ascending", True) else "↓"
if st.session_state.get("bookmark_sort_field") not in {"股票代號", "行業"}:
    st.session_state.bookmark_sort_field = sort_field_label
with st.container(key="bookmark_sort_row"):
    sort_label_col, sort_field_col, sort_direction_col, _ = st.columns(
        [0.42, 1.15, 0.62, 6.2],
        vertical_alignment="center",
    )
    with sort_label_col:
        st.caption("排序")
    with sort_field_col:
        st.segmented_control(
            "排序欄位",
            ["股票代號", "行業"],
            default=sort_field_label,
            label_visibility="collapsed",
            key="bookmark_sort_field",
        )
        selected_sort_label = st.session_state.get("bookmark_sort_field", sort_field_label)
        if selected_sort_label != sort_field_label:
            _set_bookmark_sort_field()
    with sort_direction_col:
        if st.button(
            sort_direction_icon,
            help="切換升序／降序",
            key="bookmark_sort_direction",
        ):
            _toggle_bookmark_sort_direction()

# Card View

num_cols = 4
with st.container(key="bookmark_card_grid"):
    cols = st.columns(num_cols)
    for index, row in df_stocks.iterrows():
        ticker = row["symbol"]
        setup_status = normalize_setup_status(
            row.get("setup_status", row.get("status", "watchlist")),
            row.get("setup_phase"),
        )
        holding_action = normalize_holding_action(
            row.get("holding_action", "hold"),
            row.get("setup_phase"),
        )
        phase_label, phase_icon, phase_color = setup_phase_badge(row.get("setup_phase"))

        target_col = cols[index % num_cols]

        with target_col.container(border=True):
            with st.container(key=f"stock_card_header_{ticker}"):
                header_col, view_col, delete_col = st.columns([1, 0.14, 0.14], vertical_alignment="top")
                with header_col:
                    render_stock_card_heading(row, ticker)
                with view_col:
                    st.link_button('', f'http://{st.context.headers.get("host")}/view_stock?symbol={ticker}', key=f"view_{ticker}", icon=':material/document_search:', help='查看詳情')
                with delete_col:
                    if st.button('', key=f"delete_{ticker}", icon=':material/delete:', help='刪除'):
                        with st.spinner("正在更新清單..."):
                            delete_stock(ticker)

            render_live_card(row, ticker)

            with st.container(horizontal=True, vertical_alignment='center', gap='xsmall'):
                match setup_status:
                    case 'breakout_buy' | 'breakout':
                        st.badge("空倉：留意突破", icon="🚀", color="green")
                    case 'pullback_wait' | 'pullback_buy' | 'pullback':
                        st.badge("空倉：留意META", icon="🛡️", color="violet")
                    case 'watchlist':
                        st.badge("空倉：觀望後續", icon="👀", color="blue")
                    case 'not_recommended':
                        st.badge("空倉：不宜買入", icon="❌", color="grey")
                match holding_action:
                    case 'add':
                        st.badge("持倉：考慮加倉", icon="➕", color="green")
                    case 'sell_weakness':
                        st.badge("持倉：趁弱賣出", icon="⚠️", color="red")
                    case _:
                        st.badge("持倉：繼續持有", icon="✅", color="blue")
                st.badge(f"趨勢：{phase_label}", icon=phase_icon, color=phase_color)

# List View

if 'setup_status' not in df_stocks.columns and 'status' in df_stocks.columns:
    df_stocks['setup_status'] = df_stocks['status']
if 'setup_phase' in df_stocks.columns:
    df_stocks['setup_status'] = df_stocks.apply(
        lambda row: normalize_setup_status(row.get('setup_status'), row.get('setup_phase')),
        axis=1,
    )
if 'holding_action' not in df_stocks.columns:
    df_stocks['holding_action'] = 'hold'

list_df = _prepare_bookmark_list_view(df_stocks, st.context.headers.get("host"))
visible_columns = [column for column in MY_STOCKS_REVIEW_COLUMNS if column in list_df.columns]

with st.expander("詳細資料表", expanded=False):
    edited_df = st.data_editor(
        list_df[visible_columns],
        column_config=st.session_state.column_config,
        num_rows="delete",
        hide_index=True,
        key=f"bookmark_editor_{st.session_state.get('bookmark_view_version', 0)}",
        disabled=visible_columns,  # 只允許刪除行，不允許改數據
    )

# 💡 同步更新名單：萬一用戶喺 data_editor 撳垃圾桶刪除咗某一行
if len(edited_df) != len(st.session_state.bookmark_stocks):
    if 'symbol_link' in edited_df.columns:
        remaining_symbols = set(
            edited_df['symbol_link'].astype(str).str.extract(r'symbol=([^&]+)', expand=False).dropna()
        )
    else:
        remaining_symbols = set()
    st.session_state.bookmark_stocks = (
        st.session_state.bookmark_stocks[
            st.session_state.bookmark_stocks['symbol'].isin(remaining_symbols)
        ]
        .reset_index(drop=True)
    )
    with st.spinner("正在更新清單..."):
        _persist_bookmark_order()
        _bump_bookmark_view_version()
        st.rerun()
