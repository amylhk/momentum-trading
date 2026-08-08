import streamlit as st
import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
import pandas_market_calendars as mcal
import streamlit.components.v1 as components
from datetime import datetime, timedelta, timezone

from core.analysis.contracts import ScreeningSettings
from core.analysis.explainability import build_rule_checklist, earnings_context, phase_score, rules_passed_text, score_breakdown
from core.i18n.sector_industry_zh import translate_industry, translate_sector
from repositories.storage import load_latest_screener_metadata, save_latest_screener, save_stock_cache
from services.stock_service import (
    build_analyzed_stock,
    completed_daily_ohlcv,
    completed_session_closes_from_intraday,
    repair_completed_session_close,
)
from native_stock_chart import browser_bookmark_storage
from ui.browser_bookmarks import bookmark_symbols, hydrate_browser_bookmarks
from ui.stock_table_columns import SCREENER_REVIEW_COLUMNS, holding_action_display_text
from utils import _parse_earnings_date, normalize_holding_action, normalize_setup_status, setup_phase_display_text


def _classification_badge(setup_status, phase):
    normalized = normalize_setup_status(setup_status, phase)
    labels = {
        "breakout_buy": ("🚀", "空倉：留意突破"),
        "pullback_wait": ("🛡️", "空倉：留意META"),
        "pullback_buy": ("🛡️", "空倉：留意META"),
        "watchlist": ("👀", "空倉：觀望後續"),
        "not_recommended": ("❌", "空倉：不宜買入"),
    }
    icon, text = labels.get(normalized, ("👀", "空倉：觀望後續"))
    return f"{icon} {text}"


def _format_run_timestamp(value):
    if not value:
        return ""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Hong_Kong")
    return timestamp.tz_convert("Asia/Hong_Kong").strftime("%Y-%m-%d %H:%M")


def _last_completed_us_trading_day():
    now_utc = datetime.now(timezone.utc)
    calendar = mcal.get_calendar("NYSE")
    schedule = calendar.schedule(
        start_date=(now_utc - timedelta(days=14)).date(),
        end_date=now_utc.date(),
    )
    if schedule.empty:
        return None

    completed = schedule[schedule["market_close"] <= pd.Timestamp(now_utc)]
    if completed.empty:
        return None
    return str(completed.index[-1].date())


def _latest_result_data_date(stocks=None, metadata=None):
    metadata = metadata or {}
    if metadata.get("data_date"):
        return str(metadata["data_date"])

    data_dates = [
        str(row.get("data_date"))
        for row in (stocks or [])
        if isinstance(row, dict) and row.get("data_date")
    ]
    if data_dates:
        return max(data_dates)
    return None


def _render_latest_result_status(placeholder=None):
    metadata = st.session_state.get("screening_metadata", {}) or {}
    run_display = metadata.get("run_at_display") or _format_run_timestamp(metadata.get("run_at"))
    data_date = _latest_result_data_date(st.session_state.get("analyzed_stocks", []), metadata)
    latest_completed_day = _last_completed_us_trading_day()

    target = placeholder.container() if placeholder is not None else st.container()
    with target:
        status_columns = st.columns(2)
        status_columns[0].caption(f"⏱️ 上次篩選時間：{run_display or '尚未篩選'}")
        status_columns[1].caption(f"📅 資料日期：{data_date or '尚未有資料'}")

        if data_date and latest_completed_day and data_date != latest_completed_day:
            st.warning(
                f"目前顯示的篩選結果截至 {data_date}，不是最新已完成美股交易日 {latest_completed_day}。"
                "請按「開始篩選」重新篩選。"
            )
        elif not data_date:
            st.info("尚未有篩選結果。請按「開始篩選」取得最新結果。")


def _fixed_settings_summary(settings):
    return pd.DataFrame(
        [
            {"分類": "基本資格", "條件": "最低市值", "門檻": f"${settings.market_cap_min / 1e9:.0f}B"},
            {"分類": "基本資格", "條件": "最低股價", "門檻": f"${settings.price_min:.0f}"},
            {"分類": "基本資格", "條件": "三個月平均成交量", "門檻": f"{settings.avg_volume_3m_min / 1_000:.0f}K"},
            {"分類": "基本資格", "條件": "最低 Beta", "門檻": f"{settings.beta_min:.1f}"},
            {"分類": "基本資格", "條件": "最低成交額", "門檻": f"${settings.dollar_volume_min / 1e6:.0f}M"},
            {"分類": "基本資格", "條件": "距 52 週高位最大跌幅", "門檻": f"{settings.max_distance_52w_high:.0%}"},
            {"分類": "第二階段趨勢", "條件": "股價 >= MA50 >= MA200", "門檻": "必須通過" if settings.require_price_above_ma_stack else "不適用"},
            {"分類": "第二階段趨勢", "條件": "MA10 > MA20", "門檻": "必須通過" if settings.require_sma10_above_sma20 else "不適用"},
            {"分類": "第二階段趨勢", "條件": "200 日均線向上", "門檻": "必須通過" if settings.require_ma200_rising else "不適用"},
            {"分類": "第二階段趨勢", "條件": "跑贏大市（1 個月）", "門檻": "必須通過" if settings.require_vs_market_1m else "不適用"},
            {"分類": "第二階段趨勢", "條件": "跑贏大市（3 個月）", "門檻": "必須通過" if settings.require_vs_market_3m else "不適用"},
            {"分類": "第二階段趨勢", "條件": "跑贏大市（6 個月）", "門檻": "必須通過" if settings.require_vs_market_6m else "不適用"},
        ]
    )

yf_list_df = pd.read_csv('data/yf_list.csv')
SP500 = '^GSPC'

# Initialize the session state variable if it doesn't exist
if 'last_screening_time' not in st.session_state:
    st.session_state.last_screening_time = "尚未篩選"

if 'analyzed_stocks' not in st.session_state:
    st.session_state.analyzed_stocks = []
if 'screening_metadata' not in st.session_state:
    st.session_state.screening_metadata = load_latest_screener_metadata()

if st.session_state.last_screening_time == "尚未篩選":
    saved_run_time = _format_run_timestamp(st.session_state.screening_metadata.get("run_at"))
    if saved_run_time:
        st.session_state.last_screening_time = saved_run_time

legacy_bookmark_symbols = (
    bookmark_symbols(st.session_state.get("bookmark_stocks"))
    if str(st.context.headers.get("host", "")).split(":", 1)[0] in {"localhost", "127.0.0.1"}
    else []
)
browser_storage_state = browser_bookmark_storage(
    storage_key="momentum-trading:bookmarks:v1",
    symbols=(
        st.session_state.get("browser_bookmark_symbols", bookmark_symbols(st.session_state.bookmark_stocks))
        if st.session_state.get("browser_bookmarks_loaded")
        else None
    ),
    initial_symbols=legacy_bookmark_symbols,
    key="browser_bookmark_storage",
)
if not st.session_state.get("browser_bookmarks_loaded"):
    if not isinstance(browser_storage_state, dict) or not browser_storage_state.get("ready"):
        st.stop()
    st.session_state.bookmark_stocks = hydrate_browser_bookmarks(
        browser_storage_state.get("symbols") or [],
        existing=st.session_state.get("bookmark_stocks"),
        analyzed=st.session_state.get("analyzed_stocks"),
    )
    st.session_state.browser_bookmarks_loaded = True
    st.session_state.browser_bookmark_symbols = bookmark_symbols(st.session_state.bookmark_stocks)

# UI Starts Here
components.html(
    """
    <script>
    const title = '股票篩選器 | Momentum Trading';
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

st.header('股票篩選器')

st.write('本程式依據 Momentum Trading 的條件，先篩選出符合基本資格與第二階段趨勢的股票。')

settings = ScreeningSettings()
st.session_state.screening_settings = settings.to_dict()

with st.expander("固定篩選規則", expanded=False):
    st.dataframe(
        _fixed_settings_summary(settings),
        hide_index=True,
        use_container_width=True,
    )

latest_result_status = st.empty()
_render_latest_result_status(latest_result_status)

if st.button("開始篩選"):
    current_analyzed_stocks = []
    screen_data_date = None
    with st.status("正在篩選潛在股票", expanded=True) as status:

        q = EquityQuery('and', [
            EquityQuery('is-in', ['exchange', 'NMS', 'NGM', 'NYQ', 'ASE']), #NMS: Nasdaq大型股，NGM：Nasdaq中型股，NYQ：紐交所大型股，ASE：紐交所中型股
            EquityQuery('gte', ['lastclosemarketcap.lasttwelvemonths', settings.market_cap_min]),
            EquityQuery('gte', ['avgdailyvol3m', settings.avg_volume_3m_min]),
            EquityQuery('gte', ['eodprice', settings.price_min]),
            EquityQuery('gt', ['beta', settings.beta_min])
        ])

        all_quotes = []
        page_size = 250
        start_index = 0

        with st.status("正在套用預設篩選條件"):
            while True:
                response = yf.screen(q, offset=start_index, size=page_size, sortAsc = True)
                quotes = response.get('quotes', [])

                if not quotes:
                    break

                for q_data in quotes:
                    # 🌟 先安全提取三個核心價格指標
                    price = q_data.get('regularMarketPrice', 0.0) # Must be regularMarketPrice but no current Price
                    ma50 = q_data.get('fiftyDayAverage', 0.0)
                    ma200 = q_data.get('twoHundredDayAverage', 0.0)
                    avgvol3m = q_data.get('averageDailyVolume3Month', 0.0)

                    if (not settings.require_price_above_ma_stack or price >= ma50 >= ma200) and price > 0 and avgvol3m * price >= settings.dollar_volume_min:
                        extracted_data = {
                            'symbol': q_data.get('symbol'),  # 主鍵，若缺失根本不應存在，故不設 Fallback
                            'name': q_data.get('shortName', '未知'),
                            'exchange': q_data.get('exchange', '未知'),
                            'financial_currency': q_data.get('financialCurrency', '未知'), # TODO: Get Country Again from financial database? hkg classify by currency HKD/CNY
                            'price': price,
                            'ma50': ma50,
                            'ma200': ma200,
                            'avgvol3m': avgvol3m,

                            # --- 強勢度指標 ---
                            'market_cap': q_data.get('marketCap', 0),  # 數字型，用 0 作為安全 Fallback
                            '1y_pct_change': q_data.get('fiftyTwoWeekChangePercent', 0.0),
                            'pct_from_high': q_data.get('fiftyTwoWeekHighChangePercent', 0.0),

                            # --- 機構與風險管理 ---
                            'earnings_date': (
                                q_data.get('earningsTimestamp')
                                or q_data.get('earningsTimestampStart')
                                or q_data.get('earningsTimestampEnd')
                            ),  # 時間戳，用 None 代表未知
                            'is_earnings_estimate': q_data.get('isEarningsDateEstimate', True)
                        }

                        all_quotes.append(extracted_data)

                start_index += page_size

        all_quotes_df = pd.DataFrame(all_quotes)
        if all_quotes_df.empty:
            st.warning("Yahoo Finance 沒有回傳符合條件的股票報價。")
            st.stop()

        earnings_timestamp = pd.to_numeric(all_quotes_df['earnings_date'], errors='coerce')
        earnings_timestamp = earnings_timestamp.where(earnings_timestamp.between(-9_223_372_036, 9_223_372_036))
        all_quotes_df['earnings_date'] = pd.to_datetime(earnings_timestamp, unit='s', errors='coerce')
        filtered_df = all_quotes_df[all_quotes_df['pct_from_high'] >= -settings.max_distance_52w_high]

        prelim_stock_info = pd.merge(filtered_df, yf_list_df, how='left', on='symbol')
        #TODO: Or should I filter the screened results and add the 1/3/6 mth performance?

        prelim_list = prelim_stock_info.symbol.to_list()
        prelim_list.append(SP500) # Fetch S&P 500 as well

        with st.status("正在下載日線價格資料"):
            # Keep raw OHLC. When Yahoo delays Adj Close for the latest completed
            # session, yfinance's default auto-adjustment can blank O/H/L as well.
            prelim_ohlcv = yf.download(
                " ".join(prelim_list),
                period='18mo',
                auto_adjust=False,
                progress=False,
            )[['Open', 'High', 'Low', 'Close', 'Volume']]

            expected_screen_date = _last_completed_us_trading_day()
            quote_closes = (
                all_quotes_df.drop_duplicates('symbol', keep='last')
                .set_index('symbol')['price']
                .to_dict()
            )

            # Yahoo occasionally publishes a completed daily bar with O/H/L/V but
            # leaves Close blank until later. Recover the actual session close from
            # the final regular-session minute instead of falling back to a stale
            # screener quote.
            missing_close_tickers = []
            for ticker in prelim_list:
                if ticker not in prelim_ohlcv.columns.get_level_values(1):
                    continue
                frame = prelim_ohlcv.xs(ticker, level=1, axis=1)
                expected_rows = frame[
                    pd.DatetimeIndex(frame.index).date == pd.Timestamp(expected_screen_date).date()
                ]
                if not expected_rows.empty and pd.isna(expected_rows.iloc[-1].get('Close')):
                    missing_close_tickers.append(ticker)

            for start in range(0, len(missing_close_tickers), 80):
                ticker_chunk = missing_close_tickers[start:start + 80]
                try:
                    intraday = yf.download(
                        " ".join(ticker_chunk),
                        period='5d',
                        interval='1m',
                        prepost=False,
                        auto_adjust=False,
                        progress=False,
                    )
                except Exception:
                    continue
                quote_closes.update(
                    completed_session_closes_from_intraday(intraday, expected_screen_date)
                )

            # Classifications only use completed daily sessions. A live intraday bar
            # may still be visible in Yahoo's download during US market hours.
            daily_frames = {}
            for ticker in prelim_list:
                if ticker not in prelim_ohlcv.columns.get_level_values(1):
                    continue
                frame = prelim_ohlcv.xs(ticker, level=1, axis=1)
                frame = repair_completed_session_close(
                    frame,
                    expected_screen_date,
                    quote_closes.get(ticker),
                )
                frame = completed_daily_ohlcv(frame, ticker)
                if not frame.empty:
                    daily_frames[ticker] = frame

            if SP500 in daily_frames and not daily_frames[SP500].empty:
                screen_data_date = str(daily_frames[SP500].index[-1].date())
            else:
                frame_dates = [
                    str(frame.index[-1].date())
                    for frame in daily_frames.values()
                    if not frame.empty
                ]
                screen_data_date = max(frame_dates) if frame_dates else None

            close_prices = pd.DataFrame({
                ticker: frame['Close'] for ticker, frame in daily_frames.items()
            })

            has_sp500_benchmark = (
                SP500 in close_prices.columns
                and close_prices[SP500].dropna().shape[0] >= 126
            )
            if has_sp500_benchmark:
                sp500_close = close_prices[SP500]
                sp500_pct_1m = (sp500_close.iloc[-1] / sp500_close.iloc[-21]) - 1
                sp500_pct_3m = (sp500_close.iloc[-1] / sp500_close.iloc[-63]) - 1
                sp500_pct_6m = (sp500_close.iloc[-1] / sp500_close.iloc[-126]) - 1
            else:
                st.warning("未能下載足夠 S&P 500 資料；今次略過相對大市篩選條件。")
                sp500_pct_1m = sp500_pct_3m = sp500_pct_6m = None

            ma200_history = close_prices.rolling(window=200).mean()
            ma200_is_trending_up = ma200_history.iloc[-1] > ma200_history.iloc[-6]

            ma10 = close_prices.rolling(window=10).mean().iloc[-1]
            ma20 = close_prices.rolling(window=20).mean().iloc[-1]

            # 計算 1, 3, 6 個月 Price Pct Change (用交易日估算：1m=21, 3m=63, 6m=126)
            pct_1m = (close_prices.iloc[-1] / close_prices.iloc[-21]) - 1
            pct_3m = (close_prices.iloc[-1] / close_prices.iloc[-63]) - 1
            pct_6m = (close_prices.iloc[-1] / close_prices.iloc[-126]) - 1

            # 💡 終極篩選條件
            # 10MA > 20MA，200MA向上，1,3,6個月回報皆優於大市
            condition = (
                ((ma10 > ma20) if settings.require_sma10_above_sma20 else True)
                & (ma200_is_trending_up if settings.require_ma200_rising else True)
                & ((pct_1m > sp500_pct_1m) if settings.require_vs_market_1m and has_sp500_benchmark else True)
                & ((pct_3m > sp500_pct_3m) if settings.require_vs_market_3m and has_sp500_benchmark else True)
                & ((pct_6m > sp500_pct_6m) if settings.require_vs_market_6m and has_sp500_benchmark else True)
            )
            vs_market_1m = pct_1m - sp500_pct_1m if has_sp500_benchmark else pd.NA
            vs_market_3m = pct_3m - sp500_pct_3m if has_sp500_benchmark else pd.NA
            vs_market_6m = pct_6m - sp500_pct_6m if has_sp500_benchmark else pd.NA
            relative_strength_df = pd.DataFrame({
                'symbol': close_prices.columns,
                'vs_market_1m': vs_market_1m,
                'vs_market_3m': vs_market_3m,
                'vs_market_6m': vs_market_6m,
                'stock_pct_1m': pct_1m,
                'stock_pct_3m': pct_3m,
                'stock_pct_6m': pct_6m,
                'sp500_pct_1m': sp500_pct_1m,
                'sp500_pct_3m': sp500_pct_3m,
                'sp500_pct_6m': sp500_pct_6m,
                'ma200_is_trending_up': ma200_is_trending_up,
            })
            prelim_stock_info = pd.merge(prelim_stock_info, relative_strength_df, how='left', on='symbol')

            # 抽出過關的名單
            passed_tickers = condition[condition == True].index.tolist()
            if SP500 in passed_tickers:
                passed_tickers.remove(SP500)

            passed_tickers_ohlcv = {
                ticker: daily_frames[ticker]
                for ticker in passed_tickers
                if ticker in daily_frames
            }

        status.update(label="篩選完成", state="complete", expanded=False)

    with (st.status('正在進行技術分析', expanded=True)):
        ticker_frames = passed_tickers_ohlcv.copy()
        passed_tickers = list(ticker_frames)
        st.write(f"📊 篩選出 {len(passed_tickers)} 隻符合第二階段條件的股票")
        st.write("🎯 正在進行技術分析【突破 / 回調 / 觀望】分類")

        for ticker in passed_tickers:
            ohlcv = ticker_frames[ticker]

            meta_rows = prelim_stock_info[prelim_stock_info.symbol == ticker].to_dict(orient='records')
            if not meta_rows:
                continue
            stock_dict = build_analyzed_stock(meta_rows[0], ohlcv, settings=settings, record_history=True)
            current_analyzed_stocks.append(stock_dict)

            save_stock_cache(ticker, stock_dict, source='screener')

        st.session_state.analyzed_stocks = current_analyzed_stocks
        run_at = pd.Timestamp.now(tz="Asia/Hong_Kong").isoformat()
        metadata = {
            "run_at": run_at,
            "run_at_display": _format_run_timestamp(run_at),
            "data_date": _latest_result_data_date(current_analyzed_stocks, {"data_date": screen_data_date}),
            "expected_data_date": _last_completed_us_trading_day(),
            "settings_fingerprint": settings.fingerprint,
            "result_count": len(current_analyzed_stocks),
        }
        st.session_state.screening_metadata = metadata
        st.session_state.last_screening_time = metadata["run_at_display"]
        save_latest_screener(current_analyzed_stocks, metadata=metadata)
        # Start a clean render from the newly saved snapshot. Rewriting the
        # multi-element placeholder in the same run can leave its old warning
        # visible beside the new dataframe until the browser is refreshed.
        st.rerun()

if len(st.session_state.analyzed_stocks) > 0:
    st.header("🏁 分析結果")

    full_df = pd.DataFrame(st.session_state.analyzed_stocks) # full_df should be kept intact
    if 'setup_status' not in full_df.columns and 'status' in full_df.columns:
        full_df['setup_status'] = full_df['status']
    if 'holding_action' not in full_df.columns:
        full_df['holding_action'] = 'hold'
    if 'setup_phase' not in full_df.columns:
        full_df['setup_phase'] = 'unclear_structure'
    else:
        full_df['setup_phase'] = full_df['setup_phase'].fillna('unclear_structure')
    full_df['setup_status'] = full_df.apply(
        lambda row: normalize_setup_status(row.get('setup_status'), row.get('setup_phase')),
        axis=1,
    )

    df = full_df.copy()
    df['sector_display'] = df.apply(lambda row: translate_sector(row.get('sector_name')), axis=1)
    df['industry_display'] = df.apply(lambda row: translate_industry(row.get('industry_name')), axis=1)
    latest_list_columns = [
        'volume_ratio', 'dry_up_ratio', 'dist_pivot_20', 'ATR_pct', 'RSI',
    ]
    for column in latest_list_columns:
        if column in df.columns:
            df[column] = df[column].apply(lambda value: value[-1] if isinstance(value, list) and value else value)

    df['market_cap_b'] = df['market_cap'] / 1e9
    for period in ['1m', '3m', '6m']:
        old_column = f'rs_{period}'
        new_column = f'vs_market_{period}'
        if new_column not in df.columns and old_column in df.columns:
            df[new_column] = df[old_column]

    for column in ['vs_market_1m', 'vs_market_3m', 'vs_market_6m', 'dist_pivot_20', 'ATR_pct', 'risk_pct']:
        if column in df.columns:
            df[column] = df[column] * 100
    df['setup_phase_display'] = df['setup_phase'].apply(setup_phase_display_text)
    df['holding_action_display'] = df.apply(
        lambda row: holding_action_display_text(
            normalize_holding_action(row.get('holding_action'), row.get('setup_phase'))
        ),
        axis=1,
    )

    if 'earnings_date' in df.columns:
        df['earnings_date'] = pd.to_datetime(df['earnings_date'].apply(_parse_earnings_date), errors='coerce')
        df['earnings_date'] = df['earnings_date'].mask(df['earnings_date'].dt.year < 2000)
    else:
        df['earnings_date'] = pd.NaT
    earnings_contexts = df.apply(lambda row: earnings_context(row.to_dict()), axis=1)
    df['days_to_earnings'] = earnings_contexts.apply(lambda value: value['days'])
    df['days_to_earnings_display'] = earnings_contexts.apply(
        lambda value: "需覆核" if value['days'] is None else f"⚠ {value['days']} 日" if value['status'] == 'Needs review' else f"{value['days']} 日"
    )
    df['earnings_warning'] = earnings_contexts.apply(lambda value: value['warning'])
    df['view_link'] = [f'http://{st.context.headers.get("host")}/view_stock?symbol={x}' for x in df['symbol']]
    df['symbol_link'] = df['view_link']

    def classification_bucket(phase):
        if phase in {'near_breakout', 'fresh_breakout'}:
            return 'Breakout'
        if phase in {'pullback_forming', 'pullback_entry'}:
            return 'Pullback'
        return 'Watch'

    df['category_bucket'] = df['setup_phase'].apply(classification_bucket)
    df['classification'] = df.apply(
        lambda row: _classification_badge(row.get('setup_status'), row.get('setup_phase')),
        axis=1,
    )
    scores = df.apply(lambda row: phase_score(row.to_dict()), axis=1)
    df['score'] = scores.apply(lambda value: value[0])
    df['score_label'] = scores.apply(lambda value: value[1])
    score_details = df.apply(lambda row: score_breakdown(row.to_dict()), axis=1)
    df['score_equivalent'] = score_details.apply(lambda value: value['score_display'])
    df['score_label'] = score_details.apply(lambda value: value['display_label'])
    df['score_basis'] = score_details.apply(lambda value: value['basis'])
    # Rebuild this from the current ruleset instead of trusting a persisted
    # snapshot field. Older snapshots used a different denominator and could
    # show e.g. 20/23 after the checklist had changed.
    df['rules_passed'] = full_df.apply(
        lambda row: rules_passed_text(
            build_rule_checklist(
                row.to_dict(),
                ScreeningSettings.from_mapping(row.get('settings')),
            )
        ),
        axis=1,
    ).to_numpy()
    df['data_date'] = df.get('data_date', pd.Series('', index=df.index)).fillna('')
    df['notes'] = df.get('deterministic_summary', df['setup_caption']).fillna(df['setup_caption'])
    rank_order = {'Breakout': 0, 'Pullback': 1, 'Watch': 2}
    df['_rank'] = df['category_bucket'].map(rank_order)
    df = df.sort_values(['_rank', 'score', 'data_date', 'vs_market_3m'], ascending=[True, False, False, False])

    category_labels = {'📋 全部': 'All', '🚀 突破': 'Breakout', '🛡️ META': 'Pullback', '👀 觀望': 'Watch'}
    category = st.segmented_control(
        '分類',
        list(category_labels),
        default='📋 全部',
        key='screener_category_filter',
    )
    selected_bucket = category_labels[category]
    visible = df if selected_bucket == 'All' else df[df['category_bucket'] == selected_bucket]
    visible = visible[[column for column in SCREENER_REVIEW_COLUMNS if column in visible.columns]]
    st.caption(
        '點擊「代號」連結可開啟個股圖表及詳細分析；勾選最左側的股票選取框後，再按「加入至我的股票」'
        '即可加入收藏名單。觀望類股票僅供排序，不代表可直接交易。所有數字均採用最後一個完成的日線交易時段。'
    )
    selection_nonce = st.session_state.setdefault('screener_selection_nonce', 0)
    event = st.dataframe(
        visible,
        hide_index=True,
        use_container_width=True,
        key=f'mvp_results_{selection_nonce}',
        column_config=st.session_state.column_config,
        on_select='rerun',
        selection_mode='multi-row',
    )
    selected_rows = event.selection.rows
    added_count = st.session_state.pop("bookmark_add_count", None)
    if added_count is not None:
        st.success(f'已加入 {added_count} 隻股票至「我的股票」。')
    if st.button(f"加入 {len(selected_rows)}/{len(visible)} 隻至「我的股票」", type='primary') and selected_rows:
        selected_symbols = visible.iloc[selected_rows]['symbol_link'].str.extract(r'symbol=([^&]+)')[0].tolist()
        rows_to_add = full_df[full_df.symbol.isin(selected_symbols)]
        existing_symbols = set(bookmark_symbols(st.session_state.bookmark_stocks))
        new_symbols = [
            symbol for symbol in selected_symbols
            if str(symbol).upper() not in existing_symbols
        ]
        st.session_state.bookmark_stocks = pd.concat([st.session_state.bookmark_stocks, rows_to_add]).drop_duplicates(subset=['symbol'], keep='last').reset_index(drop=True)
        st.session_state.browser_bookmark_symbols = bookmark_symbols(st.session_state.bookmark_stocks)
        st.session_state.bookmark_add_count = len(new_symbols)
        st.session_state.screener_selection_nonce += 1
        st.rerun()
