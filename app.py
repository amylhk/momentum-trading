import streamlit as st

from repositories.storage import load_bookmarks, load_latest_screener, load_latest_screener_metadata

st.set_page_config(page_title="我的股票 | Momentum Trading", page_icon=':rocket:', layout="wide")

if 'bookmark_stocks' not in st.session_state:
    st.session_state.bookmark_stocks = load_bookmarks()

if 'analyzed_stocks' not in st.session_state:
    st.session_state.analyzed_stocks = load_latest_screener()

if 'screening_metadata' not in st.session_state:
    st.session_state.screening_metadata = load_latest_screener_metadata()

column_config = {
            'symbol': '代號',
            'symbol_link': st.column_config.LinkColumn(
                "代號 ↗",
                help="點擊代號查看股票詳情。",
                display_text=r".*[?&]symbol=([^&]+).*",
                pinned=True,
            ),
            'name': '名稱',
            'price': st.column_config.NumberColumn(
                label="股價",
                format="$%,.2f",
                help="分析所使用的最近收市價或最近更新價格。",
            ),
            'close': st.column_config.LineChartColumn(
                label="價格圖（20日）",
                color="auto",
                help="最近 20 個交易日收市價迷你走勢圖。",
            ),
            'sector_name': st.column_config.TextColumn("行業", help="Yahoo Finance 行業分類。"),
            'industry_name': st.column_config.TextColumn("子行業", help="Yahoo Finance 子行業分類。"),
            'sector_display': st.column_config.TextColumn("行業", help="Yahoo Finance 行業分類（中文顯示）。"),
            'industry_display': st.column_config.TextColumn("子行業", help="Yahoo Finance 子行業分類（中文顯示）。"),
            'setup_status': st.column_config.TextColumn("進場設定", help="技術篩選器的進場分類。"),
            'setup_status_display': st.column_config.TextColumn("進場設定", help="以圖示和中文顯示的進場分類。"),
            'setup_phase': st.column_config.TextColumn(
                "趨勢階段",
                help="詳細的結構階段。確認突破及確認回調可作行動參考；接近突破、已延伸、突破失敗和回調形成中則需要覆核。",
            ),
            'setup_phase_display': st.column_config.TextColumn(
                "趨勢階段",
                help="以圖示顯示的詳細結構階段。",
            ),
            'setup_caption': st.column_config.TextColumn("進場說明", help="按階段提供的行動說明，例如等待突破確認或回調反轉。"),
            'classification': st.column_config.TextColumn("分類"),
            'score': st.column_config.NumberColumn("評分", format="%d"),
            'score_equivalent': st.column_config.TextColumn("評分"),
            'score_label': st.column_config.TextColumn("評分類型"),
            'score_basis': st.column_config.TextColumn("評分依據"),
            'rules_passed': st.column_config.TextColumn("規則通過數"),
            'data_date': st.column_config.TextColumn("資料日期"),
            'notes': st.column_config.TextColumn("備註"),
            'holding_action': st.column_config.TextColumn("持倉建議", help="只適用於已持有該股票時的建議。"),
            'holding_action_display': st.column_config.TextColumn("持倉建議", help="只適用於已持有該股票時的建議。"),
            'status': '進場設定',
            'market_cap': st.column_config.NumberColumn(
                label="市值",
                format="$%,d",
                help="公司市值。",
            ),
            'market_cap_b': st.column_config.NumberColumn(
                label="市值",
                format="$%.2fB",
                help="以十億美元顯示的公司市值。",
            ),
            'avgvol3m': st.column_config.NumberColumn(label="三個月平均成交量", format="%,d"),
            'vs_market_1m': st.column_config.NumberColumn(label="跑贏大市 1 個月", format="%.1f%%", help="個股 1 個月回報減去 S&P 500 同期回報；數值愈高代表同期跑贏大市愈多。"),
            'vs_market_3m': st.column_config.NumberColumn(label="跑贏大市 3 個月", format="%.1f%%", help="個股 3 個月回報減去 S&P 500 同期回報；數值愈高代表同期跑贏大市愈多。"),
            'vs_market_6m': st.column_config.NumberColumn(label="跑贏大市 6 個月", format="%.1f%%", help="個股 6 個月回報減去 S&P 500 同期回報；數值愈高代表同期跑贏大市愈多。"),
            'volume_ratio': st.column_config.NumberColumn(label="成交量比率", format="%.2fx", help="當日成交量除以 20 日平均成交量。突破時較高有助確認需求；過高亦可能代表已延伸。"),
            'dry_up_ratio': st.column_config.NumberColumn(label="量能收縮", format="%.2fx", help="最近 3 日平均成交量除以 20 日平均成交量。回調時較低可反映建設性的量能收縮。"),
            'dist_pivot_20': st.column_config.NumberColumn(label="距前 20 日高位", format="%.1f%%", help="最近收市價相對前 20 日高位的距離；正數代表已高於該突破位。"),
            'ATR_pct': st.column_config.NumberColumn(label="波動率", format="%.1f%%", help="14 日 ATR 除以收市價；愈低代表日內區間較緊，愈高代表波動和止損距離較闊。"),
            'entry_price': st.column_config.NumberColumn(label="入場參考", format="$%,.2f", help="由進場階段推導的入場或觸發參考價。"),
            'stop_price': st.column_config.NumberColumn(label="止損參考", format="$%,.2f", help="由最近技術支持及 ATR 緩衝推導的止損參考；支持不清晰時以 8% 數學止損作後備。"),
            'risk_pct': st.column_config.NumberColumn(label="風險", format="%.1f%%", help="入場參考價與止損參考價之差，除以入場參考價。"),
            'entry_note': st.column_config.TextColumn("入場說明", help="簡述入場和止損參考價的計算方式。"),
            'breakout_score': st.column_config.NumberColumn(label="突破評分", format="%d", help="突破結構證據評分，滿分 90。"),
            'meta_score': st.column_config.NumberColumn(label="META 評分", format="%d", help="回調／META 結構證據評分，滿分 80。"),
            'RSI': st.column_config.NumberColumn(label="RSI", format="%.1f", help="14 日 RSI；約 50 至 75 通常較符合動能階段，過高可能代表延伸。"),
            'trendline_dist': st.column_config.NumberColumn(label="趨勢線距離", format="%.1f%%", help="實驗性趨勢線指標：最近收市價相對估算阻力趨勢線的距離。"),
            'trendline_slope': st.column_config.NumberColumn(label="趨勢線斜率", format="%.2f%%", help="實驗性趨勢線指標：估算阻力趨勢線的每日斜率。"),
            'trendline_breakout': st.column_config.CheckboxColumn(label="突破趨勢線", help="最近收市價是否高於估算阻力趨勢線。"),
            'days_to_earnings_display': st.column_config.TextColumn(label="距業績日", help="距下一次業績公布的日數；未來 7 個曆日內需在入場前覆核。"),
            'earnings_warning': st.column_config.TextColumn(label="業績風險", help="獨立風險提示；因利潤緩衝取決於持倉成本，故不會直接改變技術分數。"),
            'days_to_earnings': st.column_config.NumberColumn(label="距業績日", format="%d 日", help="距下一次業績公布的日數。"),
            'earnings_date': st.column_config.DateColumn(label="業績日期"),
            'view_link': st.column_config.LinkColumn(
                "查看詳情",
                help="點擊查看詳情",
                display_text=":material/open_in_new:",
                pinned=True
            ),
        }

if 'column_config' not in st.session_state:
    st.session_state.column_config = column_config
else:
    st.session_state.column_config.update(column_config)

# Read CSS

with open('style.css') as f:
    st.html(f"<style>{f.read()}</style>")

# Define pages
my_stocks = st.Page("pages/my_stocks.py", title="我的股票", icon="⭐")
screener = st.Page("pages/screener.py", title="股票篩選器", icon="🔎", url_path="screener")
# Preserve the detail links already stored in bookmarks and screener snapshots.
view_stock = st.Page("pages/stock_detail.py", url_path="view_stock", visibility="hidden")

# Configure navigation without a visible section heading.
nav = st.navigation([my_stocks, screener, view_stock])

nav.run()
