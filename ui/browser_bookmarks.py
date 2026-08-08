import pandas as pd

from repositories.storage import load_stock_cache


def bookmark_symbols(bookmarks):
    if not isinstance(bookmarks, pd.DataFrame) or "symbol" not in bookmarks.columns:
        return []
    return list(dict.fromkeys(
        bookmarks["symbol"].dropna().astype(str).str.upper().str.strip().tolist()
    ))


def hydrate_browser_bookmarks(symbols, *, existing=None, analyzed=None):
    existing = existing if isinstance(existing, pd.DataFrame) else pd.DataFrame()
    existing_rows = {
        str(row.get("symbol", "")).upper(): row.to_dict()
        for _, row in existing.iterrows()
        if str(row.get("symbol", "")).strip()
    }
    analyzed_rows = analyzed.to_dict(orient="records") if isinstance(analyzed, pd.DataFrame) else analyzed
    for row in analyzed_rows or []:
        if isinstance(row, dict) and row.get("symbol"):
            existing_rows.setdefault(str(row["symbol"]).upper(), row)

    hydrated = []
    for symbol in symbols:
        ticker = str(symbol).upper().strip()
        row = existing_rows.get(ticker) or load_stock_cache(ticker)
        hydrated.append(dict(row) if isinstance(row, dict) else {"symbol": ticker})
    return pd.DataFrame(hydrated)
