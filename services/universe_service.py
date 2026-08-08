from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


UNIVERSE_PATH = Path("data/yf_list.csv")


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "nan"}:
        return ""
    return text


@lru_cache(maxsize=1)
def load_symbol_universe(path: str = str(UNIVERSE_PATH)) -> pd.DataFrame:
    universe_path = Path(path)
    if not universe_path.exists():
        return pd.DataFrame(columns=["symbol", "name", "sector_name", "industry_name", "exchange"])

    df = pd.read_csv(universe_path)
    if "symbol" not in df.columns:
        return pd.DataFrame(columns=["symbol", "name", "sector_name", "industry_name", "exchange"])

    for column in ["name", "shortName", "longName", "sector_name", "industry_name", "exchange"]:
        if column not in df.columns:
            df[column] = ""

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["name"] = df[["name", "shortName", "longName"]].bfill(axis=1).iloc[:, 0].fillna("")
    df = df[df["symbol"].ne("")].drop_duplicates(subset=["symbol"], keep="first")
    return df.reset_index(drop=True)


def format_symbol_option(row: pd.Series) -> tuple[str, dict[str, str]]:
    symbol = _clean_text(row.get("symbol")).upper()
    name = _clean_text(row.get("name"))
    sector = _clean_text(row.get("sector_name"))
    industry = _clean_text(row.get("industry_name"))
    exchange = _clean_text(row.get("exchange"))

    details = [value for value in [name, exchange, sector, industry] if value]
    label = symbol if not details else f"{symbol} · {' · '.join(details)}"
    return label, {
        "symbol": symbol,
        "name": name,
        "sector_name": sector,
        "industry_name": industry,
        "exchange": exchange,
    }


def search_symbol_universe(searchterm: str, limit: int = 20) -> list[tuple[str, dict[str, str]]]:
    query = _clean_text(searchterm).upper()
    if not query:
        return []

    df = load_symbol_universe()
    if df.empty:
        return [(query, {"symbol": query})]

    symbol = df["symbol"].astype(str).str.upper()
    name = df["name"].astype(str).str.upper()
    sector = df["sector_name"].astype(str).str.upper()
    industry = df["industry_name"].astype(str).str.upper()

    matches = df[
        symbol.str.contains(query, na=False)
        | name.str.contains(query, na=False)
        | sector.str.contains(query, na=False)
        | industry.str.contains(query, na=False)
    ].copy()

    if matches.empty:
        return [(query, {"symbol": query})]

    matches["_rank"] = 100
    matches.loc[symbol[matches.index].eq(query), "_rank"] = 0
    matches.loc[symbol[matches.index].str.startswith(query, na=False), "_rank"] = matches["_rank"].clip(upper=5)
    matches.loc[name[matches.index].str.startswith(query, na=False), "_rank"] = matches["_rank"].clip(upper=20)
    matches.loc[name[matches.index].str.contains(query, na=False), "_rank"] = matches["_rank"].clip(upper=30)
    matches.loc[sector[matches.index].str.contains(query, na=False), "_rank"] = matches["_rank"].clip(upper=40)
    matches.loc[industry[matches.index].str.contains(query, na=False), "_rank"] = matches["_rank"].clip(upper=45)

    matches = matches.sort_values(["_rank", "symbol"]).head(limit)
    options = [format_symbol_option(row) for _, row in matches.iterrows()]
    if query not in {option[1].get("symbol") for option in options}:
        options.insert(0, (query, {"symbol": query}))
    return options[:limit]


def selected_symbol(value: Any) -> str:
    if isinstance(value, dict):
        return _clean_text(value.get("symbol")).upper()
    if isinstance(value, str):
        return value.split("·", 1)[0].strip().upper()
    return ""
