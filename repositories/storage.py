import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


DATA_DIR = Path('data')
STOCK_CACHE_DIR = DATA_DIR / 'stock_cache'
OHLCV_CACHE_DIR = DATA_DIR / 'ohlcv_cache'
ANALYSIS_HISTORY_DIR = DATA_DIR / 'analysis_history'
BOOKMARKS_PATH = DATA_DIR / 'bookmark_stocks.pkl'
LATEST_SCREENER_PATH = DATA_DIR / 'latest_screener_results.pkl'


def _ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    STOCK_CACHE_DIR.mkdir(exist_ok=True)
    OHLCV_CACHE_DIR.mkdir(exist_ok=True)
    ANALYSIS_HISTORY_DIR.mkdir(exist_ok=True)


def _read_pickle(path, default):
    if not path.exists():
        return default

    try:
        with path.open('rb') as f:
            return pickle.load(f)
    except Exception:
        return default


def _write_pickle(path, data):
    _ensure_dirs()
    with path.open('wb') as f:
        pickle.dump(data, f)


def load_bookmarks():
    data = _read_pickle(BOOKMARKS_PATH, pd.DataFrame())
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def save_bookmarks(bookmarks):
    if bookmarks is None:
        bookmarks = pd.DataFrame()
    if not isinstance(bookmarks, pd.DataFrame):
        bookmarks = pd.DataFrame(bookmarks)
    _write_pickle(BOOKMARKS_PATH, bookmarks.reset_index(drop=True))


def _latest_screener_payload(data):
    if isinstance(data, dict) and 'stocks' in data:
        stocks = data.get('stocks') or []
        metadata = data.get('metadata') or {}
    elif isinstance(data, pd.DataFrame):
        stocks = data.to_dict(orient='records')
        metadata = {}
    else:
        stocks = data or []
        metadata = {}

    if isinstance(stocks, pd.DataFrame):
        stocks = stocks.to_dict(orient='records')
    if not isinstance(stocks, list):
        stocks = []
    if not isinstance(metadata, dict):
        metadata = {}

    if not metadata.get('data_date'):
        data_dates = [
            str(row.get('data_date'))
            for row in stocks
            if isinstance(row, dict) and row.get('data_date')
        ]
        if data_dates:
            metadata = {**metadata, 'data_date': max(data_dates)}

    return stocks, metadata


def load_latest_screener():
    stocks, _ = _latest_screener_payload(_read_pickle(LATEST_SCREENER_PATH, []))
    return stocks


def load_latest_screener_metadata():
    _, metadata = _latest_screener_payload(_read_pickle(LATEST_SCREENER_PATH, []))
    return metadata


def save_latest_screener(stocks, metadata=None):
    if isinstance(stocks, pd.DataFrame):
        stocks = stocks.to_dict(orient='records')
    _write_pickle(LATEST_SCREENER_PATH, {
        'stocks': stocks or [],
        'metadata': metadata or {},
    })


def _stock_cache_path(symbol):
    clean_symbol = symbol.upper().replace('/', '-')
    return STOCK_CACHE_DIR / f'{clean_symbol}.pkl'


def _ohlcv_cache_path(symbol):
    clean_symbol = symbol.upper().replace('/', '-')
    return OHLCV_CACHE_DIR / f'{clean_symbol}.pkl'


def _analysis_history_path(symbol):
    clean_symbol = symbol.upper().replace('/', '-')
    return ANALYSIS_HISTORY_DIR / f'{clean_symbol}.jsonl'


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, 'item'):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def record_analysis_snapshot(snapshot):
    """Append one deterministic daily state and return the snapshot data date."""
    _ensure_dirs()
    payload = _json_safe(dict(snapshot))
    path = _analysis_history_path(payload['symbol'])
    records = []
    if path.exists():
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    identity = (
        payload.get('data_date'),
        payload.get('settings_fingerprint'),
        payload.get('ruleset_version'),
    )
    if not any(
        (item.get('data_date'), item.get('settings_fingerprint'), item.get('ruleset_version')) == identity
        for item in records
    ):
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + '\n')
        records.append(payload)

    return payload.get('data_date')


def save_stock_cache(symbol, stock, source):
    payload = {
        'symbol': symbol.upper(),
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'source': source,
        'data': dict(stock),
    }
    _write_pickle(_stock_cache_path(symbol), payload)


def load_stock_cache(symbol, max_age_hours=None):
    payload = _read_pickle(_stock_cache_path(symbol), None)
    if not payload:
        return None

    if max_age_hours is not None:
        try:
            updated_at = datetime.fromisoformat(payload.get('updated_at'))
        except (TypeError, ValueError):
            return None
        if datetime.now() - updated_at > timedelta(hours=max_age_hours):
            return None

    return payload.get('data')


def save_ohlcv_cache(symbol, ohlcv, market_profile):
    payload = {
        'symbol': symbol.upper(),
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'market_profile': market_profile,
        'data': ohlcv,
    }
    _write_pickle(_ohlcv_cache_path(symbol), payload)


def load_ohlcv_cache(symbol):
    return _read_pickle(_ohlcv_cache_path(symbol), None)

