import pandas as pd


def comparable_timestamp(value):
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.Timestamp.min
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def row_updated_at(row_data):
    return comparable_timestamp(row_data.get("updated_at"))


def row_series_last_date(row_data):
    dates = row_data.get("date")
    if not isinstance(dates, list) or not dates:
        return pd.Timestamp.min
    return comparable_timestamp(dates[-1])


def row_freshness(row_data):
    """Rank snapshots by their stated date and actual saved series coverage."""
    data_date = comparable_timestamp(row_data.get("data_date"))
    dates = row_data.get("date")
    return (
        data_date,
        row_series_last_date(row_data),
        len(dates) if isinstance(dates, list) else 0,
        row_updated_at(row_data),
    )


def row_ohlcv_frame(row_data):
    """Build daily OHLCV from a saved screener/detail snapshot when complete."""
    dates = row_data.get("date")
    if not isinstance(dates, list) or not dates:
        return pd.DataFrame()

    source_fields = {
        "Open": ("open", "Open"),
        "High": ("high", "High"),
        "Low": ("low", "Low"),
        "Close": ("close", "Close"),
        "Volume": ("volume", "Volume"),
    }
    columns = {}
    for output_name, candidates in source_fields.items():
        values = next(
            (row_data.get(name) for name in candidates if isinstance(row_data.get(name), list)),
            None,
        )
        if values is None or len(values) != len(dates):
            return pd.DataFrame()
        columns[output_name] = values

    frame = pd.DataFrame(columns, index=pd.to_datetime(dates, errors="coerce"))
    frame = frame[~frame.index.isna()].sort_index()
    return frame[~frame.index.duplicated(keep="last")]
