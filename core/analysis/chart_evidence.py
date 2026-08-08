"""Deterministic price-level evidence shared by scoring and chart consumers.

This module deliberately returns plain dictionaries.  The chart may decide how to
draw an item, but classification and checklist logic always use the same prices,
date range, and source labels.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _detect_pivot_ranges(
    highs: list[float],
    lows: list[float],
    start_idx: int,
    end_idx: int,
    min_days: int,
    max_range_pct: float,
) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    idx = start_idx
    while idx <= end_idx - min_days:
        window_high = max(highs[idx:idx + min_days])
        window_low = min(lows[idx:idx + min_days])
        range_pct = (window_high / window_low - 1) * 100 if window_low else 999
        if range_pct <= max_range_pct:
            pivot_start = idx
            pivot_end = idx + min_days - 1
            while pivot_end + 1 < end_idx:
                next_high = max(highs[pivot_start:pivot_end + 2])
                next_low = min(lows[pivot_start:pivot_end + 2])
                next_range_pct = (next_high / next_low - 1) * 100 if next_low else 999
                if next_range_pct > max_range_pct:
                    break
                pivot_end += 1
                window_high, window_low, range_pct = next_high, next_low, next_range_pct
            ranges.append({
                "low": window_low,
                "mid": (window_high + window_low) / 2,
                "high": window_high,
                "first_touch": pivot_start,
                "last_touch": pivot_end,
                "days": pivot_end - pivot_start + 1,
                "range_pct": range_pct,
            })
            idx = pivot_end + 1
        else:
            idx += 1
    return ranges


def _pivot_range_schedule(max_range_pct: float) -> list[tuple[float, int]]:
    range_ceiling = max(5.0, float(max_range_pct))
    wider_ranges: list[float] = []
    value = 5.5
    while value <= range_ceiling + 0.001:
        wider_ranges.append(round(value, 1))
        value += 0.5
    schedule = [(5.0, 10)]
    schedule.extend((5.0, days) for days in range(9, 2, -1))
    for range_pct in wider_ranges:
        schedule.append((range_pct, 10))
        schedule.extend((range_pct, days) for days in range(9, 2, -1))
    return schedule


def _zones_overlap(first: dict[str, Any], second: dict[str, Any], tolerance_pct: float = 0.2) -> bool:
    first_low = float(first["low"])
    first_high = float(first["high"])
    second_low = float(second["low"])
    second_high = float(second["high"])
    tolerance = min(first_low, second_low) * tolerance_pct / 100
    return first_low <= second_high + tolerance and second_low <= first_high + tolerance


def detect_priority_pivot_ranges(
    highs,
    lows,
    closes,
    display_count: int = 5,
    max_range_pct: float = 10.0,
) -> list[dict[str, Any]]:
    """Return the same priority pivot ranges used by analysis and charts."""

    high_values = [float(value) for value in highs]
    low_values = [float(value) for value in lows]
    close_values = [float(value) for value in closes]
    if not close_values or len(high_values) != len(close_values) or len(low_values) != len(close_values):
        return []

    latest_close = close_values[-1]
    recent_start = max(0, len(close_values) - 30)
    selected: list[dict[str, Any]] = []
    for start_idx, end_idx in ((recent_start, len(close_values)), (0, recent_start)):
        if end_idx - start_idx < 3:
            continue
        for range_pct, min_days in _pivot_range_schedule(max_range_pct):
            if range_pct > max_range_pct and max_range_pct >= 5:
                continue
            candidates = _detect_pivot_ranges(
                high_values,
                low_values,
                start_idx,
                end_idx,
                min_days,
                range_pct,
            )
            for candidate in candidates:
                candidate["distance"] = (
                    abs(latest_close / candidate["mid"] - 1)
                    if latest_close and candidate["mid"]
                    else 999
                )
            candidates.sort(key=lambda item: (item["distance"], -item["days"], -item["last_touch"]))
            for candidate in candidates:
                if any(_zones_overlap(candidate, existing) for existing in selected):
                    continue
                selected.append(candidate)
                if len(selected) >= display_count:
                    return selected
    return selected


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _zone(source: str, family: str, low: float, high: float, **extra: Any) -> dict[str, Any]:
    return {
        "source": source,
        "family": family,
        "low": round(min(low, high), 6),
        "high": round(max(low, high), 6),
        **extra,
    }


def _mean_true_range(frame: pd.DataFrame) -> float:
    if len(frame) < 2:
        return 0.0
    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    ranges = max(
        len(highs) - 1,
        0,
    )
    if not ranges:
        return 0.0
    true_ranges = pd.DataFrame({
        "range": highs[1:] - lows[1:],
        "high_gap": abs(highs[1:] - closes[:-1]),
        "low_gap": abs(lows[1:] - closes[:-1]),
    }).max(axis=1)
    return float(true_ranges.tail(14).mean()) if not true_ranges.empty else 0.0


def _support_zone_candidate(frame: pd.DataFrame) -> dict[str, Any] | None:
    """Find the nearest multi-touch support area used by the chart overlay."""
    recent = frame.tail(90).reset_index(drop=True)
    if len(recent) < 20:
        return None
    lows = pd.to_numeric(recent["low"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(recent["high"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(recent["close"], errors="coerce").to_numpy(dtype=float)
    latest_close = closes[-1]
    if not latest_close or pd.isna(latest_close):
        return None

    atr = _mean_true_range(recent)
    tolerance = max(atr * 0.35, latest_close * 0.01, 0.01)
    prices: list[tuple[float, int]] = []
    for index in range(2, len(recent) - 2):
        low_window = lows[index - 2:index + 3]
        if lows[index] <= low_window.min():
            prices.append((float(lows[index]), index))
        # A former local high below current price can act as support after
        # being reclaimed, matching the chart's support-zone treatment.
        high_window = highs[index - 2:index + 3]
        if highs[index] >= high_window.max() and highs[index] < latest_close:
            prices.append((float(highs[index]), index))
    if len(prices) < 2:
        return None

    prices.sort(key=lambda item: item[0])
    clusters: list[list[tuple[float, int]]] = []
    for point in prices:
        if not clusters or point[0] - clusters[-1][-1][0] > tolerance:
            clusters.append([point])
        else:
            clusters[-1].append(point)
    clusters = [cluster for cluster in clusters if len(cluster) >= 2 and sum(price for price, _ in cluster) / len(cluster) <= latest_close]
    if not clusters:
        return None

    cluster = min(
        clusters,
        key=lambda item: (
            abs(latest_close - sum(price for price, _ in item) / len(item)),
            -max(index for _, index in item),
            -len(item),
        ),
    )
    midpoint = sum(price for price, _ in cluster) / len(cluster)
    half_width = max(tolerance, midpoint * 0.0075)
    return _zone(
        "Support Zone",
        "support_zone",
        midpoint - half_width,
        midpoint + half_width,
        touches=len(cluster),
        first_touch=min(index for _, index in cluster),
        last_touch=max(index for _, index in cluster),
    )


def _confluence_zones(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return genuine price intersections between independent evidence zones.

    A META zone should represent a shared price area, not just the nearest
    single indicator.  Keep the original candidates as evidence, while adding
    only intersections that have a real geometric overlap.
    """
    intersections: list[dict[str, Any]] = []
    for index, first in enumerate(candidates):
        # The compact-base envelope is intentionally broad and would make
        # almost every nearby indicator look like a confluence by overlap.
        if first.get("family") in {"pivot_zone", "compact_base"}:
            continue
        first_low = _number(first.get("low"))
        first_high = _number(first.get("high"))
        if first_low is None or first_high is None:
            continue
        for second in candidates[index + 1:]:
            if second.get("family") in {"pivot_zone", "compact_base"}:
                continue
            second_low = _number(second.get("low"))
            second_high = _number(second.get("high"))
            if second_low is None or second_high is None:
                continue
            low = max(first_low, second_low)
            high = min(first_high, second_high)
            if low > high:
                continue
            sources = [str(first.get("source")), str(second.get("source"))]
            families = [str(first.get("family")), str(second.get("family"))]
            intersections.append(_zone(
                " + ".join(sources),
                "confluence",
                low,
                high,
                is_confluence=True,
                sources=sources,
                families=families,
            ))
    return intersections


def build_chart_evidence(frame: pd.DataFrame) -> dict[str, Any]:
    """Build current technical evidence from the completed daily OHLCV frame.

    This is intentionally conservative: it records deterministic MA, swing,
    open-gap, and compact-base evidence without claiming that a visual overlay is
    necessarily an actionable support level.
    """
    if frame.empty:
        return {"levels": [], "zones": [], "meta_candidates": [], "base": {}}

    data = frame.copy()
    latest = data.iloc[-1]
    close = _number(latest.get("close"), 0.0) or 0.0
    atr = _number(latest.get("ATR"), 0.0) or (close * 0.01)
    half_atr = max(atr * 0.25, close * 0.0025)
    levels: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    sma10 = _number(latest.get("SMA_10"))
    sma20 = _number(latest.get("SMA_20"))
    sma50 = _number(latest.get("SMA_50"))
    for name, value in (("MA 10", sma10), ("MA 20", sma20), ("MA 50", sma50)):
        if value and value > 0:
            levels.append({"source": name, "value": value})
    if sma10 and sma20:
        candidates.append(_zone("MA 10 / MA 20", "short_ma_cluster", sma10, sma20))
    if sma50:
        candidates.append(_zone("MA 50", "sma50", sma50 - half_atr, sma50 + half_atr))

    recent_low = _number(data["low"].tail(20).min())
    if recent_low:
        candidates.append(_zone("20d swing low", "swing_low", recent_low - half_atr, recent_low + half_atr))

    support_zone = _support_zone_candidate(data)
    if support_zone:
        candidates.append(support_zone)

    # Use exactly the same pivot detector and 90-bar lookback as the chart.
    # The first range is the primary visual Pivot Zone and therefore also the
    # structural source used by a breakout stop.
    pivot_data = data.tail(90)
    pivot_ranges = detect_priority_pivot_ranges(
        pivot_data["high"],
        pivot_data["low"],
        pivot_data["close"],
        display_count=5,
        max_range_pct=10.0,
    )
    pivot_offset = len(data) - len(pivot_data)
    for index, pivot_range in enumerate(pivot_ranges):
        candidates.append(_zone(
            "Pivot Zone",
            "pivot_zone",
            pivot_range["low"],
            pivot_range["high"],
            days=int(pivot_range["days"]),
            range_pct=float(pivot_range["range_pct"]) / 100,
            first_touch=int(pivot_range["first_touch"]) + pivot_offset,
            last_touch=int(pivot_range["last_touch"]) + pivot_offset,
            is_primary_pivot=index == 0,
            stop_eligible=True,
        ))

    pivot = _number(latest.get("pivot_high_20"))
    # A former breakout level only becomes META evidence after price has reclaimed it.
    if pivot and close >= pivot:
        candidates.append(_zone("Prior 20d pivot held", "role_reversal", pivot - half_atr, pivot + half_atr))

    gap_zones: list[dict[str, Any]] = []
    if len(data) >= 2:
        for idx in range(1, len(data)):
            prev_close = _number(data.iloc[idx - 1].get("close"))
            day_low = _number(data.iloc[idx].get("low"))
            day_high = _number(data.iloc[idx].get("high"))
            if not prev_close or not day_low or not day_high:
                continue
            if day_low > prev_close:
                zone = _zone("Gap Fill", "gap_fill", prev_close, day_low, direction="up", index=idx)
            elif day_high < prev_close:
                zone = _zone("Gap Fill", "gap_fill", day_high, prev_close, direction="down", index=idx)
            else:
                continue
            # Retain only a currently relevant recent gap.  Filled gaps are not META evidence.
            if idx >= len(data) - 90 and close >= zone["low"] * 0.97:
                gap_zones.append(zone)
    if gap_zones:
        candidates.append(min(gap_zones, key=lambda item: abs(((item["low"] + item["high"]) / 2) - close)))

    base_window = data.tail(20)
    base_low = _number(base_window["low"].min(), close)
    base_high = _number(base_window["high"].max(), close)
    base_range_pct = ((base_high - base_low) / close) if close else None
    base = {
        "duration_bars": int(len(base_window)),
        "low": base_low,
        "high": base_high,
        "range_pct": base_range_pct,
        "range_contracted": bool(latest.get("range_contraction", False)),
    }
    if base_range_pct is not None and base_range_pct <= 0.15:
        candidates.append(_zone(
            "20d compact base",
            "compact_base",
            base_low,
            base_high,
            duration_bars=len(base_window),
            stop_eligible=False,
        ))

    # Preserve every raw edge for auditability, but let scoring and the UI use
    # a true overlap when two independent evidence zones meet.
    candidates.extend(_confluence_zones(candidates))

    return {
        "levels": levels,
        "zones": [*gap_zones],
        "meta_candidates": candidates,
        "base": base,
        "as_of_close": close,
    }
