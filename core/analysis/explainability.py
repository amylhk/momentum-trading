from __future__ import annotations

from typing import Any

import pandas as pd

from core.analysis.contracts import EARNINGS_EVENT_WINDOW_DAYS, RuleResult, ScreeningSettings

BREAKOUT_VOLUME_CONFIRM_RATIO = 1.10
BREAKOUT_CHASE_CEILING_PCT = 0.01
FAILURE_VOLUME_RATIO = 1.20
MAX_RISK_PCT = 0.08
EXTENDED_FROM_SMA10_PCT = 0.08
EXTENDED_FROM_SMA20_ATR = 3


def latest_value(stock: dict[str, Any], key: str, default=None):
    value = stock.get(key, default)
    if isinstance(value, list):
        return value[-1] if value else default
    return value


def previous_value(stock: dict[str, Any], key: str, default=None):
    value = stock.get(key, default)
    if isinstance(value, list):
        return value[-2] if len(value) >= 2 else default
    return default


def _number(value, digits=2):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _rule(group, rule, passed, actual, threshold, comparison, detail="", not_applicable=False, needs_review=False):
    status = "Not applicable" if not_applicable else "Needs review" if needs_review else "Pass" if passed else "Fail"
    return RuleResult(group, rule, status, actual, threshold, comparison, detail=detail).to_dict()


def _safe_div(numerator, denominator, default=None):
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return default
    return numerator / denominator


def _with_fallback_detail(saved: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(saved, dict):
        saved = {}
    return {**fallback, **saved}


def _fallback_failure_detail(stock: dict[str, Any]) -> dict[str, Any]:
    latest_close = latest_value(stock, "close")
    prev_close = previous_value(stock, "close")
    prev_low = previous_value(stock, "low")
    pivot_high_20 = previous_value(stock, "pivot_high_20") or latest_value(stock, "pivot_high_20")
    entry_price = pivot_high_20 * 1.005 if pivot_high_20 is not None else stock.get("entry_price")
    volume_ratio = previous_value(stock, "volume_ratio")
    if volume_ratio is None:
        volume_ratio = latest_value(stock, "volume_ratio")
    prev_above_entry = bool(prev_close is not None and entry_price is not None and prev_close >= entry_price)
    volume_confirm = bool(volume_ratio is not None and volume_ratio >= BREAKOUT_VOLUME_CONFIRM_RATIO)
    latest_below_pivot = bool(latest_close is not None and pivot_high_20 is not None and latest_close < pivot_high_20)
    return {
        "holds_breakout_pivot": {
            "prev_close": prev_close,
            "latest_close": latest_close,
            "pivot_high_20": pivot_high_20,
            "entry_price": entry_price,
            "volume_ratio": volume_ratio,
            "volume_threshold": BREAKOUT_VOLUME_CONFIRM_RATIO,
            "prev_above_entry": prev_above_entry,
            "latest_below_pivot": latest_below_pivot,
            "volume_confirm": volume_confirm,
            "confirmed_prior_breakout": prev_above_entry and volume_confirm,
        },
        "holds_short_term_support": {
            "latest_close": latest_close,
            "sma20": latest_value(stock, "SMA_20"),
            "sma50": latest_value(stock, "SMA_50"),
            "prev_low": prev_low,
            "volume_ratio": volume_ratio,
            "volume_threshold": FAILURE_VOLUME_RATIO,
            "below_sma20": bool(latest_close is not None and latest_value(stock, "SMA_20") is not None and latest_close < latest_value(stock, "SMA_20")),
            "below_sma50": bool(latest_close is not None and latest_value(stock, "SMA_50") is not None and latest_close < latest_value(stock, "SMA_50")),
            "below_prev_low_with_volume": bool(
                latest_close is not None
                and prev_low is not None
                and volume_ratio is not None
                and latest_close < prev_low
                and volume_ratio >= FAILURE_VOLUME_RATIO
            ),
        },
        "risk_too_wide": {
            "risk_pct": stock.get("risk_pct"),
            "risk_threshold": MAX_RISK_PCT,
        },
        "long_term_trend_broken": {
            "latest_close": latest_close,
            "sma200": latest_value(stock, "SMA_200"),
        },
    }


def _fallback_extension_detail(stock: dict[str, Any]) -> dict[str, Any]:
    latest_close = latest_value(stock, "close")
    entry_price = stock.get("entry_price")
    sma10 = latest_value(stock, "SMA_10")
    sma20 = latest_value(stock, "SMA_20")
    atr = latest_value(stock, "ATR")
    return {
        "within_chase_limit": {
            "latest_close": latest_close,
            "entry_price": entry_price,
            "entry_ceiling": entry_price * (1 + BREAKOUT_CHASE_CEILING_PCT) if entry_price else None,
            "ceiling_pct": BREAKOUT_CHASE_CEILING_PCT,
        },
        "extended_from_sma10": {
            "latest_close": latest_close,
            "sma10": sma10,
            "distance_pct": _safe_div((latest_close - sma10) if latest_close is not None and sma10 is not None else None, sma10),
            "threshold_pct": EXTENDED_FROM_SMA10_PCT,
        },
        "extended_by_atr": {
            "latest_close": latest_close,
            "sma20": sma20,
            "atr": atr,
            "distance_atr": _safe_div((latest_close - sma20) if latest_close is not None and sma20 is not None else None, atr),
            "threshold_atr": EXTENDED_FROM_SMA20_ATR,
        },
    }


def _coerce_market_date(value: Any) -> pd.Timestamp:
    """Parse Yahoo dates without treating Unix timestamps as 1970-era datetimes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT

    if isinstance(value, (int, float)):
        magnitude = abs(value)
        unit = "s" if magnitude < 100_000_000_000 else "ms" if magnitude < 100_000_000_000_000 else "ns"
        return pd.to_datetime(value, unit=unit, errors="coerce")

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return _coerce_market_date(int(stripped))
        return pd.to_datetime(stripped, errors="coerce")

    return pd.to_datetime(value, errors="coerce")


def earnings_context(stock: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic event-risk context relative to the daily data date."""
    raw_date = stock.get("earnings_date")
    earnings_date = _coerce_market_date(raw_date)
    data_date = _coerce_market_date(stock.get("data_date"))
    if pd.isna(earnings_date) or pd.isna(data_date):
        return {
            "days": None,
            "warning": "未公佈下一次業績日期，入場前需自行覆核。",
            "status": "Needs review",
        }

    days = int((earnings_date.normalize() - data_date.normalize()).days)
    if days < 0:
        return {
            "days": days,
            "warning": "未公佈下一次業績日期，入場前需自行覆核。",
            "status": "Needs review",
        }
    if days <= EARNINGS_EVENT_WINDOW_DAYS:
        return {
            "days": days,
            "warning": (
                f"距離業績公布尚有 {days} 日；如沒有利潤緩衝，不宜新開倉或持貨過業績。"
            ),
            "status": "Needs review",
        }
    return {
        "days": days,
        "warning": f"距離下一次業績公布尚有 {days} 日。",
        "status": "Pass",
    }


def build_rule_checklist(stock: dict[str, Any], settings: ScreeningSettings) -> list[dict[str, Any]]:
    """Turn already-computed deterministic inputs into auditable rows."""
    price = _number(stock.get("price"))
    ma50, ma200 = _number(stock.get("ma50")), _number(stock.get("ma200"))
    market_cap = _number(stock.get("market_cap"), 0)
    avgvol = _number(stock.get("avgvol3m"), 0)
    pct_high = _number(stock.get("pct_from_high"), 4)
    distance_from_high = -pct_high if pct_high is not None else None
    high_52w = _number(price / (1 + pct_high), 2) if price is not None and pct_high is not None and pct_high > -1 else None
    phase = stock.get("setup_phase", "unclear_structure")
    evidence = stock.get("setup_evidence") or {}
    failure = evidence.get("failure_flags") or {}
    validity = evidence.get("validity_flags") or {}
    structural_validity = {
        "holds_breakout_pivot": bool(
            validity.get(
                "holds_breakout_pivot",
                not bool(failure.get("failed_breakout", failure.get("close_back_below_pivot"))),
            )
        ),
        "holds_short_term_support": bool(
            validity.get("holds_short_term_support", not bool(failure.get("breaks_support")))
        ),
    }
    extension = evidence.get("extension_flags") or {}
    within_chase_limit = bool(
        extension.get(
            "within_chase_limit",
            not bool(extension.get("above_entry_threshold")),
        )
    )
    failure_detail = _with_fallback_detail(
        evidence.get("failure_flag_details"),
        _fallback_failure_detail(stock),
    )
    saved_validity_detail = evidence.get("validity_flag_details") or {}
    structural_validity_detail = {
        "holds_breakout_pivot": saved_validity_detail.get(
            "holds_breakout_pivot",
            failure_detail.get("failed_breakout", failure_detail.get("holds_breakout_pivot", {})),
        ),
        "holds_short_term_support": saved_validity_detail.get(
            "holds_short_term_support",
            failure_detail.get("breaks_support", failure_detail.get("holds_short_term_support", {})),
        ),
    }
    breakout_context = bool(evidence.get("breakout_context"))
    for detail in structural_validity_detail.values():
        if isinstance(detail, dict):
            detail.setdefault("breakout_context", breakout_context)
    extension_detail = _with_fallback_detail(
        evidence.get("extension_flag_details"),
        _fallback_extension_detail(stock),
    )
    values = {
        key: latest_value(stock, key)
        for key in ("SMA_10", "SMA_20", "SMA_50", "SMA_200", "volume_ratio", "dry_up_ratio", "dist_pivot_20", "RSI", "ATR_pct")
    }
    # The classifier evaluates breakout proximity against the buffered entry
    # price, not the raw pivot. Keep the raw pivot only as a legacy fallback for
    # snapshots saved before dist_to_entry was persisted.
    dist_to_entry = _number(evidence.get("dist_to_entry"), 4)
    if dist_to_entry is None:
        dist_to_entry = _number(values["dist_pivot_20"], 4)
    rules = [
        _rule("Universe", "市值達標", market_cap is not None and market_cap >= settings.market_cap_min, market_cap, settings.market_cap_min, ">="),
        _rule("Universe", "股價達標", price is not None and price >= settings.price_min, price, settings.price_min, ">="),
        _rule("Universe", "三個月平均成交量達標", avgvol is not None and avgvol >= settings.avg_volume_3m_min, avgvol, settings.avg_volume_3m_min, ">="),
        _rule("Universe", "成交額達標", price is not None and avgvol is not None and price * avgvol >= settings.dollar_volume_min, _number((price or 0) * (avgvol or 0), 0), settings.dollar_volume_min, ">="),
        _rule(
            "Universe",
            f"現價距 52 週高位不超過 {settings.max_distance_52w_high:.0%}",
            distance_from_high is not None and distance_from_high <= settings.max_distance_52w_high,
            {"現時股價": price, "52 週高位": high_52w, "距高位": distance_from_high},
            settings.max_distance_52w_high,
            "<=",
        ),
        _rule("Stage 2", "股價 >= MA50 >= MA200", price is not None and ma50 is not None and ma200 is not None and price >= ma50 >= ma200, {"股價": price, "MA50": ma50, "MA200": ma200}, "股價 >= MA50 >= MA200", ">=", not_applicable=not settings.require_price_above_ma_stack),
        _rule("Stage 2", "MA10 > MA20", values["SMA_10"] is not None and values["SMA_20"] is not None and values["SMA_10"] > values["SMA_20"], {"MA10": _number(values["SMA_10"]), "MA20": _number(values["SMA_20"])}, "MA10 > MA20", ">", not_applicable=not settings.require_sma10_above_sma20),
        _rule("Stage 2", "200 日均線向上", bool(stock.get("ma200_is_trending_up")), stock.get("ma200_is_trending_up"), True, "=", not_applicable=not settings.require_ma200_rising),
    ]
    for period, required in (
        ("1m", settings.require_vs_market_1m),
        ("3m", settings.require_vs_market_3m),
        ("6m", settings.require_vs_market_6m),
    ):
        value = _number(stock.get(f"vs_market_{period}"), 4)
        period_label = {"1m": "1 個月", "3m": "3 個月", "6m": "6 個月"}[period]
        rules.append(_rule("Stage 2", f"跑贏大市（{period_label}）", value is not None and value > 0, value, 0, ">", not_applicable=not required))

    risk = _number(stock.get("risk_pct"), 4)
    earnings = earnings_context(stock)
    rules.extend([
        _rule("Setup", "距離突破入場參考", dist_to_entry is not None and -0.03 <= dist_to_entry <= 0.01, dist_to_entry, {"下限距離": -0.03, "上限距離": 0.01}, "範圍"),
        _rule("Setup", "突破成交量確認（最新交易日 / 20 日均量）", values["volume_ratio"] is not None and values["volume_ratio"] >= 1.10, _number(values["volume_ratio"]), 1.10, ">="),
        _rule("Setup", "META 量縮滿分（3 日均量 / 20 日均量）", values["dry_up_ratio"] is not None and values["dry_up_ratio"] <= 0.90, _number(values["dry_up_ratio"]), 0.90, "<="),
        _rule(
            "Setup",
            "突破結構仍然有效",
            all(structural_validity.values()),
            structural_validity,
            True,
            "兩項均成立",
            detail=structural_validity_detail,
            needs_review=phase == "unclear_structure",
        ),
        _rule("Risk", "技術止損風險不超過 8%", risk is not None and risk <= 0.08, risk, 0.08, "<="),
        _rule(
            "Risk",
            "仍在合理追價範圍",
            within_chase_limit,
            {"within_chase_limit": within_chase_limit},
            True,
            "不超過追價上限",
            detail={
                "within_chase_limit": extension_detail.get(
                    "within_chase_limit",
                    extension_detail.get("above_entry_threshold", {}),
                )
            },
            needs_review=phase == "extended_breakout",
        ),
        _rule(
            "Earnings Risk",
            f"下一次業績公布距今超過 {EARNINGS_EVENT_WINDOW_DAYS} 日",
            earnings["status"] == "Pass",
            earnings["days"],
            EARNINGS_EVENT_WINDOW_DAYS,
            ">",
            detail=earnings["warning"],
            needs_review=earnings["status"] == "Needs review",
        ),
    ])
    return rules


def phase_score(stock: dict[str, Any]) -> tuple[int, str]:
    phase = stock.get("setup_phase", "unclear_structure")
    breakout = int(stock.get("breakout_score") or 0)
    meta = int(stock.get("meta_score") or 0)
    if phase in {"near_breakout", "fresh_breakout", "extended_breakout", "failed_breakout"}:
        return breakout, "突破評分"
    if phase in {"pullback_forming", "pullback_entry"}:
        return meta, "META 評分"
    if breakout >= meta:
        return breakout, "觀察評分（突破依據）"
    return meta, "觀察評分（META 依據）"


_SCORE_COMPONENT_LABELS = {
    "trend_structure": "趨勢結構",
    "pivot_proximity": "樞紐距離",
    "base_quality": "底部品質",
    "volume_context": "成交量背景",
    "market_leadership": "大市表現",
    "range_contraction": "區間收窄",
    "risk_quality": "風險品質",
    "meta_confluence": "META 匯聚度",
    "distance_to_meta": "距離 META",
    "volume_dry_up": "成交量收縮",
    "reversal_signal": "反轉確認",
}

_BREAKOUT_SCORE_MAXIMUMS = {
    "trend_structure": 25, "pivot_proximity": 20, "base_quality": 10,
    "volume_context": 10, "market_leadership": 10, "range_contraction": 5,
    "risk_quality": 10,
}

_META_SCORE_MAXIMUMS = {
    "meta_confluence": 30, "distance_to_meta": 15, "volume_dry_up": 10,
    "range_contraction": 5, "reversal_signal": 10, "risk_quality": 10,
}


def _score_source(stock: dict[str, Any], source: str | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, int], str]:
    """Return the evidence dictionary that supplies the displayed phase score."""
    evidence = stock.get("setup_evidence") or {}
    phase = stock.get("setup_phase", "unclear_structure")
    if source == "breakout":
        return evidence.get("breakout_score_parts") or {}, evidence.get("breakout_score_detail") or {}, _BREAKOUT_SCORE_MAXIMUMS, "breakout"
    if source in {"meta", "pullback"}:
        return evidence.get("meta_score_parts") or {}, evidence.get("meta_score_detail") or {}, _META_SCORE_MAXIMUMS, "pullback"
    if phase in {"near_breakout", "fresh_breakout", "extended_breakout", "failed_breakout"}:
        return evidence.get("breakout_score_parts") or {}, evidence.get("breakout_score_detail") or {}, _BREAKOUT_SCORE_MAXIMUMS, "breakout"
    if phase in {"pullback_forming", "pullback_entry"}:
        return evidence.get("meta_score_parts") or {}, evidence.get("meta_score_detail") or {}, _META_SCORE_MAXIMUMS, "pullback"
    if int(stock.get("breakout_score") or 0) >= int(stock.get("meta_score") or 0):
        return evidence.get("breakout_score_parts") or {}, evidence.get("breakout_score_detail") or {}, _BREAKOUT_SCORE_MAXIMUMS, "breakout"
    return evidence.get("meta_score_parts") or {}, evidence.get("meta_score_detail") or {}, _META_SCORE_MAXIMUMS, "pullback"


def score_breakdown(stock: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    """Expose the phase-specific weighted score instead of conflating it with rule count."""
    if source == "breakout":
        score, label = int(stock.get("breakout_score") or 0), "突破評分"
    elif source in {"meta", "pullback"}:
        score, label = int(stock.get("meta_score") or 0), "META 評分"
    else:
        score, label = phase_score(stock)
    parts, details, maximums, resolved_source = _score_source(stock, source)
    components = []
    for key, maximum in maximums.items():
        points = min(max(int(parts.get(key) or 0), 0), maximum)
        components.append({
            "key": key,
            "component": _SCORE_COMPONENT_LABELS.get(key, key.replace("_", " ").title()),
            "points": points,
            "maximum": maximum,
            "group": "Risk Score" if key == "risk_quality" else "Setup Score",
            "items": details.get(key) or [],
        })
    source_label = "突破依據" if resolved_source == "breakout" else "回調依據"
    display_label = label if not label.startswith("Watch Score") else label
    basis = "; ".join(f"{item['component']} {item['points']}/{item['maximum']}" for item in components)
    return {
        "score": score,
        "possible": sum(maximums.values()),
        "score_display": f"{score}/{sum(maximums.values())}",
        "label": label,
        "display_label": display_label,
        "source": resolved_source,
        "components": components,
        "basis": f"{display_label}: {score}/{sum(maximums.values())} ({basis})",
    }


def build_score_checklist(stock: dict[str, Any], source: str | None = None) -> list[dict[str, Any]]:
    """Expose each deterministic score component as earned/possible points."""
    breakdown = score_breakdown(stock, source=source)
    rows = []

    def latest_stock_value(key):
        value = stock.get(key)
        if isinstance(value, list):
            return value[-1] if value else None
        return value

    for component in breakdown["components"]:
        points, maximum = component["points"], component["maximum"]
        items = component.get("items") or []
        if items:
            for item in items:
                item_points = int(item.get("points") or 0)
                item_maximum = int(item.get("maximum") or maximum)
                item_status = item.get("status")
                actual = item.get("actual")
                if component["key"] == "meta_confluence" and isinstance(actual, dict):
                    actual = dict(actual)
                    chart_candidates = (stock.get("setup_evidence") or {}).get("chart_evidence", {}).get("meta_candidates", [])
                    all_families = {
                        family
                        for candidate in chart_candidates
                        for family in (
                            candidate.get("families", [])
                            if candidate.get("is_confluence")
                            else [candidate.get("family")]
                        )
                        if family and family != "confluence"
                    }
                    if all_families:
                        actual.setdefault("所有現存條件", sorted(all_families))
                        actual.setdefault("符合條件", actual.get("價格條件類別", []))
                if component["key"] == "distance_to_meta" and isinstance(actual, dict):
                    actual = dict(actual)
                    current_price = latest_stock_value("close")
                    if current_price is not None:
                        actual.setdefault("現時股價", current_price)
                        low = actual.get("low")
                        high = actual.get("high")
                        try:
                            current_number = float(current_price)
                            low_number = float(low)
                            high_number = float(high)
                        except (TypeError, ValueError):
                            current_number = low_number = high_number = None
                        if current_number is not None and low_number is not None and high_number:
                            actual.setdefault("是否在區內", low_number <= current_number <= high_number)
                            actual.setdefault("距離區頂", (current_number - high_number) / high_number)
                rows.append(_rule(
                    component["group"], item.get("label", component["component"]),
                    item_points == item_maximum,
                    f"{item_points}/{item_maximum}",
                    actual, "計分條件",
                    detail=item.get("detail", ""),
                    needs_review=item_status == "Needs review" or 0 < item_points < item_maximum,
                ))
        else:
            rows.append(_rule(
                component["group"], component["component"], points == maximum,
                f"{points}/{maximum}", "", "計分條件",
                detail=f"此為舊快取資料，未保存細項；目前依據該評分組合給予 {points}/{maximum} 分。重新分析此股票後可查看完整細項。",
                needs_review=0 < points < maximum,
            ))
    return rows


def build_decision_tree(stock: dict[str, Any], settings: ScreeningSettings) -> list[dict[str, Any]]:
    """Expose the hard classification gates separately from quality scoring."""
    evidence = stock.get("setup_evidence") or {}
    phase = stock.get("setup_phase", "unclear_structure")
    rules = build_rule_checklist(stock, settings)
    prerequisites = [rule for rule in rules if rule["group"] in {"Universe", "Stage 2"} and rule["status"] != "Not applicable"]
    prerequisites_passed = bool(prerequisites) and all(rule["status"] == "Pass" for rule in prerequisites)
    failures = evidence.get("failure_flags") or {}
    validity = evidence.get("validity_flags") or {
        "holds_breakout_pivot": not bool(failures.get("failed_breakout")),
        "holds_short_term_support": not bool(failures.get("breaks_support")),
    }
    extensions = evidence.get("extension_flags") or {}
    fresh_breakout = phase == "fresh_breakout"
    chart_evidence = evidence.get("chart_evidence") or {}
    has_meta_zone = bool(chart_evidence.get("selected_meta_zone"))
    has_saved_chart_evidence = "chart_evidence" in evidence
    # Older snapshots may not have chart evidence at all; keep those as a
    # review item.  Once chart evidence exists, a selected META overlap is a
    # hard requirement for either pullback phase.
    meta_entry = phase == "pullback_entry" and (has_meta_zone or not has_saved_chart_evidence)
    meta_forming = phase == "pullback_forming" and (has_meta_zone or not has_saved_chart_evidence)
    final_detail = {
        "near_breakout": "尚未符合確認突破條件，繼續等待價格及成交量確認。",
        "pullback_forming": "已形成 META 回調候選，但反轉訊號尚未確認。",
        "unclear_structure": "突破及 META 回調條件均未完整成立。",
    }.get(phase, "分數用於排序品質，不能覆蓋硬性的分類條件。")
    return [
        _rule("分類流程", "日線資料完整", bool(stock.get("data_date")), stock.get("data_date"), "已完成的日線交易時段", "可用", detail="分類只使用最後一個完成的日線交易時段，不使用即時報價。"),
        _rule("分類流程", "通過基本資格與第二階段條件", prerequisites_passed, f"{sum(rule['status'] == 'Pass' for rule in prerequisites)}/{len(prerequisites)}", "所有適用規則通過", "全部通過", detail="評分只在必要條件通過後，用於比較結構品質。"),
        _rule(
            "分類流程",
            "突破結構仍然有效",
            all(bool(value) for value in validity.values()),
            validity,
            True,
            "兩項均成立",
            detail="已確認突破後，必須同時守在原突破樞紐及短中期支持之上。",
        ),
        _rule("分類流程", "過度延伸檢查", phase != "extended_breakout", extensions, False, "必須為否", detail="即使趨勢評分高，已延伸股票亦不追高。", needs_review=phase == "extended_breakout"),
        _rule("分類流程", "確認突破條件", fresh_breakout, phase, "收市突破樞紐並有成交量確認", "分類", detail="只有最近完成交易日的突破才分類為突破買入。", not_applicable=phase in {"pullback_forming", "pullback_entry"}),
        _rule("分類流程", "META 入場條件", meta_entry, phase, "META 匯聚加上反轉確認", "分類", detail="確認回調需要先選定 META 區域，再通過匯聚及反轉訊號。", needs_review=meta_forming, not_applicable=fresh_breakout),
        _rule("分類流程", "最終分類", phase not in {"unclear_structure", "not_recommended"}, phase, "突破或 META 入場條件", "分類", detail=final_detail, needs_review=phase in {"near_breakout", "pullback_forming", "unclear_structure"}),
    ]


def deterministic_summary(stock: dict[str, Any]) -> str:
    phase = stock.get("setup_phase", "unclear_structure")
    distance = latest_value(stock, "dist_pivot_20")
    volume = latest_value(stock, "volume_ratio")
    risk = stock.get("risk_pct")
    confirmation = (
        latest_value(stock, "confirmation_price")
        or latest_value(stock, "entry_zone_high")
        or latest_value(stock, "entry_price")
    )
    chase_ceiling = confirmation * (1 + BREAKOUT_CHASE_CEILING_PCT) if confirmation else None
    if phase == "fresh_breakout":
        if confirmation and chase_ceiling:
            summary = (
                f"已收市突破 ${confirmation:,.2f}，成交量為20日均量 {_number(volume)}x"
                f"（門檻 {BREAKOUT_VOLUME_CONFIRM_RATIO:.2f}x）；現價未高於追價上限 "
                f"${chase_ceiling:,.2f}，可留意突破入場。"
            )
        else:
            summary = (
                "已帶量突破入場參考，目前未觸發過度延伸訊號；"
                f"成交量為20日均量 {_number(volume)}x，風險 {_number((risk or 0) * 100, 1)}%。"
            )
    elif phase == "near_breakout":
        if confirmation and chase_ceiling:
            summary = (
                f"等待收市突破 ${confirmation:,.2f}，並以成交量達20日均量 "
                f"{BREAKOUT_VOLUME_CONFIRM_RATIO:.2f}x 確認；確認後而未高於追價上限 "
                f"${chase_ceiling:,.2f}，可留意突破入場。目前成交量為 "
                f"{_number(volume)}x。"
            )
        else:
            summary = f"距前 20 日突破樞紐 {_number((distance or 0) * 100, 1)}%，等待收市突破及成交量確認。"
    elif phase in {"pullback_forming", "pullback_entry"}:
        summary = f"回調結構評分 {int(stock.get('meta_score') or 0)}；3日成交量為20日均量 {_number(latest_value(stock, 'dry_up_ratio'))}x。"
    elif phase == "extended_breakout":
        summary = f"趨勢仍強但已延伸；風險參考 {_number((risk or 0) * 100, 1)}%，等待新的底部結構或回調。"
    elif phase == "failed_breakout":
        summary = "突破後出現結構失敗訊號，暫不作新買入。"
    elif phase == "not_recommended":
        summary = "未通過基本資格或第二階段必要條件，暫不作新買入。"
    else:
        summary = "技術結構未清晰；保留觀察，等待突破或回調條件成形。"
    earnings = earnings_context(stock)
    if earnings["status"] == "Needs review":
        return f"{summary} {earnings['warning']}"
    return summary


def rules_passed_text(rules: list[dict[str, Any]]) -> str:
    applicable = [
        rule for rule in rules
        if rule["status"] != "Not applicable" and rule.get("group") not in {"Setup Score", "Risk Score", "Decision Tree"}
    ]
    return f"{sum(rule['status'] == 'Pass' for rule in applicable)}/{len(applicable)}"
