from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SignalDefinition:
    label: str
    detail: str


SIGNAL_DEFINITIONS = {
    "holds_breakout_pivot": SignalDefinition(
        "守在原突破樞紐上",
        "確認突破後，最新收市仍守在該次突破所依據的 20 日樞紐之上。",
    ),
    "close_back_below_pivot": SignalDefinition(
        "跌回突破樞紐下方",
        "舊快取訊號：突破後收市跌回樞紐下方。",
    ),
    "holds_short_term_support": SignalDefinition(
        "守在短中期支持上",
        "確認突破後，最新收市仍守在 SMA20 及 SMA50 之上。",
    ),
    "risk_too_wide": SignalDefinition(
        "止損風險過闊",
        "入場參考價至止損參考價的距離超過 8%。",
    ),
    "long_term_trend_broken": SignalDefinition(
        "跌穿長期趨勢",
        "最新收市價低於 SMA200。",
    ),
    "within_chase_limit": SignalDefinition(
        "仍在追價上限內",
        "最新收市價沒有超過入場參考價加上追價上限。",
    ),
    "extended_from_sma10": SignalDefinition(
        "偏離 SMA10 過遠",
        "最新收市價相對 SMA10 的距離達到過度延伸門檻。",
    ),
    "extended_by_atr": SignalDefinition(
        "偏離 SMA20 超過 ATR 門檻",
        "最新收市價相對 SMA20 的距離超過指定 ATR 倍數。",
    ),
    "pivot": SignalDefinition(
        "突破已延伸",
        "舊快取訊號：突破後已超過合理追價範圍。",
    ),
}


SIGNAL_RULES = {
    "沒有已確認結構失敗",
    "突破結構仍然有效",
    "未過度延伸",
    "仍在合理追價範圍",
}


def _signal_definition(key: str) -> SignalDefinition | None:
    return SIGNAL_DEFINITIONS.get(str(key))


def _price(value: Any) -> str:
    number = coerce_number(value)
    if number is None:
        return "-"
    return f"${number:.2f}"


def _ratio(value: Any) -> str:
    number = coerce_number(value)
    if number is None:
        return "-"
    return f"{number:.2f}x"


def _percent(value: Any) -> str:
    number = coerce_number(value)
    if number is None:
        return "-"
    return f"{number * 100:.1f}%"


def _bool_result(value: Any) -> str:
    return "是" if bool(value) else "否"


def _detail_value(detail: Any, key: str) -> Any:
    return detail.get(key) if isinstance(detail, dict) else None


def _holds_breakout_pivot_detail(detail: dict[str, Any]) -> str:
    text = (
        f"最新收市 {_price(detail.get('latest_close'))}；"
        f"原 20 日突破樞紐 {_price(detail.get('pivot_high_20'))}。"
    )
    if detail.get("pivot_source_date"):
        text += f" 樞紐最高價出現於 {detail['pivot_source_date']}。"
    return text


def _holds_short_term_support_detail(detail: dict[str, Any]) -> str:
    return (
        f"最新收市 {_price(detail.get('latest_close'))} | "
        f"SMA20 {_price(detail.get('sma20'))} | SMA50 {_price(detail.get('sma50'))}"
    )


def _risk_too_wide_detail(detail: dict[str, Any]) -> str:
    return (
        f"實際風險：{_percent(detail.get('risk_pct'))}\n"
        f"門檻：不可超過 {_percent(detail.get('risk_threshold'))}\n"
        "判斷：入場價至止損價距離超過門檻，才算止損風險過闊。"
    )


def _long_term_trend_broken_detail(detail: dict[str, Any]) -> str:
    return (
        f"最新收市：{_price(detail.get('latest_close'))}\n"
        f"SMA200：{_price(detail.get('sma200'))}\n"
        "判斷：最新收市低於 SMA200，才算跌穿長期趨勢。"
    )


def _within_chase_limit_detail(detail: dict[str, Any]) -> str:
    return (
        f"最新收市：{_price(detail.get('latest_close'))}\n"
        f"入場價：{_price(detail.get('entry_price'))}\n"
        f"追價上限：{_price(detail.get('entry_ceiling'))}\n"
        f"上限百分比：{_percent(detail.get('ceiling_pct'))}\n"
        "判斷：最新收市高於追價上限，才算高於入場價太多。"
    )


def _extended_from_sma10_detail(detail: dict[str, Any]) -> str:
    return (
        f"最新收市：{_price(detail.get('latest_close'))}\n"
        f"SMA10：{_price(detail.get('sma10'))}\n"
        f"實際距離：{_percent(detail.get('distance_pct'))}\n"
        f"門檻：{_percent(detail.get('threshold_pct'))}\n"
        "判斷：收市價相對 SMA10 的距離達到門檻，才算偏離 SMA10 過遠。"
    )


def _extended_by_atr_detail(detail: dict[str, Any]) -> str:
    return (
        f"最新收市：{_price(detail.get('latest_close'))}\n"
        f"SMA20：{_price(detail.get('sma20'))}\n"
        f"ATR：{_price(detail.get('atr'))}\n"
        f"實際距離：{_ratio(detail.get('distance_atr'))}\n"
        f"門檻：{_ratio(detail.get('threshold_atr'))}\n"
        "判斷：收市價高於 SMA20 的距離達到指定 ATR 倍數，才算偏離 SMA20 超過 ATR 門檻。"
    )


DETAIL_BUILDERS = {
    "holds_breakout_pivot": _holds_breakout_pivot_detail,
    "holds_short_term_support": _holds_short_term_support_detail,
    "risk_too_wide": _risk_too_wide_detail,
    "long_term_trend_broken": _long_term_trend_broken_detail,
    "within_chase_limit": _within_chase_limit_detail,
    "extended_from_sma10": _extended_from_sma10_detail,
    "extended_by_atr": _extended_by_atr_detail,
}


def _logic_status(*values: Any, mode: str = "and") -> str:
    passed = any(bool(value) for value in values) if mode == "or" else all(bool(value) for value in values)
    return _bool_result(passed)


def _holds_breakout_pivot_rows(detail: dict[str, Any]) -> list[dict[str, str]]:
    breakout_context = detail.get("breakout_context")
    if breakout_context is None:
        breakout_context = detail.get("confirmed_prior_breakout")
    holds_pivot = not bool(detail.get("latest_below_pivot"))
    source = ""
    if detail.get("pivot_source_date"):
        source = f" | 最高價日期 {detail['pivot_source_date']}"
    if detail.get("pivot_window_start") and detail.get("pivot_window_end"):
        source += (
            f" | 計算窗口 {detail['pivot_window_start']} 至 "
            f"{detail['pivot_window_end']}"
        )
    return [
        {
            "rule": "判斷",
            "status": "✅" if breakout_context and holds_pivot else "❌" if breakout_context else "—",
            "actual": (
                f"最新收市 {_price(detail.get('latest_close'))} | "
                f"原 20 日樞紐 {_price(detail.get('pivot_high_20'))}{source}"
            ),
            "threshold": f"原 20 日樞紐 {_price(detail.get('pivot_high_20'))}",
            "comparison": ">=" if breakout_context else "確認突破後才檢查",
            "detail": "",
        },
    ]


def _holds_short_term_support_rows(detail: dict[str, Any]) -> list[dict[str, str]]:
    breakout_context = detail.get("breakout_context")
    holds_sma20 = not bool(detail.get("below_sma20"))
    holds_sma50 = not bool(detail.get("below_sma50"))
    parent_status = "✅" if holds_sma20 and holds_sma50 else "❌"
    if breakout_context is False:
        parent_status = "—"
    return [
        {
            "rule": "判斷",
            "status": parent_status,
            "actual": (
                f"最新收市 {_price(detail.get('latest_close'))} | "
                f"SMA20 {_price(detail.get('sma20'))} | SMA50 {_price(detail.get('sma50'))}"
            ),
            "threshold": "最新收市 >= SMA20 及 SMA50",
            "comparison": "AND",
            "detail": "",
        },
    ]


def _risk_too_wide_rows(detail: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "rule": "實際風險",
            "status": _bool_result(
                coerce_number(detail.get("risk_pct")) is not None
                and coerce_number(detail.get("risk_threshold")) is not None
                and coerce_number(detail.get("risk_pct")) > coerce_number(detail.get("risk_threshold"))
            ),
            "actual": _percent(detail.get("risk_pct")),
            "threshold": f"不可超過 {_percent(detail.get('risk_threshold'))}",
            "comparison": ">",
            "detail": "",
        },
    ]


def _long_term_trend_broken_rows(detail: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "rule": "長期趨勢",
            "status": _bool_result(
                coerce_number(detail.get("latest_close")) is not None
                and coerce_number(detail.get("sma200")) is not None
                and coerce_number(detail.get("latest_close")) < coerce_number(detail.get("sma200"))
            ),
            "actual": f"最新收市 {_price(detail.get('latest_close'))}",
            "threshold": f"SMA200 {_price(detail.get('sma200'))}",
            "comparison": "<",
            "detail": "",
        },
    ]


def _within_chase_limit_rows(detail: dict[str, Any]) -> list[dict[str, str]]:
    latest_close = coerce_number(detail.get("latest_close"))
    entry_ceiling = coerce_number(detail.get("entry_ceiling"))
    within_limit = (
        latest_close is not None
        and entry_ceiling is not None
        and latest_close <= entry_ceiling
    )
    return [
        {
            "rule": "判斷",
            "status": "✅" if within_limit else "❌",
            "actual": (
                f"最新收市 {_price(detail.get('latest_close'))} | "
                f"入場參考 {_price(detail.get('entry_price'))} | "
                f"追價上限 {_price(detail.get('entry_ceiling'))}"
            ),
            "threshold": (
                f"入場參考 {_price(detail.get('entry_price'))} + "
                f"{_percent(detail.get('ceiling_pct'))} = {_price(detail.get('entry_ceiling'))}"
            ),
            "comparison": "<=",
            "detail": "",
        },
    ]


def _extended_from_sma10_rows(detail: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "rule": "SMA10 距離",
            "status": _bool_result(
                coerce_number(detail.get("distance_pct")) is not None
                and coerce_number(detail.get("threshold_pct")) is not None
                and coerce_number(detail.get("distance_pct")) >= coerce_number(detail.get("threshold_pct"))
            ),
            "actual": _percent(detail.get("distance_pct")),
            "threshold": _percent(detail.get("threshold_pct")),
            "comparison": ">=",
            "detail": f"最新收市 {_price(detail.get('latest_close'))}；SMA10 {_price(detail.get('sma10'))}",
        },
    ]


def _extended_by_atr_rows(detail: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "rule": "SMA20 / ATR 距離",
            "status": _bool_result(
                coerce_number(detail.get("distance_atr")) is not None
                and coerce_number(detail.get("threshold_atr")) is not None
                and coerce_number(detail.get("distance_atr")) >= coerce_number(detail.get("threshold_atr"))
            ),
            "actual": _ratio(detail.get("distance_atr")),
            "threshold": _ratio(detail.get("threshold_atr")),
            "comparison": ">=",
            "detail": f"最新收市 {_price(detail.get('latest_close'))}；SMA20 {_price(detail.get('sma20'))}；ATR {_price(detail.get('atr'))}",
        },
    ]


ROW_BUILDERS = {
    "holds_breakout_pivot": _holds_breakout_pivot_rows,
    "holds_short_term_support": _holds_short_term_support_rows,
    "risk_too_wide": _risk_too_wide_rows,
    "long_term_trend_broken": _long_term_trend_broken_rows,
    "within_chase_limit": _within_chase_limit_rows,
    "extended_from_sma10": _extended_from_sma10_rows,
    "extended_by_atr": _extended_by_atr_rows,
}


def _signal_detail(key: str, row_detail: Any, definition: SignalDefinition) -> str:
    signal_detail = _detail_value(row_detail, key)
    builder = DETAIL_BUILDERS.get(key)
    if builder and isinstance(signal_detail, dict):
        return builder(signal_detail)
    return definition.detail


def _signal_condition_rows(key: str, row_detail: Any) -> list[dict[str, str]]:
    signal_detail = _detail_value(row_detail, key)
    builder = ROW_BUILDERS.get(key)
    if builder and isinstance(signal_detail, dict):
        return builder(signal_detail)
    return []


def _split_detail_line(line: str) -> tuple[str, str]:
    text = str(line or "").strip()
    if "：" not in text:
        return "", text
    label, detail = text.split("：", 1)
    return label.strip(), detail.strip()


def _condition_rows(base_row: dict[str, Any], detail: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in str(detail or "").splitlines() if line.strip()]
    if len(lines) <= 1:
        return []

    rows = []
    for line in lines:
        label, content = _split_detail_line(line)
        child = base_row.copy()
        child["rule"] = label or "條件"
        child["status"] = ""
        child["actual"] = ""
        child["threshold"] = ""
        child["comparison"] = ""
        child["detail"] = content
        rows.append(child)
    return rows


def _structured_condition_rows(base_row: dict[str, Any], items: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        child = base_row.copy()
        child.update(item)
        rows.append(child)
    return rows


def _apply_parent_judgement(parent_row: dict[str, Any], items: list[dict[str, str]]) -> list[dict[str, str]]:
    if not items or items[0].get("rule") != "判斷":
        return items

    judgement = items[0]
    parent_row["status"] = judgement.get("status", parent_row.get("status"))
    parent_row["actual"] = judgement.get("actual", parent_row.get("actual"))
    parent_row["threshold"] = judgement.get("threshold", parent_row.get("threshold"))
    parent_row["comparison"] = judgement.get("comparison", parent_row.get("comparison"))
    parent_row["detail"] = judgement.get("detail", "")
    return items[1:]


def _status_for_signal(parent_status: str, value: Any) -> str:
    if parent_status == "Not applicable":
        return parent_status
    if bool(value):
        return "Needs review" if parent_status == "Needs review" else "Fail"
    return "Pass"


def expand_signal_dict_rows(rule_frame: pd.DataFrame) -> pd.DataFrame:
    """Expand internal signal dictionaries into one readable row per flag."""
    if rule_frame.empty or "actual" not in rule_frame:
        return rule_frame

    rows: list[dict[str, Any]] = []
    changed = False
    for _, row in rule_frame.iterrows():
        actual = row.get("actual")
        rule_name = row.get("rule")
        if not isinstance(actual, dict) or rule_name not in SIGNAL_RULES:
            rows.append(row.to_dict())
            continue

        known_items = [
            (key, value, _signal_definition(str(key)))
            for key, value in actual.items()
            if _signal_definition(str(key)) is not None
        ]
        if not known_items:
            rows.append(row.to_dict())
            continue

        changed = True
        row_detail = row.get("detail")
        for key, value, definition in known_items:
            expanded = row.to_dict()
            expanded["rule"] = f"{definition.label} ({key})"
            expanded["status"] = _status_for_signal(str(row.get("status")), value)
            expanded["actual"] = bool(value)
            expanded["threshold"] = False
            expanded["comparison"] = "必須為否"
            detail = _signal_detail(str(key), row_detail, definition)
            structured_items = _signal_condition_rows(str(key), row_detail)
            has_structured_judgement = bool(structured_items)
            structured_items = _apply_parent_judgement(expanded, structured_items)
            child_rows = (
                _structured_condition_rows(expanded, structured_items)
                if structured_items
                else [] if has_structured_judgement else _condition_rows(expanded, detail)
            )
            expanded["detail"] = "" if has_structured_judgement or child_rows else detail
            rows.append(expanded)
            rows.extend(child_rows)

    return pd.DataFrame(rows, columns=rule_frame.columns) if changed else rule_frame


def trim_number(text: str) -> str:
    return text.rstrip("0").rstrip(".")


def compact_amount(value: float, currency: bool = False, shares: bool = False) -> str:
    prefix = "$" if currency else ""
    suffix = ""
    divisor = 1
    if abs(value) >= 1_000_000_000:
        divisor = 1_000_000_000
        suffix = "B"
    elif abs(value) >= 1_000_000:
        divisor = 1_000_000
        suffix = "M"
    elif abs(value) >= 1_000:
        divisor = 1_000
        suffix = "K"
    unit = "" if suffix else (" 股" if shares else "")
    return f"{prefix}{trim_number(f'{value / divisor:.2f}')}{suffix}{unit}"


def coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def format_rule_number(value: Any, rule_text: str = "") -> str | None:
    rule_text = str(rule_text or "")
    number = coerce_number(value)
    if number is None:
        return None
    if "市值" in rule_text:
        return compact_amount(number, currency=True)
    if "美元成交額" in rule_text or "成交額" in rule_text:
        return compact_amount(number, currency=True)
    if "成交量" in rule_text:
        if abs(number) <= 20:
            return f"{trim_number(f'{number:.2f}')}x"
        return compact_amount(number, shares=True)
    if "業績" in rule_text:
        return f"{int(round(number))} 日"
    if any(token in rule_text for token in ("距", "幅度", "大市", "回撤", "延伸", "風險", "高位", "%")) and abs(number) <= 5:
        return f"{number * 100:+.1f}%"
    if any(token in rule_text for token in ("股價", "價格", "收市", "MA", "SMA", "高位", "低位")) and abs(number) >= 1:
        return f"${trim_number(f'{number:.2f}')}"
    if abs(number) >= 1_000_000:
        return compact_amount(number)
    if isinstance(value, int) or float(number).is_integer():
        return str(int(number))
    return trim_number(f"{number:.2f}")


def display_rule_value(value: Any, rule_text: str = "") -> str:
    """Keep raw RuleResult evidence intact while giving Arrow one display type."""
    if isinstance(value, dict):
        return ", ".join(f"{key}={display_rule_value(item, key)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(display_rule_value(item, rule_text) for item in value)
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    formatted_number = format_rule_number(value, rule_text)
    if formatted_number is not None:
        return formatted_number
    return (
        str(value)
        .replace("K 線", "陰陽燭")
        .replace("K線", "陰陽燭")
        .replace("反轉 陰陽燭", "反轉陰陽燭")
    )


# Score evidence is stored with stable internal keys because it is also used by
# the classifier.  Keep those keys out of the user-facing score checklist,
# including for older snapshots that saved the raw dictionaries.
_SCORE_FIELD_LABELS = {
    "價格條件類別": "價格條件類別",
    "來源": "來源",
    "source": "來源",
    "sources": "來源",
    "family": "技術條件",
    "families": "技術條件類別",
    "所有現存條件": "所有現存條件",
    "符合條件": "符合條件",
    "low": "區域下限",
    "high": "區域上限",
    "mid": "區域中位",
    "distance": "距離",
    "range_pct": "區間幅度",
    "risk_pct": "風險百分比",
    "ATR_pct": "ATR 百分比",
    "pct_from_high": "距離高位",
    "range_contracted": "區間已收窄",
    "is_confluence": "是否為匯聚區",
    "direction": "方向",
    "days": "持續日數",
    "duration_bars": "持續交易日",
    "first_touch": "開始位置",
    "last_touch": "結束位置",
    "is_primary_pivot": "主要樞紐區",
    "stop_eligible": "可作止損參考",
    "index": "資料位置",
}

_SCORE_VALUE_LABELS = {
    "gap_fill": "裂口回補",
    "pivot_zone": "樞紐區",
    "short_ma_cluster": "短期均線匯聚",
    "support_zone": "支持區",
    "sma50": "50MA",
    "swing_low": "波段低位",
    "role_reversal": "原阻力轉支持",
    "confluence": "多重條件匯聚",
    "compact_base": "緊密底部區",
    "MA 10 / MA 20": "10MA / 20MA 匯聚區",
    "MA 50": "50MA 附近",
    "Support Zone": "支持區",
    "20d swing low": "近 20 日波段低位",
    "Prior 20d pivot held": "原 20 日突破位轉為支持",
    "Gap Fill": "未完全回補裂口",
    "Pivot Zone": "樞紐區",
    "20d compact base": "20 日緊密底部區",
    "up": "向上",
    "down": "向下",
}


def _score_field_label(field: Any) -> str:
    text = str(field)
    return _SCORE_FIELD_LABELS.get(text, text if any("\u4e00" <= char <= "\u9fff" for char in text) else "資料")


def _score_value_label(value: Any) -> str:
    if isinstance(value, str):
        if "+" in value:
            return " + ".join(
                _SCORE_VALUE_LABELS.get(part.strip(), part.strip())
                for part in value.split("+")
                if part.strip()
            )
        return _SCORE_VALUE_LABELS.get(value, value)
    return str(value)


def _score_scalar(value: Any, field: Any = "") -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    field_name = str(field)
    if field_name in {
        "low", "high", "mid", "close", "latest_close", "entry_price", "price",
        "收市價", "股價", "MA10", "MA20", "MA50", "MA200", "SMA10", "SMA20", "SMA50", "SMA200", "ATR",
    }:
        number = coerce_number(value)
        return f"${number:,.2f}" if number is not None else _score_value_label(value)
    if field_name in {"distance", "distance_pct", "range_pct", "risk_pct", "ATR_pct", "pct_from_high"}:
        number = coerce_number(value)
        if number is None:
            return _score_value_label(value)
        return "0.0%" if abs(number) < 0.0005 else f"{number:+.1%}"
    if field_name in {"dry_up_ratio", "volume_ratio", "當日/20日均量", "3日/20日均量"}:
        number = coerce_number(value)
        return f"{number:.2f}x" if number is not None else _score_value_label(value)
    if field_name in {"days", "duration_bars", "first_touch", "last_touch", "index"}:
        number = coerce_number(value)
        return f"{int(number)} 日" if number is not None and field_name in {"days", "duration_bars"} else str(int(number)) if number is not None else _score_value_label(value)
    return _score_value_label(value)


def display_score_value(value: Any) -> str:
    """Render score evidence in plain Chinese for current and old snapshots."""
    if isinstance(value, dict):
        parts = []
        for field, nested in value.items():
            label = _score_field_label(field)
            if isinstance(nested, (dict, list, tuple)):
                rendered = display_score_value(nested)
            else:
                rendered = _score_scalar(nested, field)
            parts.append(f"{label}：{rendered}")
        return "\n".join(parts)
    if isinstance(value, (list, tuple)):
        return "、".join(
            display_score_value(item) if isinstance(item, (dict, list, tuple)) else _score_scalar(item)
            for item in value
        )
    return _score_scalar(value)


def display_score_actual(value: Any, rule_text: str = "") -> str:
    """Render one score condition as a concise, readable actual value."""
    rule = str(rule_text or "")
    if rule == "META 匯聚度" and isinstance(value, dict):
        all_categories = value.get("所有現存條件") or value.get("價格條件類別") or value.get("families") or []
        passed_categories = value.get("符合條件") or value.get("價格條件類別") or value.get("families") or []
        if not isinstance(all_categories, (list, tuple)):
            all_categories = [all_categories]
        if not isinstance(passed_categories, (list, tuple)):
            passed_categories = [passed_categories]
        if not all_categories:
            return "沒有獨立匯聚條件"
        passed_set = {str(item) for item in passed_categories}
        condition_lines = []
        passed_index = 0
        for category in all_categories:
            label = display_score_value(category)
            if str(category) in passed_set:
                condition_points = 8 if passed_index < 3 else 6 if passed_index == 3 else 0
                points_note = "已達上限，額外 0 分" if condition_points == 0 else f"計入 {condition_points} 分"
                condition_lines.append(f"✅ {label}：{points_note}")
                passed_index += 1
            else:
                condition_lines.append(f"❌ {label}：0 分")
        subtotal = min(len(passed_categories) * 8, 30)
        return "\n".join([
            *condition_lines,
            f"合計：{len(passed_categories)} 種符合條件，計入 {subtotal} 分／30 分",
        ])
    if rule == "距離選定 META 區" and isinstance(value, dict):
        low = _score_scalar(value.get("low"), "low")
        high = _score_scalar(value.get("high"), "high")
        source = _score_scalar(value.get("source"), "source")
        zone = f"META 區 {low}–{high}" if low != "-" and high != "-" else "META 區資料不足"
        current = value.get("現時股價")
        distance = value.get("距離區頂")
        position = value.get("是否在區內")
        context = []
        if current is not None:
            context.append(f"現價 {_score_scalar(current, 'price')}")
        if distance is not None:
            context.append(f"距離區頂 {_score_scalar(distance, 'distance_pct')}")
        if position is not None:
            context.append("位於 META 區內" if position else "位於 META 區外")
        zone_lines = [zone, *context]
        if source != "-":
            zone_lines.append(f"來源：{source}")
        return "\n".join(zone_lines)
    if rule in {"三日成交量收縮", "技術止損風險"} and not isinstance(value, (dict, list, tuple)):
        field = "dry_up_ratio" if rule == "三日成交量收縮" else "risk_pct"
        return _score_scalar(value, field)
    if rule.startswith("跑贏大市") and not isinstance(value, (dict, list, tuple)):
        return _score_scalar(value, "distance_pct")
    if rule in {"區間收窄", "陽燭收市", "MACD 柱狀體回升", "看漲反轉陰陽燭"}:
        return _score_scalar(value)
    if rule == "距離突破樞紐" and not isinstance(value, (dict, list, tuple)):
        return _score_scalar(value, "distance_pct")
    if rule == "RSI 動能區間" and not isinstance(value, (dict, list, tuple)):
        number = coerce_number(value)
        return f"{number:.1f}" if number is not None else _score_value_label(value)
    return display_score_value(value)


def display_rule_frame(group_rules: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = expand_signal_dict_rows(group_rules)[columns].copy()
    if "rule" in frame:
        frame["rule"] = frame["rule"].map(
            lambda value: (
                str(value)
                .replace("K 線", "陰陽燭")
                .replace("K線", "陰陽燭")
                .replace("反轉 陰陽燭", "反轉陰陽燭")
            )
        )
    for column in ("actual", "threshold", "detail"):
        if column in frame:
            frame[column] = [
                display_rule_value(value, rule)
                for value, rule in zip(frame[column], frame.get("rule", pd.Series([""] * len(frame))))
            ]
    return frame
