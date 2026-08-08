import html
import math

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from core.analysis.contracts import ScreeningSettings
from core.analysis.explainability import build_decision_tree, build_rule_checklist, build_score_checklist, earnings_context, score_breakdown
from core.analysis.momentum import rebuild_entry_plan_from_snapshot, run_tech_analysis
from repositories.storage import load_ohlcv_cache, load_stock_cache, save_stock_cache
from services.stock_service import analyze_single_stock, completed_daily_ohlcv
from services.universe_service import load_symbol_universe
from ui.rule_display import display_rule_frame, display_score_actual
from ui.stock_snapshot import comparable_timestamp, row_freshness, row_ohlcv_frame, row_series_last_date, row_updated_at
from utils import normalize_holding_action, normalize_setup_status, render_stock_detail, setup_phase_display_text


def _rows_from_state(key):
    data = st.session_state.get(key)
    if data is None:
        return pd.DataFrame()
    return data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


def _has_display_value(value, *, field=None, symbol=None):
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    if field == "name" and symbol and text.upper() == str(symbol).upper():
        return False
    return bool(text) and text.lower() not in {"nan", "none", "未知"}


def _freshest_row(symbol, matched_rows):
    candidates = matched_rows.to_dict(orient="records") if not matched_rows.empty else []
    cached_stock = load_stock_cache(symbol)
    if cached_stock:
        candidates.append(cached_stock)
    universe = load_symbol_universe()
    if not universe.empty and "symbol" in universe.columns:
        universe_match = universe[universe["symbol"].astype(str).str.upper() == symbol]
        if not universe_match.empty:
            candidates.append(universe_match.iloc[0].to_dict())
    if not candidates:
        return None

    newest = dict(max(candidates, key=row_freshness))
    metadata_fields = (
        "name", "exchange", "financial_currency", "market_cap", "earnings_date",
        "is_earnings_estimate", "sector_name", "industry_name", "avgvol3m",
    )
    for field in metadata_fields:
        if _has_display_value(newest.get(field), field=field, symbol=symbol):
            continue
        for candidate in sorted(candidates, key=row_updated_at, reverse=True):
            value = candidate.get(field)
            if _has_display_value(value, field=field, symbol=symbol):
                newest[field] = value
                break
    return pd.Series(newest)


def _display_rule_frame(group_rules, columns):
    return display_rule_frame(group_rules, columns)


def _polish_trace_rule_wording(display_frame):
    if "rule" not in display_frame:
        return display_frame

    data_rule_mask = display_frame["rule"] == "日線資料完整"
    if data_rule_mask.any():
        if "threshold" in display_frame:
            display_frame.loc[data_rule_mask, "threshold"] = "最後一個完成交易日"
        if "comparison" in display_frame:
            display_frame.loc[data_rule_mask, "comparison"] = "有資料"

    volume_ratio_mask = display_frame["rule"] == "META 量縮滿分（3 日均量 / 20 日均量）"
    for column in ("actual", "threshold"):
        if column not in display_frame:
            continue
        display_frame.loc[volume_ratio_mask, column] = display_frame.loc[
            volume_ratio_mask, column
        ].map(lambda value: f"{value}x" if value not in (None, "") and not str(value).endswith("x") else value)
    for column in ("actual", "threshold", "detail"):
        if column in display_frame:
            display_frame[column] = display_frame[column].map(
                lambda value: str(value)
                .replace("K 線", "陰陽燭")
                .replace("K線", "陰陽燭")
                .replace("反轉 陰陽燭", "反轉陰陽燭")
                .replace("｜", "\n")
                .replace(" | ", "\n")
                .replace("；", "\n")
                .replace(", ", "\n")
                .replace("共同 3% META", "共同 5% META")
                if isinstance(value, str)
                else value
            )
    return display_frame


_STATUS_DISPLAY = {
    "Pass": "通過",
    "Fail": "未通過",
    "Needs review": "需覆核",
    "Not applicable": "不適用",
}

_GROUP_DISPLAY = {
    "Universe": "基本資格",
    "Stage 2": "第二階段趨勢條件",
    "Setup": "進場結構條件",
    "Risk": "風險條件",
    "Earnings Risk": "業績風險",
    "Setup Score": "進場評分明細",
    "Risk Score": "風險評分明細",
}


def _rule_table_columns(score_table=False):
    return (
        ["rule", "actual", "status", "score", "detail"]
        if score_table
        else ["rule", "status", "actual", "threshold", "comparison", "detail"]
    )


def _rule_table_column_config(score_table=False):
    return {
        "rule": "規則",
        "status": "結果",
        "actual": "實際值",
        "score": "得分",
        "threshold": "門檻",
        "comparison": "比較方式",
        "detail": "計分條件" if score_table else "說明",
    }


def _html_cell(value):
    text = "" if value is None or value is pd.NA else str(value)
    return html.escape(text).replace("\n", "<br>")


def _render_static_rule_table(
    display_frame,
    *,
    score_table=False,
    compact=False,
    narrow_status=False,
):
    labels = _rule_table_column_config(score_table)
    header = "".join(
        f'<th class="trace-rule-col-{html.escape(column)}">{html.escape(labels.get(column, column))}</th>'
        for column in display_frame.columns
    )
    body_rows = []
    for _, row in display_frame.iterrows():
        row_rule = str(row.get("rule", "")).strip()
        row_class = "trace-subrow" if _is_trace_subrow(row_rule) else ""
        cells = "".join(
            f'<td class="trace-rule-col-{html.escape(column)}">{_html_cell(row[column])}</td>'
            for column in display_frame.columns
        )
        body_rows.append(f'<tr class="{row_class}">{cells}</tr>')
    st.markdown(
        (
            '<div class="trace-rule-table-wrap">'
            f'<table class="trace-rule-table{" compact-gate-table" if compact else ""}'
            f'{" narrow-status" if narrow_status else ""}">'
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _is_trace_subrow(rule_name):
    return rule_name in {
        "條件 A",
        "條件 B",
        "條件 C",
        "成交量確認",
        "判斷",
        "實際風險",
        "長期趨勢",
        "追價上限",
        "SMA10 距離",
        "SMA20 / ATR 距離",
    }


def _render_rule_table(rule_frame, *, score_table=False, key=None):
    if rule_frame.empty:
        st.info("此步驟沒有適用規則。")
        return

    if score_table:
        source_frame = _display_rule_frame(rule_frame, ["rule", "status", "actual", "threshold", "detail"])
        display_frame = pd.DataFrame({
            "rule": source_frame["rule"],
            # In a score row, the internal ``actual`` field is the earned
            # points while ``threshold`` stores the evidence value.  Render
            # that evidence separately so old snapshots do not leak raw
            # classifier names such as ``gap_fill`` into the UI.
            "actual": [
                display_score_actual(value, rule)
                for value, rule in zip(rule_frame["threshold"], rule_frame["rule"])
            ],
            "status": source_frame["status"].map(_SCORE_STATUS_DISPLAY).fillna("—"),
            "score": source_frame["actual"],
            "detail": source_frame["detail"],
        })
    else:
        columns = _rule_table_columns(False)
        display_frame = _display_rule_frame(rule_frame, columns)
        display_frame["status"] = display_frame["status"].map(_STATUS_DISPLAY).fillna(display_frame["status"])
    display_frame = _polish_trace_rule_wording(display_frame)
    if "detail" in display_frame and not score_table:
        display_frame = display_frame.drop(columns=["detail"])
    elif "detail" in display_frame:
        detail_values = display_frame["detail"].fillna("").astype(str).str.strip()
        if detail_values.eq("").all():
            display_frame = display_frame.drop(columns=["detail"])
    _render_static_rule_table(display_frame, score_table=score_table)


_COMPACT_STATUS_DISPLAY = {
    "Pass": "✅",
    "Fail": "❌",
    "Needs review": "⚠️",
    "Not applicable": "—",
}

_SCORE_STATUS_DISPLAY = {
    "Pass": "✅ 符合",
    "Fail": "❌ 不符合",
    "Needs review": "⚠️ 部分符合",
    "Not applicable": "— 不適用",
}


def _compact_gate_rule_text(rule, threshold, comparison):
    rule_text = str(rule or "").strip()
    threshold_text = str(threshold or "").strip()
    comparison_text = str(comparison or "").strip()
    if rule_text == "日線資料完整":
        return "日線資料完整"
    if rule_text == "最終分類":
        return "最終分類"
    if rule_text == "距離突破入場參考":
        normalized_threshold = threshold_text.replace(", ", " | ")
        bounds = [
            part.split("=", 1)[-1]
            for part in normalized_threshold.split(" | ")
            if part.strip()
        ]
        if len(bounds) == 2:
            return f"{rule_text}：{bounds[0]} 至 {bounds[1]}"
    if "距 52 週高位不超過" in rule_text:
        return rule_text
    if rule_text in {"200 日均線向上"} or rule_text.startswith("跑贏大市（"):
        return rule_text
    if any(
        token in rule_text
        for token in ("(holds_breakout_pivot)", "(holds_short_term_support)", "(within_chase_limit)")
    ):
        return rule_text
    rule_text = rule_text.removesuffix("達標")
    if not comparison_text or not threshold_text or threshold_text == "-":
        return rule_text
    if threshold_text in rule_text or comparison_text in rule_text:
        return rule_text
    return f"{rule_text} {comparison_text} {threshold_text}"


def _render_compact_gate_table(rule_frame, *, narrow_status=False):
    if rule_frame.empty:
        st.info("此步驟沒有適用規則。")
        return
    display_frame = _display_rule_frame(
        rule_frame,
        ["rule", "status", "actual", "threshold", "comparison", "detail"],
    ).reset_index(drop=True)
    display_frame = _polish_trace_rule_wording(display_frame)
    raw_rows = rule_frame.reset_index(drop=True).to_dict(orient="records")

    def price_part(label, value):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return f"{label} 資料不足"
        if not math.isfinite(numeric_value):
            return f"{label} 資料不足"
        return f"{label} ${numeric_value:.2f}"

    for index, raw_row in enumerate(raw_rows):
        actual = raw_row.get("actual")
        rule_name = str(raw_row.get("rule", ""))
        if rule_name == "最終分類":
            display_frame.loc[index, "actual"] = _candidate_structure_label(str(actual))
        if not isinstance(actual, dict):
            continue
        if "距 52 週高位不超過" in rule_name:
            price = actual.get("現時股價")
            high = actual.get("52 週高位")
            distance = actual.get("距高位")
            if not all(value is not None for value in (price, high, distance)):
                continue
            display_frame.loc[index, "actual"] = (
                f"現價 ${float(price):.2f} | 52 週高位 ${float(high):.2f} | "
                f"距 52 週高位 {float(distance):.1%}"
            )
        elif rule_name == "股價 >= MA50 >= MA200":
            display_frame.loc[index, "actual"] = " | ".join(
                price_part(label, actual.get(label))
                for label in ("股價", "MA50", "MA200")
            )
        elif rule_name == "MA10 > MA20":
            display_frame.loc[index, "actual"] = " | ".join(
                price_part(label, actual.get(label))
                for label in ("MA10", "MA20")
            )
    display_frame["rule"] = [
        _compact_gate_rule_text(rule, threshold, comparison)
        for rule, threshold, comparison in zip(
            display_frame["rule"],
            display_frame["threshold"],
            display_frame["comparison"],
        )
    ]
    raw_status = display_frame["status"].copy()
    display_frame["status"] = raw_status.map(_COMPACT_STATUS_DISPLAY).fillna(raw_status).fillna("—")
    review_mask = raw_status.eq("Needs review")
    review_notes = [
        str(row.get("detail") or "").strip() or "需要覆核"
        for _, row in display_frame.loc[review_mask].iterrows()
    ]
    _render_static_rule_table(
        display_frame[["rule", "actual", "status"]],
        compact=True,
        narrow_status=narrow_status,
    )
    if review_notes:
        note_items = "".join(
            f'<div class="trace-warning-note">⚠️ {_html_cell(note)}</div>'
            for note in review_notes
        )
        st.markdown(
            f'<div class="trace-warning-notes">{note_items}</div>',
            unsafe_allow_html=True,
        )


def _render_logic_summary(items):
    if not items:
        return

    blocks = []
    for item in items:
        tone = "pass" if item.get("favourable", False) else "review"
        checks = item.get("checks") or []
        checks_html = ""
        if checks:
            check_rows = []
            for check in checks:
                check_tone = "pass" if check.get("passed", False) else "fail"
                icon = "✅" if check.get("passed", False) else "❌"
                detail = check.get("detail", "")
                detail_html = (
                    f'<small>{html.escape(str(detail))}</small>'
                    if detail
                    else ""
                )
                check_rows.append(
                    (
                        f'<div class="trace-logic-check {check_tone}">'
                        f'<span class="trace-logic-check-icon">{icon}</span>'
                        '<div class="trace-logic-check-copy">'
                        f'<span>{html.escape(str(check["label"]))}</span>'
                        f"{detail_html}"
                        "</div></div>"
                    )
                )
            checks_html = f'<div class="trace-logic-checks">{"".join(check_rows)}</div>'
        hint_html = ""
        if not checks and item.get("hint"):
            hint_html = f'<small>{html.escape(str(item["hint"]))}</small>'
        blocks.append(
            (
                f'<div class="trace-logic-card {tone}">'
                f'<span>{html.escape(item["label"])}</span>'
                f'<strong>{html.escape(item["value"])}</strong>'
                f"{hint_html}{checks_html}"
                "</div>"
            )
        )
    detailed_class = " detailed" if any(item.get("checks") for item in items) else ""
    st.markdown(
        f'<div class="trace-logic-grid{detailed_class}">{"".join(blocks)}</div>',
        unsafe_allow_html=True,
    )


def _render_step_card(step_label, title, answer, favourable, final=False, hide_answer=False):
    tone = "pass" if favourable else "review"
    final_class = " decision-final" if final else ""
    answer_html = "" if hide_answer else f'<span>{html.escape(answer)}</span>'
    st.markdown(
        (
            f'<div class="decision-step-card {tone}{final_class}">'
            f'<span class="decision-step-label">{html.escape(step_label)}</span>'
            f'<strong>{html.escape(title)}</strong>'
            f"{answer_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _flag_logic_item(label, active, hint):
    return {
        "label": label,
        "value": "否" if not active else "是",
        "hint": hint,
        "favourable": not active,
    }


def _positive_logic_item(label, passed, hint, applicable=True):
    return {
        "label": label,
        "value": "是" if passed else "否" if applicable else "不適用",
        "hint": hint,
        "favourable": passed and applicable,
    }


def _validity_logic_summary(validity_flags, validity_details=None, breakout_context=False):
    flags = validity_flags or {}
    details = validity_details or {}
    holds_pivot = bool(flags.get("holds_breakout_pivot"))
    holds_support = bool(flags.get("holds_short_term_support"))
    pivot_detail = details.get("holds_breakout_pivot") or {}
    pivot_source = pivot_detail.get("pivot_source_date")
    pivot_window_start = pivot_detail.get("pivot_window_start")
    pivot_window_end = pivot_detail.get("pivot_window_end")
    pivot_hint = "最新收市 >= 原 20 日突破樞紐"
    if pivot_source and pivot_window_start and pivot_window_end:
        pivot_hint += (
            f"；樞紐取自確認突破前 20 個完成交易日 "
            f"{pivot_window_start} 至 {pivot_window_end}，最高價日期 {pivot_source}"
        )
    items = [
        _positive_logic_item("守在原突破樞紐上", holds_pivot, pivot_hint, breakout_context),
        _positive_logic_item("守在短中期支持上", holds_support, "最新收市 >= MA20 及 MA50", breakout_context),
    ]
    breakout_valid = bool(breakout_context and holds_pivot and holds_support)
    items.append({
        "label": "突破仍然有效？",
        "value": "是" if breakout_valid else "否" if breakout_context else "不適用",
        "hint": "兩者需同時生效",
        "favourable": breakout_valid,
    })
    return items


def _extension_logic_summary(extension_flags, extension_details=None):
    flags = extension_flags or {}
    detail = (
        (extension_details or {}).get("within_chase_limit")
        or (extension_details or {}).get("above_entry_threshold")
        or {}
    )
    within_limit = bool(flags.get("within_chase_limit", not flags.get("above_entry_threshold")))
    latest = detail.get("latest_close")
    entry = detail.get("entry_price")
    ceiling = detail.get("entry_ceiling")
    ceiling_pct = detail.get("ceiling_pct")
    if all(value is not None for value in (latest, entry, ceiling, ceiling_pct)):
        hint = (
            f"現價 ${float(latest):.2f}；入場參考 ${float(entry):.2f} + "
            f"{float(ceiling_pct):.1%} = 追價上限 ${float(ceiling):.2f}"
        )
    else:
        hint = "現價不可超過入場參考加追價上限"
    return [_positive_logic_item("現價未超過追價上限", within_limit, hint)]


def _candidate_structure_label(phase):
    return {
        "fresh_breakout": "確認突破",
        "near_breakout": "接近突破",
        "extended_breakout": "延伸突破",
        "failed_breakout": "突破失敗",
        "pullback_entry": "META 回調入場",
        "pullback_forming": "META 回調形成中",
        "not_recommended": "未通過必要條件",
        "unclear_structure": "未形成清晰結構",
    }.get(phase, "未形成清晰結構")


def _latest_number(row_data, key):
    value = row_data.get(key)
    if isinstance(value, list):
        value = value[-1] if value else None
    try:
        return None if value is None or pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None


def _candidate_flags_for_display(row_data):
    """Rebuild candidate gates for older snapshots that predate saved flags."""
    evidence = row_data.get("setup_evidence") or {}
    saved = evidence.get("candidate_flags")
    if isinstance(saved, dict) and saved:
        return saved

    phase = str(row_data.get("setup_phase", "unclear_structure"))
    failure_flags = evidence.get("failure_flags") or {}
    validity_flags = evidence.get("validity_flags") or {}
    if not validity_flags:
        validity_flags = {
            "holds_breakout_pivot": not bool(failure_flags.get("failed_breakout")),
            "holds_short_term_support": not bool(failure_flags.get("breaks_support")),
        }
    extension_flags = evidence.get("extension_flags") or {}
    breakout_score = int(row_data.get("breakout_score") or 0)
    meta_score = int(row_data.get("meta_score") or 0)
    meta_parts = evidence.get("meta_score_parts") or {}
    chart_evidence = evidence.get("chart_evidence") or {}
    has_meta_zone = bool(chart_evidence.get("selected_meta_zone"))
    dist_to_entry = _latest_number(evidence, "dist_to_entry")
    if dist_to_entry is None:
        dist_to_entry = _latest_number(row_data, "dist_pivot_20")
    volume_ratio = _latest_number(row_data, "volume_ratio")
    extension_clear = bool(
        extension_flags.get(
            "within_chase_limit",
            not bool(extension_flags.get("above_entry_threshold")),
        )
    )
    structural_failure = not all(bool(value) for value in validity_flags.values())
    major_failure = not bool(validity_flags.get("holds_short_term_support"))
    confluence = int(meta_parts.get("meta_confluence") or 0)
    reversal = int(meta_parts.get("reversal_signal") or 0)

    fresh_breakout = bool(
        breakout_score >= 70
        and dist_to_entry is not None
        and 0 <= dist_to_entry <= 0.01
        and volume_ratio is not None
        and volume_ratio >= 1.10
        and extension_clear
    )
    near_breakout = bool(
        phase == "near_breakout"
        or (
            breakout_score >= 65
            and dist_to_entry is not None
            and -0.03 <= dist_to_entry < 0
            and not structural_failure
        )
    )
    pullback_entry = bool(meta_score >= 55 and confluence >= 16 and reversal >= 6 and not major_failure)
    pullback_forming = bool(meta_score >= 45 and confluence >= 16 and reversal < 6 and not major_failure)

    # Very old snapshots may contain only the final phase and aggregate scores.
    fresh_breakout = fresh_breakout or phase == "fresh_breakout"
    pullback_entry = pullback_entry or (phase == "pullback_entry" and has_meta_zone)
    pullback_forming = pullback_forming or (phase == "pullback_forming" and has_meta_zone)
    return {
        "fresh_breakout": fresh_breakout,
        "near_breakout": near_breakout,
        "pullback_entry": pullback_entry,
        "pullback_forming": pullback_forming,
    }


def _candidate_flag_details_for_display(row_data):
    """Return auditable candidate inputs for both current and older snapshots."""
    evidence = row_data.get("setup_evidence") or {}
    saved = evidence.get("candidate_flag_details") or {}
    meta_parts = evidence.get("meta_score_parts") or {}
    chart_evidence = evidence.get("chart_evidence") or {}
    failure_flags = evidence.get("failure_flags") or {}
    validity_flags = evidence.get("validity_flags") or {}
    if not validity_flags:
        validity_flags = {
            "holds_breakout_pivot": not bool(failure_flags.get("failed_breakout")),
            "holds_short_term_support": not bool(failure_flags.get("breaks_support")),
        }

    breakout_score = int(row_data.get("breakout_score") or evidence.get("breakout_score") or 0)
    meta_score = int(row_data.get("meta_score") or evidence.get("meta_score") or 0)
    dist_to_entry = _latest_number(evidence, "dist_to_entry")
    if dist_to_entry is None:
        dist_to_entry = _latest_number(row_data, "dist_pivot_20")
    volume_ratio = _latest_number(row_data, "volume_ratio")
    has_meta_zone = bool(chart_evidence.get("selected_meta_zone"))
    structural_failure = not all(bool(value) for value in validity_flags.values())
    major_failure = not bool(validity_flags.get("holds_short_term_support", True))

    defaults = {
        "fresh_breakout": {
            "score": breakout_score,
            "score_threshold": 70,
            "dist_to_entry": dist_to_entry,
            "distance_min": 0.0,
            "distance_max": 0.01,
            "volume_ratio": volume_ratio,
            "volume_threshold": 1.10,
            "extension_clear": True,
        },
        "near_breakout": {
            "score": breakout_score,
            "score_threshold": 65,
            "dist_to_entry": dist_to_entry,
            "distance_min": -0.03,
            "distance_max": 0.0,
            "recent_breakout": False,
            "failure_clear": not structural_failure,
        },
        "pullback_entry": {
            "score": meta_score,
            "score_threshold": 55,
            "meta_confluence": int(meta_parts.get("meta_confluence") or 0),
            "meta_confluence_threshold": 16,
            "reversal_signal": int(meta_parts.get("reversal_signal") or 0),
            "reversal_threshold": 6,
            "has_meta_zone": has_meta_zone,
            "major_failure_clear": not major_failure,
        },
        "pullback_forming": {
            "score": meta_score,
            "score_threshold": 45,
            "meta_confluence": int(meta_parts.get("meta_confluence") or 0),
            "meta_confluence_threshold": 16,
            "reversal_signal": int(meta_parts.get("reversal_signal") or 0),
            "reversal_threshold": 6,
            "has_meta_zone": has_meta_zone,
            "major_failure_clear": not major_failure,
        },
    }
    return {
        key: {**default, **(saved.get(key) or {}), "has_meta_zone": has_meta_zone}
        if key.startswith("pullback_")
        else {**default, **(saved.get(key) or {})}
        for key, default in defaults.items()
    }


def _candidate_checks_for_display(candidate, details):
    def numeric(value):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def pct(value):
        return "欠缺" if value is None else f"{float(value):+.1%}"

    def multiple(value):
        return "欠缺" if value is None else f"{float(value):.2f}x"

    def check(label, passed, detail):
        return {"label": label, "passed": bool(passed), "detail": detail}

    score = int(details.get("score") or 0)
    score_threshold = int(details.get("score_threshold") or 0)
    if candidate == "fresh_breakout":
        distance = details.get("dist_to_entry")
        distance_min = float(details.get("distance_min", 0.0))
        distance_max = float(details.get("distance_max", 0.01))
        volume = details.get("volume_ratio")
        volume_threshold = float(details.get("volume_threshold", 1.10))
        return [
            check("突破評分", score >= score_threshold, f"{score}/90；門檻 >= {score_threshold}"),
            check(
                "已突破且仍在追價範圍",
                distance is not None and distance_min <= float(distance) <= distance_max,
                f"距入場價 {pct(distance)}；範圍 {pct(distance_min)} 至 {pct(distance_max)}",
            ),
            check(
                "突破成交量",
                volume is not None and float(volume) >= volume_threshold,
                f"{multiple(volume)}；門檻 >= {volume_threshold:.2f}x",
            ),
        ]
    if candidate == "near_breakout":
        distance = details.get("dist_to_entry")
        distance_min = float(details.get("distance_min", -0.03))
        distance_max = float(details.get("distance_max", 0.0))
        recent_breakout = bool(details.get("recent_breakout"))
        recent_detail = details.get("recent_breakout_detail") or {}
        recent_date = recent_detail.get("date")
        if recent_breakout:
            breakout_close = numeric(recent_detail.get("close"))
            confirmation = numeric(recent_detail.get("confirmation_price"))
            volume_ratio = numeric(recent_detail.get("volume_ratio"))
            recent_hint = " | ".join(
                value
                for value in [
                    str(recent_date) if recent_date else None,
                    f"收市 ${breakout_close:,.2f}" if breakout_close is not None else None,
                    f"確認價 ${confirmation:,.2f}" if confirmation is not None else None,
                    f"成交量 {volume_ratio:.2f}x" if volume_ratio is not None else None,
                ]
                if value
            )
        else:
            recent_hint = "近 3 日沒有同時通過價格及成交量確認"
        failure_clear = bool(details.get("failure_clear"))
        return [
            check("突破評分", score >= score_threshold, f"{score}/90；門檻 >= {score_threshold}"),
            check(
                "接近但尚未到入場價",
                distance is not None and distance_min <= float(distance) < distance_max,
                f"距入場價 {pct(distance)}；範圍 {pct(distance_min)} 至 <{pct(distance_max)}",
            ),
            check("近 3 日未確認突破", not recent_breakout, recent_hint),
            check("結構仍然有效", failure_clear, "是" if failure_clear else "否；出現結構失效訊號"),
        ]

    confluence = int(details.get("meta_confluence") or 0)
    confluence_threshold = int(details.get("meta_confluence_threshold") or 16)
    reversal = int(details.get("reversal_signal") or 0)
    reversal_threshold = int(details.get("reversal_threshold") or 6)
    has_meta_zone = bool(details.get("has_meta_zone"))
    failure_clear = bool(details.get("major_failure_clear"))
    is_entry = candidate == "pullback_entry"
    reversal_passed = reversal >= reversal_threshold if is_entry else reversal < reversal_threshold
    reversal_rule = f">= {reversal_threshold}" if is_entry else f"< {reversal_threshold}"
    return [
        check("已選中 META 匯聚區", has_meta_zone, "是" if has_meta_zone else "否；未找到真正匯聚交集"),
        check("META 評分", score >= score_threshold, f"{score}/80；門檻 >= {score_threshold}"),
        check("匯聚條件", confluence >= confluence_threshold, f"{confluence}；門檻 >= {confluence_threshold}"),
        check("反轉確認", reversal_passed, f"{reversal}；門檻 {reversal_rule}"),
        check("短中期支持未失守", failure_clear, "是" if failure_clear else "否；已跌穿短中期支持"),
    ]


def _breakout_level_context(row_data):
    level = _latest_number(row_data, "pivot_high_20")
    dates = row_data.get("date")
    highs = row_data.get("high")
    source_date = None
    if isinstance(dates, list) and isinstance(highs, list) and len(dates) >= 2 and len(highs) >= 2:
        window = list(zip(dates[:-1], highs[:-1]))[-20:]
        valid = []
        for date, high in window:
            try:
                value = float(high)
                if not pd.isna(value):
                    valid.append((date, value))
            except (TypeError, ValueError):
                continue
        if valid:
            source_date, source_high = max(valid, key=lambda item: item[1])
            if level is None:
                level = source_high
    entry = level * 1.005 if level is not None else None
    return level, entry, source_date


def _meta_source_label(source):
    labels = {
        "MA 10 / MA 20": "10MA / 20MA 匯聚區",
        "MA 50": "50MA 附近",
        "20d swing low": "近 20 日波段低位",
        "Prior 20d pivot held": "原 20 日突破位轉為支持",
        "Gap Fill": "未完全回補裂口",
        "20d compact base": "20 日緊密底部區",
        "Support Zone": "支持區",
    }
    parts = [labels.get(part.strip(), part.strip()) for part in str(source or "").split("+")]
    return " + ".join(part for part in parts if part) or "未有來源"


def _structure_level_summaries(row_data):
    level, entry, source_date = _breakout_level_context(row_data)
    zone_low = _latest_number(row_data, "entry_zone_low") or level
    zone_high = _latest_number(row_data, "entry_zone_high") or entry
    confirmation = _latest_number(row_data, "confirmation_price") or entry
    stop_price = _latest_number(row_data, "stop_price")
    breakout_value = (
        f"${zone_low:,.2f}–${zone_high:,.2f}"
        if zone_low is not None and zone_high is not None
        else "資料不足"
    )
    source_text = f"；最高價出現於 {source_date}" if source_date else ""
    confirmation_text = (
        f"確認價 ${confirmation:,.2f} = 樞紐 ${level:,.2f} × 1.005"
        if level is not None and confirmation is not None
        else "未能計算突破確認價"
    )
    stop_text = f"${stop_price:,.2f}" if stop_price is not None else "資料不足"

    evidence = row_data.get("setup_evidence") or {}
    chart_evidence = evidence.get("chart_evidence") or {}
    selected = chart_evidence.get("selected_meta_zone")
    if isinstance(selected, dict):
        low = _latest_number(selected, "low")
        high = _latest_number(selected, "high")
        meta_value = (
            f"${low:,.2f}–${high:,.2f}"
            if low is not None and high is not None
            else "資料不足"
        )
        meta_source = _meta_source_label(selected.get("source"))
    else:
        meta_value = "未選定"
        meta_source = "目前沒有被選作參考的 META 區"

    return [
        {
            "label": "突破入場價",
            "value": breakout_value,
            "hint": f"以最新交易日前 20 個完成交易日的最高價作為突破樞紐{source_text}；入場區由樞紐至 +0.5% 確認價。",
        },
        {
            "label": "突破確認價",
            "value": f"${confirmation:,.2f}" if confirmation is not None else "資料不足",
            "hint": f"{confirmation_text}；收市價站上此價，並通過突破成交量確認，才算確認突破。",
        },
        {
            "label": "突破止損參考",
            "value": stop_text,
            "hint": "突破類優先以包含現價的 Pivot Zone 下界作基礎，再加入 0.25 ATR 緩衝；沒有合適 Pivot Zone 才退回最近支持位，若風險超過上限則套用風險上限。",
        },
        {
            "label": "META 區",
            "value": meta_value,
            "hint": (
                f"META 用多個支持及匯聚條件描述承接範圍，因此顯示區間。選定來源：{meta_source}。"
                "只有獨立價格條件實際重疊時才形成交匯區；是否成為 META 類仍須通過匯聚、量縮及反轉規則。"
            ),
        },
    ]


def _final_price_summaries(row_data):
    """Show only the price plan that belongs to the final classification."""

    phase = str(row_data.get("setup_phase") or "")
    stop_price = _latest_number(row_data, "stop_price")
    stop_value = f"${stop_price:,.2f}" if stop_price is not None else "資料不足"
    stop_item = {
        "label": "止損參考",
        "value": stop_value,
        "hint": "以該分類的下方結構位作基礎，再加入 0.25 ATR 緩衝；若風險超過上限，套用風險上限。",
    }

    if phase in {"near_breakout", "fresh_breakout", "extended_breakout", "failed_breakout"}:
        confirmation = _latest_number(row_data, "confirmation_price")
        if confirmation is None:
            confirmation = _latest_number(row_data, "entry_price")
        chase_ceiling = confirmation * 1.01 if confirmation is not None else None
        volume_ratio = _latest_number(row_data, "volume_ratio")
        pivot_low = _latest_number(row_data, "stop_pivot_zone_low")
        pivot_high = _latest_number(row_data, "stop_pivot_zone_high")
        atr = _latest_number(row_data, "ATR")
        if pivot_low is not None and pivot_high is not None and atr is not None:
            stop_item["hint"] = (
                f"突破樞紐區 ${pivot_low:,.2f}–${pivot_high:,.2f}；"
                f"止損 = 區間下界 ${pivot_low:,.2f} - 0.25 × ATR ${atr:,.2f}"
                f" = {stop_value}。8% 只作最大風險上限，不是預設止損距離。"
            )
        return [
            {
                "label": "突破入場價",
                "value": f"${confirmation:,.2f}" if confirmation is not None else "資料不足",
                "hint": (
                    f"收市突破此價且成交量達20日均量 1.10x 才算確認；"
                    f"目前成交量為 {volume_ratio:.2f}x；確認後而未高於追價上限 "
                    f"${chase_ceiling:,.2f}，可留意突破入場。"
                    if volume_ratio is not None and chase_ceiling is not None
                    else "由突破區間上限計算；收市突破並通過成交量確認後，才可作入場參考。"
                ),
            },
            stop_item,
        ]

    if phase in {"pullback_forming", "pullback_entry"}:
        evidence = row_data.get("setup_evidence") or {}
        chart_evidence = evidence.get("chart_evidence") or {}
        selected = chart_evidence.get("selected_meta_zone")
        low = _latest_number(selected, "low") if isinstance(selected, dict) else None
        high = _latest_number(selected, "high") if isinstance(selected, dict) else None
        return [
            {
                "label": "META 入場區",
                "value": f"${low:,.2f}–${high:,.2f}" if low is not None and high is not None else "資料不足",
                "hint": "由實際重疊的多重支持及匯聚條件形成入場區；META 類需等待反轉確認。",
            },
            stop_item,
        ]

    return []


def _render_structure_levels(items):
    blocks = []
    for item in items or []:
        blocks.append(
            f'<div class="trace-level-card">'
            f'<span>{html.escape(item["label"])}</span>'
            f'<strong>{html.escape(item["value"])}</strong>'
            f'<small>{html.escape(item["hint"])}</small>'
            "</div>"
        )
    if blocks:
        st.markdown(
            f'<div class="trace-level-grid">{"".join(blocks)}</div>',
            unsafe_allow_html=True,
        )


def _render_decision_step(step, index):
    with st.container(border=True):
        step_col, table_col = st.columns([1, 4], vertical_alignment="top")
        with step_col:
            _render_step_card(
                step["label"],
                step["title"],
                step["answer"],
                step["favourable"],
                final=step.get("final", False),
                hide_answer=step.get("hide_answer", False),
            )
        with table_col:
            st.markdown(f"**{step['table_title']}**")
            _render_logic_summary(step.get("logic_summary"))
            _render_structure_levels(step.get("structure_levels"))
            if step.get("rule_sections"):
                for section in step["rule_sections"]:
                    st.caption(section["title"])
                    _render_rule_table(
                        section["rules"],
                        score_table=section.get("score_table", False),
                        key=f"trace_step_{index}_{section['title']}",
                    )
            elif step.get("hide_rules", False):
                pass
            elif step.get("compact_rules", False):
                _render_compact_gate_table(
                    step["rules"],
                    narrow_status=step.get("narrow_status", False),
                )
            else:
                _render_rule_table(
                    step["rules"],
                    score_table=step.get("score_table", False),
                    key=f"trace_step_{index}_{step['label']}",
                )


def _render_decision_flow(row_data, settings, rules):
    """Show the classification path with each step's metrics beside it."""
    phase = str(row_data.get("setup_phase", "unclear_structure"))
    prerequisites = [
        rule for rule in rules.to_dict(orient="records")
        if rule["group"] in {"Universe", "Stage 2"} and rule["status"] != "Not applicable"
    ]
    prerequisites_passed = bool(prerequisites) and all(rule["status"] == "Pass" for rule in prerequisites)
    evidence = row_data.get("setup_evidence") or {}
    chart_evidence = evidence.get("chart_evidence") or {}
    if phase in {"pullback_forming", "pullback_entry"} and "chart_evidence" in evidence and not chart_evidence.get("selected_meta_zone"):
        # Keep the explanation consistent with the same hard gate used by the
        # classifier: a META pullback cannot exist without a selected overlap.
        phase = "unclear_structure"
    failure_flags = evidence.get("failure_flags") or {}
    validity_flags = evidence.get("validity_flags") or {
        "holds_breakout_pivot": not bool(failure_flags.get("failed_breakout")),
        "holds_short_term_support": not bool(failure_flags.get("breaks_support")),
    }
    failure_details = evidence.get("failure_flag_details") or {}
    validity_details = evidence.get("validity_flag_details") or {
        "holds_breakout_pivot": failure_details.get("failed_breakout", {}),
        "holds_short_term_support": failure_details.get("breaks_support", {}),
    }
    breakout_context = bool(evidence.get("breakout_context"))
    breakout_invalid = bool(evidence.get("breakout_invalid"))
    extension_flags = evidence.get("extension_flags") or {}
    extension_details = evidence.get("extension_flag_details") or {}
    within_chase_range = bool(
        extension_flags.get(
            "within_chase_limit",
            not bool(extension_flags.get("above_entry_threshold")),
        )
    )
    final_labels = {
        "fresh_breakout": "突破買入",
        "near_breakout": "接近突破，繼續觀察",
        "pullback_entry": "回調買入",
        "pullback_forming": "等待回調確認",
        "extended_breakout": "突破已延伸，不追高",
        "failed_breakout": "突破失敗，不宜買入",
        "not_recommended": "未通過基本條件，不宜買入",
        "unclear_structure": "結構未形成，繼續觀察",
    }
    decision_rules = pd.DataFrame(build_decision_tree(row_data, settings))
    earnings = earnings_context(row_data)
    data_rule = decision_rules[decision_rules["rule"] == "日線資料完整"]
    setup_rules = rules[rules["group"] == "Setup"]
    risk_rules = rules[rules["group"] == "Risk"]
    has_entry = phase in {"fresh_breakout", "pullback_entry"}
    breakout_score_details = score_breakdown(row_data, source="breakout")
    meta_score_details = score_breakdown(row_data, source="meta")
    meta_score_title = meta_score_details["score_display"] + "　META 評分"
    if phase == "failed_breakout":
        meta_score_title += "（輔助參考；突破失敗優先）"
    candidate_flags = _candidate_flags_for_display(row_data)
    candidate_details = _candidate_flag_details_for_display(row_data)
    candidate_logic = [
        {
            "label": "確認突破候選",
            "value": "是" if candidate_flags.get("fresh_breakout") else "否",
            "checks": _candidate_checks_for_display(
                "fresh_breakout", candidate_details["fresh_breakout"]
            ),
            "favourable": bool(candidate_flags.get("fresh_breakout")),
        },
        {
            "label": "接近突破候選",
            "value": "是" if candidate_flags.get("near_breakout") else "否",
            "checks": _candidate_checks_for_display(
                "near_breakout", candidate_details["near_breakout"]
            ),
            "favourable": bool(candidate_flags.get("near_breakout")),
        },
        {
            "label": "META 回調入場候選",
            "value": "是" if candidate_flags.get("pullback_entry") else "否",
            "checks": _candidate_checks_for_display(
                "pullback_entry", candidate_details["pullback_entry"]
            ),
            "favourable": bool(candidate_flags.get("pullback_entry")),
        },
        {
            "label": "META 回調形成中",
            "value": "是" if candidate_flags.get("pullback_forming") else "否",
            "checks": _candidate_checks_for_display(
                "pullback_forming", candidate_details["pullback_forming"]
            ),
            "favourable": bool(candidate_flags.get("pullback_forming")),
        },
    ]
    matched_candidates = [
        label
        for key, label in (
            ("fresh_breakout", "確認突破"),
            ("near_breakout", "接近突破"),
            ("pullback_entry", "META 回調入場"),
            ("pullback_forming", "META 回調形成中"),
        )
        if candidate_flags.get(key)
    ]
    candidate_answer = " + ".join(matched_candidates) if matched_candidates else "未形成候選"
    final_rule = decision_rules[decision_rules["rule"] == "最終分類"]
    setup_status = normalize_setup_status(
        row_data.get("setup_status", row_data.get("status", "watchlist")),
        phase,
    )
    holding_action = normalize_holding_action(row_data.get("holding_action", "hold"), phase)
    action_display = {
        "breakout_buy": "空倉：留意突破",
        "breakout": "空倉：留意突破",
        "pullback_wait": "空倉：留意META",
        "pullback_buy": "空倉：留意META",
        "pullback": "空倉：留意META",
        "watchlist": "空倉：觀望後續",
        "not_recommended": "空倉：不宜買入",
    }
    holding_display = {
        "add": "持倉：考慮加倉",
        "sell_weakness": "持倉：趁弱賣出",
        "hold": "持倉：繼續持有",
    }
    extended_for_holding = bool(
        extension_flags.get("extended_from_sma10")
        or extension_flags.get("extended_by_atr")
    )
    holds_support = bool(validity_flags.get("holds_short_term_support", True))
    phase_display = _candidate_structure_label(phase)
    if setup_status in {"breakout_buy", "breakout"}:
        empty_action_hint = f"趨勢階段：{phase_display} | 接近或確認突破均歸入突破類，是否已觸發以趨勢階段為準"
    elif setup_status in {"pullback_wait", "pullback_buy", "pullback"}:
        empty_action_hint = f"趨勢階段：{phase_display} | META 回調形成中或已確認回調均歸入等待回調類"
    else:
        empty_action_hint = f"趨勢階段：{phase_display} | 入場尚未確認，因此維持觀望"

    if holding_action == "sell_weakness":
        holding_hint = "短中期支持失守 | 優先趁弱賣出"
    elif holding_action == "add":
        holding_hint = (
            f"趨勢階段：{phase_display} | 支持未失守 | "
            f"{'已過度延伸，需自行控制倉位' if extended_for_holding else '未過度延伸'} | "
            "重新出現可入場形態，因此可考慮加倉"
        )
    else:
        holding_hint = (
            f"趨勢階段：{phase_display} | "
            f"支持{'未失守' if holds_support else '已失守'} | "
            f"{'已過度延伸，但不自動等於趁強減持' if extended_for_holding else '未過度延伸'} | "
            "未跌穿支持且未有重新加倉訊號，因此繼續持有；倉位過重或已有利潤時，可自行考慮趁強減持"
        )

    badge_logic = [
        {
            "label": "空倉建議",
            "value": action_display.get(setup_status, "空倉：觀望後續"),
            "hint": empty_action_hint,
            "favourable": setup_status in {"breakout_buy", "breakout", "pullback_wait", "pullback_buy", "pullback"},
        },
        {
            "label": "持倉建議",
            "value": holding_display.get(holding_action, "持倉：繼續持有"),
            "hint": holding_hint,
            "favourable": holding_action in {"add", "hold"},
        },
        {
            "label": "趨勢階段",
            "value": setup_phase_display_text(phase),
            "hint": "固定優先次序：突破失敗 | 過度延伸 | 確認突破 | 接近突破 | META 回調入場 | META 回調形成中",
            "favourable": phase in {"fresh_breakout", "pullback_entry"},
        },
    ]

    steps = [
        {
            "label": "步驟 1",
            "title": "日線資料完整？",
            "answer": "是" if row_data.get("data_date") else "否",
            "favourable": bool(row_data.get("data_date")),
            "table_title": "日線資料",
            "rules": data_rule,
            "compact_rules": True,
        },
        {
            "label": "步驟 2",
            "title": "通過基本資格與第二階段條件？",
            "answer": "是" if prerequisites_passed else "否",
            "favourable": prerequisites_passed,
            "table_title": "基本資格與第二階段趨勢條件",
            "rules": rules[rules["group"].isin(["Universe", "Stage 2"])],
            "compact_rules": True,
            "narrow_status": True,
        },
        {
            "label": "步驟 3",
            "title": "並行計算兩套結構評分",
            "answer": f"突破 {breakout_score_details['score_display']} | META {meta_score_details['score_display']}",
            "favourable": phase not in {"unclear_structure", "not_recommended"},
            "table_title": "突破與 META 評分各自計算，不以兩者高低直接定案",
            "rules": pd.DataFrame(),
            "rule_sections": [
                {
                    "title": breakout_score_details["score_display"] + "　突破評分",
                    "rules": pd.DataFrame(build_score_checklist(row_data, source="breakout")),
                    "score_table": True,
                },
                {
                    "title": meta_score_title,
                    "rules": pd.DataFrame(build_score_checklist(row_data, source="meta")),
                    "score_table": True,
                },
            ],
        },
        {
            "label": "步驟 4",
            "title": "符合哪些候選結構？",
            "answer": candidate_answer,
            "favourable": bool(matched_candidates),
            "table_title": "各候選獨立判斷，可同時成立",
            "rules": pd.DataFrame(),
            "logic_summary": candidate_logic,
            "hide_rules": True,
        },
        {
            "label": "步驟 5",
            "title": "突破仍然有效？",
            "answer": "是" if breakout_context and not breakout_invalid else "否" if breakout_context else "不適用",
            "favourable": breakout_context and not breakout_invalid,
            "table_title": "突破後結構有效性檢查",
            "rules": setup_rules[
                setup_rules["rule"].isin(["突破結構仍然有效", "沒有已確認結構失敗"])
            ],
            "logic_summary": _validity_logic_summary(
                validity_flags,
                validity_details,
                breakout_context,
            ),
            "compact_rules": True,
        },
        {
            "label": "步驟 6",
            "title": "現價仍在合理追價範圍？",
            "answer": "是" if within_chase_range else "否",
            "favourable": within_chase_range,
            "table_title": "追價上限檢查",
            "rules": risk_rules[
                risk_rules["rule"].isin(["仍在合理追價範圍", "未過度延伸"])
            ],
            "compact_rules": True,
        },
        {
            "label": "業績風險",
            "title": "下一次業績公布安全？",
            "answer": "需覆核" if earnings.get("days") is None else f"{earnings['days']} 日",
            "favourable": earnings.get("status") == "Pass",
            "table_title": "業績風險",
            "rules": rules[rules["group"] == "Earnings Risk"],
            "compact_rules": True,
        },
        {
            "label": "最終分類",
            "title": f"趨勢：{setup_phase_display_text(phase).split('趨勢：', 1)[-1]}",
            "answer": "",
            "favourable": has_entry,
            "table_title": "判定趨勢優先次序：突破失敗 → 過度延伸 → 確認突破 → 接近突破 → META 回調入場 → META 回調形成中",
            "rules": final_rule,
            "logic_summary": badge_logic,
            "structure_levels": _final_price_summaries(row_data),
            "compact_rules": True,
            "final": True,
            "hide_answer": True,
        },
    ]

    st.markdown(
        """
        <style>
        .decision-step-card {min-height:112px; padding:12px 14px; border:1px solid #4b5563; border-radius:7px; background:#1d232d; color:#f9fafb; display:flex; flex-direction:column; justify-content:center; gap:7px;}
        .decision-step-card.pass {border-color:#1f8d63; background:#102d25;}
        .decision-step-card.review {border-color:#a66b1c; background:#332410;}
        .decision-step-card.decision-final {min-height:132px;}
        .decision-step-label {font-size:0.78rem; color:#b5c0cf;}
        .decision-step-card strong {font-size:1rem; line-height:1.35;}
        .trace-logic-grid {display:grid; grid-template-columns:repeat(auto-fit, minmax(132px, 1fr)); gap:8px; margin:8px 0 10px;}
        .trace-logic-grid.detailed {grid-template-columns:repeat(4, minmax(0, 1fr)); align-items:stretch;}
        .trace-logic-card {border:1px solid #4b5563; border-radius:7px; padding:8px 10px; background:#1b2029; display:flex; flex-direction:column; gap:3px; min-height:72px;}
        .trace-logic-card.pass {border-color:#1f8d63; background:#102d25;}
        .trace-logic-card.review {border-color:#a66b1c; background:#332410;}
        .trace-logic-card span {font-size:0.82rem; color:#aeb8c7;}
        .trace-logic-card strong {font-size:1.02rem; color:#f9fafb; line-height:1.15;}
        .trace-logic-card small {font-size:0.74rem; color:#aeb8c7; line-height:1.25;}
        .trace-logic-checks {display:grid; gap:6px; margin-top:6px; padding-top:7px; border-top:1px solid rgba(174,184,199,0.24);}
        .trace-logic-check {display:grid; grid-template-columns:18px minmax(0, 1fr); gap:5px; align-items:start;}
        .trace-logic-check > .trace-logic-check-icon {font-size:0.72rem; line-height:1.35; color:#f9fafb;}
        .trace-logic-check-copy {display:flex; flex-direction:column; gap:1px; min-width:0;}
        .trace-logic-check-copy > span {font-size:0.73rem; line-height:1.25; color:#e5e7eb;}
        .trace-logic-check-copy > small {font-size:0.67rem; line-height:1.3; color:#aeb8c7; overflow-wrap:anywhere;}
        .trace-level-grid {display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:8px; margin:2px 0 12px;}
        .trace-level-card {border:1px solid #3c4655; border-radius:7px; padding:10px 12px; background:#111821; display:flex; flex-direction:column; gap:4px;}
        .trace-level-card span {font-size:0.82rem; color:#aeb8c7;}
        .trace-level-card strong {font-size:1rem; color:#f9fafb;}
        .trace-level-card small {font-size:0.76rem; color:#b8c1ce; line-height:1.4;}
        .trace-rule-table-wrap {width:100%; overflow-x:auto;}
        .trace-rule-table {width:100%; border-collapse:separate; border-spacing:0; table-layout:fixed; border:1px solid #2f3641; border-radius:8px; overflow:hidden;}
        .trace-rule-table th, .trace-rule-table td {padding:9px 10px; border-right:1px solid #2f3641; border-bottom:1px solid #2f3641; vertical-align:top; white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; line-height:1.45;}
        .trace-rule-table th {background:#1d2129; color:#aeb4bf; font-weight:700; text-align:left;}
        .trace-rule-table td {background:#0f1218; color:#f3f4f6;}
        .trace-rule-table tr.trace-subrow td {background:#111821; color:#d7dde7; font-size:0.92rem; border-top:0;}
        .trace-rule-table tr.trace-subrow .trace-rule-col-rule {color:#9aa4b2; padding-left:1.65rem; position:relative;}
        .trace-rule-table tr.trace-subrow .trace-rule-col-rule::before {content:""; position:absolute; left:0.78rem; top:0.92rem; bottom:0.92rem; width:2px; background:#3b4656; border-radius:999px;}
        .trace-rule-table tr:last-child td {border-bottom:0;}
        .trace-rule-table th:last-child, .trace-rule-table td:last-child {border-right:0;}
        .trace-rule-col-rule {width:18%;}
        .trace-rule-col-status {width:12%; white-space:nowrap;}
        .trace-rule-col-actual {width:9%;}
        .trace-rule-col-score {width:14%; text-align:center;}
        .trace-rule-col-threshold {width:9%;}
        .trace-rule-col-comparison {width:12%;}
        .trace-rule-col-detail {width:40%;}
        .trace-rule-table:has(.trace-rule-col-score) .trace-rule-col-rule {width:20%;}
        .trace-rule-table:has(.trace-rule-col-score) .trace-rule-col-actual {width:24%;}
        .trace-rule-table:has(.trace-rule-col-score) .trace-rule-col-status {width:14%; text-align:center; white-space:nowrap;}
        .trace-rule-table:has(.trace-rule-col-score) .trace-rule-col-detail {width:28%;}
        .trace-rule-table.compact-gate-table .trace-rule-col-rule {width:46%;}
        .trace-rule-table.compact-gate-table .trace-rule-col-actual {width:46%;}
        .trace-rule-table.compact-gate-table .trace-rule-col-status {width:8%; text-align:center; white-space:nowrap;}
        .trace-rule-table.compact-gate-table.narrow-status .trace-rule-col-rule {width:46%;}
        .trace-rule-table.compact-gate-table.narrow-status .trace-rule-col-actual {width:46%;}
        .trace-rule-table.compact-gate-table.narrow-status .trace-rule-col-status {width:8%; text-align:center; white-space:nowrap;}
        .trace-warning-notes {margin:10px 0 24px; color:#d0d6df; font-size:0.9rem; line-height:1.45;}
        .trace-warning-note {margin:3px 0;}
        @media (max-width: 1180px) {
            .trace-logic-grid.detailed {grid-template-columns:repeat(2, minmax(0, 1fr));}
        }
        @media (max-width: 700px) {
            .trace-logic-grid.detailed {grid-template-columns:1fr;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    for index, step in enumerate(steps, start=1):
        _render_decision_step(step, index)


def _with_rebuilt_breakout_validity(row_data):
    """Backfill breakout state from indicator series saved in older snapshots."""

    closes = row_data.get("close")
    pivots = row_data.get("pivot_high_20")
    volumes = row_data.get("volume_ratio")
    if not all(isinstance(values, list) and values for values in (closes, pivots, volumes)):
        return row_data

    confirmed = []
    for index, (close, pivot, volume_ratio) in enumerate(zip(closes, pivots, volumes)):
        try:
            if (
                float(close) >= float(pivot) * 1.005
                and float(volume_ratio) >= 1.10
            ):
                confirmed.append(index)
        except (TypeError, ValueError):
            continue

    evidence = dict(row_data.get("setup_evidence") or {})
    failure_flags = dict(evidence.get("failure_flags") or {})
    failure_details = dict(evidence.get("failure_flag_details") or {})
    validity_flags = dict(evidence.get("validity_flags") or {})
    validity_details = dict(evidence.get("validity_flag_details") or {})
    pivot_detail = dict(
        validity_details.get("holds_breakout_pivot")
        or failure_details.get("failed_breakout")
        or {}
    )
    support_detail = dict(
        validity_details.get("holds_short_term_support")
        or failure_details.get("breaks_support")
        or {}
    )

    breakout_context = bool(confirmed)
    breakout_index = confirmed[-1] if confirmed else None
    latest_close = _latest_number(row_data, "close")
    breakout_pivot = (
        float(pivots[breakout_index])
        if breakout_index is not None
        else _latest_number(row_data, "pivot_high_20")
    )
    pivot_window_start = None
    pivot_window_end = None
    pivot_source_date = None
    dates = row_data.get("date")
    highs = row_data.get("high")
    if (
        breakout_index is not None
        and isinstance(dates, list)
        and isinstance(highs, list)
    ):
        window_start = max(0, breakout_index - 20)
        window_end = min(breakout_index, len(dates), len(highs))
        window = []
        for position in range(window_start, window_end):
            try:
                value = float(highs[position])
            except (TypeError, ValueError):
                continue
            if pd.notna(value):
                window.append((position, value))
        if window:
            pivot_window_start = str(pd.Timestamp(dates[window[0][0]]).date())
            pivot_window_end = str(pd.Timestamp(dates[window[-1][0]]).date())
            pivot_source_position = max(window, key=lambda item: item[1])[0]
            pivot_source_date = str(pd.Timestamp(dates[pivot_source_position]).date())
    latest_below_pivot = bool(
        breakout_context
        and latest_close is not None
        and breakout_pivot is not None
        and latest_close < breakout_pivot
    )
    sma20 = _latest_number(row_data, "SMA_20")
    sma50 = _latest_number(row_data, "SMA_50")
    below_sma20 = bool(latest_close is not None and sma20 is not None and latest_close < sma20)
    below_sma50 = bool(latest_close is not None and sma50 is not None and latest_close < sma50)
    breaks_support = below_sma20 or below_sma50

    pivot_detail.update({
        "breakout_context": breakout_context,
        "breakout_close": float(closes[breakout_index]) if breakout_index is not None else None,
        "latest_close": latest_close,
        "pivot_high_20": breakout_pivot,
        "pivot_window_start": pivot_window_start,
        "pivot_window_end": pivot_window_end,
        "pivot_source_date": pivot_source_date,
        "latest_below_pivot": latest_below_pivot,
        "confirmed_prior_breakout": breakout_context,
    })
    support_detail.update({
        "breakout_context": breakout_context,
        "latest_close": latest_close,
        "sma20": sma20,
        "sma50": sma50,
        "below_sma20": below_sma20,
        "below_sma50": below_sma50,
    })
    validity_flags.update({
        "holds_breakout_pivot": not latest_below_pivot,
        "holds_short_term_support": not breaks_support,
    })
    validity_details.update({
        "holds_breakout_pivot": pivot_detail,
        "holds_short_term_support": support_detail,
    })
    failure_flags.pop("failed_breakout", None)
    failure_flags.pop("breaks_support", None)
    failure_details.pop("failed_breakout", None)
    failure_details.pop("breaks_support", None)
    evidence.update({
        "failure_flags": failure_flags,
        "failure_flag_details": failure_details,
        "validity_flags": validity_flags,
        "validity_flag_details": validity_details,
        "breakout_context": breakout_context,
        "breakout_invalid": bool(
            breakout_context and (latest_below_pivot or breaks_support)
        ),
    })
    return {**row_data, "setup_evidence": evidence}


def _demote_meta_phase_without_zone(row_data):
    """Prevent stale snapshots from presenting a pullback without META overlap."""
    phase = str(row_data.get("setup_phase") or "")
    evidence = row_data.get("setup_evidence") or {}
    chart_evidence = evidence.get("chart_evidence") or {}
    if phase not in {"pullback_forming", "pullback_entry"}:
        return row_data
    if "chart_evidence" not in evidence or chart_evidence.get("selected_meta_zone"):
        return row_data
    updated_evidence = dict(evidence)
    candidate_flags = dict(updated_evidence.get("candidate_flags") or {})
    candidate_flags["pullback_forming"] = False
    candidate_flags["pullback_entry"] = False
    updated_evidence["candidate_flags"] = candidate_flags
    return {
        **row_data,
        "setup_phase": "unclear_structure",
        "setup_status": "watchlist",
        "status": "watchlist",
        "setup_caption": "未找到真正 META 匯聚區，回調分類需重新確認。",
        "setup_evidence": updated_evidence,
    }


def _with_rebuilt_score_evidence(row_data):
    """Refresh deterministic presentation fields from locally cached daily OHLCV."""

    symbol = str(row_data.get("symbol", "")).upper().strip()
    cached = load_ohlcv_cache(symbol) if symbol else None
    cached_ohlcv = cached.get("data") if isinstance(cached, dict) else None
    saved_ohlcv = row_ohlcv_frame(row_data)
    cached_last_date = pd.Timestamp.min
    if isinstance(cached_ohlcv, pd.DataFrame) and not cached_ohlcv.empty:
        cached_last_date = comparable_timestamp(cached_ohlcv.index[-1])
    ohlcv = saved_ohlcv if row_series_last_date(row_data) > cached_last_date else cached_ohlcv
    if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
        return _demote_meta_phase_without_zone(_with_rebuilt_breakout_validity(row_data))

    try:
        daily = completed_daily_ohlcv(ohlcv, symbol, row_data.get("exchange"))
        if daily.empty:
            return _demote_meta_phase_without_zone(row_data)
        rebuilt = run_tech_analysis(daily, market_metrics=row_data)
    except Exception:
        rebuilt_entry = rebuild_entry_plan_from_snapshot(row_data)
        return _demote_meta_phase_without_zone(_with_rebuilt_breakout_validity(rebuilt_entry))

    # Analysis fields are derived solely from the local completed daily series;
    # preserve snapshot metadata such as the saved name, dates, and chart flags.
    merged = {**row_data, **rebuilt}
    metadata_fields = (
        "name", "exchange", "financial_currency", "market_cap", "earnings_date",
        "is_earnings_estimate", "sector_name", "industry_name", "avgvol3m",
    )
    for field in metadata_fields:
        if _has_display_value(row_data.get(field), field=field, symbol=symbol):
            merged[field] = row_data.get(field)
    merged = rebuild_entry_plan_from_snapshot(merged)
    return _demote_meta_phase_without_zone(_with_rebuilt_breakout_validity(merged))


def _render_explanation(row):
    row_data = _with_rebuilt_score_evidence(row.to_dict())
    # Older saved snapshots stored these fields as scalar values. Rebuild the
    # structured display payload so detail pages remain backward compatible.
    score_details = score_breakdown(row_data)
    if not isinstance(score_details, dict):
        score_details = {}
    settings = ScreeningSettings.from_mapping(row_data.get("settings"))
    st.subheader("詳細分析")
    left, right = st.columns(2)
    left.metric("分類", setup_phase_display_text(row_data.get("setup_phase")))
    right.metric(score_details.get("display_label", score_details.get("label", row_data.get("score_label", "評分"))), score_details.get("score_display", str(int(score_details.get("score", row_data.get("mvp_score", 0)) or 0))))

    st.markdown("**分類決策流程**")

    # Rebuild rules from the current deterministic engine. Saved snapshots from
    # earlier versions contained stale English/Beta rows and incomplete detail.
    rules = pd.DataFrame([
        *build_rule_checklist(row_data, settings),
        *build_score_checklist(row_data),
    ])
    if rules.empty:
        st.info("此舊資料未保存可追溯規則；請重新分析股票後再查看。")
        return
    _render_decision_flow(row_data, settings, rules)


symbol = st.query_params.get("symbol", "").upper()
if not symbol:
    st.warning("缺少股票代號。")
    st.stop()

components.html(
    f"""
    <script>
    const title = {repr(symbol + ' | Momentum Trading')};
    const parentWindow = window.parent;
    parentWindow.__momentumPageTitle = title;
    const syncTitle = () => {{
        if (parentWindow.__momentumPageTitle !== title) {{
            observer.disconnect();
            window.clearInterval(timer);
            return;
        }}
        if (parentWindow.document.title !== title) parentWindow.document.title = title;
    }};
    const observer = new MutationObserver(syncTitle);
    observer.observe(parentWindow.document.head, {{childList: true, subtree: true, characterData: true}});
    const timer = window.setInterval(syncTitle, 500);
    syncTitle();
    </script>
    """,
    height=0,
)

stock_rows = pd.concat(
    [_rows_from_state("bookmark_stocks"), _rows_from_state("analyzed_stocks")],
    ignore_index=True,
)
matched_rows = stock_rows[stock_rows["symbol"] == symbol] if "symbol" in stock_rows.columns else pd.DataFrame()
row = _freshest_row(symbol, matched_rows)

if row is None:
    with st.spinner(f"正在分析 {symbol}..."):
        try:
            row = pd.Series(analyze_single_stock(symbol))
            save_stock_cache(symbol, row.to_dict(), source="view_stock")
        except Exception as error:
            st.warning(f"找不到 {symbol} 的已儲存資料，且即時分析失敗：{error}")
            st.stop()

# Older saved snapshots may contain stale display strings and incomplete score
# evidence. Rebuild all detail-page presentation from local completed daily data.
row = pd.Series(_with_rebuilt_score_evidence(row.to_dict()))
trendline_metadata = render_stock_detail(row, symbol)
if trendline_metadata is not None and row.get("trendline_analysis") != trendline_metadata:
    row_data = row.to_dict()
    row_data["trendline_analysis"] = trendline_metadata
    save_stock_cache(symbol, row_data, source="trendline_analysis")

_render_explanation(row)
