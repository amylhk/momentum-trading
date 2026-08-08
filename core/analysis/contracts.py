from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any


RULESET_VERSION = "momentum_mvp_v3"
EARNINGS_EVENT_WINDOW_DAYS = 7


@dataclass(frozen=True)
class ScreeningSettings:
    """Versioned, user-visible inputs to the deterministic screener."""

    market_cap_min: float = 2_000_000_000
    price_min: float = 10.0
    avg_volume_3m_min: float = 500_000
    beta_min: float = 1.0
    dollar_volume_min: float = 30_000_000
    max_distance_52w_high: float = 0.25
    require_price_above_ma_stack: bool = True
    require_sma10_above_sma20: bool = True
    require_ma200_rising: bool = True
    require_vs_market_1m: bool = True
    require_vs_market_3m: bool = True
    require_vs_market_6m: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "ScreeningSettings":
        # Older cached rows may contain NaN instead of a settings mapping.
        # Treat those rows as default-profile data rather than raising while
        # rebuilding the current deterministic checklist.
        if not isinstance(values, dict):
            values = {}
        allowed = {field: values[field] for field in cls.__dataclass_fields__ if field in values}
        return cls(**allowed)


@dataclass(frozen=True)
class RuleResult:
    group: str
    rule: str
    status: str
    actual: Any
    threshold: Any
    comparison: str
    source: str = "Yahoo Finance daily data"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisSnapshot:
    symbol: str
    classification: str
    setup_status: str
    score: int
    score_label: str
    data_date: str
    settings_fingerprint: str
    ruleset_version: str
    summary: str
    rules: list[dict[str, Any]]
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
