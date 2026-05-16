"""
Feature Vectorizer — 18-dimensional trade setup embedding.

Converts a mix of technical indicators, insider signals, and sentiment
into a fixed-length numpy array suitable for cosine similarity search.

Dimension map:
  [0]  mom_12_1          Jegadeesh-Titman 11-month return
  [1]  mom_6m            6-month return
  [2]  mom_3m            3-month return
  [3]  rs_vs_spy         Excess return over S&P 500 (12m)
  [4]  high_52w_pct      Current price / 52-week high
  [5]  adx_norm          ADX / 50
  [6]  rsi_norm          RSI / 100
  [7]  vol_trend_norm    Volume ratio (clipped, normalised)
  [8]  golden_cross      Binary: 50d SMA > 200d SMA
  [9]  hist_vol_norm     Annualised vol / 0.8
  [10] congress_buys     Log-normalised count (last 30d)
  [11] insider_buys      Log-normalised count + seniority-weighted
  [12] options_bullish   Log-normalised bullish flow count
  [13] suspicious_vol    Binary: volume/insider correlation flag
  [14] sector_idx        GICS sector ordinal / 11
  [15] earnings_prox     Days to earnings / 30
  [16] market_regime     0=bear, 0.5=mixed, 1=bull
  [17] sentiment_score   NLP sentiment  0=negative, 0.5=neutral, 1=positive
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

VECTOR_DIM = 18   # Upgraded from 17 with NLP sentiment dimension

SECTOR_MAP: dict[str, int] = {
    "information technology": 0, "technology":  0,
    "health care":            1, "healthcare":  1,
    "financials":             2,
    "consumer discretionary": 3,
    "communication services": 4,
    "industrials":            5,
    "consumer staples":       6,
    "energy":                 7,
    "utilities":              8,
    "real estate":            9,
    "materials":              10,
}

FEATURE_NAMES: list[str] = [
    "mom_12_1", "mom_6m", "mom_3m", "rs_vs_spy",
    "high_52w_pct", "adx_norm", "rsi_norm", "vol_trend_norm",
    "golden_cross", "hist_vol_norm",
    "congress_buys", "insider_buys", "options_bullish", "suspicious_vol",
    "sector_idx", "earnings_prox", "market_regime", "sentiment_score",
]
assert len(FEATURE_NAMES) == VECTOR_DIM


@dataclass
class VectorInput:
    """
    All raw inputs needed to build one feature vector.
    Every field is optional — missing values get a sensible neutral default.
    """
    # ── Technical ──────────────────────────────────────────────────────
    mom_12_1:       float | None = None
    mom_6m:         float | None = None
    mom_3m:         float | None = None
    rs_vs_spy:      float | None = None
    high_52w_pct:   float | None = None
    adx:            float | None = None
    rsi:            float | None = None
    vol_trend:      float | None = None
    golden_cross:   bool  | None = None
    hist_vol:       float | None = None
    # ── Intelligence signals ────────────────────────────────────────────
    congress_buys:    int   = 0
    insider_buys:     int   = 0
    insider_weighted: float = 0.0   # seniority-weighted buy score
    options_bullish:  int   = 0
    suspicious_vol:   bool  = False
    # ── Context ─────────────────────────────────────────────────────────
    sector:            str  = ""
    days_to_earnings:  int | None = None
    market_regime:     str  = "mixed"
    sentiment_score:   float | None = None   # -1 to 1 → normalised to 0-1


def _safe(v: Any, fallback: float = 0.5) -> float:
    if v is None:
        return fallback
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _norm_mom(v: float | None) -> float:
    """Return ∈ (-0.50 … +1.50) → [0, 1].  None → 0.5 (neutral)."""
    return float(np.clip((_safe(v, 0.0) + 0.5) / 2.0, 0.0, 1.0))


def _log_norm(v: float | int | None, scale: float = 5.0) -> float:
    """Log-normalise a count/score.  5 events → ~0.70."""
    if not v:
        return 0.0
    return float(np.clip(math.log1p(abs(float(v))) / math.log1p(scale), 0.0, 1.0))


def _norm_sentiment(v: float | None) -> float:
    """Map [-1, 1] → [0, 1].  None → 0.5 (neutral)."""
    if v is None:
        return 0.5
    return float(np.clip((_safe(v, 0.0) + 1.0) / 2.0, 0.0, 1.0))


def build_vector(inp: VectorInput) -> np.ndarray:
    """
    Convert a VectorInput into a normalised 18-dim float32 numpy array.

    Every dimension is clipped to [0, 1].
    Missing values resolve to semantically neutral positions (usually 0.5).
    """
    # Insider buy signal: combine count + seniority weighting
    insider_signal = max(
        _log_norm(inp.insider_buys),
        _log_norm(inp.insider_weighted, scale=3.0),
    )

    regime_map = {
        "bull": 1.0, "buying": 1.0,
        "mixed": 0.5, "unknown": 0.5,
        "bear": 0.0, "profit_taking": 0.3,
    }
    regime_val = regime_map.get((inp.market_regime or "mixed").lower(), 0.5)

    sector_idx = SECTOR_MAP.get((inp.sector or "").lower().strip(), 5) / 10.0

    earn_prox = (
        float(np.clip(_safe(inp.days_to_earnings, 15) / 30.0, 0.0, 1.0))
        if inp.days_to_earnings is not None
        else 0.5
    )

    v = np.array([
        _norm_mom(inp.mom_12_1),
        _norm_mom(inp.mom_6m),
        _norm_mom(inp.mom_3m),
        _norm_mom(inp.rs_vs_spy),
        float(np.clip(_safe(inp.high_52w_pct, 0.9), 0.0, 1.0)),
        float(np.clip(_safe(inp.adx, 25.0) / 50.0, 0.0, 1.0)),
        float(_safe(inp.rsi, 50.0) / 100.0),
        float(np.clip((_safe(inp.vol_trend, 1.0) - 0.5) / 1.5, 0.0, 1.0)),
        1.0 if inp.golden_cross is True else 0.0,
        float(np.clip(_safe(inp.hist_vol, 0.3) / 0.8, 0.0, 1.0)),
        _log_norm(inp.congress_buys),
        insider_signal,
        _log_norm(inp.options_bullish),
        1.0 if inp.suspicious_vol else 0.0,
        sector_idx,
        earn_prox,
        regime_val,
        _norm_sentiment(inp.sentiment_score),
    ], dtype=np.float32)

    assert v.shape[0] == VECTOR_DIM, f"Vector dim mismatch: {v.shape[0]} != {VECTOR_DIM}"
    return v


def vector_from_momentum_score(
    score: Any,
    congress_buys: int = 0,
    insider_buys: int = 0,
    insider_weighted: float = 0.0,
    options_bullish: int = 0,
    suspicious_vol: bool = False,
    sector: str = "",
    days_to_earnings: int | None = None,
    market_regime: str = "mixed",
    sentiment_score: float | None = None,
) -> np.ndarray:
    """
    Convenience wrapper: build a vector directly from a MomentumScore object.
    Accepts any object with the standard MomentumScore attribute names.
    """
    return build_vector(VectorInput(
        mom_12_1       = getattr(score, "mom_12_1",    None),
        mom_6m         = getattr(score, "mom_6m",      None),
        mom_3m         = getattr(score, "mom_3m",      None),
        rs_vs_spy      = getattr(score, "rs_vs_spy",   None),
        high_52w_pct   = getattr(score, "high_52w_pct",None),
        adx            = getattr(score, "adx",         None),
        rsi            = getattr(score, "rsi",         None),
        vol_trend      = getattr(score, "vol_trend",   None),
        golden_cross   = getattr(score, "golden_cross",None),
        hist_vol       = getattr(score, "hist_vol",    None),
        congress_buys    = congress_buys,
        insider_buys     = insider_buys,
        insider_weighted = insider_weighted,
        options_bullish  = options_bullish,
        suspicious_vol   = suspicious_vol,
        sector           = sector,
        days_to_earnings = days_to_earnings,
        market_regime    = market_regime,
        sentiment_score  = sentiment_score,
    ))


def describe_vector(v: np.ndarray) -> dict[str, float]:
    """Return a human-readable dict mapping feature names to values."""
    if v.shape[0] != VECTOR_DIM:
        return {}
    return {name: round(float(v[i]), 4) for i, name in enumerate(FEATURE_NAMES)}


def vector_diff(
    v1: np.ndarray,
    v2: np.ndarray,
    threshold: float = 0.15,
) -> dict[str, dict[str, float]]:
    """
    Compare two vectors dimension-by-dimension.
    Returns only dimensions where |v1 - v2| > threshold.

    Useful for explaining why two setups are similar/different.
    """
    diffs: dict[str, dict[str, float]] = {}
    for i, name in enumerate(FEATURE_NAMES):
        d = float(v1[i]) - float(v2[i])
        if abs(d) > threshold:
            diffs[name] = {
                "current":    round(float(v1[i]), 4),
                "historical": round(float(v2[i]), 4),
                "delta":      round(d, 4),
            }
    return diffs
