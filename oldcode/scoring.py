"""
Composite momentum scoring.

Takes raw indicator values for a ticker and produces a single
0–100 score plus a grade (A+ → F) and a human-readable breakdown.

Weight rationale (long-term momentum, weeks–months horizon):
  12-1 month momentum   25%  — the strongest academic predictor
  Relative strength     20%  — alpha over the market matters most
  52-week high prox.    15%  — George & Hwang proximity effect
  Sharpe momentum       10%  — rewards smooth, consistent up-trends
  Golden cross / trend  10%  — macro trend filter
  ADX                    8%  — confirms there *is* a trend
  Volume trend           7%  — institutional accumulation signal
  RSI zone               5%  — mild sentiment confirmation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MomentumScore:
    ticker: str
    price: float
    score: float                    # 0-100
    grade: str                      # A+, A, B+, B, C, D, F
    rank: int = 0                   # set by scanner after sorting
    # Raw signals
    mom_12_1: float | None = None
    mom_6m: float | None = None
    mom_3m: float | None = None
    rs_vs_spy: float | None = None
    rs_ratio: float | None = None
    high_52w_pct: float | None = None
    sharpe_mom: float | None = None
    golden_cross: bool | None = None
    price_vs_200: float | None = None
    adx: float | None = None
    vol_trend: float | None = None
    rsi: float | None = None
    macd_bullish: bool | None = None
    # Risk
    atr: float | None = None
    hist_vol: float | None = None
    # Factor sub-scores (0-1) for transparency
    factor_scores: dict[str, float] = field(default_factory=dict)
    # Human summary
    signals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score_mom_12_1(value: float | None) -> float:
    """0.0 → 1.0.  Best in class is ~80%+ over 11 months."""
    if value is None:
        return 0.0
    # Sigmoid centred at 20% return, saturates at 80%
    return _clamp(0.5 + value / 0.80 * 0.5)


def _score_rs_vs_spy(value: float | None) -> float:
    """Excess return over SPY.  +30% excess → 1.0."""
    if value is None:
        return 0.0
    return _clamp(0.5 + value / 0.60)


def _score_high_52w(value: float | None) -> float:
    """
    Proximity to 52-week high.
    0.97-1.0  → 1.0 (near highs — strong momentum)
    0.85-0.97 → linear
    < 0.85    → 0 (too far from high)
    """
    if value is None:
        return 0.0
    if value >= 0.97:
        return 1.0
    if value < 0.80:
        return 0.0
    return _clamp((value - 0.80) / (0.97 - 0.80))


def _score_sharpe(value: float | None) -> float:
    """Sharpe > 2.0 → 1.0.  Negative → 0."""
    if value is None:
        return 0.0
    return _clamp(value / 2.5)


def _score_trend(golden_cross: bool | None, price_vs_200: float | None) -> float:
    """Combined golden-cross + price-above-200 SMA score."""
    s = 0.0
    if golden_cross is True:
        s += 0.5
    if price_vs_200 is not None and price_vs_200 > 0:
        # Max bonus for being 20%+ above 200 SMA
        s += _clamp(price_vs_200 / 0.20) * 0.5
    return _clamp(s)


def _score_adx(value: float | None) -> float:
    """ADX > 40 → 1.0.  < 20 → 0 (no trend)."""
    if value is None:
        return 0.0
    if value < 20:
        return 0.0
    return _clamp((value - 20) / 30)


def _score_volume(value: float | None) -> float:
    """Volume ratio.  1.5× average volume → 1.0."""
    if value is None:
        return 0.5  # neutral if unknown
    return _clamp((value - 0.7) / 0.8)


def _score_rsi(value: float | None) -> float:
    """
    Ideal RSI for momentum entries: 50-70 (in the zone, not overbought).
    < 40 or > 80 → low score.
    """
    if value is None:
        return 0.5
    if 55 <= value <= 70:
        return 1.0
    if 45 <= value < 55:
        return 0.7
    if 70 < value <= 80:
        return 0.6
    if 40 <= value < 45:
        return 0.3
    if value > 80:
        return 0.2  # overbought
    return 0.1  # oversold / no momentum


# ── Weights ────────────────────────────────────────────────────────────────────
# Must sum to 1.0
WEIGHTS: dict[str, float] = {
    "mom_12_1":   0.25,
    "rs_vs_spy":  0.20,
    "high_52w":   0.15,
    "sharpe":     0.10,
    "trend":      0.10,
    "adx":        0.08,
    "volume":     0.07,
    "rsi":        0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

GRADE_THRESHOLDS = [
    (90, "A+"),
    (80, "A"),
    (70, "B+"),
    (60, "B"),
    (50, "C"),
    (40, "D"),
    (0,  "F"),
]


def _grade(score: float) -> str:
    for threshold, label in GRADE_THRESHOLDS:
        if score >= threshold:
            return label
    return "F"


def compute_score(
    ticker: str,
    price: float,
    mom_12_1: float | None,
    mom_6m: float | None,
    mom_3m: float | None,
    rs_vs_spy: float | None,
    rs_ratio: float | None,
    high_52w_pct: float | None,
    sharpe_mom: float | None,
    golden_cross: bool | None,
    price_vs_200: float | None,
    adx_val: float | None,
    vol_trend: float | None,
    rsi_val: float | None,
    macd_bullish: bool | None,
    atr_val: float | None,
    hist_vol: float | None,
) -> MomentumScore:
    """
    Compute a composite momentum score (0–100) for one ticker.

    Returns a fully populated MomentumScore dataclass.
    """
    factors = {
        "mom_12_1": _score_mom_12_1(mom_12_1),
        "rs_vs_spy": _score_rs_vs_spy(rs_vs_spy),
        "high_52w":  _score_high_52w(high_52w_pct),
        "sharpe":    _score_sharpe(sharpe_mom),
        "trend":     _score_trend(golden_cross, price_vs_200),
        "adx":       _score_adx(adx_val),
        "volume":    _score_volume(vol_trend),
        "rsi":       _score_rsi(rsi_val),
    }

    weighted = sum(factors[k] * WEIGHTS[k] for k in WEIGHTS)
    raw_score = round(weighted * 100, 2)

    # MACD confirmation: tiny bonus/penalty
    if macd_bullish is True:
        raw_score = min(100, raw_score + 2.0)
    elif macd_bullish is False:
        raw_score = max(0, raw_score - 2.0)

    grade = _grade(raw_score)

    # ── Build signal narrative ──────────────────────────────────────────
    signals: list[str] = []
    warnings: list[str] = []

    if mom_12_1 is not None:
        pct = f"{mom_12_1:.0%}"
        if mom_12_1 > 0.30:
            signals.append(f"Strong 12-1m momentum: {pct}")
        elif mom_12_1 > 0.10:
            signals.append(f"Positive 12-1m momentum: {pct}")
        else:
            warnings.append(f"Weak 12-1m momentum: {pct}")

    if rs_vs_spy is not None:
        if rs_vs_spy > 0.15:
            signals.append(f"Outperforming SPY by {rs_vs_spy:.0%}")
        elif rs_vs_spy < -0.10:
            warnings.append(f"Underperforming SPY by {abs(rs_vs_spy):.0%}")

    if high_52w_pct is not None:
        if high_52w_pct >= 0.97:
            signals.append(f"Near 52-week high ({high_52w_pct:.1%})")
        elif high_52w_pct < 0.85:
            warnings.append(f"Far from 52-week high ({high_52w_pct:.1%})")

    if golden_cross is True:
        signals.append("Golden cross (50 SMA > 200 SMA)")
    elif golden_cross is False:
        warnings.append("Death cross (50 SMA < 200 SMA)")

    if adx_val is not None:
        if adx_val >= 35:
            signals.append(f"Strong trend (ADX {adx_val:.0f})")
        elif adx_val < 20:
            warnings.append(f"Weak trend (ADX {adx_val:.0f})")

    if rsi_val is not None:
        if rsi_val > 80:
            warnings.append(f"RSI overbought ({rsi_val:.0f})")
        elif rsi_val < 40:
            warnings.append(f"RSI oversold ({rsi_val:.0f})")

    if vol_trend is not None and vol_trend > 1.3:
        signals.append(f"Volume expanding ({vol_trend:.1f}× avg)")

    if macd_bullish is True:
        signals.append("MACD bullish crossover")

    if hist_vol is not None and hist_vol > 0.60:
        warnings.append(f"High volatility ({hist_vol:.0%} annualised)")

    return MomentumScore(
        ticker=ticker,
        price=price,
        score=raw_score,
        grade=grade,
        mom_12_1=mom_12_1,
        mom_6m=mom_6m,
        mom_3m=mom_3m,
        rs_vs_spy=rs_vs_spy,
        rs_ratio=rs_ratio,
        high_52w_pct=high_52w_pct,
        sharpe_mom=sharpe_mom,
        golden_cross=golden_cross,
        price_vs_200=price_vs_200,
        adx=adx_val,
        vol_trend=vol_trend,
        rsi=rsi_val,
        macd_bullish=macd_bullish,
        atr=atr_val,
        hist_vol=hist_vol,
        factor_scores=factors,
        signals=signals,
        warnings=warnings,
    )
