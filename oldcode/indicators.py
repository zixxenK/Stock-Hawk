"""
Technical indicators for momentum analysis.

All functions accept a pandas DataFrame with at minimum a 'Close' column
(and optionally 'High', 'Low', 'Volume').  They return scalar values or
small dicts so the scoring layer can stay clean.

Momentum literature used:
- Jegadeesh & Titman (1993): 12-1 month momentum
- George & Hwang (2004): 52-week high as anchor
- Moskowitz & Grinblatt (1999): industry momentum
- Carhart (1997): 4-factor model — momentum factor
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─── Price Momentum ───────────────────────────────────────────────────────────

def momentum_12_1(close: pd.Series) -> float | None:
    """
    Classic 12-minus-1 month price momentum (Jegadeesh & Titman).

    Returns the percentage return from 252 trading days ago to 21 trading
    days ago, skipping the most recent month to avoid short-term reversal.
    Returns None if insufficient history.
    """
    if len(close) < 273:
        return None
    p_start = close.iloc[-252]
    p_end = close.iloc[-21]
    if p_start <= 0:
        return None
    return (p_end / p_start) - 1.0


def momentum_3m(close: pd.Series) -> float | None:
    """3-month (63 trading day) price momentum."""
    if len(close) < 63:
        return None
    p_start = close.iloc[-63]
    if p_start <= 0:
        return None
    return (close.iloc[-1] / p_start) - 1.0


def momentum_6m(close: pd.Series) -> float | None:
    """6-month (126 trading day) price momentum."""
    if len(close) < 126:
        return None
    p_start = close.iloc[-126]
    if p_start <= 0:
        return None
    return (close.iloc[-1] / p_start) - 1.0


def momentum_1m(close: pd.Series) -> float | None:
    """1-month (21 trading day) short-term momentum (note: often mean-reverts)."""
    if len(close) < 21:
        return None
    p_start = close.iloc[-21]
    if p_start <= 0:
        return None
    return (close.iloc[-1] / p_start) - 1.0


def sharpe_momentum(close: pd.Series, window: int = 252) -> float | None:
    """
    Volatility-adjusted momentum: annualised Sharpe ratio of daily returns
    over the lookback window.  Rewards consistent risers over volatile ones.
    """
    if len(close) < window + 1:
        return None
    rets = close.pct_change().dropna().iloc[-window:]
    if rets.std() == 0:
        return None
    return float((rets.mean() / rets.std()) * np.sqrt(252))


# ─── Relative Strength ────────────────────────────────────────────────────────

def relative_strength_vs_benchmark(
    close: pd.Series,
    bench_close: pd.Series,
    window: int = 252,
) -> float | None:
    """
    Return the stock's excess return over the benchmark over `window` days.
    Positive = outperformed SPY.
    """
    if len(close) < window or len(bench_close) < window:
        return None
    stock_ret = (close.iloc[-1] / close.iloc[-window]) - 1.0
    bench_ret = (bench_close.iloc[-1] / bench_close.iloc[-window]) - 1.0
    return stock_ret - bench_ret


def relative_strength_ratio(
    close: pd.Series,
    bench_close: pd.Series,
    window: int = 63,
) -> float | None:
    """
    Mansfield Relative Strength: ratio of stock performance to benchmark.
    > 1 means stock is outperforming.
    """
    if len(close) < window or len(bench_close) < window:
        return None
    stock_ret = close.iloc[-1] / close.iloc[-window]
    bench_ret = bench_close.iloc[-1] / bench_close.iloc[-window]
    if bench_ret == 0:
        return None
    return stock_ret / bench_ret


# ─── 52-Week High Proximity ───────────────────────────────────────────────────

def high_52w_proximity(close: pd.Series, high: pd.Series | None = None) -> float | None:
    """
    Fraction of 52-week high the current price sits at.
    1.0 = at all-time 52-week high. < 0.85 is considered weak.

    George & Hwang (2004) show proximity to the 52-week high is the single
    best predictor of subsequent 6-12 month returns.
    """
    series = high if high is not None else close
    if len(series) < 252:
        return None
    high_52 = series.iloc[-252:].max()
    if high_52 <= 0:
        return None
    return float(close.iloc[-1] / high_52)


def low_52w_proximity(close: pd.Series, low: pd.Series | None = None) -> float | None:
    """How far the current price is above the 52-week low (% above)."""
    series = low if low is not None else close
    if len(series) < 252:
        return None
    low_52 = series.iloc[-252:].min()
    if low_52 <= 0:
        return None
    return float((close.iloc[-1] / low_52) - 1.0)


# ─── Trend / Moving Averages ─────────────────────────────────────────────────

def sma(close: pd.Series, window: int) -> float | None:
    """Simple moving average of the last `window` prices."""
    if len(close) < window:
        return None
    return float(close.iloc[-window:].mean())


def price_vs_sma(close: pd.Series, window: int) -> float | None:
    """How many % above (+) or below (-) the SMA the current price is."""
    s = sma(close, window)
    if s is None or s == 0:
        return None
    return float((close.iloc[-1] / s) - 1.0)


def golden_cross(close: pd.Series) -> bool | None:
    """
    True if the 50-day SMA is above the 200-day SMA (golden cross).
    The most widely tracked trend filter on institutional desks.
    """
    if len(close) < 200:
        return None
    s50 = close.iloc[-50:].mean()
    s200 = close.iloc[-200:].mean()
    return bool(s50 > s200)


def ema(close: pd.Series, window: int) -> float | None:
    """Exponential moving average."""
    if len(close) < window:
        return None
    return float(close.ewm(span=window, adjust=False).mean().iloc[-1])


# ─── Trend Strength (ADX) ─────────────────────────────────────────────────────

def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> float | None:
    """
    Average Directional Index.
    > 25: strong trend.  > 50: very strong.  < 20: ranging / no trend.
    """
    n = period + 1
    if len(close) < n * 3:
        return None

    h = high.values
    l = low.values
    c = close.values

    # True Range
    tr = np.maximum(h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))

    # Directional movements
    dm_plus = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]),
                       np.maximum(h[1:] - h[:-1], 0), 0)
    dm_minus = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]),
                        np.maximum(l[:-1] - l[1:], 0), 0)

    # Wilder smoothing
    def wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
        out = np.zeros(len(arr))
        out[p - 1] = arr[:p].sum()
        for i in range(p, len(arr)):
            out[i] = out[i - 1] - out[i - 1] / p + arr[i]
        return out

    atr_s = wilder_smooth(tr, period)
    dm_plus_s = wilder_smooth(dm_plus, period)
    dm_minus_s = wilder_smooth(dm_minus, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus = np.where(atr_s != 0, 100 * dm_plus_s / atr_s, 0)
        di_minus = np.where(atr_s != 0, 100 * dm_minus_s / atr_s, 0)
        dx = np.where(
            (di_plus + di_minus) != 0,
            100 * np.abs(di_plus - di_minus) / (di_plus + di_minus),
            0,
        )

    if len(dx) < period:
        return None

    adx_series = np.zeros(len(dx))
    adx_series[period - 1] = dx[:period].mean()
    for i in range(period, len(dx)):
        adx_series[i] = (adx_series[i - 1] * (period - 1) + dx[i]) / period

    return float(adx_series[-1])


# ─── Volume ───────────────────────────────────────────────────────────────────

def volume_trend(volume: pd.Series, short: int = 20, long: int = 60) -> float | None:
    """
    Ratio of short-term average volume to long-term average volume.
    > 1 means volume is expanding — confirms momentum.
    """
    if len(volume) < long:
        return None
    avg_short = volume.iloc[-short:].mean()
    avg_long = volume.iloc[-long:].mean()
    if avg_long == 0:
        return None
    return float(avg_short / avg_long)


def on_balance_volume_trend(close: pd.Series, volume: pd.Series, window: int = 20) -> float | None:
    """
    Slope of OBV over the last `window` bars, normalised by mean OBV.
    Positive = buying pressure accumulating.
    """
    if len(close) < window + 1:
        return None
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    obv_recent = obv.iloc[-window:]
    mean_obv = obv_recent.abs().mean()
    if mean_obv == 0:
        return None
    slope = (obv_recent.iloc[-1] - obv_recent.iloc[0]) / window
    return float(slope / mean_obv)


# ─── RSI ─────────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> float | None:
    """
    Relative Strength Index.
    For momentum entries: ideally 50-70 (momentum without being overbought).
    """
    if len(close) < period + 1:
        return None
    delta = close.diff().dropna()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.ewm(com=period - 1, adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(com=period - 1, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


# ─── MACD ─────────────────────────────────────────────────────────────────────

def macd_signal(close: pd.Series) -> dict | None:
    """
    MACD (12, 26, 9).
    Returns dict with 'macd', 'signal', 'histogram', 'bullish'.
    bullish = True when histogram is positive AND MACD > signal.
    """
    if len(close) < 35:
        return None
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    return {
        "macd": float(macd_line.iloc[-1]),
        "signal": float(signal_line.iloc[-1]),
        "histogram": float(hist.iloc[-1]),
        "bullish": bool(hist.iloc[-1] > 0 and macd_line.iloc[-1] > signal_line.iloc[-1]),
    }


# ─── Risk / ATR ──────────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float | None:
    """Average True Range — used for stop-loss placement."""
    if len(close) < period + 1:
        return None
    h = high.values
    l = low.values
    c = close.values
    tr = np.maximum(h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    # Wilder smoothing
    atr_val = tr[:period].mean()
    for t in tr[period:]:
        atr_val = (atr_val * (period - 1) + t) / period
    return float(atr_val)


def historical_volatility(close: pd.Series, window: int = 30) -> float | None:
    """Annualised historical volatility from log returns."""
    if len(close) < window + 1:
        return None
    log_rets = np.log(close / close.shift(1)).dropna().iloc[-window:]
    return float(log_rets.std() * np.sqrt(252))
