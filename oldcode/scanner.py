"""
Core scan loop.

Pulls the stock universe, fetches price history, computes all indicators,
scores and ranks every ticker, and returns a sorted list of MomentumScore
objects ready for display or export.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from .data import fetch_batch, fetch_benchmark
from .indicators import (
    adx,
    atr,
    golden_cross,
    high_52w_proximity,
    historical_volatility,
    macd_signal,
    momentum_12_1,
    momentum_3m,
    momentum_6m,
    on_balance_volume_trend,
    price_vs_sma,
    relative_strength_ratio,
    relative_strength_vs_benchmark,
    rsi,
    sharpe_momentum,
    volume_trend,
)
from .scoring import MomentumScore, compute_score
from .universe import get_universe

logger = logging.getLogger(__name__)


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series | None:
    """Return a column series or None if it doesn't exist / is all-NaN."""
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return s if len(s) > 0 else None


def _analyze_ticker(
    ticker: str,
    df: pd.DataFrame,
    bench_close: pd.Series,
) -> MomentumScore | None:
    """
    Run all indicators and produce a MomentumScore for one ticker.
    Returns None if there's insufficient data.
    """
    try:
        close_raw = _safe_col(df, "Close")
        if close_raw is None or len(close_raw) < 60:
            return None

        close = close_raw
        high = _safe_col(df, "High")
        low = _safe_col(df, "Low")
        volume = _safe_col(df, "Volume")

        price = float(close.iloc[-1])
        if price <= 0:
            return None

        # Align benchmark to the same dates
        bench_aligned = bench_close.reindex(close.index, method="ffill").dropna()

        return compute_score(
            ticker=ticker,
            price=price,
            mom_12_1=momentum_12_1(close),
            mom_6m=momentum_6m(close),
            mom_3m=momentum_3m(close),
            rs_vs_spy=relative_strength_vs_benchmark(close, bench_aligned),
            rs_ratio=relative_strength_ratio(close, bench_aligned),
            high_52w_pct=high_52w_proximity(close, high),
            sharpe_mom=sharpe_momentum(close),
            golden_cross=golden_cross(close),
            price_vs_200=price_vs_sma(close, 200),
            adx_val=adx(high, low, close) if high is not None and low is not None else None,
            vol_trend=volume_trend(volume) if volume is not None else None,
            rsi_val=rsi(close),
            macd_bullish=(macd_signal(close) or {}).get("bullish"),
            atr_val=atr(high, low, close) if high is not None and low is not None else None,
            hist_vol=historical_volatility(close),
        )
    except Exception as exc:
        logger.debug("Analysis failed for %s: %s", ticker, exc)
        return None


def run_scan(
    tickers: list[str] | None = None,
    top_n: int = 30,
    min_score: float = 50.0,
    min_price: float = 5.0,
    max_price: float | None = None,
    benchmark: str = "SPY",
    show_progress: bool = True,
    max_workers: int = 8,
) -> list[MomentumScore]:
    """
    Run the full momentum scan.

    Args:
        tickers: List of tickers to scan.  None = full S&P 500 + Nasdaq 100.
        top_n: Return only the top N results.
        min_score: Minimum composite score to include.
        min_price: Skip stocks below this price (avoids penny stocks).
        max_price: Optional upper price filter.
        benchmark: Benchmark ticker for relative-strength calculation.
        show_progress: Show tqdm progress bars.
        max_workers: Threads for concurrent indicator calculation.

    Returns:
        List of MomentumScore objects, sorted descending by score.
    """
    if tickers is None:
        logger.info("Fetching stock universe...")
        tickers = get_universe()
        logger.info("Universe: %d tickers", len(tickers))

    # Fetch benchmark first
    logger.info("Fetching benchmark (%s)...", benchmark)
    bench_df = fetch_benchmark(benchmark)
    if bench_df is None or "Close" not in bench_df.columns:
        raise RuntimeError(f"Could not fetch benchmark {benchmark} — check your connection")
    bench_close: pd.Series = bench_df["Close"].dropna()

    # Batch-fetch all price history (with caching)
    logger.info("Fetching price history for %d tickers...", len(tickers))
    all_data = fetch_batch(tickers, show_progress=show_progress)
    logger.info("Data available for %d/%d tickers", len(all_data), len(tickers))

    # Analyse each ticker
    results: list[MomentumScore] = []

    if show_progress:
        items = tqdm(all_data.items(), desc="Analyzing", unit="ticker")
    else:
        items = all_data.items()

    for ticker, df in items:
        score = _analyze_ticker(ticker, df, bench_close)
        if score is None:
            continue
        if score.price < min_price:
            continue
        if max_price and score.price > max_price:
            continue
        if score.score < min_score:
            continue
        results.append(score)

    # Sort and assign ranks
    results.sort(key=lambda s: s.score, reverse=True)
    for i, s in enumerate(results, start=1):
        s.rank = i

    logger.info(
        "Scan complete. %d tickers scored above %.0f. Returning top %d.",
        len(results),
        min_score,
        min(top_n, len(results)),
    )
    return results[:top_n]
