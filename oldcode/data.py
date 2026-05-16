"""
Market data fetching with disk caching.

Uses yfinance to pull OHLCV history for a list of tickers.
Data is cached locally (Parquet or CSV) to avoid hammering the API
every time you run the scanner.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "prices"
CACHE_TTL_HOURS = 4          # Re-download if data is older than this
HISTORY_YEARS = 2            # How far back to pull (need 12 months + buffer)
BATCH_SIZE = 100             # yfinance download batch size (>100 gets noisy)
MIN_TRADING_DAYS = 252       # Minimum history required for a full momentum calc


def _ticker_cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('-', '_')}.parquet"


def _is_stale(path: Path) -> bool:
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime > timedelta(hours=CACHE_TTL_HOURS)


def _load_cached(ticker: str) -> pd.DataFrame | None:
    path = _ticker_cache_path(ticker)
    if path.exists() and not _is_stale(path):
        try:
            df = pd.read_parquet(path)
            return df
        except Exception as exc:
            logger.debug("Cache read failed for %s: %s", ticker, exc)
    return None


def _save_cached(ticker: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(_ticker_cache_path(ticker))
    except Exception as exc:
        logger.debug("Cache write failed for %s: %s", ticker, exc)


def fetch_ticker(ticker: str, years: int = HISTORY_YEARS) -> pd.DataFrame | None:
    """
    Fetch adjusted OHLCV history for a single ticker.

    Returns a DataFrame indexed by date with columns:
    Open, High, Low, Close, Volume, Dividends, Stock Splits.
    Returns None if the ticker is invalid or data is insufficient.
    """
    cached = _load_cached(ticker)
    if cached is not None:
        return cached

    try:
        end = datetime.today()
        start = end - timedelta(days=years * 365 + 30)
        df = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df.empty or len(df) < MIN_TRADING_DAYS // 2:
            return None

        # Flatten MultiIndex columns (yfinance quirk with single tickers)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        _save_cached(ticker, df)
        return df

    except Exception as exc:
        logger.debug("Fetch failed for %s: %s", ticker, exc)
        return None


def fetch_batch(
    tickers: list[str],
    years: int = HISTORY_YEARS,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV history for a list of tickers, with local caching.

    First loads from cache for tickers whose cache is fresh.  Remaining
    tickers are fetched from yfinance in batches for efficiency.

    Args:
        tickers: List of ticker symbols.
        years: Years of history to fetch.
        show_progress: Show tqdm progress bar.

    Returns:
        Dict mapping ticker → DataFrame.  Tickers that fail are omitted.
    """
    result: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for ticker in tickers:
        cached = _load_cached(ticker)
        if cached is not None:
            result[ticker] = cached
        else:
            to_fetch.append(ticker)

    if not to_fetch:
        return result

    logger.info("Fetching %d tickers from yfinance...", len(to_fetch))
    end = datetime.today()
    start = end - timedelta(days=years * 365 + 30)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    batches = [to_fetch[i : i + BATCH_SIZE] for i in range(0, len(to_fetch), BATCH_SIZE)]
    iterator = tqdm(batches, desc="Downloading", unit="batch") if show_progress else batches

    for batch in iterator:
        try:
            raw = yf.download(
                batch,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            if raw.empty:
                continue

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                    else:
                        if ticker not in raw.columns.get_level_values(0):
                            continue
                        df = raw[ticker].copy()
                        df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns

                    df.index = pd.to_datetime(df.index)
                    df = df.dropna(how="all").sort_index()

                    if len(df) < MIN_TRADING_DAYS // 2:
                        continue

                    _save_cached(ticker, df)
                    result[ticker] = df
                except Exception as exc:
                    logger.debug("Parse failed for %s: %s", ticker, exc)

        except Exception as exc:
            logger.warning("Batch download failed: %s", exc)

    return result


def fetch_benchmark(symbol: str = "SPY", years: int = HISTORY_YEARS) -> pd.DataFrame | None:
    """Fetch benchmark (SPY / QQQ) for relative-strength calculation."""
    return fetch_ticker(symbol, years)
