"""
Stock universe management.

Fetches and caches the S&P 500 and Nasdaq 100 constituent lists from
Wikipedia.  Falls back to a curated ~200-ticker seed list when offline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent / ".cache" / "universe.json"
CACHE_TTL_DAYS = 7

# ── Curated seed list (used when Wikipedia is unreachable) ────────────────────
# Covers the largest and most liquid names across sectors so the scanner
# still works without internet access to Wikipedia.
SEED_UNIVERSE: list[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AVGO", "TSLA",
    "AMD", "INTC", "ORCL", "CRM", "ADBE", "NOW", "INTU", "QCOM",
    "TXN", "AMAT", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "FTNT",
    "PANW", "CRWD", "ZS", "DDOG", "SNOW", "MDB",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP",
    "SPGI", "MCO", "ICE", "CME", "BLK", "SCHW", "COF", "USB",
    # Healthcare
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR",
    "AMGN", "ISRG", "SYK", "BSX", "ELV", "HUM", "CI", "CVS",
    "VRTX", "REGN", "BIIB", "MRNA",
    # Consumer Discretionary
    "AMZN", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG",
    "CMG", "ABNB", "LULU", "ORLY", "AZO", "DECK",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "CL", "MDLZ", "KHC", "GIS",
    # Industrials
    "CAT", "RTX", "HON", "UNP", "UPS", "DE", "GE", "BA", "LMT",
    "NOC", "GD", "ETN", "EMR", "ITW", "PH", "MMM",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO",
    # Materials
    "LIN", "APD", "SHW", "FCX", "NEM", "DOW", "DD",
    # Communication Services
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA", "TTWO",
    # Real Estate & Utilities
    "PLD", "AMT", "EQIX", "CCI", "NEE", "DUK", "SO",
    # ETFs (for benchmark comparisons)
    "SPY", "QQQ", "IWM",
]


def _load_cache() -> dict | None:
    try:
        if CACHE_PATH.exists():
            data = json.loads(CACHE_PATH.read_text())
            saved = datetime.fromisoformat(data["saved_at"])
            if datetime.utcnow() - saved < timedelta(days=CACHE_TTL_DAYS):
                return data
    except Exception:
        pass
    return None


def _save_cache(tickers: list[str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps({"saved_at": datetime.utcnow().isoformat(), "tickers": tickers})
    )


def _fetch_sp500_wikipedia() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()


def _fetch_nasdaq100_wikipedia() -> list[str]:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    # The ticker column varies — try common names
    for tbl in tables:
        for col in ("Ticker", "Symbol", "Ticker symbol"):
            if col in tbl.columns:
                return tbl[col].str.replace(".", "-", regex=False).tolist()
    return []


def get_universe(
    include_sp500: bool = True,
    include_nasdaq100: bool = True,
    extra_tickers: list[str] | None = None,
    use_cache: bool = True,
) -> list[str]:
    """
    Return a deduplicated list of ticker symbols to scan.

    Fetches S&P 500 and/or Nasdaq 100 constituents from Wikipedia,
    caching the result for ``CACHE_TTL_DAYS`` days.  Falls back to the
    curated SEED_UNIVERSE when Wikipedia is unreachable.

    Args:
        include_sp500: Include S&P 500 constituents.
        include_nasdaq100: Include Nasdaq 100 constituents.
        extra_tickers: Additional tickers to append (e.g. ["MSTR", "PLTR"]).
        use_cache: Read from / write to the local cache.

    Returns:
        Sorted, deduplicated list of ticker symbols.
    """
    if use_cache:
        cached = _load_cache()
        if cached:
            logger.debug("Universe loaded from cache (%d tickers)", len(cached["tickers"]))
            tickers = list(cached["tickers"])
            if extra_tickers:
                tickers += extra_tickers
            return sorted(set(tickers))

    tickers: list[str] = []

    if include_sp500:
        try:
            tickers += _fetch_sp500_wikipedia()
            logger.info("S&P 500 universe fetched from Wikipedia (%d tickers)", len(tickers))
        except Exception as exc:
            logger.warning("S&P 500 fetch failed (%s), using seed list", exc)
            tickers += SEED_UNIVERSE

    if include_nasdaq100:
        try:
            ndq = _fetch_nasdaq100_wikipedia()
            tickers += ndq
            logger.info("Nasdaq 100 added (%d extra tickers)", len(ndq))
        except Exception as exc:
            logger.warning("Nasdaq 100 fetch failed: %s", exc)

    if not tickers:
        logger.warning("All fetches failed — falling back to seed universe")
        tickers = list(SEED_UNIVERSE)

    # Strip ETF benchmarks from the scan universe (kept only as comparators)
    benchmarks = {"SPY", "QQQ", "IWM"}
    tickers = [t for t in tickers if t not in benchmarks]

    deduped = sorted(set(tickers))
    _save_cache(deduped)
    return deduped + (extra_tickers or [])
