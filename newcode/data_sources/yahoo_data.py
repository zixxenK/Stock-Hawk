from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _safe_list(values: list | None, idx: int, default=None):
    if not values or idx >= len(values) or values[idx] is None:
        return default
    return values[idx]


def _compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50.0)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    sig_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"] = ((macd_line - sig_line) / df["Close"]).clip(-0.05, 0.05) / 0.05

    df["sma50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["sma200"] = df["Close"].rolling(200, min_periods=1).mean()
    df["high_52w_pct"] = (df["Close"] / df["Close"].rolling(252, min_periods=1).max()).clip(0.0, 1.0)
    df["vol_trend"] = (df["Volume"] / df["Volume"].rolling(20, min_periods=1).mean()).clip(0.0, 10.0)
    df["hist_vol"] = df["log_return"].rolling(20, min_periods=1).std().fillna(df["log_return"].std())

    return df.fillna(0.0)


def fetch_price_history(
    ticker: str,
    period: Literal["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"] = "1y",
    interval: Literal["1d", "1wk", "1mo"] = "1d",
) -> pd.DataFrame:
    """Fetch daily OHLCV history for a ticker from Yahoo Finance."""
    ticker = ticker.upper().strip()
    url = _YAHOO_CHART_URL.format(ticker=ticker)
    params = {
        "range": period,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits",
    }
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        result = payload.get("chart", {}).get("result")
        if not result:
            logger.warning("Yahoo chart payload missing result for %s", ticker)
            return pd.DataFrame()
        result = result[0]
        timestamps = result.get("timestamp") or []
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []

        records = []
        for i, ts in enumerate(timestamps):
            close = _safe_list(closes, i)
            if close is None:
                continue
            records.append({
                "Date": datetime.utcfromtimestamp(int(ts)).date(),
                "Open": float(_safe_list(opens, i, close) or close),
                "High": float(_safe_list(highs, i, close) or close),
                "Low": float(_safe_list(lows, i, close) or close),
                "Close": float(close),
                "Volume": float(_safe_list(volumes, i, 0) or 0.0),
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = _compute_technical_indicators(df)
        return df

    except Exception as exc:
        logger.warning("Failed to fetch Yahoo market data for %s: %s", ticker, exc)
        return pd.DataFrame()
