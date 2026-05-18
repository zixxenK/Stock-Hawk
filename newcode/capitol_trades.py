"""
Congress trade scraper.

Sources (in priority order):
  1. QuiverQuant public API   — JSON, reliable, no JS required
  2. Capitol Trades HTML      — fallback, parsed with BeautifulSoup
  3. House/Senate EDGAR API   — official XML disclosure feeds

All raw responses are saved to the data lake before parsing.

Congress trades are legally required to be disclosed within 45 days of
execution under the STOCK Act (2012).  Purchases by legislators in
sectors they regulate are the primary "suspicious volume" signal.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from bs4 import BeautifulSoup

from base import BaseScraper

logger = logging.getLogger(__name__)

# ── Amount range midpoints (Congress uses ranges, not exact) ────────────────
# Official House/Senate disclosure buckets
_AMOUNT_BUCKETS: dict[str, tuple[int, int]] = {
    "$1,001 - $15,000":        (1_001,    15_000),
    "$15,001 - $50,000":       (15_001,   50_000),
    "$50,001 - $100,000":      (50_001,  100_000),
    "$100,001 - $250,000":     (100_001, 250_000),
    "$250,001 - $500,000":     (250_001, 500_000),
    "$500,001 - $1,000,000":   (500_001, 1_000_000),
    "$1,000,001 - $5,000,000": (1_000_001, 5_000_000),
    "Over $5,000,000":         (5_000_001, 25_000_000),
    # Capitol Trades format variants
    "1001 - 15000":   (1_001,    15_000),
    "15001 - 50000":  (15_001,   50_000),
    "50001 - 100000": (50_001,  100_000),
    "100001 - 250000":(100_001, 250_000),
    "250001 - 500000":(250_001, 500_000),
    "500001 - 1000000":(500_001, 1_000_000),
    "1000001 - 5000000":(1_000_001, 5_000_000),
}


def _parse_amount(raw: str) -> tuple[int | None, int | None, int | None]:
    """
    Parse a raw amount range string into (min, max, mid).
    Returns (None, None, None) if unparseable.
    """
    if not raw:
        return None, None, None

    raw = raw.strip()

    # Exact number
    exact = re.sub(r"[$,]", "", raw).strip()
    if exact.lstrip("-").isdigit():
        v = int(exact)
        return v, v, v

    # Bucket lookup
    for key, (lo, hi) in _AMOUNT_BUCKETS.items():
        if key.lower() in raw.lower():
            return lo, hi, (lo + hi) // 2

    # Regex: "1,234,567 - 5,678,900" or "$1M - $5M"
    nums = re.findall(r"[\d,]+", raw.replace("$", ""))
    cleaned = [int(n.replace(",", "")) for n in nums if n.replace(",", "").isdigit()]
    if len(cleaned) >= 2:
        lo, hi = cleaned[0], cleaned[-1]
        return lo, hi, (lo + hi) // 2
    if len(cleaned) == 1:
        v = cleaned[0]
        return v, v, v

    return None, None, None


def _normalize_trade_type(raw: str) -> str:
    raw = (raw or "").lower().strip()
    if any(w in raw for w in ("purchase", "buy", "bought", "p")):
        return "purchase"
    if any(w in raw for w in ("sale", "sell", "sold", "s")):
        return "sale"
    if "exchange" in raw:
        return "exchange"
    return raw


class QuiverQuantCongressScraper(BaseScraper):
    """
    Scrapes Congress trading data from the QuiverQuant public API.
    No authentication required for the live feed.
    API docs: https://api.quiverquant.com/
    """

    source_name = "quiverquant_congress"
    base_url    = "https://api.quiverquant.com"

    _LIVE_URL   = "https://api.quiverquant.com/beta/live/congresstrading"
    _HIST_URL   = "https://api.quiverquant.com/beta/historical/congresstrading/{ticker}"

    def fetch_records(
        self,
        ticker: str | None = None,
        days_back: int = 90,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """
        Fetch congress trades.

        Args:
            ticker:    If given, fetch only trades for that ticker.
                       If None, fetch the recent live feed for all tickers.
            days_back: Filter to trades within this many days.
        """
        if ticker:
            return self._fetch_ticker(ticker, days_back)
        return self._fetch_live(days_back)

    def _fetch_live(self, days_back: int) -> list[dict[str, Any]]:
        result = self.get(
            self._LIVE_URL,
            extra_headers={"Accept": "application/json"},
        )
        if not result.ok:
            logger.warning("QuiverQuant live feed failed: %s", result.error)
            return []

        try:
            raw = json.loads(result.text)
        except json.JSONDecodeError as exc:
            logger.warning("QuiverQuant JSON parse error: %s", exc)
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        records: list[dict[str, Any]] = []

        for row in raw if isinstance(raw, list) else raw.get("data", []):
            rec = self._normalise_quiver_row(row)
            if rec is None:
                continue
            try:
                td = datetime.fromisoformat(rec["trade_date"].replace("Z", "+00:00"))
                if td.replace(tzinfo=timezone.utc) < cutoff.replace(tzinfo=timezone.utc):
                    continue
            except Exception:
                pass
            records.append(rec)

        logger.info("QuiverQuant live: %d records", len(records))
        return records

    def _fetch_ticker(self, ticker: str, days_back: int) -> list[dict[str, Any]]:
        url = self._HIST_URL.format(ticker=ticker.upper())
        result = self.get(url, extra_headers={"Accept": "application/json"})
        if not result.ok:
            return []
        try:
            raw = json.loads(result.text)
        except Exception:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        records: list[dict[str, Any]] = []
        for row in raw if isinstance(raw, list) else raw.get("data", []):
            rec = self._normalise_quiver_row(row)
            if rec:
                try:
                    td = datetime.fromisoformat(rec["trade_date"].replace("Z", "+00:00"))
                    if td.replace(tzinfo=timezone.utc) >= cutoff.replace(tzinfo=timezone.utc):
                        records.append(rec)
                except Exception:
                    records.append(rec)
        return records

    def _normalise_quiver_row(self, row: dict) -> dict[str, Any] | None:
        """Map QuiverQuant API fields → our schema."""
        ticker = (row.get("Ticker") or row.get("ticker") or "").upper().strip()
        if not ticker:
            return None

        amount_raw = str(row.get("Amount") or row.get("amount") or "")
        lo, hi, mid = _parse_amount(amount_raw)

        politician = (
            row.get("Representative")
            or row.get("representative")
            or row.get("politician")
            or ""
        ).strip()

        trade_date = (
            row.get("TransactionDate")
            or row.get("transaction_date")
            or row.get("trade_date")
            or row.get("Date")
            or ""
        )
        if trade_date and len(trade_date) > 10:
            trade_date = trade_date[:10]

        return {
            "source":           self.source_name,
            "politician":       politician,
            "party":            row.get("Party") or row.get("party"),
            "chamber":          (row.get("Chamber") or row.get("chamber") or "").lower(),
            "state":            row.get("State") or row.get("state"),
            "ticker":           ticker,
            "asset_name":       row.get("Asset") or row.get("asset_name") or ticker,
            "trade_type":       _normalize_trade_type(
                                    row.get("Transaction") or row.get("transaction") or ""
                                ),
            "trade_date":       trade_date,
            "disclosure_date":  row.get("DisclosureDate") or row.get("disclosure_date"),
            "amount_min":       lo,
            "amount_max":       hi,
            "amount_mid":       mid,
            "comment":          row.get("Comment") or row.get("comment"),
        }


class CapitolTradesScraper(BaseScraper):
    """
    Scrapes https://www.capitoltrades.com/ using HTML parsing.
    Used as fallback when QuiverQuant is unavailable.

    Capitol Trades provides a well-structured public table — no login required.
    """

    source_name = "capitol_trades"
    base_url    = "https://www.capitoltrades.com"

    _TRADES_URL = "https://www.capitoltrades.com/trades"

    def fetch_records(
        self,
        page_limit: int = 5,
        ticker: str | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """
        Scrape the public trades table.

        Args:
            page_limit: Max pages to scrape (each has ~20 rows).
            ticker:     If set, appended as ?ticker= filter.
        """
        all_records: list[dict[str, Any]] = []

        for page in range(1, page_limit + 1):
            params: dict[str, Any] = {"page": page}
            if ticker:
                params["ticker"] = ticker.upper()

            result = self.get(self._TRADES_URL, params=params)
            if not result.ok:
                logger.warning("CapitolTrades page %d failed: %s", page, result.error)
                break

            rows = self._parse_html_table(result.text)
            if not rows:
                logger.debug("CapitolTrades page %d: no rows, stopping", page)
                break

            all_records.extend(rows)
            logger.debug("CapitolTrades page %d: %d rows", page, len(rows))

        logger.info("CapitolTrades total: %d records", len(all_records))
        return all_records

    def _parse_html_table(self, html: str) -> list[dict[str, Any]]:
        """Parse the trades table from Capitol Trades HTML."""
        records: list[dict[str, Any]] = []

        try:
            soup = BeautifulSoup(html, "lxml")

            # Capitol Trades renders rows as <tr> elements inside a <table>
            # or as <div data-row> elements depending on the page variant
            rows = soup.select("table tbody tr")
            if not rows:
                # Try alternative layout
                rows = soup.select("[data-testid='trade-row'], .trade-row, tr.q-tr")

            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 6:
                    continue

                texts = [c.get_text(strip=True) for c in cells]

                # Capitol Trades column order (may vary by layout version):
                # [0] Politician  [1] Chamber  [2] Ticker  [3] Asset
                # [4] Type        [5] Date      [6] Amount  [7] Disclosed
                try:
                    politician = texts[0]
                    chamber    = texts[1].lower() if len(texts) > 1 else ""
                    ticker_raw = texts[2] if len(texts) > 2 else ""
                    trade_type = _normalize_trade_type(texts[4] if len(texts) > 4 else "")
                    trade_date = self._parse_date(texts[5] if len(texts) > 5 else "")
                    amount_raw = texts[6] if len(texts) > 6 else ""
                    disc_date  = self._parse_date(texts[7] if len(texts) > 7 else "")

                    # Ticker might have exchange suffix like "AAPL (NASDAQ)"
                    ticker = re.split(r"[\s(]", ticker_raw.upper())[0].strip()
                    if not ticker or len(ticker) > 6:
                        continue

                    lo, hi, mid = _parse_amount(amount_raw)

                    records.append({
                        "source":          self.source_name,
                        "politician":      politician.strip(),
                        "party":           self._extract_party(row),
                        "chamber":         chamber,
                        "state":           self._extract_state(row),
                        "ticker":          ticker,
                        "asset_name":      ticker_raw,
                        "trade_type":      trade_type,
                        "trade_date":      trade_date,
                        "disclosure_date": disc_date,
                        "amount_min":      lo,
                        "amount_max":      hi,
                        "amount_mid":      mid,
                        "comment":         None,
                    })
                except Exception as exc:
                    logger.debug("Row parse error: %s — row text: %s", exc, texts[:4])

        except Exception as exc:
            logger.warning("HTML parse error: %s", exc)

        return records

    @staticmethod
    def _parse_date(raw: str) -> str | None:
        """Attempt to normalise a date string to YYYY-MM-DD."""
        raw = raw.strip()
        if not raw:
            return None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y", "%d %b %Y",
                    "%B %d, %Y", "%m-%d-%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        # Grab first date-like substring
        m = re.search(r"\d{4}-\d{2}-\d{2}", raw)
        if m:
            return m.group(0)
        return raw[:10] if len(raw) >= 10 else None

    @staticmethod
    def _extract_party(row: Any) -> str | None:
        text = row.get_text()
        if "(R)" in text or "Republican" in text:
            return "R"
        if "(D)" in text or "Democrat" in text:
            return "D"
        if "(I)" in text or "Independent" in text:
            return "I"
        return None

    @staticmethod
    def _extract_state(row: Any) -> str | None:
        text = row.get_text()
        m = re.search(r"\(([A-Z]{2})\)", text)
        return m.group(1) if m else None


class CongressScraper:
    """
    Facade that tries QuiverQuant first, falls back to Capitol Trades HTML.

    Usage:
        scraper = CongressScraper(db=db)
        total, new = scraper.run(days_back=30)
    """

    def __init__(self, db: Any | None = None) -> None:
        self.db = db
        self._quiver = QuiverQuantCongressScraper(db=db)
        self._capitol = CapitolTradesScraper(db=db)

    def run(self, days_back: int = 30, ticker: str | None = None) -> tuple[int, int]:
        """Run both scrapers, returning (total_fetched, new_saved)."""
        total = 0
        new   = 0

        # Try QuiverQuant first
        t, n = self._quiver.run(days_back=days_back, ticker=ticker)
        total += t
        new   += n

        if t == 0:
            # Fall back to Capitol Trades HTML
            logger.info("QuiverQuant empty — falling back to Capitol Trades HTML")
            t2, n2 = self._capitol.run(
                page_limit=5 if not ticker else 2,
                ticker=ticker,
            )
            total += t2
            new   += n2

        return total, new

    def fetch_for_tickers(
        self,
        tickers: list[str],
        days_back: int = 48,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Fetch congress trades for a batch of tickers.
        Returns dict: ticker → list of trade dicts.
        Used by the Golden Loop for 48-hour correlation window.
        """
        result: dict[str, list[dict[str, Any]]] = {}
        ticker_set = {t.upper() for t in tickers}

        # Try the fast QuiverQuant live feed first.
        live_records = self._quiver.fetch_records(days_back=days_back)

        # If the live feed returns nothing, try ticker-specific Quiver calls.
        if not live_records:
            for ticker in ticker_set:
                live_records.extend(
                    self._quiver.fetch_records(ticker=ticker, days_back=days_back)
                )

        # If Quiver is unavailable, fall back to CapitolTrades HTML for each ticker.
        if not live_records:
            for ticker in ticker_set:
                fallback = self._capitol.fetch_records(page_limit=2, ticker=ticker)
                live_records.extend(fallback)

        cutoff_str = (
            datetime.now(timezone.utc) - timedelta(hours=days_back)
        ).strftime("%Y-%m-%d")

        for rec in live_records:
            t = (rec.get("ticker") or "").upper()
            if t not in ticker_set:
                continue
            if rec.get("trade_date", "") < cutoff_str:
                continue
            result.setdefault(t, []).append(rec)

        return result
