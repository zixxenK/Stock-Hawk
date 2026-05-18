"""
Insider trade scraper (Section 16 / Form 4 filings).

Sources:
  1. SEC EDGAR full-text search API  — official, free, structured JSON
  2. OpenInsider.com                 — aggregated, human-readable, scrape-able
  3. InsiderFinance.io proxy pattern  — mirrors SEC data with enrichment

Form 4 is the SEC filing that insiders (officers, directors, ≥10% holders)
must file within 2 business days of a transaction.  Clusters of purchases
by multiple insiders on the same ticker are the strongest "smart money"
signal in the system.

Key insight: We care about PURCHASES (code P or M), not sales.
Sales are often tax-driven.  Purchases are almost always conviction buys.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from xml.etree import ElementTree

from bs4 import BeautifulSoup

from base import BaseScraper

logger = logging.getLogger(__name__)

# Transaction type codes on SEC Form 4
_PURCHASE_CODES = {"P", "M", "A"}   # Purchase, Option exercise, Award
_SALE_CODES     = {"S", "D", "F"}   # Sale, Disposition, Tax withholding

# Title seniority map — we weight C-suite buys more than rank-and-file
_TITLE_WEIGHT: dict[str, float] = {
    "ceo":       1.0,
    "cfo":       0.95,
    "coo":       0.90,
    "cto":       0.85,
    "president": 0.90,
    "chairman":  0.95,
    "director":  0.75,
    "svp":       0.70,
    "evp":       0.75,
    "vp":        0.65,
    "officer":   0.60,
    "10%":       0.80,
}


def _title_seniority(title: str) -> float:
    """Return a 0-1 weight based on the insider's title."""
    t = (title or "").lower()
    for key, weight in _TITLE_WEIGHT.items():
        if key in t:
            return weight
    return 0.55  # generic insider


class SECEdgarForm4Scraper(BaseScraper):
    """
    Fetches Form 4 filings from SEC EDGAR's full-text search API.

    EDGAR EFTS endpoint:
      GET https://efts.sec.gov/LATEST/search-index?q=%22form+4%22&dateRange=custom
          &startdt=YYYY-MM-DD&enddt=YYYY-MM-DD&forms=4

    Individual filing XML:
      https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany
          &CIK=AAPL&type=4&dateb=&owner=include&count=40&search_text=
    """

    source_name = "sec_edgar"
    base_url    = "https://efts.sec.gov"

    _SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
    _FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    _DETAIL_URL = "https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&type=4"

    def fetch_records(
        self,
        days_back: int = 7,
        ticker: str | None = None,
        purchases_only: bool = True,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent Form 4 filings.

        Args:
            days_back:      Days of history to fetch (max 30 recommended).
            ticker:         Restrict to a single ticker symbol.
            purchases_only: If True, skip sale transactions.
        """
        end_dt   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_dt = (
            datetime.now(timezone.utc) - timedelta(days=days_back)
        ).strftime("%Y-%m-%d")

        params: dict[str, Any] = {
            "q":         f'"form 4"' + (f' "{ticker.upper()}"' if ticker else ""),
            "forms":     "4",
            "dateRange": "custom",
            "startdt":   start_dt,
            "enddt":     end_dt,
            "_source":   "full_search",
            "hits.hits.total.value": 40,
        }

        result = self.get(self._SEARCH_URL, params=params)
        if not result.ok:
            logger.warning("EDGAR search failed: %s", result.error)
            return []

        try:
            data = json.loads(result.text)
        except json.JSONDecodeError:
            logger.warning("EDGAR JSON parse failed")
            return []

        hits = (data.get("hits") or {}).get("hits") or []
        records: list[dict[str, Any]] = []

        for hit in hits:
            src = hit.get("_source") or {}
            filing_url = (
                "https://www.sec.gov/Archives/"
                + (src.get("file_date") or "")
                + "/"
                + (src.get("period_of_report") or "")
            )
            # Parse the actual XML filing for transaction details
            recs = self._parse_filing_index(src, purchases_only)
            records.extend(recs)

        logger.info("EDGAR Form 4: %d records (days_back=%d)", len(records), days_back)
        return records

    def _parse_filing_index(
        self, src: dict, purchases_only: bool
    ) -> list[dict[str, Any]]:
        """Extract structured data from an EDGAR search hit."""
        records: list[dict[str, Any]] = []

        tickers_raw = src.get("entity_name") or src.get("file_date") or ""
        period      = src.get("period_of_report") or ""
        filed_at    = src.get("file_date") or ""
        cik         = src.get("cik") or ""

        # The EFTS endpoint doesn't always include ticker directly.
        # We extract from the display_names field or entity_name.
        display = src.get("display_names") or []
        ticker  = ""
        issuer  = ""
        for item in display if isinstance(display, list) else [display]:
            name = str(item)
            # Look for TICKER in parentheses: "Apple Inc. (AAPL)"
            m = re.search(r"\(([A-Z]{1,6})\)", name)
            if m:
                ticker = m.group(1)
                issuer = name
                break

        if not ticker:
            return records

        # Build a minimal record from index data (full XML parse is rate-limited)
        code = (src.get("transaction_code") or "P").upper()
        if purchases_only and code not in _PURCHASE_CODES:
            return records

        insider = src.get("reporting_owner_name") or src.get("filer_name") or ""
        title   = src.get("reporting_owner_relationship") or ""
        shares  = self._safe_int(src.get("transaction_shares"))
        price   = self._safe_float(src.get("transaction_price_per_share"))

        records.append({
            "source":             self.source_name,
            "ticker":             ticker.upper(),
            "company_name":       issuer,
            "insider_name":       insider,
            "insider_title":      title,
            "insider_seniority":  _title_seniority(title),
            "trade_type":         "P" if code in _PURCHASE_CODES else "S",
            "trade_date":         period[:10] if period else filed_at[:10],
            "shares":             shares,
            "price_per_share":    price,
            "total_value":        (shares or 0) * (price or 0) if shares and price else None,
            "shares_owned_after": self._safe_int(src.get("shares_owned_following")),
            "form_url":           f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4",
        })
        return records

    @staticmethod
    def _safe_int(v: Any) -> int | None:
        try:
            return int(float(str(v).replace(",", "")))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(v: Any) -> float | None:
        try:
            return float(str(v).replace(",", "").replace("$", ""))
        except (TypeError, ValueError):
            return None


class OpenInsiderScraper(BaseScraper):
    """
    Scrapes OpenInsider.com for recent insider purchases.
    OpenInsider aggregates SEC Form 4 data with sector/industry enrichment.
    URL: https://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=7&td=0
         &tdr=&fdlyl=&fdlyh=&daysago=&xp=1&vl=100&vh=&ocl=&och=&sic1=-1&
         sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=
         &oc=&sortcol=0&cnt=100&page=1
    """

    source_name = "openinsider"
    base_url    = "https://openinsider.com"

    _SCREENER_URL = "https://openinsider.com/screener"

    def fetch_records(
        self,
        days_back: int = 7,
        min_value: int = 50_000,
        purchases_only: bool = True,
        page_limit: int = 3,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent insider purchases from OpenInsider screener.

        Args:
            days_back:      Filter to last N days.
            min_value:      Minimum transaction value ($).
            purchases_only: Only return purchases.
            page_limit:     Max pages to scrape.
        """
        all_records: list[dict[str, Any]] = []

        for page in range(1, page_limit + 1):
            params = {
                "fd":      days_back,
                "td":      0,
                "xp":      1 if purchases_only else 0,   # purchases only toggle
                "vl":      max(1, min_value // 1000),     # $k lower bound
                "sortcol": 0,
                "cnt":     100,
                "page":    page,
            }

            result = self.get(self._SCREENER_URL, params=params)
            if not result.ok:
                logger.warning("OpenInsider page %d failed: %s", page, result.error)
                break

            rows = self._parse_screener(result.text)
            if not rows:
                break

            all_records.extend(rows)

        logger.info("OpenInsider: %d records", len(all_records))
        return all_records

    def _parse_screener(self, html: str) -> list[dict[str, Any]]:
        """Parse the OpenInsider screener HTML table."""
        records: list[dict[str, Any]] = []

        try:
            soup = BeautifulSoup(html, "lxml")
            table = soup.find("table", {"class": re.compile(r"tinytable|body-table")})
            if not table:
                # Try generic table
                tables = soup.find_all("table")
                table = tables[0] if tables else None
            if not table:
                return records

            headers: list[str] = []
            for th in table.find_all("th"):
                headers.append(th.get_text(strip=True).lower())

            for tr in table.find_all("tr")[1:]:  # Skip header row
                cells = tr.find_all(["td"])
                if len(cells) < 8:
                    continue

                row_data = {
                    h: cells[i].get_text(strip=True)
                    for i, h in enumerate(headers)
                    if i < len(cells)
                }

                # OpenInsider column names vary; try multiple keys
                ticker = (
                    row_data.get("ticker")
                    or row_data.get("symbol")
                    or self._get_cell_link_text(cells, 2)
                    or ""
                ).upper().strip().split()[0]

                if not ticker or len(ticker) > 6:
                    continue

                trade_date = (
                    row_data.get("filing\xa0date")
                    or row_data.get("date")
                    or row_data.get("trade\xa0date")
                    or ""
                )
                trade_date = trade_date[:10].replace("/", "-") if trade_date else ""

                insider_name = (
                    row_data.get("insider name")
                    or row_data.get("insider\xa0name")
                    or row_data.get("insider")
                    or ""
                ).strip()

                title = (
                    row_data.get("title")
                    or row_data.get("relationship")
                    or ""
                ).strip()

                trade_type_raw = (
                    row_data.get("transaction")
                    or row_data.get("type")
                    or "P"
                ).strip().upper()[:1]

                shares_raw = re.sub(r"[^\d]", "",
                    row_data.get("qty") or row_data.get("shares") or "0")
                price_raw  = re.sub(r"[^\d.]", "",
                    row_data.get("price") or row_data.get("share price") or "0")
                value_raw  = re.sub(r"[^\d]", "",
                    row_data.get("value") or row_data.get("$ value") or "0")

                shares = int(shares_raw) if shares_raw.isdigit() else None
                try:
                    price = float(price_raw) if price_raw else None
                except ValueError:
                    price = None
                try:
                    total_value = int(value_raw) if value_raw else None
                except ValueError:
                    total_value = None

                records.append({
                    "source":             self.source_name,
                    "ticker":             ticker,
                    "company_name":       row_data.get("company name", ""),
                    "insider_name":       insider_name,
                    "insider_title":      title,
                    "insider_seniority":  _title_seniority(title),
                    "trade_type":         "P" if trade_type_raw == "P" else "S",
                    "trade_date":         trade_date,
                    "shares":             shares,
                    "price_per_share":    price,
                    "total_value":        total_value,
                    "shares_owned_after": None,
                    "form_url":           None,
                })

        except Exception as exc:
            logger.warning("OpenInsider parse error: %s", exc)

        return records

    @staticmethod
    def _get_cell_link_text(cells: list, idx: int) -> str:
        """Extract text from an anchor tag inside a cell."""
        try:
            a = cells[idx].find("a")
            return a.get_text(strip=True) if a else cells[idx].get_text(strip=True)
        except Exception:
            return ""


class InsiderScraper:
    """
    Facade: tries EDGAR first, falls back to OpenInsider.
    Always deduplicates across both sources before upserting.
    """

    def __init__(self, db: Any | None = None) -> None:
        self.db     = db
        self._edgar = SECEdgarForm4Scraper(db=db)
        self._open  = OpenInsiderScraper(db=db)

    def run(
        self,
        days_back: int = 7,
        min_value: int = 25_000,
    ) -> tuple[int, int]:
        """Fetch from both sources, upsert to db, return (total, new)."""
        total = 0
        new   = 0

        # EDGAR (official) first
        edgar_records = self._edgar.fetch_records(days_back=days_back)
        for rec in edgar_records:
            total += 1
            try:
                if self.db and self.db.upsert_insider_trade(rec):
                    new += 1
            except Exception as exc:
                logger.debug("Upsert error: %s", exc)

        # OpenInsider (enriched) supplements
        open_records = self._open.fetch_records(
            days_back=days_back, min_value=min_value
        )
        for rec in open_records:
            total += 1
            try:
                if self.db and self.db.upsert_insider_trade(rec):
                    new += 1
            except Exception as exc:
                logger.debug("Upsert error: %s", exc)

        logger.info("InsiderScraper run: fetched %d, new %d", total, new)
        return total, new

    def fetch_for_tickers(
        self,
        tickers: list[str],
        days_back: int = 2,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Return recent insider PURCHASES grouped by ticker.
        Used by the Golden Loop 48-hour correlation window.
        """
        if not self.db:
            return {}

        result: dict[str, list[dict[str, Any]]] = {}
        ticker_set = {t.upper() for t in tickers}

        for ticker in ticker_set:
            trades = self.db.get_insider_trades(
                ticker=ticker,
                days_back=days_back,
                purchases_only=True,
            )

            if not trades:
                # Try live EDGAR fetch if history is missing.
                edgar_records = self._edgar.fetch_records(
                    days_back=days_back,
                    ticker=ticker,
                    purchases_only=True,
                )
                for rec in edgar_records:
                    self.db.upsert_insider_trade(rec)

                if not edgar_records:
                    open_records = self._open.fetch_records(
                        days_back=days_back,
                        min_value=25_000,
                        purchases_only=True,
                        page_limit=2,
                    )
                    for rec in open_records:
                        if rec.get("ticker", "").upper() == ticker:
                            self.db.upsert_insider_trade(rec)

                trades = self.db.get_insider_trades(
                    ticker=ticker,
                    days_back=days_back,
                    purchases_only=True,
                )

            if trades:
                result[ticker] = trades

        return result
