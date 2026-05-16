"""
Base scraper infrastructure.

Every scraper inherits BaseScraper, which handles:
  - Rotating User-Agent headers to avoid bot detection
  - Exponential back-off on 429 / 5xx responses
  - Raw HTML/JSON persistence to the data-lake table in store.py
  - Per-domain rate limiting (min seconds between requests)
  - Session reuse with connection pooling

Design principle: SAVE FIRST, PARSE LATER.
Raw bytes land in `raw_lake` before any parsing happens, so a parsing
bug never loses data and every scrape is fully auditable.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ── User-Agent pool ──────────────────────────────────────────────────────────
# Mix of real browser UAs across Chrome/Firefox on Win/Mac/Linux
_USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
]

# Default seconds to wait between requests to the same domain
_DEFAULT_RATE_SECONDS: dict[str, float] = {
    "capitoltrades.com": 2.0,
    "insiderfinance.io": 3.0,
    "www.sec.gov":       1.0,
    "efts.sec.gov":      1.0,
    "openinsider.com":   1.5,
    "finance.yahoo.com": 0.5,
}


@dataclass
class ScrapeResult:
    """Container returned by every BaseScraper.fetch() call."""
    url:          str
    source:       str
    status_code:  int
    content_type: str
    raw_bytes:    bytes
    headers:      dict[str, str]
    fetched_at:   str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error:        str | None = None

    @property
    def ok(self) -> bool:
        return self.status_code == 200 and self.error is None

    @property
    def text(self) -> str:
        try:
            return self.raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            return ""

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()[:16]


class BaseScraper(ABC):
    """
    Abstract base class for all Flippy scrapers.

    Subclasses implement:
      source_name   : str — identifier stored in the data lake
      base_url      : str — root URL of the target site
      fetch_records : the actual parsing logic

    The base class handles all HTTP mechanics.
    """

    source_name: str = "unknown"
    base_url:    str = ""

    def __init__(
        self,
        db: Any | None = None,          # IntelligenceDB instance (optional)
        rate_seconds: float | None = None,
        timeout: int = 20,
        max_retries: int = 3,
        save_raw: bool = True,
    ) -> None:
        self.db = db
        self.timeout = timeout
        self.save_raw = save_raw
        self._last_request_time: dict[str, float] = {}

        # Determine rate limit for this scraper's domain
        domain = self._extract_domain(self.base_url)
        self._rate_seconds = rate_seconds or _DEFAULT_RATE_SECONDS.get(domain, 1.5)

        # Build a requests.Session with automatic retry on transient failures
        self._session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1.5,         # wait 1.5s, 3s, 4.5s …
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://",  adapter)

    # ── HTTP mechanics ────────────────────────────────────────────────────

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lstrip("www.")
        except Exception:
            return ""

    def _rotate_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "User-Agent":      random.choice(_USER_AGENTS),
            "Accept":          "text/html,application/xhtml+xml,application/json,*/*;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection":      "keep-alive",
            "DNT":             "1",
            "Upgrade-Insecure-Requests": "1",
        }
        if extra:
            headers.update(extra)
        return headers

    def _rate_limit(self, domain: str) -> None:
        """Block until the rate-limit window has elapsed for `domain`."""
        last = self._last_request_time.get(domain, 0.0)
        elapsed = time.time() - last
        wait = self._rate_seconds - elapsed
        if wait > 0:
            # Add small random jitter to look more human
            jitter = random.uniform(0.1, 0.4)
            time.sleep(wait + jitter)
        self._last_request_time[domain] = time.time()

    def get(
        self,
        url: str,
        params: dict | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> ScrapeResult:
        """
        Perform a rate-limited GET and return a ScrapeResult.
        Raw response is saved to the data lake if self.db is set.
        """
        domain = self._extract_domain(url)
        self._rate_limit(domain)

        error: str | None = None
        raw_bytes = b""
        status_code = 0
        content_type = ""
        resp_headers: dict[str, str] = {}

        try:
            resp = self._session.get(
                url,
                params=params,
                headers=self._rotate_headers(extra_headers),
                timeout=self.timeout,
                allow_redirects=True,
            )
            status_code = resp.status_code
            content_type = resp.headers.get("Content-Type", "")
            resp_headers = dict(resp.headers)
            raw_bytes = resp.content

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                logger.warning(
                    "Rate-limited by %s — sleeping %ds", domain, retry_after
                )
                time.sleep(retry_after + random.uniform(1, 5))
                return self.get(url, params, extra_headers)

            if resp.status_code not in (200, 201):
                error = f"HTTP {resp.status_code}"
                logger.warning("Non-200 from %s: %d", url, resp.status_code)

        except requests.exceptions.Timeout:
            error = "timeout"
            logger.warning("Timeout fetching %s", url)
        except requests.exceptions.ConnectionError as exc:
            error = f"connection_error: {exc}"
            logger.warning("Connection error for %s: %s", url, exc)
        except Exception as exc:
            error = f"unexpected: {exc}"
            logger.exception("Unexpected error fetching %s", url)

        result = ScrapeResult(
            url=url,
            source=self.source_name,
            status_code=status_code,
            content_type=content_type,
            raw_bytes=raw_bytes,
            headers=resp_headers,
            error=error,
        )

        # Persist raw bytes to the data lake
        if self.save_raw and self.db is not None and raw_bytes:
            try:
                self.db.save_raw_page(
                    source=self.source_name,
                    url=url,
                    status_code=status_code,
                    content_type=content_type,
                    raw_bytes=raw_bytes,
                    content_hash=result.content_hash,
                )
            except Exception as exc:
                logger.debug("Raw lake save failed: %s", exc)

        return result

    def post(
        self,
        url: str,
        json_body: dict | None = None,
        data: dict | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> ScrapeResult:
        """Rate-limited POST."""
        domain = self._extract_domain(url)
        self._rate_limit(domain)

        headers = self._rotate_headers(extra_headers)
        if json_body is not None:
            headers["Content-Type"] = "application/json"

        error: str | None = None
        raw_bytes = b""
        status_code = 0
        content_type = ""
        resp_headers: dict[str, str] = {}

        try:
            resp = self._session.post(
                url,
                json=json_body,
                data=data,
                headers=headers,
                timeout=self.timeout,
                allow_redirects=True,
            )
            status_code = resp.status_code
            content_type = resp.headers.get("Content-Type", "")
            resp_headers = dict(resp.headers)
            raw_bytes = resp.content

            if resp.status_code not in (200, 201):
                error = f"HTTP {resp.status_code}"

        except Exception as exc:
            error = str(exc)

        result = ScrapeResult(
            url=url,
            source=self.source_name,
            status_code=status_code,
            content_type=content_type,
            raw_bytes=raw_bytes,
            headers=resp_headers,
            error=error,
        )

        if self.save_raw and self.db is not None and raw_bytes:
            try:
                self.db.save_raw_page(
                    source=self.source_name,
                    url=url,
                    status_code=status_code,
                    content_type=content_type,
                    raw_bytes=raw_bytes,
                    content_hash=result.content_hash,
                )
            except Exception as exc:
                logger.debug("Raw lake save failed: %s", exc)

        return result

    # ── Abstract interface ────────────────────────────────────────────────

    @abstractmethod
    def fetch_records(self, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Fetch, parse, and return a list of structured record dicts.
        Each dict should be ready for IntelligenceDB upsert methods.
        """
        ...

    def run(self, **kwargs: Any) -> tuple[int, int]:
        """
        Convenience method: fetch_records + upsert to db.
        Returns (total_fetched, new_records) counts.
        """
        if self.db is None:
            logger.warning("%s.run() called without a db — results not persisted",
                           self.source_name)
        records = self.fetch_records(**kwargs)
        new = 0
        for rec in records:
            try:
                if "politician" in rec:
                    saved = self.db.upsert_congress_trade(rec) if self.db else True
                elif "insider_name" in rec:
                    saved = self.db.upsert_insider_trade(rec) if self.db else True
                else:
                    saved = False
                if saved:
                    new += 1
            except Exception as exc:
                logger.debug("Upsert failed: %s", exc)
        logger.info("%s: fetched %d, new %d", self.source_name, len(records), new)
        return len(records), new
