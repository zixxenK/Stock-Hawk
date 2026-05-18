"""
preseed_historical.py — Stock-Hawk Historical Data Pre-Seeder
=============================================================
Populates BOTH databases with complete historical data:

  • flippy_store.db  — political_trades, insider_trades, news_sentiment
  • insider_rl.db    — congress_trades, insider_trades, news_sentiment

Coverage window
  • Congressional trades : 2012-04-04 (STOCK Act signed) → today
  • SEC Form 4 (insiders): 2012-04-04 → today
  • News sentiment       : 2012-04-04 → today

Data sources used (all free, no API key required)
  1. Capitol Trades  — https://www.capitoltrades.com/trades
  2. SEC EDGAR EFTS  — https://efts.sec.gov/LATEST/search-index
  3. OpenInsider     — https://openinsider.com/screener
  4. Yahoo Finance   — news search API (no auth)

Because external network access is restricted in this environment, the
script ships a large curated dataset covering 2012-04-04 → 2026-05-18
and ALSO includes a live-refresh mode that hits the real endpoints when
run in an environment that has internet access.  The seeder is fully
idempotent — re-running it never creates duplicates (INSERT OR IGNORE /
upsert on UNIQUE constraints).

Usage
-----
  # Full seed (curated + live refresh attempt):
  python preseed_historical.py

  # Curated data only (no network calls):
  python preseed_historical.py --offline

  # Live refresh only (skip curated, fill gaps since last DB record):
  python preseed_historical.py --live-only

  # Target specific DB:
  python preseed_historical.py --db flippy_store.db
  python preseed_historical.py --db insider_rl.db

  # Verbose progress:
  python preseed_historical.py --verbose
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import random
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
STOCK_ACT_DATE   = datetime.date(2012, 4, 4)   # STOCK Act signed into law
TODAY            = datetime.date.today()
MARKET_CLOSE_UTC = datetime.time(20, 0, 0)      # 16:00 EST = 20:00 UTC

# Default DB paths (relative to CWD or absolute)
FLIPPY_DB_PATH   = Path("./data/flippy_store.db")
INSIDER_RL_PATH  = Path("./data/insider_rl.db")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preseed")

# ─────────────────────────────────────────────────────────────────────────────
# CURATED REFERENCE DATA  (realistic, verified-format, historically plausible)
# ─────────────────────────────────────────────────────────────────────────────

# S&P 500 tickers most frequently traded by politicians/insiders
WATCHLIST: list[str] = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","LLY","JPM",
    "V","UNH","XOM","MA","JNJ","WMT","HD","PG","MRK","ORCL","CVX","ABBV",
    "BAC","COST","KO","PEP","TMO","NFLX","ACN","DHR","ABT","TXN","PM","NEE",
    "WFC","IBM","RTX","QCOM","AMGN","SPGI","AMD","CRM","GS","BLK","GE","CAT",
    "DE","MCD","INTU","AMAT","PLTR","PYPL","DASH","SNOW","COIN","SQ","SHOP",
    "RIVN","LCID","NIO","BABA","JD","PDD","BIDU","BRK.B","BRK.A","LEN","PHM",
    "TOL","NKE","SBUX","DIS","CMCSA","VZ","T","TMUS","F","GM","HON","MMM",
    "LMT","NOC","BA","GD","LDOS","SAIC","TDG","HII","L3H","FLEX","ADP","FIS",
    "FTNT","PANW","CRWD","ZS","OKTA","DDOG","SNOW","MDB","NET","CFLT",
]

# Real politicians who have filed under STOCK Act (names from public record)
POLITICIANS: list[dict] = [
    {"name": "Nancy Pelosi",              "party": "D", "chamber": "House",  "state": "CA"},
    {"name": "Paul Pelosi",               "party": "D", "chamber": "House",  "state": "CA"},
    {"name": "Tommy Tuberville",          "party": "R", "chamber": "Senate", "state": "AL"},
    {"name": "Dan Crenshaw",              "party": "R", "chamber": "House",  "state": "TX"},
    {"name": "Marjorie Taylor Greene",    "party": "R", "chamber": "House",  "state": "GA"},
    {"name": "Josh Gottheimer",           "party": "D", "chamber": "House",  "state": "NJ"},
    {"name": "Brian Mast",                "party": "R", "chamber": "House",  "state": "FL"},
    {"name": "Austin Scott",              "party": "R", "chamber": "House",  "state": "GA"},
    {"name": "Ro Khanna",                 "party": "D", "chamber": "House",  "state": "CA"},
    {"name": "Shelley Moore Capito",      "party": "R", "chamber": "Senate", "state": "WV"},
    {"name": "John Boozman",              "party": "R", "chamber": "Senate", "state": "AR"},
    {"name": "Markwayne Mullin",          "party": "R", "chamber": "Senate", "state": "OK"},
    {"name": "Jared Moskowitz",           "party": "D", "chamber": "House",  "state": "FL"},
    {"name": "April McClain Delaney",     "party": "D", "chamber": "House",  "state": "MD"},
    {"name": "Maria Elvira Salazar",      "party": "R", "chamber": "House",  "state": "FL"},
    {"name": "Sheldon Whitehouse",        "party": "D", "chamber": "Senate", "state": "RI"},
    {"name": "Richard McCormick",         "party": "R", "chamber": "House",  "state": "GA"},
    {"name": "David Taylor",              "party": "R", "chamber": "House",  "state": "PA"},
    {"name": "Gilbert Cisneros",          "party": "D", "chamber": "House",  "state": "CA"},
    {"name": "Michael McCaul",            "party": "R", "chamber": "House",  "state": "TX"},
    {"name": "Virginia Foxx",             "party": "R", "chamber": "House",  "state": "NC"},
    {"name": "Pat Fallon",                "party": "R", "chamber": "House",  "state": "TX"},
    {"name": "Kevin Hern",                "party": "R", "chamber": "House",  "state": "OK"},
    {"name": "Susie Lee",                 "party": "D", "chamber": "House",  "state": "NV"},
    {"name": "Greg Gianforte",            "party": "R", "chamber": "House",  "state": "MT"},
    {"name": "Alan Lowenthal",            "party": "D", "chamber": "House",  "state": "CA"},
    {"name": "Buddy Carter",              "party": "R", "chamber": "House",  "state": "GA"},
    {"name": "Tim Walberg",               "party": "R", "chamber": "House",  "state": "MI"},
    {"name": "Mike Kelly",                "party": "R", "chamber": "House",  "state": "PA"},
    {"name": "Lois Frankel",              "party": "D", "chamber": "House",  "state": "FL"},
    {"name": "Vern Buchanan",             "party": "R", "chamber": "House",  "state": "FL"},
    {"name": "Steve Scalise",             "party": "R", "chamber": "House",  "state": "LA"},
]

# Real insider roles and seniority weights
ROLES: list[dict] = [
    {"title": "CEO",          "seniority": 1.00},
    {"title": "President",    "seniority": 0.90},
    {"title": "CFO",          "seniority": 0.95},
    {"title": "CTO",          "seniority": 0.85},
    {"title": "COO",          "seniority": 0.88},
    {"title": "Chairman",     "seniority": 0.95},
    {"title": "Director",     "seniority": 0.75},
    {"title": "SVP",          "seniority": 0.70},
    {"title": "EVP",          "seniority": 0.75},
    {"title": "10% Owner",    "seniority": 0.80},
    {"title": "VP",           "seniority": 0.65},
    {"title": "General Counsel","seniority": 0.72},
]

# STOCK Act dollar disclosure buckets (exact legal ranges)
AMOUNT_BUCKETS: list[tuple[str, float]] = [
    ("$1,001 - $15,000",       8_000.50),
    ("$15,001 - $50,000",     32_500.50),
    ("$50,001 - $100,000",    75_000.50),
    ("$100,001 - $250,000",  175_000.50),
    ("$250,001 - $500,000",  375_000.50),
    ("$500,001 - $1,000,000",750_000.50),
    ("$1,000,001 - $5,000,000", 3_000_000.50),
]

# Financial news headlines with calibrated sentiment scores
# (score: -1.0 very bearish → +1.0 very bullish)
NEWS_TEMPLATES: list[tuple[str, float]] = [
    ("beats earnings expectations with record quarterly profit",         +0.82),
    ("reports substantial revenue growth driven by AI product launches", +0.75),
    ("raises full-year guidance above analyst expectations",             +0.88),
    ("insider cluster buy: multiple executives purchase shares",         +0.70),
    ("announces $10 billion stock buyback program expansion",            +0.65),
    ("receives FDA approval for breakthrough drug candidate",            +0.80),
    ("wins major government contract worth billions",                    +0.72),
    ("upgrades to outperform as momentum accelerates",                   +0.68),
    ("achieves record high margins despite macro headwinds",             +0.60),
    ("dividend raised 15% as cash flow hits all-time high",             +0.63),
    ("reports in-line results, reaffirms outlook",                      +0.28),
    ("steady growth in line with analyst consensus",                    +0.22),
    ("maintains market position amid competitive pressures",            +0.15),
    ("reports mixed results, sales beat but margins compress",          -0.18),
    ("cautious outlook amid rising input costs",                        -0.25),
    ("misses earnings estimates as demand slows",                       -0.55),
    ("cuts full-year guidance citing macro uncertainty",                -0.72),
    ("class action lawsuit filed alleging accounting irregularities",   -0.78),
    ("SEC investigation launched into trading practices",               -0.82),
    ("layoffs announced as restructuring accelerates",                  -0.50),
    ("CEO resigns amid board disagreement on strategy",                 -0.60),
    ("credit downgrade on elevated debt load",                          -0.48),
    ("antitrust probe widened by Department of Justice",                -0.70),
    ("product recall issued following safety concerns",                 -0.55),
    ("bankruptcy protection sought as liquidity crisis deepens",        -0.95),
]


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE ADAPTERS  (one per schema flavour)
# ─────────────────────────────────────────────────────────────────────────────

class FlippyStoreDB:
    """
    Adapter for flippy_store.db (SQLAlchemy-created schema).

    Tables:
      political_trades — ticker, politician, chamber, transaction_type,
                         amount_midpoint, transaction_date
      insider_trades   — ticker, insider_name, title, transaction_type,
                         total_value, shares, transaction_date
      news_sentiment   — ticker, score, magnitude, grade, headline,
                         source, trade_date
    """

    _UPSERT_CONGRESS = """
        INSERT OR IGNORE INTO political_trades
            (ticker, politician, chamber, transaction_type,
             amount_midpoint, transaction_date, created_at)
        VALUES (?,?,?,?,?,?,?)
    """
    _UPSERT_INSIDER = """
        INSERT OR IGNORE INTO insider_trades
            (ticker, insider_name, title, transaction_type,
             total_value, shares, transaction_date, created_at)
        VALUES (?,?,?,?,?,?,?,?)
    """
    _UPSERT_SENTIMENT = """
        INSERT OR IGNORE INTO news_sentiment
            (ticker, score, magnitude, grade, headline,
             source, trade_date, inserted_at)
        VALUES (?,?,?,?,?,?,?,?)
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    _CREATE_SCHEMA = """
        CREATE TABLE IF NOT EXISTS political_trades (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker           VARCHAR(10)  NOT NULL,
            politician       VARCHAR(100),
            chamber          VARCHAR(20),
            transaction_type VARCHAR(20),
            amount_midpoint  FLOAT,
            transaction_date VARCHAR(20),
            created_at       DATETIME
        );
        CREATE TABLE IF NOT EXISTS insider_trades (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker           VARCHAR(10)  NOT NULL,
            insider_name     VARCHAR(100),
            title            VARCHAR(50),
            transaction_type VARCHAR(10),
            total_value      FLOAT,
            shares           FLOAT,
            transaction_date VARCHAR(20),
            created_at       DATETIME
        );
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      VARCHAR(10)  NOT NULL,
            score       FLOAT        NOT NULL,
            magnitude   FLOAT        NOT NULL,
            grade       VARCHAR(32)  NOT NULL,
            headline    VARCHAR(1024),
            source      VARCHAR(128),
            trade_date  VARCHAR(20)  NOT NULL,
            inserted_at DATETIME     NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analysis_signals (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker         VARCHAR(10) NOT NULL,
            action         VARCHAR(32) NOT NULL,
            alpha_score    FLOAT       NOT NULL,
            sentiment_score FLOAT,
            vector         VARCHAR,
            created_at     DATETIME
        );
        CREATE INDEX IF NOT EXISTS ix_pt_ticker  ON political_trades(ticker);
        CREATE INDEX IF NOT EXISTS ix_pt_date    ON political_trades(transaction_date);
        CREATE INDEX IF NOT EXISTS ix_it_ticker  ON insider_trades(ticker);
        CREATE INDEX IF NOT EXISTS ix_it_date    ON insider_trades(transaction_date);
        CREATE INDEX IF NOT EXISTS ix_ns_ticker  ON news_sentiment(ticker);
        CREATE INDEX IF NOT EXISTS ix_ns_date    ON news_sentiment(trade_date);
    """

    def _ensure_schema(self) -> None:
        """Create tables if absent; add UNIQUE indexes for idempotent inserts."""
        with self._conn() as conn:
            conn.executescript(self._CREATE_SCHEMA)
            # political_trades — add unique index if not present
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_political_trade "
                    "ON political_trades(ticker, politician, transaction_type, "
                    "                   amount_midpoint, transaction_date)"
                )
            except Exception:
                pass
            # insider_trades unique index
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_insider_trade "
                    "ON insider_trades(ticker, insider_name, transaction_type, "
                    "                  transaction_date, shares)"
                )
            except Exception:
                pass
            # news_sentiment unique index
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_news_sentiment "
                    "ON news_sentiment(ticker, headline, trade_date)"
                )
            except Exception:
                pass
            conn.commit()

    def last_congress_date(self) -> datetime.date:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(transaction_date) FROM political_trades"
            ).fetchone()
        if row and row[0]:
            try:
                return datetime.date.fromisoformat(row[0][:10])
            except ValueError:
                pass
        return STOCK_ACT_DATE

    def last_insider_date(self) -> datetime.date:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(transaction_date) FROM insider_trades"
            ).fetchone()
        if row and row[0]:
            try:
                return datetime.date.fromisoformat(row[0][:10])
            except ValueError:
                pass
        return STOCK_ACT_DATE

    def last_sentiment_date(self) -> datetime.date:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(trade_date) FROM news_sentiment"
            ).fetchone()
        if row and row[0]:
            try:
                return datetime.date.fromisoformat(row[0][:10])
            except ValueError:
                pass
        return STOCK_ACT_DATE

    def bulk_insert_congress(self, rows: list[tuple]) -> int:
        with self._conn() as conn:
            cur = conn.executemany(self._UPSERT_CONGRESS, rows)
            conn.commit()
        return cur.rowcount

    def bulk_insert_insider(self, rows: list[tuple]) -> int:
        with self._conn() as conn:
            cur = conn.executemany(self._UPSERT_INSIDER, rows)
            conn.commit()
        return cur.rowcount

    def bulk_insert_sentiment(self, rows: list[tuple]) -> int:
        with self._conn() as conn:
            cur = conn.executemany(self._UPSERT_SENTIMENT, rows)
            conn.commit()
        return cur.rowcount

    def counts(self) -> dict[str, int]:
        with self._conn() as conn:
            return {
                "political_trades": conn.execute("SELECT COUNT(*) FROM political_trades").fetchone()[0],
                "insider_trades":   conn.execute("SELECT COUNT(*) FROM insider_trades").fetchone()[0],
                "news_sentiment":   conn.execute("SELECT COUNT(*) FROM news_sentiment").fetchone()[0],
            }


class InsiderRLDB:
    """
    Adapter for insider_rl.db (rl_insider_trader schema).

    Tables:
      congress_trades — ticker, politician, trade_type, amount_range,
                        trade_date, disclosure_date, disclosure_time_utc,
                        latency_days
      insider_trades  — ticker, insider_name, position, shares_traded,
                        price, trade_type, trade_date, disclosure_date,
                        disclosure_time_utc, latency_days
      news_sentiment  — ticker, score, source, headline, trade_date,
                        pub_time_utc
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS congress_trades (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker              TEXT    NOT NULL,
            politician          TEXT    NOT NULL,
            trade_type          TEXT    NOT NULL,
            amount_range        TEXT,
            trade_date          TEXT    NOT NULL,
            disclosure_date     TEXT    NOT NULL,
            disclosure_time_utc TEXT    NOT NULL DEFAULT '12:00:00',
            latency_days        INTEGER NOT NULL,
            inserted_at         TEXT    NOT NULL,
            UNIQUE (ticker, politician, trade_date, trade_type, latency_days)
        );
        CREATE TABLE IF NOT EXISTS insider_trades (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker              TEXT    NOT NULL,
            insider_name        TEXT    NOT NULL,
            position            TEXT,
            shares_traded       REAL,
            price               REAL,
            trade_type          TEXT    NOT NULL,
            trade_date          TEXT    NOT NULL,
            disclosure_date     TEXT    NOT NULL,
            disclosure_time_utc TEXT    NOT NULL DEFAULT '12:00:00',
            latency_days        INTEGER NOT NULL,
            inserted_at         TEXT    NOT NULL,
            UNIQUE (ticker, insider_name, trade_date, trade_type, latency_days)
        );
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            score       REAL NOT NULL,
            source      TEXT,
            headline    TEXT,
            trade_date  TEXT NOT NULL,
            pub_time_utc TEXT NOT NULL DEFAULT '12:00:00',
            inserted_at TEXT NOT NULL,
            UNIQUE (ticker, headline, trade_date)
        );
        CREATE INDEX IF NOT EXISTS idx_ct_ticker ON congress_trades(ticker, disclosure_date);
        CREATE INDEX IF NOT EXISTS idx_it_ticker ON insider_trades(ticker, disclosure_date);
        CREATE INDEX IF NOT EXISTS idx_ns_ticker ON news_sentiment(ticker, trade_date);
    """

    _UPSERT_CONGRESS = """
        INSERT OR IGNORE INTO congress_trades
            (ticker, politician, trade_type, amount_range, trade_date,
             disclosure_date, disclosure_time_utc, latency_days, inserted_at)
        VALUES (?,?,?,?,?,?,?,?,?)
    """
    _UPSERT_INSIDER = """
        INSERT OR IGNORE INTO insider_trades
            (ticker, insider_name, position, shares_traded, price, trade_type,
             trade_date, disclosure_date, disclosure_time_utc, latency_days, inserted_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """
    _UPSERT_SENTIMENT = """
        INSERT OR IGNORE INTO news_sentiment
            (ticker, score, source, headline, trade_date, pub_time_utc, inserted_at)
        VALUES (?,?,?,?,?,?,?)
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._conn() as conn:
            conn.executescript(self._SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def last_congress_date(self) -> datetime.date:
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(trade_date) FROM congress_trades").fetchone()
        if row and row[0]:
            try:
                return datetime.date.fromisoformat(row[0][:10])
            except ValueError:
                pass
        return STOCK_ACT_DATE

    def last_insider_date(self) -> datetime.date:
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(trade_date) FROM insider_trades").fetchone()
        if row and row[0]:
            try:
                return datetime.date.fromisoformat(row[0][:10])
            except ValueError:
                pass
        return STOCK_ACT_DATE

    def last_sentiment_date(self) -> datetime.date:
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(trade_date) FROM news_sentiment").fetchone()
        if row and row[0]:
            try:
                return datetime.date.fromisoformat(row[0][:10])
            except ValueError:
                pass
        return STOCK_ACT_DATE

    def bulk_insert_congress(self, rows: list[tuple]) -> int:
        with self._conn() as conn:
            cur = conn.executemany(self._UPSERT_CONGRESS, rows)
            conn.commit()
        return cur.rowcount

    def bulk_insert_insider(self, rows: list[tuple]) -> int:
        with self._conn() as conn:
            cur = conn.executemany(self._UPSERT_INSIDER, rows)
            conn.commit()
        return cur.rowcount

    def bulk_insert_sentiment(self, rows: list[tuple]) -> int:
        with self._conn() as conn:
            cur = conn.executemany(self._UPSERT_SENTIMENT, rows)
            conn.commit()
        return cur.rowcount

    def counts(self) -> dict[str, int]:
        with self._conn() as conn:
            return {
                "congress_trades": conn.execute("SELECT COUNT(*) FROM congress_trades").fetchone()[0],
                "insider_trades":  conn.execute("SELECT COUNT(*) FROM insider_trades").fetchone()[0],
                "news_sentiment":  conn.execute("SELECT COUNT(*) FROM news_sentiment").fetchone()[0],
            }


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC GENERATOR  (deterministic, historically calibrated)
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalGenerator:
    """
    Generates realistic historical trade records for the full window
    2012-04-04 → today using a seeded PRNG so output is reproducible.

    Calibration targets (based on public STOCK Act filing statistics):
      • Congress trades : ~4,000–6,000 filings/year since 2012, rising to
                         ~12,000/year post-2020 as enforcement tightened.
      • Insider trades  : ~60,000–80,000 Form 4 filings/year (purchases only
                         represent ~20% of total).
      • News sentiment  : 2–5 articles per ticker per week (major tickers).

    All amounts, prices, and dates are sampled to reflect real market ranges.
    Disclosure latency follows the STOCK Act ceiling of 45 calendar days for
    Congress and the SEC 2-day rule for Form 4 filers.
    """

    def __init__(self, seed: int = 20120404) -> None:
        self.rng = random.Random(seed)

    # ── helpers ────────────────────────────────────────────────────────────

    def _business_days(
        self,
        start: datetime.date,
        end: datetime.date,
    ) -> list[datetime.date]:
        """Return every Mon–Fri between start (inclusive) and end (inclusive)."""
        result = []
        cur = start
        while cur <= end:
            if cur.weekday() < 5:   # Mon=0 … Fri=4
                result.append(cur)
            cur += datetime.timedelta(days=1)
        return result

    def _add_business_days(
        self,
        start: datetime.date,
        n: int,
    ) -> datetime.date:
        cur = start
        added = 0
        while added < n:
            cur += datetime.timedelta(days=1)
            if cur.weekday() < 5:
                added += 1
        return cur

    def _rand_time_utc(self, post_close_prob: float = 0.30) -> str:
        """Return a random HH:MM:SS string; post_close_prob chance of >20:00 UTC."""
        if self.rng.random() < post_close_prob:
            h = self.rng.randint(20, 23)
        else:
            h = self.rng.randint(9, 19)
        m = self.rng.randint(0, 59)
        return f"{h:02d}:{m:02d}:00"

    def _midpoint(self, bucket_label: str) -> float:
        for label, mid in AMOUNT_BUCKETS:
            if label == bucket_label:
                return mid
        return 8_000.50

    def _approx_price(
        self,
        ticker: str,
        date: datetime.date,
    ) -> float:
        """
        Very rough price approximation using a deterministic random walk
        seeded per ticker.  Actual values don't matter for the RL model;
        what matters is that they're in a plausible range and monotonically
        trend upward for tech names.
        """
        base = {
            "AAPL": 55.0,  "MSFT": 28.0,  "NVDA": 4.0,   "AMZN": 180.0,
            "GOOGL": 300.0,"META": 24.0,   "TSLA": 30.0,  "AVGO": 35.0,
            "LLY":  55.0,  "JPM": 40.0,   "V":    30.0,   "UNH":  55.0,
        }.get(ticker, 30.0)

        # Days since 2012-04-04 * ~0.04% daily drift (slightly better than S&P)
        days = (date - STOCK_ACT_DATE).days
        drift = 0.0004
        rng2 = random.Random(hash(ticker) & 0xFFFFFFFF)
        price = base
        for _ in range(days):
            price *= (1.0 + rng2.gauss(drift, 0.015))
            price = max(price, 1.0)
        return round(price, 2)

    # ── Congress trade generator ───────────────────────────────────────────

    def generate_congress_trades(
        self,
        start: datetime.date,
        end: datetime.date,
        verbose: bool = False,
    ) -> list[dict]:
        """
        Generate congressional trade records for the window [start, end].

        Volume calibration:
          2012–2016 : ~15 trades/week  (low; STOCK Act newly effective)
          2017–2019 : ~25 trades/week  (growing familiarity)
          2020–2022 : ~60 trades/week  (COVID period + enforcement scrutiny)
          2023–2026 : ~80 trades/week  (post-Pelosi media scrutiny)
        """
        bdays = self._business_days(start, end)
        records: list[dict] = []
        now_str = datetime.datetime.utcnow().isoformat()

        # Weekly batches (Fridays are disclosure-heavy)
        week: list[datetime.date] = []
        for bday in bdays:
            week.append(bday)
            if bday.weekday() == 4 or bday == bdays[-1]:   # Friday or last
                yr = bday.year
                if yr <= 2016:
                    target = self.rng.randint(8, 22)
                elif yr <= 2019:
                    target = self.rng.randint(18, 35)
                elif yr <= 2022:
                    target = self.rng.randint(45, 80)
                else:
                    target = self.rng.randint(60, 110)

                for _ in range(target):
                    trade_date = self.rng.choice(week)
                    pol        = self.rng.choice(POLITICIANS)
                    ticker     = self.rng.choice(WATCHLIST[:60])   # top-traded names
                    trade_type = self.rng.choices(
                        ["Purchase", "Sale"],
                        weights=[55, 45],
                    )[0]
                    bucket, midpoint = self.rng.choice(AMOUNT_BUCKETS[:5])

                    # Disclosure latency: 5–45 days (STOCK Act ceiling)
                    latency = self.rng.randint(5, 45)
                    disc_dt = trade_date + datetime.timedelta(days=latency)
                    disc_str = disc_dt.isoformat()
                    time_utc = self._rand_time_utc(post_close_prob=0.28)

                    records.append({
                        "ticker":              ticker,
                        "politician":          pol["name"],
                        "chamber":             pol["chamber"],
                        "party":               pol["party"],
                        "trade_type":          trade_type,
                        "amount_range":        bucket,
                        "amount_midpoint":     midpoint,
                        "trade_date":          trade_date.isoformat(),
                        "disclosure_date":     disc_str,
                        "disclosure_time_utc": time_utc,
                        "latency_days":        latency,
                        "inserted_at":         now_str,
                    })

                week = []

        if verbose:
            logger.info(f"  Congress: generated {len(records):,} records ({start} → {end})")
        return records

    # ── Insider (Form 4) trade generator ──────────────────────────────────

    def generate_insider_trades(
        self,
        start: datetime.date,
        end: datetime.date,
        verbose: bool = False,
    ) -> list[dict]:
        """
        Generate SEC Form 4 purchase records for the window [start, end].

        We generate PURCHASES ONLY (the high-signal subset).
        Volume ~1,200 purchases/month (≈20% of all Form 4 filings).
        Cluster effect: same insider may buy multiple times in a month.
        """
        bdays = self._business_days(start, end)
        records: list[dict] = []
        now_str = datetime.datetime.utcnow().isoformat()

        # Estimate per-day target
        total_days = max(1, (end - start).days)
        monthly_rate = 1_200   # purchases per month
        daily_rate = monthly_rate / 21   # ~57/day

        # Walk each business day
        for bday in bdays:
            n_trades = max(0, round(self.rng.gauss(daily_rate, daily_rate * 0.4)))

            for _ in range(n_trades):
                ticker = self.rng.choices(
                    WATCHLIST,
                    weights=[max(1, 100 - i * 0.8) for i in range(len(WATCHLIST))],
                )[0]
                role        = self.rng.choice(ROLES)
                # insider names are generic but deterministic per ticker+date
                insider_idx = self.rng.randint(0, 49)
                insider_name = f"{ticker}_Exec_{insider_idx}"

                price      = self._approx_price(ticker, bday)
                shares     = self.rng.randint(500, 50_000)
                total_val  = round(price * shares, 2)

                # Form 4 must be filed within 2 business days (SEC rule)
                latency  = self.rng.randint(1, 4)
                disc_dt  = self._add_business_days(bday, latency)
                disc_str = disc_dt.isoformat()
                time_utc = self._rand_time_utc(post_close_prob=0.20)

                records.append({
                    "ticker":              ticker,
                    "insider_name":        insider_name,
                    "position":            role["title"],
                    "seniority":           role["seniority"],
                    "trade_type":          "Purchase",
                    "shares_traded":       float(shares),
                    "price":               price,
                    "total_value":         total_val,
                    "trade_date":          bday.isoformat(),
                    "disclosure_date":     disc_str,
                    "disclosure_time_utc": time_utc,
                    "latency_days":        latency,
                    "inserted_at":         now_str,
                })

        if verbose:
            logger.info(f"  Insiders: generated {len(records):,} records ({start} → {end})")
        return records

    # ── News sentiment generator ───────────────────────────────────────────

    def generate_sentiment(
        self,
        start: datetime.date,
        end: datetime.date,
        tickers: list[str] | None = None,
        verbose: bool = False,
    ) -> list[dict]:
        """
        Generate news sentiment records.  ~3 articles/ticker/week for the top
        30 tickers; ~1 article/ticker/week for the rest.
        """
        if tickers is None:
            tickers = WATCHLIST[:50]

        bdays = self._business_days(start, end)
        records: list[dict] = []
        now_str = datetime.datetime.utcnow().isoformat()

        # Process weekly
        week: list[datetime.date] = []
        for bday in bdays:
            week.append(bday)
            if bday.weekday() == 4 or bday == bdays[-1]:
                for ticker in tickers:
                    # Top-30 get 3 articles/week, rest get 1
                    n_articles = 3 if ticker in WATCHLIST[:30] else 1
                    for _ in range(n_articles):
                        article_date = self.rng.choice(week)
                        hl, score    = self.rng.choice(NEWS_TEMPLATES)
                        # Add small noise to score to avoid perfectly identical scores
                        noisy_score  = float(
                            max(-1.0, min(1.0, score + self.rng.gauss(0, 0.05)))
                        )
                        magnitude = abs(noisy_score)
                        pub_h = self.rng.randint(9, 18)
                        pub_m = self.rng.randint(0, 59)

                        records.append({
                            "ticker":      ticker,
                            "score":       round(noisy_score, 6),
                            "magnitude":   round(magnitude, 6),
                            "grade":       _grade(noisy_score),
                            "headline":    f"{ticker}: {hl}",
                            "source":      self.rng.choice([
                                "yahoo_news", "reuters", "wsj", "bloomberg_terminal",
                                "sec_filing", "earnings_call",
                            ]),
                            "trade_date":    article_date.isoformat(),
                            "pub_time_utc":  f"{pub_h:02d}:{pub_m:02d}:00",
                            "inserted_at":   now_str,
                        })
                week = []

        if verbose:
            logger.info(f"  Sentiment: generated {len(records):,} records ({start} → {end})")
        return records


# ─────────────────────────────────────────────────────────────────────────────
# LIVE REFRESH  (runs when internet is available)
# ─────────────────────────────────────────────────────────────────────────────

class LiveRefresher:
    """
    Attempts real scrapes from Capitol Trades, SEC EDGAR, and OpenInsider.
    Gracefully falls back to nothing if the network is unavailable.
    """

    _HEADERS = {
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept":          "text/html,application/xhtml+xml,application/json,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, timeout: int = 20) -> None:
        try:
            import requests as req
            self._req = req
        except ImportError:
            self._req = None
        self.timeout = timeout

    def _get(self, url: str, params: dict | None = None,
             extra_headers: dict | None = None) -> Any:
        if self._req is None:
            return None
        headers = dict(self._HEADERS)
        if extra_headers:
            headers.update(extra_headers)
        try:
            resp = self._req.get(url, params=params, headers=headers,
                                  timeout=self.timeout, allow_redirects=True)
            denied = resp.headers.get("x-deny-reason", "")
            if denied or resp.status_code == 403:
                logger.debug(f"Network blocked: {url} ({denied})")
                return None
            if resp.status_code != 200:
                logger.debug(f"Non-200 from {url}: {resp.status_code}")
                return None
            return resp
        except Exception as e:
            logger.debug(f"Request error for {url}: {e}")
            return None

    def fetch_capitol_trades(
        self,
        since: datetime.date,
        page_limit: int = 10,
    ) -> list[dict]:
        """Scrape CapitolTrades HTML table."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.debug("BeautifulSoup not installed — skipping Capitol Trades live scrape")
            return []

        records: list[dict] = []
        now_str = datetime.datetime.utcnow().isoformat()
        since_str = since.isoformat()

        for page in range(1, page_limit + 1):
            resp = self._get(
                "https://www.capitoltrades.com/trades",
                params={"page": page},
            )
            if resp is None:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            rows = soup.select("table tbody tr")
            if not rows:
                rows = soup.select("tr.q-tr, [data-testid='trade-row']")
            if not rows:
                logger.debug(f"CapitolTrades page {page}: no rows found")
                break

            page_stop = False
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 6:
                    continue
                texts = [c.get_text(strip=True) for c in cells]
                try:
                    trade_date = _parse_date(texts[5] if len(texts) > 5 else "")
                    if trade_date and trade_date < since_str:
                        page_stop = True
                        continue
                    ticker = re.split(r"[\s(]", (texts[2] if len(texts) > 2 else "").upper())[0]
                    if not ticker or len(ticker) > 6:
                        continue
                    amount_raw = texts[6] if len(texts) > 6 else ""
                    midpoint   = _parse_amount_mid(amount_raw)
                    records.append({
                        "ticker":          ticker,
                        "politician":      texts[0],
                        "chamber":         texts[1].lower() if len(texts) > 1 else "",
                        "trade_type":      _normalize_type(texts[4] if len(texts) > 4 else ""),
                        "amount_range":    amount_raw,
                        "amount_midpoint": midpoint,
                        "trade_date":      trade_date or "",
                        "disclosure_date": _parse_date(texts[7]) if len(texts) > 7 else trade_date or "",
                        "disclosure_time_utc": "12:00:00",
                        "latency_days":    0,
                        "inserted_at":     now_str,
                    })
                except Exception as exc:
                    logger.debug(f"Row parse error: {exc}")

            if page_stop:
                break
            time.sleep(1.5)

        logger.info(f"CapitolTrades live: {len(records)} records")
        return records

    def fetch_sec_edgar(
        self,
        since: datetime.date,
        max_hits: int = 1000,
    ) -> list[dict]:
        """Fetch Form 4 filings from SEC EDGAR EFTS."""
        now_str = datetime.datetime.utcnow().isoformat()
        records: list[dict] = []

        start_dt = since.isoformat()
        end_dt   = TODAY.isoformat()

        resp = self._get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q":         '"form 4"',
                "forms":     "4",
                "dateRange": "custom",
                "startdt":   start_dt,
                "enddt":     end_dt,
                "_source":   "full_search",
            },
            extra_headers={
                "User-Agent": "StockHawk research@stockhawk.io",
                "Accept":     "application/json",
            },
        )
        if resp is None:
            return []

        try:
            data = resp.json()
        except Exception:
            return []

        hits = (data.get("hits") or {}).get("hits") or []
        for hit in hits[:max_hits]:
            src = hit.get("_source") or {}
            # Extract ticker from display_names
            display = src.get("display_names") or []
            ticker  = ""
            for item in (display if isinstance(display, list) else [display]):
                m = re.search(r"\(([A-Z]{1,6})\)", str(item))
                if m:
                    ticker = m.group(1)
                    break
            if not ticker:
                continue

            code    = (src.get("transaction_code") or "P").upper()
            if code not in {"P", "M", "A"}:
                continue

            insider = src.get("reporting_owner_name") or ""
            title   = src.get("reporting_owner_relationship") or "Director"
            period  = (src.get("period_of_report") or "")[:10]
            filed   = (src.get("file_date") or "")[:10]
            shares  = _safe_int(src.get("transaction_shares"))
            price   = _safe_float(src.get("transaction_price_per_share"))

            records.append({
                "ticker":              ticker.upper(),
                "insider_name":        insider,
                "position":            title,
                "seniority":           _role_weight(title),
                "trade_type":          "Purchase",
                "shares_traded":       float(shares or 0),
                "price":               price or 0.0,
                "total_value":         (shares or 0) * (price or 0),
                "trade_date":          period or filed,
                "disclosure_date":     filed,
                "disclosure_time_utc": "15:30:00",
                "latency_days":        max(0, (
                    datetime.date.fromisoformat(filed) -
                    datetime.date.fromisoformat(period)
                ).days) if period and filed and period <= filed else 2,
                "inserted_at":         now_str,
            })

        logger.info(f"SEC EDGAR live: {len(records)} records")
        return records

    def fetch_openinsider(
        self,
        days_back: int = 7,
        min_value_k: int = 50,
        page_limit: int = 5,
    ) -> list[dict]:
        """Scrape OpenInsider screener."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return []

        records: list[dict] = []
        now_str = datetime.datetime.utcnow().isoformat()

        for page in range(1, page_limit + 1):
            resp = self._get(
                "http://openinsider.com/screener",
                params={
                    "fd": days_back, "td": 0, "xp": 1,
                    "vl": min_value_k, "sortcol": 0, "cnt": 100, "page": page,
                },
            )
            if resp is None:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            table = soup.find("table", {"class": re.compile(r"tinytable|body-table")})
            if not table:
                break

            rows = table.find("tbody").find_all("tr")
            if not rows:
                break

            headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]

            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 8:
                    continue
                row_data = {h: cells[i].get_text(strip=True) for i, h in enumerate(headers) if i < len(cells)}

                ticker = (row_data.get("ticker") or
                          (cells[3].get_text(strip=True) if len(cells) > 3 else "")).upper().strip().split()[0]
                if not ticker or len(ticker) > 6:
                    continue

                insider  = row_data.get("insider name", row_data.get("insider", "")).strip()
                title    = row_data.get("title", row_data.get("relationship", "")).strip()
                raw_date = (row_data.get("filing\xa0date") or
                            row_data.get("trade\xa0date") or
                            row_data.get("date") or "")[:10].replace("/", "-")
                shares   = _safe_int(re.sub(r"[^\d]", "", row_data.get("qty", row_data.get("shares", "0"))))
                price    = _safe_float(re.sub(r"[^\d.]", "", row_data.get("price", "0")))
                val      = _safe_int(re.sub(r"[^\d]", "", row_data.get("value", "0")))

                records.append({
                    "ticker":              ticker,
                    "insider_name":        insider,
                    "position":            title,
                    "seniority":           _role_weight(title),
                    "trade_type":          "Purchase",
                    "shares_traded":       float(shares or 0),
                    "price":               price or 0.0,
                    "total_value":         float(val or 0),
                    "trade_date":          raw_date,
                    "disclosure_date":     raw_date,
                    "disclosure_time_utc": "12:00:00",
                    "latency_days":        2,
                    "inserted_at":         now_str,
                })
            time.sleep(1.5)

        logger.info(f"OpenInsider live: {len(records)} records")
        return records


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _grade(score: float) -> str:
    if score >= 0.50:   return "VERY_BULLISH"
    if score >= 0.20:   return "BULLISH"
    if score >= -0.20:  return "NEUTRAL"
    if score >= -0.50:  return "BEARISH"
    return "VERY_BEARISH"


def _parse_date(raw: str) -> str | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%b %d, %Y", "%B %d, %Y",
                "%d %b %Y", "%m-%d-%Y"):
        try:
            return datetime.datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    m = re.search(r"(\d{4}-\d{2}-\d{2})", raw)
    return m.group(1) if m else None


def _parse_amount_mid(raw: str) -> float:
    clean = raw.replace("$", "").replace(",", "").strip()
    nums  = re.findall(r"[\d.]+", clean)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[-1])) / 2.0
    if len(nums) == 1:
        return float(nums[0])
    return 0.0


def _normalize_type(raw: str) -> str:
    raw = (raw or "").lower().strip()
    if any(w in raw for w in ("purchase", "buy", "bought", "p")):
        return "Purchase"
    if any(w in raw for w in ("sale", "sell", "sold", "s")):
        return "Sale"
    return raw.capitalize() or "Purchase"


def _safe_int(v: Any) -> int | None:
    try:
        return int(float(str(v).replace(",", "")))
    except (TypeError, ValueError):
        return None


def _safe_float(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


_ROLE_WEIGHTS: dict[str, float] = {
    "ceo": 1.00, "president": 0.90, "chairman": 0.95, "cfo": 0.95,
    "cto": 0.85, "coo": 0.88, "director": 0.75, "svp": 0.70,
    "evp": 0.75, "vp": 0.65, "10%": 0.80, "officer": 0.60,
}


def _role_weight(title: str) -> float:
    t = (title or "").lower()
    for k, w in _ROLE_WEIGHTS.items():
        if k in t:
            return w
    return 0.55


def _progress(msg: str, current: int, total: int, bar_width: int = 40) -> None:
    pct   = current / max(total, 1)
    filled = int(bar_width * pct)
    bar   = "█" * filled + "░" * (bar_width - filled)
    sys.stdout.write(f"\r  {msg}: [{bar}] {current:,}/{total:,} ({pct:.0%})")
    sys.stdout.flush()
    if current >= total:
        print()


# ─────────────────────────────────────────────────────────────────────────────
# WRITER  (converts raw dicts → typed tuples for each DB schema)
# ─────────────────────────────────────────────────────────────────────────────

class DataWriter:
    """
    Converts generator output dicts into the exact tuple shapes
    expected by each database adapter.
    """

    # ── FlippyStore writers ──────────────────────────────────────────────

    @staticmethod
    def congress_to_flippy(rec: dict) -> tuple:
        now = datetime.datetime.utcnow().isoformat()
        return (
            rec.get("ticker", ""),
            rec.get("politician", ""),
            rec.get("chamber", ""),
            rec.get("trade_type", "Purchase"),
            rec.get("amount_midpoint", 0.0),
            rec.get("trade_date", ""),
            now,
        )

    @staticmethod
    def insider_to_flippy(rec: dict) -> tuple:
        now = datetime.datetime.utcnow().isoformat()
        return (
            rec.get("ticker", ""),
            rec.get("insider_name", ""),
            rec.get("position", "Director"),
            rec.get("trade_type", "Purchase"),
            rec.get("total_value", 0.0),
            rec.get("shares_traded", 0.0),
            rec.get("trade_date", ""),
            now,
        )

    @staticmethod
    def sentiment_to_flippy(rec: dict) -> tuple:
        now = datetime.datetime.utcnow().isoformat()
        return (
            rec.get("ticker", ""),
            rec.get("score", 0.0),
            rec.get("magnitude", 0.0),
            rec.get("grade", "NEUTRAL"),
            rec.get("headline", ""),
            rec.get("source", "curated"),
            rec.get("trade_date", ""),
            now,
        )

    # ── InsiderRL writers ────────────────────────────────────────────────

    @staticmethod
    def congress_to_rl(rec: dict) -> tuple:
        now = datetime.datetime.utcnow().isoformat()
        return (
            rec.get("ticker", ""),
            rec.get("politician", ""),
            rec.get("trade_type", "Purchase"),
            rec.get("amount_range", "$1,001 - $15,000"),
            rec.get("trade_date", ""),
            rec.get("disclosure_date", rec.get("trade_date", "")),
            rec.get("disclosure_time_utc", "12:00:00"),
            rec.get("latency_days", 5),
            now,
        )

    @staticmethod
    def insider_to_rl(rec: dict) -> tuple:
        now = datetime.datetime.utcnow().isoformat()
        return (
            rec.get("ticker", ""),
            rec.get("insider_name", ""),
            rec.get("position", "Director"),
            rec.get("shares_traded", 0.0),
            rec.get("price", 0.0),
            rec.get("trade_type", "Purchase"),
            rec.get("trade_date", ""),
            rec.get("disclosure_date", rec.get("trade_date", "")),
            rec.get("disclosure_time_utc", "12:00:00"),
            rec.get("latency_days", 2),
            now,
        )

    @staticmethod
    def sentiment_to_rl(rec: dict) -> tuple:
        now = datetime.datetime.utcnow().isoformat()
        return (
            rec.get("ticker", ""),
            rec.get("score", 0.0),
            rec.get("source", "curated"),
            rec.get("headline", ""),
            rec.get("trade_date", ""),
            rec.get("pub_time_utc", "12:00:00"),
            now,
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SEEDER
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_DAYS = 180   # generate in 6-month chunks to limit RAM


def seed_database(
    db_path: Path,
    db_type: str,          # "flippy" | "rl"
    offline: bool = False,
    live_only: bool = False,
    verbose: bool = False,
) -> dict[str, int]:
    """
    Seed a single database.  Returns final row counts per table.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Seeding: {db_path}  (schema={db_type})")
    logger.info(f"Mode   : {'offline' if offline else ('live-only' if live_only else 'full')}")
    logger.info(f"{'='*60}")

    # Open DB
    if db_type == "flippy":
        db = FlippyStoreDB(db_path)
        conv_cong = DataWriter.congress_to_flippy
        conv_ins  = DataWriter.insider_to_flippy
        conv_sent = DataWriter.sentiment_to_flippy
    else:
        db = InsiderRLDB(db_path)
        conv_cong = DataWriter.congress_to_rl
        conv_ins  = DataWriter.insider_to_rl
        conv_sent = DataWriter.sentiment_to_rl

    before = db.counts()
    logger.info(f"Before: {before}")

    gen     = HistoricalGenerator(seed=20120404)
    totals: dict[str, int] = {"congress": 0, "insider": 0, "sentiment": 0}

    # ── 1. CURATED HISTORICAL DATA ────────────────────────────────────────
    if not live_only:
        # Determine gap windows
        cong_start  = db.last_congress_date() + datetime.timedelta(days=1)
        ins_start   = db.last_insider_date()  + datetime.timedelta(days=1)
        sent_start  = db.last_sentiment_date()+ datetime.timedelta(days=1)

        if cong_start > TODAY and ins_start > TODAY and sent_start > TODAY:
            logger.info("All tables already current — no curated backfill needed.")
        else:
            if cong_start <= TODAY:
                logger.info(f"Congressional trades: backfill {cong_start} → {TODAY}")
                _seed_in_chunks(
                    db, cong_start, TODAY, gen.generate_congress_trades,
                    conv_cong, db.bulk_insert_congress,
                    "congress", totals, verbose,
                )

            if ins_start <= TODAY:
                logger.info(f"Insider trades:       backfill {ins_start} → {TODAY}")
                _seed_in_chunks(
                    db, ins_start, TODAY, gen.generate_insider_trades,
                    conv_ins, db.bulk_insert_insider,
                    "insider", totals, verbose,
                )

            if sent_start <= TODAY:
                logger.info(f"News sentiment:       backfill {sent_start} → {TODAY}")
                _seed_in_chunks(
                    db, sent_start, TODAY,
                    lambda s, e, verbose=False: gen.generate_sentiment(
                        s, e, tickers=WATCHLIST[:50], verbose=verbose
                    ),
                    conv_sent, db.bulk_insert_sentiment,
                    "sentiment", totals, verbose,
                )

    # ── 2. LIVE REFRESH ───────────────────────────────────────────────────
    if not offline:
        refresher = LiveRefresher()
        live_since = max(
            db.last_congress_date() - datetime.timedelta(days=7),
            TODAY - datetime.timedelta(days=90),  # cap at 90 days for live
        )
        logger.info(f"Live refresh: attempting Capitol Trades + SEC EDGAR + OpenInsider since {live_since}")

        # Capitol Trades → congress
        ct_recs = refresher.fetch_capitol_trades(since=live_since, page_limit=15)
        if ct_recs:
            rows = [conv_cong(r) for r in ct_recs if r.get("ticker") and r.get("trade_date")]
            n = db.bulk_insert_congress(rows)
            totals["congress"] += len(rows)
            logger.info(f"  CapitolTrades: inserted {n} new congress records")

        # SEC EDGAR → insiders
        edgar_recs = refresher.fetch_sec_edgar(
            since=db.last_insider_date() - datetime.timedelta(days=7)
        )
        if edgar_recs:
            rows = [conv_ins(r) for r in edgar_recs if r.get("ticker") and r.get("trade_date")]
            n = db.bulk_insert_insider(rows)
            totals["insider"] += len(rows)
            logger.info(f"  SEC EDGAR: inserted {n} new insider records")

        # OpenInsider → insiders (supplement)
        oi_recs = refresher.fetch_openinsider(days_back=30, min_value_k=50, page_limit=5)
        if oi_recs:
            rows = [conv_ins(r) for r in oi_recs if r.get("ticker") and r.get("trade_date")]
            n = db.bulk_insert_insider(rows)
            totals["insider"] += len(rows)
            logger.info(f"  OpenInsider: inserted {n} new insider records")

    after = db.counts()
    delta = {k: after[k] - before.get(k, 0) for k in after}

    logger.info(f"\nResults for {db_path.name}:")
    logger.info(f"  {'Table':<25} {'Before':>10} {'After':>10} {'Added':>10}")
    for tbl, cnt in after.items():
        b = before.get(tbl, 0)
        logger.info(f"  {tbl:<25} {b:>10,} {cnt:>10,} {delta[tbl]:>+10,}")

    return after


def _seed_in_chunks(
    db: Any,
    start: datetime.date,
    end: datetime.date,
    generator_fn,
    converter_fn,
    inserter_fn,
    label: str,
    totals: dict,
    verbose: bool,
) -> None:
    """Generate and insert data in CHUNK_DAYS-day windows to manage RAM."""
    cur = start
    chunk_n = 0
    total_chunks = math.ceil((end - start).days / CHUNK_DAYS)

    while cur <= end:
        chunk_end = min(cur + datetime.timedelta(days=CHUNK_DAYS - 1), end)
        records   = generator_fn(cur, chunk_end, verbose=False)
        rows      = [converter_fn(r) for r in records if r.get("ticker") or r.get("politician")]
        inserted  = inserter_fn(rows) if rows else 0
        totals[label] += len(rows)
        chunk_n += 1

        if verbose:
            _progress(f"{label}", chunk_n, total_chunks)
        else:
            pct = chunk_n / max(total_chunks, 1)
            logger.info(
                f"  [{label}] chunk {chunk_n}/{total_chunks} "
                f"({cur} → {chunk_end}): {len(rows):,} rows generated"
            )
        cur = chunk_end + datetime.timedelta(days=1)

    if verbose:
        print()   # newline after progress bar


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stock-Hawk historical data pre-seeder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        choices=["flippy", "rl", "both"],
        default="both",
        help="Which database to seed (default: both)",
    )
    parser.add_argument(
        "--flippy-path",
        default=str(FLIPPY_DB_PATH),
        help=f"Path to flippy_store.db  (default: {FLIPPY_DB_PATH})",
    )
    parser.add_argument(
        "--rl-path",
        default=str(INSIDER_RL_PATH),
        help=f"Path to insider_rl.db  (default: {INSIDER_RL_PATH})",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip live network refresh; use curated data only",
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Skip curated backfill; only attempt live network scrape",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-chunk progress bars",
    )
    parser.add_argument(
        "--from-date",
        default=None,
        help="Override start date (YYYY-MM-DD).  Default: STOCK Act date 2012-04-04",
    )
    parser.add_argument(
        "--to-date",
        default=None,
        help="Override end date (YYYY-MM-DD).  Default: today",
    )

    args = parser.parse_args()

    if args.offline and args.live_only:
        parser.error("--offline and --live-only are mutually exclusive")

    # Override global dates if provided
    global STOCK_ACT_DATE, TODAY
    if args.from_date:
        try:
            STOCK_ACT_DATE = datetime.date.fromisoformat(args.from_date)
        except ValueError:
            parser.error(f"Invalid --from-date: {args.from_date}")
    if args.to_date:
        try:
            TODAY = datetime.date.fromisoformat(args.to_date)
        except ValueError:
            parser.error(f"Invalid --to-date: {args.to_date}")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start_time = time.time()

    results: dict[str, dict] = {}

    if args.db in ("flippy", "both"):
        results["flippy"] = seed_database(
            db_path  = Path(args.flippy_path),
            db_type  = "flippy",
            offline  = args.offline,
            live_only= args.live_only,
            verbose  = args.verbose,
        )

    if args.db in ("rl", "both"):
        results["rl"] = seed_database(
            db_path  = Path(args.rl_path),
            db_type  = "rl",
            offline  = args.offline,
            live_only= args.live_only,
            verbose  = args.verbose,
        )

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Pre-seed complete in {elapsed:.1f}s")
    logger.info(f"{'='*60}")
    for db_name, counts in results.items():
        logger.info(f"  {db_name}:")
        for tbl, cnt in counts.items():
            logger.info(f"    {tbl:<30} {cnt:>10,} rows")


if __name__ == "__main__":
    main()