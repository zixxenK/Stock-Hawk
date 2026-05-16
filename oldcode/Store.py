"""
SQLite historical database.

Stores every scraped event (congress trade, insider trade, options alert,
momentum snapshot) so the vector engine can learn from the past.

Schema is intentionally flat — one row per event, rich JSON payload, so
we can add new fields without migrations.

Tables:
  congress_trades   — House / Senate / Capitol Trades disclosures
  insider_trades    — SEC EDGAR Form 4 (officers, directors, 10%+ holders)
  options_alerts    — Unusual options flow (large sweeps, whale prints)
  momentum_snapshots— Daily scan results per ticker (for outcome tracking)
  price_outcomes    — Forward returns attached to each event after the fact
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "intelligence.db"


CREATE_STATEMENTS = [
    # ── Congress trades ──────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS congress_trades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        scraped_at      TEXT NOT NULL,
        source          TEXT NOT NULL,        -- 'capitol_trades' | 'house' | 'senate'
        politician      TEXT NOT NULL,
        party           TEXT,
        chamber         TEXT,                 -- 'house' | 'senate'
        state           TEXT,
        ticker          TEXT NOT NULL,
        asset_name      TEXT,
        trade_type      TEXT NOT NULL,        -- 'purchase' | 'sale' | 'exchange'
        trade_date      TEXT,
        disclosure_date TEXT,
        amount_min      INTEGER,              -- lower bound of reported range (USD)
        amount_max      INTEGER,
        amount_mid      INTEGER,              -- midpoint used for vectorisation
        comment         TEXT,
        raw_json        TEXT,
        UNIQUE(source, politician, ticker, trade_date, trade_type)
    )
    """,
    # ── SEC EDGAR Form 4 insider trades ──────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS insider_trades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        scraped_at      TEXT NOT NULL,
        source          TEXT NOT NULL,        -- 'sec_edgar' | 'openinsider'
        ticker          TEXT NOT NULL,
        company_name    TEXT,
        insider_name    TEXT NOT NULL,
        insider_title   TEXT,
        trade_type      TEXT NOT NULL,        -- 'P' purchase | 'S' sale | 'M' option exercise
        trade_date      TEXT NOT NULL,
        shares          INTEGER,
        price_per_share REAL,
        total_value     REAL,
        shares_owned_after INTEGER,
        form_url        TEXT,
        raw_json        TEXT,
        UNIQUE(source, insider_name, ticker, trade_date, trade_type, shares)
    )
    """,
    # ── Unusual options flow ──────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS options_alerts (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        scraped_at      TEXT NOT NULL,
        source          TEXT NOT NULL,
        ticker          TEXT NOT NULL,
        expiry          TEXT,
        strike          REAL,
        option_type     TEXT,                 -- 'call' | 'put'
        sentiment       TEXT,                 -- 'bullish' | 'bearish' | 'neutral'
        premium         REAL,                 -- total premium paid (USD)
        contracts       INTEGER,
        open_interest   INTEGER,
        iv_percentile   REAL,
        trade_date      TEXT NOT NULL,
        raw_json        TEXT
    )
    """,
    # ── Daily momentum snapshots ──────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS momentum_snapshots (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date   TEXT NOT NULL,
        ticker          TEXT NOT NULL,
        price           REAL,
        score           REAL,
        grade           TEXT,
        mom_12_1        REAL,
        mom_6m          REAL,
        mom_3m          REAL,
        rs_vs_spy       REAL,
        high_52w_pct    REAL,
        adx             REAL,
        rsi             REAL,
        vol_trend       REAL,
        golden_cross    INTEGER,
        atr             REAL,
        hist_vol        REAL,
        feature_vector  TEXT,                 -- JSON array for similarity search
        UNIQUE(snapshot_date, ticker)
    )
    """,
    # ── Forward price outcomes (filled in retrospectively) ─────────────────
    """
    CREATE TABLE IF NOT EXISTS price_outcomes (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        event_table     TEXT NOT NULL,        -- which table the event is in
        event_id        INTEGER NOT NULL,
        ticker          TEXT NOT NULL,
        event_date      TEXT NOT NULL,
        price_at_event  REAL,
        ret_1w          REAL,
        ret_2w          REAL,
        ret_1m          REAL,
        ret_3m          REAL,
        ret_6m          REAL,
        filled_at       TEXT,
        UNIQUE(event_table, event_id)
    )
    """,
    # ── Signal composite (ties events + momentum together) ────────────────
    """
    CREATE TABLE IF NOT EXISTS composite_signals (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at      TEXT NOT NULL,
        ticker          TEXT NOT NULL,
        signal_date     TEXT NOT NULL,
        signal_score    REAL NOT NULL,        -- 0-100, higher = stronger
        signal_grade    TEXT,
        congress_buys   INTEGER DEFAULT 0,
        insider_buys    INTEGER DEFAULT 0,
        options_bullish INTEGER DEFAULT 0,
        momentum_score  REAL,
        suspicious_vol  INTEGER DEFAULT 0,    -- 1 if volume looks coordinated
        notes           TEXT,
        feature_vector  TEXT                  -- JSON array for similarity search
    )
    """
]

INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_congress_ticker ON congress_trades(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_congress_date   ON congress_trades(trade_date)",
    "CREATE INDEX IF NOT EXISTS idx_insider_ticker  ON insider_trades(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_insider_date    ON insider_trades(trade_date)",
    "CREATE INDEX IF NOT EXISTS idx_options_ticker  ON options_alerts(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_options_date    ON options_alerts(trade_date)",
    "CREATE INDEX IF NOT EXISTS idx_snap_ticker     ON momentum_snapshots(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_snap_date       ON momentum_snapshots(snapshot_date)",
    "CREATE INDEX IF NOT EXISTS idx_sig_ticker      ON composite_signals(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_sig_score       ON composite_signals(signal_score)",
]


class IntelligenceDB:
    """Thin wrapper around the SQLite intelligence database."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            for stmt in CREATE_STATEMENTS:
                conn.execute(stmt)
            for idx in INDEX_STATEMENTS:
                conn.execute(idx)

    # ── Congress trades ─────────────────────────────────────────────────

    def upsert_congress_trade(self, row: dict) -> bool:
        """Insert or ignore a congress trade record. Returns True if new."""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            INSERT OR IGNORE INTO congress_trades
            (scraped_at, source, politician, party, chamber, state,
             ticker, asset_name, trade_type, trade_date, disclosure_date,
             amount_min, amount_max, amount_mid, comment, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._conn() as conn:
            cur = conn.execute(sql, (
                now,
                row.get("source", ""),
                row.get("politician", ""),
                row.get("party"),
                row.get("chamber"),
                row.get("state"),
                row.get("ticker", "").upper(),
                row.get("asset_name"),
                row.get("trade_type", ""),
                row.get("trade_date"),
                row.get("disclosure_date"),
                row.get("amount_min"),
                row.get("amount_max"),
                row.get("amount_mid"),
                row.get("comment"),
                json.dumps(row),
            ))
            return cur.rowcount > 0

    def get_congress_trades(
        self,
        ticker: str | None = None,
        days_back: int = 90,
        trade_type: str | None = None,
    ) -> list[dict]:
        sql = """
            SELECT * FROM congress_trades
            WHERE trade_date >= date('now', ?)
        """
        params: list = [f"-{days_back} days"]
        if ticker:
            sql += " AND ticker = ?"
            params.append(ticker.upper())
        if trade_type:
            sql += " AND trade_type = ?"
            params.append(trade_type)
        sql += " ORDER BY trade_date DESC"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ── Insider trades ──────────────────────────────────────────────────

    def upsert_insider_trade(self, row: dict) -> bool:
        sql = """
            INSERT OR IGNORE INTO insider_trades
            (scraped_at, source, ticker, company_name, insider_name,
             insider_title, trade_type, trade_date, shares, price_per_share,
             total_value, shares_owned_after, form_url, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cur = conn.execute(sql, (
                now,
                row.get("source", ""),
                row.get("ticker", "").upper(),
                row.get("company_name"),
                row.get("insider_name", ""),
                row.get("insider_title"),
                row.get("trade_type", ""),
                row.get("trade_date", ""),
                row.get("shares"),
                row.get("price_per_share"),
                row.get("total_value"),
                row.get("shares_owned_after"),
                row.get("form_url"),
                json.dumps(row),
            ))
            return cur.rowcount > 0

    def get_insider_trades(
        self,
        ticker: str | None = None,
        days_back: int = 90,
        purchases_only: bool = True,
    ) -> list[dict]:
        sql = """
            SELECT * FROM insider_trades
            WHERE trade_date >= date('now', ?)
        """
        params: list = [f"-{days_back} days"]
        if ticker:
            sql += " AND ticker = ?"
            params.append(ticker.upper())
        if purchases_only:
            sql += " AND trade_type = 'P'"
        sql += " ORDER BY trade_date DESC"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ── Options alerts ──────────────────────────────────────────────────

    def insert_options_alert(self, row: dict) -> None:
        sql = """
            INSERT INTO options_alerts
            (scraped_at, source, ticker, expiry, strike, option_type,
             sentiment, premium, contracts, open_interest, iv_percentile,
             trade_date, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(sql, (
                now,
                row.get("source", ""),
                row.get("ticker", "").upper(),
                row.get("expiry"),
                row.get("strike"),
                row.get("option_type"),
                row.get("sentiment"),
                row.get("premium"),
                row.get("contracts"),
                row.get("open_interest"),
                row.get("iv_percentile"),
                row.get("trade_date", ""),
                json.dumps(row),
            ))

    def get_options_alerts(
        self,
        ticker: str | None = None,
        days_back: int = 30,
        bullish_only: bool = False,
    ) -> list[dict]:
        sql = "SELECT * FROM options_alerts WHERE trade_date >= date('now', ?)"
        params: list = [f"-{days_back} days"]
        if ticker:
            sql += " AND ticker = ?"
            params.append(ticker.upper())
        if bullish_only:
            sql += " AND sentiment = 'bullish'"
        sql += " ORDER BY premium DESC"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ── Momentum snapshots ──────────────────────────────────────────────

    def upsert_momentum_snapshot(self, snap: dict, feature_vector: list[float]) -> None:
        sql = """
            INSERT OR REPLACE INTO momentum_snapshots
            (snapshot_date, ticker, price, score, grade, mom_12_1, mom_6m,
             mom_3m, rs_vs_spy, high_52w_pct, adx, rsi, vol_trend,
             golden_cross, atr, hist_vol, feature_vector)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        today = datetime.now(timezone.utc).date().isoformat()
        with self._conn() as conn:
            conn.execute(sql, (
                today,
                snap.get("ticker", "").upper(),
                snap.get("price"),
                snap.get("score"),
                snap.get("grade"),
                snap.get("mom_12_1"),
                snap.get("mom_6m"),
                snap.get("mom_3m"),
                snap.get("rs_vs_spy"),
                snap.get("high_52w_pct"),
                snap.get("adx"),
                snap.get("rsi"),
                snap.get("vol_trend"),
                int(snap.get("golden_cross") or 0),
                snap.get("atr"),
                snap.get("hist_vol"),
                json.dumps(feature_vector),
            ))

    def get_all_snapshots_with_vectors(
        self, days_back: int = 365
    ) -> list[dict]:
        sql = """
            SELECT * FROM momentum_snapshots
            WHERE snapshot_date >= date('now', ?)
              AND feature_vector IS NOT NULL
            ORDER BY snapshot_date DESC
        """
        with self._conn() as conn:
            rows = conn.execute(sql, [f"-{days_back} days"]).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["feature_vector"] = json.loads(d["feature_vector"] or "[]")
            result.append(d)
        return result

    # ── Composite signals ───────────────────────────────────────────────

    def insert_signal(self, sig: dict, feature_vector: list[float]) -> int:
        sql = """
            INSERT INTO composite_signals
            (created_at, ticker, signal_date, signal_score, signal_grade,
             congress_buys, insider_buys, options_bullish, momentum_score,
             suspicious_vol, notes, feature_vector)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """
        now = datetime.now(timezone.utc).isoformat()
        today = datetime.now(timezone.utc).date().isoformat()
        with self._conn() as conn:
            cur = conn.execute(sql, (
                now,
                sig.get("ticker", "").upper(),
                sig.get("signal_date", today),
                sig.get("signal_score", 0),
                sig.get("signal_grade"),
                sig.get("congress_buys", 0),
                sig.get("insider_buys", 0),
                sig.get("options_bullish", 0),
                sig.get("momentum_score"),
                int(sig.get("suspicious_vol", False)),
                sig.get("notes"),
                json.dumps(feature_vector),
            ))
            return cur.lastrowid

    def get_recent_signals(self, days_back: int = 30, min_score: float = 60) -> list[dict]:
        sql = """
            SELECT * FROM composite_signals
            WHERE signal_date >= date('now', ?)
              AND signal_score >= ?
            ORDER BY signal_score DESC
        """
        with self._conn() as conn:
            rows = conn.execute(sql, [f"-{days_back} days", min_score]).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["feature_vector"] = json.loads(d.get("feature_vector") or "[]")
            result.append(d)
        return result

    # ── Price outcomes ──────────────────────────────────────────────────

    def upsert_outcome(self, outcome: dict) -> None:
        sql = """
            INSERT OR REPLACE INTO price_outcomes
            (event_table, event_id, ticker, event_date, price_at_event,
             ret_1w, ret_2w, ret_1m, ret_3m, ret_6m, filled_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(sql, (
                outcome["event_table"],
                outcome["event_id"],
                outcome["ticker"],
                outcome["event_date"],
                outcome.get("price_at_event"),
                outcome.get("ret_1w"),
                outcome.get("ret_2w"),
                outcome.get("ret_1m"),
                outcome.get("ret_3m"),
                outcome.get("ret_6m"),
                now,
            ))

    def get_outcomes_for_table(self, event_table: str) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM price_outcomes WHERE event_table=?", [event_table]
            ).fetchall()]

    # ── Stats ───────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._conn() as conn:
            def count(tbl):
                return conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            return {
                "congress_trades":    count("congress_trades"),
                "insider_trades":     count("insider_trades"),
                "options_alerts":     count("options_alerts"),
                "momentum_snapshots": count("momentum_snapshots"),
                "composite_signals":  count("composite_signals"),
                "price_outcomes":     count("price_outcomes"),
                "db_path": str(self.db_path),
            }