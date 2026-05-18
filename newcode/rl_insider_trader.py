"""
rl_insider_trader.py  ·  InsiderRL v2.0 — Hardened Edition
Stock-Hawk Multi-Dimensional RL Trading Agent

Audit fixes applied (see AUDIT_LOG at bottom for full change manifest):
  [A1] Temporal integrity  — get_disclosed_signals_on_date enforces
       disclosure_date ≤ sim_step_date AND post-market-close hold-back.
  [A2] Vector zero-guard  — epsilon=1e-8 prevents NaN propagation into
       the policy gradient when the feature vector collapses to zero.
  [A3] Gymnasium warm-up  — env starts at step 26 (max indicator lag)
       to guarantee RSI-14 and MACD-26 arrays are fully initialized.
  [A4] Reward rebalancing — ALPHA_CHURN scaled from 0.02 → 0.002 so the
       churn penalty does NOT dominate daily PnL (~0.03 range), preventing
       the agent from locking into a permanent cash/flat policy.
  [A5] Fee order-of-ops   — transaction fee is deducted from the CURRENT
       portfolio value BEFORE the next-period price return is applied,
       matching real-world broker mechanics (no leverage-free borrowing).
  [A6] Streamlit thread   — trained model and backtest results are stored
       in st.session_state so widget interactions do NOT restart training;
       WAL journal mode already present (retained & documented).

Run:
    streamlit run rl_insider_trader.py

Install:
    pip install streamlit gymnasium stable-baselines3 scikit-learn plotly pandas numpy
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import math
import random
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import gymnasium as gym
from gymnasium import spaces
from sklearn.decomposition import PCA
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from database.db_manager import DBManager, create_db_adapter, validate_db_adapter
from data_sources.yahoo_data import fetch_price_history
from data_sources.scraper_manager import ScraperManager
from intelligence.llm_advisor import LocalLLMAdvisor
from intelligence.training_manager import AutonomousTrainingConfig, AutonomousTrainingManager
from processing.sentiment_processor import SentimentProcessor
from preseedhistory import seed_database

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("InsiderRL")


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = Path("./data/insider_rl.db")

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "V", "MA", "UNH", "PG", "KO", "PEP", "JNJ",
    "WMT", "DIS", "BAC", "XOM", "CVX", "PFE", "MRK",
    "CRM", "ADBE", "NFLX", "PYPL", "INTC", "CSCO", "ORCL",
    "COST", "NKE", "MCD", "SBUX", "T", "VZ", "BA", "GE",
    "IBM", "ABT",
]

POLITICIANS = [
    "Nancy Pelosi", "Dan Crenshaw", "Tommy Tuberville",
    "Marjorie Taylor Greene", "Josh Gottheimer", "Brian Mast",
    "Austin Scott", "Ro Khanna",
]
ROLES = ["CEO", "CFO", "Director", "SVP", "10% Owner", "President", "CTO", "EVP"]

ROLE_SENIORITY: dict[str, float] = {
    "CEO": 1.00, "President": 0.90, "Chairman": 0.95, "CFO": 0.95,
    "CTO": 0.85, "COO": 0.90, "Director": 0.75, "SVP": 0.70,
    "EVP": 0.75, "VP": 0.65, "10% Owner": 0.80, "Officer": 0.60,
}

DEFAULT_FOCUS_SETTINGS: dict[str, Any] = {
    "insider_flow_weight": 1.0,
    "congress_flow_weight": 1.0,
    "momentum_horizon_days": 63,
    "max_trade_size_pct": 0.05,
    "per_ticker_risk_budget_pct": 0.10,
    "use_watchlist_only": True,
}

# Canonical "suspicious accumulation" feature template vectors (4-dim, normalized)
INSIDER_TEMPLATE  = np.array([0.90, 0.95, 0.80, 0.75], dtype=np.float32)
CONGRESS_TEMPLATE = np.array([0.85, 1.00, 0.70, 0.65], dtype=np.float32)

# ── [A4] Reward shaping — rebalanced coefficients ───────────────────────────
# ALPHA_CHURN was 0.02 (≈ 67% of a typical daily PnL swing of 0.03).
# At that scale PPO immediately learns "never trade" to dodge the fee.
# Scaled down by 10× to 0.002 so it acts as a soft anti-churn nudge.
ALPHA_CHURN         = 0.002   # was 0.02  — [A4]
BETA_ALIGNMENT      = 0.015   # alignment bonus for holding at high insider-sim
GAMMA_DRAWDOWN      = 0.05    # drawdown penalty coefficient
FEE_PCT             = 0.0015  # 0.15% transaction fee
MAX_TRADE_EXPOSURE_PCT = 0.10  # maximum single trade exposure as fraction of equity
STOP_LOSS_PCT       = 0.05    # trailing stop-loss at 5% below entry
ALIGN_THRESHOLD     = 0.70    # cosine sim threshold for alignment bonus
DRAWDOWN_THRESHOLD  = 0.05    # 5% decline from peak triggers penalty

# Indicator warm-up minimum (MACD-26 is the longest look-back)  — [A3]
INDICATOR_WARMUP    = 26

# Market close time (EST / UTC-4 during EDT) — post-close disclosures
# held back until next trading day                              — [A1]
MARKET_CLOSE_UTC    = datetime.time(20, 0, 0)   # 16:00 EST = 20:00 UTC


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — DATABASE & DATA MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

class MarketIntelligenceDB:
    """
    SQLite persistence layer for congressional disclosures, SEC Form 4 filings,
    and news-sentiment scores.

    Design choices
    ──────────────
    • WAL journal mode for concurrent read safety (Streamlit widget re-runs)  [A6]
    • Disclosure-date latency emulation eliminates look-ahead bias:
        - Congress (STOCK Act): 5–45 calendar days
        - SEC Form 4 (insiders): 2–4 business days
    • get_disclosed_signals_on_date enforces D_disclosure ≤ D_sim_step AND
      applies a post-market-close hold-back (disclosures published after
      16:00 EST on day D are not visible until day D+1).              [A1]
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
            inserted_at         TEXT    NOT NULL
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
            inserted_at         TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            score       REAL NOT NULL,
            source      TEXT,
            headline    TEXT,
            trade_date  TEXT NOT NULL,
            pub_time_utc TEXT NOT NULL DEFAULT '12:00:00',
            inserted_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analysis_signals (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker         TEXT NOT NULL,
            action         TEXT NOT NULL,
            alpha_score    REAL NOT NULL,
            sentiment_score REAL,
            vector         TEXT,
            inserted_at    TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ct_ticker  ON congress_trades(ticker, disclosure_date);
        CREATE INDEX IF NOT EXISTS idx_it_ticker  ON insider_trades(ticker, disclosure_date);
        CREATE INDEX IF NOT EXISTS idx_ns_ticker  ON news_sentiment(ticker, trade_date);
        CREATE INDEX IF NOT EXISTS idx_as_ticker  ON analysis_signals(ticker, inserted_at);
        CREATE TABLE IF NOT EXISTS strategy_sessions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id         TEXT    NOT NULL,
            ticker             TEXT    NOT NULL,
            strategy_name      TEXT,
            training_steps     INTEGER,
            learning_rate      REAL,
            entropy_coef       REAL,
            backtest_days      INTEGER,
            performance_metrics TEXT,
            suggested_change   TEXT,
            notes              TEXT,
            created_at         TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS watchlist_tickers (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            note        TEXT,
            added_at    TEXT    NOT NULL,
            UNIQUE(ticker)
        );
        CREATE TABLE IF NOT EXISTS focus_settings (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_name   TEXT    NOT NULL UNIQUE,
            setting_value  TEXT    NOT NULL,
            updated_at     TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ss_ticker ON strategy_sessions(ticker, created_at DESC);
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(self._SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        # WAL mode supports concurrent reads while a write is in progress,
        # preventing SQLite "database is locked" errors during Streamlit re-runs [A6]
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── Latency emulation engine ──────────────────────────────────────────

    @staticmethod
    def emulate_latency(
        trade_date: str,
        min_days: int = 2,
        max_days: int = 45,
        force_time_utc: str | None = None,
    ) -> tuple[str, int, str]:
        """
        Shift a trade date forward by a simulated regulatory disclosure delay.

        Congress (STOCK Act):   5–45 calendar days (legally mandated ceiling)
        SEC Form 4 (insiders):  2–4 business days

        The disclosure_time_utc is randomised to simulate filings arriving both
        before and after market close.  Post-close filings are held back one
        additional trading day in get_disclosed_signals_on_date.   [A1]

        Returns (disclosure_date_str, latency_days, time_utc_str).
        """
        try:
            dt = datetime.datetime.strptime(trade_date, "%Y-%m-%d")
        except ValueError:
            return trade_date, 0, "12:00:00"

        delay = random.randint(min_days, max_days)
        disc_dt = dt + datetime.timedelta(days=delay)
        disc_str = disc_dt.strftime("%Y-%m-%d")

        # Simulate publication time: 70% chance of pre-close, 30% post-close
        if force_time_utc:
            time_utc = force_time_utc
        elif random.random() < 0.70:
            # Pre-close: 09:30–19:59 UTC
            h = random.randint(9, 19)
            m = random.randint(0, 59)
            time_utc = f"{h:02d}:{m:02d}:00"
        else:
            # Post-close: 20:00–23:59 UTC
            h = random.randint(20, 23)
            m = random.randint(0, 59)
            time_utc = f"{h:02d}:{m:02d}:00"

        return disc_str, delay, time_utc

    # ── Insert methods ────────────────────────────────────────────────────

    def insert_congress_trade(self, rec: dict) -> None:
        disc, lat, t_utc = self.emulate_latency(
            rec["trade_date"], min_days=5, max_days=45
        )
        now = datetime.datetime.utcnow().isoformat()
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO congress_trades
                   (ticker, politician, trade_type, amount_range,
                    trade_date, disclosure_date, disclosure_time_utc,
                    latency_days, inserted_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (rec["ticker"].upper(), rec["politician"], rec["trade_type"],
                 rec.get("amount_range", "$15,001 – $50,000"),
                 rec["trade_date"], disc, t_utc, lat, now),
            )

    def insert_insider_trade(self, rec: dict) -> None:
        disc, lat, t_utc = self.emulate_latency(
            rec["trade_date"], min_days=2, max_days=4
        )
        now = datetime.datetime.utcnow().isoformat()
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO insider_trades
                   (ticker, insider_name, position, shares_traded, price,
                    trade_type, trade_date, disclosure_date, disclosure_time_utc,
                    latency_days, inserted_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (rec["ticker"].upper(), rec["insider_name"],
                 rec.get("position", "Director"), rec.get("shares_traded", 1000),
                 rec.get("price", 100.0), rec.get("trade_type", "Purchase"),
                 rec["trade_date"], disc, t_utc, lat, now),
            )

    def insert_sentiment(self, rec: dict) -> None:
        now = datetime.datetime.utcnow().isoformat()
        # Sentiment is published intraday; default to pre-close
        pub_time = rec.get("pub_time_utc", "14:30:00")
        with self._conn() as c:
            c.execute(
                """INSERT INTO news_sentiment
                   (ticker, score, source, headline, trade_date, pub_time_utc, inserted_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (rec["ticker"].upper(), rec["score"],
                 rec.get("source", "Yahoo Finance"),
                 rec.get("headline", ""), rec["trade_date"], pub_time, now),
            )

    # ── [A1] Strict temporal query ────────────────────────────────────────

    def get_disclosed_signals_on_date(
        self,
        ticker: str,
        sim_date: str,
        sim_date_is_after_close: bool = False,
    ) -> dict[str, list[dict]]:
        """
        Return all alternative-data signals that were PUBLICLY AVAILABLE
        to an agent trading on `sim_date`.

        Temporal enforcement rules:
          1. disclosure_date MUST be strictly ≤ sim_date.
          2. If sim_date_is_after_close is False (i.e. we are simulating
             intraday / at open), disclosures published after market close
             (disclosure_time_utc ≥ '20:00:00') on the SAME day as sim_date
             are held back until the NEXT trading day.
          3. This mirrors the real institutional constraint: a Form 4 filed
             at 17:15 UTC on Monday cannot be acted upon until Tuesday's open.

        Returns dict with keys: 'congress', 'insider', 'sentiment'.
        """
        ticker = ticker.upper()

        # Congress trades
        if sim_date_is_after_close:
            # After close: include same-day disclosures regardless of time
            cong_q = """
                SELECT * FROM congress_trades
                WHERE ticker=? AND trade_type='Purchase'
                  AND disclosure_date <= ?
                ORDER BY disclosure_date DESC LIMIT 20"""
            cong_rows = self._fetch(cong_q, (ticker, sim_date))
        else:
            # Before / at close: exclude same-day post-close filings
            cong_q = """
                SELECT * FROM congress_trades
                WHERE ticker=? AND trade_type='Purchase'
                  AND (
                    disclosure_date < ?
                    OR (disclosure_date = ? AND disclosure_time_utc < '20:00:00')
                  )
                ORDER BY disclosure_date DESC LIMIT 20"""
            cong_rows = self._fetch(cong_q, (ticker, sim_date, sim_date))

        # Insider trades (same logic)
        if sim_date_is_after_close:
            ins_q = """
                SELECT * FROM insider_trades
                WHERE ticker=? AND trade_type='Purchase'
                  AND disclosure_date <= ?
                ORDER BY disclosure_date DESC LIMIT 20"""
            ins_rows = self._fetch(ins_q, (ticker, sim_date))
        else:
            ins_q = """
                SELECT * FROM insider_trades
                WHERE ticker=? AND trade_type='Purchase'
                  AND (
                    disclosure_date < ?
                    OR (disclosure_date = ? AND disclosure_time_utc < '20:00:00')
                  )
                ORDER BY disclosure_date DESC LIMIT 20"""
            ins_rows = self._fetch(ins_q, (ticker, sim_date, sim_date))

        # Sentiment — always safe to include (intraday; pre-close default)
        sent_q = """
            SELECT * FROM news_sentiment
            WHERE ticker=? AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT 10"""
        sent_rows = self._fetch(sent_q, (ticker, sim_date))

        return {
            "congress": [dict(r) for r in cong_rows],
            "insider":  [dict(r) for r in ins_rows],
            "sentiment":[dict(r) for r in sent_rows],
        }

    def _fetch(self, query: str, params: tuple) -> list:
        with self._conn() as c:
            return c.execute(query, params).fetchall()

    def get_congress_trades(
        self,
        ticker: str,
        days_back: int = 30,
        purchases_only: bool = True,
        start_date: str | None = None,
    ) -> list[dict]:
        query = [
            "SELECT * FROM congress_trades",
            "WHERE ticker=?",
        ]
        params: list = [ticker.upper()]
        if purchases_only:
            query.append("AND trade_type='Purchase'")
        if start_date is not None:
            query.append("AND trade_date >= ?")
            params.append(start_date)
        else:
            query.append("AND trade_date >= date('now',?)")
            params.append(f"-{days_back} days")
        query.append("ORDER BY trade_date DESC")

        rows = self._fetch("\n".join(query), tuple(params))
        return [dict(r) for r in rows]

    def get_insider_trades(
        self,
        ticker: str,
        days_back: int = 30,
        purchases_only: bool = True,
        start_date: str | None = None,
    ) -> list[dict]:
        query = [
            "SELECT * FROM insider_trades",
            "WHERE ticker=?",
        ]
        params: list = [ticker.upper()]
        if purchases_only:
            query.append("AND trade_type='Purchase'")
        if start_date is not None:
            query.append("AND trade_date >= ?")
            params.append(start_date)
        else:
            query.append("AND trade_date >= date('now',?)")
            params.append(f"-{days_back} days")
        query.append("ORDER BY trade_date DESC")

        rows = self._fetch("\n".join(query), tuple(params))
        return [dict(r) for r in rows]
        if start_date is not None:
            rows = self._fetch(
                """SELECT * FROM insider_trades
                   WHERE ticker=? AND trade_type='Purchase'
                     AND trade_date >= ?
                   ORDER BY trade_date DESC""",
                (ticker.upper(), start_date),
            )
        else:
            rows = self._fetch(
                """SELECT * FROM insider_trades
                   WHERE ticker=? AND trade_type='Purchase'
                     AND trade_date >= date('now',?)
                   ORDER BY trade_date DESC""",
                (ticker.upper(), f"-{days_back} days"),
            )
        return [dict(r) for r in rows]

    def get_news_sentiment(
        self,
        ticker: str,
        days_back: int = 30,
    ) -> list[dict]:
        if days_back is not None:
            rows = self._fetch(
                """SELECT * FROM news_sentiment
                   WHERE ticker=? AND trade_date >= date('now',?)
                   ORDER BY trade_date DESC""",
                (ticker.upper(), f"-{days_back} days"),
            )
        else:
            rows = self._fetch(
                """SELECT * FROM news_sentiment
                   WHERE ticker=?
                   ORDER BY trade_date DESC""",
                (ticker.upper(),),
            )
        return [dict(r) for r in rows]

    def get_watchlist_tickers(self) -> list[dict[str, Any]]:
        rows = self._fetch(
            """SELECT ticker, source, note, added_at
               FROM watchlist_tickers
               ORDER BY added_at DESC""",
            (),
        )
        return [dict(r) for r in rows]

    def add_watchlist_ticker(self, ticker: str, source: str = "manual", note: str | None = None) -> bool:
        ticker = ticker.strip().upper()
        if not ticker:
            return False
        now = datetime.datetime.utcnow().isoformat()
        try:
            with self._conn() as c:
                c.execute(
                    """INSERT OR IGNORE INTO watchlist_tickers
                       (ticker, source, note, added_at)
                       VALUES (?,?,?,?)""",
                    (ticker, source, note or "", now),
                )
            return True
        except Exception as exc:
            logger.error("Failed to add watchlist ticker: %s", exc)
            return False

    def remove_watchlist_ticker(self, ticker: str) -> int:
        ticker = ticker.strip().upper()
        with self._conn() as c:
            cursor = c.execute(
                """DELETE FROM watchlist_tickers WHERE ticker=?""",
                (ticker,),
            )
        return cursor.rowcount

    def clear_watchlist(self) -> int:
        with self._conn() as c:
            cursor = c.execute("""DELETE FROM watchlist_tickers""")
        return cursor.rowcount

    def get_focus_settings(self) -> dict[str, Any]:
        rows = self._fetch(
            """SELECT setting_name, setting_value FROM focus_settings""",
            (),
        )
        settings: dict[str, Any] = {}
        for row in rows:
            try:
                settings[row["setting_name"]] = json.loads(row["setting_value"])
            except Exception:
                settings[row["setting_name"]] = row["setting_value"]
        return {**DEFAULT_FOCUS_SETTINGS, **settings}

    def save_focus_settings(self, settings: dict[str, Any]) -> bool:
        now = datetime.datetime.utcnow().isoformat()
        try:
            with self._conn() as c:
                for name, value in settings.items():
                    c.execute(
                        """INSERT INTO focus_settings
                           (setting_name, setting_value, updated_at)
                           VALUES (?,?,?)
                           ON CONFLICT(setting_name) DO UPDATE SET
                             setting_value=excluded.setting_value,
                             updated_at=excluded.updated_at""",
                        (name, json.dumps(value), now),
                    )
            return True
        except Exception as exc:
            logger.error("Failed to save focus settings: %s", exc)
            return False

    def get_recent_hit_tickers(self, days_back: int = 30) -> list[str]:
        rows = self._fetch(
            """SELECT DISTINCT ticker FROM (
                   SELECT ticker FROM congress_trades
                   WHERE trade_type='Purchase' AND trade_date >= date('now',?)
                   UNION
                   SELECT ticker FROM insider_trades
                   WHERE trade_type='Purchase' AND trade_date >= date('now',?)
               ) ORDER BY ticker""",
            (f"-{days_back} days", f"-{days_back} days"),
        )
        return [r["ticker"] for r in rows]

    def get_avg_sentiment(self, ticker: str, days_back: int = 7) -> float:
        with self._conn() as c:
            row = c.execute(
                """SELECT AVG(score) FROM news_sentiment
                   WHERE ticker=? AND trade_date >= date('now',?)""",
                (ticker.upper(), f"-{days_back} days"),
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def get_stats(self) -> dict[str, int]:
        with self._conn() as c:
            def count(tbl: str) -> int:
                return c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            return {
                "congress_trades": count("congress_trades"),
                "insider_trades":  count("insider_trades"),
                "news_sentiment":  count("news_sentiment"),
                "analysis_signals": count("analysis_signals"),
                "strategy_sessions": count("strategy_sessions"),
            }

    def insert_strategy_session(
        self,
        ticker: str,
        strategy_name: str,
        training_steps: int,
        learning_rate: float,
        entropy_coef: float,
        backtest_days: int,
        performance_metrics: dict[str, float],
        suggested_change: str,
        notes: str | None = None,
    ) -> bool:
        now = datetime.datetime.utcnow().isoformat()
        metrics_payload = json.dumps(performance_metrics)
        with self._conn() as c:
            try:
                c.execute(
                    """INSERT INTO strategy_sessions
                       (session_id, ticker, strategy_name, training_steps,
                        learning_rate, entropy_coef, backtest_days,
                        performance_metrics, suggested_change, notes, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        uuid.uuid4().hex,
                        ticker.upper(),
                        strategy_name,
                        training_steps,
                        float(learning_rate),
                        float(entropy_coef),
                        backtest_days,
                        metrics_payload,
                        suggested_change,
                        notes or "",
                        now,
                    ),
                )
                return True
            except Exception as exc:
                logger.error("Failed to insert strategy session: %s", exc)
                return False

    def get_strategy_sessions(
        self,
        ticker: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        if ticker is not None:
            rows = self._fetch(
                """SELECT * FROM strategy_sessions
                   WHERE ticker=?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (ticker.upper(), limit),
            )
        else:
            rows = self._fetch(
                """SELECT * FROM strategy_sessions
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            )
        sessions = []
        for row in rows:
            metrics = {}
            try:
                metrics = json.loads(row["performance_metrics"]) if row["performance_metrics"] else {}
            except Exception:
                metrics = {}
            sessions.append({
                "session_id": row["session_id"],
                "ticker": row["ticker"],
                "strategy_name": row["strategy_name"],
                "training_steps": row["training_steps"],
                "learning_rate": row["learning_rate"],
                "entropy_coef": row["entropy_coef"],
                "backtest_days": row["backtest_days"],
                "performance_metrics": metrics,
                "suggested_change": row["suggested_change"],
                "notes": row["notes"],
                "created_at": row["created_at"],
            })
        return sessions

    def prune_strategy_sessions(self, keep_days: int = 365) -> int:
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=keep_days)).strftime("%Y-%m-%d")
        with self._conn() as c:
            cursor = c.execute(
                """DELETE FROM strategy_sessions WHERE created_at < ?""",
                (cutoff,),
            )
            return cursor.rowcount

    def upsert_signal_metadata(
        self,
        ticker: str,
        action: str,
        alpha_score: float,
        sentiment_score: float | None,
        vector: list[float] | str | None = None,
    ) -> bool:
        now = datetime.datetime.utcnow().isoformat()
        vector_payload = None
        if vector is not None:
            vector_payload = json.dumps(vector)
        with self._conn() as c:
            try:
                c.execute(
                    """INSERT INTO analysis_signals
                       (ticker, action, alpha_score, sentiment_score, vector, inserted_at)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        ticker.upper(),
                        action,
                        float(alpha_score),
                        float(sentiment_score) if sentiment_score is not None else None,
                        vector_payload,
                        now,
                    ),
                )
                return True
            except Exception as exc:
                logger.error("Failed to upsert signal metadata: %s", exc)
                return False

    def get_ticker_history(self, ticker: str, days_back: int = 365) -> list[dict]:
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
        rows = self._fetch(
            """SELECT * FROM analysis_signals
               WHERE ticker=? AND inserted_at >= ?
               ORDER BY inserted_at DESC""",
            (ticker.upper(), cutoff),
        )
        results = []
        for row in rows:
            vector_value = row["vector"]
            try:
                vector_payload = json.loads(vector_value) if vector_value else None
            except Exception:
                vector_payload = None
            results.append({
                "ticker": row["ticker"],
                "action": row["action"],
                "alpha_score": row["alpha_score"],
                "sentiment_score": row["sentiment_score"],
                "vector": vector_payload,
                "inserted_at": row["inserted_at"],
            })
        return results

    def seed_mock_data(
        self,
        ticker: str,
        n_days: int = 60,
        start_date: datetime.date | None = None,
    ) -> None:
        """Populate the DB with realistic synthetic activity for demonstration."""
        _sentiment = SentimentEngine()
        if start_date is None:
            start_date = datetime.date.today() - datetime.timedelta(days=n_days - 1)

        headlines = [
            "beats earnings expectations substantial growth record profit",
            "insider buy cluster record high breakout bullish momentum",
            "SEC investigation fraud lawsuit class action",
            "dividend increase acquisition expansion partnership approved",
            "guidance cut restructuring layoffs disappointing miss",
            "upgrade outperform all-time high bullish recovery",
        ]
        for i in range(n_days):
            dt = (start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            if i % 9 == 0:
                self.insert_congress_trade({
                    "ticker": ticker,
                    "politician": random.choice(POLITICIANS),
                    "trade_type": random.choice(["Purchase", "Sale"]),
                    "trade_date": dt,
                    "amount_range": random.choice([
                        "$1,001 – $15,000", "$15,001 – $50,000",
                        "$50,001 – $100,000", "$100,001 – $250,000",
                    ]),
                })
            if i % 7 == 0:
                role = random.choice(ROLES)
                self.insert_insider_trade({
                    "ticker": ticker,
                    "insider_name": f"Executive_{i}",
                    "position": role,
                    "shares_traded": random.randint(500, 50_000),
                    "price": round(random.uniform(50, 500), 2),
                    "trade_type": "Purchase",
                    "trade_date": dt,
                })
            hl = random.choice(headlines)
            self.insert_sentiment({
                "ticker": ticker,
                "score": _sentiment.analyze_text(hl),
                "source": "Yahoo Finance",
                "headline": hl,
                "trade_date": dt,
            })

    def get_watchlist_tickers_as_symbols(self) -> list[str]:
        return [row["ticker"] for row in self.get_watchlist_tickers()]

    def build_llm_context(self) -> dict[str, Any]:
        return {
            "focus_settings": self.get_focus_settings(),
            "watchlist_tickers": self.get_watchlist_tickers_as_symbols(),
            "recent_hit_tickers": self.get_recent_hit_tickers(days_back=30),
        }


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — VECTOR GEOMETRY & MATHEMATICAL ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class SignalGeometry:
    """
    Linear algebra engine for matching live alternative data against
    historical high-conviction accumulation profiles.

    Feature vector (4-dim):
        V = [v_volume_percentile, v_role_seniority, v_gex_regime, v_sentiment]

    All dimensions are normalized to [0, 1] before computation.

    [A2] Zero-vector guard: both cosine_similarity and wedge_magnitude
    check ||v|| < 1e-8 and return safe defaults (0.0) rather than
    propagating NaN into the policy network's gradient update.
    """

    FEATURE_NAMES = ["volume_pct", "role_seniority", "gex_regime", "sentiment"]

    @staticmethod
    def build_feature_vector(
        volume_percentile: float,
        role_seniority: float,
        gex_regime: float,
        sentiment: float,  # expected in [-1, 1]
    ) -> np.ndarray:
        """
        Construct a normalized 4-dim alternative data feature vector.
        Sentiment is mapped from [-1, 1] → [0, 1].
        """
        return np.array([
            float(np.clip(volume_percentile,           0.0, 1.0)),
            float(np.clip(role_seniority,              0.0, 1.0)),
            float(np.clip(gex_regime,                  0.0, 1.0)),
            float(np.clip((sentiment + 1.0) / 2.0,    0.0, 1.0)),
        ], dtype=np.float32)

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Directional alignment between current signal vector and a template.

            Sim(v1, v2) = (v1 · v2) / (||v1|| ||v2||)  ∈ [-1, +1]

        +1.0 = perfect alignment with suspicious insider accumulation profile.
        -1.0 = complete opposite (distribution / liquidation).
         0.0 = orthogonal — no relationship to the template.

        [A2] On zero-activity days the feature vector may degenerate to
        ||v|| < 1e-8. Dividing by that norm would produce ±Inf/NaN which
        corrupts all subsequent PPO gradient computations. We return 0.0
        (neutral / no signal) instead — mathematically: if there is no
        directional information, the similarity is undefined and we treat
        it as orthogonal.
        """
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-8 or n2 < 1e-8:   # [A2] hardened guard (was 1e-9, now 1e-8)
            return 0.0
        raw = float(np.dot(v1, v2) / (n1 * n2))
        # Extra safety: clip to [-1, 1] to absorb any floating-point drift
        return float(np.clip(raw, -1.0, 1.0))

    @staticmethod
    def wedge_magnitude(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Frobenius norm of the exterior (wedge) product matrix.

            Exterior product:  v1 ∧ v2 = v1 v2^T − v2 v1^T   (skew-symmetric n×n)
            Magnitude:         ||v1 ∧ v2||_F = √(Σ_ij |(v1 ∧ v2)_ij|²)

        Near zero → vectors are collinear (high-confidence pattern match).
        Large     → structural deviation; current pattern has diverged
                    from historical norms (potential novel regime).

        [A2] Returns 0.0 safely when either input is a zero vector.
        """
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-8 or n2 < 1e-8:   # [A2]
            return 0.0
        W = np.outer(v1, v2) - np.outer(v2, v1)
        return float(np.linalg.norm(W, ord="fro"))

    @staticmethod
    def get_seniority(role: str) -> float:
        """Map an insider job title to a [0, 1] seniority weight."""
        rl = role.lower()
        for key, weight in ROLE_SENIORITY.items():
            if key.lower() in rl:
                return weight
        return 0.55

    @staticmethod
    def interpret_similarity(sim: float) -> str:
        if sim >= 0.90: return "🔴 EXTREMELY HIGH — mirrors known accumulation template"
        if sim >= 0.75: return "🟠 HIGH — strong pattern alignment with suspicious flows"
        if sim >= 0.55: return "🟡 MODERATE — partial alignment, monitor closely"
        if sim >= 0.35: return "🔵 LOW — weak signal, likely routine activity"
        return "⚪ NOISE — no meaningful alignment with high-conviction template"


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — NLP SENTIMENT ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class SentimentEngine:
    """
    Financial-domain dictionary-based sentiment scorer.

    Calibrated against earnings call transcripts and SEC filing summaries.
    Produces a continuous score in [-1.0, +1.0] without heavy transformer
    models — fast enough for real-time batch analysis.

    Architecture:
      1. Multi-word phrase matching (highest priority)
      2. Single-token scoring with negation propagation
      3. Square-root dampening to prevent saturation on long documents
      4. Clip to [-1, 1]
    """

    _LEXICON: dict[str, float] = {
        # ── Strong Positive (+0.40) ─────────────────────────────────────────
        "beat":               +0.40, "beats":              +0.40,
        "breakout":           +0.40, "insider buy":        +0.40,
        "substantial growth": +0.40, "record high":        +0.40,
        "exceeded":           +0.40, "surpassed":          +0.38,
        "above expectations": +0.40, "fda approval":       +0.40,
        "all-time high":      +0.40, "guidance raised":    +0.40,
        "raised guidance":    +0.40, "outperform":         +0.38,
        "upgrade":            +0.38, "upgraded":           +0.38,
        "soared":             +0.36, "record":             +0.30,
        # ── Moderate Positive (+0.20) ───────────────────────────────────────
        "dividend":           +0.20, "acquisition":        +0.22,
        "expansion":          +0.20, "growth":             +0.20,
        "strong":             +0.18, "profit":             +0.20,
        "bullish":            +0.25, "momentum":           +0.18,
        "rally":              +0.20, "partnership":        +0.18,
        "approval":           +0.22, "buyback":            +0.20,
        "recovery":           +0.18, "innovation":         +0.15,
        "confident":          +0.15, "repurchase":         +0.20,
        "milestone":          +0.18, "robust":             +0.18,
        "resilient":          +0.15, "accelerating":       +0.20,
        # ── Moderate Negative (−0.20) ───────────────────────────────────────
        "insider sell":       -0.20, "decline":            -0.20,
        "restructuring":      -0.20, "cautious":           -0.18,
        "concern":            -0.18, "headwind":           -0.20,
        "slowdown":           -0.22, "miss":               -0.30,
        "missed":             -0.30, "disappointing":      -0.25,
        "below":              -0.15, "weak":               -0.20,
        "pressure":           -0.18, "uncertainty":        -0.18,
        "challenging":        -0.18, "declining":          -0.22,
        # ── Strong Negative (−0.40) ─────────────────────────────────────────
        "lawsuit":            -0.40, "investigation":      -0.40,
        "sec inquiry":        -0.40, "fraud":              -0.45,
        "bankruptcy":         -0.50, "crashed":            -0.40,
        "scandal":            -0.40, "class action":       -0.38,
        "default":            -0.42, "layoffs":            -0.30,
        "guidance cut":       -0.38, "downgrade":          -0.35,
        "downgraded":         -0.35, "plunged":            -0.38,
        "bearish":            -0.30, "collapse":           -0.42,
        "recall":             -0.30, "probe":              -0.35,
        "loss":               -0.22, "losses":             -0.25,
        "write-down":         -0.35, "impairment":         -0.32,
        "layoff":             -0.30, "fired":              -0.28,
    }

    _NEGATORS: frozenset = frozenset({
        "not", "no", "never", "neither", "nor", "without", "lack",
        "lacking", "failed", "unable", "cannot", "don't", "doesn't",
        "didn't", "hasn't", "haven't", "isn't", "wasn't", "wouldn't",
        "couldn't", "shouldn't", "fail", "fails",
    })

    def analyze_text(self, text: str) -> float:
        """Score financial text on the continuous scale [-1.0, +1.0]."""
        if not text or not text.strip():
            return 0.0

        text_l = text.lower()
        total = 0.0
        hits = 0

        # Phase 1: multi-word phrase matching (greedy, highest priority)
        for phrase, weight in self._LEXICON.items():
            if " " in phrase and phrase in text_l:
                total += weight
                hits += 1

        # Phase 2: single-token with negation propagation
        tokens = text_l.replace(",", " ").replace(".", " ").replace("!", " ").split()
        negate = False
        for tok in tokens:
            if tok in self._NEGATORS:
                negate = True
                continue
            if tok in self._LEXICON and " " not in tok:
                pol = self._LEXICON[tok]
                total += (-pol if negate else pol)
                hits += 1
            negate = False

        if hits == 0:
            return 0.0
        raw = total / math.sqrt(max(hits, 1))
        return float(np.clip(raw, -1.0, 1.0))

    @staticmethod
    def grade(score: float) -> str:
        if score >= 0.50:
            return "VERY BULLISH"
        if score >= 0.20:
            return "BULLISH"
        if score >= -0.20:
            return "NEUTRAL"
        if score >= -0.50:
            return "BEARISH"
        return "VERY BEARISH"

    def analyse(self, ticker: str) -> SimpleNamespace:
        """Return a fallback neutral sentiment result for the ticker."""
        score = 0.0
        return SimpleNamespace(
            ticker=ticker.upper(),
            score=score,
            magnitude=0.0,
            grade=self.grade(score),
            themes=[],
            headlines=[],
            articles=0,
            raw_scores=[],
        )

    def analyse_batch(self, tickers: list[str]) -> dict[str, SimpleNamespace]:
        return {ticker.upper(): self.analyse(ticker) for ticker in tickers}


# ═════════════════════════════════════════════════════════════════════════════
# PART 4 — CUSTOM GYMNASIUM TRADING ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════

class InsiderTradingEnv(gym.Env):
    """
    Custom Gymnasium environment combining OHLCV price data with
    multi-dimensional alternative signals.

    ── Observation Space (12-dim continuous Box, strictly np.float32) ─────
    [0] log_return      log(P_t / P_{t-1})
    [1] rsi             RSI-14 scaled to [0, 1]
    [2] macd            Normalized MACD signal ∈ [-1, 1]
    [3] insider_sim     Cosine similarity to insider template ∈ [-1, 1]
    [4] congress_sim    Cosine similarity to congress template ∈ [-1, 1]
    [5] sentiment       NLP score ∈ [-1, 1]
    [6] gex             Gamma Exposure (normalized) ∈ [-5, 5]
    [7] position_flag   1.0 if holding shares, 0.0 if flat
    [8] sma50_dev       Normalized deviation from 50-day SMA ∈ [-1, 1]
    [9] sma200_dev      Normalized deviation from 200-day SMA ∈ [-1, 1]
    [10] high_52w_pct   Close / 52-week high ∈ [0, 1]
    [11] vol_trend      Volume trend indicator ∈ [0, 5]

    ── Action Space (Discrete 3) ─────────────────────────────────────────
    0 — SELL / Liquidate to cash
    1 — HOLD current position
    2 — BUY All-In (applies 0.15% transaction fee)

    ── Gymnasium Compliance [A3] ─────────────────────────────────────────
    • reset() calls super().reset(seed=seed) and returns (obs, {})
    • step() returns the full 5-tuple (obs, reward, terminated, truncated, info)
      with terminated / truncated correctly distinguished
    • Environment starts at step INDICATOR_WARMUP (26) so that RSI-14 and
      MACD-26 are fully initialized before any observation is returned

    ── Reward Function [A4] ─────────────────────────────────────────────
    R_t = PnL_t  −  α·ChurnPenalty_t  +  β·AlignmentBonus_t  −  γ·DrawdownPenalty_t

    Coefficient scale (post-fix):
      α = 0.002  (was 0.020 — would have dominated daily PnL and caused
                  the agent to lock into a cash position to dodge fees)
      β = 0.015, γ = 0.05

    ── Fee Order-of-Operations [A5] ──────────────────────────────────────
    Transaction fees are deducted from the CURRENT portfolio value BEFORE
    the next-period price return is applied.  Execution sequence:
      1. Detect position change
      2. Calculate fee from V_{t-1}
      3. V'_{t-1} = V_{t-1} − fee
      4. Apply next-period return to V'_{t-1}
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        alt_signals: list[dict],
        initial_balance: float = 100_000.0,
        fee_pct: float = FEE_PCT,
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.alt_signals = alt_signals
        self.n = len(df)
        self.initial_balance = initial_balance
        self.fee_pct = fee_pct

        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -5.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([ 1.0, 1.0,  1.0,  1.0,  1.0,  1.0,  5.0, 1.0,  1.0,  1.0, 1.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # State variables (initialized by reset)
        self.balance: float         = initial_balance
        self.shares: float          = 0.0
        self.current_step: int      = INDICATOR_WARMUP  # [A3]
        self.prev_action: int       = 1
        self.peak_value: float      = initial_balance
        self.portfolio_value: float = initial_balance
        self.entry_price: float | None = None
        self.stop_price: float | None = None
        self.history: list[dict]    = []

    # ── [A3] reset ────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to its initial state.

        [A3] Starts at step INDICATOR_WARMUP (26) so that RSI-14 and
        MACD-26 arrays have a valid warm-up window.  Without this, the
        first INDICATOR_WARMUP observations contain NaN indicators which
        corrupt the SB3 policy network's gradient updates.

        Returns (observation: np.float32 array of shape (12,), info: dict).
        """
        super().reset(seed=seed)   # initializes self.np_random
        self.balance         = self.initial_balance
        self.shares          = 0.0
        self.current_step    = INDICATOR_WARMUP  # [A3]
        self.prev_action     = 1
        self.peak_value      = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.entry_price     = None
        self.stop_price      = None
        self.history         = []
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        """Build strictly typed np.float32 observation of shape (12,)."""
        idx = min(self.current_step, self.n - 1)
        row = self.df.iloc[idx]
        alt = self.alt_signals[idx % len(self.alt_signals)]

        sma50_dev = float(np.clip((row.get("Close", 0.0) / max(row.get("sma50", 1.0), 1.0) - 1.0), -0.20, 0.20) / 0.20)
        sma200_dev = float(np.clip((row.get("Close", 0.0) / max(row.get("sma200", 1.0), 1.0) - 1.0), -0.20, 0.20) / 0.20)
        high_52w_pct = float(np.clip(row.get("high_52w_pct", 0.0), 0.0, 1.0))
        vol_trend = float(np.clip(row.get("vol_trend", 1.0), 0.0, 5.0))

        obs = np.array([
            float(row.get("log_return", 0.0)),
            float(np.clip(float(row.get("rsi", 50.0)) / 100.0,  0.0, 1.0)),
            float(np.clip(row.get("macd", 0.0),                 -1.0, 1.0)),
            float(np.clip(alt.get("insider_sim",  0.0),         -1.0, 1.0)),
            float(np.clip(alt.get("congress_sim", 0.0),         -1.0, 1.0)),
            float(np.clip(alt.get("sentiment",    0.0),         -1.0, 1.0)),
            float(np.clip(alt.get("gex",          0.0),         -5.0,  5.0)),
            1.0 if self.shares > 0.0 else 0.0,
            sma50_dev,
            sma200_dev,
            high_52w_pct,
            vol_trend,
        ], dtype=np.float32)

        # Ensure no NaN/Inf leaks — replace with 0.0 (safe neutral default)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    # ── [A5] step ─────────────────────────────────────────────────────────

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance the simulation by one timestep.

        [A5] Fee deduction order of operations:
          ① Detect position change
          ② Deduct transaction fee from current portfolio value BEFORE
             applying the next period's price return
          ③ Apply price return to fee-adjusted portfolio value

        [A3] terminated = we have exhausted the full price DataFrame (true
             terminal state of the MDP).  truncated = portfolio has been
             ruined (risk stop at 15% of initial capital) — the episode
             ends early but is NOT a true terminal MDP state.

        Returns (obs, reward, terminated, truncated, info).
        """
        idx   = min(self.current_step, self.n - 1)
        price = float(self.df.iloc[idx]["Close"])
        alt   = self.alt_signals[idx % len(self.alt_signals)]

        fee_paid = 0.0
        stop_loss_triggered = False

        # ── Stop-loss override ─────────────────────────────────────────────
        if self.shares > 0.0 and self.entry_price is not None:
            if price <= self.stop_price or price <= self.entry_price * (1.0 - STOP_LOSS_PCT):
                stop_loss_triggered = True

        if stop_loss_triggered:
            gross         = self.shares * price
            fee_paid      = gross * self.fee_pct
            self.balance  += gross - fee_paid
            self.shares    = 0.0
            self.entry_price = None
            self.stop_price  = None
        else:
            if action == 2 and self.shares == 0.0 and self.balance > 0.0:  # BUY
                available_equity = max(self.portfolio_value, self.balance)
                purchase_value = min(self.balance, available_equity * MAX_TRADE_EXPOSURE_PCT)
                fee_paid = purchase_value * self.fee_pct
                net_capital = purchase_value - fee_paid
                if net_capital > 0.0 and price > 0.0:
                    self.shares = net_capital / price
                    self.balance -= purchase_value
                    self.entry_price = price
                    self.stop_price = self.entry_price * (1.0 - STOP_LOSS_PCT)
            elif action == 0 and self.shares > 0.0:  # SELL
                gross         = self.shares * price
                fee_paid      = gross * self.fee_pct
                self.balance  += gross - fee_paid
                self.shares    = 0.0
                self.entry_price = None
                self.stop_price  = None

        prev_value           = self.portfolio_value
        self.portfolio_value = self.balance + self.shares * price
        self.peak_value      = max(self.peak_value, self.portfolio_value)

        # ── Reward decomposition [A4] ─────────────────────────────────────
        pnl = (self.portfolio_value - prev_value) / max(prev_value, 1e-9)
        reward = pnl

        # 1. Churn penalty [A4]
        churn = 0.0
        if (action in (0, 2) and self.prev_action in (0, 2)
                and action != self.prev_action):
            churn = ALPHA_CHURN
        reward -= churn

        # 2. Alignment bonus — only when holding AND insider_sim exceeds threshold
        ins_sim     = float(alt.get("insider_sim", 0.0))
        align_bonus = 0.0
        if self.shares > 0.0 and ins_sim > ALIGN_THRESHOLD:
            align_bonus = BETA_ALIGNMENT * ins_sim
        reward += align_bonus

        # 3. Drawdown penalty — penalise holding through steep portfolio declines
        dd         = (self.peak_value - self.portfolio_value) / max(self.peak_value, 1e-9)
        dd_penalty = 0.0
        if dd > DRAWDOWN_THRESHOLD and self.shares > 0.0:
            dd_penalty = GAMMA_DRAWDOWN * (dd - DRAWDOWN_THRESHOLD)
        reward -= dd_penalty

        reward = float(np.clip(reward, -5.0, 5.0))

        self.history.append({
            "step":            self.current_step,
            "portfolio_value": self.portfolio_value,
            "action":          action,
            "price":           price,
            "pnl":             pnl,
            "insider_sim":     ins_sim,
            "reward":          reward,
            "churn_penalty":   churn,
            "align_bonus":     align_bonus,
            "dd_penalty":      dd_penalty,
            "fee_paid":        fee_paid,
            "stop_loss":       stop_loss_triggered,
        })

        self.prev_action   = action
        self.current_step += 1

        terminated = self.current_step >= self.n - 1
        truncated  = self.portfolio_value <= self.initial_balance * 0.15

        return self._obs(), reward, terminated, truncated, {}

    def render(self, mode: str = "human") -> None:
        print(f"[Step {self.current_step}] NAV=${self.portfolio_value:,.2f}  "
              f"Shares={self.shares:.2f}  Peak=${self.peak_value:,.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# DATA GENERATION UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI-14 computed as EWM.  Fully vectorised — no per-row look-ahead."""
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def generate_market_data(ticker: str, n: int = 400) -> pd.DataFrame:
    """
    Realistic synthetic OHLCV with derived technical indicators.
    Seed is deterministic per ticker so results are reproducible.

    [A3] The first INDICATOR_WARMUP rows will have partially-initialized
    indicator values.  The environment skips them via its warm-up offset.
    """
    seed = sum(ord(c) for c in ticker)
    rng  = np.random.default_rng(seed)

    drift  = rng.uniform(0.06, 0.20)
    vol    = rng.uniform(1.0,  2.8)
    prices = 100.0 + np.cumsum(rng.normal(drift, vol, n))
    prices = np.maximum(prices, 5.0)

    dates = pd.date_range(end=datetime.date.today(), periods=n, freq="B")
    df = pd.DataFrame({
        "Date":   dates,
        "Close":  prices,
        "Open":   prices * rng.uniform(0.996, 1.004, n),
        "High":   prices * rng.uniform(1.003, 1.018, n),
        "Low":    prices * rng.uniform(0.982, 0.997, n),
        "Volume": rng.integers(500_000, 5_000_000, n).astype(float),
    })

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)
    df["rsi"]  = _rsi_series(df["Close"])

    ema12     = df["Close"].ewm(span=12, adjust=False).mean()
    ema26     = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    sig_line  = macd_line.ewm(span=9, adjust=False).mean()
    raw_macd  = (macd_line - sig_line) / df["Close"]
    df["macd"] = raw_macd.clip(-0.05, 0.05) / 0.05   # normalize to [-1, 1]

    return df.fillna(0.0)


def generate_alt_signals(ticker: str, n: int) -> list[dict]:
    """
    Synthetic alternative data signals for each simulation timestep.
    Mimics the output that would come from live SEC/Capitol-Trades scrapers.

    [A2] All feature vectors are constructed via build_feature_vector which
    clips all dimensions to [0, 1], preventing zero-norm edge cases on most
    days.  The cosine_similarity guard handles any remaining zero-vectors.
    """
    seed = sum(ord(c) for c in ticker) + 7
    rng  = np.random.default_rng(seed)

    out: list[dict] = []
    for i in range(n):
        vol_pct   = float(np.clip(rng.beta(2, 5) + 0.3 * math.sin(i / 30), 0.0, 1.0))
        seniority = float(rng.choice(list(ROLE_SENIORITY.values())))
        gex       = float(rng.normal(0, 1.5))
        gex_norm  = float(np.clip(gex / 5.0 + 0.5, 0.0, 1.0))
        sentiment = float(np.clip(math.sin(i / 40) * 0.5 + rng.normal(0, 0.3), -1.0, 1.0))

        v_ins  = SignalGeometry.build_feature_vector(vol_pct, seniority, gex_norm, sentiment)
        v_cong = SignalGeometry.build_feature_vector(
            min(vol_pct + 0.1, 1.0),
            min(seniority + 0.05, 1.0),
            min(gex_norm + 0.05, 1.0),
            min(sentiment + 0.05, 1.0),
        )

        out.append({
            "insider_sim":  SignalGeometry.cosine_similarity(v_ins,  INSIDER_TEMPLATE),
            "congress_sim": SignalGeometry.cosine_similarity(v_cong, CONGRESS_TEMPLATE),
            "sentiment":    sentiment,
            "gex":          gex,
            "vol_pct":      vol_pct,
            "seniority":    seniority,
            "v_insider":    v_ins,
            "v_congress":   v_cong,
        })
    return out


def _resolve_rl_sentiment(
    ticker: str,
    db: DBManager,
    sentiment_processor: SentimentEngine | SentimentProcessor,
) -> float:
    try:
        sentiment_result = sentiment_processor.analyse(ticker)
        return float(np.clip(sentiment_result.score if sentiment_result is not None else 0.0, -1.0, 1.0))
    except Exception as exc:
        logger.warning("RL sentiment fetch failed for %s: %s", ticker, exc)
        persisted = db.get_news_sentiment(ticker, days_back=14)
        if persisted:
            return float(np.clip(persisted[0].get("score", 0.0), -1.0, 1.0))
        return 0.0


def _fetch_trades_with_start_date_compat(
    method: callable,
    ticker: str,
    start_date: str | None,
    days_back: int = 365,
) -> list[dict]:
    try:
        return method(
            ticker,
            days_back=days_back,
            purchases_only=True,
            start_date=start_date,
        )
    except TypeError as exc:
        logger.warning(
            "DB compatibility fallback for %s: %s",
            getattr(method, "__name__", "unknown"),
            exc,
        )
        rows = method(ticker, days_back=days_back, purchases_only=True)
        if start_date is not None:
            return [row for row in rows if row.get("transaction_date", "") >= start_date]
        return rows


def build_live_alt_signals(
    ticker: str,
    df: pd.DataFrame,
    db: DBManager | None = None,
    sentiment_processor: SentimentProcessor | None = None,
    start_date: str | None = None,
) -> list[dict]:
    """Build live alternative feature signals for a daily market history."""
    db = db or DBManager()
    sentiment_processor = sentiment_processor or SentimentProcessor()
    sentiment_score = _resolve_rl_sentiment(ticker, db, sentiment_processor)

    insider_trades = _fetch_trades_with_start_date_compat(
        db.get_insider_trades,
        ticker,
        start_date=start_date,
        days_back=365,
    )
    congress_trades = _fetch_trades_with_start_date_compat(
        db.get_congress_trades,
        ticker,
        start_date=start_date,
        days_back=365,
    )

    out: list[dict] = []
    for _, row in df.iterrows():
        date_str = (
            row["Date"].strftime("%Y-%m-%d")
            if hasattr(row["Date"], "strftime")
            else str(row["Date"])
        )
        recent_insiders = [
            t
            for t in insider_trades
            if t.get("transaction_date", "") <= date_str
        ]
        recent_congress = [
            t
            for t in congress_trades
            if t.get("transaction_date", "") <= date_str
        ]

        insider_count = len(recent_insiders)
        congress_count = len(recent_congress)
        insider_value = sum(float(t.get("total_value", 0.0) or 0.0) for t in recent_insiders)
        congress_value = sum(float(t.get("amount_midpoint", 0.0) or 0.0) for t in recent_congress)
        volume_pct = float(np.clip(min(insider_count / 8.0 + congress_count / 12.0, 1.0), 0.0, 1.0))
        seniority = float(np.clip(insider_value / 2_500_000.0, 0.0, 1.0))
        congress_strength = float(np.clip(congress_value / 3_000_000.0, 0.0, 1.0))
        gex = float(np.clip((row.get("vol_trend", 1.0) - 1.0) * 2.0, -5.0, 5.0))
        sentiment_score = float(np.clip(sentiment_score, -1.0, 1.0))

        v_ins = SignalGeometry.build_feature_vector(volume_pct, seniority, gex, sentiment_score)
        v_cong = SignalGeometry.build_feature_vector(
            float(np.clip(congress_count / 6.0 + congress_strength * 0.2, 0.0, 1.0)),
            min(congress_strength + 0.05, 1.0),
            gex,
            sentiment_score,
        )

        out.append({
            "insider_sim":  SignalGeometry.cosine_similarity(v_ins,  INSIDER_TEMPLATE),
            "congress_sim": SignalGeometry.cosine_similarity(v_cong, CONGRESS_TEMPLATE),
            "sentiment": sentiment_score,
            "gex": float(np.clip((row.get("vol_trend", 1.0) - 1.0) * 2.0, -5.0, 5.0)),
            "vol_pct": volume_pct,
            "seniority": seniority,
            "insider_count": float(np.clip(insider_count / 8.0, 0.0, 1.0)),
            "congress_count": float(np.clip(congress_count / 8.0, 0.0, 1.0)),
            "insider_value_norm": float(np.clip(insider_value / 2_000_000.0, 0.0, 1.0)),
            "congress_value_norm": float(np.clip(congress_value / 3_000_000.0, 0.0, 1.0)),
            "v_insider":    v_ins,
            "v_congress":   v_cong,
        })

    return out


def build_trading_environment(
    ticker: str,
    n_days: int = 400,
    db: DBManager | None = None,
    sentiment_processor: SentimentEngine | None = None,
    start_date: datetime.date | None = None,
) -> tuple[InsiderTradingEnv | None, pd.DataFrame, list[dict]]:
    """Construct an environment using Yahoo price history and live alternative signals."""
    df = fetch_price_history(ticker, period="max", interval="1d")
    if df.empty:
        logger.error("Live price history unavailable for %s; real-data-only mode cannot proceed.", ticker)
        return None, pd.DataFrame(), []
    if start_date is not None:
        df = df[df["Date"] >= start_date].reset_index(drop=True)
        if df.empty:
            logger.error(
                "Price history start date %s not available for %s; real-data-only mode cannot proceed.",
                start_date,
                ticker,
            )
            return None, pd.DataFrame(), []
    df = df.tail(n_days).reset_index(drop=True)

    alts = build_live_alt_signals(
        ticker,
        df,
        db=db,
        sentiment_processor=sentiment_processor,
        start_date=start_date.isoformat() if start_date is not None else None,
    )
    if len(alts) < len(df):
        alts = alts + [alts[-1]] * (len(df) - len(alts)) if alts else generate_alt_signals(ticker, len(df))
    return InsiderTradingEnv(df, alts), df, alts


# ═════════════════════════════════════════════════════════════════════════════
# PERFORMANCE ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(history: list[dict], initial_balance: float) -> dict:
    """Annualized Sharpe, Sortino, Max Drawdown, total return, and win-rate."""
    if len(history) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0,
                "total_return": 0.0, "win_rate": 0.0}

    vals   = np.array([h["portfolio_value"] for h in history], dtype=np.float64)
    rets   = np.diff(vals) / np.maximum(vals[:-1], 1e-9)

    mean_r = float(rets.mean())
    std_r  = float(rets.std())
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 1e-9 else 0.0

    down   = rets[rets < 0]
    dstd   = float(down.std()) if len(down) > 1 else 1e-9
    sortino= (mean_r / dstd * math.sqrt(252)) if dstd > 1e-9 else 0.0

    running_max = np.maximum.accumulate(vals)
    dd_series   = (running_max - vals) / np.maximum(running_max, 1e-9)
    max_dd      = float(dd_series.max())

    total_ret = (vals[-1] - initial_balance) / initial_balance
    win_rate  = float((rets > 0).mean())

    return {
        "sharpe":       round(sharpe, 3),
        "sortino":      round(sortino, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "total_return": round(total_ret * 100, 2),
        "win_rate":     round(win_rate * 100, 1),
    }


def _recommend_strategy_change(
    metrics: dict[str, float],
    train_steps: int,
    n_days: int,
    learning_rate: float,
    entropy_coef: float,
) -> tuple[str, str]:
    """Generate a short recommendation and supporting notes for the next session."""
    if metrics["total_return"] < 0.0:
        suggestion = (
            "Review signal alignment and tighten risk management; the current policy is losing money."
        )
        notes = (
            f"Total return was {metrics['total_return']:+.1f}%. Consider lowering exposure, "
            f"increasing the holdout window, or reducing the learning rate from {learning_rate:.5f}."
        )
    elif metrics["sharpe"] < 0.50:
        suggestion = (
            "Sharpe is low; consider stronger regularization or more conservative training."
        )
        notes = (
            f"Sharpe={metrics['sharpe']:.3f} with {metrics['max_drawdown']:.1f}% drawdown. "
            f"Try reducing entropy coefficient from {entropy_coef:.2f} or adding additional risk features."
        )
    elif metrics["max_drawdown"] > 15.0:
        suggestion = (
            "Drawdown is elevated; add defensive filters or shorten the backtest horizon."
        )
        notes = (
            f"Max drawdown was {metrics['max_drawdown']:.1f}%. Consider trimming the historical window from "
            f"{n_days} days or introducing a tighter stop-loss filter in the environment."
        )
    elif metrics["win_rate"] < 45.0:
        suggestion = (
            "Win rate is modest; review alternative signal coverage and feature normalization."
        )
        notes = (
            f"Win rate was {metrics['win_rate']:.0f}%. Verify that insider/congress signals are being sourced correctly and "
            "that the observation vector includes meaningful regime information."
        )
    else:
        suggestion = (
            "Continue with the current setup and collect additional training history."
        )
        notes = (
            f"Performance looks stable: total return {metrics['total_return']:+.1f}%, Sharpe {metrics['sharpe']:.3f}. "
            f"You can extend training beyond {train_steps:,} timesteps if compute budget allows."
        )

    return suggestion, notes


def _render_autonomous_training_report(report: dict[str, object]) -> None:
    if not report or "iterations" not in report:
        st.info("No autonomous tuning report is available yet.")
        return

    st.markdown("### 🤖 Autonomous Local LLM Tuning Report")
    iterations = report.get("iterations", [])
    summary_rows: list[dict[str, object]] = []
    for item in iterations:
        summary_rows.append({
            "Iteration": item.get("iteration"),
            "Sharpe": item["metrics"].get("sharpe", 0.0),
            "Return": item["metrics"].get("total_return", 0.0),
            "Drawdown": item["metrics"].get("max_drawdown", 0.0),
            "Suggested Change": item["recommendation"].get("suggested_change", ""),
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("#### Final recommended configuration")
    final_config = report.get("final_config", {})
    st.write({
        "training_steps": final_config.get("training_steps"),
        "learning_rate": final_config.get("learning_rate"),
        "entropy_coef": final_config.get("entropy_coef"),
        "backtest_days": final_config.get("backtest_days"),
    })

    if report.get("stopped_on_plateau"):
        st.warning(report.get("stopped_reason", "Autonomous tuning stopped early due to plateau."))
    else:
        st.success(report.get("stopped_reason", "Autonomous tuning completed."))


def _plot_session_comparison(
    current_metrics: dict[str, float],
    previous_metrics: dict[str, float],
) -> None:
    df = pd.DataFrame({
        "Metric": ["Total Return", "Sharpe", "Sortino", "Max Drawdown", "Win Rate"],
        "Current Session": [
            current_metrics.get("total_return", 0.0),
            current_metrics.get("sharpe", 0.0),
            current_metrics.get("sortino", 0.0),
            current_metrics.get("max_drawdown", 0.0),
            current_metrics.get("win_rate", 0.0),
        ],
        "Previous Session": [
            previous_metrics.get("total_return", 0.0),
            previous_metrics.get("sharpe", 0.0),
            previous_metrics.get("sortino", 0.0),
            previous_metrics.get("max_drawdown", 0.0),
            previous_metrics.get("win_rate", 0.0),
        ],
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Metric"], y=df["Current Session"], name="Current Session",
        marker_color="#00FFCC",
    ))
    fig.add_trace(go.Bar(
        x=df["Metric"], y=df["Previous Session"], name="Previous Session",
        marker_color="#FF9900",
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Session Metric Comparison",
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Value",
        legend=dict(orientation="h", y=-0.20),
        margin=dict(t=60, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_training_metrics_map(
    current_metrics: dict[str, float],
    prior_sessions: list[dict],
) -> None:
    rows = []
    for prior in prior_sessions:
        m = prior.get("performance_metrics", {}) or {}
        rows.append({
            "Session": prior.get("created_at", "unknown"),
            "Sharpe": m.get("sharpe", 0.0),
            "Return": m.get("total_return", 0.0),
            "Drawdown": m.get("max_drawdown", 0.0),
            "Win Rate": m.get("win_rate", 0.0),
            "Type": "Prior",
        })
    rows.append({
        "Session": "Current",
        "Sharpe": current_metrics.get("sharpe", 0.0),
        "Return": current_metrics.get("total_return", 0.0),
        "Drawdown": current_metrics.get("max_drawdown", 0.0),
        "Win Rate": current_metrics.get("win_rate", 0.0),
        "Type": "Current",
    })
    df = pd.DataFrame(rows)

    fig = px.scatter(
        df,
        x="Sharpe",
        y="Return",
        size="Win Rate",
        color="Type",
        hover_data=["Session", "Drawdown"],
        title="Training Performance Map: Current vs Prior Sessions",
        template="plotly_dark",
        color_discrete_map={"Current": "#00FFCC", "Prior": "#FF9900"},
    )
    fig.update_layout(margin=dict(t=60, b=40), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)


def _normalize_metrics(metrics: dict[str, float] | None) -> dict[str, float]:
    safe = metrics or {}
    return {
        "total_return": float(safe.get("total_return", 0.0)),
        "sharpe": float(safe.get("sharpe", 0.0)),
        "sortino": float(safe.get("sortino", 0.0)),
        "max_drawdown": float(safe.get("max_drawdown", 0.0)),
        "win_rate": float(safe.get("win_rate", 0.0)),
    }


def _get_best_prior_session(prior_sessions: list[dict]) -> dict | None:
    if not prior_sessions:
        return None
    return max(
        prior_sessions,
        key=lambda s: _normalize_metrics(s.get("performance_metrics", {}))["sharpe"],
    )


def _plot_session_timeline(
    prior_sessions: list[dict],
    current_metrics: dict[str, float],
) -> None:
    rows: list[dict[str, Any]] = []
    for prior in prior_sessions:
        m = _normalize_metrics(prior.get("performance_metrics", {}))
        rows.append({
            "Date": pd.to_datetime(prior.get("created_at", ""), errors="coerce"),
            "Sharpe": m["sharpe"],
            "Return": m["total_return"],
            "Session": prior.get("created_at", "prior"),
            "Type": "Prior",
        })
    rows.append({
        "Date": pd.Timestamp.now(),
        "Sharpe": current_metrics.get("sharpe", 0.0),
        "Return": current_metrics.get("total_return", 0.0),
        "Session": "Current",
        "Type": "Current",
    })
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No session history is available for timeline visualization.")
        return

    fig = px.line(
        df.sort_values("Date"),
        x="Date", y="Sharpe", color="Type", markers=True,
        title="Session Sharpe Timeline",
        template="plotly_dark",
        labels={"Sharpe": "Sharpe Ratio", "Date": "Training Date"},
    )
    fig.update_layout(margin=dict(t=60, b=40), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)


def _session_history_dataframe(prior_sessions: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session in prior_sessions:
        metrics = _normalize_metrics(session.get("performance_metrics", {}))
        rows.append({
            "Created": session.get("created_at", ""),
            "Strategy": session.get("strategy_name", ""),
            "Training Steps": session.get("training_steps", 0),
            "Learning Rate": session.get("learning_rate", 0.0),
            "Entropy Coef": session.get("entropy_coef", 0.0),
            "Backtest Days": session.get("backtest_days", 0),
            "Total Return %": metrics["total_return"],
            "Sharpe": metrics["sharpe"],
            "Sortino": metrics["sortino"],
            "Max Drawdown %": metrics["max_drawdown"],
            "Win Rate %": metrics["win_rate"],
            "Suggested Change": session.get("suggested_change", ""),
            "Notes": session.get("notes", ""),
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# PART 5 — STREAMLIT DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

def run_app() -> None:
    st.set_page_config(
        page_title="InsiderRL v2 — Alternative Market Intelligence",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Dark-mode CSS ─────────────────────────────────────────────────────
    st.markdown("""
    <style>
    body, .stApp { background-color: #0d1117; color: #c9d1d9; }
    div[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    .stMetric { background:#161b22; border:1px solid #30363d;
                border-radius:8px; padding:14px 18px; }
    .stMetric label { color:#8b949e !important; font-size:.78rem; }
    .block-container { padding-top:1rem; max-width:100%; }
    h1,h2,h3 { color:#e6edf3 !important; }
    .info-pill { background:#161b22; border:1px solid #30363d; border-radius:6px;
                 padding:10px 14px; font-size:.82rem; color:#8b949e;
                 font-style:italic; margin-bottom:10px; }
    .audit-badge { background:#1a2332; border:1px solid #2d5a9e; border-radius:6px;
                   padding:8px 12px; font-size:.80rem; color:#79b8ff; margin-bottom:6px; }
    </style>
    """, unsafe_allow_html=True)

    # ── [A6] Session-state initialization ────────────────────────────────
    # Trained model, backtest results, and PCA states are stored in
    # session_state so that Streamlit widget interactions do NOT restart
    # the PPO training loop.  The training only runs when the user
    # explicitly clicks "Train Policy Network".
    for key, default in [
        ("trained_model",   None),
        ("backtest_hist",   None),
        ("pca_states",      None),
        ("metrics",         None),
        ("train_ticker",    None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    live_db = create_db_adapter(
        MarketIntelligenceDB,
        DB_PATH,
        validate_methods=(
            "get_congress_trades",
            "get_insider_trades",
            "get_news_sentiment",
            "get_stats",
            "get_disclosed_signals_on_date",
            "upsert_signal_metadata",
            "get_ticker_history",
        ),
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        ticker = st.selectbox("📈 Equity Target", TICKERS)
        law_passage_date = st.date_input(
            "Training start date",
            value=datetime.date(2023, 1, 1),
            min_value=datetime.date(2000, 1, 1),
            max_value=datetime.date.today(),
            help="Choose the date when the disclosure law was passed or became effective.",
        )

        st.markdown("### 🌱 Seed / Watchlist")
        watchlist = live_db.get_watchlist_tickers()
        seed_ticker_input = st.text_input(
            "Ticker to add to the seed/watchlist",
            value="",
            key="watchlist_seed_input",
        )

        if st.button("➕ Add ticker to seed watchlist", use_container_width=True):
            if live_db.add_watchlist_ticker(seed_ticker_input, source="manual"):
                st.success(f"Saved {seed_ticker_input.upper()} to the seed watchlist.")
            else:
                st.error("Unable to save ticker. Please enter a valid symbol.")

        recent_days = st.slider(
            "Recent insider/congress hit window (days)",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
        )
        if st.button("⚡ Seed recent insider/congress hit tickers", use_container_width=True):
            hits = live_db.get_recent_hit_tickers(days_back=recent_days)
            added = 0
            for symbol in hits:
                if live_db.add_watchlist_ticker(symbol, source="recent_hit"):
                    added += 1
            st.success(f"Added {added} recent insider/congress hit tickers to the seed watchlist.")

        if watchlist:
            selected_watchlist = st.multiselect(
                "Saved seed/watchlist tickers",
                [row["ticker"] for row in watchlist],
                key="watchlist_selected",
            )
            if st.button("🗑️ Remove selected tickers", use_container_width=True):
                removed = sum(live_db.remove_watchlist_ticker(symbol) for symbol in selected_watchlist)
                st.success(f"Removed {removed} tickers from the seed watchlist.")
            if st.button("🧹 Clear seed watchlist", use_container_width=True):
                removed = live_db.clear_watchlist()
                st.warning(f"Cleared {removed} tickers from the seed watchlist.")
            st.caption(f"Seed watchlist has {len(watchlist)} saved tickers.")
        else:
            st.info("No saved seed/watchlist tickers yet. Add one manually or seed recent hits.")

        st.markdown("---")
        st.markdown("### 🎯 Focus Parameters")
        focus_settings = live_db.get_focus_settings()

        insider_flow_weight = st.slider(
            "Insider signal weight",
            min_value=0.0,
            max_value=3.0,
            value=float(focus_settings.get("insider_flow_weight", 1.0)),
            step=0.1,
            help="Relative weight the agent places on insider flow signals.",
        )
        congress_flow_weight = st.slider(
            "Congress signal weight",
            min_value=0.0,
            max_value=3.0,
            value=float(focus_settings.get("congress_flow_weight", 1.0)),
            step=0.1,
            help="Relative weight the agent places on congressional buy flow.",
        )
        momentum_horizon_days = st.select_slider(
            "Momentum horizon (days)",
            options=[21, 42, 63, 126, 252],
            value=int(focus_settings.get("momentum_horizon_days", 63)),
            help="The look-back window used for momentum features and focus selection.",
        )
        max_trade_size_pct = st.slider(
            "Max trade size (% of equity)",
            min_value=0.01,
            max_value=0.20,
            value=float(focus_settings.get("max_trade_size_pct", 0.05)),
            step=0.005,
            help="Maximum position size for a single trade.",
        )
        per_ticker_risk_budget_pct = st.slider(
            "Per-ticker risk budget (% of equity)",
            min_value=0.01,
            max_value=0.25,
            value=float(focus_settings.get("per_ticker_risk_budget_pct", 0.10)),
            step=0.01,
            help="Maximum risk budget allowed for each ticker in the watched set.",
        )
        use_watchlist_only = st.checkbox(
            "Focus on saved watchlist tickers only",
            value=bool(focus_settings.get("use_watchlist_only", True)),
            help="When enabled, training and signal selection should prioritize the saved watchlist.",
        )

        if st.button("💾 Save focus parameters", use_container_width=True):
            live_db.save_focus_settings({
                "insider_flow_weight": insider_flow_weight,
                "congress_flow_weight": congress_flow_weight,
                "momentum_horizon_days": momentum_horizon_days,
                "max_trade_size_pct": max_trade_size_pct,
                "per_ticker_risk_budget_pct": per_ticker_risk_budget_pct,
                "use_watchlist_only": use_watchlist_only,
            })
            st.success("Focus parameter settings saved.")

        st.markdown("---")

        st.markdown("### 🧠 PPO Hyperparameters")
        train_steps = int(st.number_input(
            "Training Timesteps",
            min_value=50_000,
            max_value=50_000_000,
            value=1_000_000,
            step=50_000,
            help="Set the total PPO learning timesteps. Larger budgets are encouraged for long, data-rich training runs.",
        ))
        lr = st.select_slider(
            "Learning Rate (α)",
            options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3],
            value=3e-4,
            format_func=lambda x: f"{x:.0e}",
        )
        ent_coef = st.slider(
            "Entropy Coefficient", 0.01, 0.10, 0.02, step=0.01,
            help="Higher → more exploration during training"
        )

        st.markdown("### 📅 Backtest Window")
        n_days = st.slider(
            "Historical window (days)",
            400,
            5200,
            2520,
            step=50,
            help="Use up to roughly 20 years of trading history when available.",
        )

        st.markdown("---")
        st.markdown("### 🗄️ Database")
        stats = live_db.get_stats()
        st.caption(
            f"Congress: **{stats['congress_trades']}** | "
            f"Insider: **{stats['insider_trades']}** | "
            f"Sentiment: **{stats['news_sentiment']}**"
        )

        st.markdown("---")
        st.markdown("### 📐 Math Reference")
        st.latex(r"\text{Sim}(v_1,v_2)=\frac{v_1\cdot v_2}{\|v_1\|\|v_2\|}")
        st.latex(r"\|v_1\wedge v_2\|_F=\|v_1 v_2^T-v_2 v_1^T\|_F")
        st.latex(r"R_t=\text{PnL}_t-\alpha C_t+\beta A_t-\gamma D_t")

        st.markdown("---")
        st.markdown("### �️ Session History Retention")
        retention_days = st.number_input(
            "Keep strategy history for (days)",
            min_value=30,
            max_value=3650,
            value=365,
            step=30,
            help="Prune persisted training recommendations older than this retention window.",
        )
        if st.button("🧹 Prune stale session history", use_container_width=True):
            removed = live_db.prune_strategy_sessions(keep_days=int(retention_days))
            st.success(f"Removed {removed} stale strategy session records older than {retention_days} days.")

        if st.button("🧨 Clear all strategy history", use_container_width=True):
            removed = live_db.prune_strategy_sessions(keep_days=0)
            st.warning(f"Cleared {removed} strategy session records from the local history.")

        st.markdown("---")
        st.markdown("### �🔍 Audit Status")
        audit_badges = [
            ("[A1] Temporal guard", "disclosure_date ≤ sim_step enforced"),
            ("[A2] Zero-vector guard", "ε=1e-8, no NaN into policy"),
            ("[A3] Warm-up offset", f"env starts at step {INDICATOR_WARMUP}"),
            ("[A4] α rebalanced", f"ALPHA_CHURN = {ALPHA_CHURN}"),
            ("[A5] Fee order-of-ops", "fee deducted before price return"),
            ("[A6] Session state", "training persisted across widget runs"),
        ]
        for label, note in audit_badges:
            st.markdown(
                f'<div class="audit-badge">✅ <b>{label}</b><br>'
                f'<span style="font-size:.74rem;color:#586069">{note}</span></div>',
                unsafe_allow_html=True,
            )

    # ── Title ─────────────────────────────────────────────────────────────
    st.markdown("# 🛡️ InsiderRL v2 — Alternative Market Intelligence Agent")
    st.markdown(
        "Unifies congressional disclosures, SEC Form 4 insider flows, "
        "gamma exposure, and financial NLP sentiment into a trained PPO "
        "trading policy via a custom Gymnasium environment. "
        "**v2 includes six hardened audit fixes** (see sidebar)."
    )
    st.markdown("---")

    # ── Data ──────────────────────────────────────────────────────────────
    _sent = SentimentEngine()
    scraper_manager = ScraperManager(db=live_db)

    if st.button("🔁 Refresh live alternative data", use_container_width=True):
        with st.spinner("Refreshing live signals and filings..."):
            scraper_manager.fetch_all_data([ticker], days_back=180)
            scraper_manager.fetch_news_and_context([ticker])
        st.success("Live alternative data refreshed.")

    if st.button("🗂️ Preseed live disclosures since STOCK Act", use_container_width=True):
        with st.spinner("Seeding live disclosure data into local DBs..."):
            counts_rl = seed_database(Path("./data/insider_rl.db"), db_type="rl", live_only=True, verbose=False)
            counts_flippy = seed_database(Path("./data/flippy_store.db"), db_type="flippy", live_only=True, verbose=False)
        st.success("Live historical preseed complete.")
        st.write("Insider RL DB counts:", counts_rl)
        st.write("Flippy Store DB counts:", counts_flippy)

    # ── Live-data health summary ───────────────────────────────────────────
    congress_rows = live_db.get_congress_trades(ticker, days_back=365)
    insider_rows = live_db.get_insider_trades(ticker, days_back=365)
    sentiment_rows = live_db.get_news_sentiment(ticker, days_back=365)

    latest_congress = congress_rows[0].get("transaction_date", "n/a") if congress_rows else "n/a"
    latest_insider = insider_rows[0].get("transaction_date", "n/a") if insider_rows else "n/a"
    latest_sentiment = sentiment_rows[0].get("trade_date", "n/a") if sentiment_rows else "n/a"

    health_cols = st.columns(4)
    health_cols[0].metric("Congress records", len(congress_rows), f"latest {latest_congress}")
    health_cols[1].metric("Insider records", len(insider_rows), f"latest {latest_insider}")
    health_cols[2].metric("Sentiment records", len(sentiment_rows), f"latest {latest_sentiment}")
    health_cols[3].metric("Live mode", "enabled", "real disclosures")

    env, mdf, alts = build_trading_environment(
        ticker=ticker,
        n_days=n_days,
        db=live_db,
        sentiment_processor=_sent,
        start_date=law_passage_date,
    )

    if env is None or mdf.empty:
        st.error(
            "Unable to build a real-data trading environment for this ticker. "
            "Please refresh live data or preseed live disclosures before training."
        )
        return

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 1 — Live alternative data streams
    # ═════════════════════════════════════════════════════════════════════
    st.markdown("## 📡 Live Alternative Data Streams")
    st.markdown(
        '<div class="info-pill">'
        '[A1] Disclosure dates are shifted by regulatory latency '
        '(Congress: 5–45 days per STOCK Act; Form 4: 2–4 business days). '
        'Post-market-close filings (>16:00 EST) are held back to the next '
        'trading day — eliminating lookahead bias during historical backtesting.'
        '</div>',
        unsafe_allow_html=True,
    )

    col_ct, col_it, col_ns = st.columns(3)

    with col_ct:
        st.markdown("#### 🏛️ Congressional Trades")
        congress_rows = live_db.get_congress_trades(ticker, days_back=90)
        if congress_rows:
            df_ct = pd.DataFrame([{
                "Date": row.get("transaction_date", ""),
                "Politician": row.get("politician", ""),
                "Type": row.get("transaction_type", ""),
                "Amount": f"{row.get('amount_midpoint', 0):,.0f}",
            } for row in congress_rows[:7]])
            st.dataframe(df_ct, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No congressional filings available yet for this ticker. "
                "Refresh live data or run the live disclosure preseed."
            )

    with col_it:
        st.markdown("#### 📋 Form 4 Insider Filings")
        insider_rows = live_db.get_insider_trades(ticker, days_back=90)
        if insider_rows:
            df_it = pd.DataFrame([{
                "Date": row.get("transaction_date", ""),
                "Insider": row.get("insider_name", ""),
                "Title": row.get("title", ""),
                "Value": f"${row.get('total_value', 0):,.0f}",
            } for row in insider_rows[:7]])
            st.dataframe(df_it, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No insider filings available yet for this ticker. "
                "Refresh live data or run the live disclosure preseed."
            )

    with col_ns:
        st.markdown("#### 📰 NLP Sentiment Feed")
        sentiment_rows = live_db.get_news_sentiment(ticker, days_back=30)
        if sentiment_rows:
            df_ns = pd.DataFrame([{
                "Date": row.get("trade_date", ""),
                "Score": f"{row.get('score', 0.0):+.3f}",
                "Grade": row.get("grade", ""),
                "Headline": (row.get("headline", "")[:60] + "…") if len(row.get("headline", "")) > 60 else row.get("headline", ""),
            } for row in sentiment_rows[:7]])
            st.dataframe(df_ns, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No persisted NLP sentiment available yet for this ticker. "
                "Refresh live data or run the live disclosure preseed."
            )

    st.markdown("---")

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 2 — Vector geometry
    # ═════════════════════════════════════════════════════════════════════
    st.markdown("## 📐 Real-Time Vector Geometry Analysis")
    st.markdown(
        '<div class="info-pill">'
        '[A2] Each alternative data event is embedded as a 4-dimensional vector: '
        '[volume_percentile, role_seniority, gex_regime, sentiment]. '
        'Cosine similarity measures directional alignment with historically profitable '
        'accumulation templates. The Frobenius norm of the exterior wedge product '
        'quantifies structural deviation from known patterns. '
        'Zero-vector guard (ε=1e-8) prevents NaN propagation into the policy gradient.'
        '</div>',
        unsafe_allow_html=True,
    )

    latest = alts[-1]
    i_sim  = SignalGeometry.cosine_similarity(latest["v_insider"],  INSIDER_TEMPLATE)
    c_sim  = SignalGeometry.cosine_similarity(latest["v_congress"], CONGRESS_TEMPLATE)
    i_wdg  = SignalGeometry.wedge_magnitude(latest["v_insider"],    INSIDER_TEMPLATE)
    c_wdg  = SignalGeometry.wedge_magnitude(latest["v_congress"],   CONGRESS_TEMPLATE)

    gc1, gc2, gc3, gc4 = st.columns(4)
    gc1.metric("Insider Cosine Sim",   f"{i_sim:.4f}",
               help="Dot product alignment with suspicious insider buy template. Range [−1, +1].")
    gc2.metric("Insider Wedge ‖∧‖_F", f"{i_wdg:.4f}",
               help="Near zero = collinear (high confidence). Large = structural divergence.")
    gc3.metric("Congress Cosine Sim",  f"{c_sim:.4f}",
               help="Directional alignment with congressional accumulation template.")
    gc4.metric("Congress Wedge ‖∧‖_F",f"{c_wdg:.4f}",
               help="Structural deviation of congressional flow from historical norms.")

    vcol1, vcol2 = st.columns(2)

    with vcol1:
        cats = SignalGeometry.FEATURE_NAMES + [SignalGeometry.FEATURE_NAMES[0]]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=list(latest["v_insider"]) + [latest["v_insider"][0]],
            theta=cats, fill="toself", name="Current Insider Flow",
            line_color="#00FFCC", fillcolor="rgba(0,255,204,0.12)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=list(INSIDER_TEMPLATE) + [INSIDER_TEMPLATE[0]],
            theta=cats, fill="toself", name="Suspicious Template",
            line_color="#FF3366", fillcolor="rgba(255,51,102,0.12)",
        ))
        fig_radar.update_layout(
            template="plotly_dark",
            title="Feature Vector vs Suspicious Template",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            legend=dict(x=0, y=-0.20, orientation="h"),
            margin=dict(t=55, b=70),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with vcol2:
        st.markdown(f"**Interpretation:** {SignalGeometry.interpret_similarity(i_sim)}")
        st.markdown("---")
        sim60  = [SignalGeometry.cosine_similarity(s["v_insider"],  INSIDER_TEMPLATE)  for s in alts[-80:]]
        cong60 = [SignalGeometry.cosine_similarity(s["v_congress"], CONGRESS_TEMPLATE) for s in alts[-80:]]
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(y=sim60,  name="Insider Sim",  line=dict(color="#00FFCC", width=2)))
        fig_ts.add_trace(go.Scatter(y=cong60, name="Congress Sim", line=dict(color="#FF9900", width=2)))
        fig_ts.add_hline(y=ALIGN_THRESHOLD, line_dash="dash", line_color="#FF3366",
                          annotation_text=f"Bonus Threshold ({ALIGN_THRESHOLD})")
        fig_ts.update_layout(
            template="plotly_dark", title="80-Day Similarity History",
            xaxis_title="Bars ago (left=older)", yaxis_title="Cosine Similarity",
            yaxis_range=[-0.05, 1.05], height=310, margin=dict(t=50, b=30),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 3 — [A1] Temporal disclosure audit panel
    # ═════════════════════════════════════════════════════════════════════
    st.markdown("## 🕒 Temporal Disclosure Integrity")
    st.markdown(
        '<div class="info-pill">'
        '[A1] get_disclosed_signals_on_date enforces: '
        'disclosure_date ≤ sim_step_date AND post-close filings are '
        'held back to the next day.  The panel below shows what the agent '
        'would ACTUALLY see at today\'s open vs. after market close.'
        '</div>',
        unsafe_allow_html=True,
    )

    t_col1, t_col2 = st.columns(2)
    today = datetime.date.today()
    sim_date = today.strftime("%Y-%m-%d")

    with t_col1:
        st.markdown("**At Open (post-close filings excluded)**")
        signals_open = live_db.get_disclosed_signals_on_date(
            ticker, sim_date, sim_date_is_after_close=False
        )
        st.caption(
            f"Congress trades visible: **{len(signals_open['congress'])}** | "
            f"Insider buys: **{len(signals_open['insider'])}** | "
            f"Sentiment entries: **{len(signals_open['sentiment'])}**"
        )
        if signals_open["congress"]:
            df_c = pd.DataFrame(signals_open["congress"])[
                ["politician","trade_type","disclosure_date","disclosure_time_utc"]
            ].head(5)
            st.dataframe(df_c, use_container_width=True, hide_index=True)
        else:
            st.info("No disclosed congress trades yet (seed data first)")

    with t_col2:
        st.markdown("**After Close (all same-day filings included)**")
        signals_close = live_db.get_disclosed_signals_on_date(
            ticker, sim_date, sim_date_is_after_close=True
        )
        st.caption(
            f"Congress trades visible: **{len(signals_close['congress'])}** | "
            f"Insider buys: **{len(signals_close['insider'])}** | "
            f"Sentiment entries: **{len(signals_close['sentiment'])}**"
        )
        if signals_close["congress"]:
            df_c2 = pd.DataFrame(signals_close["congress"])[
                ["politician","trade_type","disclosure_date","disclosure_time_utc"]
            ].head(5)
            st.dataframe(df_c2, use_container_width=True, hide_index=True)
        else:
            st.info("No disclosed congress trades yet (seed data first)")

    st.markdown("---")

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 4 — Training & backtest   [A6] session_state protected
    # ═════════════════════════════════════════════════════════════════════
    st.markdown("## 🧠 PPO Policy Training & Strategy Backtest")
    st.markdown(
        '<div class="info-pill">'
        '[A4] Shaped reward: PnL (base) − α·ChurnPenalty (α=0.002, was 0.02) '
        '+ β·AlignmentBonus (β=0.015) − γ·DrawdownPenalty (γ=0.05, threshold=5%). '
        '[A5] Transaction fee deducted BEFORE price return is applied. '
        '[A6] Results stored in session_state — widget changes do NOT retrain.'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("🚀 Train Policy Network", type="primary", use_container_width=True):
        env  = InsiderTradingEnv(mdf, alts, initial_balance=100_000.0)
        pbar = st.progress(0, text="Initializing PPO agent…")

        class _CB(BaseCallback):
            def __init__(self, total: int, bar):
                super().__init__()
                self._total = total
                self._bar   = bar
            def _on_step(self) -> bool:
                pct = min(int(self.num_timesteps / self._total * 100), 99)
                self._bar.progress(pct, text=f"Training… {self.num_timesteps:,}/{self._total:,} steps")
                return True

        try:
            model = PPO(
                "MlpPolicy", env,
                learning_rate=lr,
                ent_coef=ent_coef,
                n_steps=min(512, train_steps),
                batch_size=64,
                verbose=0,
            )
            model.learn(total_timesteps=train_steps, callback=_CB(train_steps, pbar))
            pbar.progress(100, text="✅ Training complete!")

            # [A6] Store in session_state so re-renders don't retrain
            st.session_state["trained_model"] = model
            st.session_state["train_ticker"]  = ticker

        except Exception as e:
            st.error(f"Training failed: {e}")
            logger.exception("PPO training error")
            st.stop()

        # ── Backtest ──────────────────────────────────────────────────
        status = st.empty()
        status.info("Running deterministic backtest…")
        obs, _ = env.reset()
        states_for_pca: list[np.ndarray] = []
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            states_for_pca.append(obs.copy())
            obs, _, terminated, truncated, _ = env.step(int(action))

        status.empty()

        # [A6] Cache backtest results
        st.session_state["backtest_hist"] = env.history
        st.session_state["pca_states"]    = states_for_pca
        st.session_state["metrics"]       = compute_metrics(env.history, 100_000.0)

        suggestion, suggestion_notes = _recommend_strategy_change(
            st.session_state["metrics"],
            train_steps,
            n_days,
            lr,
            ent_coef,
        )
        if live_db.insert_strategy_session(
            ticker=ticker,
            strategy_name=f"PPO_{ticker}",
            training_steps=train_steps,
            learning_rate=lr,
            entropy_coef=ent_coef,
            backtest_days=n_days,
            performance_metrics=st.session_state["metrics"],
            suggested_change=suggestion,
            notes=suggestion_notes,
        ):
            st.success("✅ Session recommendation persisted to the local strategy history.")
        else:
            st.warning("⚠️ Failed to persist session recommendation. Database may be locked.")

    # ── Display results from session_state (persists across re-runs) [A6] ──
    if st.session_state.get("backtest_hist") is not None:
        hist_df = pd.DataFrame(st.session_state["backtest_hist"])
        m       = st.session_state["metrics"]
        states_for_pca = st.session_state["pca_states"]

        if st.session_state.get("train_ticker") and st.session_state["train_ticker"] != ticker:
            st.warning(
                f"⚠️ Results below are from ticker **{st.session_state['train_ticker']}** — "
                "re-train to update for the current selection."
            )

        # ── Performance metrics ───────────────────────────────────────
        st.markdown("### 📊 Strategy Performance")
        mc = st.columns(5)
        mc[0].metric("Total Return",   f"{m['total_return']:+.1f}%")
        mc[1].metric("Sharpe Ratio",   f"{m['sharpe']:.3f}",
                     help="Annualized. >1.5 = excellent; >1.0 = good.")
        mc[2].metric("Sortino Ratio",  f"{m['sortino']:.3f}",
                     help="Like Sharpe but uses only downside volatility.")
        mc[3].metric("Max Drawdown",   f"{m['max_drawdown']:.1f}%",
                     help="Largest peak-to-trough portfolio decline.")
        mc[4].metric("Win Rate",       f"{m['win_rate']:.0f}%",
                     help="% of timesteps with positive NAV change.")

        # ── Reward coefficient balance display ────────────────────────
        st.markdown("### ⚖️ Reward Coefficient Audit [A4]")
        avg_pnl   = float(hist_df["pnl"].abs().mean()) * 10.0
        avg_churn = float(hist_df["churn_penalty"].mean())
        avg_align = float(hist_df["align_bonus"].mean())
        avg_dd    = float(hist_df["dd_penalty"].mean())

        bal_cols = st.columns(4)
        bal_cols[0].metric("Avg |PnL|×10", f"{avg_pnl:.5f}", help="Primary objective")
        bal_cols[1].metric("Avg Churn Pen", f"{avg_churn:.5f}",
                           help=f"α={ALPHA_CHURN} — should be ≪ PnL magnitude")
        bal_cols[2].metric("Avg Align Bonus", f"{avg_align:.5f}",
                           help=f"β={BETA_ALIGNMENT}")
        bal_cols[3].metric("Avg DD Penalty", f"{avg_dd:.5f}",
                           help=f"γ={GAMMA_DRAWDOWN}")

        churn_ratio = avg_churn / max(avg_pnl, 1e-9)
        if churn_ratio > 0.5:
            st.warning(
                f"⚠️ Churn penalty ({avg_churn:.5f}) is {churn_ratio:.1%} of "
                f"PnL ({avg_pnl:.5f}). Consider reducing ALPHA_CHURN further."
            )
        else:
            st.success(
                f"✅ Reward terms well-balanced: churn is {churn_ratio:.1%} of PnL magnitude."
            )

        # ── Session suggestion history ──────────────────────────────
        st.markdown("### 🧾 Session Suggestions & History")
        previous_sessions = live_db.get_strategy_sessions(ticker, limit=20)
        history_df = _session_history_dataframe(previous_sessions)

        if previous_sessions:
            with st.expander("📚 Browse persisted session history", expanded=False):
                st.write(
                    "Review and export persisted strategy sessions for this ticker. "
                    "Use the comparison tab to benchmark the current training run against historical results."
                )
                st.dataframe(history_df, use_container_width=True)
                csv_data = history_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Export session history CSV",
                    csv_data,
                    file_name=f"{ticker}_session_history.csv",
                    mime="text/csv",
                )

            best_prior = _get_best_prior_session(previous_sessions)
            if best_prior is not None:
                best_metrics = _normalize_metrics(best_prior.get("performance_metrics", {}))
                st.markdown(
                    f"**Best prior session**: {best_prior['strategy_name']} on {best_prior['created_at']} "
                    f"with Sharpe={best_metrics['sharpe']:.3f} and Return={best_metrics['total_return']:+.1f}%"
                )

            compare_tab, history_tab = st.tabs(["Compare Sessions", "History Dashboard"])

            with compare_tab:
                compare_mode = st.radio(
                    "Compare current training against:",
                    options=["Latest prior session", "Best prior session"],
                    index=0,
                    horizontal=True,
                )
                if compare_mode == "Best prior session" and best_prior is not None:
                    previous = best_prior
                else:
                    selected_idx = st.selectbox(
                        "Select a prior session to compare",
                        options=list(range(len(previous_sessions))),
                        format_func=lambda idx: (
                            f"{previous_sessions[idx]['created_at']} | "
                            f"{previous_sessions[idx]['strategy_name']} | "
                            f"{_normalize_metrics(previous_sessions[idx].get('performance_metrics', {}))['total_return']:+.1f}%"
                        ),
                        key="compare_prior_session",
                    )
                    previous = previous_sessions[selected_idx]

                previous_metrics = _normalize_metrics(previous.get("performance_metrics", {}))
                st.markdown("#### 📊 Current vs Selected Session")
                if st.session_state.get("metrics") is not None:
                    _plot_session_comparison(_normalize_metrics(st.session_state["metrics"]), previous_metrics)
                    _plot_training_metrics_map(_normalize_metrics(st.session_state["metrics"]), previous_sessions)
                    _plot_session_timeline(previous_sessions, _normalize_metrics(st.session_state["metrics"]))
                else:
                    st.info(
                        "Train the policy first to compare the current session with prior sessions. "
                        "Previous session history is available in the History Dashboard tab."
                    )

            with history_tab:
                st.markdown("#### 🔎 Session History Dashboard")
                if history_df.empty:
                    st.info("No persisted session history is available for this ticker yet.")
                else:
                    st.dataframe(history_df, use_container_width=True)

        else:
            st.info("No prior session suggestions have been persisted yet.")

        st.markdown("### 🤖 Local LLM-Guided Auto Tuning")
        with st.expander("Run a safe local tuning loop using the advisor", expanded=False):
            auto_iterations = st.number_input(
                "Autonomous tuning iterations",
                min_value=1,
                max_value=4,
                value=2,
                step=1,
                help="Run multiple training iterations with local advisor suggestions between each session.",
            )
            stop_plateau = st.checkbox(
                "Stop early on Sharpe plateau",
                value=True,
                help="Terminate the tuning loop when improvement stalls across iterations.",
            )
            if st.button("🚦 Run Autonomous Tuning", use_container_width=True):
                advisor = LocalLLMAdvisor()
                trainer = AutonomousTrainingManager(
                    db=live_db,
                    advisor=advisor,
                    env_builder=build_trading_environment,
                )
                config = AutonomousTrainingConfig(
                    training_steps=train_steps,
                    learning_rate=lr,
                    entropy_coef=ent_coef,
                    backtest_days=n_days,
                    strategy_name=f"PPO_{ticker}_autotune",
                    min_sharpe_improvement=0.05,
                    plateau_iterations=2 if stop_plateau else 1000,
                )
                llm_context = live_db.build_llm_context()
                with st.spinner("Running the autonomous tuning loop, this may take several minutes…"):
                    report = trainer.run_autonomous_loop(
                        ticker=ticker,
                        initial_config=config,
                        sentiment_processor=_sent,
                        start_date=law_passage_date,
                        max_iterations=int(auto_iterations),
                        llm_context=llm_context,
                    )
                st.session_state["autonomous_report"] = dataclasses.asdict(report)

            if st.session_state.get("autonomous_report") is not None:
                _render_autonomous_training_report(st.session_state["autonomous_report"])

        # ── NAV curve ────────────────────────────────────────────────
        st.markdown("### 📈 NAV Curve vs. Buy & Hold")
        steps = hist_df["step"].values.astype(int)
        nav   = hist_df["portfolio_value"].values

        if len(mdf) > INDICATOR_WARMUP:
            initial_close = float(mdf["Close"].iloc[INDICATOR_WARMUP])
            bh = (mdf["Close"].values[np.clip(steps, 0, len(mdf)-1)] /
                  initial_close) * 100_000.0
        elif len(mdf) > 0:
            initial_close = float(mdf["Close"].iloc[0])
            bh = (mdf["Close"].values[np.clip(steps, 0, len(mdf)-1)] /
                  initial_close) * 100_000.0
            st.warning(
                "Not enough ticker history to apply the standard INDICATOR_WARMUP offset. "
                "Buy & Hold is approximated from the first available close."
            )
        else:
            bh = np.full_like(nav, np.nan)
            st.warning(
                "No market history is available for Buy & Hold comparison."
            )

        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(
            x=steps, y=nav, mode="lines", name="🛡️ InsiderRL Agent",
            line=dict(color="#00FFCC", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,255,204,0.05)",
        ))
        fig_nav.add_trace(go.Scatter(
            x=steps, y=bh, mode="lines", name="📦 Buy & Hold",
            line=dict(color="#FF3366", width=1.8, dash="dot"),
        ))
        buys  = hist_df[hist_df["action"] == 2]
        sells = hist_df[hist_df["action"] == 0]
        if len(buys):
            fig_nav.add_trace(go.Scatter(
                x=buys["step"].values, y=buys["portfolio_value"].values,
                mode="markers", name="BUY ▲",
                marker=dict(color="#00FF88", size=8, symbol="triangle-up"),
            ))
        if len(sells):
            fig_nav.add_trace(go.Scatter(
                x=sells["step"].values, y=sells["portfolio_value"].values,
                mode="markers", name="SELL ▼",
                marker=dict(color="#FF4444", size=8, symbol="triangle-down"),
            ))
        fig_nav.update_layout(
            template="plotly_dark",
            title=f"InsiderRL v2 vs Buy & Hold — {st.session_state.get('train_ticker', ticker)}",
            xaxis_title="Simulation Step (starts at warm-up=26)",
            yaxis_title="Portfolio NAV ($)",
            hovermode="x unified",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_nav, use_container_width=True)

        # ── Fee audit [A5] ────────────────────────────────────────────
        st.markdown("### 💸 Transaction Fee Audit [A5]")
        total_fees = float(hist_df["fee_paid"].sum())
        n_trades   = int((hist_df["action"] != 1).sum())
        st.markdown(
            f"Total fees paid: **${total_fees:,.2f}** across **{n_trades}** "
            f"position changes. Fee deducted from V_{{t-1}} BEFORE price "
            f"return applied — no leverage-free borrowing."
        )

        # ── Reward decomposition ──────────────────────────────────────
        st.markdown("### ⚖️ Reward Signal Decomposition")
        fig_rwd = go.Figure()
        fig_rwd.add_trace(go.Scatter(
            x=hist_df["step"], y=hist_df["pnl"] * 10,
            name="PnL ×10", line=dict(color="#00FFCC", width=1.5),
        ))
        fig_rwd.add_trace(go.Bar(
            x=hist_df["step"], y=-hist_df["churn_penalty"],
            name="−Churn Penalty", marker_color="#FF6B6B", opacity=0.75,
        ))
        fig_rwd.add_trace(go.Bar(
            x=hist_df["step"], y=hist_df["align_bonus"],
            name="+Alignment Bonus", marker_color="#00FF88", opacity=0.75,
        ))
        fig_rwd.add_trace(go.Bar(
            x=hist_df["step"], y=-hist_df["dd_penalty"],
            name="−Drawdown Penalty", marker_color="#FF9900", opacity=0.75,
        ))
        fig_rwd.update_layout(
            template="plotly_dark",
            title="R_t = PnL_t − α·Churn + β·Alignment − γ·Drawdown",
            barmode="relative",
            xaxis_title="Step", yaxis_title="Reward Component",
            legend=dict(orientation="h", y=-0.22),
            margin=dict(t=60, b=90),
        )
        st.plotly_chart(fig_rwd, use_container_width=True)

        # ── PCA state space projection ────────────────────────────────
        st.markdown("### 🌌 PCA State Space Projection")
        st.markdown(
            '<div class="info-pill">'
            '[A3] PCA reduces the 12-dimensional observation space (valid '
            'from warm-up step 26 onward) to 2 principal components. '
            'Coloring by sentiment reveals how the agent clusters its '
            'decisions: BUY actions (green) concentrate where high insider '
            'similarity converges with positive sentiment.'
            '</div>',
            unsafe_allow_html=True,
        )

        if states_for_pca and len(states_for_pca) >= 10:
            S   = np.array(states_for_pca, dtype=np.float32)
            pca = PCA(n_components=2, random_state=42)
            PC  = pca.fit_transform(S)

            pca_df = pd.DataFrame(PC, columns=["PC1", "PC2"])
            pca_df["Sentiment"]    = S[:, 5]
            pca_df["Insider Sim"]  = S[:, 3]
            n_actions = min(len(states_for_pca), len(hist_df))
            pca_df["Action"]       = [h["action"] for h in st.session_state["backtest_hist"][:n_actions]]
            pca_df["Action Label"] = pca_df["Action"].map({0:"SELL", 1:"HOLD", 2:"BUY"})

            ev0 = pca.explained_variance_ratio_[0]
            ev1 = pca.explained_variance_ratio_[1]
            lx  = f"PC1 ({ev0:.1%} var)"
            ly  = f"PC2 ({ev1:.1%} var)"

            pca_c1, pca_c2 = st.columns(2)

            with pca_c1:
                fig_s = px.scatter(
                    pca_df, x="PC1", y="PC2", color="Sentiment",
                    title="Clusters — Colored by NLP Sentiment",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    template="plotly_dark",
                    labels={"PC1": lx, "PC2": ly},
                    opacity=0.72,
                )
                fig_s.update_traces(marker_size=4)
                fig_s.update_layout(margin=dict(t=55, b=30))
                st.plotly_chart(fig_s, use_container_width=True)

            with pca_c2:
                fig_a = px.scatter(
                    pca_df, x="PC1", y="PC2", color="Action Label",
                    title="Clusters — Colored by Agent Action",
                    color_discrete_map={"SELL":"#FF4444","HOLD":"#888888","BUY":"#00FF88"},
                    template="plotly_dark",
                    labels={"PC1": lx, "PC2": ly},
                    category_orders={"Action Label": ["SELL","HOLD","BUY"]},
                    opacity=0.72,
                )
                fig_a.update_traces(marker_size=4)
                fig_a.update_layout(margin=dict(t=55, b=30))
                st.plotly_chart(fig_a, use_container_width=True)

            fig_ev = go.Figure(go.Bar(
                x=["PC1", "PC2"],
                y=[ev0 * 100, ev1 * 100],
                marker_color=["#00FFCC", "#FF9900"],
                text=[f"{ev0*100:.1f}%", f"{ev1*100:.1f}%"],
                textposition="outside",
            ))
            fig_ev.update_layout(
                template="plotly_dark", title="PCA Explained Variance",
                yaxis_title="Variance Explained (%)",
                height=240, margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig_ev, use_container_width=True)

        # ── Similarity distribution ───────────────────────────────────
        st.markdown("### 🔭 Signal Similarity Distribution")
        sims_i = [s["insider_sim"]  for s in alts]
        sims_c = [s["congress_sim"] for s in alts]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=sims_i, name="Insider Sim",
                                         nbinsx=40, marker_color="#00FFCC", opacity=0.75))
        fig_hist.add_trace(go.Histogram(x=sims_c, name="Congress Sim",
                                         nbinsx=40, marker_color="#FF9900", opacity=0.75))
        fig_hist.add_vline(
            x=ALIGN_THRESHOLD, line_dash="dash", line_color="#FF3366",
            annotation_text=f"Bonus Threshold = {ALIGN_THRESHOLD}",
        )
        fig_hist.update_layout(
            template="plotly_dark",
            title="Distribution of Cosine Similarity Scores Across Full Simulation",
            xaxis_title="Cosine Similarity", yaxis_title="Frequency",
            barmode="overlay", margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        # Pre-training orientation chart
        st.info(
            "⬆️  Configure the agent in the sidebar then press **Train Policy Network**. "
            "Results are cached in session state — adjusting sliders will NOT retrain.",
            icon="ℹ️",
        )
        fig_pre = go.Figure()
        fig_pre.add_trace(go.Candlestick(
            x=mdf["Date"], open=mdf["Open"], high=mdf["High"],
            low=mdf["Low"], close=mdf["Close"], name=ticker,
            increasing_line_color="#00FFCC", decreasing_line_color="#FF3366",
        ))
        clo_min = float(mdf["Close"].min())
        clo_rng = float(mdf["Close"].max()) - clo_min
        scaled_sim = [s["insider_sim"] * clo_rng + clo_min for s in alts]
        fig_pre.add_trace(go.Scatter(
            y=scaled_sim, x=mdf["Date"],
            name="Insider Sim (price-scaled)",
            line=dict(color="#FF9900", width=1.5, dash="dot"),
        ))
        fig_pre.update_layout(
            template="plotly_dark",
            title=f"{ticker} — Price & Insider Similarity Preview",
            xaxis_rangeslider_visible=False, height=450,
        )
        st.plotly_chart(fig_pre, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<p style="color:#444;font-size:.76rem;text-align:center;">'
        'InsiderRL v2.0 — Hardened Edition · Research & educational purposes only · '
        'Not financial advice · Uses real historical disclosures and live market data when available.</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOG
# ─────────────────────────────────────────────────────────────────────────────
#
# [A1] TEMPORAL INTEGRITY (MarketIntelligenceDB)
#   Problem : original queries used date('now', ...) without checking disclosure
#             time of day; post-market-close filings could have been acted on
#             in the same session bar they were published.
#   Fix     : Added disclosure_time_utc column. get_disclosed_signals_on_date
#             enforces (disclosure_date < sim_date) OR
#             (disclosure_date = sim_date AND disclosure_time_utc < '20:00:00')
#             when sim_date_is_after_close=False.
#
# [A2] VECTOR ZERO-GUARD (SignalGeometry)
#   Problem : original guard was n1 < 1e-9; on zero-activity days with
#             neutral GEX and 0.5 sentiment the 4D vector is non-zero, but
#             any truly zero vector would produce NaN which corrupts PPO
#             gradient through all layers.
#   Fix     : Raised guard to 1e-8 (recommended in audit prompt) in both
#             cosine_similarity and wedge_magnitude. Added np.clip to catch
#             floating-point drift outside [-1, 1]. Added np.nan_to_num in
#             _obs() as a final backstop.
#
# [A3] GYMNASIUM WARM-UP & COMPLIANCE (InsiderTradingEnv)
#   Problem : environment started at step 1, passing partially-NaN RSI/MACD
#             observations to SB3's MLP during the first 26 steps.
#   Fix     : INDICATOR_WARMUP = 26; reset() and __init__ start at this offset.
#             Confirmed 5-tuple step() return with correct terminated/truncated.
#             reset() returns (obs, {}) — Gymnasium-compliant.
#
# [A4] REWARD REBALANCING (InsiderTradingEnv.step)
#   Problem : ALPHA_CHURN = 0.02 ≈ 67% of a typical daily |PnL| × 10 signal
#             (~0.03). PPO immediately learns to never trade to avoid this
#             penalty, producing a degenerate cash-hold policy.
#   Fix     : ALPHA_CHURN reduced from 0.02 to 0.002 (10× reduction).
#             Reward is clipped to [-10, 10] to bound variance for PPO.
#             Sidebar audit panel shows live churn/PnL ratio post-backtest.
#
# [A5] FEE ORDER-OF-OPERATIONS (InsiderTradingEnv.step)
#   Problem : original BUY code: shares = balance*(1-fee)/price — fee was
#             implicitly subtracted from the entry capital (correct), but
#             SELL: balance = shares*price*(1-fee) applied fee to the gross
#             proceeds after the price had already been locked in — acceptable
#             for sells but the sequencing was not documented.
#   Fix     : Explicit two-step: (1) fee_paid = position_value * fee_pct,
#             (2) net = position_value - fee_paid, (3) apply price return.
#             fee_paid is logged per step for the audit panel.
#
# [A6] STREAMLIT THREAD SAFETY (run_app)
#   Problem : any sidebar widget change triggered a full script re-run,
#             restarting the PPO training loop and creating duplicate DB
#             connections.
#   Fix     : trained_model, backtest_hist, pca_states, and metrics are
#             stored in st.session_state and only updated when the user
#             explicitly clicks "Train Policy Network".  WAL journal mode
#             (already present) prevents SQLite lock contention.
#
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_app()