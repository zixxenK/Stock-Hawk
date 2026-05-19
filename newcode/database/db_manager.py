from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any
import json
import logging
import os
import uuid

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    LargeBinary,
    UniqueConstraint,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)
Base = declarative_base()

REQUIRED_DB_METHODS = (
    "get_congress_trades",
    "get_insider_trades",
    "get_news_sentiment",
    "get_stats",
    "get_disclosed_signals_on_date",
    "upsert_signal_metadata",
    "get_ticker_history",
    "get_strategy_sessions",
    "insert_strategy_session",
    "prune_strategy_sessions",
    "save_recommended_candidates",
    "get_recommended_candidates",
)


def validate_db_adapter(adapter: Any, required_methods: tuple[str, ...] | None = None) -> None:
    if required_methods is None:
        required_methods = REQUIRED_DB_METHODS
    missing = [name for name in required_methods if not callable(getattr(adapter, name, None))]
    if missing:
        raise TypeError(
            f"Database adapter missing required methods: {', '.join(missing)}"
        )


def create_db_adapter(
    adapter_class: type["BaseDatabaseAdapter"] | None = None,
    *args: Any,
    validate_methods: tuple[str, ...] | None = None,
    **kwargs: Any,
) -> "BaseDatabaseAdapter":
    """Create and validate a database adapter instance.

    Defaults to `DBManager` when no adapter_class is provided.
    """
    if adapter_class is None:
        adapter_class = DBManager  # type: ignore[assignment]

    adapter = adapter_class(*args, **kwargs)  # type: ignore[call-arg]
    validate_db_adapter(adapter, required_methods=validate_methods)
    return adapter


class BaseDatabaseAdapter(ABC):
    """Abstract contract for database adapters used by the Flippy platform."""

    @abstractmethod
    def get_congress_trades(
        self,
        ticker: str,
        days_back: int = 30,
        purchases_only: bool = True,
        start_date: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_insider_trades(
        self,
        ticker: str,
        days_back: int = 30,
        purchases_only: bool = True,
        start_date: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_news_sentiment(
        self,
        ticker: str,
        days_back: int = 30,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, int]:
        ...

    @abstractmethod
    def get_disclosed_signals_on_date(
        self,
        ticker: str,
        sim_date: str,
        sim_date_is_after_close: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        ...

    @abstractmethod
    def upsert_signal_metadata(
        self,
        ticker: str,
        action: str,
        alpha_score: float,
        sentiment_score: float | None,
        vector: list[float] | str | None = None,
    ) -> bool:
        ...

    @abstractmethod
    def save_recommended_candidates(
        self,
        candidates: list[dict[str, Any]],
        source: str = "recommend",
    ) -> bool:
        ...

    @abstractmethod
    def get_recommended_candidates(
        self,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_strategy_sessions(self, ticker: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        ...

    def check_health(self) -> dict[str, Any]:
        raise NotImplementedError("Health checks are not supported by this database adapter.")

    def compact_database(self) -> bool:
        raise NotImplementedError("Database compaction is not supported by this database adapter.")

    @abstractmethod
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
        ...

    @abstractmethod
    def prune_strategy_sessions(self, keep_days: int = 365) -> int:
        ...

    @abstractmethod
    def get_ticker_history(self, ticker: str, days_back: int = 365) -> list[dict[str, Any]]:
        ...

# --- Schemas ---
class DBPoliticalTrade(Base):
    __tablename__ = "political_trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), index=True, nullable=False)
    politician = Column(String(100))
    chamber = Column(String(20))
    transaction_type = Column(String(20))
    amount_midpoint = Column(Float)
    transaction_date = Column(String(20), index=True)
    disclosure_date = Column(String(20), nullable=False, default="")
    disclosure_time_utc = Column(String(20), nullable=False, default="12:00:00")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "politician",
            "transaction_type",
            "amount_midpoint",
            "transaction_date",
            name="uq_political_trade",
        ),
    )

class DBInsiderTrade(Base):
    __tablename__ = "insider_trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), index=True, nullable=False)
    insider_name = Column(String(100))
    title = Column(String(50))
    transaction_type = Column(String(10))
    total_value = Column(Float)
    shares = Column(Float)
    transaction_date = Column(String(20), index=True)
    disclosure_date = Column(String(20), nullable=False, default="")
    disclosure_time_utc = Column(String(20), nullable=False, default="12:00:00")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "insider_name",
            "transaction_type",
            "transaction_date",
            "shares",
            name="uq_insider_trade",
        ),
    )

class DBSignal(Base):
    __tablename__ = "analysis_signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), index=True, nullable=False)
    action = Column(String(32), nullable=False)
    alpha_score = Column(Float, nullable=False)
    sentiment_score = Column(Float, nullable=True)
    vector = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class DBRecommendedCandidate(Base):
    __tablename__ = "recommended_candidates"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), index=True, nullable=False)
    action = Column(String(32), nullable=False)
    alpha_score = Column(Float, nullable=False)
    sentiment_score = Column(Float, nullable=True)
    vector = Column(String, nullable=True)
    source = Column(String(64), nullable=False, default="recommend")
    created_at = Column(DateTime, default=datetime.utcnow)

class DBNewsSentiment(Base):
    __tablename__ = "news_sentiment"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), index=True, nullable=False)
    score = Column(Float, nullable=False)
    magnitude = Column(Float, nullable=False)
    grade = Column(String(32), nullable=False)
    headline = Column(String(1024), nullable=True)
    source = Column(String(128), nullable=True)
    trade_date = Column(String(20), index=True, nullable=False)
    inserted_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "headline",
            "trade_date",
            name="uq_news_sentiment",
        ),
    )

class DBRawPage(Base):
    __tablename__ = "raw_pages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(64), nullable=False, index=True)
    url = Column(String(1024), nullable=False)
    status_code = Column(Integer, nullable=False)
    content_type = Column(String(256), nullable=True)
    headers = Column(String, nullable=True)
    content_hash = Column(String(64), nullable=False, index=True)
    raw_bytes = Column(LargeBinary, nullable=False)
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class DBStrategySession(Base):
    __tablename__ = "strategy_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, unique=True, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    strategy_name = Column(String(128), nullable=True)
    training_steps = Column(Integer, nullable=True)
    learning_rate = Column(Float, nullable=True)
    entropy_coef = Column(Float, nullable=True)
    backtest_days = Column(Integer, nullable=True)
    performance_metrics = Column(String, nullable=True)
    suggested_change = Column(String(512), nullable=True)
    notes = Column(String(1024), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# --- Manager ---
class DatabaseManager(BaseDatabaseAdapter):
    """
    Handles connections and specific queries to the Flippy SQLite database.
    """
    def __init__(self, db_url: str = "sqlite:///./data/flippy_store.db"):
        if db_url.startswith("sqlite:///"):
            db_path = db_url.replace("sqlite:///", "")
            folder = os.path.dirname(db_path)
            if folder and not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

        engine_kwargs = {
            "echo": False,
            "future": True,
        }
        if db_url.startswith("sqlite:///"):
            engine_kwargs["connect_args"] = {
                "check_same_thread": False,
                "timeout": 30,
            }

        self.engine = create_engine(db_url, **engine_kwargs)
        with self.engine.begin() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA busy_timeout=5000"))
        Base.metadata.create_all(self.engine)
        self._ensure_analysis_signals_schema()
        self._run_migrations()
        self.Session = sessionmaker(bind=self.engine, future=True)

    def _ensure_analysis_signals_schema(self) -> None:
        """Add missing columns to the analysis_signals table for older SQLite databases."""
        with self.engine.begin() as conn:
            result = conn.execute(text("PRAGMA table_info(analysis_signals)"))
            columns = [row[1] for row in result]
            if "created_at" not in columns:
                conn.execute(text(
                    "ALTER TABLE analysis_signals "
                    "ADD COLUMN created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP"
                ))

    def _run_migrations(self) -> None:
        """Run schema migrations using SQLite PRAGMA user_version."""
        with self.engine.begin() as conn:
            result = conn.execute(text("PRAGMA user_version"))
            if hasattr(result, "scalar_one_or_none"):
                current_version = result.scalar_one_or_none() or 0
            elif isinstance(result, (list, tuple)):
                current_version = result[0] if result else 0
            else:
                current_version = 0
            if isinstance(current_version, tuple):
                current_version = current_version[0]

            if current_version < 1:
                self._ensure_disclosure_columns(conn)
                conn.execute(text("PRAGMA user_version = 1"))
                current_version = 1

    def _ensure_disclosure_columns(self, conn: Any) -> None:
        """Add disclosure timestamp fields to older trade tables."""
        for table_name in ("political_trades", "insider_trades"):
            result = conn.execute(text(f"PRAGMA table_info({table_name})"))
            existing = [row[1] for row in result]
            if "disclosure_date" not in existing:
                conn.execute(text(
                    f"ALTER TABLE {table_name} "
                    "ADD COLUMN disclosure_date TEXT NOT NULL DEFAULT ''"
                ))
            if "disclosure_time_utc" not in existing:
                conn.execute(text(
                    f"ALTER TABLE {table_name} "
                    "ADD COLUMN disclosure_time_utc TEXT NOT NULL DEFAULT '12:00:00'"
                ))

    @staticmethod
    def _normalize_disclosure_fields(record: dict[str, Any], date_field: str) -> tuple[str, str]:
        disclosure_date = record.get("disclosure_date") or record.get("disclosure_date_str") or record.get(date_field)
        disclosure_time_utc = record.get("disclosure_time_utc") or "12:00:00"
        if not disclosure_date:
            disclosure_date = record.get(date_field, "")
        return str(disclosure_date), str(disclosure_time_utc)

    def save_congress_trades(self, trades: list[dict[str, Any]]):
        """Save a batch of congressional trades to the DB."""
        session = self.Session()
        try:
            for t in trades:
                disclosure_date, disclosure_time_utc = self._normalize_disclosure_fields(t, "trade_date")
                db_trade = DBPoliticalTrade(
                    ticker=t.get("ticker", ""),
                    politician=t.get("politician"),
                    chamber=t.get("chamber"),
                    transaction_type=t.get("transaction_type"),
                    amount_midpoint=float(t.get("amount_midpoint") or 0.0),
                    transaction_date=t.get("trade_date"),
                    disclosure_date=disclosure_date,
                    disclosure_time_utc=disclosure_time_utc,
                )
                session.add(db_trade)
            session.commit()
            logger.info("Saved %s congressional trades to database.", len(trades))
        except IntegrityError:
            session.rollback()
            logger.warning("Duplicate congressional trade ignored during save.")
        except Exception as e:
            session.rollback()
            logger.error("Failed to save congress trades: %s", e)
        finally:
            session.close()

    def upsert_congress_trade(self, record: dict[str, Any]) -> bool:
        """Insert or ignore a single congressional trade record."""
        session = self.Session()
        try:
            disclosure_date, disclosure_time_utc = self._normalize_disclosure_fields(record, "trade_date")
            db_trade = DBPoliticalTrade(
                ticker=record.get("ticker", ""),
                politician=record.get("politician"),
                chamber=record.get("chamber"),
                transaction_type=record.get("transaction_type"),
                amount_midpoint=float(record.get("amount_midpoint") or 0.0),
                transaction_date=record.get("trade_date"),
                disclosure_date=disclosure_date,
                disclosure_time_utc=disclosure_time_utc,
            )
            session.add(db_trade)
            session.commit()
            return True
        except IntegrityError:
            session.rollback()
            return False
        except Exception as exc:
            session.rollback()
            logger.error("Failed to upsert congress trade: %s", exc)
            return False
        finally:
            session.close()

    def get_congress_trades(
        self,
        ticker: str,
        days_back: int = 30,
        purchases_only: bool = True,
        start_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return congress trades matching the ticker and time window."""
        session = self.Session()
        try:
            query = session.query(DBPoliticalTrade).filter(DBPoliticalTrade.ticker == ticker.upper())
            if start_date is not None:
                query = query.filter(DBPoliticalTrade.transaction_date >= start_date)
            else:
                cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                query = query.filter(DBPoliticalTrade.transaction_date >= cutoff)
            if purchases_only:
                query = query.filter(DBPoliticalTrade.transaction_type.in_(["purchase", "buy", "p"]))
            rows = query.order_by(DBPoliticalTrade.transaction_date.desc()).all()
            return [
                {
                    "ticker": row.ticker,
                    "politician": row.politician,
                    "chamber": row.chamber,
                    "transaction_type": row.transaction_type,
                    "amount_midpoint": row.amount_midpoint,
                    "transaction_date": row.transaction_date,
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("Failed to query congress trades: %s", exc)
            return []
        finally:
            session.close()

    def save_insider_trades(self, trades: list[dict[str, Any]]):
        """Save a batch of insider trades to the DB."""
        session = self.Session()
        try:
            for t in trades:
                transaction_type = t.get("transaction_type") or t.get("trade_type") or ""
                disclosure_date, disclosure_time_utc = self._normalize_disclosure_fields(t, "transaction_date")
                db_trade = DBInsiderTrade(
                    ticker=t.get("ticker", ""),
                    insider_name=t.get("insider_name"),
                    title=t.get("title"),
                    transaction_type=transaction_type,
                    total_value=float(t.get("total_value") or 0.0),
                    shares=float(t.get("shares") or 0.0),
                    transaction_date=t.get("transaction_date"),
                    disclosure_date=disclosure_date,
                    disclosure_time_utc=disclosure_time_utc,
                )
                session.add(db_trade)
            session.commit()
            logger.info("Saved %s insider trades to database.", len(trades))
        except IntegrityError:
            session.rollback()
            logger.warning("Duplicate insider trade ignored during save.")
        except Exception as e:
            session.rollback()
            logger.error("Failed to save insider trades: %s", e)
        finally:
            session.close()

    def save_news_sentiment(
        self,
        ticker: str,
        score: float,
        magnitude: float,
        grade: str,
        headline: str | None,
        source: str | None,
        trade_date: str,
    ) -> bool:
        """Persist a single sentiment analysis record."""
        session = self.Session()
        try:
            sentiment = DBNewsSentiment(
                ticker=ticker.upper(),
                score=score,
                magnitude=magnitude,
                grade=grade,
                headline=headline,
                source=source,
                trade_date=trade_date,
            )
            session.add(sentiment)
            session.commit()
            return True
        except IntegrityError:
            session.rollback()
            return False
        except Exception as exc:
            session.rollback()
            logger.error("Failed to save news sentiment: %s", exc)
            return False
        finally:
            session.close()

    def get_news_sentiment(
        self,
        ticker: str,
        days_back: int = 30,
    ) -> list[dict[str, Any]]:
        """Return saved sentiment records for a ticker."""
        session = self.Session()
        try:
            cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            rows = (
                session.query(DBNewsSentiment)
                .filter(DBNewsSentiment.ticker == ticker.upper())
                .filter(DBNewsSentiment.trade_date >= cutoff)
                .order_by(DBNewsSentiment.trade_date.desc())
                .all()
            )
            return [
                {
                    "ticker": row.ticker,
                    "score": row.score,
                    "magnitude": row.magnitude,
                    "grade": row.grade,
                    "headline": row.headline,
                    "source": row.source,
                    "trade_date": row.trade_date,
                    "inserted_at": row.inserted_at.isoformat() if row.inserted_at else None,
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("Failed to query news sentiment: %s", exc)
            return []
        finally:
            session.close()

    def get_disclosed_signals_on_date(
        self,
        ticker: str,
        sim_date: str,
        sim_date_is_after_close: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return all publicly available signals for a date using disclosure latency rules."""
        ticker = ticker.upper()
        session = self.Session()
        try:
            if sim_date_is_after_close:
                cong_rows = (
                    session.query(DBPoliticalTrade)
                    .filter(DBPoliticalTrade.ticker == ticker)
                    .filter(DBPoliticalTrade.transaction_type.in_(["purchase", "buy", "p", "PURCHASE", "BUY"]))
                    .filter(DBPoliticalTrade.disclosure_date <= sim_date)
                    .order_by(DBPoliticalTrade.disclosure_date.desc())
                    .limit(20)
                    .all()
                )
                ins_rows = (
                    session.query(DBInsiderTrade)
                    .filter(DBInsiderTrade.ticker == ticker)
                    .filter(DBInsiderTrade.transaction_type.in_(["purchase", "buy", "p", "PURCHASE", "BUY"]))
                    .filter(DBInsiderTrade.disclosure_date <= sim_date)
                    .order_by(DBInsiderTrade.disclosure_date.desc())
                    .limit(20)
                    .all()
                )
            else:
                cong_rows = (
                    session.query(DBPoliticalTrade)
                    .filter(DBPoliticalTrade.ticker == ticker)
                    .filter(DBPoliticalTrade.transaction_type.in_(["purchase", "buy", "p", "PURCHASE", "BUY"]))
                    .filter(
                        (DBPoliticalTrade.disclosure_date < sim_date) |
                        ((DBPoliticalTrade.disclosure_date == sim_date) & (DBPoliticalTrade.disclosure_time_utc < "20:00:00"))
                    )
                    .order_by(DBPoliticalTrade.disclosure_date.desc())
                    .limit(20)
                    .all()
                )
                ins_rows = (
                    session.query(DBInsiderTrade)
                    .filter(DBInsiderTrade.ticker == ticker)
                    .filter(DBInsiderTrade.transaction_type.in_(["purchase", "buy", "p", "PURCHASE", "BUY"]))
                    .filter(
                        (DBInsiderTrade.disclosure_date < sim_date) |
                        ((DBInsiderTrade.disclosure_date == sim_date) & (DBInsiderTrade.disclosure_time_utc < "20:00:00"))
                    )
                    .order_by(DBInsiderTrade.disclosure_date.desc())
                    .limit(20)
                    .all()
                )

            sent_rows = (
                session.query(DBNewsSentiment)
                .filter(DBNewsSentiment.ticker == ticker)
                .filter(DBNewsSentiment.trade_date <= sim_date)
                .order_by(DBNewsSentiment.trade_date.desc())
                .limit(10)
                .all()
            )

            return {
                "congress": [
                    {
                        "ticker": row.ticker,
                        "politician": row.politician,
                        "chamber": row.chamber,
                        "transaction_type": row.transaction_type,
                        "amount_midpoint": row.amount_midpoint,
                        "transaction_date": row.transaction_date,
                        "disclosure_date": row.disclosure_date,
                        "disclosure_time_utc": row.disclosure_time_utc,
                    }
                    for row in cong_rows
                ],
                "insider": [
                    {
                        "ticker": row.ticker,
                        "insider_name": row.insider_name,
                        "title": row.title,
                        "transaction_type": row.transaction_type,
                        "total_value": row.total_value,
                        "shares": row.shares,
                        "transaction_date": row.transaction_date,
                        "disclosure_date": row.disclosure_date,
                        "disclosure_time_utc": row.disclosure_time_utc,
                    }
                    for row in ins_rows
                ],
                "sentiment": [
                    {
                        "ticker": row.ticker,
                        "score": row.score,
                        "magnitude": row.magnitude,
                        "grade": row.grade,
                        "headline": row.headline,
                        "source": row.source,
                        "trade_date": row.trade_date,
                        "inserted_at": row.inserted_at.isoformat() if row.inserted_at else None,
                    }
                    for row in sent_rows
                ],
            }
        except Exception as exc:
            logger.error("Failed to query disclosure-aware signals: %s", exc)
            return {"congress": [], "insider": [], "sentiment": []}
        finally:
            session.close()

    def get_stats(self) -> dict[str, int]:
        session = self.Session()
        try:
            return {
                "congress_trades": session.query(DBPoliticalTrade).count(),
                "insider_trades": session.query(DBInsiderTrade).count(),
                "news_sentiment": session.query(DBNewsSentiment).count(),
            }
        except Exception as exc:
            logger.error("Failed to query database stats: %s", exc)
            return {"congress_trades": 0, "insider_trades": 0, "news_sentiment": 0}
        finally:
            session.close()

    def check_health(self) -> dict[str, Any]:
        try:
            with self.engine.begin() as conn:
                journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar_one_or_none()
                if isinstance(journal_mode, tuple):
                    journal_mode = journal_mode[0]
            return {
                "healthy": True,
                "journal_mode": str(journal_mode).upper() if journal_mode else None,
                "table_counts": self.get_stats(),
                "message": "Database is operational.",
            }
        except Exception as exc:
            logger.error("Database health check failed: %s", exc)
            return {
                "healthy": False,
                "journal_mode": None,
                "table_counts": {},
                "message": str(exc),
            }

    def compact_database(self) -> bool:
        try:
            with self.engine.begin() as conn:
                conn.execute(text("PRAGMA wal_checkpoint(FULL)"))
                conn.execute(text("VACUUM"))
            return True
        except Exception as exc:
            logger.error("Database compact failed: %s", exc)
            return False

    def save_raw_page(
        self,
        source: str,
        url: str,
        status_code: int,
        content_type: str,
        raw_bytes: bytes,
        content_hash: str,
        headers: dict[str, str] | None = None,
    ) -> bool:
        """Persist a raw HTTP response to the raw_pages audit table."""
        session = self.Session()
        try:
            raw_record = DBRawPage(
                source=source,
                url=url,
                status_code=status_code,
                content_type=content_type,
                headers=json.dumps(headers or {}),
                content_hash=content_hash,
                raw_bytes=raw_bytes,
            )
            session.add(raw_record)
            session.commit()
            return True
        except Exception as exc:
            session.rollback()
            logger.error("Failed to save raw page: %s", exc)
            return False
        finally:
            session.close()

    def upsert_insider_trade(self, record: dict[str, Any]) -> bool:
        """Insert or ignore a single insider trade record."""
        session = self.Session()
        try:
            transaction_type = record.get("transaction_type") or record.get("trade_type") or ""
            disclosure_date, disclosure_time_utc = self._normalize_disclosure_fields(record, "transaction_date")
            trade = DBInsiderTrade(
                ticker=record.get("ticker", ""),
                insider_name=record.get("insider_name"),
                title=record.get("title"),
                transaction_type=transaction_type,
                total_value=float(record.get("total_value") or 0.0),
                shares=float(record.get("shares") or 0.0),
                transaction_date=record.get("transaction_date"),
                disclosure_date=disclosure_date,
                disclosure_time_utc=disclosure_time_utc,
            )
            session.add(trade)
            session.commit()
            return True
        except IntegrityError:
            session.rollback()
            return False
        except Exception as exc:
            session.rollback()
            logger.error("Failed to upsert insider trade: %s", exc)
            return False
        finally:
            session.close()

    def get_insider_trades(
        self,
        ticker: str,
        days_back: int = 30,
        purchases_only: bool = True,
        start_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return insider trades matching the ticker and time window."""
        session = self.Session()
        try:
            query = session.query(DBInsiderTrade).filter(DBInsiderTrade.ticker == ticker.upper())
            if start_date is not None:
                query = query.filter(DBInsiderTrade.transaction_date >= start_date)
            else:
                cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                query = query.filter(DBInsiderTrade.transaction_date >= cutoff)
            if purchases_only:
                query = query.filter(DBInsiderTrade.transaction_type.in_(["P", "B", "BUY", "PURCHASE"]))
            rows = query.order_by(DBInsiderTrade.transaction_date.desc()).all()
            return [
                {
                    "ticker": row.ticker,
                    "insider_name": row.insider_name,
                    "title": row.title,
                    "transaction_type": row.transaction_type,
                    "total_value": row.total_value,
                    "shares": row.shares,
                    "transaction_date": row.transaction_date,
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("Failed to query insider trades: %s", exc)
            return []
        finally:
            session.close()

    def upsert_signal_metadata(
        self,
        ticker: str,
        action: str,
        alpha_score: float,
        sentiment_score: float | None,
        vector: list[float] | str | None = None,
    ) -> bool:
        """Save a computed signal to the analysis history table."""
        session = self.Session()
        try:
            payload = DBSignal(
                ticker=ticker.upper(),
                action=action,
                alpha_score=float(alpha_score),
                sentiment_score=float(sentiment_score) if sentiment_score is not None else None,
                vector=json.dumps(vector) if vector is not None else None,
            )
            session.add(payload)
            session.commit()
            return True
        except Exception as exc:
            session.rollback()
            logger.error("Failed to save signal metadata: %s", exc)
            return False
        finally:
            session.close()

    def get_ticker_history(self, ticker: str, days_back: int = 365) -> list[dict[str, Any]]:
        """Retrieve saved signal history for a ticker."""
        session = self.Session()
        try:
            cutoff = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            rows = (
                session.query(DBSignal)
                .filter(DBSignal.ticker == ticker.upper())
                .filter(DBSignal.created_at >= cutoff)
                .order_by(DBSignal.created_at.desc())
                .all()
            )
            return [
                {
                    "ticker": row.ticker,
                    "action": row.action,
                    "alpha_score": row.alpha_score,
                    "sentiment_score": row.sentiment_score,
                    "vector": json.loads(row.vector) if row.vector else None,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("Failed to fetch ticker history: %s", exc)
            return []
        finally:
            session.close()

    def save_recommended_candidates(
        self,
        candidates: list[dict[str, Any]],
        source: str = "recommend",
    ) -> bool:
        session = self.Session()
        try:
            for candidate in candidates:
                payload = DBRecommendedCandidate(
                    ticker=candidate.get("ticker", "").upper(),
                    action=candidate.get("action", "RECOMMEND"),
                    alpha_score=float(candidate.get("alpha_score", 0.0)),
                    sentiment_score=(
                        float(candidate.get("sentiment_score"))
                        if candidate.get("sentiment_score") is not None
                        else None
                    ),
                    vector=json.dumps(candidate.get("vector")) if candidate.get("vector") is not None else None,
                    source=source,
                )
                session.add(payload)
            session.commit()
            return True
        except Exception as exc:
            session.rollback()
            logger.error("Failed to save recommended candidates: %s", exc)
            return False
        finally:
            session.close()

    def get_recommended_candidates(
        self,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        session = self.Session()
        try:
            rows = (
                session.query(DBRecommendedCandidate)
                .filter(DBRecommendedCandidate.alpha_score >= float(min_score))
                .order_by(DBRecommendedCandidate.alpha_score.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "ticker": row.ticker,
                    "action": row.action,
                    "alpha_score": row.alpha_score,
                    "sentiment_score": row.sentiment_score,
                    "vector": json.loads(row.vector) if row.vector else None,
                    "source": row.source,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("Failed to fetch recommended candidates: %s", exc)
            return []
        finally:
            session.close()

    def get_strategy_sessions(self, ticker: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
        session = self.Session()
        try:
            query = session.query(DBStrategySession)
            if ticker is not None:
                query = query.filter(DBStrategySession.ticker == ticker.upper())
            rows = (
                query.order_by(DBStrategySession.created_at.desc())
                .limit(limit)
                .all()
            )
            sessions: list[dict[str, Any]] = []
            for row in rows:
                metrics = {}
                try:
                    metrics = json.loads(row.performance_metrics) if row.performance_metrics else {}
                except Exception:
                    metrics = {}
                sessions.append({
                    "session_id": row.session_id,
                    "ticker": row.ticker,
                    "strategy_name": row.strategy_name,
                    "training_steps": row.training_steps,
                    "learning_rate": row.learning_rate,
                    "entropy_coef": row.entropy_coef,
                    "backtest_days": row.backtest_days,
                    "performance_metrics": metrics,
                    "suggested_change": row.suggested_change,
                    "notes": row.notes,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                })
            return sessions
        except Exception as exc:
            logger.error("Failed to fetch strategy sessions: %s", exc)
            return []
        finally:
            session.close()

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
        session = self.Session()
        try:
            payload = DBStrategySession(
                session_id=str(uuid.uuid4()),
                ticker=ticker.upper(),
                strategy_name=strategy_name,
                training_steps=int(training_steps),
                learning_rate=float(learning_rate),
                entropy_coef=float(entropy_coef),
                backtest_days=int(backtest_days),
                performance_metrics=json.dumps(performance_metrics),
                suggested_change=suggested_change,
                notes=notes,
            )
            session.add(payload)
            session.commit()
            return True
        except Exception as exc:
            session.rollback()
            logger.error("Failed to insert strategy session: %s", exc)
            return False
        finally:
            session.close()

    def prune_strategy_sessions(self, keep_days: int = 365) -> int:
        session = self.Session()
        try:
            cutoff = datetime.utcnow() - timedelta(days=keep_days)
            deleted = session.query(DBStrategySession).filter(DBStrategySession.created_at < cutoff).delete()
            session.commit()
            return deleted
        except Exception as exc:
            session.rollback()
            logger.error("Failed to prune strategy sessions: %s", exc)
            return 0
        finally:
            session.close()


DBManager = DatabaseManager
