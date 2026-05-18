import json
import os
from pathlib import Path

import pytest

from database.db_manager import DBManager, create_db_adapter, validate_db_adapter
from rl_insider_trader import MarketIntelligenceDB


def test_db_manager_complies_with_api_adapter_contract():
    db = DBManager()
    validate_db_adapter(
        db,
        required_methods=(
            "get_congress_trades",
            "get_insider_trades",
            "get_news_sentiment",
            "get_stats",
            "upsert_signal_metadata",
            "get_ticker_history",
        ),
    )


def test_market_intelligence_db_complies_with_realtime_adapter_contract():
    db = MarketIntelligenceDB()
    validate_db_adapter(db)


def test_market_intelligence_db_analysis_signal_history():
    tmp_path = Path("./data/test_insider_rl.db")
    if tmp_path.exists():
        tmp_path.unlink()
    db = MarketIntelligenceDB(tmp_path)

    assert db.upsert_signal_metadata(
        ticker="AAPL",
        action="BUY",
        alpha_score=0.35,
        sentiment_score=0.12,
        vector=[0.1, 0.2, 0.3],
    ) is True

    history = db.get_ticker_history("AAPL", days_back=7)
    assert len(history) == 1
    assert history[0]["action"] == "BUY"
    assert history[0]["alpha_score"] == 0.35
    assert history[0]["vector"] == [0.1, 0.2, 0.3]

    stats = db.get_stats()
    assert stats["analysis_signals"] >= 1

    # Leave the temporary test DB in place; Windows can hold file locks for active connections.


def test_market_intelligence_db_watchlist_persistence(tmp_path):
    db_path = tmp_path / "watchlist.db"
    db = MarketIntelligenceDB(db_path)

    assert db.get_watchlist_tickers() == []
    assert db.add_watchlist_ticker("TSLA", source="manual") is True
    assert len(db.get_watchlist_tickers()) == 1

    assert db.add_watchlist_ticker("tsla", source="manual") is True
    assert len(db.get_watchlist_tickers()) == 1

    assert db.remove_watchlist_ticker("TSLA") == 1
    assert db.get_watchlist_tickers() == []

    assert db.add_watchlist_ticker("AAPL", source="manual") is True
    assert db.clear_watchlist() == 1
    assert db.get_watchlist_tickers() == []


def test_market_intelligence_db_focus_settings_persistence(tmp_path):
    db_path = tmp_path / "focus.db"
    db = MarketIntelligenceDB(db_path)

    settings = {
        "insider_flow_weight": 2.1,
        "congress_flow_weight": 1.9,
        "momentum_horizon_days": 126,
        "max_trade_size_pct": 0.12,
        "per_ticker_risk_budget_pct": 0.15,
        "use_watchlist_only": False,
    }
    assert db.save_focus_settings(settings) is True

    loaded = db.get_focus_settings()
    assert loaded["insider_flow_weight"] == 2.1
    assert loaded["congress_flow_weight"] == 1.9
    assert loaded["momentum_horizon_days"] == 126
    assert loaded["max_trade_size_pct"] == 0.12
    assert loaded["per_ticker_risk_budget_pct"] == 0.15
    assert loaded["use_watchlist_only"] is False


def test_market_intelligence_db_recent_hit_tickers(tmp_path):
    db_path = tmp_path / "hits.db"
    db = MarketIntelligenceDB(db_path)

    db.insert_congress_trade({
        "ticker": "MSFT",
        "politician": "Alice Smith",
        "trade_type": "Purchase",
        "trade_date": "2025-12-01",
        "amount_range": "$50,001 – $100,000",
    })
    db.insert_insider_trade({
        "ticker": "AAPL",
        "insider_name": "Executive_1",
        "position": "CEO",
        "shares_traded": 1000,
        "price": 150.0,
        "trade_type": "Purchase",
        "trade_date": "2025-12-02",
    })

    hits = db.get_recent_hit_tickers(days_back=365)
    assert "MSFT" in hits
    assert "AAPL" in hits


def test_market_intelligence_db_temporal_disclosure_holdback(tmp_path):
    db_path = tmp_path / "disclosure.db"
    db = MarketIntelligenceDB(db_path)
    sim_date = "2026-05-17"
    inserted_at = "2026-05-17T12:00:00"

    with db._conn() as conn:
        conn.execute(
            "INSERT INTO congress_trades (ticker, politician, trade_type, amount_range, trade_date, disclosure_date, disclosure_time_utc, latency_days, inserted_at) VALUES (?,?,?,?,?,?,?,?,?)",
            ("AAPL", "Alice Smith", "Purchase", "$15,001 – $50,000", sim_date, sim_date, "20:30:00", 1, inserted_at),
        )

    signals_open = db.get_disclosed_signals_on_date("AAPL", sim_date, sim_date_is_after_close=False)
    assert len(signals_open["congress"]) == 0

    signals_close = db.get_disclosed_signals_on_date("AAPL", sim_date, sim_date_is_after_close=True)
    assert len(signals_close["congress"]) == 1


def test_validate_db_adapter_rejects_missing_methods():
    class BadAdapter:
        pass

    with pytest.raises(TypeError, match="missing required methods"):
        validate_db_adapter(BadAdapter())


def test_create_db_adapter_returns_valid_manager():
    adapter = create_db_adapter(validate_methods=(
        "get_congress_trades",
        "get_insider_trades",
        "get_news_sentiment",
        "get_stats",
        "upsert_signal_metadata",
        "get_ticker_history",
    ))
    assert adapter is not None
    assert hasattr(adapter, "get_congress_trades")
    assert hasattr(adapter, "get_insider_trades")


def test_db_manager_disclosed_signal_method_exists_and_is_explicit():
    db = DBManager()
    with pytest.raises(NotImplementedError):
        db.get_disclosed_signals_on_date("AAPL", "2026-05-17")
