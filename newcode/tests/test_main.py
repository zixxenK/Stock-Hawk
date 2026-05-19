from datetime import datetime

import pytest

import main


class DummyDB:
    def __init__(self, sessions=None, candidates=None):
        self.sessions = sessions or {}
        self.candidates = candidates or []
        self.calls = []

    def get_strategy_sessions(self, ticker: str, limit: int = 1):
        self.calls.append(("get_strategy_sessions", ticker, limit))
        return self.sessions.get(ticker.upper(), [])

    def get_recommended_candidates(self, limit=10, min_score=0.0):
        self.calls.append(("get_recommended_candidates", limit, min_score))
        return self.candidates[:limit]


def test_recommend_training_candidates_filters_and_limits():
    signals = [
        {"ticker": "AAPL", "alpha_score": 0.10},
        {"ticker": "MSFT", "alpha_score": 0.88},
        {"ticker": "NVDA", "alpha_score": 0.72},
    ]

    candidates = main._recommend_training_candidates(signals, limit=2)

    assert len(candidates) == 2
    assert candidates[0]["ticker"] == "MSFT"
    assert candidates[1]["ticker"] == "NVDA"


def test_get_persisted_recommended_candidates_queries_db_manager():
    expected = [{"ticker": "AAPL", "alpha_score": 0.9}]
    db = DummyDB(candidates=expected)

    candidates = main._get_persisted_recommended_candidates(db, limit=1, min_score=0.5)

    assert candidates == expected
    assert ("get_recommended_candidates", 1, 0.5) in db.calls


def test_run_autonomous_training_candidates_skips_recent_sessions(monkeypatch):
    now = datetime.utcnow().isoformat()
    db = DummyDB(sessions={"SKIPPED": [{"created_at": now}]})
    candidates = [
        {"ticker": "SKIPPED", "alpha_score": 0.95},
        {"ticker": "RAN", "alpha_score": 0.92},
    ]

    reports = []

    def fake_run_autotune_loop(**kwargs):
        reports.append(kwargs)
        return {"status": "completed", "ticker": kwargs["ticker"]}

    monkeypatch.setattr(main, "run_autotune_loop", fake_run_autotune_loop)

    result = main._run_autonomous_training_candidates(
        db_manager=db,
        candidates=candidates,
        training_steps=10,
        learning_rate=0.001,
        entropy_coef=0.01,
        backtest_days=30,
        max_iterations=1,
        stop_on_plateau=True,
        start_date=None,
        skip_recent_hours=24,
    )

    assert len(result) == 1
    assert result[0]["ticker"] == "RAN"
    assert reports[0]["ticker"] == "RAN"
    assert ("get_strategy_sessions", "SKIPPED", 1) in db.calls
    assert ("get_strategy_sessions", "RAN", 1) in db.calls
