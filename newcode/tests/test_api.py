from fastapi.testclient import TestClient

from api.fast_api_app import app, get_db_manager


class FakeDBAdapter:
    def get_congress_trades(self, ticker, days_back=30, purchases_only=True, start_date=None):
        return []

    def get_insider_trades(self, ticker, days_back=30, purchases_only=True, start_date=None):
        return []

    def get_news_sentiment(self, ticker, days_back=30):
        return []

    def get_stats(self):
        return {
            "congress_trades": 0,
            "insider_trades": 0,
            "news_sentiment": 0,
            "analysis_signals": 0,
            "strategy_sessions": 0,
        }

    def get_disclosed_signals_on_date(self, ticker, sim_date, sim_date_is_after_close=False):
        return {"congress": [], "insider": [], "sentiment": []}

    def upsert_signal_metadata(self, *args, **kwargs):
        return True

    def get_strategy_sessions(self, ticker, limit=10):
        ticker = ticker or "AAPL"
        return [
            {
                "session_id": "abc123",
                "ticker": ticker,
                "strategy_name": "PPO_TEST",
                "training_steps": 100000,
                "learning_rate": 0.0003,
                "entropy_coef": 0.02,
                "backtest_days": 2520,
                "performance_metrics": {
                    "total_return": 12.3,
                    "sharpe": 1.23,
                    "sortino": 1.55,
                    "max_drawdown": 10.1,
                    "win_rate": 52.0,
                },
                "suggested_change": "Continue training",
                "notes": "Test session.",
                "created_at": "2026-05-18T00:00:00",
            }
        ]

    def prune_strategy_sessions(self, keep_days=365):
        return 1

    def get_ticker_history(self, ticker, days_back=365):
        return []


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "2.0"}


def test_strategy_sessions_endpoint_returns_history():
    app.dependency_overrides[get_db_manager] = lambda: FakeDBAdapter()
    client = TestClient(app)

    response = client.get("/api/v1/strategy-sessions/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data[0]["ticker"] == "AAPL"
    assert data[0]["performance_metrics"]["sharpe"] == 1.23

    app.dependency_overrides.clear()


def test_list_strategy_sessions_endpoint_returns_all():
    app.dependency_overrides[get_db_manager] = lambda: FakeDBAdapter()
    client = TestClient(app)

    response = client.get("/api/v1/strategy-sessions?limit=1")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data[0]["ticker"] == "AAPL"

    app.dependency_overrides.clear()


def test_prune_strategy_sessions_endpoint():
    app.dependency_overrides[get_db_manager] = lambda: FakeDBAdapter()
    client = TestClient(app)

    response = client.delete("/api/v1/strategy-sessions?keep_days=30")
    assert response.status_code == 200
    assert response.json() == {"removed": 1, "keep_days": 30}

    app.dependency_overrides.clear()


def test_autotune_recommend_endpoint_returns_safe_update():
    app.dependency_overrides[get_db_manager] = lambda: FakeDBAdapter()
    client = TestClient(app)

    response = client.post(
        "/api/v1/autotune/recommend",
        json={
            "ticker": "AAPL",
            "training_steps": 250000,
            "learning_rate": 0.0003,
            "entropy_coef": 0.02,
            "backtest_days": 2520,
            "max_iterations": 1,
            "stop_on_plateau": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert "suggested_change" in data
    assert "config_updates" in data

    app.dependency_overrides.clear()


def test_autotune_history_endpoint_returns_sessions():
    app.dependency_overrides[get_db_manager] = lambda: FakeDBAdapter()
    client = TestClient(app)

    response = client.get("/api/v1/autotune/history?ticker=AAPL&limit=1")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert data[0]["ticker"] == "AAPL"
    assert data[0]["strategy_name"] == "PPO_TEST"

    app.dependency_overrides.clear()
