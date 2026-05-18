import pandas as pd
import numpy as np
import pytest
from types import SimpleNamespace

import main
import rl_insider_trader as rlt


class DummyInnerScraper:
    def __init__(self, db=None):
        self.db = db

    def run(self, *args, **kwargs):
        return None

    def fetch_for_tickers(self, tickers, days_back=2):
        return {ticker.upper(): [] for ticker in tickers}


class DummyManager:
    def __init__(self, db=None):
        self.db = db
        self.congress_scraper = DummyInnerScraper(db=db)
        self.insider_scraper = DummyInnerScraper(db=db)

    def fetch_news_and_context(self, tickers):
        return {ticker.upper(): SimpleNamespace(score=0.0) for ticker in tickers}


class DummyAgent:
    def __init__(self, *args, **kwargs):
        pass

    def select_action(self, state, momentum_score=0.0, sentiment_score=0.0):
        return 1, {"alpha_score": 0.42}


class DummyDB:
    def upsert_signal_metadata(self, *args, **kwargs):
        return True


def test_run_golden_loop_with_safe_stubs(monkeypatch):
    monkeypatch.setattr(main, "DBManager", DummyDB)
    monkeypatch.setattr(main, "ScraperManager", DummyManager)
    monkeypatch.setattr(main, "FlippyAgent", DummyAgent)
    monkeypatch.setattr(
        main,
        "fetch_price_history",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "Close": [100.0, 101.0, 102.0, 103.0],
                "Volume": [1000, 1200, 1100, 1050],
                "rsi": [55.0, 56.0, 57.0, 58.0],
                "hist_vol": [0.2, 0.21, 0.19, 0.18],
                "sma50": [95.0, 96.0, 97.0, 98.0],
                "sma200": [90.0, 91.0, 92.0, 93.0],
                "high_52w_pct": [0.3, 0.32, 0.33, 0.34],
            }
        ),
    )

    signals = main.run_golden_loop(["AAPL"], days_back=1)

    assert isinstance(signals, list)
    assert len(signals) == 1
    assert signals[0]["ticker"] == "AAPL"
    assert isinstance(signals[0]["action"], str)
    assert signals[0]["alpha_score"] == 0.42


def test_build_trading_environment_returns_valid_env(monkeypatch):
    def fake_price_history(*args, **kwargs):
        n = kwargs.get("n_days", 30) if "n_days" in kwargs else 30
        dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
        return pd.DataFrame({
            "Date": dates,
            "Close": np.linspace(100.0, 130.0, n),
            "Open": np.linspace(99.5, 129.0, n),
            "High": np.linspace(100.5, 131.0, n),
            "Low": np.linspace(99.0, 128.0, n),
            "Volume": np.full(n, 1_000_000.0),
        })

    monkeypatch.setattr(rlt, "fetch_price_history", fake_price_history)
    monkeypatch.setattr(
        rlt,
        "build_live_alt_signals",
        lambda ticker, df, db=None, sentiment_processor=None, start_date=None: rlt.generate_alt_signals(ticker, len(df)),
    )

    env, df, alts = rlt.build_trading_environment("AAPL", n_days=30, db=None, sentiment_processor=None)

    assert df.shape[0] == 30
    assert len(alts) == 30
    obs, info = env.reset(seed=0)

    assert obs.shape == (12,)
    assert obs.dtype == np.float32
    assert info == {}
    assert env.action_space.n == 3
    assert env.observation_space.shape == (12,)


def _build_test_env(prices: list[float]) -> rlt.InsiderTradingEnv:
    n = len(prices)
    df = pd.DataFrame({
        "Date": pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B"),
        "Close": prices,
        "Open": prices,
        "High": prices,
        "Low": prices,
        "Volume": np.full(n, 1_000_000.0),
        "rsi": np.full(n, 50.0),
        "macd": np.full(n, 0.0),
        "sma50": prices,
        "sma200": prices,
        "high_52w_pct": np.minimum(np.array(prices) / max(prices), 1.0),
        "vol_trend": np.full(n, 1.0),
        "log_return": np.zeros(n),
    })
    alt_signals = [
        {"insider_sim": 0.0, "congress_sim": 0.0, "sentiment": 0.0, "gex": 0.0}
        for _ in range(n)
    ]
    return rlt.InsiderTradingEnv(df, alt_signals)


def test_buy_action_is_capped_to_ten_percent_exposure():
    env = _build_test_env([100.0] * 30)
    _, _ = env.reset(seed=0)

    _, reward, terminated, truncated, _ = env.step(2)

    assert env.balance == pytest.approx(90_000.0, rel=1e-6)
    assert env.shares == pytest.approx((10_000.0 - 10_000.0 * 0.0015) / 100.0, rel=1e-6)
    assert env.entry_price == 100.0
    assert env.stop_price == pytest.approx(95.0, rel=1e-6)
    assert reward <= 0.0 or reward >= -5.0
    assert not terminated
    assert not truncated


def test_trailing_stop_loss_triggers_at_five_percent_drawdown():
    prices = [100.0] * 27 + [94.9, 100.0, 100.0]
    env = _build_test_env(prices)
    _, _ = env.reset(seed=0)

    env.step(2)
    obs, reward, terminated, truncated, _ = env.step(1)

    assert env.shares == 0.0
    assert env.entry_price is None
    assert env.stop_price is None
    assert env.history[-1]["stop_loss"] is True
    assert not terminated
    assert not truncated


def test_reward_is_clipped_to_five():
    prices = [1.0] * 27 + [1000.0, 1000.0, 1000.0]
    env = _build_test_env(prices)
    _, _ = env.reset(seed=0)

    env.step(2)
    obs, reward, terminated, truncated, _ = env.step(1)

    assert reward == pytest.approx(5.0, rel=1e-6)
    assert env.history[-1]["reward"] == pytest.approx(5.0, rel=1e-6)
    assert not terminated
    assert not truncated
