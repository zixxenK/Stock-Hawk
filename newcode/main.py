from __future__ import annotations

from datetime import datetime, timedelta
import argparse
import dataclasses
import logging

import pandas as pd
from database.db_manager import DBManager
from data_sources.scraper_manager import ScraperManager
from data_sources.yahoo_data import fetch_price_history
from processing.sentiment_processor import SentimentProcessor
from processing.vectorizer import VectorInput, build_vector
from intelligence.agent import ACTION_NAMES, FlippyAgent
from intelligence.backtest import simulate_backtest_for_tickers
from intelligence.llm_advisor import LocalLLMAdvisor
from intelligence.training_manager import AutonomousTrainingConfig, AutonomousTrainingManager
from config.settings import SETTINGS
from api.fast_api_app import app, TickerAnalysisRequest, BatchTickerAnalysisRequest, StrategyBacktestRequest
from rl_insider_trader import build_trading_environment

logger = logging.getLogger(__name__)


def _normalize_ticker_universe(tickers: list[str] | None) -> list[str]:
    if not tickers:
        return []
    return sorted({ticker.strip().upper() for ticker in tickers if isinstance(ticker, str) and ticker.strip()})


def _resolve_default_ticker_universe(db_manager: DBManager) -> list[str]:
    if SETTINGS.scan.use_watchlist_only:
        watchlist = [row["ticker"] for row in db_manager.get_watchlist_tickers()]
        if watchlist:
            return _normalize_ticker_universe(watchlist)

    universe = _normalize_ticker_universe(SETTINGS.scan.default_universe)
    if not universe:
        universe = db_manager.get_recent_hit_tickers(days_back=SETTINGS.scan.recent_hit_days)

    if not universe:
        universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]

    watchlist = [row["ticker"] for row in db_manager.get_watchlist_tickers()]
    merged = _normalize_ticker_universe(universe + watchlist)
    if len(merged) > SETTINGS.scan.max_scan_tickers:
        merged = merged[:SETTINGS.scan.max_scan_tickers]
    return merged


def _recommend_training_candidates(signals: list[dict[str, object]], limit: int | None = None) -> list[dict[str, object]]:
    threshold = SETTINGS.scan.candidate_score_threshold
    top_candidates = [signal for signal in sorted(signals, key=lambda x: float(x.get("alpha_score", 0.0)), reverse=True) if float(signal.get("alpha_score", 0.0)) >= threshold]
    if limit is None:
        limit = SETTINGS.scan.top_candidates
    return top_candidates[:limit]


def _persist_recommendations(db_manager: DBManager, candidates: list[dict[str, object]], source: str = "cli_recommend") -> None:
    try:
        if not candidates:
            return
        if hasattr(db_manager, "save_recommended_candidates"):
            db_manager.save_recommended_candidates(candidates, source=source)
    except Exception as exc:
        logger.warning("Unable to persist recommended candidates: %s", exc)


def _get_persisted_recommended_candidates(
    db_manager: DBManager,
    limit: int | None = None,
    min_score: float | None = None,
) -> list[dict[str, object]]:
    if not hasattr(db_manager, "get_recommended_candidates"):
        return []
    if limit is None:
        limit = SETTINGS.scan.top_candidates
    if min_score is None:
        min_score = SETTINGS.scan.candidate_score_threshold
    return db_manager.get_recommended_candidates(limit=limit, min_score=min_score)


def _recent_training_session_exists(
    db_manager: DBManager,
    ticker: str,
    hours: int = 24,
) -> bool:
    if not hasattr(db_manager, "get_strategy_sessions"):
        return False
    sessions = db_manager.get_strategy_sessions(ticker=ticker, limit=1)
    if not sessions:
        return False
    created_at = sessions[0].get("created_at")
    if not created_at:
        return False
    try:
        created = datetime.fromisoformat(created_at)
    except Exception:
        return False
    return datetime.utcnow() - created < timedelta(hours=hours)


def _run_autonomous_training_candidates(
    db_manager: DBManager,
    candidates: list[dict[str, object]],
    training_steps: int,
    learning_rate: float,
    entropy_coef: float,
    backtest_days: int,
    max_iterations: int,
    stop_on_plateau: bool,
    start_date: str | None = None,
    skip_recent_hours: int = 24,
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for candidate in candidates:
        ticker = str(candidate.get("ticker", "")).upper().strip()
        if not ticker:
            continue

        if _recent_training_session_exists(db_manager, ticker, hours=skip_recent_hours):
            logger.info("Skipping autonomous training for %s because a recent session exists.", ticker)
            continue

        print(f"Starting autonomous training for recommended candidate {ticker} (score={candidate.get('alpha_score', 0.0):.4f})...")
        report = run_autotune_loop(
            ticker=ticker,
            training_steps=training_steps,
            learning_rate=learning_rate,
            entropy_coef=entropy_coef,
            backtest_days=backtest_days,
            max_iterations=max_iterations,
            stop_on_plateau=stop_on_plateau,
            start_date=start_date,
        )
        reports.append({"ticker": ticker, "report": report})

    return reports


def _build_ticker_vector(
    ticker: str,
    congress_trades: list[dict],
    insider_trades: list[dict],
    sentiment_score: float,
    price_data: pd.DataFrame | None = None,
    vol_trend: float | None = None,
) -> "np.ndarray":
    mom_12_1 = None
    mom_6m = None
    mom_3m = None
    rs_vs_spy = None
    high_52w_pct = None
    adx = None
    rsi = None
    hist_vol = None
    golden_cross = False

    if price_data is not None and not price_data.empty:
        close = price_data["Close"]
        if len(close) >= 252 and close.iloc[-252] > 0:
            mom_12_1 = float(close.iloc[-1] / close.iloc[-252] - 1.0)
        if len(close) >= 126 and close.iloc[-126] > 0:
            mom_6m = float(close.iloc[-1] / close.iloc[-126] - 1.0)
        if len(close) >= 63 and close.iloc[-63] > 0:
            mom_3m = float(close.iloc[-1] / close.iloc[-63] - 1.0)
        if "rsi" in price_data.columns:
            rsi = float(price_data["rsi"].iloc[-1])
        if "hist_vol" in price_data.columns:
            hist_vol = float(price_data["hist_vol"].iloc[-1])
        if "sma50" in price_data.columns and "sma200" in price_data.columns:
            golden_cross = float(price_data["sma50"].iloc[-1]) > float(price_data["sma200"].iloc[-1])
        if "high_52w_pct" in price_data.columns:
            high_52w_pct = float(price_data["high_52w_pct"].iloc[-1])
        if vol_trend is None and "Volume" in price_data.columns:
            vol_trend = float(price_data["Volume"].iloc[-1] / max(price_data["Volume"].rolling(20, min_periods=1).mean().iloc[-1], 1.0))

    return build_vector(VectorInput(
        mom_12_1=mom_12_1,
        mom_6m=mom_6m,
        mom_3m=mom_3m,
        rs_vs_spy=rs_vs_spy,
        high_52w_pct=high_52w_pct,
        adx=adx,
        rsi=rsi,
        vol_trend=vol_trend,
        golden_cross=golden_cross,
        hist_vol=hist_vol,
        congress_buys=sum(
            1
            for trade in congress_trades
            if str(trade.get("transaction_type", "")).upper() in {"B", "BUY", "P", "PURCHASE"}
        ),
        insider_buys=sum(
            1
            for trade in insider_trades
            if str(trade.get("transaction_type", trade.get("trade_type", ""))).upper() in {"P", "B", "BUY", "PURCHASE"}
        ),
        insider_weighted=sum(
            float(trade.get("total_value", 0) or 0) / 1_000_000
            for trade in insider_trades
            if str(trade.get("transaction_type", trade.get("trade_type", ""))).upper() in {"P", "B", "BUY", "PURCHASE"}
        ),
        options_bullish=0,
        suspicious_vol=bool(congress_trades and insider_trades),
        sector="",
        days_to_earnings=None,
        market_regime="mixed",
        sentiment_score=sentiment_score,
    ))




def run_golden_loop(tickers: list[str] | None = None, days_back: int = 2):
    """
    The main function that executes the entire analysis pipeline for a batch of tickers.
    If tickers are not provided, the runtime universe is resolved from settings and watchlist state.
    """
    print("--- Starting Analysis Cycle ---")

    # 1. Initialization (Setup)
    db_manager = DBManager()
    if not tickers:
        tickers = _resolve_default_ticker_universe(db_manager)

    tickers = _normalize_ticker_universe(tickers)
    if not tickers:
        raise ValueError("No tickers available for analysis. Please configure a default universe or seed the watchlist.")

    scraper_manager = ScraperManager(db=db_manager)
    agent = FlippyAgent()

    # 2. Data Acquisition (Scan & Correlate)
    print(f"Step 2/5: Fetching and unifying raw data for {len(tickers)} tickers...")
    signals: list[dict[str, object]] = []

    for ticker in tickers:
        try:
            all_trades = scraper_manager.fetch_all_data([ticker], days_back=days_back)
            ticker_trades = all_trades.get(ticker.upper(), [])
            congress_trades = [trade for trade in ticker_trades if trade.get("politician") is not None]
            insider_trades = [trade for trade in ticker_trades if trade.get("insider_name") is not None]
        except Exception as exc:
            logger.warning("Trade ingestion failed for %s: %s", ticker, exc)
            congress_trades = []
            insider_trades = []

        sentiment_results = scraper_manager.fetch_news_and_context([ticker])
        sentiment_result = sentiment_results.get(ticker.upper())

        market_df = fetch_price_history(ticker, period="1y", interval="1d")
        if market_df.empty:
            logger.warning("Yahoo price history unavailable for %s; continuing with neutral market vector.", ticker)
            market_df = pd.DataFrame()

        vol_trend = 1.0
        if not market_df.empty and "Volume" in market_df.columns:
            vol_trend = float(market_df["Volume"].iloc[-1] / max(market_df["Volume"].rolling(20, min_periods=1).mean().iloc[-1], 1.0))

        current_vector = _build_ticker_vector(
            ticker=ticker,
            congress_trades=congress_trades,
            insider_trades=insider_trades,
            sentiment_score=sentiment_result.score if sentiment_result else 0.0,
            price_data=market_df if not market_df.empty else None,
            vol_trend=vol_trend,
        )

        action_idx, action_details = agent.select_action(
            state=current_vector,
            momentum_score=50.0,
            sentiment_score=sentiment_result.score if sentiment_result else 0.0,
        )
        action_name = ACTION_NAMES.get(action_idx, "UNKNOWN")

        print(f"\n\t✅ Signal Found for {ticker}: Action={action_name} Score={action_details['alpha_score']:.2f}")
        db_manager.upsert_signal_metadata(
            ticker=ticker,
            action=action_name,
            alpha_score=action_details.get("alpha_score", 0.0),
            sentiment_score=sentiment_result.score if sentiment_result else None,
            vector=current_vector.tolist(),
        )

        signals.append({
            "ticker": ticker,
            "action": action_name,
            "alpha_score": action_details.get("alpha_score", 0.0),
            "sentiment_score": sentiment_result.score if sentiment_result else None,
            "vector": current_vector.tolist(),
        })

    print("--- Analysis Cycle Complete ---")
    return signals


def run_autotune_loop(
    ticker: str,
    training_steps: int = 250_000,
    learning_rate: float = 3e-4,
    entropy_coef: float = 0.02,
    backtest_days: int = 2520,
    max_iterations: int = 2,
    stop_on_plateau: bool = True,
    start_date: str | None = None,
) -> dict[str, object]:
    db_manager = DBManager()
    advisor = LocalLLMAdvisor()
    trainer = AutonomousTrainingManager(
        db=db_manager,
        advisor=advisor,
        env_builder=build_trading_environment,
    )
    config = AutonomousTrainingConfig(
        training_steps=training_steps,
        learning_rate=learning_rate,
        entropy_coef=entropy_coef,
        backtest_days=backtest_days,
        strategy_name=f"PPO_{ticker}_autotune",
        min_sharpe_improvement=0.05,
        plateau_iterations=2 if stop_on_plateau else 1000,
    )
    start_date_obj = None
    if start_date:
        start_date_obj = datetime.fromisoformat(start_date).date()

    report = trainer.run_autonomous_loop(
        ticker=ticker,
        initial_config=config,
        sentiment_processor=SentimentProcessor(),
        start_date=start_date_obj,
        max_iterations=max_iterations,
    )

    return dataclasses.asdict(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flippy Intelligence Engine command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI server.")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the Streamlit high-dimensional dashboard.")
    dashboard_parser.add_argument(
        "--script",
        default="visualization/high_dim_state_dashboard.py",
        help="Path to the Streamlit dashboard script.",
    )

    autotune_parser = subparsers.add_parser("autotune", help="Run an autonomous tuning loop for a ticker.")
    autotune_parser.add_argument("ticker", help="Ticker symbol to tune.")
    autotune_parser.add_argument("--training-steps", type=int, default=250000)
    autotune_parser.add_argument("--learning-rate", type=float, default=3e-4)
    autotune_parser.add_argument("--entropy-coef", type=float, default=0.02)
    autotune_parser.add_argument("--backtest-days", type=int, default=2520)
    autotune_parser.add_argument("--max-iterations", type=int, default=2)
    autotune_parser.add_argument("--stop-on-plateau", action="store_true")
    autotune_parser.add_argument("--start-date", default=None)

    scan_parser = subparsers.add_parser("scan", help="Scan the configured universe for candidate signals.")
    scan_parser.add_argument("--tickers", nargs="*", help="Optional tickers to scan. If omitted, the default universe is used.")
    scan_parser.add_argument("--days-back", type=int, default=2, help="How many days of disclosure data to scan.")

    recommend_parser = subparsers.add_parser("recommend", help="Recommend training candidates from a scan.")
    recommend_parser.add_argument("--tickers", nargs="*", help="Optional tickers to recommend from.")
    recommend_parser.add_argument("--days-back", type=int, default=2, help="How many days of disclosure data to scan.")

    autonomous_parser = subparsers.add_parser(
        "autonomous",
        help="Run autonomous training on recommended tickers the agent identifies as high-opportunity.",
    )
    autonomous_parser.add_argument("--tickers", nargs="*", help="Optional tickers to scan or train.")
    autonomous_parser.add_argument("--days-back", type=int, default=2, help="How many days of disclosure data to scan for recommendations.")
    autonomous_parser.add_argument("--limit", type=int, default=SETTINGS.scan.top_candidates, help="Maximum number of recommended tickers to train on.")
    autonomous_parser.add_argument("--min-score", type=float, default=SETTINGS.scan.candidate_score_threshold, help="Minimum alpha score to qualify as a candidate.")
    autonomous_parser.add_argument("--refresh", action="store_true", help="Refresh recommendations by scanning again before training.")
    autonomous_parser.add_argument("--skip-recent-hours", type=int, default=24, help="Skip tickers that were trained within the last n hours.")
    autonomous_parser.add_argument("--training-steps", type=int, default=250000)
    autonomous_parser.add_argument("--learning-rate", type=float, default=3e-4)
    autonomous_parser.add_argument("--entropy-coef", type=float, default=0.02)
    autonomous_parser.add_argument("--backtest-days", type=int, default=2520)
    autonomous_parser.add_argument("--max-iterations", type=int, default=2)
    autonomous_parser.add_argument("--stop-on-plateau", action="store_true")
    autonomous_parser.add_argument("--start-date", default=None)

    analyze_parser = subparsers.add_parser("analyze", help="Run the golden analysis loop for a list of tickers.")
    analyze_parser.add_argument("tickers", nargs="+", help="Tickers to analyze.")
    analyze_parser.add_argument("--days-back", type=int, default=2)

    ingest_parser = subparsers.add_parser("ingest", help="Run a managed data ingestion pass for specified tickers.")
    ingest_parser.add_argument("--tickers", nargs="*", help="Optional tickers to ingest. If omitted, the default universe is used.")
    ingest_parser.add_argument("--days-back", type=int, default=30)

    backtest_parser = subparsers.add_parser("backtest", help="Run a trade execution backtest against historical signals.")
    backtest_parser.add_argument("tickers", nargs="+", help="Tickers to backtest.")
    backtest_parser.add_argument("--days-back", type=int, default=252)

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.command == "dashboard":
        import subprocess
        import sys

        print("Launching Streamlit dashboard...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", args.script],
            check=True,
        )

    elif args.command == "autotune":
        print(f"Running autonomous tuning for {args.ticker}...")
        report = run_autotune_loop(
            ticker=args.ticker,
            training_steps=args.training_steps,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            backtest_days=args.backtest_days,
            max_iterations=args.max_iterations,
            stop_on_plateau=args.stop_on_plateau,
            start_date=args.start_date,
        )
        print(report)

    elif args.command == "scan":
        targets = args.tickers if args.tickers else None
        print("Scanning universe for signals...")
        signals = run_golden_loop(targets, days_back=args.days_back)
        print(signals)

    elif args.command == "recommend":
        targets = args.tickers if args.tickers else None
        print("Scanning universe and recommending top candidates...")
        signals = run_golden_loop(targets, days_back=args.days_back)
        candidates = _recommend_training_candidates(signals)
        db_manager = DBManager()
        _persist_recommendations(db_manager, candidates, source="cli_recommend")
        print({"recommended_candidates": candidates})

    elif args.command == "autonomous":
        targets = args.tickers if args.tickers else None
        db_manager = DBManager()
        candidates: list[dict[str, object]] = []

        if args.refresh:
            print("Refreshing recommendations before autonomous training...")
            signals = run_golden_loop(targets, days_back=args.days_back)
            candidates = _recommend_training_candidates(signals, limit=args.limit)
            _persist_recommendations(db_manager, candidates, source="cli_autonomous")
        else:
            candidates = _get_persisted_recommended_candidates(
                db_manager,
                limit=args.limit,
                min_score=args.min_score,
            )
            if not candidates:
                print("No persisted recommended candidates found. Scanning to generate fresh recommendations...")
                signals = run_golden_loop(targets, days_back=args.days_back)
                candidates = _recommend_training_candidates(signals, limit=args.limit)
                _persist_recommendations(db_manager, candidates, source="cli_autonomous")

        if not candidates:
            print("No recommended candidates available for autonomous training.")
        else:
            reports = _run_autonomous_training_candidates(
                db_manager=db_manager,
                candidates=candidates,
                training_steps=args.training_steps,
                learning_rate=args.learning_rate,
                entropy_coef=args.entropy_coef,
                backtest_days=args.backtest_days,
                max_iterations=args.max_iterations,
                stop_on_plateau=args.stop_on_plateau,
                start_date=args.start_date,
                skip_recent_hours=args.skip_recent_hours,
            )
            print({"autonomous_training_reports": reports})

    elif args.command == "analyze":
        print(f"Running analysis loop for {args.tickers}...")
        signals = run_golden_loop(args.tickers, days_back=args.days_back)
        print(signals)

    elif args.command == "ingest":
        print("Running ingestion cycle...")
        db_manager = DBManager()
        scraper_manager = ScraperManager(db=db_manager)
        tickers = args.tickers if args.tickers else _resolve_default_ticker_universe(db_manager)
        summary = scraper_manager.ingest_tickers(tickers, days_back=args.days_back)
        print(summary)

    elif args.command == "backtest":
        print(f"Running execution backtest for {args.tickers}...")
        db_manager = DBManager()
        ticker_histories = {
            ticker.upper(): db_manager.get_ticker_history(ticker.upper(), args.days_back)
            for ticker in args.tickers
        }
        summary = simulate_backtest_for_tickers(ticker_histories, days_back=args.days_back)
        print(summary)
