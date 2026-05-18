from __future__ import annotations

from datetime import datetime
from typing import List
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
from intelligence.llm_advisor import LocalLLMAdvisor
from intelligence.training_manager import AutonomousTrainingConfig, AutonomousTrainingManager
from config.settings import SETTINGS
from api.fast_api_app import app, TickerAnalysisRequest, BatchTickerAnalysisRequest, StrategyBacktestRequest
from rl_insider_trader import build_trading_environment

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    # Start the FastAPI application with Uvicorn
    print("Starting Flippy Intelligence Engine API...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_golden_loop(tickers: List[str], days_back: int = 2):
    """
    The main function that executes the entire analysis pipeline for a batch of tickers.
    """
    print("--- Starting Analysis Cycle ---")

    # 1. Initialization (Setup)
    db_manager = DBManager()
    scraper_manager = ScraperManager(db=db_manager)
    agent = FlippyAgent()

    # 2. Data Acquisition (Scan & Correlate)
    print("Step 2/5: Fetching and unifying raw data...")
    signals: list[dict[str, object]] = []

    for ticker in tickers:
        try:
            try:
                scraper_manager.congress_scraper.run(days_back=days_back, ticker=ticker)
            except Exception as exc:
                logger.warning("Congress scraper failed for %s: %s", ticker, exc)

            try:
                scraper_manager.insider_scraper.run(days_back=days_back)
            except Exception as exc:
                logger.warning("Insider scraper failed for %s: %s", ticker, exc)

            try:
                congress_trades = scraper_manager.congress_scraper.fetch_for_tickers([ticker], days_back=days_back)
            except Exception as exc:
                logger.warning("Congress fetch failed for %s: %s", ticker, exc)
                congress_trades = {ticker.upper(): []}

            try:
                insider_trades = scraper_manager.insider_scraper.fetch_for_tickers([ticker], days_back=days_back)
            except Exception as exc:
                logger.warning("Insider fetch failed for %s: %s", ticker, exc)
                insider_trades = {ticker.upper(): []}

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
                congress_trades=congress_trades.get(ticker.upper(), []),
                insider_trades=insider_trades.get(ticker.upper(), []),
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

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

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

    autotune_parser = subparsers.add_parser("autotune", help="Run an autonomous tuning loop for a ticker.")
    autotune_parser.add_argument("ticker", help="Ticker symbol to tune.")
    autotune_parser.add_argument("--training-steps", type=int, default=250000)
    autotune_parser.add_argument("--learning-rate", type=float, default=3e-4)
    autotune_parser.add_argument("--entropy-coef", type=float, default=0.02)
    autotune_parser.add_argument("--backtest-days", type=int, default=2520)
    autotune_parser.add_argument("--max-iterations", type=int, default=2)
    autotune_parser.add_argument("--stop-on-plateau", action="store_true")
    autotune_parser.add_argument("--start-date", default=None)

    analyze_parser = subparsers.add_parser("analyze", help="Run the golden analysis loop for a list of tickers.")
    analyze_parser.add_argument("tickers", nargs="+", help="Tickers to analyze.")
    analyze_parser.add_argument("--days-back", type=int, default=2)

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)

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

    elif args.command == "analyze":
        print(f"Running analysis loop for {args.tickers}...")
        signals = run_golden_loop(args.tickers, days_back=args.days_back)
        print(signals)
