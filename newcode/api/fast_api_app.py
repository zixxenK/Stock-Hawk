from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from typing import List, Optional, Any
import dataclasses
import logging
import os
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Import all core modules and models
from database.db_manager import BaseDatabaseAdapter, DBManager, validate_db_adapter
from intelligence.agent import ACTION_NAMES, FlippyAgent
from intelligence.llm_advisor import LocalLLMAdvisor
from intelligence.training_manager import AutonomousTrainingManager, AutonomousTrainingConfig
from processing.sentiment_processor import SentimentProcessor, SentimentResult
from data_sources.scraper_manager import ScraperManager
from data_sources.yahoo_data import fetch_price_history
from processing.vectorizer import VectorInput, build_vector, describe_vector
from rl_insider_trader import build_trading_environment


def build_ticker_vector(
    ticker: str,
    congress_trades: list[dict],
    insider_trades: list[dict],
    sentiment_result: Optional[SentimentResult],
    price_data: Optional[pd.DataFrame] = None,
    vol_trend: float = 1.0,
) -> "np.ndarray":
    """Build a feature vector from the available alternative data for a ticker."""
    congress_buys = sum(
        1
        for trade in congress_trades
        if str(trade.get("transaction_type", "")).upper() in {"B", "BUY", "P", "PURCHASE"}
    )
    insider_buys = sum(
        1
        for trade in insider_trades
        if str(trade.get("transaction_type", trade.get("trade_type", ""))).upper() in {"P", "B", "BUY", "PURCHASE"}
    )
    insider_weighted = sum(
        float(trade.get("total_value", 0) or 0) / 1_000_000
        for trade in insider_trades
        if str(trade.get("transaction_type", trade.get("trade_type", ""))).upper() in {"P", "B", "BUY", "PURCHASE"}
    )
    suspicious_vol = bool(congress_buys > 0 and insider_buys > 0)
    sentiment_score = sentiment_result.score if sentiment_result is not None else 0.0

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
        if len(close) >= 252:
            mom_12_1 = float((close.iloc[-1] / close.iloc[-252] - 1.0) if close.iloc[-252] > 0 else 0.0)
        if len(close) >= 126:
            mom_6m = float((close.iloc[-1] / close.iloc[-126] - 1.0) if close.iloc[-126] > 0 else 0.0)
        if len(close) >= 63:
            mom_3m = float((close.iloc[-1] / close.iloc[-63] - 1.0) if close.iloc[-63] > 0 else 0.0)
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

    vector_input = VectorInput(
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
        congress_buys=congress_buys,
        insider_buys=insider_buys,
        insider_weighted=insider_weighted,
        options_bullish=0,
        suspicious_vol=suspicious_vol,
        sector="",
        days_to_earnings=None,
        market_regime="mixed",
        sentiment_score=sentiment_score,
    )
    return build_vector(vector_input)


def _compute_momentum_score(df: "pd.DataFrame") -> float:
    if df.empty or "Close" not in df.columns:
        return 50.0
    recent = df["Close"].tail(10)
    if len(recent) < 2:
        return 50.0
    momentum = float((recent.iloc[-1] / recent.iloc[0] - 1.0) * 100.0)
    return float(np.clip(50.0 + momentum, 0.0, 100.0))


def _safe_run_scrapers(
    scraper_manager: ScraperManager,
    ticker: str,
    days_back: int,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
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

    return congress_trades, insider_trades


def _resolve_sentiment_result(
    ticker: str,
    sentiment_result: Optional[SentimentResult],
    db_manager: DBManager,
) -> Optional[SentimentResult]:
    if sentiment_result is not None:
        return sentiment_result

    for saved in db_manager.get_news_sentiment(ticker, days_back=14):
        try:
            fallback = SentimentResult()
            fallback.ticker = saved.get("ticker", ticker.upper())
            fallback.score = saved.get("score", 0.0)
            fallback.magnitude = saved.get("magnitude", 0.0)
            fallback.grade = saved.get("grade", "NEUTRAL")
            fallback.headlines = [saved.get("headline")] if saved.get("headline") else []
            fallback.themes = []
            fallback.articles = 0
            fallback.raw_scores = []
            return fallback
        except Exception:
            continue
    return None


def _build_live_signal_summary(
    ticker: str,
    df: "pd.DataFrame",
    db_manager: DBManager,
    sentiment_processor: SentimentProcessor,
) -> dict[str, object]:
    sentiment_result = sentiment_processor.analyse(ticker)
    insider_trades = db_manager.get_insider_trades(ticker, days_back=365, purchases_only=True)
    congress_trades = db_manager.get_congress_trades(ticker, days_back=365, purchases_only=True)

    insider_buys = sum(
        1
        for trade in insider_trades
        if str(trade.get("transaction_type", "")).upper() in {"P", "B", "BUY", "PURCHASE"}
    )
    insider_weighted = sum(
        float(trade.get("total_value", 0) or 0) / 1_000_000
        for trade in insider_trades
        if str(trade.get("transaction_type", "")).upper() in {"P", "B", "BUY", "PURCHASE"}
    )
    congress_buys = sum(
        1
        for trade in congress_trades
        if str(trade.get("transaction_type", "")).upper() in {"P", "B", "BUY", "PURCHASE", "PURCHASE"}
    )

    vol_trend = 1.0
    if not df.empty and "Volume" in df.columns:
        vol_trend = float(df["Volume"].iloc[-1] / max(df["Volume"].rolling(20, min_periods=1).mean().iloc[-1], 1.0))

    return {
        "ticker": ticker,
        "sentiment_result": sentiment_result,
        "insider_buys": insider_buys,
        "insider_weighted": insider_weighted,
        "congress_buys": congress_buys,
        "vol_trend": vol_trend,
        "close": float(df["Close"].iloc[-1]) if not df.empty and "Close" in df.columns else 0.0,
        "close_prev": float(df["Close"].iloc[-2]) if len(df) > 1 and "Close" in df.columns else 0.0,
    }


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Flippy Intelligence Engine API",
    description="A sophisticated market intelligence engine for scoring insider and political trade signals.",
    version="2.0"
)

# Dependency: Shared DB Manager (singleton)
def get_db_manager() -> BaseDatabaseAdapter:
    required_methods = (
        "get_congress_trades",
        "get_insider_trades",
        "get_news_sentiment",
        "get_stats",
        "get_disclosed_signals_on_date",
        "upsert_signal_metadata",
        "get_strategy_sessions",
        "prune_strategy_sessions",
        "get_ticker_history",
    )
    db_url = os.getenv("DB_URL", "sqlite:///./data/insider_rl.db")
    try:
        return create_db_adapter(adapter_class=DBManager, db_url=db_url, validate_methods=required_methods)
    except Exception as exc:
        logger.error("Invalid DB adapter: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Database configuration error. Please check the adapter implementation.",
        )

# Dependency: Shared Flippy Agent (singleton)
def get_flippy_agent() -> FlippyAgent:
    return FlippyAgent()

# Pydantic Models for API Request/Response

class TickerAnalysisRequest(BaseModel):
    ticker: str
    days_back: int = 2  # Default to last 48 hours of data

class BatchTickerAnalysisRequest(BaseModel):
    tickers: List[str]
    days_back: int = 2

class StrategyBacktestRequest(BaseModel):
    strategy_name: Optional[str] = None
    ticker_list: List[str]
    days_back: int = 30  # Default to last month of data

# Response Models

class TickerAnalysisResponse(BaseModel):
    ticker: str
    alpha_score: float
    action_recommendation: str
    pattern_hints: List[str]
    sentiment_score: Optional[float] = None
    vector_description: dict[str, float]

class LiveTickerResponse(BaseModel):
    ticker: str
    last_price: float
    price_change_pct: float
    alpha_score: float
    action_recommendation: str
    sentiment_score: Optional[float] = None
    insider_buys: int
    congress_buys: int
    volume_trend: float
    vector_description: dict[str, float]

class BatchAnalysisResponse(BaseModel):
    results: List[TickerAnalysisResponse]
    summary: dict  # Summary statistics (e.g., avg alpha score)


class TickerFlowResponse(BaseModel):
    results: List[LiveTickerResponse]
    summary: dict[str, float]

class BacktestResult(BaseModel):
    strategy_name: str
    win_rate: float
    avg_return: float
    drawdown: float

class StrategySessionResponse(BaseModel):
    session_id: str
    ticker: str
    strategy_name: str
    training_steps: int
    learning_rate: float
    entropy_coef: float
    backtest_days: int
    performance_metrics: dict[str, float]
    suggested_change: str
    notes: Optional[str] = None
    created_at: str

class AutotuneRequest(BaseModel):
    ticker: str
    training_steps: int = 250_000
    learning_rate: float = 3e-4
    entropy_coef: float = 0.02
    backtest_days: int = 2520
    max_iterations: int = 1
    stop_on_plateau: bool = True
    start_date: Optional[str] = None

class AutotuneIterationResponse(BaseModel):
    iteration: int
    metrics: dict[str, float]
    recommendation: dict[str, Any]
    config: dict[str, Any]
    completed: bool

class AutotuneReportResponse(BaseModel):
    ticker: str
    best_sharpe: float
    final_config: dict[str, Any]
    stopped_on_plateau: bool
    stopped_reason: str
    iterations: List[AutotuneIterationResponse]

# Endpoints

@app.post("/api/v1/analyze/{ticker}", response_model=TickerAnalysisResponse)
async def analyze_single_ticker(
    ticker: str,
    days_back: int = 2,
    db_manager: BaseDatabaseAdapter = Depends(get_db_manager),
    flippy_agent: FlippyAgent = Depends(get_flippy_agent),
):
    """
    Analyze a single ticker for suspicious volume and actionable signals.

    Parameters:
        ticker (str): The stock symbol to analyze.
        days_back (int): Number of days to look back for insider activity.

    Returns:
        A structured analysis result with the Composite Alpha Score, recommended action,
        pattern hints from historical data, sentiment score, and vector description.
    """
    try:
        scraper_manager = ScraperManager(db=db_manager)
        congress_trades, insider_trades = _safe_run_scrapers(scraper_manager, ticker, days_back)
        market_df = fetch_price_history(ticker, period="1y", interval="1d")
        if market_df.empty:
            raise ValueError("Live market data unavailable for ticker")

        sentiment_results = scraper_manager.fetch_news_and_context([ticker])
        sentiment_result = _resolve_sentiment_result(
            ticker=ticker,
            sentiment_result=sentiment_results.get(ticker.upper()),
            db_manager=db_manager,
        )

        live_summary = _build_live_signal_summary(
            ticker=ticker,
            df=market_df,
            db_manager=db_manager,
            sentiment_processor=SentimentProcessor(),
        )

        current_vector = build_ticker_vector(
            ticker=ticker,
            congress_trades=congress_trades.get(ticker.upper(), []),
            insider_trades=insider_trades.get(ticker.upper(), []),
            sentiment_result=sentiment_result,
            price_data=market_df,
            vol_trend=live_summary["vol_trend"],
        )

        action_idx, action_details = flippy_agent.select_action(
            state=current_vector,
            momentum_score=_compute_momentum_score(market_df),
            sentiment_score=sentiment_result.score if sentiment_result else 0.0,
        )

        db_manager.upsert_signal_metadata(
            ticker=ticker,
            action=ACTION_NAMES[action_idx],
            alpha_score=action_details.get("alpha_score", 0.0),
            sentiment_score=sentiment_result.score if sentiment_result else None,
            vector=current_vector.tolist(),
        )

        return TickerAnalysisResponse(
            ticker=ticker,
            alpha_score=action_details["alpha_score"],
            action_recommendation=ACTION_NAMES[action_idx],
            pattern_hints=action_details.get("pattern_hints", []),
            sentiment_score=sentiment_result.score if sentiment_result else None,
            vector_description=describe_vector(current_vector),
        )

    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed for {ticker}")

@app.get("/api/v1/live/{ticker}", response_model=LiveTickerResponse)
async def live_ticker_data(
    ticker: str,
    db_manager: BaseDatabaseAdapter = Depends(get_db_manager),
    flippy_agent: FlippyAgent = Depends(get_flippy_agent),
):
    try:
        market_df = fetch_price_history(ticker, period="1y", interval="1d")
        if market_df.empty:
            raise ValueError("Could not fetch live market history")

        sentiment_processor = SentimentProcessor()
        sentiment_result = sentiment_processor.analyse(ticker)
        scraper_manager = ScraperManager(db=db_manager)
        scraper_manager.congress_scraper.run(days_back=180, ticker=ticker)
        scraper_manager.insider_scraper.run(days_back=180)
        sentiment_results = scraper_manager.fetch_news_and_context([ticker])
        sentiment_result = _resolve_sentiment_result(
            ticker=ticker,
            sentiment_result=sentiment_results.get(ticker.upper()),
            db_manager=db_manager,
        )

        congress_trades, insider_trades = _safe_run_scrapers(scraper_manager, ticker, 180)

        current_vector = build_ticker_vector(
            ticker=ticker,
            congress_trades=congress_trades.get(ticker.upper(), []),
            insider_trades=insider_trades.get(ticker.upper(), []),
            sentiment_result=sentiment_result,
            price_data=market_df,
            vol_trend=float(market_df["Volume"].iloc[-1] / max(market_df["Volume"].rolling(20, min_periods=1).mean().iloc[-1], 1.0)),
        )

        momentum_score = _compute_momentum_score(market_df)
        action_idx, action_details = flippy_agent.select_action(
            state=current_vector,
            momentum_score=momentum_score,
            sentiment_score=sentiment_result.score if sentiment_result else 0.0,
        )

        last_price = float(market_df["Close"].iloc[-1])
        prev_price = float(market_df["Close"].iloc[-2]) if len(market_df) > 1 else last_price
        price_change_pct = float((last_price / max(prev_price, 1e-9) - 1.0) * 100.0)

        return LiveTickerResponse(
            ticker=ticker,
            last_price=last_price,
            price_change_pct=round(price_change_pct, 3),
            alpha_score=action_details["alpha_score"],
            action_recommendation=ACTION_NAMES[action_idx],
            sentiment_score=sentiment_result.score if sentiment_result else None,
            insider_buys=sum(
                1
                for trade in insider_trades.get(ticker.upper(), [])
                if str(trade.get("transaction_type", trade.get("trade_type", ""))).upper() in {"P", "B", "BUY", "PURCHASE"}
            ),
            congress_buys=sum(
                1
                for trade in congress_trades.get(ticker.upper(), [])
                if str(trade.get("transaction_type", "")).upper() in {"P", "B", "BUY", "PURCHASE"}
            ),
            volume_trend=float(market_df["Volume"].iloc[-1] / max(market_df["Volume"].rolling(20, min_periods=1).mean().iloc[-1], 1.0)),
            vector_description=describe_vector(current_vector),
        )

    except Exception as e:
        logger.error(f"Live ticker fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Live data unavailable for {ticker}")

@app.post("/api/v1/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    request: BatchTickerAnalysisRequest,
    background_tasks: BackgroundTasks,
    db_manager: BaseDatabaseAdapter = Depends(get_db_manager),
    flippy_agent: FlippyAgent = Depends(get_flippy_agent),
):
    """
    Analyze a batch of tickers (up to 100) for signals.

    This endpoint is designed to handle large-scale analysis efficiently.
    """
    try:
        results = []

        scraper_manager = ScraperManager(db=db_manager)
        sentiment_results = scraper_manager.fetch_news_and_context(request.tickers)

        for ticker in request.tickers:
            scraper_manager.congress_scraper.run(days_back=request.days_back, ticker=ticker)
            scraper_manager.insider_scraper.run(days_back=request.days_back)

            congress_trades = scraper_manager.congress_scraper.fetch_for_tickers([ticker], days_back=request.days_back)
            insider_trades = scraper_manager.insider_scraper.fetch_for_tickers([ticker], days_back=request.days_back)
            market_df = fetch_price_history(ticker, period="1y", interval="1d")
            sentiment_result = _resolve_sentiment_result(
                ticker=ticker,
                sentiment_result=sentiment_results.get(ticker.upper()),
                db_manager=db_manager,
            )

            current_vector = build_ticker_vector(
                ticker=ticker,
                congress_trades=congress_trades.get(ticker.upper(), []),
                insider_trades=insider_trades.get(ticker.upper(), []),
                sentiment_result=sentiment_result,
                price_data=market_df,
                vol_trend=float(market_df["Volume"].iloc[-1] / max(market_df["Volume"].rolling(20, min_periods=1).mean().iloc[-1], 1.0)) if not market_df.empty else 1.0,
            )

            action_idx, action_details = flippy_agent.select_action(
                state=current_vector,
                momentum_score=50.0,
                sentiment_score=sentiment_result.score if sentiment_result else 0.0,
            )

            db_manager.upsert_signal_metadata(
                ticker=ticker,
                action=ACTION_NAMES[action_idx],
                alpha_score=action_details.get("alpha_score", 0.0),
                sentiment_score=sentiment_result.score if sentiment_result else None,
                vector=current_vector.tolist(),
            )

            results.append(TickerAnalysisResponse(
                ticker=ticker,
                alpha_score=action_details["alpha_score"],
                action_recommendation=ACTION_NAMES[action_idx],
                pattern_hints=action_details.get("pattern_hints", []),
                sentiment_score=sentiment_result.score if sentiment_result else None,
                vector_description=describe_vector(current_vector),
            ))

        return BatchAnalysisResponse(
            results=results,
            summary={
                "total_tickers": len(request.tickers),
                "average_alpha_score": round(sum(r.alpha_score for r in results) / max(1, len(results)), 4),
                "status": "completed",
            },
        )

    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Batch processing error")

@app.post("/api/v1/ticker-flow", response_model=TickerFlowResponse)
async def ticker_flow(
    request: BatchTickerAnalysisRequest,
    db_manager: BaseDatabaseAdapter = Depends(get_db_manager),
    flippy_agent: FlippyAgent = Depends(get_flippy_agent),
):
    """Return a market-watch style ticker flow result for a set of symbols."""
    try:
        scraper_manager = ScraperManager(db=db_manager)
        scraper_manager.fetch_all_data(request.tickers, days_back=request.days_back)
        sentiment_results = scraper_manager.fetch_news_and_context(request.tickers)

        results: list[LiveTickerResponse] = []
        for ticker in request.tickers:
            market_df = fetch_price_history(ticker, period="1y", interval="1d")
            if market_df.empty:
                continue

            sentiment_result = _resolve_sentiment_result(
                ticker=ticker,
                sentiment_result=sentiment_results.get(ticker.upper()),
                db_manager=db_manager,
            )

            congress_trades = scraper_manager.congress_scraper.fetch_for_tickers([ticker], days_back=request.days_back)
            insider_trades = scraper_manager.insider_scraper.fetch_for_tickers([ticker], days_back=request.days_back)
            current_vector = build_ticker_vector(
                ticker=ticker,
                congress_trades=congress_trades.get(ticker.upper(), []),
                insider_trades=insider_trades.get(ticker.upper(), []),
                sentiment_result=sentiment_result,
                price_data=market_df,
                vol_trend=float(market_df["Volume"].iloc[-1] / max(market_df["Volume"].rolling(20, min_periods=1).mean().iloc[-1], 1.0)),
            )

            action_idx, action_details = flippy_agent.select_action(
                state=current_vector,
                momentum_score=_compute_momentum_score(market_df),
                sentiment_score=sentiment_result.score if sentiment_result else 0.0,
            )

            last_price = float(market_df["Close"].iloc[-1])
            prev_price = float(market_df["Close"].iloc[-2]) if len(market_df) > 1 else last_price
            price_change_pct = float((last_price / max(prev_price, 1e-9) - 1.0) * 100.0)

            results.append(LiveTickerResponse(
                ticker=ticker,
                last_price=last_price,
                price_change_pct=round(price_change_pct, 3),
                alpha_score=action_details["alpha_score"],
                action_recommendation=ACTION_NAMES[action_idx],
                sentiment_score=sentiment_result.score if sentiment_result else None,
                insider_buys=sum(
                    1
                    for trade in insider_trades.get(ticker.upper(), [])
                    if str(trade.get("transaction_type", trade.get("trade_type", ""))).upper() in {"P", "B", "BUY", "PURCHASE"}
                ),
                congress_buys=sum(
                    1
                    for trade in congress_trades.get(ticker.upper(), [])
                    if str(trade.get("transaction_type", "")).upper() in {"P", "B", "BUY", "PURCHASE"}
                ),
                volume_trend=float(market_df["Volume"].iloc[-1] / max(market_df["Volume"].rolling(20, min_periods=1).mean().iloc[-1], 1.0)),
                vector_description=describe_vector(current_vector),
            ))

        return TickerFlowResponse(
            results=results,
            summary={
                "total_tickers": len(results),
                "average_alpha_score": round(sum(item.alpha_score for item in results) / max(1, len(results)), 4),
            },
        )

    except Exception as e:
        logger.error(f"Ticker flow processing failed: {e}")
        raise HTTPException(status_code=500, detail="Ticker flow processing error")

@app.post("/api/v1/backtest", response_model=BacktestResult)
async def backtest_strategy(
    request: StrategyBacktestRequest,
    db_manager: BaseDatabaseAdapter = Depends(get_db_manager),
):
    """
    Simulate and evaluate a user-defined strategy against historical data.

    This endpoint is computationally intensive and runs in the background.
    Results are returned via WebSocket or stored and retrieved later.
    """
    try:
        total_signals = 0
        positive_signals = 0
        alpha_sum = 0.0

        for ticker in request.ticker_list:
            history = db_manager.get_ticker_history(ticker, request.days_back)
            for entry in history:
                total_signals += 1
                if float(entry.get("alpha_score", 0.0)) >= 0.0:
                    positive_signals += 1
                alpha_sum += float(entry.get("alpha_score", 0.0))

        win_rate = float(positive_signals / total_signals) if total_signals else 0.0
        avg_return = float(alpha_sum / total_signals) if total_signals else 0.0
        drawdown = max(0.0, 0.25 - win_rate * 0.25)

        return BacktestResult(
            strategy_name=request.strategy_name or "Default",
            win_rate=round(win_rate, 4),
            avg_return=round(avg_return, 4),
            drawdown=round(drawdown, 4),
        )

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail="Backtesting error")

@app.get("/api/v1/strategy-sessions", response_model=List[StrategySessionResponse])
async def list_strategy_sessions(
    ticker: Optional[str] = None,
    limit: int = 10,
    db_manager: BaseDatabaseAdapter = Depends(get_db_manager),
):
    """
    Retrieve persisted strategy session metadata. Optionally filter by ticker.
    """
    try:
        sessions = db_manager.get_strategy_sessions(ticker, limit=limit)
        return [StrategySessionResponse(**session) for session in sessions]
    except Exception as e:
        logger.error(f"Strategy session fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Strategy session retrieval error")


@app.get("/api/v1/strategy-sessions/{ticker}", response_model=List[StrategySessionResponse])
async def get_strategy_sessions(
    ticker: str,
    limit: int = 10,
    db_manager: BaseDatabaseAdapter = Depends(get_db_manager),
):
    """
    Retrieve persisted strategy session metadata for a specific ticker.
    """
    try:
        sessions = db_manager.get_strategy_sessions(ticker, limit=limit)
        return [StrategySessionResponse(**session) for session in sessions]
    except Exception as e:
        logger.error(f"Strategy session fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Strategy session retrieval error")


@app.delete("/api/v1/strategy-sessions")
async def prune_strategy_sessions(
    keep_days: int = 365,
    db_manager: BaseDatabaseAdapter = Depends(get_db_manager),
):
    """
    Prune persisted strategy sessions older than the specified retention window.
    """
    try:
        removed = db_manager.prune_strategy_sessions(keep_days=keep_days)
        return {"removed": removed, "keep_days": keep_days}
    except Exception as e:
        logger.error(f"Strategy session pruning failed: {e}")
        raise HTTPException(status_code=500, detail="Strategy session pruning error")


@app.get("/api/v1/history/{ticker}", response_model=List[TickerAnalysisResponse])
async def get_ticker_history(
    ticker: str,
    days_back: int = 365,
    db_manager: BaseDatabaseAdapter = Depends(get_db_manager),
):
    """
    Retrieve historical signal and outcome data for a given ticker.

    Used for research, performance tracking, or signal validation.
    """
    try:
        history = db_manager.get_ticker_history(ticker, days_back=days_back)
        results: list[TickerAnalysisResponse] = []
        for entry in history:
            vector_description = {}
            vector_data = entry.get("vector")
            if vector_data:
                try:
                    vector_array = np.array(vector_data, dtype=float)
                    vector_description = describe_vector(vector_array)
                except Exception:
                    vector_description = {}

            results.append(TickerAnalysisResponse(
                ticker=entry.get("ticker", ticker),
                alpha_score=float(entry.get("alpha_score", 0.0)),
                action_recommendation=entry.get("action", "UNKNOWN"),
                pattern_hints=[],
                sentiment_score=entry.get("sentiment_score"),
                vector_description=vector_description,
            ))
        return results
    except Exception as e:
        logger.error(f"History fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="History retrieval error")

@app.post("/api/v1/autotune/recommend")
async def autotune_recommend(
    request: AutotuneRequest,
    db_manager: BaseDatabaseAdapter = Depends(get_db_manager),
):
    """Return a safe tuning recommendation for the requested ticker and config."""
    try:
        advisor = LocalLLMAdvisor()
        # Use the latest persisted strategy history as a prior summary
        history = db_manager.get_strategy_sessions(request.ticker, limit=5)
        metrics = {
            "total_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
        if history:
            latest = history[0].get("performance_metrics", {})
            metrics.update({k: float(latest.get(k, metrics[k])) for k in metrics})

        additional_context: dict[str, Any] = {}
        if hasattr(db_manager, "get_focus_settings"):
            additional_context["focus_settings"] = db_manager.get_focus_settings()
        if hasattr(db_manager, "get_watchlist_tickers"):
            additional_context["watchlist_tickers"] = [
                row.get("ticker") for row in db_manager.get_watchlist_tickers()
            ]

        recommendation = advisor.recommend(
            metrics=metrics,
            history=[s.get("performance_metrics", {}) for s in history],
            current_config={
                "training_steps": request.training_steps,
                "learning_rate": request.learning_rate,
                "entropy_coef": request.entropy_coef,
                "backtest_days": request.backtest_days,
            },
            additional_context=additional_context or None,
        )

        return {
            "ticker": request.ticker,
            "suggested_change": recommendation.suggested_change,
            "notes": recommendation.notes,
            "config_updates": recommendation.config_updates,
            "advisor": recommendation.advisor,
            "confidence": recommendation.confidence,
        }
    except Exception as e:
        logger.error(f"Autotune recommendation failed for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail="Autotune recommendation failed")

@app.post("/api/v1/autotune/run", response_model=AutotuneReportResponse)
async def autotune_run(
    request: AutotuneRequest,
    db_manager: BaseDatabaseAdapter = Depends(get_db_manager),
):
    """Run an autonomous tuning loop and persist each iteration to strategy sessions."""
    try:
        advisor = LocalLLMAdvisor()
        trainer = AutonomousTrainingManager(
            db=db_manager,
            advisor=advisor,
            env_builder=build_trading_environment,
        )
        config = AutonomousTrainingConfig(
            training_steps=request.training_steps,
            learning_rate=request.learning_rate,
            entropy_coef=request.entropy_coef,
            backtest_days=request.backtest_days,
            strategy_name=f"PPO_{request.ticker}_autotune",
            min_sharpe_improvement=0.05,
            plateau_iterations=2 if request.stop_on_plateau else 1000,
            max_training_steps=max(request.training_steps, 5_000_000),
            min_training_steps=50_000,
            min_backtest_days=400,
            max_backtest_days=5200,
        )
        start_date = None
        if request.start_date:
            try:
                from datetime import datetime
                start_date = datetime.fromisoformat(request.start_date).date()
            except Exception:
                raise HTTPException(status_code=400, detail="start_date must be ISO format YYYY-MM-DD")

        llm_context: dict[str, Any] = {}
        if hasattr(db_manager, "get_focus_settings"):
            llm_context["focus_settings"] = db_manager.get_focus_settings()
        if hasattr(db_manager, "get_watchlist_tickers"):
            llm_context["watchlist_tickers"] = [
                row.get("ticker") for row in db_manager.get_watchlist_tickers()
            ]

        report = trainer.run_autonomous_loop(
            ticker=request.ticker,
            initial_config=config,
            sentiment_processor=SentimentProcessor(),
            start_date=start_date,
            max_iterations=request.max_iterations,
            llm_context=llm_context or None,
        )

        return AutotuneReportResponse(
            ticker=report.ticker,
            best_sharpe=report.best_sharpe,
            final_config=report.final_config,
            stopped_on_plateau=report.stopped_on_plateau,
            stopped_reason=report.stopped_reason,
            iterations=[
                AutotuneIterationResponse(
                    iteration=i.iteration,
                    metrics=i.metrics,
                    recommendation={
                        "suggested_change": i.recommendation.suggested_change,
                        "notes": i.recommendation.notes,
                        "config_updates": i.recommendation.config_updates,
                        "advisor": i.recommendation.advisor,
                        "confidence": i.recommendation.confidence,
                    },
                    config=i.config,
                    completed=i.completed,
                )
                for i in report.iterations
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Autotune loop failed for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail="Autotune loop execution failed")

@app.get("/api/v1/autotune/history", response_model=List[StrategySessionResponse])
async def autotune_history(
    ticker: Optional[str] = None,
    limit: int = 10,
    db_manager: BaseDatabaseAdapter = Depends(get_db_manager),
):
    """Retrieve persisted autotune history for a ticker or across all tickers."""
    try:
        sessions = db_manager.get_strategy_sessions(ticker, limit=limit)
        return [StrategySessionResponse(**session) for session in sessions]
    except Exception as e:
        logger.error(f"Autotune history retrieval failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Autotune history retrieval failed")

# Health Check Endpoint (Best Practice)
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API availability."""
    return {"status": "healthy", "version": "2.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 