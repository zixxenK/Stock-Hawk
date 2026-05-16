from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from typing import List, Optional
import logging
from pydantic import BaseModel

# Import all core modules and models
from database.db_manager import DBManager
from intelligence.agent import FlippyAgent
from processing.sentiment_processor import SentimentResult
from data_sources.scraper_manager import ScraperManager
from processing.vectorizer import VectorInputModel, describe_vector

# Initialize logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Flippy Intelligence Engine API",
    description="A sophisticated market intelligence engine for scoring insider and political trade signals.",
    version="2.0"
)

# Dependency: Shared DB Manager (singleton)
def get_db_manager() -> DBManager:
    return DBManager()

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

class BatchAnalysisResponse(BaseModel):
    results: List[TickerAnalysisResponse]
    summary: dict  # Summary statistics (e.g., avg alpha score)

class BacktestResult(BaseModel):
    strategy_name: str
    win_rate: float
    avg_return: float
    drawdown: float

# Endpoints

@app.post("/api/v1/analyze/{ticker}", response_model=TickerAnalysisResponse)
async def analyze_single_ticker(
    ticker: str,
    days_back: int = 2,
    db_manager: DBManager = Depends(get_db_manager),
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
        # 1. Fetch all relevant data using ScraperManager
        scraper_manager = ScraperManager(db=db_manager)
        raw_trades = scraper_manager.fetch_all_data([ticker], days_back=days_back)
        sentiment_result = scraper_manager.fetch_news_and_context([ticker], days_back=days_back)

        # 2. Build the feature vector
        current_vector = build_vector(...) 

        # 3. Get action and score from FlippyAgent
        action_idx, action_details = flippy_agent.select_action(
            state=current_vector,
            momentum_score=50.0,
            sentiment_score=sentiment_result.get(ticker, {}).score if sentiment_result else 0.0
        )

        # 4. Format response
        return TickerAnalysisResponse(
            ticker=ticker,
            alpha_score=action_details["alpha_score"],
            action_recommendation=ACTION_NAMES[action_idx],
            pattern_hints=action_details.get("pattern_hints", []),
            sentiment_score=sentiment_result.get(ticker, {}).score if sentiment_result else None,
            vector_description=describe_vector(current_vector)
        )

    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed for {ticker}")

@app.post("/api/v1/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    request: BatchTickerAnalysisRequest,
    background_tasks: BackgroundTasks,
    db_manager: DBManager = Depends(get_db_manager),
    flippy_agent: FlippyAgent = Depends(get_flippy_agent),
):
    """
    Analyze a batch of tickers (up to 100) for signals.

    This endpoint is designed to handle large-scale analysis efficiently.
    """
    try:
        results = []

        # Process in background to avoid blocking the user
        def run_batch_analysis():
            scraper_manager = ScraperManager(db=db_manager)
            sentiment_results = scraper_manager.fetch_news_and_context(request.tickers, request.days_back)

            for ticker in request.tickers:
                raw_trades = scraper_manager.fetch_all_data([ticker], days_back=request.days_back)
                current_vector = build_vector(...)

                action_idx, action_details = flippy_agent.select_action(
                    state=current_vector,
                    momentum_score=50.0,
                    sentiment_score=sentiment_results.get(ticker, {}).score if sentiment_results else 0.0
                )

                results.append(TickerAnalysisResponse(
                    ticker=ticker,
                    alpha_score=action_details["alpha_score"],
                    action_recommendation=ACTION_NAMES[action_idx],
                    pattern_hints=action_details.get("pattern_hints", []),
                    sentiment_score=sentiment_results.get(ticker, {}).score if sentiment_results else None,
                    vector_description=describe_vector(current_vector)
                ))

        # Run the analysis in a background thread
        background_tasks.add_task(run_batch_analysis)

        # Return immediate response with placeholder results (real data returned later via WebSocket or polling)
        return BatchAnalysisResponse(
            results=[],
            summary={"total_tickers": len(request.tickers), "status": "processing"}
        )

    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Batch processing error")

@app.post("/api/v1/backtest", response_model=BacktestResult)
async def backtest_strategy(
    request: StrategyBacktestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Simulate and evaluate a user-defined strategy against historical data.

    This endpoint is computationally intensive and runs in the background.
    Results are returned via WebSocket or stored and retrieved later.
    """
    try:
        # TODO: Implement backtesting logic using ModelTrainer
        def run_backtest():
            pass  # Placeholder for actual implementation

        background_tasks.add_task(run_backtest)
        return BacktestResult(
            strategy_name=request.strategy_name or "Default",
            win_rate=0.65,  # Placeholder value
            avg_return=0.12,
            drawdown=0.18
        )

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail="Backtesting error")

@app.get("/api/v1/history/{ticker}", response_model=List[TickerAnalysisResponse])
async def get_ticker_history(
    ticker: str,
    days_back: int = 365,
    db_manager: DBManager = Depends(get_db_manager),
):
    """
    Retrieve historical signal and outcome data for a given ticker.

    Used for research, performance tracking, or signal validation.
    """
    try:
        # TODO: Implement retrieval logic from DB
        return []
    except Exception as e:
        logger.error(f"History fetch failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="History retrieval error")

# Health Check Endpoint (Best Practice)
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API availability."""
    return {"status": "healthy", "version": "2.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
