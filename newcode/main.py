from typing import List
from database.db_manager import DBManager
from data_sources.scraper_manager import ScraperManager
from processing.vectorizer import build_vector
from intelligence.agent import FlippyAgent
from config.settings import SETTINGS
from api.fast_api_app import app, TickerAnalysisRequest, BatchTickerAnalysisRequest, StrategyBacktestRequest   


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
    # Run scrapers for the batch of tickers
    congress_trades = scraper_manager.fetch_all_data(tickers, days_back)

    # 3. Feature Vectorization (The Core Processing)
    print("Step 3/5: Vectorizing setups and generating signals...")
    for ticker in tickers:
        try:
            # A. Collect all relevant data points for the current setup
            raw_data = {
                "congress": congress_trades.get(ticker),
                "insider": db_manager.get_insider_trades(ticker, days_back)
            }

            # B. Build the 18D vector (The most complex data aggregation step)
            current_vector = build_vector(...) # Requires a dedicated function to aggregate all signals.

            # C. Get NLP Sentiment Context
            sentiment_result = agent.get_sentiment(ticker)
            # Update vector with sentiment score and themes.

            # 4. Scoring & Decision Making (The Alpha Score)
            print(f"  -> Analyzing {ticker}: Calculating Composite Alpha Score...")
            composite_score, action_details = agent.select_action(current_vector, ...)

            # 5. Output & Learning Loop
            if composite_score > 0:
                print(f"\n\t✅ Signal Found for {ticker}: Score={composite_score:.2f}")
                db_manager.upsert_signal_metadata(...) # Store the analysis result
                # Simulate trade and teach the agent (Self-correction loop)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
