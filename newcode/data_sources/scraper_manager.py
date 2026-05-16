from typing import List
from database.db_manager import DBManager
# Import all necessary scrapers (CongressScraper, InsiderScraper)
# from . import CongressScraper, InsiderScraper

class ScraperManager:
    def __init__(self, db: 'DBManager'):
        self.db = db
        self.congress_scraper = self.congress_scraper # Initialized with DB
        self.insider_scraper = self.insider_scraper   # Initialized with DB

    def fetch_all_data(self, tickers: List[str], days_back: int) -> dict[str, list[dict]]:
        """
        Orchestrates fetching all data sources for the batch of tickers.
        Returns a dictionary grouped by Ticker -> [trade_records]
        """
        all_trades: dict[str, list[dict]] = {}

        # 1. Run Congress Scraper (Highest Priority)
        congress_data = self.congress_scraper.fetch_for_tickers(tickers, days_back=days_back)
        print("[SCRAPER] Processed Congress trades.")
        all_trades.update(congress_data)

        # 2. Run Insider Scraper (Secondary Source)
        insider_data = self.insider_scraper.fetch_for_tickers(tickers, days_back=days_back)
        print("[SCRAPER] Processed Insider trades.")
        # Merge/merge logic needed here to prevent duplicate entries per ticker.
        for ticker, trades in insider_data.items():
            if ticker in all_trades:
                all_trades[ticker].extend(trades)
            else:
                all_trades[ticker] = trades
        return all_trades

    def fetch_news_and_context(self, tickers: List[str], days_back: int) -> dict:
        """Fetches NLP context and other resources."""
        # Use the SentimentProcessor from processing/sentiment_processor.py
        from processing.sentiment_processor import SentimentProcessor
        nlp_processor = SentimentProcessor()
        return nlp_processor.analyse_batch(tickers)
