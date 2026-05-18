from datetime import datetime
from typing import Any, List

from database.db_manager import DBManager
from capitol_trades import CongressScraper
from insider_finance import InsiderScraper
from processing.sentiment_processor import SentimentProcessor, SentimentResult


class ScraperManager:
    def __init__(self, db: DBManager):
        self.db = db
        self.congress_scraper = CongressScraper(db=db)
        self.insider_scraper = InsiderScraper(db=db)
        self.sentiment_processor = SentimentProcessor()

    def fetch_all_data(self, tickers: List[str], days_back: int) -> dict[str, list[dict[str, Any]]]:
        """
        Orchestrates fetching all data sources for the batch of tickers.
        Ensures the ingestion step is executed and structured records are persisted.
        Returns a dictionary grouped by ticker -> list of trade records.
        """
        all_trades: dict[str, list[dict[str, Any]]] = {}

        try:
            self.congress_scraper.run(days_back=days_back)
            print("[SCRAPER] Persisted congress trades.")
        except Exception as exc:
            print(f"[SCRAPER] Congress scraper failed: {exc}")

        try:
            self.insider_scraper.run(days_back=days_back)
            print("[SCRAPER] Persisted insider trades.")
        except Exception as exc:
            print(f"[SCRAPER] Insider scraper failed: {exc}")

        try:
            congress_data = self.congress_scraper.fetch_for_tickers(tickers, days_back=days_back)
            print("[SCRAPER] Processed Congress trades.")
            for ticker, trades in congress_data.items():
                all_trades.setdefault(ticker.upper(), []).extend(trades)
        except Exception as exc:
            print(f"[SCRAPER] Congress fetch failed: {exc}")

        try:
            insider_data = self.insider_scraper.fetch_for_tickers(tickers, days_back=days_back)
            print("[SCRAPER] Processed Insider trades.")
            for ticker, trades in insider_data.items():
                all_trades.setdefault(ticker.upper(), []).extend(trades)
        except Exception as exc:
            print(f"[SCRAPER] Insider fetch failed: {exc}")

        return all_trades

    def fetch_news_and_context(self, tickers: List[str]) -> dict[str, Any]:
        """Fetches NLP sentiment and context for a batch of tickers."""
        try:
            results = self.sentiment_processor.analyse_batch(tickers)
        except Exception as exc:
            print(f"[SENTIMENT] Batch sentiment failed: {exc}")
            results = {}

        if self.db is not None:
            for ticker in tickers:
                key = ticker.upper()
                result = results.get(key)
                if result is None:
                    persisted = self.db.get_news_sentiment(ticker, days_back=7)
                    if persisted:
                        latest = persisted[0]
                        fallback = SentimentResult()
                        fallback.ticker = key
                        fallback.score = latest.get("score", 0.0)
                        fallback.magnitude = latest.get("magnitude", 0.0)
                        fallback.grade = latest.get("grade", "NEUTRAL")
                        fallback.headlines = [latest.get("headline")] if latest.get("headline") else []
                        fallback.themes = []
                        fallback.articles = 0
                        fallback.raw_scores = []
                        results[key] = fallback
                else:
                    try:
                        self.db.save_news_sentiment(
                            ticker=key,
                            score=result.score,
                            magnitude=result.magnitude,
                            grade=result.grade,
                            headline=(result.headlines[0] if result.headlines else None),
                            source="yahoo_news",
                            trade_date=datetime.utcnow().strftime("%Y-%m-%d"),
                        )
                    except Exception:
                        continue

        return results
