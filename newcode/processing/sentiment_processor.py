from __future__ import annotations
from typing import List

from sentiment import (
    SentimentProcessor as _SentimentProcessor,
    SentimentResult as _SentimentResult,
)


class SentimentResult(_SentimentResult):
    """A typed bridge object for the processing layer."""


class SentimentProcessor:
    """Wrapper around the core sentiment module.

    Provides a stable interface for the application pipeline.
    """

    def __init__(self, cache_ttl_seconds: int = 3600) -> None:
        self._impl = _SentimentProcessor(cache_ttl_seconds=cache_ttl_seconds)

    def analyse(self, ticker: str) -> SentimentResult:
        """Score a single ticker and return normalized sentiment metadata."""
        result = self._impl.analyse(ticker)
        return SentimentResult(
            ticker=result.ticker,
            score=result.score,
            magnitude=result.magnitude,
            grade=result.grade,
            themes=result.themes,
            articles=result.articles,
            headlines=result.headlines,
            raw_scores=result.raw_scores,
        )

    def analyse_batch(self, tickers: List[str]) -> dict[str, SentimentResult]:
        """Score a batch of tickers, normalising ticker keys to upper case."""
        return {ticker.upper(): self.analyse(ticker) for ticker in tickers}
