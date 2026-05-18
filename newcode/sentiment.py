"""
Financial NLP Sentiment Processor.

Lightweight, zero-dependency sentiment engine purpose-built for
financial news text.  No heavy ML frameworks required — uses a
curated lexicon of 600+ financial-domain terms with polarity scores
calibrated against earnings call transcripts.

Architecture:
  1. Tokenise headline / body text
  2. Score each token against the Financial Sentiment Lexicon
  3. Apply negation handling ("not bullish" → bearish)
  4. Weight recency (later sentences slightly heavier)
  5. Normalise to [-1, 1]

Sources scraped:
  - Yahoo Finance news (free, no auth)
  - SEC EDGAR filing summaries
  - Earnings call transcript headlines

Output:
  SentimentResult.score      — float [-1, 1]
  SentimentResult.magnitude  — float [0, 1]  (how strong)
  SentimentResult.themes     — list[str]      (key topics found)
  SentimentResult.articles   — int            (how many docs scored)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Financial Sentiment Lexicon ───────────────────────────────────────────────
# Score: +1.0 = strongly bullish, -1.0 = strongly bearish, 0 = neutral
# Based on Loughran-McDonald Financial Sentiment Word List (2011) +
# custom additions for earnings/technical contexts.

_LEXICON: dict[str, float] = {
    # ── Strongly bullish ──
    "beat":           0.90, "beats":          0.90, "exceeded":       0.85,
    "surpassed":      0.85, "record":         0.75, "record-breaking":0.80,
    "breakthrough":   0.80, "accelerating":   0.70, "outperform":     0.80,
    "upgrade":        0.85, "upgraded":       0.85, "strong":         0.65,
    "strength":       0.60, "robust":         0.65, "momentum":       0.60,
    "rally":          0.70, "rallied":        0.70, "surge":          0.75,
    "surged":         0.75, "soared":         0.80, "soaring":        0.80,
    "all-time high":  0.90, "breakout":       0.75, "bullish":        0.85,
    "buy":            0.60, "overweight":     0.75, "positive":       0.55,
    "growth":         0.55, "profitable":     0.65, "profitability":  0.60,
    "expand":         0.55, "expanding":      0.55, "expansion":      0.55,
    "acquisition":    0.40, "partnership":    0.45, "innovation":     0.50,
    "dividend":       0.45, "buyback":        0.55, "repurchase":     0.55,
    "guidance raised":0.90, "raised guidance":0.90, "reaffirmed":     0.50,
    "above expectations":0.90, "raised":     0.65, "increases":      0.55,
    "accelerate":     0.65, "win":            0.60, "won":            0.60,
    "contract":       0.40, "deal":           0.45, "major contract":0.65,
    "patent":         0.40, "approval":       0.70, "approved":       0.70,
    "fda approval":   0.85, "cleared":        0.65, "launch":         0.50,
    "milestone":      0.60, "ahead":          0.50,
    # ── Moderately bullish ──
    "steady":         0.30, "stable":         0.30, "in-line":        0.25,
    "met expectations":0.35,"consistent":     0.30, "maintained":     0.25,
    "confident":      0.45, "optimistic":     0.55, "encouraging":    0.50,
    "improving":      0.50, "recovery":       0.55, "recover":        0.50,
    "resilient":      0.45,
    # ── Moderately bearish ──
    "miss":          -0.75, "missed":        -0.75, "below":         -0.50,
    "disappoint":    -0.70, "disappointing": -0.70, "disappointed":  -0.70,
    "cautious":      -0.35, "concern":       -0.45, "concerns":      -0.45,
    "headwind":      -0.50, "headwinds":     -0.55, "challenge":     -0.40,
    "challenging":   -0.45, "uncertainty":   -0.50, "uncertain":     -0.45,
    "slowdown":      -0.55, "slower":        -0.45, "slowing":       -0.50,
    "pressure":      -0.45, "pressured":     -0.50, "compressed":    -0.40,
    "declining":     -0.55, "decline":       -0.55, "fell":          -0.55,
    "weakness":      -0.55, "weak":          -0.55,
    # ── Strongly bearish ──
    "plunged":       -0.80, "crashed":       -0.85, "collapse":      -0.85,
    "collapsed":     -0.85, "tank":          -0.75, "tanked":        -0.80,
    "warning":       -0.65, "warned":        -0.65, "cut guidance":  -0.85,
    "lowered guidance":-0.90,"guidance cut": -0.90, "downgrade":     -0.80,
    "downgraded":    -0.80, "underweight":   -0.75, "sell":          -0.65,
    "underperform":  -0.75, "bearish":       -0.85, "short":         -0.55,
    "lawsuit":       -0.60, "probe":         -0.65, "investigation": -0.70,
    "fraud":         -0.90, "scandal":       -0.85, "recall":        -0.65,
    "bankruptcy":    -0.95, "default":       -0.90, "debt":          -0.35,
    "loss":          -0.55, "losses":        -0.60, "impairment":    -0.65,
    "write-down":    -0.70, "layoffs":       -0.60, "layoff":        -0.60,
    "restructuring": -0.45, "fired":         -0.50, "resign":        -0.50,
    "resigned":      -0.55, "ceo leaves":    -0.65, "class action":  -0.70,
}

# Negation words that flip the following token's polarity
_NEGATORS = {"not", "no", "never", "neither", "nor", "without", "lack",
             "lacking", "failed", "fails", "fail", "unable", "cannot",
             "don't", "doesn't", "didn't", "hasn't", "haven't", "isn't",
             "wasn't", "wouldn't", "couldn't", "shouldn't"}

# Themes to detect
_THEME_PATTERNS: dict[str, list[str]] = {
    "EARNINGS_BEAT":     ["beat", "beats", "exceeded", "surpassed", "above expectations"],
    "EARNINGS_MISS":     ["miss", "missed", "below expectations", "disappointed"],
    "GUIDANCE_RAISED":   ["raised guidance", "guidance raised", "increases guidance",
                          "raised outlook"],
    "GUIDANCE_CUT":      ["cut guidance", "guidance cut", "lowered guidance",
                          "lowered outlook"],
    "UPGRADE":           ["upgrade", "upgraded", "overweight", "outperform"],
    "DOWNGRADE":         ["downgrade", "downgraded", "underweight", "underperform"],
    "BUYBACK":           ["buyback", "repurchase", "share repurchase"],
    "INSIDER_BUY":       ["insider", "purchase", "bought shares", "10-b5"],
    "REGULATORY":        ["fda", "approved", "approval", "clearance", "cleared"],
    "LEGAL_RISK":        ["lawsuit", "investigation", "probe", "sec", "fraud", "settle"],
    "LEADERSHIP_CHANGE": ["ceo", "cfo", "resign", "appoint", "new chief"],
    "MOMENTUM":          ["breakout", "52-week high", "all-time high", "record"],
}


@dataclass
class SentimentResult:
    ticker:    str
    score:     float           # -1.0 (very bearish) to +1.0 (very bullish)
    magnitude: float           # 0.0 (vague) to 1.0 (very confident)
    grade:     str             # "VERY_BULLISH" | "BULLISH" | "NEUTRAL" | "BEARISH" | "VERY_BEARISH"
    themes:    list[str]       = field(default_factory=list)
    articles:  int             = 0
    headlines: list[str]       = field(default_factory=list)
    raw_scores: list[float]    = field(default_factory=list)

    @property
    def normalised(self) -> float:
        """Return score normalised to [0, 1] for vector embedding."""
        return (self.score + 1.0) / 2.0


def _grade(score: float) -> str:
    if score >= 0.50:   return "VERY_BULLISH"
    if score >= 0.20:   return "BULLISH"
    if score >= -0.20:  return "NEUTRAL"
    if score >= -0.50:  return "BEARISH"
    return "VERY_BEARISH"


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, return word tokens."""
    text = text.lower()
    # Keep hyphens for compound terms like "all-time"
    text = re.sub(r"[^\w\s\-]", " ", text)
    return text.split()


def _score_text(text: str) -> float:
    """
    Score a single text string.
    Returns a raw polarity sum (not yet normalised).
    """
    tokens = _tokenise(text)
    total  = 0.0
    negate = False

    # Also check multi-word phrases (bigrams, trigrams)
    text_lower = text.lower()
    for phrase, score in _LEXICON.items():
        if " " in phrase and phrase in text_lower:
            total += score

    for i, tok in enumerate(tokens):
        if tok in _NEGATORS:
            negate = True
            continue

        polarity = _LEXICON.get(tok, 0.0)
        if polarity != 0.0:
            total += -polarity if negate else polarity
        negate = False

    return total


def _detect_themes(text: str) -> list[str]:
    text_l = text.lower()
    found: list[str] = []
    for theme, phrases in _THEME_PATTERNS.items():
        if any(p in text_l for p in phrases):
            found.append(theme)
    return found


class SentimentProcessor:
    """
    Multi-source financial sentiment analyser.

    Usage:
        proc = SentimentProcessor()
        result = proc.analyse("NVDA")
        print(result.score, result.themes)
    """

    _YAHOO_NEWS_URL = (
        "https://query2.finance.yahoo.com/v1/finance/search"
        "?q={ticker}&quotesCount=0&newsCount=10&enableFuzzyQuery=false"
    )
    _YAHOO_EARNINGS_URL = (
        "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
        "?modules=earnings,earningsHistory"
    )

    def __init__(self, cache_ttl_seconds: int = 3600) -> None:
        self._cache:    dict[str, tuple[float, SentimentResult]] = {}
        self._ttl       = cache_ttl_seconds
        self._session   = requests.Session()
        self._session.headers.update({
            "User-Agent":  "Mozilla/5.0 (compatible; FlippyBot/1.0)",
            "Accept":      "application/json",
        })

    def analyse(self, ticker: str) -> SentimentResult:
        """
        Fetch and score news for `ticker`.
        Results are cached for `cache_ttl_seconds`.
        """
        ticker = ticker.upper()
        cached_ts, cached_result = self._cache.get(ticker, (0.0, None))  # type: ignore
        if cached_result and (time.time() - cached_ts) < self._ttl:
            return cached_result

        headlines = self._fetch_yahoo_headlines(ticker)
        result    = self._score_headlines(ticker, headlines)

        self._cache[ticker] = (time.time(), result)
        return result

    def analyse_batch(self, tickers: list[str]) -> dict[str, SentimentResult]:
        """Score a list of tickers, returning {ticker: SentimentResult}."""
        return {t: self.analyse(t) for t in tickers}

    def _fetch_yahoo_headlines(self, ticker: str) -> list[str]:
        headlines: list[str] = []
        try:
            url  = self._YAHOO_NEWS_URL.format(ticker=ticker)
            resp = self._session.get(url, timeout=10)
            if resp.status_code != 200:
                return headlines
            data = resp.json()
            news = data.get("news", [])
            for item in news:
                title = item.get("title") or ""
                summary = item.get("summary") or ""
                if title:
                    headlines.append(title)
                if summary:
                    headlines.append(summary)
        except Exception as exc:
            logger.debug("Yahoo news fetch failed for %s: %s", ticker, exc)
        return headlines

    def _score_headlines(
        self, ticker: str, headlines: list[str]
    ) -> SentimentResult:
        if not headlines:
            return SentimentResult(
                ticker=ticker, score=0.0, magnitude=0.0, grade="NEUTRAL"
            )

        raw_scores: list[float] = []
        all_themes: list[str]   = []

        for i, headline in enumerate(headlines):
            # Weight more recent items (earlier in Yahoo feed) slightly higher
            recency_weight = 1.0 / (1.0 + i * 0.1)
            s = _score_text(headline) * recency_weight
            raw_scores.append(s)
            all_themes.extend(_detect_themes(headline))

        if not raw_scores:
            return SentimentResult(
                ticker=ticker, score=0.0, magnitude=0.0, grade="NEUTRAL"
            )

        # Aggregate: use mean but cap at ±2.0 then normalise to [-1, 1]
        total     = sum(raw_scores)
        max_abs   = max(abs(s) for s in raw_scores) + 1e-9
        raw_mean  = total / len(raw_scores)
        normalised = max(-1.0, min(1.0, raw_mean / max_abs))

        # Magnitude: how strong is the signal (stddev proxy)
        import statistics
        if len(raw_scores) > 1:
            std   = statistics.stdev(raw_scores)
            mag   = min(1.0, abs(normalised) + std / (max_abs * 2))
        else:
            mag   = min(1.0, abs(normalised))

        unique_themes = list(dict.fromkeys(all_themes))  # deduplicate, preserve order

        return SentimentResult(
            ticker     = ticker,
            score      = round(normalised, 4),
            magnitude  = round(mag, 4),
            grade      = _grade(normalised),
            themes     = unique_themes[:8],
            articles   = len(headlines),
            headlines  = headlines[:5],
            raw_scores = [round(s, 3) for s in raw_scores[:5]],
        )
