"""
Vector similarity engine.

Every trade event and momentum snapshot is embedded as a fixed-length
feature vector.  Cosine similarity is used to find historical setups
that most closely resemble a new opportunity.  The matched historical
events include outcome labels (forward returns), so the agent can ask:
"what happened the last N times I saw a setup like this?"

Vector schema  (17 dimensions, all normalised 0-1 unless noted):
  [0]  mom_12_1         normalised to [-1, 1] → [0, 1]
  [1]  mom_6m           normalised
  [2]  mom_3m           normalised
  [3]  rs_vs_spy        normalised
  [4]  high_52w_pct     already 0-1
  [5]  adx              / 50 → 0-1
  [6]  rsi              / 100 → 0-1
  [7]  vol_trend        clipped [0.5, 2.0] → [0, 1]
  [8]  golden_cross     0 or 1
  [9]  hist_vol         clipped [0, 0.8] → [0, 1]
  [10] congress_buys    log-normalised (0-1)
  [11] insider_buys     log-normalised
  [12] options_bullish  log-normalised
  [13] suspicious_vol   0 or 1
  [14] sector_idx       one-hot ordinal (sector / 11)
  [15] earnings_prox    days to earnings / 30, clipped [0, 1]
  [16] market_regime    0=bear 0.5=mixed 1=bull
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

VECTOR_DIM = 17

# Map GICS sector name → ordinal index (0-10)
SECTOR_MAP: dict[str, int] = {
    "information technology": 0,
    "technology":             0,
    "health care":            1,
    "healthcare":             1,
    "financials":             2,
    "consumer discretionary": 3,
    "communication services": 4,
    "industrials":            5,
    "consumer staples":       6,
    "energy":                 7,
    "utilities":              8,
    "real estate":            9,
    "materials":              10,
}


def _safe(v: Any, fallback: float = 0.5) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return fallback
    return float(v)


def _norm_momentum(v: float | None) -> float:
    """Map return (e.g. -0.5 to +1.5) into [0, 1]."""
    if v is None:
        return 0.5
    return float(np.clip((v + 0.5) / 2.0, 0.0, 1.0))


def _log_norm(v: int | float | None, scale: float = 5.0) -> float:
    """Log-normalise a count into [0, 1].  scale=5 → 5 events ≈ 0.70."""
    if not v:
        return 0.0
    return float(np.clip(math.log1p(v) / math.log1p(scale), 0.0, 1.0))


def build_vector(
    *,
    mom_12_1: float | None = None,
    mom_6m: float | None = None,
    mom_3m: float | None = None,
    rs_vs_spy: float | None = None,
    high_52w_pct: float | None = None,
    adx: float | None = None,
    rsi: float | None = None,
    vol_trend: float | None = None,
    golden_cross: bool | None = None,
    hist_vol: float | None = None,
    congress_buys: int = 0,
    insider_buys: int = 0,
    options_bullish: int = 0,
    suspicious_vol: bool = False,
    sector: str = "",
    days_to_earnings: int | None = None,
    market_regime: str = "mixed",
) -> np.ndarray:
    """
    Construct a normalised VECTOR_DIM-length feature vector.
    All inputs are optional; missing values default to neutral (0.5).
    """
    sector_idx = SECTOR_MAP.get(sector.lower().strip(), 5) / 10.0

    if days_to_earnings is None:
        earn_prox = 0.5   # unknown → neutral
    else:
        earn_prox = float(np.clip(days_to_earnings / 30.0, 0.0, 1.0))

    regime_map = {"bull": 1.0, "buying": 1.0,
                  "mixed": 0.5, "unknown": 0.5,
                  "bear": 0.0, "profit_taking": 0.3}
    regime_val = regime_map.get(market_regime.lower(), 0.5)

    v = np.array([
        _norm_momentum(mom_12_1),
        _norm_momentum(mom_6m),
        _norm_momentum(mom_3m),
        _norm_momentum(rs_vs_spy),
        float(np.clip(_safe(high_52w_pct, 0.9), 0.0, 1.0)),
        float(np.clip(_safe(adx, 25.0) / 50.0, 0.0, 1.0)),
        float(_safe(rsi, 50.0) / 100.0),
        float(np.clip((_safe(vol_trend, 1.0) - 0.5) / 1.5, 0.0, 1.0)),
        1.0 if golden_cross else 0.0,
        float(np.clip(_safe(hist_vol, 0.3) / 0.8, 0.0, 1.0)),
        _log_norm(congress_buys),
        _log_norm(insider_buys),
        _log_norm(options_bullish),
        1.0 if suspicious_vol else 0.0,
        sector_idx,
        earn_prox,
        regime_val,
    ], dtype=np.float32)

    assert v.shape[0] == VECTOR_DIM, f"Expected {VECTOR_DIM} dims, got {v.shape[0]}"
    return v


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0-1."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def dot_product_score(a: np.ndarray, b: np.ndarray) -> float:
    """Raw dot product (unnormalised strength × alignment)."""
    return float(np.dot(a, b))


class VectorIndex:
    """
    In-memory nearest-neighbour index for historical event vectors.

    Supports:
      add(key, vector, metadata)      — insert a record
      query(vector, top_k)            — return top-k similar records
      wedge_score(a, b)               — cross-source signal alignment
    """

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._matrix: np.ndarray | None = None   # (N, VECTOR_DIM)
        self._meta: list[dict] = []

    def add(self, key: str, vector: np.ndarray, metadata: dict) -> None:
        self._keys.append(key)
        self._meta.append(metadata)
        v = vector.reshape(1, -1).astype(np.float32)
        if self._matrix is None:
            self._matrix = v
        else:
            self._matrix = np.vstack([self._matrix, v])

    def __len__(self) -> int:
        return len(self._keys)

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.70,
    ) -> list[dict]:
        """
        Return the top_k most similar historical records.

        Each result dict contains:
          key, similarity, dot_score, metadata (including forward returns).
        """
        if self._matrix is None or len(self._keys) == 0:
            return []

        q = vector.reshape(1, -1).astype(np.float32)

        # Vectorised cosine similarity across all stored vectors
        norms_db = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norm_q = np.linalg.norm(q)

        if norm_q == 0:
            return []

        with np.errstate(divide="ignore", invalid="ignore"):
            sims = (self._matrix @ q.T).flatten() / (norms_db.flatten() * norm_q + 1e-9)

        dots = (self._matrix @ q.T).flatten()

        # Sort descending by similarity
        order = np.argsort(sims)[::-1]

        results = []
        for idx in order[:top_k]:
            sim = float(sims[idx])
            if sim < min_similarity:
                break
            results.append({
                "key":        self._keys[idx],
                "similarity": sim,
                "dot_score":  float(dots[idx]),
                "metadata":   self._meta[idx],
            })
        return results

    def wedge_score(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        label_a: str = "momentum",
        label_b: str = "insider",
    ) -> dict:
        """
        'Wedge' product score: measures how well two signal vectors
        are aligned (dot product) vs. diverging (cross-magnitude).

        High wedge → both signals point the same direction strongly.
        Used to flag 'suspicious coordination' between e.g. congress
        buys and unusual options flow on the same ticker.

        Returns:
          alignment   — cosine similarity (0-1)
          magnitude   — geometric mean of both norms (raw strength)
          wedge_score — alignment × magnitude (0-∞, higher = stronger)
          verdict     — 'STRONG_ALIGNMENT' | 'MODERATE' | 'WEAK' | 'DIVERGING'
        """
        cos = cosine_similarity(vec_a, vec_b)
        mag = math.sqrt(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        wedge = cos * mag

        if cos > 0.85:
            verdict = "STRONG_ALIGNMENT"
        elif cos > 0.65:
            verdict = "MODERATE"
        elif cos > 0.40:
            verdict = "WEAK"
        else:
            verdict = "DIVERGING"

        return {
            "label_a":    label_a,
            "label_b":    label_b,
            "alignment":  round(cos, 4),
            "magnitude":  round(mag, 4),
            "wedge_score": round(wedge, 4),
            "verdict":    verdict,
        }

    def load_from_db_rows(self, rows: list[dict], key_field: str = "ticker") -> int:
        """
        Bulk-load historical snapshots from the database into the index.
        Each row must have a 'feature_vector' list and optional outcome fields.
        Returns number of rows loaded.
        """
        loaded = 0
        for row in rows:
            fv = row.get("feature_vector")
            if not fv or not isinstance(fv, list):
                continue
            vec = np.array(fv, dtype=np.float32)
            if vec.shape[0] != VECTOR_DIM:
                continue
            key = f"{row.get(key_field, '?')}_{row.get('snapshot_date', '')}"
            self.add(key, vec, {
                "ticker":       row.get("ticker"),
                "date":         row.get("snapshot_date") or row.get("signal_date"),
                "score":        row.get("score") or row.get("signal_score"),
                "ret_1w":       row.get("ret_1w"),
                "ret_2w":       row.get("ret_2w"),
                "ret_1m":       row.get("ret_1m"),
                "ret_3m":       row.get("ret_3m"),
                "congress_buys": row.get("congress_buys", 0),
                "insider_buys":  row.get("insider_buys", 0),
            })
            loaded += 1
        return loaded

    def expected_return_from_similar(
        self,
        vector: np.ndarray,
        top_k: int = 20,
        horizon: str = "ret_1m",
    ) -> dict:
        """
        Query the index, pull forward returns from similar historical setups,
        and return summary statistics.

        horizon: 'ret_1w' | 'ret_2w' | 'ret_1m' | 'ret_3m'

        Returns:
          mean_ret, median_ret, win_rate, n_samples, similar_setups (list)
        """
        similar = self.query(vector, top_k=top_k, min_similarity=0.65)

        returns = [
            s["metadata"].get(horizon)
            for s in similar
            if s["metadata"].get(horizon) is not None
        ]

        if not returns:
            return {
                "mean_ret":    None,
                "median_ret":  None,
                "win_rate":    None,
                "n_samples":   0,
                "similar_setups": similar,
            }

        arr = np.array(returns, dtype=np.float64)
        return {
            "mean_ret":    float(arr.mean()),
            "median_ret":  float(np.median(arr)),
            "win_rate":    float((arr > 0).mean()),
            "n_samples":   len(arr),
            "similar_setups": similar,
        }
    