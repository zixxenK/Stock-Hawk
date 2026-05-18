from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, LargeBinary
# Assuming SQLAlchemy ORM is used for persistence
# from sqlalchemy.ext.declarative import declarative_base
# Base = declarative_base()

class VectorInputModel(BaseModel):
    """Schema for the 18-dimensional input vector."""
    mom_12_1: Optional[float] = None
    rsi: Optional[float] = None
    insider_weighted: float = 0.0
    sentiment_score: Optional[float] = None
    congress_buys: int = 0

class TradeRecord(BaseModel):
    """Standardized record for any insider trade (Congress or SEC)."""
    ticker: str = Field(description="Stock Ticker Symbol")
    trade_type: str = Field(description="P/S/E (Purchase/Sale/Exchange)")
    shares: Optional[int] = None
    price_per_share: Optional[float] = None
    total_value: Optional[float] = None
    source: str = Field(description="Source: SEC, OpenInsider, QuiverQuant")

class AnalysisSignal(BaseModel):
    """The final output structure for a single ticker analysis."""
    ticker: str
    composite_alpha_score: float # The final ranked score
    action_recommendation: str   # e.g., BUY_MEDIUM
    history_win_rate: float     # Pattern memory input
    sentiment_score: float
    pattern_hints: list[str] = []

class SignalMetadata(BaseModel):
    """Stores the metadata used to calculate and explain a signal."""
    vector_fingerprint: str # Deterministic key for the state vector.
    timestamp: datetime
    source_data: dict # Raw data points that contributed to the signal.
    additional_info: Optional[dict] = None # Any extra metadata or notes