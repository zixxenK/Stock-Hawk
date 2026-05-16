from pydantic import BaseModel, Field
from typing import Literal
import os

class Settings(BaseModel):
    """Global application settings."""
    # Database connection string (using SQLite for local development)
    DATABASE_URL: str = "sqlite:///./data/flippy_store.db"
    # Placeholder API Keys and Secrets
    API_KEYS: dict[str, str] = Field(default_factory=lambda: {
        "YAHOO": os.getenv("FLIPPY_YAHOO_KEY", ""),
        "BROKERAGE": os.getenv("FLIPPY_BROKER_KEY", "")
    })
    MODEL_PATH: str = "./models/"
    LOGGING_LEVEL: Literal["DEBUG", "INFO", "WARNING"] = "INFO"

# Instantiate the settings object globally for easy access throughout the application
SETTINGS: Settings = Settings()
