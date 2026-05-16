import requests
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Pydantic model for strict data validation
class CongressTrade(BaseModel):
    ticker: str
    politician: str
    transaction_date: str
    transaction_type: str # "Purchase" or "Sale"
    amount_midpoint: float
    party: Optional[str] = None
    chamber: Optional[str] = None

class QuiverQuantCongressScraper:
    """
    Ingests live Congressional trades using the QuiverQuant API.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.quiverquant.com/beta/live/congresstrading"
        self.headers = {
            "accept": "application/json",
            "X-CSRFToken": "TyTJwjuEC7VV7mOqZ622haRaaUr0x0Ng4nrwSRFKQs7vdoBcJlK9qjAS69ghzhFu",
            "Authorization": f"Token {self.api_key}"
        }

    def _calculate_midpoint(self, amount_range: str) -> float:
        """Converts ranges like '$1,001 - $15,000' to a numeric midpoint."""
        try:
            # Strip symbols and split
            clean_str = amount_range.replace('$', '').replace(',', '')
            if '-' in clean_str:
                low, high = clean_str.split('-')
                return (float(low.strip()) + float(high.strip())) / 2.0
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to parse amount range: {amount_range}. Error: {e}")
            return 0.0

    def fetch_recent_trades(self) -> List[CongressTrade]:
        """Fetches the latest congressional trades and normalizes the data."""
        logger.info("Fetching live congressional trades from QuiverQuant...")
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            raw_data = response.json()
            
            normalized_trades = []
            for trade in raw_data:
                # Filter out trades without a valid ticker
                if not trade.get("Ticker"):
                    continue
                    
                midpoint = self._calculate_midpoint(trade.get("Amount", "0-0"))
                
                normalized_trades.append(
                    CongressTrade(
                        ticker=trade.get("Ticker"),
                        politician=trade.get("Representative", "Unknown"),
                        transaction_date=trade.get("TransactionDate", datetime.now().strftime("%Y-%m-%d")),
                        transaction_type=trade.get("Transaction", "Unknown").lower(),
                        amount_midpoint=midpoint,
                        party=trade.get("Party"),
                        chamber=trade.get("House") # e.g., "Senate" or "House"
                    )
                )
            return normalized_trades
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return []