import requests
from bs4 import BeautifulSoup
import logging
from typing import List
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class InsiderTrade(BaseModel):
    ticker: str
    insider_name: str
    title: str
    transaction_date: str
    trade_type: str
    shares_traded: float
    price_per_share: float
    total_value: float

class OpenInsiderScraper:
    """
    Scrapes OpenInsider.com for recent cluster purchases by corporate insiders.
    """
    def __init__(self):
        # URL configured to look for Purchases (P) filed in the last 7 days
        self.url = "http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=7&td=0&fdlyl=&fdlyh=&daysago=&xp=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc=&sortcol=0&cnt=100&page=1"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }

    def fetch_insider_buys(self) -> List[InsiderTrade]:
        logger.info("Scraping live insider purchases from OpenInsider...")
        trades = []
        try:
            response = requests.get(self.url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            table = soup.find('table', {'class': 'tinytable'})
            if not table:
                logger.warning("Could not find the data table on OpenInsider.")
                return trades

            rows = table.find('tbody').find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 16:
                    continue
                
                # Extract relevant fields based on OpenInsider's table structure
                trade = InsiderTrade(
                    ticker=cols[3].text.strip(),
                    insider_name=cols[4].text.strip(),
                    title=cols[5].text.strip(),
                    transaction_date=cols[1].text.strip()[:10], # Get YYYY-MM-DD
                    trade_type=cols[6].text.strip(),
                    shares_traded=float(cols[8].text.replace(',', '').replace('+', '') or 0),
                    price_per_share=float(cols[9].text.replace('$', '').replace(',', '') or 0),
                    total_value=float(cols[11].text.replace('$', '').replace(',', '') or 0)
                )
                # We only want actual purchases
                if trade.trade_type == "Purchase":
                    trades.append(trade)
                    
            return trades
            
        except Exception as e:
            logger.error(f"Failed to scrape OpenInsider: {e}")
            return []