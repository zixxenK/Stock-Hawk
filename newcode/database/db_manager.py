from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import logging

logger = logging.getLogger(__name__)
Base = declarative_base()

# --- Schemas ---
class DBPoliticalTrade(Base):
    __tablename__ = "political_trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), index=True, nullable=False)
    politician = Column(String(100))
    chamber = Column(String(20))
    transaction_type = Column(String(20))
    amount_midpoint = Column(Float)
    transaction_date = Column(String(20), index=True)

class DBInsiderTrade(Base):
    __tablename__ = "insider_trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), index=True, nullable=False)
    insider_name = Column(String(100))
    title = Column(String(50))
    total_value = Column(Float)
    transaction_date = Column(String(20), index=True)

# --- Manager ---
class DatabaseManager:
    """
    Handles connections and specific queries to the Flippy SQLite database.
    """
    def __init__(self, db_url: str = "sqlite:///market_intelligence.db"):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_congress_trades(self, trades: list):
        """Saves a list of CongressTrade Pydantic models to the DB."""
        session = self.Session()
        try:
            for t in trades:
                # Basic deduplication logic could be added here
                db_trade = DBPoliticalTrade(
                    ticker=t.ticker,
                    politician=t.politician,
                    chamber=t.chamber,
                    transaction_type=t.transaction_type,
                    amount_midpoint=t.amount_midpoint,
                    transaction_date=t.transaction_date
                )
                session.add(db_trade)
            session.commit()
            logger.info(f"Saved {len(trades)} congressional trades to database.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save congress trades: {e}")
        finally:
            session.close()

    def save_insider_trades(self, trades: list):
        """Saves a list of InsiderTrade Pydantic models to the DB."""
        session = self.Session()
        try:
            for t in trades:
                db_trade = DBInsiderTrade(
                    ticker=t.ticker,
                    insider_name=t.insider_name,
                    title=t.title,
                    total_value=t.total_value,
                    transaction_date=t.transaction_date
                )
                session.add(db_trade)
            session.commit()
            logger.info(f"Saved {len(trades)} corporate insider trades to database.")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save insider trades: {e}")
        finally:
            session.close()