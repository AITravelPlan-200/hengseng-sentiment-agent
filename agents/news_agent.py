"""
Agent 2: News Intelligence Agent.
Coordinates financial news retrieval via Exa API (and Tavily fallback),
executes financial sentiment analysis, and computes recency-weighted news signals.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from config.tickers import ASSET_UNIVERSE, get_asset_metadata
from config.settings import NEWS_LOOKBACK_DAYS
from data_ingestion.exa_client import ExaNewsClient
from sentiment.sentiment_engine import SentimentEngine
from database.db import db

logger = logging.getLogger("hengseng.agents.news")

class NewsIntelligenceAgent:
    """Agent responsible for financial news ingestion and sentiment scoring."""

    def __init__(self, exa_client: Optional[ExaNewsClient] = None):
        self.news_client = exa_client or ExaNewsClient()
        self.sentiment_engine = SentimentEngine()

    def run_for_ticker(self, ticker: str, days_back: int = NEWS_LOOKBACK_DAYS, max_articles: int = 15) -> List[Dict[str, Any]]:
        """
        Retrieves news articles for ticker, runs sentiment analysis,
        and saves enriched records to SQLite database.
        """
        logger.info(f"[NewsAgent] Ingesting news for {ticker}")
        
        # 1. Attempt live news ingestion
        raw_articles = self.news_client.fetch_company_news(ticker, days_back=days_back, max_articles=max_articles)

        # 2. Check local database cache if live ingestion produced zero
        if not raw_articles:
            cached_news = db.get_news(ticker=ticker, limit=50)
            if not cached_news.empty:
                logger.info(f"[NewsAgent] Using {len(cached_news)} cached news articles for {ticker}.")
                return cached_news.to_dict(orient="records")

        # 3. Enrich articles with sentiment probabilities and recency weights
        if raw_articles:
            enriched = self.sentiment_engine.enrich_news_articles(raw_articles)
            db.save_news_articles(enriched)
            logger.info(f"[NewsAgent] Saved {len(enriched)} enriched articles for {ticker}.")
            return enriched

        return []

    def run_for_universe(self, days_back: int = NEWS_LOOKBACK_DAYS) -> Dict[str, List[Dict[str, Any]]]:
        """Runs news intelligence collection across all assets."""
        results = {}
        for ticker in ASSET_UNIVERSE:
            articles = self.run_for_ticker(ticker, days_back=days_back)
            results[ticker] = articles
        return results
