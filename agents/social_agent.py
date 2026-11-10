"""
Agent 1: Social Intelligence Agent.
Coordinates multi-source retail social media ingestion and sentiment extraction
for Hong Kong (LIHKG, Telegram) and South Korea (Naver Finance, Daum Finance).
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from config.tickers import ASSET_UNIVERSE, get_asset_metadata
from scrapers.apify_client import ApifyScraperClient
from scrapers.lihkg_scraper import LIHKGScraper
from scrapers.telegram_scraper import TelegramScraper
from scrapers.naver_scraper import NaverFinanceScraper
from scrapers.daum_scraper import DaumFinanceScraper
from sentiment.sentiment_engine import SentimentEngine
from database.db import db

logger = logging.getLogger("hengseng.agents.social")

class SocialIntelligenceAgent:
    """Agent responsible for retail social sentiment discovery, extraction, and normalization."""

    def __init__(self, apify_client: Optional[ApifyScraperClient] = None):
        self.apify_client = apify_client or ApifyScraperClient()
        self.sentiment_engine = SentimentEngine()
        
        # Instantiate dedicated scrapers
        self.lihkg_scraper = LIHKGScraper(self.apify_client)
        self.telegram_scraper = TelegramScraper(self.apify_client)
        self.naver_scraper = NaverFinanceScraper(self.apify_client)
        self.daum_scraper = DaumFinanceScraper(self.apify_client)

    def run_for_ticker(self, ticker: str, days_back: int = 10, max_posts_per_source: int = 25) -> List[Dict[str, Any]]:
        """
        Executes social scraping pipeline for a target ticker across 2 exchange sources.
        Enriches posts with sentiment scores and persists to SQLite.
        """
        meta = get_asset_metadata(ticker)
        exchange = meta["exchange"]
        collected_posts = []

        logger.info(f"[SocialAgent] Running collection for {ticker} ({exchange})")

        # Select sources based on exchange
        if exchange == "HKEX":
            # Source 1: LIHKG
            try:
                lihkg_items = self.lihkg_scraper.fetch_posts(ticker, days_back=days_back, max_posts=max_posts_per_source)
                collected_posts.extend(lihkg_items)
            except Exception as e:
                logger.error(f"[SocialAgent] LIHKG collection error for {ticker}: {e}")

            # Source 2: Telegram
            try:
                tg_items = self.telegram_scraper.fetch_posts(ticker, days_back=days_back, max_posts=max_posts_per_source)
                collected_posts.extend(tg_items)
            except Exception as e:
                logger.error(f"[SocialAgent] Telegram collection error for {ticker}: {e}")

        elif exchange == "KRX":
            # Source 1: Naver Finance
            try:
                naver_items = self.naver_scraper.fetch_posts(ticker, days_back=days_back, max_posts=max_posts_per_source)
                collected_posts.extend(naver_items)
            except Exception as e:
                logger.error(f"[SocialAgent] Naver Finance collection error for {ticker}: {e}")

            # Source 2: Daum Finance
            try:
                daum_items = self.daum_scraper.fetch_posts(ticker, days_back=days_back, max_posts=max_posts_per_source)
                collected_posts.extend(daum_items)
            except Exception as e:
                logger.error(f"[SocialAgent] Daum Finance collection error for {ticker}: {e}")

        # If live scraping returned zero posts (offline/no token), load from database
        if not collected_posts:
            cached_posts = db.get_social_posts(ticker=ticker, limit=100)
            if not cached_posts.empty:
                logger.info(f"[SocialAgent] Using {len(cached_posts)} cached posts for {ticker}.")
                return cached_posts.to_dict(orient="records")

        # Enrich with sentiment
        if collected_posts:
            enriched = self.sentiment_engine.enrich_social_posts(collected_posts)
            db.save_social_posts(enriched)
            logger.info(f"[SocialAgent] Saved {len(enriched)} enriched posts for {ticker}.")
            return enriched

        return []

    def run_for_universe(self, days_back: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Runs social intelligence collection across all 10 assets in universe."""
        results = {}
        for ticker in ASSET_UNIVERSE:
            posts = self.run_for_ticker(ticker, days_back=days_back)
            results[ticker] = posts
        return results
