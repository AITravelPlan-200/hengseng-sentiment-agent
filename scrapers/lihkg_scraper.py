"""
LIHKG Social Media Scraper (Hong Kong Exchange).
Integrates Apify actor for LIHKG with resilient fallback mechanisms.
Extracts Hong Kong retail investor sentiment discussions on target HKEX stocks.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import requests
from config.apify_actors import APIFY_ACTORS
from config.tickers import get_asset_metadata
from scrapers.apify_client import ApifyScraperClient
from data_ingestion.data_cleaner import DataCleaner

logger = logging.getLogger("hengseng.scraper.lihkg")

class LIHKGScraper:
    """Scrapes LIHKG Finance forum discussions for HKEX stocks."""

    def __init__(self, apify_client: Optional[ApifyScraperClient] = None):
        self.client = apify_client or ApifyScraperClient()
        self.actor_config = APIFY_ACTORS.get("hk_lihkg", {})
        self.source_name = "LIHKG"

    def fetch_posts(self, ticker: str, days_back: int = 10, max_posts: int = 50) -> List[Dict[str, Any]]:
        """
        Fetches LIHKG posts discussing the specified HKEX stock.
        Uses Apify actor if available; otherwise falls back to direct HTTP or cached crawl.
        """
        meta = get_asset_metadata(ticker)
        asset_name = meta["name"]
        exchange = meta["exchange"]
        search_terms = meta.get("social_search_queries", [ticker])

        logger.info(f"[{self.source_name}] Starting collection for {ticker} ({asset_name}) - Query terms: {search_terms}")

        results: List[Dict[str, Any]] = []

        # 1. Attempt Apify actor collection if authenticated
        if self.client.is_authenticated:
            actor_id = self.actor_config.get("actor_id", "curious_coder/lihkg-scraper")
            run_input = {
                "searchKeywords": search_terms,
                "maxPosts": max_posts,
                "daysBack": days_back,
                "category": "finance"
            }
            logger.info(f"[{self.source_name}] Invoking Apify actor {actor_id}...")
            items = self.client.run_actor(actor_id=actor_id, run_input=run_input)
            if items:
                logger.info(f"[{self.source_name}] Successfully retrieved {len(items)} items via Apify.")
                results = DataCleaner.process_and_deduplicate_posts(
                    raw_posts=items,
                    ticker=ticker,
                    asset_name=asset_name,
                    exchange=exchange,
                    source=self.source_name
                )
                if results:
                    DataCleaner.save_raw_snapshot(items, self.source_name, ticker)
                    return results

        # 2. Resilient Direct HTTP Fallback to LIHKG Finance Board API
        logger.info(f"[{self.source_name}] Attempting direct public board query fallback...")
        direct_items = self._fetch_direct_public(ticker, search_terms, max_posts)
        if direct_items:
            logger.info(f"[{self.source_name}] Retrieved {len(direct_items)} items from LIHKG direct API.")
            results = DataCleaner.process_and_deduplicate_posts(
                raw_posts=direct_items,
                ticker=ticker,
                asset_name=asset_name,
                exchange=exchange,
                source=self.source_name
            )
            DataCleaner.save_raw_snapshot(direct_items, self.source_name, ticker)

        return results

    def _fetch_direct_public(self, ticker: str, search_terms: List[str], max_items: int) -> List[Dict[str, Any]]:
        """
        Direct HTTP request to LIHKG public category threads.
        Category 16 is Finance / Stock market on LIHKG.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Referer": "https://lihkg.com/category/16"
        }
        url = "https://lihkg.com/api_v2/thread/category?cat_id=16&page=1&count=60&type=now"
        posts: List[Dict[str, Any]] = []

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                threads = data.get("response", {}).get("items", [])
                
                # Filter threads containing any search term
                lower_terms = [t.lower() for t in search_terms]
                for th in threads:
                    title = th.get("title", "")
                    title_lower = title.lower()
                    if any(st in title_lower for st in lower_terms):
                        posts.append({
                            "id": f"lihkg_{th.get('thread_id')}",
                            "title": title,
                            "text": f"{title} - {th.get('remark', '')}",
                            "createdAt": th.get("create_time"),
                            "likes": th.get("like_count", 0),
                            "reposts": th.get("reply_count", 0),
                            "views": th.get("reply_count", 0) * 15,  # Estimated view multiplier
                            "author_followers": 0,
                            "url": f"https://lihkg.com/thread/{th.get('thread_id')}/page/1"
                        })
                        if len(posts) >= max_items:
                            break
        except Exception as e:
            logger.debug(f"Direct LIHKG API fetch skipped: {e}")

        return posts
