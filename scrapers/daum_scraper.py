"""
Daum Finance Discussion Board Scraper (South Korea Exchange).
Collects retail discussions and sentiment from Daum Finance (다음 금융).
Integrates Apify cheerio actor with direct Daum JSON endpoint fallback.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import requests
from config.apify_actors import APIFY_ACTORS
from config.tickers import get_asset_metadata
from scrapers.apify_client import ApifyScraperClient
from data_ingestion.data_cleaner import DataCleaner

logger = logging.getLogger("hengseng.scraper.daum")

class DaumFinanceScraper:
    """Scrapes Daum Finance Discussions (finance.daum.net) for KRX stocks."""

    def __init__(self, apify_client: Optional[ApifyScraperClient] = None):
        self.client = apify_client or ApifyScraperClient()
        self.actor_config = APIFY_ACTORS.get("kr_daum", {})
        self.source_name = "Daum Finance"

    def fetch_posts(self, ticker: str, days_back: int = 10, max_posts: int = 50) -> List[Dict[str, Any]]:
        """
        Fetches retail investor posts from Daum Finance.
        """
        meta = get_asset_metadata(ticker)
        asset_name = meta["name"]
        exchange = meta["exchange"]
        symbol = meta.get("symbol", ticker.split(".")[0])

        logger.info(f"[{self.source_name}] Starting collection for {ticker} ({asset_name})")

        results: List[Dict[str, Any]] = []

        # 1. Apify actor run if authenticated
        if self.client.is_authenticated:
            actor_id = self.actor_config.get("actor_id", "apify/cheerio-scraper")
            run_input = {
                "startUrls": [{"url": f"https://finance.daum.net/quotes/A{symbol}#talk"}],
                "maxRequestsPerCrawl": 10
            }
            logger.info(f"[{self.source_name}] Invoking Apify actor {actor_id} for symbol {symbol}...")
            items = self.client.run_actor(actor_id=actor_id, run_input=run_input)
            if items:
                logger.info(f"[{self.source_name}] Retrieved {len(items)} items from Apify Daum actor.")
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

        # 2. Resilient Direct Endpoint Fallback
        logger.info(f"[{self.source_name}] Querying Daum Finance talk discussion API...")
        raw_items = self._fetch_daum_talk_direct(symbol=symbol, max_items=max_posts)
        if raw_items:
            results = DataCleaner.process_and_deduplicate_posts(
                raw_posts=raw_items,
                ticker=ticker,
                asset_name=asset_name,
                exchange=exchange,
                source=self.source_name
            )
            DataCleaner.save_raw_snapshot(raw_items, self.source_name, ticker)

        return results

    def _fetch_daum_talk_direct(self, symbol: str, max_items: int) -> List[Dict[str, Any]]:
        """
        Queries Daum Finance community talk endpoints.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": f"https://finance.daum.net/quotes/A{symbol}"
        }
        url = f"https://finance.daum.net/api/quotes/A{symbol}/talks?perPage={max_items}"
        collected: List[Dict[str, Any]] = []

        try:
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                talks = data.get("data", [])
                for talk in talks:
                    content = talk.get("content") or talk.get("title", "")
                    collected.append({
                        "id": f"daum_{talk.get('id', len(collected))}",
                        "title": talk.get("title", ""),
                        "text": content,
                        "createdAt": talk.get("createdAt") or datetime.now(timezone.utc).isoformat(),
                        "likes": talk.get("upCount", 0),
                        "reposts": talk.get("commentCount", 0),
                        "views": talk.get("viewCount", 0),
                        "author_followers": 0,
                        "url": f"https://finance.daum.net/quotes/A{symbol}#talk"
                    })
        except Exception as e:
            logger.debug(f"Direct Daum Finance API fetch skipped: {e}")

        return collected
