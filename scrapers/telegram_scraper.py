"""
Telegram Social Media Scraper.
Collects retail discussions and sentiment from public financial Telegram channels
using Apify's Telegram Scraper actor with direct preview fallback.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup
from config.apify_actors import APIFY_ACTORS
from config.tickers import get_asset_metadata
from scrapers.apify_client import ApifyScraperClient
from data_ingestion.data_cleaner import DataCleaner

logger = logging.getLogger("hengseng.scraper.telegram")

class TelegramScraper:
    """Scrapes public Telegram trading channels for retail sentiment."""

    def __init__(self, apify_client: Optional[ApifyScraperClient] = None):
        self.client = apify_client or ApifyScraperClient()
        self.actor_config = APIFY_ACTORS.get("hk_telegram", {})
        self.source_name = "Telegram"

    def fetch_posts(self, ticker: str, days_back: int = 10, max_posts: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves Telegram channel messages mentioning the ticker or company keywords.
        """
        meta = get_asset_metadata(ticker)
        asset_name = meta["name"]
        exchange = meta["exchange"]
        search_terms = meta.get("social_search_queries", [ticker])

        logger.info(f"[{self.source_name}] Starting collection for {ticker} ({asset_name})")

        results: List[Dict[str, Any]] = []

        # 1. Apify actor execution if authenticated
        if self.client.is_authenticated:
            actor_id = self.actor_config.get("actor_id", "apify/telegram-scraper")
            channels = self.actor_config.get("target_channels", ["hkstocks_discussion"])
            run_input = {
                "channels": channels,
                "messagesLimit": max_posts,
                "query": search_terms[0] if search_terms else ticker
            }
            logger.info(f"[{self.source_name}] Invoking Apify actor {actor_id}...")
            items = self.client.run_actor(actor_id=actor_id, run_input=run_input)
            if items:
                logger.info(f"[{self.source_name}] Retrieved {len(items)} items from Apify Telegram actor.")
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

        # 2. Public Channel Web-Preview Scraper Fallback (t.me/s/<channel>)
        logger.info(f"[{self.source_name}] Checking public web previews for relevant messages...")
        public_channels = [
            "hkstocks_discussion",
            "hangsengindexdaily",
            "krx_market_brief"
        ] if exchange == "HKEX" else ["krx_market_brief", "koreastocktalk", "seoultraders"]

        web_items = self._scrape_public_previews(public_channels, search_terms, max_posts)
        if web_items:
            results = DataCleaner.process_and_deduplicate_posts(
                raw_posts=web_items,
                ticker=ticker,
                asset_name=asset_name,
                exchange=exchange,
                source=self.source_name
            )
            DataCleaner.save_raw_snapshot(web_items, self.source_name, ticker)

        return results

    def _scrape_public_previews(self, channels: List[str], search_terms: List[str], max_items: int) -> List[Dict[str, Any]]:
        """Scrape public t.me/s/ channel previews without requiring bot authentication."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        collected: List[Dict[str, Any]] = []
        lower_terms = [t.lower() for t in search_terms]

        for channel in channels:
            if len(collected) >= max_items:
                break
            url = f"https://t.me/s/{channel}"
            try:
                resp = requests.get(url, headers=headers, timeout=8)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    messages = soup.find_all("div", class_="tgme_widget_message")
                    for msg in messages:
                        text_div = msg.find("div", class_="tgme_widget_message_text")
                        if not text_div:
                            continue
                        text = text_div.get_text(separator=" ", strip=True)
                        text_lower = text.lower()
                        if any(term in text_lower for term in lower_terms):
                            date_tag = msg.find("time")
                            date_str = date_tag.get("datetime") if date_tag else None
                            views_tag = msg.find("span", class_="tgme_widget_message_views")
                            views_str = views_tag.text.strip() if views_tag else "0"
                            views = self._parse_views(views_str)

                            collected.append({
                                "id": msg.get("data-post", f"tg_{len(collected)}"),
                                "text": text,
                                "createdAt": date_str or datetime.now(timezone.utc).isoformat(),
                                "likes": 5,
                                "reposts": 1,
                                "views": views,
                                "author_followers": 1500,
                                "url": f"https://t.me/{msg.get('data-post', '')}"
                            })
            except Exception as e:
                logger.debug(f"Public telegram preview fetch skipped for {channel}: {e}")

        return collected

    @staticmethod
    def _parse_views(v_str: str) -> int:
        try:
            v_str = v_str.upper().strip()
            if "K" in v_str:
                return int(float(v_str.replace("K", "")) * 1000)
            if "M" in v_str:
                return int(float(v_str.replace("M", "")) * 1000000)
            return int(v_str)
        except Exception:
            return 0
