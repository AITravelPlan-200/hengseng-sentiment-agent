"""
Naver Finance Stock Discussion Board Scraper (South Korea Exchange).
Collects Korean retail investor posts from Naver Finance (네이버 금융 종목토론실).
Integrates Apify web/cheerio scraper with direct public board parser fallback.
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

logger = logging.getLogger("hengseng.scraper.naver")

class NaverFinanceScraper:
    """Scrapes Naver Finance Discussion Boards (네이버 금융 종목토론실) for KRX assets."""

    def __init__(self, apify_client: Optional[ApifyScraperClient] = None):
        self.client = apify_client or ApifyScraperClient()
        self.actor_config = APIFY_ACTORS.get("kr_naver", {})
        self.source_name = "Naver Finance"

    def fetch_posts(self, ticker: str, days_back: int = 10, max_posts: int = 50) -> List[Dict[str, Any]]:
        """
        Fetches retail investor posts from Naver Finance discussion board for the ticker.
        """
        meta = get_asset_metadata(ticker)
        asset_name = meta["name"]
        exchange = meta["exchange"]
        symbol = meta.get("symbol", ticker.split(".")[0])

        logger.info(f"[{self.source_name}] Starting collection for {ticker} ({asset_name}) - Symbol: {symbol}")

        results: List[Dict[str, Any]] = []

        # 1. Apify actor run if authenticated
        if self.client.is_authenticated:
            actor_id = self.actor_config.get("actor_id", "apify/web-scraper")
            run_input = {
                "startUrls": [{"url": f"https://finance.naver.com/item/board.naver?code={symbol}"}],
                "maxPages": 3
            }
            logger.info(f"[{self.source_name}] Invoking Apify actor {actor_id} for code {symbol}...")
            items = self.client.run_actor(actor_id=actor_id, run_input=run_input)
            if items:
                logger.info(f"[{self.source_name}] Retrieved {len(items)} items from Apify Naver actor.")
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

        # 2. Resilient Direct HTTP Fallback
        logger.info(f"[{self.source_name}] Executing direct board parser fallback for {symbol}...")
        raw_items = self._scrape_naver_board_direct(symbol=symbol, max_items=max_posts)
        if raw_items:
            logger.info(f"[{self.source_name}] Direct crawl collected {len(raw_items)} discussion posts.")
            results = DataCleaner.process_and_deduplicate_posts(
                raw_posts=raw_items,
                ticker=ticker,
                asset_name=asset_name,
                exchange=exchange,
                source=self.source_name
            )
            DataCleaner.save_raw_snapshot(raw_items, self.source_name, ticker)

        return results

    def _scrape_naver_board_direct(self, symbol: str, max_items: int) -> List[Dict[str, Any]]:
        """
        Scrapes Naver's public item discussion table.
        Naver Finance uses EUC-KR encoding for classic board HTML.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": f"https://finance.naver.com/item/board.naver?code={symbol}"
        }
        collected: List[Dict[str, Any]] = []

        # Scrape first 2 pages
        for page in range(1, 3):
            if len(collected) >= max_items:
                break
            url = f"https://finance.naver.com/item/board.naver?code={symbol}&page={page}"
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue

                # Naver board is typically encoded in EUC-KR / CP949
                try:
                    resp.encoding = "cp949"
                    html = resp.text
                except Exception:
                    html = resp.text

                soup = BeautifulSoup(html, "html.parser")
                table = soup.find("table", class_="type2")
                if not table:
                    continue

                rows = table.find_all("tr")
                for row in rows:
                    title_td = row.find("td", class_="title")
                    if not title_td or not title_td.find("a"):
                        continue

                    link = title_td.find("a")
                    title = link.get_text(strip=True)
                    href = link.get("href", "")
                    
                    # Extract post ID from href nid=XXXX
                    post_id = f"naver_{symbol}_{href.split('nid=')[-1].split('&')[0]}" if "nid=" in href else ""

                    # Date td
                    date_td = row.find("td", class_="p9")
                    date_str = date_td.get_text(strip=True) if date_td else datetime.now(timezone.utc).isoformat()

                    # Views td
                    tds = row.find_all("td")
                    views = 0
                    likes = 0
                    if len(tds) >= 6:
                        try:
                            views = int(tds[3].get_text(strip=True).replace(",", ""))
                            likes = int(tds[4].get_text(strip=True).replace(",", ""))
                        except Exception:
                            pass

                    collected.append({
                        "id": post_id or f"naver_{len(collected)}",
                        "title": title,
                        "text": title,
                        "createdAt": date_str,
                        "likes": likes,
                        "reposts": 0,
                        "views": views,
                        "author_followers": 0,
                        "url": f"https://finance.naver.com{href}"
                    })

                    if len(collected) >= max_items:
                        break

            except Exception as e:
                logger.debug(f"Direct Naver board parse error for page {page}: {e}")

        return collected
