"""
Financial News Ingestion Engine via Exa API and Tavily API.
Fetches high-signal corporate news (earnings, M&A, regulatory, product developments)
for target assets across HKEX and KRX exchanges.
Supports both official SDKs and high-resilience direct REST API execution.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import requests

from config.settings import EXA_API_KEY, TAVILY_API_KEY, NEWS_LOOKBACK_DAYS, REQUEST_TIMEOUT_SECONDS
from config.tickers import get_asset_metadata
from data_ingestion.data_cleaner import DataCleaner

logger = logging.getLogger("hengseng.news.exa")

EXA_SEARCH_ENDPOINT = "https://api.exa.ai/search"
TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"

class ExaNewsClient:
    """Client for retrieving financial news using Exa API, with Tavily fallback."""

    def __init__(self, exa_api_key: Optional[str] = None, tavily_api_key: Optional[str] = None):
        self.exa_key = exa_api_key or EXA_API_KEY
        self.tavily_key = tavily_api_key or TAVILY_API_KEY
        self._exa_client = None
        self._tavily_client = None
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize Exa and Tavily clients if credentials and packages are present."""
        if self.exa_key and self.exa_key.strip() and "your_" not in self.exa_key:
            try:
                from exa_py import Exa
                self._exa_client = Exa(api_key=self.exa_key)
                logger.info("Exa SDK client initialized.")
            except ImportError:
                logger.info("Exa SDK not installed; using direct Exa REST API endpoint.")
        else:
            logger.info("No EXA_API_KEY configured. Exa live news queries will be bypassed.")

        if self.tavily_key and self.tavily_key.strip() and "your_" not in self.tavily_key:
            try:
                from tavily import TavilyClient
                self._tavily_client = TavilyClient(api_key=self.tavily_key)
                logger.info("Tavily SDK client initialized.")
            except ImportError:
                logger.info("Tavily SDK not installed; using direct Tavily REST API endpoint.")

    @property
    def has_live_provider(self) -> bool:
        """Return True if at least one live news API provider is configured with credentials."""
        has_exa = bool(self.exa_key and self.exa_key.strip() and "your_" not in self.exa_key)
        has_tavily = bool(self.tavily_key and self.tavily_key.strip() and "your_" not in self.tavily_key)
        return has_exa or has_tavily

    def fetch_company_news(self, ticker: str, days_back: int = NEWS_LOOKBACK_DAYS, max_articles: int = 15) -> List[Dict[str, Any]]:
        """
        Retrieves company news for a ticker across financial categories.
        Prioritizes Exa API, falls back to Tavily search, then cached snapshots.
        """
        meta = get_asset_metadata(ticker)
        asset_name = meta["name"]
        exchange = meta["exchange"]
        queries = meta.get("news_search_queries", [f"{asset_name} {ticker} financial news"])

        logger.info(f"[News] Ingesting news for {ticker} ({asset_name})")

        # 1. Attempt Exa API (SDK or direct REST)
        if self.exa_key and self.exa_key.strip() and "your_" not in self.exa_key:
            articles = self._fetch_via_exa(ticker, asset_name, exchange, queries[0], days_back, max_articles)
            if articles:
                logger.info(f"[News] Exa returned {len(articles)} articles for {ticker}.")
                return articles

        # 2. Attempt Tavily API fallback (SDK or direct REST)
        if self.tavily_key and self.tavily_key.strip() and "your_" not in self.tavily_key:
            articles = self._fetch_via_tavily(ticker, asset_name, exchange, queries[0], max_articles)
            if articles:
                logger.info(f"[News] Tavily returned {len(articles)} articles for {ticker}.")
                return articles

        logger.info(f"[News] No live news retrieved for {ticker}. Check API keys or load cached snapshot.")
        return []

    def _fetch_via_exa(
        self,
        ticker: str,
        asset_name: str,
        exchange: str,
        query: str,
        days_back: int,
        max_articles: int
    ) -> List[Dict[str, Any]]:
        """Query Exa neural search via SDK or direct REST API."""
        articles = []
        start_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # SDK route
        if self._exa_client:
            try:
                response = self._exa_client.search_and_contents(
                    query=query,
                    type="neural",
                    start_published_date=start_date,
                    num_results=max_articles,
                    text={"max_characters": 1200}
                )
                seen_urls = set()
                for res in getattr(response, "results", []):
                    url = getattr(res, "url", "")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    articles.append({
                        "ticker": ticker,
                        "exchange": exchange,
                        "title": DataCleaner.clean_text(getattr(res, "title", f"News on {asset_name}")),
                        "url": url,
                        "published_utc": DataCleaner.normalize_timestamp(getattr(res, "published_date", None)),
                        "source": str(getattr(res, "author", "Exa Financial News")),
                        "summary": DataCleaner.clean_text(getattr(res, "text", "")[:400]),
                        "content": DataCleaner.clean_text(getattr(res, "text", "")),
                        "provider": "Exa API (SDK)"
                    })
                return articles
            except Exception as e:
                logger.warning(f"Exa SDK execution failed: {e}. Falling back to Exa REST endpoint...")

        # Direct REST route
        try:
            headers = {
                "x-api-key": self.exa_key,
                "Content-Type": "application/json",
                "User-Agent": "HengSeng-Sentiment-Agent/1.0"
            }
            payload = {
                "query": query,
                "type": "neural",
                "startPublishedDate": start_date,
                "numResults": max_articles,
                "contents": {
                    "text": {"maxCharacters": 1200}
                }
            }
            resp = requests.post(EXA_SEARCH_ENDPOINT, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                seen_urls = set()
                for item in results:
                    url = item.get("url", "")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    text = item.get("text", "")
                    articles.append({
                        "ticker": ticker,
                        "exchange": exchange,
                        "title": DataCleaner.clean_text(item.get("title", f"News on {asset_name}")),
                        "url": url,
                        "published_utc": DataCleaner.normalize_timestamp(item.get("publishedDate")),
                        "source": str(item.get("author") or "Exa Financial News"),
                        "summary": DataCleaner.clean_text(text[:400]),
                        "content": DataCleaner.clean_text(text),
                        "provider": "Exa API (REST)"
                    })
            else:
                logger.warning(f"Exa REST API returned status HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.error(f"Error querying Exa REST API for {ticker}: {e}")

        return articles

    def _fetch_via_tavily(
        self,
        ticker: str,
        asset_name: str,
        exchange: str,
        query: str,
        max_articles: int
    ) -> List[Dict[str, Any]]:
        """Query Tavily search via SDK or direct REST API."""
        articles = []
        # Direct REST endpoint
        try:
            payload = {
                "api_key": self.tavily_key,
                "query": f"{query} stock market earnings news",
                "search_depth": "advanced",
                "max_results": max_articles
            }
            resp = requests.post(TAVILY_SEARCH_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                for item in results:
                    url = item.get("url", "")
                    content = item.get("content", "")
                    articles.append({
                        "ticker": ticker,
                        "exchange": exchange,
                        "title": DataCleaner.clean_text(item.get("title", "")),
                        "url": url,
                        "published_utc": DataCleaner.normalize_timestamp(item.get("published_date")),
                        "source": "Tavily Financial Search",
                        "summary": DataCleaner.clean_text(content[:400]),
                        "content": DataCleaner.clean_text(content),
                        "provider": "Tavily API"
                    })
            else:
                logger.warning(f"Tavily REST returned status HTTP {resp.status_code}")
        except Exception as e:
            logger.error(f"Tavily REST call failed: {e}")

        return articles
