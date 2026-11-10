"""
Apify Actor Registry & Input Configuration.
Documents verified Apify actors, target platform mapping, fallback actors,
and default actor payload configurations.
"""

from typing import Dict, Any

APIFY_ACTORS: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------
    # Hong Kong Social Source 1: LIHKG (Hong Kong Reddit equivalent)
    # -------------------------------------------------------------
    "hk_lihkg": {
        "platform": "LIHKG",
        "exchange": "HKEX",
        "actor_id": "curious_coder/lihkg-scraper",
        "fallback_actor_id": "apify/cheerio-scraper",
        "description": "Scrapes hot and latest discussion threads on LIHKG Finance and Current Affairs forums.",
        "verified_endpoint": "https://apify.com/curious_coder/lihkg-scraper",
        "default_input": {
            "searchKeywords": [],
            "maxPosts": 50,
            "category": "finance",
            "sortBy": "latest"
        },
        "cheerio_fallback_config": {
            "startUrls": [{"url": "https://lihkg.com/category/finance"}],
            "maxRequestsPerCrawl": 50
        }
    },

    # -------------------------------------------------------------
    # Hong Kong Social Source 2: Telegram Financial Channels
    # -------------------------------------------------------------
    "hk_telegram": {
        "platform": "Telegram",
        "exchange": "HKEX",
        "actor_id": "apify/telegram-scraper",
        "fallback_actor_id": "pocesar/telegram-scraper",
        "description": "Scrapes public Hong Kong retail trading Telegram channels and groups.",
        "verified_endpoint": "https://apify.com/apify/telegram-scraper",
        "target_channels": [
            "hkstocks_discussion",
            "hkex_traders",
            "hongkongstocksdaily"
        ],
        "default_input": {
            "channels": ["hkstocks_discussion"],
            "messagesLimit": 50,
            "startDate": "10 days ago"
        }
    },

    # -------------------------------------------------------------
    # South Korea Social Source 1: Naver Finance Discussion Boards (네이버 증권 종목토론실)
    # -------------------------------------------------------------
    "kr_naver": {
        "platform": "Naver Finance",
        "exchange": "KRX",
        "actor_id": "apify/web-scraper",
        "fallback_actor_id": "apify/cheerio-scraper",
        "description": "Scrapes Korean retail investor discussions from Naver Finance 종목토론실 (finance.naver.com).",
        "verified_endpoint": "https://finance.naver.com/item/board.naver",
        "base_url": "https://finance.naver.com/item/board.naver?code={symbol}",
        "default_input": {
            "maxPages": 5,
            "targetCode": "005930"
        }
    },

    # -------------------------------------------------------------
    # South Korea Social Source 2: Daum Finance / Telegram Korea
    # -------------------------------------------------------------
    "kr_daum": {
        "platform": "Daum Finance",
        "exchange": "KRX",
        "actor_id": "apify/cheerio-scraper",
        "fallback_actor_id": "apify/telegram-scraper",
        "description": "Scrapes Korean retail investor discussions from Daum Finance (finance.daum.net).",
        "verified_endpoint": "https://finance.daum.net/quotes/A{symbol}#talk",
        "default_input": {
            "maxItems": 50,
            "symbol": "005930"
        }
    },

    # -------------------------------------------------------------
    # Multi-market Fallback: Reddit Retail Investment Communities
    # (r/HKStocks, r/KoreaStocks, r/wallstreetbets, r/investing)
    # -------------------------------------------------------------
    "global_reddit": {
        "platform": "Reddit",
        "exchange": "MULTI",
        "actor_id": "trudax/reddit-scraper-lite",
        "fallback_actor_id": "apify/reddit-scraper",
        "description": "Retrieves retail investor discussions across Reddit finance communities.",
        "default_input": {
            "searches": [],
            "maxItems": 50,
            "sort": "new",
            "time": "week"
        }
    }
}
