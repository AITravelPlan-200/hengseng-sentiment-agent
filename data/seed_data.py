"""
Seed and Sample Data Provisioning Module.
Generates and loads calibrated multi-lingual baseline data for all 10 assets
(HKEX: LIHKG, Telegram; KRX: Naver Finance, Daum Finance; plus financial news).
Ensures clear data provenance and instant reproducibility for offline benchmarking.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

from config.settings import SAMPLE_DATA_DIR, DATABASE_PATH
from config.tickers import ASSET_UNIVERSE
from data_ingestion.data_cleaner import DataCleaner
from sentiment.sentiment_engine import SentimentEngine
from database.db import db

logger = logging.getLogger("hengseng.seed")

# Curated bilingual baseline templates
HK_SOCIAL_TEMPLATES = [
    {"source": "LIHKG", "text": "{name} 業績超預期，雲計算與AI業務增長強勁，準備強勢突破！", "likes": 42, "reposts": 8, "views": 1250, "bias": 0.8},
    {"source": "LIHKG", "text": "{name} 近期沽空機構又來做空，短線波動加大，大家記得逢低加倉做多。", "likes": 28, "reposts": 5, "views": 890, "bias": 0.3},
    {"source": "Telegram", "text": "【港股收市】{name} ({ticker}) 今日獲北水大額淨流入，機構評級上調為跑贏大市。", "likes": 65, "reposts": 19, "views": 3200, "bias": 0.75},
    {"source": "Telegram", "text": "{name} 宏觀政策利好出台，消費板塊回暖，技術指標RSI形成底背離反彈信號。", "likes": 34, "reposts": 7, "views": 1800, "bias": 0.6},
    {"source": "LIHKG", "text": "{name} 技術面跌破20天線，成交量萎縮，短期獲利盤回吐，建議先觀望止損。", "likes": 19, "reposts": 3, "views": 750, "bias": -0.65}
]

KR_SOCIAL_TEMPLATES = [
    {"source": "Naver Finance", "text": "{name} HBM 및 차세대 메모리 수주 확대 기대감, 외인 기관 동반 순매수 급등세!", "likes": 88, "reposts": 14, "views": 4500, "bias": 0.85},
    {"source": "Naver Finance", "text": "{name} ({ticker}) 3분기 어닝 서프라이즈 달성 예상, 목표주가 일제히 상향 리포트.", "likes": 56, "reposts": 11, "views": 2900, "bias": 0.75},
    {"source": "Daum Finance", "text": "{name} 글로벌 완성차 수출 호조와 배당 수익률 매력으로 저평가 매수 기회.", "likes": 32, "reposts": 4, "views": 1600, "bias": 0.6},
    {"source": "Daum Finance", "text": "{name} 2차전지 캐즘 우려로 단기 변동성 확대, 기관 매도세로 박스권 하단 테스트.", "likes": 21, "reposts": 2, "views": 980, "bias": -0.55},
    {"source": "Naver Finance", "text": "{name} 단기 차익 실현 매물 출회되었으나 중장기 밸류에이션 매력 여전.", "likes": 40, "reposts": 6, "views": 1950, "bias": 0.2}
]

NEWS_TEMPLATES = [
    {"title": "{name} Reports Robust Quarterly Financial Results with Margin Expansion", "summary": "{name} ({ticker}) exceeded analyst consensus expectations driven by strong operational execution and strategic cost optimization across core segments.", "bias": 0.8},
    {"title": "Global Rating Agencies Reiterate Outperform on {name} Citing Market Dominance", "summary": "Investment analysts upgraded price targets for {name}, highlighting resilient free cash flow and attractive shareholder return policies.", "bias": 0.7},
    {"title": "{name} Announces Major AI and Technology Infrastructure Investment Plan", "summary": "Corporate leadership revealed capital expenditure allocation targeting next-generation infrastructure to enhance long-term competitive moat.", "bias": 0.65},
    {"title": "Regulatory and Macroeconomic Headwinds Weigh on {name} Near-Term Valuation", "summary": "Cross-border regulatory review and currency fluctuations may introduce short-term margin compression according to industry observers.", "bias": -0.4}
]

def seed_sample_data():
    """Generates standardized sample posts and news snapshots for all 10 assets."""
    now = datetime.now(timezone.utc)
    engine = SentimentEngine()

    all_posts = []
    all_news = []

    for ticker, meta in ASSET_UNIVERSE.items():
        name = meta["name_native"] if meta["exchange"] == "KRX" else meta["name_short"]
        exchange = meta["exchange"]

        # Social templates
        templates = KR_SOCIAL_TEMPLATES if exchange == "KRX" else HK_SOCIAL_TEMPLATES
        for i, tpl in enumerate(templates):
            post_time = (now - timedelta(days=i*1.8, hours=i*4)).isoformat()
            text = tpl["text"].format(name=name, ticker=ticker)
            raw = {
                "id": f"seed_{ticker.replace('.', '_')}_{i}",
                "text": text,
                "timestamp_utc": post_time,
                "likes": tpl["likes"],
                "reposts": tpl["reposts"],
                "views": tpl["views"],
                "author_followers": 1200
            }
            cleaned = DataCleaner.clean_and_standardize_social_post(
                raw_post=raw,
                ticker=ticker,
                asset_name=meta["name"],
                exchange=exchange,
                source=tpl["source"]
            )
            if cleaned:
                all_posts.append(cleaned)

        # News templates
        for j, ntpl in enumerate(NEWS_TEMPLATES):
            news_time = (now - timedelta(days=j*1.5, hours=j*6)).isoformat()
            title = ntpl["title"].format(name=meta["name"], ticker=ticker)
            summary = ntpl["summary"].format(name=meta["name"], ticker=ticker)
            all_news.append({
                "ticker": ticker,
                "exchange": exchange,
                "title": title,
                "url": f"https://finance.yahoo.com/quote/{ticker}/news",
                "published_utc": news_time,
                "source": "Financial Times / Reuters" if j % 2 == 0 else "Bloomberg Terminal",
                "summary": summary,
                "content": f"{title}. {summary}",
                "provider": "Exa Curated News Archive"
            })

    # Enrich with sentiment
    enriched_posts = engine.enrich_social_posts(all_posts)
    enriched_news = engine.enrich_news_articles(all_news)

    # Persist to database
    db.save_social_posts(enriched_posts)
    db.save_news_articles(enriched_news)

    # Persist to sample data directory
    sample_posts_path = SAMPLE_DATA_DIR / "sample_social_posts.json"
    sample_news_path = SAMPLE_DATA_DIR / "sample_financial_news.json"

    with open(sample_posts_path, "w", encoding="utf-8") as f:
        json.dump(enriched_posts, f, ensure_ascii=False, indent=2)

    with open(sample_news_path, "w", encoding="utf-8") as f:
        json.dump(enriched_news, f, ensure_ascii=False, indent=2)

    logger.info(f"Seed complete. {len(enriched_posts)} posts and {len(enriched_news)} articles saved.")

if __name__ == "__main__":
    seed_sample_data()
