"""
Sentiment Aggregator.
Aggregates sentiment across posts, sources, tickers, exchanges, and timeframes.
Clearly partitions social retail sentiment and financial news sentiment,
and computes combined multi-source signals.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from config.settings import BULLISH_THRESHOLD, BEARISH_THRESHOLD

logger = logging.getLogger("hengseng.sentiment.aggregator")

class SentimentAggregator:
    """Aggregates and aligns social and news sentiment metrics."""

    @staticmethod
    def aggregate_ticker_sentiment(
        social_df: pd.DataFrame,
        news_df: pd.DataFrame,
        ticker: str,
        social_weight: float = 0.45,
        news_weight: float = 0.55
    ) -> Dict[str, Any]:
        """
        Computes composite sentiment profile for a single ticker.
        Distinguishes social vs news components.
        """
        # 1. Social Sentiment Aggregation
        social_posts = social_df[social_df["ticker"] == ticker] if not social_df.empty and "ticker" in social_df.columns else pd.DataFrame()
        social_count = len(social_posts)

        if social_count > 0:
            avg_social_raw = float(social_posts["sentiment_score"].mean())
            weighted_social = float(social_posts["weighted_sentiment"].mean())
            social_bullish_count = int((social_posts["sentiment_score"] > BULLISH_THRESHOLD).sum())
            social_bearish_count = int((social_posts["sentiment_score"] < BEARISH_THRESHOLD).sum())
            social_neutral_count = social_count - (social_bullish_count + social_bearish_count)
            sources_breakdown = social_posts["source"].value_counts().to_dict()
        else:
            avg_social_raw = 0.0
            weighted_social = 0.0
            social_bullish_count = 0
            social_bearish_count = 0
            social_neutral_count = 0
            sources_breakdown = {}

        # 2. News Sentiment Aggregation
        news_articles = news_df[news_df["ticker"] == ticker] if not news_df.empty and "ticker" in news_df.columns else pd.DataFrame()
        news_count = len(news_articles)

        if news_count > 0:
            avg_news_raw = float(news_articles["sentiment_score"].mean())
            # Recency-weighted news sentiment
            if "recency_weight" in news_articles.columns:
                rec_weights = news_articles["recency_weight"]
                if rec_weights.sum() > 0:
                    weighted_news = float((news_articles["sentiment_score"] * rec_weights).sum() / rec_weights.sum())
                else:
                    weighted_news = avg_news_raw
            else:
                weighted_news = avg_news_raw

            news_bullish_count = int((news_articles["sentiment_score"] > BULLISH_THRESHOLD).sum())
            news_bearish_count = int((news_articles["sentiment_score"] < BEARISH_THRESHOLD).sum())
            news_neutral_count = news_count - (news_bullish_count + news_bearish_count)
        else:
            avg_news_raw = 0.0
            weighted_news = 0.0
            news_bullish_count = 0
            news_bearish_count = 0
            news_neutral_count = 0

        # 3. Combined Sentiment Computation
        # Dynamic weighting based on data availability
        if social_count > 0 and news_count > 0:
            w_s, w_n = social_weight, news_weight
        elif social_count > 0:
            w_s, w_n = 1.0, 0.0
        elif news_count > 0:
            w_s, w_n = 0.0, 1.0
        else:
            w_s, w_n = 0.5, 0.5

        combined_score = round((w_s * weighted_social) + (w_n * weighted_news), 4)

        # Direction classification
        if combined_score > BULLISH_THRESHOLD:
            direction = "Bullish"
            suggested_position = "Long"
        elif combined_score < BEARISH_THRESHOLD:
            direction = "Bearish"
            suggested_position = "Short"
        else:
            direction = "Neutral"
            suggested_position = "Neutral"

        return {
            "ticker": ticker,
            "social_mention_count": social_count,
            "social_sentiment_score": round(weighted_social, 4),
            "social_raw_mean": round(avg_social_raw, 4),
            "social_bullish": social_bullish_count,
            "social_bearish": social_bearish_count,
            "social_neutral": social_neutral_count,
            "social_sources": sources_breakdown,
            "news_article_count": news_count,
            "news_sentiment_score": round(weighted_news, 4),
            "news_raw_mean": round(avg_news_raw, 4),
            "news_bullish": news_bullish_count,
            "news_bearish": news_bearish_count,
            "news_neutral": news_neutral_count,
            "combined_sentiment_score": combined_score,
            "sentiment_direction": direction,
            "suggested_position": suggested_position
        }

    @classmethod
    def aggregate_exchange_summary(
        cls,
        social_df: pd.DataFrame,
        news_df: pd.DataFrame,
        exchange: str,
        tickers: list
    ) -> Dict[str, Any]:
        """Generate high-level market intelligence summary for an exchange."""
        exchange_tickers = [t for t in tickers]
        ticker_summaries = [
            cls.aggregate_ticker_sentiment(social_df, news_df, t)
            for t in exchange_tickers
        ]

        total_mentions = sum(s["social_mention_count"] for s in ticker_summaries)
        total_news = sum(s["news_article_count"] for s in ticker_summaries)
        
        scores = [s["combined_sentiment_score"] for s in ticker_summaries if (s["social_mention_count"] > 0 or s["news_article_count"] > 0)]
        mean_score = round(float(np.mean(scores)), 4) if scores else 0.0

        bullish_count = sum(1 for s in ticker_summaries if s["sentiment_direction"] == "Bullish")
        bearish_count = sum(1 for s in ticker_summaries if s["sentiment_direction"] == "Bearish")
        neutral_count = sum(1 for s in ticker_summaries if s["sentiment_direction"] == "Neutral")

        return {
            "exchange": exchange,
            "assets_covered": len(exchange_tickers),
            "total_social_mentions": total_mentions,
            "total_news_articles": total_news,
            "average_sentiment": mean_score,
            "bullish_assets": bullish_count,
            "bearish_assets": bearish_count,
            "neutral_assets": neutral_count,
            "ticker_summaries": ticker_summaries
        }
