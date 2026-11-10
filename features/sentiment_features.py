"""
Sentiment Feature Engineering Engine.
Aligns unstructured social and news sentiment streams into chronological daily trading bars.
Calculates mention frequency, rolling sentiment momentum, and engagement intensity.
"""

import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger("hengseng.features.sentiment")

class SentimentFeatureGenerator:
    """Generates daily aggregated sentiment features for market integration."""

    @classmethod
    def create_daily_sentiment_series(
        cls,
        social_df: pd.DataFrame,
        news_df: pd.DataFrame,
        ticker: str,
        trading_dates: List[str]
    ) -> pd.DataFrame:
        """
        Creates a time-series aligned DataFrame of sentiment features
        matching each trading day's timestamp.
        """
        records = []
        
        # Filter ticker data
        ticker_social = social_df[social_df["ticker"] == ticker].copy() if not social_df.empty and "ticker" in social_df.columns else pd.DataFrame()
        ticker_news = news_df[news_df["ticker"] == ticker].copy() if not news_df.empty and "ticker" in news_df.columns else pd.DataFrame()

        # Parse date strings to date objects for clean session alignment
        if not ticker_social.empty and "timestamp_utc" in ticker_social.columns:
            ticker_social["date"] = pd.to_datetime(ticker_social["timestamp_utc"], utc=True).dt.date
        if not ticker_news.empty and "published_utc" in ticker_news.columns:
            ticker_news["date"] = pd.to_datetime(ticker_news["published_utc"], utc=True).dt.date

        # Rolling tracker for lagged sentiment momentum
        prev_social_scores = []
        prev_news_scores = []

        for date_str in trading_dates:
            dt_obj = pd.to_datetime(date_str, utc=True).date()

            # Social metrics for session
            if not ticker_social.empty and "date" in ticker_social.columns:
                s_day = ticker_social[ticker_social["date"] == dt_obj]
                s_count = len(s_day)
                s_score = float(s_day["sentiment_score"].mean()) if s_count > 0 else 0.0
                s_weighted = float(s_day["weighted_sentiment"].mean()) if s_count > 0 else 0.0
                s_eng = float((s_day["likes"] + s_day["reposts"] * 2 + s_day["views"] * 0.01).sum()) if s_count > 0 else 0.0
            else:
                s_count = 0
                s_score = 0.0
                s_weighted = 0.0
                s_eng = 0.0

            # News metrics for session
            if not ticker_news.empty and "date" in ticker_news.columns:
                n_day = ticker_news[ticker_news["date"] == dt_obj]
                n_count = len(n_day)
                n_score = float(n_day["sentiment_score"].mean()) if n_count > 0 else 0.0
                n_weighted = float(n_day["sentiment_score"].mean()) if n_count > 0 else 0.0
            else:
                n_count = 0
                n_score = 0.0
                n_weighted = 0.0

            # Track rolling sentiment history for momentum
            prev_social_scores.append(s_score)
            prev_news_scores.append(n_score)
            
            # 3-day sentiment momentum: current - mean of last 3 sessions
            s_momentum_3d = s_score - float(np.mean(prev_social_scores[-4:-1])) if len(prev_social_scores) >= 4 else 0.0
            n_momentum_3d = n_score - float(np.mean(prev_news_scores[-4:-1])) if len(prev_news_scores) >= 4 else 0.0

            # Composite daily sentiment (50/50 when both present)
            if s_count > 0 and n_count > 0:
                composite_score = 0.45 * s_weighted + 0.55 * n_weighted
            elif s_count > 0:
                composite_score = s_weighted
            elif n_count > 0:
                composite_score = n_weighted
            else:
                composite_score = 0.0

            records.append({
                "timestamp_utc": date_str,
                "social_mention_count": s_count,
                "social_sentiment_mean": round(s_score, 4),
                "social_sentiment_weighted": round(s_weighted, 4),
                "social_total_engagement": round(s_eng, 2),
                "social_momentum_3d": round(s_momentum_3d, 4),
                "news_article_count": n_count,
                "news_sentiment_mean": round(n_score, 4),
                "news_sentiment_weighted": round(n_weighted, 4),
                "news_momentum_3d": round(n_momentum_3d, 4),
                "composite_sentiment": round(composite_score, 4)
            })

        return pd.DataFrame(records)
