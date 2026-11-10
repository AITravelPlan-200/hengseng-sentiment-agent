"""
Multi-Criteria Stock Ranking Engine.
Computes and formats ranked league tables for HKEX and KRX exchanges across:
1. Social Mention Volume
2. Positive Sentiment Score
3. Negative Sentiment Score
4. Combined Sentiment Score
5. Predicted Price Increase (Bullish leaders)
6. Predicted Price Decrease (Bearish leaders)
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger("hengseng.analysis.ranking")

class StockRankingEngine:
    """Ranks and formats asset intelligence summaries by multiple quantitative criteria."""

    @classmethod
    def rank_stocks(
        cls,
        stock_analyses: List[Dict[str, Any]],
        exchange: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generates 6 distinct ranking tables from a list of stock intelligence reports.
        Optionally filters by exchange ('HKEX' or 'KRX').
        """
        if not stock_analyses:
            return {}

        df = pd.DataFrame(stock_analyses)
        if exchange:
            df = df[df["exchange"].str.upper() == exchange.upper()].copy()

        if df.empty:
            return {}

        # Columns for standard display
        display_cols = [
            "ticker",
            "company_name",
            "exchange",
            "current_price",
            "social_mention_count",
            "social_sentiment_score",
            "news_sentiment_score",
            "combined_sentiment_score",
            "predicted_next_period_price",
            "predicted_price_movement_pct",
            "suggested_position"
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        base_df = df[available_cols].copy()

        rankings = {
            # 1. Rank by Social Mentions
            "by_social_mentions": base_df.sort_values(
                by="social_mention_count", ascending=False
            ).reset_index(drop=True),

            # 2. Rank by Positive Sentiment (Highest positive social/news sentiment)
            "by_positive_sentiment": base_df.sort_values(
                by="combined_sentiment_score", ascending=False
            ).reset_index(drop=True),

            # 3. Rank by Negative Sentiment (Lowest / most negative sentiment)
            "by_negative_sentiment": base_df.sort_values(
                by="combined_sentiment_score", ascending=True
            ).reset_index(drop=True),

            # 4. Rank by Combined Sentiment (Absolute sentiment magnitude or score)
            "by_combined_sentiment": base_df.sort_values(
                by="combined_sentiment_score", ascending=False
            ).reset_index(drop=True),

            # 5. Rank by Predicted Price Increase (Highest positive expected return)
            "by_predicted_increase": base_df.sort_values(
                by="predicted_price_movement_pct", ascending=False
            ).reset_index(drop=True),

            # 6. Rank by Predicted Price Decrease (Largest downward projection)
            "by_predicted_decrease": base_df.sort_values(
                by="predicted_price_movement_pct", ascending=True
            ).reset_index(drop=True)
        }

        # Add clean 1-based rank column
        for key in rankings:
            rankings[key].insert(0, "Rank", range(1, len(rankings[key]) + 1))

        return rankings
