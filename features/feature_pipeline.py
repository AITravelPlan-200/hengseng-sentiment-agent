"""
Feature Engineering Pipeline for Supervised Next-Day Price Prediction.
Merges market data with aligned sentiment features, constructs lag features,
defines target Y(t) = Close(t+1), and enforces strict no-leakage chronological alignment.
"""

import logging
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from features.technical_indicators import TechnicalIndicatorCalculator
from features.sentiment_features import SentimentFeatureGenerator

logger = logging.getLogger("hengseng.features.pipeline")

# Canonical feature subsets for benchmarking
PRICE_TECHNICAL_FEATURES = [
    "close",
    "returns_1d",
    "returns_5d",
    "volatility_20",
    "volume_change",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "dist_sma_20",
    "dist_sma_50",
    "bb_percent_b"
]

SENTIMENT_FEATURES = [
    "social_mention_count",
    "social_sentiment_weighted",
    "social_total_engagement",
    "social_momentum_3d",
    "news_article_count",
    "news_sentiment_weighted",
    "news_momentum_3d",
    "composite_sentiment"
]

FULL_FEATURE_SET = PRICE_TECHNICAL_FEATURES + SENTIMENT_FEATURES

class FeaturePipeline:
    """End-to-end pipeline preparing train/val/test feature matrices."""

    @classmethod
    def build_dataset(
        cls,
        market_df: pd.DataFrame,
        social_df: pd.DataFrame,
        news_df: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Merges market OHLCV with daily sentiment features.
        Calculates lagged returns, relative distances, and target Y(t) = Close(t+1).
        """
        if market_df.empty or len(market_df) < 30:
            logger.warning(f"Insufficient market bars ({len(market_df)}) for {ticker}.")
            return pd.DataFrame()

        df = market_df.copy()
        df = df.sort_values("timestamp_utc").reset_index(drop=True)

        # Ensure all technical indicators exist
        if "rsi_14" not in df.columns or "macd" not in df.columns:
            df = TechnicalIndicatorCalculator.compute_all_indicators(df)

        # 1. Price Momentum & Distances
        df["returns_1d"] = df["close"].pct_change(1).fillna(0.0)
        df["returns_5d"] = df["close"].pct_change(5).fillna(0.0)
        df["dist_sma_20"] = ((df["close"] / (df["sma_20"] + 1e-9)) - 1.0).fillna(0.0)
        df["dist_sma_50"] = ((df["close"] / (df["sma_50"] + 1e-9)) - 1.0).fillna(0.0)
        
        # Bollinger %B indicator: (Price - Lower) / (Upper - Lower)
        bb_width = (df["bb_upper"] - df["bb_lower"]) + 1e-9
        df["bb_percent_b"] = ((df["close"] - df["bb_lower"]) / bb_width).clip(0.0, 1.0).fillna(0.5)

        # 2. Build Daily Aligned Sentiment Features
        trading_dates = df["timestamp_utc"].tolist()
        sentiment_daily = SentimentFeatureGenerator.create_daily_sentiment_series(
            social_df=social_df,
            news_df=news_df,
            ticker=ticker,
            trading_dates=trading_dates
        )

        # Merge on timestamp_utc
        merged = pd.merge(df, sentiment_daily, on="timestamp_utc", how="left")
        for col in SENTIMENT_FEATURES:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)

        # 3. Supervised Target Variable Definition:
        # Y(t) = Next-day closing price: Close(t+1)
        # Note: The last row will have NaN target, which is reserved for LIVE forward inference!
        merged["target_next_close"] = merged["close"].shift(-1)
        merged["target_return_pct"] = ((merged["target_next_close"] - merged["close"]) / merged["close"]) * 100.0

        return merged

    @classmethod
    def get_train_test_split(
        cls,
        dataset: pd.DataFrame,
        test_days: int = 60,
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Chronologically splits dataset into Train and Out-of-Sample Test splits.
        Guarantees zero future-data leakage.
        """
        feature_cols = feature_cols or FULL_FEATURE_SET
        # Drop the live forecasting row (where target_next_close is NaN)
        labeled = dataset.dropna(subset=["target_next_close"]).copy()
        
        if len(labeled) <= test_days:
            split_idx = int(len(labeled) * 0.8)
        else:
            split_idx = len(labeled) - test_days

        train_df = labeled.iloc[:split_idx].copy()
        test_df = labeled.iloc[split_idx:].copy()

        X_train = train_df[feature_cols].values
        y_train = train_df["target_next_close"].values

        X_test = test_df[feature_cols].values
        y_test = test_df["target_next_close"].values

        # Live row for next period prediction
        live_row = dataset.iloc[[-1]][feature_cols].values
        current_close = float(dataset.iloc[-1]["close"])

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "train_df": train_df,
            "test_df": test_df,
            "live_row": live_row,
            "current_close": current_close,
            "feature_names": feature_cols,
            "train_start": str(train_df.iloc[0]["timestamp_utc"]),
            "train_end": str(train_df.iloc[-1]["timestamp_utc"]),
            "test_start": str(test_df.iloc[0]["timestamp_utc"]),
            "test_end": str(test_df.iloc[-1]["timestamp_utc"])
        }
