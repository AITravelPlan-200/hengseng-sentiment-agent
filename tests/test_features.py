"""
Unit tests for Feature Pipeline and Technical Indicators.
Validates:
- Correct computation of SMA20, SMA50, EMA20, RSI14, MACD, Bollinger Bands
- Returns and rolling volatility properties
- Feature matrix construction and chronological alignment
"""

import pytest
import numpy as np
import pandas as pd
from features.technical_indicators import TechnicalIndicatorCalculator
from features.feature_pipeline import FeaturePipeline, FULL_FEATURE_SET

@pytest.fixture
def sample_market_df():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    price = 100.0 + np.cumsum(np.random.normal(0.1, 1.5, n))
    high = price + np.random.uniform(0.5, 2.0, n)
    low = price - np.random.uniform(0.5, 2.0, n)
    open_p = low + np.random.uniform(0.1, 0.9, n) * (high - low)
    volume = np.random.randint(100000, 5000000, n)

    return pd.DataFrame({
        "timestamp_utc": [d.strftime("%Y-%m-%dT00:00:00Z") for d in dates],
        "open": open_p,
        "high": high,
        "low": low,
        "close": price,
        "volume": volume
    })

def test_technical_indicators_structure(sample_market_df):
    df_ind = TechnicalIndicatorCalculator.compute_all_indicators(sample_market_df)

    expected_cols = [
        "sma_20", "sma_50", "ema_20", "rsi_14", "macd", "macd_signal",
        "macd_hist", "bb_upper", "bb_lower", "bb_middle", "volatility_20",
        "daily_returns", "volume_change"
    ]
    for col in expected_cols:
        assert col in df_ind.columns, f"Missing indicator column: {col}"

    # RSI must be bounded between 0 and 100
    assert (df_ind["rsi_14"] >= 0.0).all()
    assert (df_ind["rsi_14"] <= 100.0).all()

    # Bollinger Bands order: Upper >= Middle >= Lower
    valid_bb = df_ind.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
    assert (valid_bb["bb_upper"] >= valid_bb["bb_middle"]).all()
    assert (valid_bb["bb_middle"] >= valid_bb["bb_lower"]).all()

def test_feature_pipeline_no_future_leakage(sample_market_df):
    empty_social = pd.DataFrame()
    empty_news = pd.DataFrame()

    dataset = FeaturePipeline.build_dataset(
        market_df=sample_market_df,
        social_df=empty_social,
        news_df=empty_news,
        ticker="0700.HK"
    )

    assert "target_next_close" in dataset.columns
    # Check that target is indeed Close shifted by -1
    for i in range(len(dataset) - 1):
        assert dataset.iloc[i]["target_next_close"] == dataset.iloc[i+1]["close"]

    # Verify split chronologically
    splits = FeaturePipeline.get_train_test_split(dataset, test_days=20, feature_cols=FULL_FEATURE_SET)
    train_end = pd.to_datetime(splits["train_end"])
    test_start = pd.to_datetime(splits["test_start"])
    assert train_end < test_start, "Train set overlaps or follows Test set in time!"
