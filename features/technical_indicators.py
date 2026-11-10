"""
Technical Indicator Calculation Engine.
Calculates mathematical and quantitative momentum, trend, and volatility indicators:
- SMA (20, 50)
- EMA (20)
- RSI (14-period Wilder smoothing)
- MACD (12, 26, 9 line, signal, and histogram)
- Bollinger Bands (20-period, 2-sigma upper, middle, lower)
- Daily returns
- Rolling Volatility (20-day)
- Volume rate of change
"""

import numpy as np
import pandas as pd
from typing import Optional

class TechnicalIndicatorCalculator:
    """Calculates quantitative technical indicators on OHLCV pandas DataFrames."""

    @classmethod
    def compute_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all required technical indicators and appends them as columns.
        Expects columns: ['open', 'high', 'low', 'close', 'volume'].
        """
        if df.empty or len(df) < 2:
            return df

        data = df.copy()

        # Ensure column lowercase for consistency
        data.columns = [c.lower() for c in data.columns]

        # 1. Simple Moving Averages
        data["sma_20"] = data["close"].rolling(window=20, min_periods=1).mean()
        data["sma_50"] = data["close"].rolling(window=50, min_periods=1).mean()

        # 2. Exponential Moving Average
        data["ema_20"] = data["close"].ewm(span=20, adjust=False).mean()

        # 3. Relative Strength Index (RSI 14)
        data["rsi_14"] = cls.compute_rsi(data["close"], period=14)

        # 4. Moving Average Convergence Divergence (MACD: 12, 26, 9)
        macd_line, signal_line, hist = cls.compute_macd(data["close"], fast=12, slow=26, signal=9)
        data["macd"] = macd_line
        data["macd_signal"] = signal_line
        data["macd_hist"] = hist

        # 5. Bollinger Bands (20-period, 2 std dev)
        bb_upper, bb_mid, bb_lower = cls.compute_bollinger_bands(data["close"], window=20, num_std=2.0)
        data["bb_upper"] = bb_upper
        data["bb_middle"] = bb_mid
        data["bb_lower"] = bb_lower

        # 6. Daily Returns and Volatility
        data["daily_returns"] = data["close"].pct_change().fillna(0.0)
        data["volatility_20"] = data["daily_returns"].rolling(window=20, min_periods=1).std().fillna(0.0)

        # 7. Volume Rate of Change
        if "volume" in data.columns:
            data["volume_change"] = data["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        else:
            data["volume_change"] = 0.0

        # Round float columns for clean representation
        float_cols = [
            "sma_20", "sma_50", "ema_20", "rsi_14", "macd", "macd_signal",
            "macd_hist", "bb_upper", "bb_middle", "bb_lower", "volatility_20",
            "daily_returns", "volume_change"
        ]
        for col in float_cols:
            if col in data.columns:
                data[col] = data[col].round(4)

        return data

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculates RSI using exponential smoothing."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0.0)).copy()
        loss = (-delta.where(delta < 0, 0.0)).copy()

        # Wilder's exponential smoothing
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Computes MACD line, Signal line, and Histogram."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
        """Computes Upper, Middle (SMA), and Lower Bollinger Bands."""
        middle = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std().fillna(0.0)
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)
        return upper, middle, lower
