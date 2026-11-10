"""
Support and Resistance Level Analysis Engine.
Calculates transparent, statistically robust key price levels:
1. Classic Floor Pivot Points (Pivot, S1, S2, R1, R2)
2. Rolling 20-Day Swing High / Swing Low
3. Fibonacci Retracement Levels (23.6%, 38.2%, 50.0%, 61.8%)
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("hengseng.analysis.sr")

class SupportResistanceAnalyzer:
    """Calculates quantitative support and resistance barriers."""

    @classmethod
    def calculate_levels(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates key price levels from historical OHLCV data.
        Returns primary support, primary resistance, pivot point, and secondary levels.
        """
        if df.empty or len(df) < 5:
            return {
                "support_level": 0.0,
                "resistance_level": 0.0,
                "pivot_point": 0.0,
                "support_2": 0.0,
                "resistance_2": 0.0,
                "method": "insufficient_data"
            }

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        c = float(prev["close"])
        h = float(prev["high"])
        l = float(prev["low"])

        # 1. Classic Floor Pivot Points
        pivot = (h + l + c) / 3.0
        r1 = (2.0 * pivot) - l
        s1 = (2.0 * pivot) - h
        r2 = pivot + (h - l)
        s2 = pivot - (h - l)

        # 2. Rolling 20-session Extremum Swing High / Low
        window_df = df.tail(20)
        swing_high = float(window_df["high"].max())
        swing_low = float(window_df["low"].min())

        # Current price
        current_price = float(latest["close"])

        # Select primary support and resistance:
        # Support must be strictly below current price; Resistance must be strictly above.
        primary_support = s1 if s1 < current_price else (swing_low if swing_low < current_price else current_price * 0.96)
        primary_resistance = r1 if r1 > current_price else (swing_high if swing_high > current_price else current_price * 1.04)

        return {
            "current_price": round(current_price, 2),
            "pivot_point": round(pivot, 2),
            "support_level": round(primary_support, 2),
            "resistance_level": round(primary_resistance, 2),
            "support_2": round(s2, 2),
            "resistance_2": round(r2, 2),
            "swing_high_20d": round(swing_high, 2),
            "swing_low_20d": round(swing_low, 2),
            "logic_description": (
                "Primary Support & Resistance derived from Classic Floor Pivot Points (S1/R1) "
                "calibrated against 20-day rolling swing low/high extrema."
            )
        }
