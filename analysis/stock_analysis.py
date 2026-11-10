"""
Stock Intelligence Engine.
Synthesizes market data, technical indicators, support/resistance levels,
sentiment analysis, and ML predictions into a unified stock intelligence report.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from config.tickers import get_asset_metadata
from analysis.support_resistance import SupportResistanceAnalyzer
from sentiment.sentiment_aggregator import SentimentAggregator

logger = logging.getLogger("hengseng.analysis.stock")

class StockIntelligenceEngine:
    """Aggregates all quantitative, sentiment, and ML signals for an asset."""

    @classmethod
    def generate_stock_intelligence(
        cls,
        ticker: str,
        market_df: pd.DataFrame,
        social_df: pd.DataFrame,
        news_df: pd.DataFrame,
        prediction_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Builds the comprehensive analysis report for a single ticker.
        Strictly satisfies all required output fields from Section 7.
        """
        meta = get_asset_metadata(ticker)
        company_name = meta["name"]
        exchange = meta["exchange"]
        currency = meta["currency"]

        # Market & Price Information
        if not market_df.empty and len(market_df) > 0:
            latest_row = market_df.iloc[-1]
            prev_row = market_df.iloc[-2] if len(market_df) > 1 else latest_row

            current_price = float(latest_row["close"])
            prev_close = float(prev_row["close"])
            price_change = round(current_price - prev_close, 2)
            price_change_pct = round(((current_price - prev_close) / prev_close) * 100.0, 2) if prev_close > 0 else 0.0

            # Technical indicator values
            rsi = float(latest_row.get("rsi_14", 50.0))
            macd = float(latest_row.get("macd", 0.0))
            macd_signal = float(latest_row.get("macd_signal", 0.0))
            sma_20 = float(latest_row.get("sma_20", current_price))
            sma_50 = float(latest_row.get("sma_50", current_price))
            volatility = float(latest_row.get("volatility_20", 0.0))

            # Technical summary
            tech_bullets = []
            if current_price > sma_20 and sma_20 > sma_50:
                tech_bullets.append("Bullish Moving Average Alignment (Price > SMA20 > SMA50)")
            elif current_price < sma_20:
                tech_bullets.append("Trading below 20-day Moving Average (Short-term weakness)")

            if rsi > 70:
                tech_bullets.append(f"Overbought conditions (RSI: {rsi:.1f})")
            elif rsi < 30:
                tech_bullets.append(f"Oversold conditions (RSI: {rsi:.1f})")
            else:
                tech_bullets.append(f"Neutral momentum (RSI: {rsi:.1f})")

            if macd > macd_signal:
                tech_bullets.append("MACD Bullish Crossover above signal line")
            else:
                tech_bullets.append("MACD Bearish Divergence below signal line")

            technical_summary = " | ".join(tech_bullets)

        else:
            current_price = 0.0
            price_change = 0.0
            price_change_pct = 0.0
            technical_summary = "Market data unavailable"
            rsi, macd, sma_20, sma_50 = 50.0, 0.0, 0.0, 0.0

        # Support & Resistance
        sr_levels = SupportResistanceAnalyzer.calculate_levels(market_df)
        support_level = sr_levels.get("support_level", current_price * 0.96)
        resistance_level = sr_levels.get("resistance_level", current_price * 1.04)

        # Sentiment Aggregation
        sentiment = SentimentAggregator.aggregate_ticker_sentiment(social_df, news_df, ticker)

        # Prediction details
        if prediction_result:
            predicted_close = prediction_result.get("predicted_close", current_price)
            pred_change_pct = prediction_result.get("predicted_change_pct", 0.0)
            confidence = prediction_result.get("confidence_score", 0.70)
            uncertainty_interval = (
                prediction_result.get("lower_bound", predicted_close * 0.98),
                prediction_result.get("upper_bound", predicted_close * 1.02)
            )
        else:
            predicted_close = current_price
            pred_change_pct = 0.0
            confidence = 0.50
            uncertainty_interval = (round(current_price * 0.98, 2), round(current_price * 1.02, 2))

        # Position suggestion logic (combining sentiment + technical + prediction)
        combined_sentiment = sentiment["combined_sentiment_score"]
        if pred_change_pct > 0.5 and combined_sentiment > 0.10:
            suggested_position = "Long"
            position_reasoning = (
                f"Bullish convergence: Positive next-day price forecast (+{pred_change_pct:.2f}%) "
                f"reinforced by favorable composite sentiment (+{combined_sentiment:.2f}) "
                f"and support base near {currency} {support_level:.2f}."
            )
        elif pred_change_pct < -0.5 and combined_sentiment < -0.10:
            suggested_position = "Short"
            position_reasoning = (
                f"Bearish confluence: Model projects negative shift ({pred_change_pct:.2f}%) "
                f"with negative sentiment ({combined_sentiment:.2f}) "
                f"and overhead resistance near {currency} {resistance_level:.2f}."
            )
        else:
            suggested_position = "Neutral"
            position_reasoning = (
                f"Mixed signals: Forecast change ({pred_change_pct:.2f}%) and sentiment ({combined_sentiment:.2f}) "
                f"within consolidation thresholds. Range-bound between {currency} {support_level:.2f} and {resistance_level:.2f}."
            )

        return {
            "ticker": ticker,
            "company_name": company_name,
            "exchange": exchange,
            "currency": currency,
            "current_price": current_price,
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "social_mention_count": sentiment["social_mention_count"],
            "social_sentiment_score": sentiment["social_sentiment_score"],
            "news_sentiment_score": sentiment["news_sentiment_score"],
            "combined_sentiment_score": combined_sentiment,
            "sentiment_direction": sentiment["sentiment_direction"],
            "suggested_position": suggested_position,
            "position_reasoning": position_reasoning,
            "support_level": support_level,
            "resistance_level": resistance_level,
            "pivot_point": sr_levels.get("pivot_point", current_price),
            "technical_indicator_summary": technical_summary,
            "predicted_next_period_price": predicted_close,
            "predicted_price_movement_pct": pred_change_pct,
            "prediction_confidence": confidence,
            "uncertainty_interval": uncertainty_interval,
            "disclaimer": (
                "DISCLAIMER: For informational and quantitative research purposes only. "
                "Machine learning projections and sentiment indicators are probabilistic estimates "
                "and do not constitute guaranteed financial or investment advice."
            )
        }
