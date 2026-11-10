"""
Agent 4: Prediction Agent.
Combines engineered market indicators with sentiment signals,
trains and validates candidate regression models, and generates next-period price forecasts.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

from config.tickers import ASSET_UNIVERSE
from models.train import ModelTrainer
from models.predict import PricePredictor
from sentiment.sentiment_aggregator import SentimentAggregator

logger = logging.getLogger("hengseng.agents.prediction")

class PredictionAgent:
    """Agent responsible for machine learning training, benchmark evaluation, and live forecasting."""

    def __init__(self):
        self.trainer = ModelTrainer()
        self.predictor = PricePredictor()

    def run_for_ticker(
        self,
        ticker: str,
        market_df: pd.DataFrame,
        social_df: pd.DataFrame,
        news_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Executes chronological model training, benchmarking, and forward price inference.
        """
        logger.info(f"[PredictionAgent] Executing prediction pipeline for {ticker}")

        # 1. Train and evaluate all 4 model architectures
        train_results = self.trainer.train_and_evaluate_all(
            market_df=market_df,
            social_df=social_df,
            news_df=news_df,
            ticker=ticker
        )

        if not train_results:
            logger.warning(f"[PredictionAgent] Model training failed for {ticker}. Using naive fallback.")
            current_close = float(market_df.iloc[-1]["close"]) if not market_df.empty else 100.0
            return {
                "ticker": ticker,
                "current_close": current_close,
                "predicted_close": current_close,
                "predicted_change_pct": 0.0,
                "direction": "Neutral",
                "lower_bound": round(current_close * 0.98, 2),
                "upper_bound": round(current_close * 1.02, 2),
                "confidence_score": 0.50,
                "best_model_name": "naive_baseline",
                "benchmarks": {}
            }

        # 2. Extract best model and live features
        best_model = train_results["best_model_name"]
        live_row = train_results["live_row_enhanced"]
        current_close = train_results["current_close"]
        model_rmse = train_results["benchmarks"][best_model]["sentiment_enhanced"]["rmse"]

        # Aggregate sentiment for direction guidance
        sentiment = SentimentAggregator.aggregate_ticker_sentiment(social_df, news_df, ticker)
        sentiment_score = sentiment["combined_sentiment_score"]

        # 3. Generate live next-period prediction
        prediction = self.predictor.predict_next_period(
            ticker=ticker,
            live_features=live_row,
            current_close=current_close,
            best_model_name=best_model,
            model_rmse=model_rmse,
            sentiment_score=sentiment_score
        )

        return {
            **prediction,
            "best_model_name": best_model,
            "benchmarks": train_results["benchmarks"],
            "feature_importance": train_results.get("feature_importance", {}),
            "training_summary": {
                "train_start": train_results["train_start"],
                "train_end": train_results["train_end"],
                "test_start": train_results["test_start"],
                "test_end": train_results["test_end"]
            }
        }
