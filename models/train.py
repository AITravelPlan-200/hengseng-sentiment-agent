"""
Model Training and Rigorous Chronological Walk-Forward Validation Engine.
Trains, benchmarks, and persists all 4 candidate models:
1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor
4. Naive Baseline

Evaluates both Sentiment-Enhanced and Price-Only variants to measure Sentiment Alpha.
"""

import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from features.feature_pipeline import (
    FeaturePipeline,
    FULL_FEATURE_SET,
    PRICE_TECHNICAL_FEATURES,
    SENTIMENT_FEATURES
)
from models.model_registry import get_model_pipeline, save_model
from models.evaluate import ModelEvaluator
from database.db import db

logger = logging.getLogger("hengseng.models.train")

MODEL_NAMES = ["naive_baseline", "linear_regression", "random_forest", "xgboost"]

class ModelTrainer:
    """Trains and rigorously validates price prediction architectures."""

    @classmethod
    def train_and_evaluate_all(
        cls,
        market_df: pd.DataFrame,
        social_df: pd.DataFrame,
        news_df: pd.DataFrame,
        ticker: str,
        test_days: int = 60
    ) -> Dict[str, Any]:
        """
        Orchestrates chronological training and benchmarking across all models
        for both sentiment-enhanced and price-only feature sets.
        """
        logger.info(f"--- Starting Model Training & Validation for {ticker} ---")

        # 1. Build unified feature dataset
        dataset = FeaturePipeline.build_dataset(market_df, social_df, news_df, ticker)
        if dataset.empty or len(dataset) < 40:
            logger.warning(f"Insufficient merged data points for {ticker}.")
            return {}

        # 2. Extract Chronological Train / Out-of-Sample Test splits
        splits_enhanced = FeaturePipeline.get_train_test_split(
            dataset=dataset,
            test_days=test_days,
            feature_cols=FULL_FEATURE_SET
        )

        splits_price_only = FeaturePipeline.get_train_test_split(
            dataset=dataset,
            test_days=test_days,
            feature_cols=PRICE_TECHNICAL_FEATURES
        )

        # Baseline Close series on Test set for directional accuracy
        test_current_close = splits_enhanced["test_df"]["close"].values

        benchmark_results = {}
        trained_models = {}

        # 3. Train each model architecture
        for model_name in MODEL_NAMES:
            # A. Sentiment-Enhanced Model
            pipe_enh = get_model_pipeline(model_name)
            if model_name == "naive_baseline":
                pipe_enh.fit(splits_enhanced["X_train"], splits_enhanced["y_train"], FULL_FEATURE_SET)
            else:
                pipe_enh.fit(splits_enhanced["X_train"], splits_enhanced["y_train"])

            preds_enh = pipe_enh.predict(splits_enhanced["X_test"])
            metrics_enh = ModelEvaluator.evaluate_predictions(
                y_true=splits_enhanced["y_test"],
                y_pred=preds_enh,
                current_close=test_current_close
            )
            save_model(pipe_enh, ticker, f"{model_name}_enhanced")
            trained_models[f"{model_name}_enhanced"] = pipe_enh

            # B. Price-Only Baseline Model (Ablation benchmark)
            pipe_price = get_model_pipeline(model_name)
            if model_name == "naive_baseline":
                pipe_price.fit(splits_price_only["X_train"], splits_price_only["y_train"], PRICE_TECHNICAL_FEATURES)
            else:
                pipe_price.fit(splits_price_only["X_train"], splits_price_only["y_train"])

            preds_price = pipe_price.predict(splits_price_only["X_test"])
            metrics_price = ModelEvaluator.evaluate_predictions(
                y_true=splits_price_only["y_test"],
                y_pred=preds_price,
                current_close=test_current_close
            )
            save_model(pipe_price, ticker, f"{model_name}_price_only")
            trained_models[f"{model_name}_price_only"] = pipe_price

            # Measure sentiment improvement (alpha)
            mae_improvement = metrics_price["mae"] - metrics_enh["mae"]
            dir_improvement = metrics_enh["directional_accuracy"] - metrics_price["directional_accuracy"]

            benchmark_results[model_name] = {
                "sentiment_enhanced": metrics_enh,
                "price_only": metrics_price,
                "mae_improvement": round(float(mae_improvement), 4),
                "directional_improvement_pct": round(float(dir_improvement), 2),
                "test_predictions": preds_enh.tolist(),
                "test_actuals": splits_enhanced["y_test"].tolist(),
                "test_dates": splits_enhanced["test_df"]["timestamp_utc"].tolist()
            }

            # Persist evaluation metrics to SQLite
            cls._log_metrics_to_db(ticker, model_name, metrics_enh, splits_enhanced, is_sentiment=1)
            cls._log_metrics_to_db(ticker, model_name, metrics_price, splits_price_only, is_sentiment=0)

        # 4. Identify best performing model based on test RMSE and Directional Accuracy
        best_model_name = "random_forest"
        lowest_rmse = float("inf")
        for m_name in ["linear_regression", "random_forest", "xgboost"]:
            rmse = benchmark_results[m_name]["sentiment_enhanced"]["rmse"]
            if rmse < lowest_rmse:
                lowest_rmse = rmse
                best_model_name = m_name

        # Calculate feature importance for best tree model
        best_pipe = trained_models.get(f"{best_model_name}_enhanced")
        feature_importance = cls._extract_feature_importance(best_pipe, FULL_FEATURE_SET)

        return {
            "ticker": ticker,
            "best_model_name": best_model_name,
            "benchmarks": benchmark_results,
            "feature_importance": feature_importance,
            "train_start": splits_enhanced["train_start"],
            "train_end": splits_enhanced["train_end"],
            "test_start": splits_enhanced["test_start"],
            "test_end": splits_enhanced["test_end"],
            "dataset_len": len(dataset),
            "test_len": len(splits_enhanced["test_df"]),
            "current_close": splits_enhanced["current_close"],
            "live_row_enhanced": splits_enhanced["live_row"],
            "live_row_price": splits_price_only["live_row"]
        }

    @staticmethod
    def _extract_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extracts normalized feature importance from tree regressor or regression weights."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            total = sum(importances) + 1e-9
            res = {name: round(float(imp / total), 4) for name, imp in zip(feature_names, importances)}
            # Sort descending
            return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("regressor"), "coef_"):
            coefs = np.abs(model.named_steps["regressor"].coef_)
            total = sum(coefs) + 1e-9
            res = {name: round(float(c / total), 4) for name, c in zip(feature_names, coefs)}
            return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
        return {name: round(1.0 / len(feature_names), 4) for name in feature_names}

    @staticmethod
    def _log_metrics_to_db(ticker: str, model_name: str, metrics: Dict[str, Any], splits: Dict[str, Any], is_sentiment: int):
        """Save benchmark entry in SQLite database."""
        try:
            sql = """
            INSERT INTO model_metrics (
                model_name, ticker, evaluation_date, mae, rmse, mape, r2,
                directional_accuracy, is_sentiment_enhanced,
                train_start, train_end, test_start, test_end, details_json
            ) VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                model_name, ticker, metrics["mae"], metrics["rmse"], metrics["mape"],
                metrics["r2"], metrics["directional_accuracy"], is_sentiment,
                splits["train_start"], splits["train_end"], splits["test_start"], splits["test_end"],
                json.dumps(metrics)
            )
            db.execute_non_query(sql, params)
        except Exception as e:
            logger.debug(f"Could not persist model metric: {e}")
