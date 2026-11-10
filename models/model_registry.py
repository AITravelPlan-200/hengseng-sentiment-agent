"""
Model Registry and Factory.
Instantiates, configures, and saves next-period price prediction models:
1. Linear Regression (Ridge regularized)
2. Random Forest Regressor
3. XGBoost Regressor (with GradientBoosting fallback)
4. Naive Baseline (Persistent Last Close Predictor)
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from config.settings import MODELS_DIR, RANDOM_STATE

logger = logging.getLogger("hengseng.models.registry")

class NaiveBaselineModel:
    """Baseline model predicting tomorrow's close equals today's close."""

    def __init__(self):
        self.feature_names = []
        self.close_idx = 0

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None):
        self.feature_names = feature_names or []
        # Find index of 'close' feature if available
        if "close" in self.feature_names:
            self.close_idx = self.feature_names.index("close")
        else:
            self.close_idx = 0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return np.array([X[self.close_idx]])
        return X[:, self.close_idx]

def get_model_pipeline(model_name: str) -> Any:
    """
    Factory function returning a configured regression model or pipeline.
    Standardizes inputs for linear models while preserving tree algorithms.
    """
    name = model_name.lower().strip()

    if name == "naive_baseline":
        return NaiveBaselineModel()

    elif name == "linear_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, random_state=RANDOM_STATE))
        ])

    elif name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif name == "xgboost":
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        except Exception as e:
            logger.info(f"XGBoost native module deferred ({e}). Using GradientBoostingRegressor.")
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=RANDOM_STATE
            )

    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: 'naive_baseline', 'linear_regression', 'random_forest', 'xgboost'.")

def save_model(model: Any, ticker: str, model_name: str) -> str:
    """Saves serialized model pipeline to disk."""
    safe_ticker = ticker.replace(".", "_")
    filename = f"{safe_ticker}_{model_name}.joblib"
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    logger.info(f"Model saved: {filepath}")
    return str(filepath)

def load_model(ticker: str, model_name: str) -> Optional[Any]:
    """Loads serialized model from disk if present."""
    safe_ticker = ticker.replace(".", "_")
    filepath = MODELS_DIR / f"{safe_ticker}_{model_name}.joblib"
    if filepath.exists():
        try:
            return joblib.load(filepath)
        except Exception as e:
            logger.warning(f"Could not load model at {filepath}: {e}")
    return None
