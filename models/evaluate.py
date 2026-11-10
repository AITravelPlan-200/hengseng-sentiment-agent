"""
Model Evaluation and Quantitative Metrics Engine.
Computes financial machine learning performance indicators:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Directional Accuracy (Hit rate on price movement sign)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelEvaluator:
    """Evaluates next-period stock price regression models."""

    @staticmethod
    def calculate_directional_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        current_close: np.ndarray
    ) -> float:
        """
        Calculates the percentage of correct directional calls:
        sign(y_pred - current_close) == sign(y_true - current_close).
        """
        if len(y_true) == 0:
            return 0.0

        actual_dir = np.sign(y_true - current_close)
        pred_dir = np.sign(y_pred - current_close)

        # Treat flat movements as neutral
        correct = np.sum((actual_dir == pred_dir) | ((actual_dir == 0) & (pred_dir == 0)))
        return round(float(correct / len(y_true)) * 100.0, 2)

    @classmethod
    def evaluate_predictions(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        current_close: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes standard regression metrics and financial directional accuracy.
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "mape": 0.0,
                "r2": 0.0,
                "directional_accuracy": 0.0
            }

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Avoid zero-division in MAPE
        denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
        mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

        r2 = r2_score(y_true, y_pred)
        dir_acc = cls.calculate_directional_accuracy(y_true, y_pred, current_close)

        return {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "mape": round(float(mape), 3),
            "r2": round(float(r2), 4),
            "directional_accuracy": round(float(dir_acc), 2)
        }

    @classmethod
    def generate_walk_forward_splits(
        cls,
        n_samples: int,
        n_splits: int = 5,
        test_size: int = 20
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates expanding-window walk-forward indices for time-series cross validation.
        """
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        indices = []
        for train_idx, test_idx in tscv.split(np.zeros(n_samples)):
            indices.append((train_idx, test_idx))
        return indices
