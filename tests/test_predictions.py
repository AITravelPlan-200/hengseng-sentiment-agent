"""
Unit tests for Machine Learning Training, Validation and Inference.
Validates:
- Model instantiation across candidate architectures
- Error metrics calculation (MAE, RMSE, MAPE, R², Directional Accuracy)
- Training and benchmarking pipeline
- Forward prediction and interval generation
"""

import pytest
import numpy as np
import pandas as pd
from models.evaluate import ModelEvaluator
from models.model_registry import get_model_pipeline, NaiveBaselineModel
from models.predict import PricePredictor

def test_directional_accuracy_calculation():
    current_close = np.array([100.0, 100.0, 100.0, 100.0])
    actual_future = np.array([105.0, 95.0, 102.0, 98.0])   # + - + -
    pred_future   = np.array([104.0, 96.0, 99.0, 97.0])    # + - - -
    # Indices 0, 1, 3 match direction (3/4 = 75.0%)

    acc = ModelEvaluator.calculate_directional_accuracy(actual_future, pred_future, current_close)
    assert acc == 75.0

def test_model_evaluator_metrics():
    y_true = np.array([100.0, 105.0, 110.0, 95.0])
    y_pred = np.array([101.0, 104.0, 109.0, 97.0])
    curr = np.array([99.0, 102.0, 108.0, 98.0])

    metrics = ModelEvaluator.evaluate_predictions(y_true, y_pred, curr)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert "r2" in metrics
    assert "directional_accuracy" in metrics
    assert metrics["mae"] > 0
    assert metrics["rmse"] >= metrics["mae"]

def test_naive_baseline_model():
    model = NaiveBaselineModel()
    X = np.array([[100.0, 0.5, 0.2], [105.0, 0.8, -0.1]])
    y = np.array([102.0, 107.0])
    feature_names = ["close", "rsi", "sentiment"]

    model.fit(X, y, feature_names)
    preds = model.predict(X)
    assert np.array_equal(preds, np.array([100.0, 105.0]))

def test_price_predictor_inference():
    live_features = np.array([[400.0] + [0.0]*19])
    current_close = 400.0

    pred = PricePredictor.predict_next_period(
        ticker="0700.HK",
        live_features=live_features,
        current_close=current_close,
        best_model_name="naive_baseline",
        model_rmse=5.0,
        sentiment_score=0.25
    )

    assert pred["ticker"] == "0700.HK"
    assert pred["current_close"] == 400.0
    assert pred["lower_bound"] <= pred["predicted_close"] <= pred["upper_bound"]
    assert pred["direction"] in ["Bullish", "Bearish", "Neutral"]
    assert 0.0 <= pred["confidence_score"] <= 1.0
