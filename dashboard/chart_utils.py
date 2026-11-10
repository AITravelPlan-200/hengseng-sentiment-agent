"""
Interactive Visualization Utilities for Financial Dashboard.
Provides dark-themed Plotly charts for OHLCV candlesticks, predictions with confidence bands,
sentiment gauges, feature importances, and distribution histograms.
Includes graceful native Streamlit fallbacks.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

THEME_BG = "#0d1117"
THEME_GRID = "#21262d"
COLOR_BULLISH = "#00e676"
COLOR_BEARISH = "#ff5252"
COLOR_NEUTRAL = "#90caf9"
COLOR_ACCENT = "#00e5ff"

def get_candlestick_chart(df: pd.DataFrame, ticker: str, title: Optional[str] = None):
    """Generates an interactive candlestick chart with SMA20, SMA50, and Bollinger Bands."""
    if not PLOTLY_AVAILABLE or df.empty:
        return None

    data = df.tail(120).copy()
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data["timestamp_utc"],
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        name="OHLC",
        increasing_line_color=COLOR_BULLISH,
        decreasing_line_color=COLOR_BEARISH
    ))

    # Moving Averages
    if "sma_20" in data.columns:
        fig.add_trace(go.Scatter(
            x=data["timestamp_utc"], y=data["sma_20"],
            mode="lines", name="SMA 20", line=dict(color="#f59e0b", width=1.5)
        ))
    if "sma_50" in data.columns:
        fig.add_trace(go.Scatter(
            x=data["timestamp_utc"], y=data["sma_50"],
            mode="lines", name="SMA 50", line=dict(color="#3b82f6", width=1.5)
        ))

    # Bollinger Bands
    if "bb_upper" in data.columns and "bb_lower" in data.columns:
        fig.add_trace(go.Scatter(
            x=data["timestamp_utc"], y=data["bb_upper"],
            mode="lines", name="BB Upper", line=dict(color="#64748b", width=1, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=data["timestamp_utc"], y=data["bb_lower"],
            mode="lines", name="BB Lower", line=dict(color="#64748b", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(100, 116, 139, 0.08)"
        ))

    fig.update_layout(
        title=title or f"{ticker} - OHLCV Technical Chart",
        template="plotly_dark",
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def get_actual_vs_predicted_chart(
    dates: List[str],
    actuals: List[float],
    predictions: List[float],
    title: str = "Out-of-Sample Predictions vs Actual Closing Prices"
):
    """Generates comparison line chart with prediction error shading."""
    if not PLOTLY_AVAILABLE or not dates:
        return None

    fig = go.Figure()

    # Actual price trace
    fig.add_trace(go.Scatter(
        x=dates, y=actuals,
        mode="lines+markers", name="Actual Close",
        line=dict(color="#ffffff", width=2),
        marker=dict(size=4)
    ))

    # Predicted price trace
    fig.add_trace(go.Scatter(
        x=dates, y=predictions,
        mode="lines+markers", name="ML Predicted",
        line=dict(color=COLOR_ACCENT, width=2, dash="dot"),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def get_sentiment_gauge(score: float, title: str = "Sentiment Meter"):
    """Generates a radial gauge indicator for sentiment (-1.0 to +1.0)."""
    if not PLOTLY_AVAILABLE:
        return None

    color = COLOR_BULLISH if score > 0.15 else (COLOR_BEARISH if score < -0.15 else COLOR_NEUTRAL)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score, 2),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#ffffff"},
            'bar': {'color': color},
            'bgcolor': "#161b22",
            'borderwidth': 2,
            'bordercolor': "#30363d",
            'steps': [
                {'range': [-1, -0.15], 'color': 'rgba(255, 82, 82, 0.2)'},
                {'range': [-0.15, 0.15], 'color': 'rgba(144, 202, 249, 0.2)'},
                {'range': [0.15, 1], 'color': 'rgba(0, 230, 118, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=THEME_BG,
        height=240,
        margin=dict(l=25, r=25, t=40, b=25)
    )
    return fig

def get_feature_importance_chart(feature_dict: Dict[str, float], top_n: int = 10):
    """Horizontal bar chart showing relative predictive power of engineered features."""
    if not PLOTLY_AVAILABLE or not feature_dict:
        return None

    items = list(feature_dict.items())[:top_n]
    features = [k for k, v in items][::-1]
    scores = [v for k, v in items][::-1]

    # Color code sentiment features differently from price features
    colors = [COLOR_ACCENT if "sentiment" in f or "social" in f or "news" in f else "#6366f1" for f in features]

    fig = go.Figure(go.Bar(
        x=scores,
        y=features,
        orientation="h",
        marker=dict(color=colors)
    ))

    fig.update_layout(
        title=f"Top {top_n} Feature Importance (Predictive Weight)",
        template="plotly_dark",
        paper_bgcolor=THEME_BG,
        plot_bgcolor=THEME_BG,
        xaxis_title="Relative Weight",
        margin=dict(l=40, r=40, t=50, b=40),
        height=320
    )
    return fig
