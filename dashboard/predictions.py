"""
Page 6: Price Predictions Dashboard.
Displays next trading session point forecasts, 95% uncertainty intervals,
actual vs predicted out-of-sample performance, and cross-model benchmarks.
"""

import streamlit as st
import pandas as pd
from dashboard.chart_utils import (
    get_actual_vs_predicted_chart,
    COLOR_BULLISH, COLOR_BEARISH, COLOR_NEUTRAL
)

def render_predictions_page(pipeline_data: dict):
    """Renders Page 6: Price Predictions."""
    st.markdown("## 🔮 Next-Period Price Prediction Engine")
    st.markdown(
        "Supervised machine learning forecasts combining technical indicators with "
        "engagement-weighted social and news sentiment."
    )

    stock_analyses = pipeline_data.get("stock_analyses", [])
    predictions_dict = pipeline_data.get("predictions_dict", {})

    if not stock_analyses:
        st.warning("No predictions available. Run the pipeline to train models and generate forecasts.")
        return

    df = pd.DataFrame(stock_analyses)

    # 1. Prediction League Table
    st.markdown("### 📈 Next-Session Price Projections")
    disp_df = df[[
        "ticker", "company_name", "exchange", "current_price",
        "predicted_next_period_price", "predicted_price_movement_pct",
        "sentiment_direction", "prediction_confidence", "uncertainty_interval", "suggested_position"
    ]].copy()

    # Format uncertainty interval as string
    disp_df["Uncertainty (95% CI)"] = disp_df["uncertainty_interval"].apply(
        lambda x: f"{x[0]:.2f} - {x[1]:.2f}" if isinstance(x, (tuple, list)) else "-"
    )
    disp_df = disp_df.drop(columns=["uncertainty_interval"])
    disp_df.columns = [
        "Ticker", "Asset", "Exchange", "Current Close", "Pred Next Close",
        "Expected Return %", "Direction", "Confidence", "Position", "95% Confidence Interval"
    ]
    st.dataframe(disp_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Detailed Inspection of Selected Asset
    selected_ticker = st.selectbox(
        "Select Asset for Actual vs Predicted Chronological Walk-Forward Plot:",
        [s["ticker"] for s in stock_analyses]
    )

    pred_info = predictions_dict.get(selected_ticker, {})
    benchmarks = pred_info.get("benchmarks", {})
    best_model_name = pred_info.get("best_model_name", "random_forest")

    if benchmarks and best_model_name in benchmarks:
        best_data = benchmarks[best_model_name]
        dates = best_data.get("test_dates", [])
        actuals = best_data.get("test_actuals", [])
        preds = best_data.get("test_predictions", [])

        st.markdown(f"#### 📉 Out-of-Sample Test Window: {selected_ticker} (Model: `{best_model_name}`)")
        fig = get_actual_vs_predicted_chart(
            dates=dates,
            actuals=actuals,
            predictions=preds,
            title=f"{selected_ticker} - Out-of-Sample Actual vs Predicted Closing Price ({best_model_name})"
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # 3. Model Architecture Comparison Table
        st.markdown("#### ⚖️ Architecture Benchmarking (Sentiment-Enhanced vs Price-Only)")
        benchmark_rows = []
        for m_name, m_data in benchmarks.items():
            enh = m_data.get("sentiment_enhanced", {})
            price = m_data.get("price_only", {})
            benchmark_rows.append({
                "Model Architecture": m_name.replace("_", " ").title(),
                "MAE (Enhanced)": enh.get("mae", 0.0),
                "RMSE (Enhanced)": enh.get("rmse", 0.0),
                "MAPE % (Enhanced)": f"{enh.get('mape', 0.0):.2f}%",
                "R² (Enhanced)": enh.get("r2", 0.0),
                "Directional Hit Rate % (Enhanced)": f"{enh.get('directional_accuracy', 0.0):.1f}%",
                "Directional Hit Rate % (Price Only)": f"{price.get('directional_accuracy', 0.0):.1f}%",
                "Sentiment Alpha Δ": f"{m_data.get('directional_improvement_pct', 0.0):+.1f}%"
            })
        st.dataframe(pd.DataFrame(benchmark_rows), use_container_width=True, hide_index=True)
    else:
        st.info(f"Historical out-of-sample test logs for {selected_ticker} will appear after full model training.")
