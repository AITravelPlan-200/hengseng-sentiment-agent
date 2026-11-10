"""
Page 8: Model Performance and Scientific Validation Dashboard.
Visualizes out-of-sample error metrics, walk-forward evaluation,
sentiment alpha attribution (enhanced vs price-only), and feature importance.
"""

import streamlit as st
import pandas as pd
from dashboard.chart_utils import get_feature_importance_chart

def render_performance_page(pipeline_data: dict):
    """Renders Page 8: Model Performance."""
    st.markdown("## 🔬 Machine Learning Evaluation & Sentiment Alpha Benchmark")
    st.markdown(
        "Rigorous chronological walk-forward validation comparing "
        "Sentiment-Enhanced regression models against Price-Only baselines."
    )

    predictions_dict = pipeline_data.get("predictions_dict", {})
    if not predictions_dict:
        st.warning("No model evaluation data available. Run the intelligence pipeline to train models.")
        return

    # Select ticker to inspect performance
    ticker_list = list(predictions_dict.keys())
    selected_ticker = st.selectbox("Select Asset to Audit Model Performance:", ticker_list)

    pred_info = predictions_dict.get(selected_ticker, {})
    benchmarks = pred_info.get("benchmarks", {})
    feature_importance = pred_info.get("feature_importance", {})
    training_summary = pred_info.get("training_summary", {})

    # Validation Period Meta
    if training_summary:
        st.info(
            f"**Validation Strategy:** Strict Chronological Split (Zero Future-Data Leakage)  \n"
            f"- **In-Sample Training Window:** `{training_summary.get('train_start', '')[:10]}` to `{training_summary.get('train_end', '')[:10]}`  \n"
            f"- **Out-of-Sample Test Window:** `{training_summary.get('test_start', '')[:10]}` to `{training_summary.get('test_end', '')[:10]}`"
        )

    st.markdown("---")

    # 1. Performance Table
    st.markdown("### 📊 Architecture Benchmark League Table")
    if benchmarks:
        rows = []
        for model_name, res in benchmarks.items():
            enh = res.get("sentiment_enhanced", {})
            price = res.get("price_only", {})
            rows.append({
                "Architecture": model_name.replace("_", " ").title(),
                "MAE (Enhanced)": f"{enh.get('mae', 0.0):.3f}",
                "RMSE (Enhanced)": f"{enh.get('rmse', 0.0):.3f}",
                "MAPE % (Enhanced)": f"{enh.get('mape', 0.0):.2f}%",
                "R² Score": f"{enh.get('r2', 0.0):.3f}",
                "Hit Rate % (Enhanced)": f"{enh.get('directional_accuracy', 0.0):.1f}%",
                "Hit Rate % (Price-Only)": f"{price.get('directional_accuracy', 0.0):.1f}%",
                "Alpha Delta (Sentiment Gain)": f"{res.get('directional_improvement_pct', 0.0):+.1f}%"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Feature Importance Chart
    st.markdown("### 🧠 Feature Importance: Technicals vs Sentiment Signals")
    st.markdown("Quantifies the proportional contribution of retail sentiment features in next-day forecasting.")

    if feature_importance:
        fig_feat = get_feature_importance_chart(feature_importance, top_n=12)
        if fig_feat:
            st.plotly_chart(fig_feat, use_container_width=True)

        # Tabular breakdown
        with st.expander("View Full Feature Weight Table"):
            feat_df = pd.DataFrame(list(feature_importance.items()), columns=["Feature Name", "Importance Weight"])
            st.dataframe(feat_df, use_container_width=True)

    # 3. Methodological Notes
    st.markdown("---")
    st.markdown(
        """
        ### 📐 Methodological Integrity & Leakage Prevention:
        1. **Chronological Splitting:** Random train/test shuffling (`train_test_split(shuffle=True)`) is strictly forbidden for time series.
        2. **Scaler Hygiene:** All preprocessing transformers (e.g. `StandardScaler`) are fit exclusively on the historical training fold.
        3. **Target Alignment:** The target $Y(t) = \\text{Close}(t+1)$ uses strictly the next session's close; features at time $t$ only contain data known at the close of session $t$.
        4. **Sentiment Alpha Quantification:** The delta between the sentiment-enhanced model and the identical price-only architecture proves whether retail social discussions add real predictive edge.
        """
    )
