"""
Page 1: Executive Overview Dashboard.
Provides high-level institutional market summary across HKEX and KRX,
aggregating social retail volume, sentiment directional bias, and cross-exchange benchmarks.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dashboard.chart_utils import get_sentiment_gauge, COLOR_BULLISH, COLOR_BEARISH, COLOR_NEUTRAL

def render_overview_page(pipeline_data: dict):
    """Renders Page 1: Executive Overview."""
    st.markdown("## 📊 Executive Overview: Multi-Market Sentiment & Forecasts")
    st.markdown(
        "Real-time synthesis of retail social volume, institutional news sentiment, "
        "and machine learning price projections across the Hong Kong and South Korean equity markets."
    )

    stock_analyses = pipeline_data.get("stock_analyses", [])
    if not stock_analyses:
        st.warning("No asset analyses loaded. Click 'Run Full Intelligence Pipeline' in the sidebar to initialize.")
        return

    df = pd.DataFrame(stock_analyses)

    # 1. High-Level KPI Summary Metrics
    total_assets = len(df)
    total_mentions = int(df["social_mention_count"].sum())
    avg_sentiment = float(df["combined_sentiment_score"].mean())
    bullish_count = int((df["sentiment_direction"] == "Bullish").sum())
    bearish_count = int((df["sentiment_direction"] == "Bearish").sum())
    neutral_count = int((df["sentiment_direction"] == "Neutral").sum())
    latest_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Assets Analyzed", f"{total_assets}", "10 Core Target")
    kpi2.metric("Total Social Mentions", f"{total_mentions:,}", "Last 10 Days")
    kpi3.metric("Composite Sentiment", f"{avg_sentiment:+.2f}", "Directional Score")
    kpi4.metric("Bullish Assets", f"{bullish_count}", f"{(bullish_count/total_assets)*100:.0f}% of total")
    kpi5.metric("Bearish Assets", f"{bearish_count}", f"{(bearish_count/total_assets)*100:.0f}% of total")
    kpi6.metric("Last Updated", latest_ts.split(" ")[1], latest_ts.split(" ")[0])

    st.markdown("---")

    # 2. Exchange Breakdown (HKEX vs KRX)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🇭🇰 Hong Kong Stock Exchange (HKEX)")
        hk_df = df[df["exchange"] == "HKEX"]
        if not hk_df.empty:
            hk_mentions = int(hk_df["social_mention_count"].sum())
            hk_sentiment = float(hk_df["combined_sentiment_score"].mean())
            hk_bullish = int((hk_df["sentiment_direction"] == "Bullish").sum())
            hk_bearish = int((hk_df["sentiment_direction"] == "Bearish").sum())
            
            c1, c2, c3 = st.columns(3)
            c1.metric("HKEX Mentions", f"{hk_mentions:,}")
            c2.metric("HKEX Sentiment", f"{hk_sentiment:+.2f}")
            c3.metric("Bullish / Bearish", f"{hk_bullish} / {hk_bearish}")

            # Top HKEX Table
            disp_hk = hk_df[[
                "ticker", "company_name", "current_price",
                "combined_sentiment_score", "predicted_price_movement_pct", "suggested_position"
            ]].copy()
            disp_hk.columns = ["Ticker", "Company", "Price (HKD)", "Sentiment", "Pred Return %", "Position"]
            st.dataframe(disp_hk, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("### 🇰🇷 Korea Exchange (KRX)")
        kr_df = df[df["exchange"] == "KRX"]
        if not kr_df.empty:
            kr_mentions = int(kr_df["social_mention_count"].sum())
            kr_sentiment = float(kr_df["combined_sentiment_score"].mean())
            kr_bullish = int((kr_df["sentiment_direction"] == "Bullish").sum())
            kr_bearish = int((kr_df["sentiment_direction"] == "Bearish").sum())
            
            c1, c2, c3 = st.columns(3)
            c1.metric("KRX Mentions", f"{kr_mentions:,}")
            c2.metric("KRX Sentiment", f"{kr_sentiment:+.2f}")
            c3.metric("Bullish / Bearish", f"{kr_bullish} / {kr_bearish}")

            # Top KRX Table
            disp_kr = kr_df[[
                "ticker", "company_name", "current_price",
                "combined_sentiment_score", "predicted_price_movement_pct", "suggested_position"
            ]].copy()
            disp_kr.columns = ["Ticker", "Company", "Price (KRW)", "Sentiment", "Pred Return %", "Position"]
            st.dataframe(disp_kr, use_container_width=True, hide_index=True)

    # 3. Market Sentiment Distribution Visual
    st.markdown("### 🧭 Cross-Market Sentiment Thermometer")
    gauge_col1, gauge_col2 = st.columns(2)
    with gauge_col1:
        hk_gauge = get_sentiment_gauge(hk_sentiment if not hk_df.empty else 0.0, "HKEX Retail Sentiment Meter")
        if hk_gauge:
            st.plotly_chart(hk_gauge, use_container_width=True)
    with gauge_col2:
        kr_gauge = get_sentiment_gauge(kr_sentiment if not kr_df.empty else 0.0, "KRX Retail Sentiment Meter")
        if kr_gauge:
            st.plotly_chart(kr_gauge, use_container_width=True)
