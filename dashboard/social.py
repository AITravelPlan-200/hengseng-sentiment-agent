"""
Page 4: Social Intelligence Dashboard.
Explores multi-platform retail social discussions across LIHKG, Telegram,
Naver Finance, and Daum Finance.
"""

import streamlit as st
import pandas as pd
import numpy as np
from dashboard.chart_utils import PLOTLY_AVAILABLE, THEME_BG, COLOR_BULLISH, COLOR_BEARISH, COLOR_NEUTRAL

if PLOTLY_AVAILABLE:
    import plotly.express as px
    import plotly.graph_objects as go

def render_social_page(pipeline_data: dict):
    """Renders Page 4: Social Intelligence."""
    st.markdown("## 💬 Social Intelligence: Retail Sentiment & Engagement")
    st.markdown(
        "Granular inspection of retail investor conversations collected via Apify actors "
        "and public board parsers across Hong Kong and South Korea."
    )

    social_df = pipeline_data.get("social_df", pd.DataFrame())
    if social_df.empty:
        st.info("No social posts ingested. Run the orchestrator pipeline to collect social discussions.")
        return

    # 1. Platform Breakdown & Mentions
    col1, col2, col3, col4 = st.columns(4)
    total_posts = len(social_df)
    total_likes = int(social_df["likes"].sum()) if "likes" in social_df.columns else 0
    total_views = int(social_df["views"].sum()) if "views" in social_df.columns else 0
    avg_sent = float(social_df["sentiment_score"].mean()) if "sentiment_score" in social_df.columns else 0.0

    col1.metric("Total Extracted Posts", f"{total_posts:,}")
    col2.metric("Total Upvotes / Likes", f"{total_likes:,}")
    col3.metric("Accumulated Views", f"{total_views:,}")
    col4.metric("Mean Social Sentiment", f"{avg_sent:+.2f}")

    st.markdown("---")

    # 2. Platform Distribution & Source Sentiment Comparison
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 📱 Posts by Social Platform")
        source_counts = social_df["source"].value_counts().reset_index()
        source_counts.columns = ["Platform", "Post Count"]
        if PLOTLY_AVAILABLE:
            fig_pie = px.pie(
                source_counts, values="Post Count", names="Platform",
                template="plotly_dark", color_discrete_sequence=["#00e5ff", "#6366f1", "#f59e0b", "#10b981"]
            )
            fig_pie.update_layout(paper_bgcolor=THEME_BG, height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.dataframe(source_counts, use_container_width=True)

    with chart_col2:
        st.markdown("#### 📊 Sentiment Distribution Across Sources")
        source_sent = social_df.groupby("source")["sentiment_score"].mean().reset_index()
        source_sent.columns = ["Platform", "Mean Sentiment"]
        if PLOTLY_AVAILABLE:
            fig_bar = px.bar(
                source_sent, x="Platform", y="Mean Sentiment",
                template="plotly_dark", color="Mean Sentiment",
                color_continuous_scale=["#ff5252", "#90caf9", "#00e676"]
            )
            fig_bar.update_layout(paper_bgcolor=THEME_BG, plot_bgcolor=THEME_BG, height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.dataframe(source_sent, use_container_width=True)

    st.markdown("---")

    # 3. Most Discussed Stocks
    st.markdown("#### 🔥 Most Discussed Stocks")
    ticker_counts = social_df.groupby(["ticker", "asset_name", "exchange"]).agg(
        Mentions=("post_id", "count"),
        Mean_Sentiment=("sentiment_score", "mean"),
        Total_Likes=("likes", "sum"),
        Total_Views=("views", "sum")
    ).reset_index().sort_values(by="Mentions", ascending=False)
    ticker_counts["Mean_Sentiment"] = ticker_counts["Mean_Sentiment"].round(2)
    st.dataframe(ticker_counts, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 4. Outlier Posts: Most Bullish and Most Bearish
    tab_bullish, tab_bearish, tab_raw = st.tabs(["🚀 Top Bullish Posts", "🔻 Top Bearish Posts", "📜 Raw Stream"])

    with tab_bullish:
        top_bullish = social_df.sort_values(by="sentiment_score", ascending=False).head(5)
        for _, p in top_bullish.iterrows():
            st.markdown(
                f"> **[{p.get('source')}] {p.get('ticker')} ({p.get('asset_name')})** — "
                f"Score: `{p.get('sentiment_score', 0):+.2f}` | Likes: {p.get('likes', 0)} | Views: {p.get('views', 0)}\n\n"
                f"{p.get('text')}\n\n"
                f"<small style='color:#94a3b8;'>Timestamp: {p.get('timestamp_utc')}</small>",
                unsafe_allow_html=True
            )
            st.markdown("---")

    with tab_bearish:
        top_bearish = social_df.sort_values(by="sentiment_score", ascending=True).head(5)
        for _, p in top_bearish.iterrows():
            st.markdown(
                f"> **[{p.get('source')}] {p.get('ticker')} ({p.get('asset_name')})** — "
                f"Score: `{p.get('sentiment_score', 0):+.2f}` | Likes: {p.get('likes', 0)} | Views: {p.get('views', 0)}\n\n"
                f"{p.get('text')}\n\n"
                f"<small style='color:#94a3b8;'>Timestamp: {p.get('timestamp_utc')}</small>",
                unsafe_allow_html=True
            )
            st.markdown("---")

    with tab_raw:
        st.dataframe(social_df[[
            "exchange", "ticker", "source", "timestamp_utc", "text", "sentiment_score", "likes", "views"
        ]], use_container_width=True)
