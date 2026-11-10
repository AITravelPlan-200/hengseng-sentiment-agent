"""
Page 5: News Intelligence Dashboard.
Presents high-signal institutional financial news ingested via Exa API and Tavily,
with clickable links, publication timestamps, and recency-weighted sentiment scoring.
"""

import streamlit as st
import pandas as pd
from dashboard.chart_utils import COLOR_BULLISH, COLOR_BEARISH, COLOR_NEUTRAL

def render_news_page(pipeline_data: dict):
    """Renders Page 5: News Intelligence."""
    st.markdown("## 📰 Financial News Intelligence (Exa API)")
    st.markdown("Live and curated corporate developments, earnings releases, and regulatory events.")

    news_df = pipeline_data.get("news_df", pd.DataFrame())
    if news_df.empty:
        st.info("No financial news articles in database. Run the intelligence pipeline to ingest news.")
        return

    # Filter by Ticker
    tickers = ["ALL"] + sorted(news_df["ticker"].unique().tolist())
    selected_ticker = st.selectbox("Filter News by Ticker", tickers)

    filtered_news = news_df if selected_ticker == "ALL" else news_df[news_df["ticker"] == selected_ticker]

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    total_arts = len(filtered_news)
    avg_score = float(filtered_news["sentiment_score"].mean()) if "sentiment_score" in filtered_news.columns else 0.0
    pos_count = int((filtered_news["sentiment_score"] > 0.15).sum()) if "sentiment_score" in filtered_news.columns else 0
    neg_count = int((filtered_news["sentiment_score"] < -0.15).sum()) if "sentiment_score" in filtered_news.columns else 0

    c1.metric("Total Articles", f"{total_arts}")
    c2.metric("Mean News Sentiment", f"{avg_score:+.2f}")
    c3.metric("Bullish Articles", f"{pos_count}")
    c4.metric("Bearish Articles", f"{neg_count}")

    st.markdown("---")

    # Positive and Negative News Summaries
    sum_col1, sum_col2 = st.columns(2)
    with sum_col1:
        st.markdown("#### 🟢 Key Positive Developments")
        pos_articles = filtered_news[filtered_news["sentiment_score"] > 0.10].head(4)
        if not pos_articles.empty:
            for _, a in pos_articles.iterrows():
                st.markdown(
                    f"- **[{a.get('ticker')}]** [{a.get('title')}]({a.get('url')})  \n"
                    f"  <small style='color:#94a3b8;'>Source: {a.get('source')} | Published: {a.get('published_utc')[:10]}</small>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No strongly positive news articles in current window.")

    with sum_col2:
        st.markdown("#### 🔴 Key Cautionary / Negative Developments")
        neg_articles = filtered_news[filtered_news["sentiment_score"] < -0.10].head(4)
        if not neg_articles.empty:
            for _, a in neg_articles.iterrows():
                st.markdown(
                    f"- **[{a.get('ticker')}]** [{a.get('title')}]({a.get('url')})  \n"
                    f"  <small style='color:#94a3b8;'>Source: {a.get('source')} | Published: {a.get('published_utc')[:10]}</small>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No strongly negative news articles in current window.")

    st.markdown("---")

    # Complete News Feed with Clickable Links
    st.markdown("### 📋 Complete Financial News Stream")
    for _, a in filtered_news.iterrows():
        score = a.get("sentiment_score", 0.0)
        score_color = COLOR_BULLISH if score > 0.15 else (COLOR_BEARISH if score < -0.15 else COLOR_NEUTRAL)
        label = a.get("sentiment_label", "Neutral")

        st.markdown(
            f"""
            <div style="background:#161b22; border-left: 4px solid {score_color}; padding:14px; border-radius:6px; margin-bottom:12px;">
                <div style="font-size:15px; font-weight:600;">
                    <a href="{a.get('url')}" target="_blank" style="color:#00e5ff; text-decoration:none;">{a.get('title')}</a>
                </div>
                <div style="color:#94a3b8; font-size:12px; margin:4px 0 8px 0;">
                    <strong>{a.get('ticker')}</strong> &nbsp;|&nbsp; 
                    Source: {a.get('source', 'Exa News')} &nbsp;|&nbsp; 
                    Published: {a.get('published_utc')} &nbsp;|&nbsp; 
                    Sentiment: <span style="color:{score_color}; font-weight:bold;">{score:+.2f} ({label})</span>
                </div>
                <div style="color:#cbd5e1; font-size:13px; line-height:1.4;">
                    {a.get('summary', a.get('content', '')[:300])}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
