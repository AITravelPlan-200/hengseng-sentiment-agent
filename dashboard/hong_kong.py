"""
Page 2: Hong Kong Stock Market (HKEX) Intelligence Dashboard.
Presents detailed analytics for the 5 target Hong Kong equities:
Tencent (0700.HK), Alibaba (9988.HK), Xiaomi (1810.HK), Meituan (3690.HK), HSBC (0005.HK).
"""

import streamlit as st
import pandas as pd
from dashboard.chart_utils import COLOR_BULLISH, COLOR_BEARISH, COLOR_NEUTRAL

def render_hong_kong_page(pipeline_data: dict):
    """Renders Page 2: Hong Kong Market."""
    st.markdown("## 🇭🇰 Hong Kong Stock Exchange (HKEX) Intelligence")
    st.markdown("Quantitative tracking of retail discussions (LIHKG, Telegram), financial news, and next-day predictions.")

    stock_analyses = [s for s in pipeline_data.get("stock_analyses", []) if s.get("exchange") == "HKEX"]
    if not stock_analyses:
        st.warning("No HKEX stock analyses available.")
        return

    df = pd.DataFrame(stock_analyses)

    # 1. League Table Overview
    st.markdown("### 🏆 HKEX Asset League Table")
    disp_df = df[[
        "ticker", "company_name", "current_price", "price_change_pct",
        "social_mention_count", "social_sentiment_score", "news_sentiment_score",
        "combined_sentiment_score", "predicted_next_period_price",
        "predicted_price_movement_pct", "suggested_position"
    ]].copy()
    disp_df.columns = [
        "Ticker", "Asset", "Price (HKD)", "24h Chg %", "Mentions",
        "Social Sent", "News Sent", "Combined Sent", "Pred Close", "Pred %", "Position"
    ]
    st.dataframe(disp_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Rankings Comparison
    col_rank1, col_rank2 = st.columns(2)
    with col_rank1:
        st.markdown("#### 🗣️ Social Retail Mention Ranking")
        rank_mentions = df.sort_values(by="social_mention_count", ascending=False)[[
            "ticker", "company_name", "social_mention_count"
        ]].reset_index(drop=True)
        rank_mentions.columns = ["Ticker", "Company", "Total Mentions"]
        st.dataframe(rank_mentions, use_container_width=True)

    with col_rank2:
        st.markdown("#### 📈 Composite Sentiment Ranking")
        rank_sent = df.sort_values(by="combined_sentiment_score", ascending=False)[[
            "ticker", "company_name", "combined_sentiment_score", "sentiment_direction"
        ]].reset_index(drop=True)
        rank_sent.columns = ["Ticker", "Company", "Sentiment Score", "Direction"]
        st.dataframe(rank_sent, use_container_width=True)

    st.markdown("---")

    # 3. Individual Stock Cards
    st.markdown("### 🏢 Individual Asset Intelligence Cards")
    for s in stock_analyses:
        ticker = s.get("ticker")
        name = s.get("company_name")
        curr_p = s.get("current_price", 0.0)
        chg_p = s.get("price_change_pct", 0.0)
        sent_dir = s.get("sentiment_direction", "Neutral")
        pos = s.get("suggested_position", "Neutral")
        pred_p = s.get("predicted_next_period_price", curr_p)
        pred_chg = s.get("predicted_price_movement_pct", 0.0)
        supp = s.get("support_level", 0.0)
        res = s.get("resistance_level", 0.0)

        with st.expander(f"**{ticker} - {name}** | Last: HKD {curr_p:.2f} ({chg_p:+.2f}%) | Signal: {pos}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predicted Next Close", f"HKD {pred_p:.2f}", f"{pred_chg:+.2f}%")
            c2.metric("Support / Resistance", f"{supp:.2f} / {res:.2f}")
            c3.metric("Social Mentions", f"{s.get('social_mention_count', 0)}")
            c4.metric("Sentiment Score", f"{s.get('combined_sentiment_score', 0):+.2f}", sent_dir)

            st.markdown(f"**Technical Summary:** `{s.get('technical_indicator_summary', 'Neutral')}`")
            st.markdown(f"**Position Rationale:** {s.get('position_reasoning')}")
