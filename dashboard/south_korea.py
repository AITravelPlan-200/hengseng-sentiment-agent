"""
Page 3: South Korea Stock Market (KRX) Intelligence Dashboard.
Presents detailed analytics for the 5 target Korean equities:
Samsung Electronics (005930.KS), SK Hynix (000660.KS), Hyundai Motor (005380.KS),
LG Energy Solution (373220.KS), Kia Corporation (000270.KS).
"""

import streamlit as st
import pandas as pd

def render_south_korea_page(pipeline_data: dict):
    """Renders Page 3: South Korea Market."""
    st.markdown("## 🇰🇷 Korea Exchange (KRX) Intelligence")
    st.markdown("Quantitative tracking of Korean retail sentiment (Naver Finance, Daum Finance), news, and next-day price forecasts.")

    stock_analyses = [s for s in pipeline_data.get("stock_analyses", []) if s.get("exchange") == "KRX"]
    if not stock_analyses:
        st.warning("No KRX stock analyses available.")
        return

    df = pd.DataFrame(stock_analyses)

    # 1. League Table Overview
    st.markdown("### 🏆 KRX Asset League Table")
    disp_df = df[[
        "ticker", "company_name", "current_price", "price_change_pct",
        "social_mention_count", "social_sentiment_score", "news_sentiment_score",
        "combined_sentiment_score", "predicted_next_period_price",
        "predicted_price_movement_pct", "suggested_position"
    ]].copy()
    disp_df.columns = [
        "Ticker", "Asset", "Price (KRW)", "24h Chg %", "Mentions",
        "Social Sent", "News Sent", "Combined Sent", "Pred Close", "Pred %", "Position"
    ]
    st.dataframe(disp_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Rankings Comparison
    col_rank1, col_rank2 = st.columns(2)
    with col_rank1:
        st.markdown("#### 🗣️ Social Retail Mention Ranking (Naver & Daum)")
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

        with st.expander(f"**{ticker} - {name}** | Last: ₩ {curr_p:,.0f} ({chg_p:+.2f}%) | Signal: {pos}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predicted Next Close", f"₩ {pred_p:,.0f}", f"{pred_chg:+.2f}%")
            c2.metric("Support / Resistance", f"{supp:,.0f} / {res:,.0f}")
            c3.metric("Social Mentions", f"{s.get('social_mention_count', 0)}")
            c4.metric("Sentiment Score", f"{s.get('combined_sentiment_score', 0):+.2f}", sent_dir)

            st.markdown(f"**Technical Summary:** `{s.get('technical_indicator_summary', 'Neutral')}`")
            st.markdown(f"**Position Rationale:** {s.get('position_reasoning')}")
