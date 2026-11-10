"""
Page 7: Stock Deep Dive Intelligence Dashboard.
Allows exhaustive drill-down on any of the 10 target HKEX / KRX stocks:
OHLCV Candlesticks, Pivot Points, Technicals, Sentiment Streams, and ML Forecasts.
"""

import streamlit as st
import pandas as pd
from config.tickers import get_asset_metadata
from dashboard.chart_utils import (
    get_candlestick_chart,
    get_sentiment_gauge,
    COLOR_BULLISH, COLOR_BEARISH, COLOR_NEUTRAL
)

def render_deep_dive_page(pipeline_data: dict):
    """Renders Page 7: Stock Deep Dive."""
    st.markdown("## 🔍 Stock Deep Dive: Multi-Signal Inspection")

    stock_analyses = pipeline_data.get("stock_analyses", [])
    market_data_dict = pipeline_data.get("market_data_dict", {})
    social_df = pipeline_data.get("social_df", pd.DataFrame())
    news_df = pipeline_data.get("news_df", pd.DataFrame())

    if not stock_analyses:
        st.warning("No stock analyses available. Run the pipeline to populate intelligence.")
        return

    # Ticker selector
    tickers = [s["ticker"] for s in stock_analyses]
    selected_ticker = st.selectbox("Select Asset to Inspect:", tickers, index=0)

    # Get stock analysis object
    stock_info = next((s for s in stock_analyses if s["ticker"] == selected_ticker), {})
    meta = get_asset_metadata(selected_ticker)

    # Header Card
    c_head1, c_head2, c_head3, c_head4 = st.columns(4)
    curr_p = stock_info.get("current_price", 0.0)
    chg_pct = stock_info.get("price_change_pct", 0.0)
    pred_p = stock_info.get("predicted_next_period_price", curr_p)
    pred_pct = stock_info.get("predicted_price_movement_pct", 0.0)
    pos = stock_info.get("suggested_position", "Neutral")
    pos_color = COLOR_BULLISH if pos == "Long" else (COLOR_BEARISH if pos == "Short" else COLOR_NEUTRAL)

    c_head1.metric(f"{selected_ticker} ({meta['name_short']})", f"{meta['currency']} {curr_p:,.2f}", f"{chg_pct:+.2f}%")
    c_head2.metric("ML Forecast Close", f"{meta['currency']} {pred_p:,.2f}", f"{pred_pct:+.2f}%")
    c_head3.metric("Support / Resistance", f"{stock_info.get('support_level', 0):,.2f} / {stock_info.get('resistance_level', 0):,.2f}")
    c_head4.markdown(
        f"<div style='background:#161b22; padding:10px; border-radius:6px; text-align:center; border:1px solid #30363d;'>"
        f"<div style='font-size:12px; color:#94a3b8;'>SIGNAL</div>"
        f"<div style='font-size:22px; font-weight:bold; color:{pos_color};'>{pos}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(f"**Rationale:** {stock_info.get('position_reasoning')}")
    st.markdown("---")

    # 1. Interactive Technical Candlestick Chart
    m_df = market_data_dict.get(selected_ticker, pd.DataFrame())
    if not m_df.empty:
        st.markdown("### 📊 OHLCV Technical Candlestick & Volatility Bands")
        fig_candle = get_candlestick_chart(m_df, selected_ticker)
        if fig_candle:
            st.plotly_chart(fig_candle, use_container_width=True)

        # Technical Key Metrics Row
        latest_row = m_df.iloc[-1]
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("RSI (14)", f"{latest_row.get('rsi_14', 50.0):.1f}")
        t2.metric("MACD", f"{latest_row.get('macd', 0.0):.2f}")
        t3.metric("SMA 20", f"{latest_row.get('sma_20', 0.0):.2f}")
        t4.metric("SMA 50", f"{latest_row.get('sma_50', 0.0):.2f}")
        t5.metric("Volatility (20d)", f"{latest_row.get('volatility_20', 0.0):.3f}")

    st.markdown("---")

    # 2. Sentiment Decomposition & Support/Resistance
    col_sent, col_sr = st.columns(2)

    with col_sent:
        st.markdown("### 🧭 Sentiment Breakdown")
        s1, s2 = st.columns(2)
        s1.metric("Social Sentiment", f"{stock_info.get('social_sentiment_score', 0):+.2f}", f"{stock_info.get('social_mention_count', 0)} posts")
        s2.metric("News Sentiment", f"{stock_info.get('news_sentiment_score', 0):+.2f}")
        
        sent_gauge = get_sentiment_gauge(stock_info.get("combined_sentiment_score", 0.0), f"{selected_ticker} Sentiment Gauge")
        if sent_gauge:
            st.plotly_chart(sent_gauge, use_container_width=True)

    with col_sr:
        st.markdown("### 🛡️ Support & Resistance Barriers")
        st.markdown(
            f"""
            - **Resistance 2 (R2):** `{meta['currency']} {stock_info.get('resistance_2', 0):,.2f}`  
            - **Primary Resistance (R1):** `{meta['currency']} {stock_info.get('resistance_level', 0):,.2f}`  
            - **Classic Pivot Point:** `{meta['currency']} {stock_info.get('pivot_point', 0):,.2f}`  
            - **Primary Support (S1):** `{meta['currency']} {stock_info.get('support_level', 0):,.2f}`  
            - **Support 2 (S2):** `{meta['currency']} {stock_info.get('support_2', 0):,.2f}`  
            - **20-Day Swing High:** `{meta['currency']} {stock_info.get('swing_high_20d', 0):,.2f}`  
            - **20-Day Swing Low:** `{meta['currency']} {stock_info.get('swing_low_20d', 0):,.2f}`  
            
            <small style='color:#94a3b8;'>Methodology: Classic Floor Pivot Points adjusted against 20-day price extrema.</small>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 3. Recent Feeds for Selected Stock
    feed_tab1, feed_tab2 = st.tabs(["💬 Recent Social Posts", "📰 Recent News Articles"])

    with feed_tab1:
        ticker_posts = social_df[social_df["ticker"] == selected_ticker].head(5) if not social_df.empty else pd.DataFrame()
        if not ticker_posts.empty:
            for _, p in ticker_posts.iterrows():
                st.markdown(
                    f"> **[{p.get('source')}]** Score: `{p.get('sentiment_score', 0):+.2f}` | Likes: {p.get('likes', 0)}  \n"
                    f"{p.get('text')}  \n"
                    f"<small style='color:#94a3b8;'>UTC Date: {p.get('timestamp_utc')}</small>",
                    unsafe_allow_html=True
                )
                st.markdown("---")
        else:
            st.info(f"No social media posts recorded for {selected_ticker}.")

    with feed_tab2:
        ticker_news = news_df[news_df["ticker"] == selected_ticker].head(5) if not news_df.empty else pd.DataFrame()
        if not ticker_news.empty:
            for _, n in ticker_news.iterrows():
                st.markdown(
                    f"- [{n.get('title')}]({n.get('url')})  \n"
                    f"  <small style='color:#94a3b8;'>Source: {n.get('source')} | Published: {n.get('published_utc')} | Sentiment: {n.get('sentiment_score', 0):+.2f}</small>",
                    unsafe_allow_html=True
                )
        else:
            st.info(f"No financial news articles recorded for {selected_ticker}.")

    # Disclaimer
    st.markdown("---")
    st.caption(stock_info.get("disclaimer", ""))
