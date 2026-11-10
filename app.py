"""
HengSeng Sentiment Price Prediction Agent.
Main Streamlit Application and Dashboard Interface.

Professional AI stock sentiment and next-period price prediction platform
covering Hong Kong Stock Exchange (HKEX) and Korea Exchange (KRX).
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from config.settings import APIFY_API_TOKEN, EXA_API_KEY, TAVILY_API_KEY, DATABASE_PATH
from config.tickers import ASSET_UNIVERSE
from agents.orchestrator import orchestrator
from database.db import db
from dashboard.overview import render_overview_page
from dashboard.hong_kong import render_hong_kong_page
from dashboard.south_korea import render_south_korea_page
from dashboard.social import render_social_page
from dashboard.news import render_news_page
from dashboard.predictions import render_predictions_page
from dashboard.deep_dive import render_deep_dive_page
from dashboard.performance import render_performance_page

# Set page configuration with dark terminal aesthetic
st.set_page_config(
    page_title="HengSeng Sentiment Price Prediction Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Financial Terminal CSS Styling
TERMINAL_CSS = """
<style>
    /* Dark Theme Background and Fonts */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    header, .stHeader {
        background-color: #0b0f19 !important;
    }
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 700;
        font-size: 26px;
        color: #f8fafc;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 13px;
        font-weight: 600;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid #1e293b;
    }
    /* Buttons */
    .stButton > button {
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #38bdf8;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #38bdf8;
        color: #0b0f19;
        border: 1px solid #38bdf8;
    }
    /* Dataframe tables */
    .dataframe {
        border-radius: 6px;
        font-size: 13px;
    }
</style>
"""
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_cached_pipeline_state():
    """Retrieve preloaded pipeline state or load latest data from SQLite."""
    return orchestrator.run_full_pipeline()

def ensure_database_ready():
    """Ensure SQLite database has initial baseline data on fresh deployment."""
    try:
        rows = db.execute_query("SELECT COUNT(*) as cnt FROM social_posts;")
        cnt = rows[0]["cnt"] if rows else 0
        if cnt == 0:
            from data.seed_data import seed_sample_data
            seed_sample_data()
    except Exception:
        pass

ensure_database_ready()

def main():
    # Sidebar Header
    st.sidebar.markdown("## 📈 HengSeng Agent")
    st.sidebar.caption("Institutional Quantitative Trading Platform")

    # Pipeline Control in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Engine Execution")

    if "pipeline_data" not in st.session_state:
        # Pre-populate on first start
        with st.spinner("Initializing HengSeng Multi-Agent Intelligence..."):
            st.session_state["pipeline_data"] = orchestrator.run_full_pipeline()

    if st.sidebar.button("🔄 Run Full Pipeline", use_container_width=True):
        progress_bar = st.sidebar.progress(0.0)
        status_text = st.sidebar.empty()

        def update_progress(msg: str, frac: float):
            progress_bar.progress(frac)
            status_text.text(msg)

        with st.spinner("Executing Multi-Agent Cycle..."):
            st.session_state["pipeline_data"] = orchestrator.run_full_pipeline(
                progress_callback=update_progress,
                force_market_refresh=True
            )
        st.sidebar.success("Pipeline refreshed successfully!")

    # API Status Diagnostic Section in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔌 API Integration Status")

    has_apify = bool(APIFY_API_TOKEN and "your_" not in APIFY_API_TOKEN)
    has_exa = bool(EXA_API_KEY and "your_" not in EXA_API_KEY)
    has_tavily = bool(TAVILY_API_KEY and "your_" not in TAVILY_API_KEY)

    st.sidebar.markdown(
        f"- **Apify (Social):** {'🟢 Active' if has_apify else '🟡 Direct / Cached'}\n"
        f"- **Exa (News):** {'🟢 Active' if has_exa else '🟡 Direct / Cached'}\n"
        f"- **Tavily (Search):** {'🟢 Active' if has_tavily else '⚪ Not Configured'}\n"
        f"- **Market Feed:** 🟢 Active (REST & yfinance)\n"
        f"- **Local DB:** 🟢 Connected (`hengseng.db`)"
    )

    # Multi-Page Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📑 Terminal Navigation")

    pages = [
        "1. Executive Overview",
        "2. Hong Kong Market",
        "3. South Korea Market",
        "4. Social Intelligence",
        "5. News Intelligence",
        "6. Price Predictions",
        "7. Stock Deep Dive",
        "8. Model Performance"
    ]

    selected_page = st.sidebar.radio("Navigate to Page:", pages)

    # Report Artifact Download Section in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Export Reports")
    pipeline_data = st.session_state.get("pipeline_data", {})
    exported = pipeline_data.get("exported_files", {})

    hk_csv = exported.get("hk_csv")
    kr_csv = exported.get("kr_csv")
    html_report = exported.get("html_report")

    if hk_csv and os.path.exists(hk_csv):
        with open(hk_csv, "rb") as f:
            st.sidebar.download_button(
                label="📥 Export HKEX Rankings (CSV)",
                data=f,
                file_name="HKEX_Stock_Rankings.csv",
                mime="text/csv",
                use_container_width=True
            )

    if kr_csv and os.path.exists(kr_csv):
        with open(kr_csv, "rb") as f:
            st.sidebar.download_button(
                label="📥 Export KRX Rankings (CSV)",
                data=f,
                file_name="KRX_Stock_Rankings.csv",
                mime="text/csv",
                use_container_width=True
            )

    if html_report and os.path.exists(html_report):
        with open(html_report, "rb") as f:
            st.sidebar.download_button(
                label="📄 Export Briefing Report (HTML)",
                data=f,
                file_name="HengSeng_Executive_Briefing.html",
                mime="text/html",
                use_container_width=True
            )

    # Route Page
    if selected_page == "1. Executive Overview":
        render_overview_page(pipeline_data)
    elif selected_page == "2. Hong Kong Market":
        render_hong_kong_page(pipeline_data)
    elif selected_page == "3. South Korea Market":
        render_south_korea_page(pipeline_data)
    elif selected_page == "4. Social Intelligence":
        render_social_page(pipeline_data)
    elif selected_page == "5. News Intelligence":
        render_news_page(pipeline_data)
    elif selected_page == "6. Price Predictions":
        render_predictions_page(pipeline_data)
    elif selected_page == "7. Stock Deep Dive":
        render_deep_dive_page(pipeline_data)
    elif selected_page == "8. Model Performance":
        render_performance_page(pipeline_data)

if __name__ == "__main__":
    main()
