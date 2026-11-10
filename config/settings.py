"""
Application Configuration and Settings.
Loads configuration from environment variables, .env file, or Streamlit secrets.
Centralizes paths, thresholds, and operational parameters.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base directory is the repository root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
load_dotenv(BASE_DIR / ".env")

def get_config_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve configuration value from os.environ, falling back to Streamlit secrets if running in Streamlit.
    """
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default

# -------------------------------------------------------------
# API Credentials
# -------------------------------------------------------------
APIFY_API_TOKEN = get_config_var("APIFY_API_TOKEN", "")
EXA_API_KEY = get_config_var("EXA_API_KEY", "")
TAVILY_API_KEY = get_config_var("TAVILY_API_KEY", "")
HUGGINGFACE_API_KEY = get_config_var("HUGGINGFACE_API_KEY", "")

# -------------------------------------------------------------
# File & Directory Paths
# -------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "hengseng.db"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
REPORTS_DIR = BASE_DIR / "reports" / "generated"

# Ensure runtime directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, DATABASE_DIR, MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Operational Parameters
# -------------------------------------------------------------
SOCIAL_HISTORY_DAYS = int(get_config_var("MAX_DAYS_HISTORY", "10"))
NEWS_LOOKBACK_DAYS = int(get_config_var("NEWS_LOOKBACK_DAYS", "7"))
PRICE_HISTORY_YEARS = int(get_config_var("PRICE_HISTORY_YEARS", "2"))
USE_SAMPLE_FALLBACK = get_config_var("USE_SAMPLE_FALLBACK", "true").lower() in ("true", "1", "yes")

# Request timeouts and retries
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5

# Sentiment thresholds
BULLISH_THRESHOLD = 0.15
BEARISH_THRESHOLD = -0.15

# Machine Learning Parameters
TEST_SPLIT_DAYS = 60
VALIDATION_SPLIT_DAYS = 60
RANDOM_STATE = 42

# Logging configuration
LOG_LEVEL = get_config_var("LOG_LEVEL", "INFO")
