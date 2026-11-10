"""
SQLite Database Schema definitions for HengSeng Sentiment Price Prediction Agent.
"""

SCHEMA_SQL = """
-- 1. Social Media Posts Table
CREATE TABLE IF NOT EXISTS social_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL,
    ticker TEXT NOT NULL,
    asset_name TEXT NOT NULL,
    source TEXT NOT NULL,
    post_id TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    text TEXT NOT NULL,
    likes INTEGER DEFAULT 0,
    reposts INTEGER DEFAULT 0,
    views INTEGER DEFAULT 0,
    author_followers INTEGER DEFAULT 0,
    positive_prob REAL DEFAULT 0.0,
    negative_prob REAL DEFAULT 0.0,
    neutral_prob REAL DEFAULT 0.0,
    sentiment_score REAL DEFAULT 0.0,
    sentiment_label TEXT DEFAULT 'Neutral',
    confidence_score REAL DEFAULT 0.0,
    weighted_sentiment REAL DEFAULT 0.0,
    is_spam_or_dup INTEGER DEFAULT 0,
    raw_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, post_id, ticker)
);

-- Indexing for fast retrieval by ticker and timestamp
CREATE INDEX IF NOT EXISTS idx_social_ticker_time ON social_posts(ticker, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_social_exchange ON social_posts(exchange);
CREATE INDEX IF NOT EXISTS idx_social_source ON social_posts(source);

-- 2. Financial News Table
CREATE TABLE IF NOT EXISTS financial_news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange TEXT NOT NULL,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    published_utc TEXT NOT NULL,
    source TEXT,
    summary TEXT,
    content TEXT,
    positive_score REAL DEFAULT 0.0,
    negative_score REAL DEFAULT 0.0,
    neutral_score REAL DEFAULT 0.0,
    sentiment_score REAL DEFAULT 0.0,
    confidence_score REAL DEFAULT 0.0,
    sentiment_label TEXT DEFAULT 'Neutral',
    recency_weight REAL DEFAULT 1.0,
    raw_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, url)
);

CREATE INDEX IF NOT EXISTS idx_news_ticker_time ON financial_news(ticker, published_utc);
CREATE INDEX IF NOT EXISTS idx_news_exchange ON financial_news(exchange);

-- 3. Historical Market Data Table
CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    adj_close REAL,
    volume REAL NOT NULL,
    sma_20 REAL,
    sma_50 REAL,
    ema_20 REAL,
    rsi_14 REAL,
    macd REAL,
    macd_signal REAL,
    macd_hist REAL,
    bb_upper REAL,
    bb_lower REAL,
    bb_middle REAL,
    volatility_20 REAL,
    daily_returns REAL,
    volume_change REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, timestamp_utc)
);

CREATE INDEX IF NOT EXISTS idx_market_ticker_time ON market_data(ticker, timestamp_utc);

-- 4. Next-Period Price Predictions Table
CREATE TABLE IF NOT EXISTS price_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    exchange TEXT NOT NULL,
    prediction_date TEXT NOT NULL,
    target_date TEXT NOT NULL,
    current_close REAL NOT NULL,
    predicted_close REAL NOT NULL,
    predicted_change_pct REAL NOT NULL,
    direction TEXT NOT NULL,
    lower_bound REAL,
    upper_bound REAL,
    confidence_score REAL,
    model_name TEXT NOT NULL,
    sentiment_score REAL,
    technical_score REAL,
    features_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pred_ticker_date ON price_predictions(ticker, prediction_date);

-- 5. Model Evaluation Logs Table
CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    ticker TEXT NOT NULL,
    evaluation_date TEXT NOT NULL,
    mae REAL,
    rmse REAL,
    mape REAL,
    r2 REAL,
    directional_accuracy REAL,
    is_sentiment_enhanced INTEGER DEFAULT 1,
    train_start TEXT,
    train_end TEXT,
    test_start TEXT,
    test_end TEXT,
    details_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
