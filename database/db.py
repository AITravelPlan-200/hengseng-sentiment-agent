"""
Database connection and operations layer.
Provides clean context-managed SQLite operations, pandas integrations, and schema migrations.
"""

import sqlite3
import logging
from typing import Optional, List, Dict, Any
import pandas as pd
from config.settings import DATABASE_PATH
from database.schema import SCHEMA_SQL

logger = logging.getLogger("hengseng.db")

class DatabaseManager:
    """Manages SQLite database connections and executes queries."""

    def __init__(self, db_path=None):
        self.db_path = str(db_path or DATABASE_PATH)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a new SQLite connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initialize database tables and indexes from schema."""
        try:
            with self.get_connection() as conn:
                conn.executescript(SCHEMA_SQL)
                conn.commit()
            logger.info(f"Database initialized successfully at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}", exc_info=True)
            raise

    def execute_query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return rows."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchall()

    def execute_non_query(self, sql: str, params: tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE query and return affected rows."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor.rowcount

    def insert_batch(self, sql: str, param_list: List[tuple]) -> int:
        """Execute a batch of statements with executemany."""
        if not param_list:
            return 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, param_list)
            conn.commit()
            return cursor.rowcount

    def query_df(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a query and return results as a Pandas DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    # -------------------------------------------------------------------------
    # Specialized Domain Methods
    # -------------------------------------------------------------------------
    def save_social_posts(self, posts: List[Dict[str, Any]]) -> int:
        """Upsert social media posts into the database."""
        if not posts:
            return 0
        import json
        normalized_posts = []
        for p in posts:
            item = dict(p)
            item.setdefault("raw_json", json.dumps(item, ensure_ascii=False))
            item.setdefault("is_spam_or_dup", 0)
            item.setdefault("likes", 0)
            item.setdefault("reposts", 0)
            item.setdefault("views", 0)
            item.setdefault("author_followers", 0)
            item.setdefault("positive_prob", 0.0)
            item.setdefault("negative_prob", 0.0)
            item.setdefault("neutral_prob", 1.0)
            item.setdefault("sentiment_score", 0.0)
            item.setdefault("sentiment_label", "Neutral")
            item.setdefault("confidence_score", 1.0)
            item.setdefault("weighted_sentiment", 0.0)
            normalized_posts.append(item)

        sql = """
        INSERT INTO social_posts (
            exchange, ticker, asset_name, source, post_id, timestamp_utc,
            text, likes, reposts, views, author_followers,
            positive_prob, negative_prob, neutral_prob,
            sentiment_score, sentiment_label, confidence_score,
            weighted_sentiment, is_spam_or_dup, raw_json
        ) VALUES (
            :exchange, :ticker, :asset_name, :source, :post_id, :timestamp_utc,
            :text, :likes, :reposts, :views, :author_followers,
            :positive_prob, :negative_prob, :neutral_prob,
            :sentiment_score, :sentiment_label, :confidence_score,
            :weighted_sentiment, :is_spam_or_dup, :raw_json
        ) ON CONFLICT(source, post_id, ticker) DO UPDATE SET
            likes=excluded.likes,
            reposts=excluded.reposts,
            views=excluded.views,
            sentiment_score=excluded.sentiment_score,
            weighted_sentiment=excluded.weighted_sentiment;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, normalized_posts)
            conn.commit()
            return cursor.rowcount

    def save_news_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Upsert financial news articles."""
        if not articles:
            return 0
        import json
        normalized_articles = []
        for a in articles:
            item = dict(a)
            item.setdefault("raw_json", json.dumps(item, ensure_ascii=False))
            item.setdefault("source", "Financial News")
            item.setdefault("summary", "")
            item.setdefault("content", "")
            item.setdefault("positive_score", 0.0)
            item.setdefault("negative_score", 0.0)
            item.setdefault("neutral_score", 1.0)
            item.setdefault("sentiment_score", 0.0)
            item.setdefault("confidence_score", 1.0)
            item.setdefault("sentiment_label", "Neutral")
            item.setdefault("recency_weight", 1.0)
            normalized_articles.append(item)

        sql = """
        INSERT INTO financial_news (
            exchange, ticker, title, url, published_utc,
            source, summary, content, positive_score,
            negative_score, neutral_score, sentiment_score,
            confidence_score, sentiment_label, recency_weight, raw_json
        ) VALUES (
            :exchange, :ticker, :title, :url, :published_utc,
            :source, :summary, :content, :positive_score,
            :negative_score, :neutral_score, :sentiment_score,
            :confidence_score, :sentiment_label, :recency_weight, :raw_json
        ) ON CONFLICT(ticker, url) DO UPDATE SET
            sentiment_score=excluded.sentiment_score,
            recency_weight=excluded.recency_weight;
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, normalized_articles)
            conn.commit()
            return cursor.rowcount

    def save_market_data(self, df: pd.DataFrame, ticker: str) -> int:
        """Save OHLCV and technical indicators DataFrame to database."""
        if df.empty:
            return 0
        
        # Ensure standard column names
        data = df.copy()
        if "ticker" not in data.columns:
            data["ticker"] = ticker
            
        with self.get_connection() as conn:
            # Upsert via temporary staging table
            data.to_sql("temp_market_data", conn, if_exists="replace", index=False)
            upsert_sql = """
            INSERT OR REPLACE INTO market_data (
                ticker, timestamp_utc, open, high, low, close, adj_close, volume,
                sma_20, sma_50, ema_20, rsi_14, macd, macd_signal, macd_hist,
                bb_upper, bb_lower, bb_middle, volatility_20, daily_returns, volume_change
            )
            SELECT 
                ticker, timestamp_utc, open, high, low, close, adj_close, volume,
                sma_20, sma_50, ema_20, rsi_14, macd, macd_signal, macd_hist,
                bb_upper, bb_lower, bb_middle, volatility_20, daily_returns, volume_change
            FROM temp_market_data;
            """
            cursor = conn.cursor()
            cursor.execute(upsert_sql)
            cursor.execute("DROP TABLE IF EXISTS temp_market_data;")
            conn.commit()
            return cursor.rowcount

    def get_latest_market_data(self, ticker: str, limit: int = 100) -> pd.DataFrame:
        """Fetch latest OHLCV and indicators for ticker ordered chronologically."""
        sql = """
        SELECT * FROM market_data
        WHERE ticker = ?
        ORDER BY timestamp_utc DESC
        LIMIT ?
        """
        df = self.query_df(sql, (ticker, limit))
        if not df.empty:
            df = df.sort_values("timestamp_utc").reset_index(drop=True)
        return df

    def get_social_posts(self, ticker: Optional[str] = None, exchange: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
        """Fetch social posts filtered by ticker or exchange."""
        conditions = []
        params = []
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        if exchange:
            conditions.append("exchange = ?")
            params.append(exchange)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
        SELECT * FROM social_posts
        {where_clause}
        ORDER BY timestamp_utc DESC
        LIMIT ?
        """
        params.append(limit)
        return self.query_df(sql, tuple(params))

    def get_news(self, ticker: Optional[str] = None, exchange: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
        """Fetch news articles filtered by ticker or exchange."""
        conditions = []
        params = []
        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)
        if exchange:
            conditions.append("exchange = ?")
            params.append(exchange)
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
        SELECT * FROM financial_news
        {where_clause}
        ORDER BY published_utc DESC
        LIMIT ?
        """
        params.append(limit)
        return self.query_df(sql, tuple(params))

# Global DB instance
db = DatabaseManager()
