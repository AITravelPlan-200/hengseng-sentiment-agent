"""
Data Cleaning and Standardization Pipeline.
Enforces schema compliance, UTC ISO-8601 timestamp normalization,
deduplication, spam/duplicate filtration, and snapshot persistence.
"""

import re
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger("hengseng.cleaner")

REQUIRED_SOCIAL_FIELDS = [
    "exchange",
    "ticker",
    "asset_name",
    "source",
    "post_id",
    "timestamp_utc",
    "text",
    "likes",
    "reposts",
    "views",
    "author_followers"
]

class DataCleaner:
    """Cleans, normalizes, and validates social posts and financial data."""

    @staticmethod
    def normalize_timestamp(raw_ts: Any) -> str:
        """
        Normalize any timestamp (Unix int/float, string, or datetime) to ISO-8601 UTC.
        """
        if raw_ts is None:
            return datetime.now(timezone.utc).isoformat()

        try:
            # Handle numeric timestamp (seconds or milliseconds)
            if isinstance(raw_ts, (int, float)):
                if raw_ts > 1e11:  # Milliseconds
                    raw_ts = raw_ts / 1000.0
                dt = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
                return dt.isoformat()

            # Handle datetime object
            if isinstance(raw_ts, datetime):
                if raw_ts.tzinfo is None:
                    dt = raw_ts.replace(tzinfo=timezone.utc)
                else:
                    dt = raw_ts.astimezone(timezone.utc)
                return dt.isoformat()

            # Handle string timestamp
            if isinstance(raw_ts, str):
                raw_str = raw_ts.strip()
                # Parse with pandas for robust multi-format support
                parsed = pd.to_datetime(raw_str, utc=True)
                return parsed.isoformat()

        except Exception as e:
            logger.debug(f"Timestamp normalization fallback for '{raw_ts}': {e}")
            return datetime.now(timezone.utc).isoformat()

        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def clean_text(text: str) -> str:
        """Strip junk HTML tags, URLs, and excessive whitespace from text."""
        if not text:
            return ""
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", " ", str(text))
        # Remove URLs
        clean = re.sub(r"http[s]?://\S+", "", clean)
        # Normalize whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @staticmethod
    def generate_post_hash(text: str, source: str) -> str:
        """Generate SHA-256 fingerprint for deduplication."""
        payload = f"{source}:{text.strip().lower()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @classmethod
    def clean_and_standardize_social_post(
        cls,
        raw_post: Dict[str, Any],
        ticker: str,
        asset_name: str,
        exchange: str,
        source: str
    ) -> Optional[Dict[str, Any]]:
        """
        Transforms any raw social post into the mandatory 11-field schema.
        Returns None if text is empty or post is invalid.
        """
        text = raw_post.get("text") or raw_post.get("content") or raw_post.get("title") or ""
        cleaned_text = cls.clean_text(text)
        if len(cleaned_text) < 4:
            return None

        # Determine unique post_id
        post_id = str(raw_post.get("id") or raw_post.get("post_id") or raw_post.get("url") or "")
        if not post_id:
            post_id = cls.generate_post_hash(cleaned_text, source)

        # Timestamps
        raw_ts = (
            raw_post.get("timestamp") or
            raw_post.get("timestamp_utc") or
            raw_post.get("createdAt") or
            raw_post.get("created_at") or
            raw_post.get("date")
        )
        timestamp_utc = cls.normalize_timestamp(raw_ts)

        # Engagement metrics (safely cast to int >= 0)
        def to_int(val: Any) -> int:
            try:
                if val is None:
                    return 0
                return max(0, int(float(val)))
            except (ValueError, TypeError):
                return 0

        likes = to_int(raw_post.get("likes") or raw_post.get("likeCount") or raw_post.get("upvotes"))
        reposts = to_int(raw_post.get("reposts") or raw_post.get("shares") or raw_post.get("retweetCount"))
        views = to_int(raw_post.get("views") or raw_post.get("viewCount"))
        author_followers = to_int(raw_post.get("author_followers") or raw_post.get("userFollowers") or raw_post.get("followers"))

        standardized = {
            "exchange": str(exchange).upper(),
            "ticker": str(ticker).upper(),
            "asset_name": str(asset_name),
            "source": str(source),
            "post_id": str(post_id),
            "timestamp_utc": timestamp_utc,
            "text": cleaned_text,
            "likes": likes,
            "reposts": reposts,
            "views": views,
            "author_followers": author_followers
        }
        return standardized

    @classmethod
    def process_and_deduplicate_posts(
        cls,
        raw_posts: List[Dict[str, Any]],
        ticker: str,
        asset_name: str,
        exchange: str,
        source: str
    ) -> List[Dict[str, Any]]:
        """
        Cleans a list of posts, drops duplicates by post_id and text hash,
        and saves snapshots.
        """
        cleaned_list: List[Dict[str, Any]] = []
        seen_ids = set()
        seen_hashes = set()

        for raw in raw_posts:
            post = cls.clean_and_standardize_social_post(
                raw_post=raw,
                ticker=ticker,
                asset_name=asset_name,
                exchange=exchange,
                source=source
            )
            if not post:
                continue

            # Deduplication
            pid = post["post_id"]
            thash = cls.generate_post_hash(post["text"], source)

            if pid in seen_ids or thash in seen_hashes:
                continue

            seen_ids.add(pid)
            seen_hashes.add(thash)
            cleaned_list.append(post)

        return cleaned_list

    @classmethod
    def save_raw_snapshot(cls, items: List[Dict[str, Any]], source: str, ticker: str) -> Path:
        """Persist raw crawl dump to data/raw/."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_ticker = ticker.replace(".", "_")
        filename = f"{source}_{safe_ticker}_{timestamp}.json"
        filepath = RAW_DATA_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logger.info(f"Raw snapshot saved ({len(items)} items): {filepath}")
        return filepath

    @classmethod
    def save_processed_snapshot(cls, df: pd.DataFrame, prefix: str) -> Path:
        """Persist processed dataset to data/processed/."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = PROCESSED_DATA_DIR / f"{prefix}_{timestamp}.parquet"
        df.to_parquet(filepath, index=False)
        logger.info(f"Processed snapshot saved ({len(df)} rows): {filepath}")
        return filepath
