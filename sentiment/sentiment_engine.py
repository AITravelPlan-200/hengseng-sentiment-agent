"""
Sentiment Engine with Engagement-Weighting and Spam Shielding.
Calculates granular sentiment probabilities, directional scores, and engagement/recency weights.

Weighting Formula Documentation:
--------------------------------
1. Raw Sentiment Score:
   Score = P(Positive) - P(Negative)  ∈ [-1.0, +1.0]

2. Engagement Weight (W_eng):
   Accounts for organic viral amplification and social proof without allowing exponential runaway.
   Metric Score M = likes*2.0 + reposts*3.0 + views*0.02 + author_followers*0.001
   W_eng = 1.0 + log10(1.0 + M)
   Clamped to [0.5, 3.5] to prevent bot manipulation.

3. Recency Weight (W_rec):
   Exponential time-decay with a half-life of 3 days (~72 hours):
   dt_days = max(0.0, (now_utc - post_utc).total_seconds() / 86400.0)
   W_rec = exp(-0.231 * dt_days)  ∈ [0.10, 1.0]

4. Spam / Duplicate Shield:
   If post matches duplicate fingerprint or repetitive spam patterns:
   W_spam = 0.05
   Else:
   W_spam = 1.0

5. Final Weighted Sentiment:
   Weighted_Sentiment = Score * W_eng * W_rec * W_spam
"""

import math
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
from sentiment.multilingual import MultilingualSentimentClassifier

logger = logging.getLogger("hengseng.sentiment.engine")

class SentimentEngine:
    """Processes textual content and generates weighted sentiment metrics."""

    def __init__(self, classifier: Optional[MultilingualSentimentClassifier] = None):
        self.classifier = classifier or MultilingualSentimentClassifier()

    def process_item(
        self,
        text: str,
        timestamp_utc: str,
        likes: int = 0,
        reposts: int = 0,
        views: int = 0,
        followers: int = 0,
        is_spam: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates sentiment probabilities and engagement-weighted score for an item.
        """
        # Run multilingual sentiment classifier
        res = self.classifier.analyze_sentiment(text)
        score = res["sentiment_score"]

        # Compute engagement weight
        w_eng = self.calculate_engagement_weight(likes, reposts, views, followers)

        # Compute recency weight
        w_rec = self.calculate_recency_weight(timestamp_utc)

        # Spam damping factor
        w_spam = 0.05 if is_spam else 1.0

        # Weighted sentiment score
        weighted_score = round(score * w_eng * w_rec * w_spam, 4)

        return {
            "positive_prob": res["positive_prob"],
            "negative_prob": res["negative_prob"],
            "neutral_prob": res["neutral_prob"],
            "sentiment_score": score,
            "sentiment_label": res["sentiment_label"],
            "confidence_score": res["confidence_score"],
            "detected_language": res["detected_language"],
            "engagement_weight": round(w_eng, 4),
            "recency_weight": round(w_rec, 4),
            "weighted_sentiment": weighted_score,
            "is_spam_or_dup": 1 if is_spam else 0
        }

    @staticmethod
    def calculate_engagement_weight(likes: int, reposts: int, views: int, followers: int) -> float:
        """
        Logarithmically scales social engagement to prevent outliers from dominating.
        """
        m = max(0, likes * 2.0 + reposts * 3.0 + views * 0.02 + followers * 0.001)
        raw_weight = 1.0 + math.log10(1.0 + m)
        return min(3.5, max(0.5, raw_weight))

    @staticmethod
    def calculate_recency_weight(timestamp_utc: str, half_life_days: float = 3.0) -> float:
        """
        Computes exponential decay based on elapsed days from event timestamp.
        """
        try:
            now = datetime.now(timezone.utc)
            event_dt = pd.to_datetime(timestamp_utc, utc=True).to_pydatetime()
            delta_days = max(0.0, (now - event_dt).total_seconds() / 86400.0)
            decay_rate = math.log(2.0) / max(0.5, half_life_days)
            recency = math.exp(-decay_rate * delta_days)
            return min(1.0, max(0.10, recency))
        except Exception:
            return 1.0

    def enrich_social_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich a batch of standardized social posts with sentiment scores."""
        enriched = []
        for p in posts:
            item_sentiment = self.process_item(
                text=p.get("text", ""),
                timestamp_utc=p.get("timestamp_utc", ""),
                likes=p.get("likes", 0),
                reposts=p.get("reposts", 0),
                views=p.get("views", 0),
                followers=p.get("author_followers", 0),
                is_spam=bool(p.get("is_spam_or_dup", 0))
            )
            # Combine post fields with sentiment analysis
            combined = {**p, **item_sentiment}
            enriched.append(combined)
        return enriched

    def enrich_news_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich news articles with sentiment scores."""
        enriched = []
        for art in articles:
            text = f"{art.get('title', '')}. {art.get('summary', '')}"
            res = self.process_item(
                text=text,
                timestamp_utc=art.get("published_utc", ""),
                likes=50,  # Baseline institutional visibility weight
                reposts=20,
                views=500,
                followers=5000,
                is_spam=False
            )
            art_enriched = {
                **art,
                "positive_score": res["positive_prob"],
                "negative_score": res["negative_prob"],
                "neutral_score": res["neutral_prob"],
                "sentiment_score": res["sentiment_score"],
                "confidence_score": res["confidence_score"],
                "sentiment_label": res["sentiment_label"],
                "recency_weight": res["recency_weight"],
                "weighted_sentiment": res["weighted_sentiment"]
            }
            enriched.append(art_enriched)
        return enriched
