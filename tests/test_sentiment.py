"""
Unit tests for Multilingual Financial Sentiment Analysis Engine.
Validates:
- Multilingual language detection (EN, ZH, KO)
- Probability distributions (positive_prob + negative_prob + neutral_prob == 1.0)
- Sentiment scores in [-1.0, 1.0] range
- Engagement weight calculations and log-capping
- Recency exponential decay
- Anti-spam / duplicate shielding
"""

import pytest
from sentiment.multilingual import MultilingualSentimentClassifier
from sentiment.sentiment_engine import SentimentEngine

@pytest.fixture
def classifier():
    return MultilingualSentimentClassifier()

@pytest.fixture
def engine(classifier):
    return SentimentEngine(classifier=classifier)

def test_language_detection(classifier):
    assert classifier.detect_language("Tencent Holdings surges on AI earnings growth") == "en"
    assert classifier.detect_language("阿里巴巴今日股價大漲，業績超預期") == "zh"
    assert classifier.detect_language("삼성전자 영업이익 급등 호실적 달성") == "ko"

def test_sentiment_probabilities_valid(classifier):
    texts = [
        "Tencent reports record high profits and massive revenue jump",
        "阿里巴巴業績超預期，雲計算暴漲突破牛市",
        "삼성전자 HBM 수주 대박 어닝서프라이즈 급등",
        "HSBC warns of severe risk and sudden drop in profit"
    ]
    for txt in texts:
        res = classifier.analyze_sentiment(txt)
        assert 0.0 <= res["positive_prob"] <= 1.0
        assert 0.0 <= res["negative_prob"] <= 1.0
        assert 0.0 <= res["neutral_prob"] <= 1.0
        prob_sum = res["positive_prob"] + res["negative_prob"] + res["neutral_prob"]
        assert pytest.approx(prob_sum, abs=1e-2) == 1.0
        assert -1.0 <= res["sentiment_score"] <= 1.0
        assert res["sentiment_label"] in ["Bullish", "Bearish", "Neutral"]
        assert 0.0 <= res["confidence_score"] <= 1.0

def test_sentiment_polarity_accuracy(classifier):
    bullish_en = classifier.analyze_sentiment("Tencent surges to record high with soaring revenue growth")
    assert bullish_en["sentiment_score"] > 0.15
    assert bullish_en["sentiment_label"] == "Bullish"

    bearish_en = classifier.analyze_sentiment("HSBC suffers catastrophic plunge, massive losses and regulatory fines")
    assert bearish_en["sentiment_score"] < -0.15
    assert bearish_en["sentiment_label"] == "Bearish"

    bullish_zh = classifier.analyze_sentiment("騰訊業績大漲暴增，利好消息接連公佈，全面做多買入")
    assert bullish_zh["sentiment_score"] > 0.15

    bearish_kr = classifier.analyze_sentiment("SK하이닉스 실적 충격 적자 전환 폭락 악재 발생")
    assert bearish_kr["sentiment_score"] < -0.15

def test_engagement_weight_bounds(engine):
    # Zero engagement baseline
    w_zero = engine.calculate_engagement_weight(likes=0, reposts=0, views=0, followers=0)
    assert 0.5 <= w_zero <= 1.5

    # Viral post with 10k likes and 500k views
    w_viral = engine.calculate_engagement_weight(likes=10000, reposts=2000, views=500000, followers=50000)
    assert w_viral <= 3.5  # Must not exceed upper cap

def test_recency_decay(engine):
    from datetime import datetime, timezone, timedelta
    now_ts = datetime.now(timezone.utc).isoformat()
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()

    w_now = engine.calculate_recency_weight(now_ts)
    w_old = engine.calculate_recency_weight(old_ts)

    assert w_now > w_old
    assert w_old >= 0.10  # Minimum floor

def test_spam_shielding(engine):
    ts = "2026-11-08T12:00:00Z"
    res_clean = engine.process_item("Tencent growth beat expectations", ts, likes=10, is_spam=False)
    res_spam = engine.process_item("Tencent growth beat expectations", ts, likes=10, is_spam=True)

    assert res_clean["is_spam_or_dup"] == 0
    assert res_spam["is_spam_or_dup"] == 1
    assert abs(res_spam["weighted_sentiment"]) < abs(res_clean["weighted_sentiment"])
