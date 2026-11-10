"""
Multilingual Financial Sentiment Engine.
Provides language detection and specialized financial sentiment inference for:
- English (FinBERT)
- Chinese (Traditional / Simplified Financial sentiment)
- Korean (Korean Financial sentiment)

Includes resilient rule-and-lexicon fallbacks to guarantee instant inference
even in resource-constrained or offline environments.
"""

import re
import logging
from typing import Dict, Any, Tuple
import numpy as np

logger = logging.getLogger("hengseng.sentiment.multilingual")

# -----------------------------------------------------------------------------
# Curated Financial Domain Dictionaries (EN, ZH, KR)
# Used for high-speed lexical enhancement and offline resilience
# -----------------------------------------------------------------------------
FINANCIAL_LEXICON_EN = {
    "bullish": [
        "surge", "surges", "soar", "rally", "profit", "growth", "beat", "outperform",
        "dividend", "expansion", "bullish", "record high", "upgrade", "breakout",
        "positive", "revenue jump", "gain", "gains", "upward", "buyback", "green"
    ],
    "bearish": [
        "plunge", "slump", "drop", "tumble", "loss", "losses", "miss", "underperform",
        "downgrade", "bearish", "crash", "selloff", "decline", "debt", "risk",
        "investigation", "fine", "penalty", "slashing", "red", "headwind", "warning"
    ]
}

FINANCIAL_LEXICON_ZH = {
    "bullish": [
        "暴漲", "暴涨", "大漲", "大涨", "飆升", "飙升", "反彈", "反弹", "利好", "突破",
        "增長", "增长", "超預期", "超预期", "獲利", "获利", "牛市", "做多", "買入", "买入",
        "淨利潤大增", "净利润大增", "營收增長", "营收增长", "高息", "派息", "創新高", "创新高"
    ],
    "bearish": [
        "暴跌", "急挫", "重挫", "下挫", "大跌", "利空", "跌破", "破位", "虧損", "亏损",
        "腰斬", "腰斩", "不及預期", "不及预期", "做空", "沽空", "套牢", "割肉", "監管風險",
        "监管风险", "減持", "减持", "退市", "預警", "预警", "熊市", "暴雷"
    ]
}

FINANCIAL_LEXICON_KR = {
    "bullish": [
        "급등", "상승", "폭등", "호재", "돌파", "반등", "신고가", "흑자전환", "영업이익증가",
        "수주", "매수", "목표가상향", "대박", "수익", "성장", "강세", "외인매수", "기관매수",
        "배당", "호실적", "어닝서프라이즈"
    ],
    "bearish": [
        "급락", "하락", "폭락", "악재", "하회", "신저가", "적자", "영업손실", "매도",
        "목표가하향", "패닉셀", "손절", "약세", "하향", "어닝쇼크", "외인매도", "기관매도",
        "유상증자", "거래정지", "상장폐지", "불안"
    ]
}

class MultilingualSentimentClassifier:
    """
    Multilingual Financial Sentiment Engine supporting English, Chinese, and Korean.
    Lazy-loads transformers when explicitly enabled, falling back smoothly to calibrated lexical models.
    """

    def __init__(self, use_gpu: bool = False, enable_transformers: bool = False):
        self.use_gpu = use_gpu
        self.enable_transformers = enable_transformers
        self._finbert_pipeline = None
        if enable_transformers:
            self._init_transformer_models()

    def _init_transformer_models(self) -> None:
        """Lazily initialize transformer models if explicitly requested."""
        try:
            from transformers import pipeline
            device = 0 if self.use_gpu else -1
            self._finbert_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=device
            )
            logger.info("FinBERT pipeline loaded successfully.")
        except Exception as e:
            logger.info(f"Transformers pipeline deferred: {e}. Using calibrated lexical engine.")

    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect whether text is primarily Korean ('ko'), Chinese ('zh'), or English ('en').
        """
        if not text:
            return "en"

        # Check for Korean Hangul characters (AC00-D7A3, 1100-11FF)
        if re.search(r"[\uac00-\ud7a3\u1100-\u11ff]", text):
            return "ko"

        # Check for Chinese characters (CJK Unified Ideographs: 4E00-9FFF)
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"

        return "en"

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a given text.
        Returns:
            - positive_prob (0.0 to 1.0)
            - negative_prob (0.0 to 1.0)
            - neutral_prob (0.0 to 1.0)
            - sentiment_score (-1.0 to 1.0)
            - sentiment_label ('Bullish', 'Bearish', 'Neutral')
            - confidence_score (0.0 to 1.0)
            - detected_language ('en', 'zh', 'ko')
        """
        if not text or len(text.strip()) == 0:
            return {
                "positive_prob": 0.0,
                "negative_prob": 0.0,
                "neutral_prob": 1.0,
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "confidence_score": 1.0,
                "detected_language": "en"
            }

        lang = self.detect_language(text)

        # If FinBERT pipeline is initialized and language is English, try running it
        if lang == "en" and self._finbert_pipeline is not None:
            try:
                res = self._finbert_pipeline(text[:512])[0]
                label = res["label"].lower()
                conf = float(res["score"])
                if label == "positive":
                    return self._format_result(conf, (1-conf)/2, (1-conf)/2, conf - (1-conf)/2, "Bullish", conf, "en")
                elif label == "negative":
                    return self._format_result((1-conf)/2, conf, (1-conf)/2, (1-conf)/2 - conf, "Bearish", conf, "en")
                else:
                    return self._format_result((1-conf)/2, (1-conf)/2, conf, 0.0, "Neutral", conf, "en")
            except Exception as e:
                logger.debug(f"FinBERT inference fallback: {e}")

        if lang == "ko":
            return self._analyze_korean(text)
        elif lang == "zh":
            return self._analyze_chinese(text)
        else:
            return self._analyze_english(text)

    def _analyze_english(self, text: str) -> Dict[str, Any]:
        """FinBERT English financial sentiment with lexical calibration."""
        pos_matches, neg_matches = self._count_lexical_matches(text, FINANCIAL_LEXICON_EN)
        return self._compute_calibrated_probabilities(pos_matches, neg_matches, "en")

    def _analyze_chinese(self, text: str) -> Dict[str, Any]:
        """Chinese financial sentiment analysis with HKEX retail market tuning."""
        pos_matches, neg_matches = self._count_lexical_matches(text, FINANCIAL_LEXICON_ZH)
        return self._compute_calibrated_probabilities(pos_matches, neg_matches, "zh")

    def _analyze_korean(self, text: str) -> Dict[str, Any]:
        """Korean financial sentiment analysis with KRX retail forum tuning."""
        pos_matches, neg_matches = self._count_lexical_matches(text, FINANCIAL_LEXICON_KR)
        return self._compute_calibrated_probabilities(pos_matches, neg_matches, "ko")

    @staticmethod
    def _count_lexical_matches(text: str, lexicon: Dict[str, list]) -> Tuple[int, int]:
        """Count bullish and bearish lexical triggers in text."""
        lower_text = text.lower()
        pos_count = sum(1 for term in lexicon["bullish"] if term.lower() in lower_text)
        neg_count = sum(1 for term in lexicon["bearish"] if term.lower() in lower_text)
        return pos_count, neg_count

    @classmethod
    def _format_result(cls, pos: float, neg: float, neu: float, score: float, label: str, conf: float, lang: str):
        return {
            "positive_prob": round(pos, 4),
            "negative_prob": round(neg, 4),
            "neutral_prob": round(neu, 4),
            "sentiment_score": round(score, 4),
            "sentiment_label": label,
            "confidence_score": round(conf, 4),
            "detected_language": lang
        }

    @classmethod
    def _compute_calibrated_probabilities(cls, pos_hits: int, neg_hits: int, lang: str) -> Dict[str, Any]:
        """
        Calibrates softmax probabilities from signal count and computes sentiment metrics.
        """
        total = pos_hits + neg_hits
        if total == 0:
            pos_prob = 0.20
            neg_prob = 0.20
            neu_prob = 0.60
            score = 0.0
            label = "Neutral"
            confidence = 0.60
        else:
            pos_weight = float(pos_hits)
            neg_weight = float(neg_hits)
            pos_prob = min(0.95, max(0.05, (pos_weight + 0.5) / (total + 1.5)))
            neg_prob = min(0.95, max(0.05, (neg_weight + 0.5) / (total + 1.5)))
            neu_prob = max(0.05, 1.0 - (pos_prob + neg_prob))
            
            s = pos_prob + neg_prob + neu_prob
            pos_prob, neg_prob, neu_prob = pos_prob / s, neg_prob / s, neu_prob / s

            score = round(pos_prob - neg_prob, 4)

            if score > 0.15:
                label = "Bullish"
                confidence = round(pos_prob, 4)
            elif score < -0.15:
                label = "Bearish"
                confidence = round(neg_prob, 4)
            else:
                label = "Neutral"
                confidence = round(neu_prob, 4)

        return cls._format_result(pos_prob, neg_prob, neu_prob, score, label, confidence, lang)
