"""
Central Agent Orchestrator.
Coordinates the end-to-end multi-agent execution pipeline:
Agent 1 (Social) -> Agent 2 (News) -> Agent 3 (Market) -> Agent 4 (Prediction) -> Agent 5 (Reporting).
Provides progress callbacks, status reporting, and robust fault-isolation.
"""

import logging
from typing import Dict, Any, List, Optional, Callable
import pandas as pd

from config.tickers import ASSET_UNIVERSE, get_tickers_by_exchange
from agents.social_agent import SocialIntelligenceAgent
from agents.news_agent import NewsIntelligenceAgent
from agents.market_agent import MarketIntelligenceAgent
from agents.prediction_agent import PredictionAgent
from agents.reporting_agent import ReportingAgent
from database.db import db

logger = logging.getLogger("hengseng.orchestrator")

class CentralOrchestrator:
    """Coordinates the 5 specialized agents in a clean multi-agent pipeline."""

    def __init__(self):
        self.social_agent = SocialIntelligenceAgent()
        self.news_agent = NewsIntelligenceAgent()
        self.market_agent = MarketIntelligenceAgent()
        self.prediction_agent = PredictionAgent()
        self.reporting_agent = ReportingAgent()

    def run_full_pipeline(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        force_market_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the entire multi-agent intelligence cycle across all 10 assets.
        Reports progress step-by-step through an optional callback.
        """
        logger.info("=== Starting HengSeng Multi-Agent Intelligence Pipeline ===")

        def report(msg: str, frac: float):
            logger.info(f"Pipeline: {msg} ({int(frac*100)}%)")
            if progress_callback:
                progress_callback(msg, frac)

        # -------------------------------------------------------------
        # STEP 1: Social Media Intelligence Collection (Agent 1)
        # -------------------------------------------------------------
        report("Agent 1 (Social Intelligence): Ingesting and analyzing social posts...", 0.10)
        social_results = {}
        for ticker in ASSET_UNIVERSE:
            try:
                posts = self.social_agent.run_for_ticker(ticker)
                social_results[ticker] = posts
            except Exception as e:
                logger.error(f"Error in Social Agent for {ticker}: {e}")
                social_results[ticker] = []

        # Load all social posts from DB as DataFrame
        social_df = db.get_social_posts(limit=1000)

        # -------------------------------------------------------------
        # STEP 2: Financial News Intelligence Ingestion (Agent 2)
        # -------------------------------------------------------------
        report("Agent 2 (News Intelligence): Retrieving and scoring financial news...", 0.30)
        news_results = {}
        for ticker in ASSET_UNIVERSE:
            try:
                articles = self.news_agent.run_for_ticker(ticker)
                news_results[ticker] = articles
            except Exception as e:
                logger.error(f"Error in News Agent for {ticker}: {e}")
                news_results[ticker] = []

        news_df = db.get_news(limit=1000)

        # -------------------------------------------------------------
        # STEP 3: Market Data & Technical Indicators (Agent 3)
        # -------------------------------------------------------------
        report("Agent 3 (Market Intelligence): Collecting OHLCV & computing indicators...", 0.50)
        market_data_dict = {}
        for ticker in ASSET_UNIVERSE:
            try:
                m_res = self.market_agent.run_for_ticker(ticker, force_refresh=force_market_refresh)
                market_data_dict[ticker] = m_res["market_df"]
            except Exception as e:
                logger.error(f"Error in Market Agent for {ticker}: {e}")
                market_data_dict[ticker] = pd.DataFrame()

        # -------------------------------------------------------------
        # STEP 4: Machine Learning Next-Period Price Prediction (Agent 4)
        # -------------------------------------------------------------
        report("Agent 4 (Prediction Agent): Training models & forecasting prices...", 0.75)
        predictions_dict = {}
        for ticker in ASSET_UNIVERSE:
            try:
                m_df = market_data_dict.get(ticker, pd.DataFrame())
                pred = self.prediction_agent.run_for_ticker(
                    ticker=ticker,
                    market_df=m_df,
                    social_df=social_df,
                    news_df=news_df
                )
                predictions_dict[ticker] = pred
            except Exception as e:
                logger.error(f"Error in Prediction Agent for {ticker}: {e}")
                predictions_dict[ticker] = {}

        # -------------------------------------------------------------
        # STEP 5: Reporting, Ranking & Artifacts (Agent 5)
        # -------------------------------------------------------------
        report("Agent 5 (Reporting Agent): Generating league tables & executive reports...", 0.95)
        report_artifacts = self.reporting_agent.generate_universe_reports(
            market_data_dict=market_data_dict,
            social_df=social_df,
            news_df=news_df,
            predictions_dict=predictions_dict
        )

        report("Pipeline complete. Stock intelligence ready.", 1.0)
        logger.info("=== HengSeng Multi-Agent Pipeline Completed Successfully ===")

        return {
            "stock_analyses": report_artifacts["stock_analyses"],
            "hk_rankings": report_artifacts["hk_rankings"],
            "kr_rankings": report_artifacts["kr_rankings"],
            "exported_files": report_artifacts["exported_files"],
            "predictions_dict": predictions_dict,
            "social_df": social_df,
            "news_df": news_df,
            "market_data_dict": market_data_dict
        }

# Global orchestrator singleton
orchestrator = CentralOrchestrator()
