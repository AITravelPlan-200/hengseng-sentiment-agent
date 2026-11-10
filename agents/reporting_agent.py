"""
Agent 5: Reporting Agent.
Synthesizes comprehensive stock intelligence reports,
computes multi-criteria rankings for HKEX and KRX,
and generates auditable CSV and HTML/PDF executive briefings.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from config.tickers import ASSET_UNIVERSE, get_tickers_by_exchange
from analysis.stock_analysis import StockIntelligenceEngine
from analysis.ranking import StockRankingEngine
from reports.report_generator import ReportGenerator

logger = logging.getLogger("hengseng.agents.reporting")

class ReportingAgent:
    """Agent responsible for intelligence aggregation, ranking tables, and report artifact compilation."""

    def __init__(self):
        self.ranking_engine = StockRankingEngine()
        self.report_generator = ReportGenerator()

    def generate_universe_reports(
        self,
        market_data_dict: Dict[str, pd.DataFrame],
        social_df: pd.DataFrame,
        news_df: pd.DataFrame,
        predictions_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Builds complete multi-exchange intelligence, rankings, and export artifacts.
        """
        logger.info("[ReportingAgent] Generating multi-asset intelligence reports...")

        # 1. Build stock intelligence report for each ticker
        stock_analyses = []
        for ticker in ASSET_UNIVERSE:
            m_df = market_data_dict.get(ticker, pd.DataFrame())
            pred = predictions_dict.get(ticker)
            report = StockIntelligenceEngine.generate_stock_intelligence(
                ticker=ticker,
                market_df=m_df,
                social_df=social_df,
                news_df=news_df,
                prediction_result=pred
            )
            stock_analyses.append(report)

        # 2. Build Exchange Rankings
        hk_rankings = self.ranking_engine.rank_stocks(stock_analyses, exchange="HKEX")
        kr_rankings = self.ranking_engine.rank_stocks(stock_analyses, exchange="KRX")

        # 3. Generate CSV exports
        hk_csv_path = self.report_generator.export_rankings_csv(hk_rankings, exchange="HKEX")
        kr_csv_path = self.report_generator.export_rankings_csv(kr_rankings, exchange="KRX")

        # 4. Generate Executive HTML Briefing
        html_report_path = self.report_generator.generate_html_report(
            stock_analyses=stock_analyses,
            exchange_summaries={"HKEX": hk_rankings, "KRX": kr_rankings}
        )

        return {
            "stock_analyses": stock_analyses,
            "hk_rankings": hk_rankings,
            "kr_rankings": kr_rankings,
            "exported_files": {
                "hk_csv": hk_csv_path,
                "kr_csv": kr_csv_path,
                "html_report": html_report_path
            }
        }
