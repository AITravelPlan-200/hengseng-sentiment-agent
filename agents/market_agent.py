"""
Agent 3: Market Intelligence Agent.
Collects historical OHLCV market bars for HKEX and KRX equities,
calculates technical indicators, and derives quantitative support and resistance barriers.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

from config.tickers import ASSET_UNIVERSE
from data_ingestion.price_collector import PriceCollector
from analysis.support_resistance import SupportResistanceAnalyzer
from database.db import db

logger = logging.getLogger("hengseng.agents.market")

class MarketIntelligenceAgent:
    """Agent responsible for market data ingestion and quantitative technical modeling."""

    def __init__(self, price_collector: Optional[PriceCollector] = None):
        self.collector = price_collector or PriceCollector()

    def run_for_ticker(self, ticker: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Retrieves market bars, calculates indicators, and extracts support/resistance levels.
        """
        logger.info(f"[MarketAgent] Processing market data for {ticker}")
        df = self.collector.fetch_historical_bars(ticker, force_refresh=force_refresh)
        
        levels = SupportResistanceAnalyzer.calculate_levels(df)

        return {
            "ticker": ticker,
            "market_df": df,
            "levels": levels,
            "bars_count": len(df),
            "latest_close": float(df.iloc[-1]["close"]) if not df.empty else 0.0
        }

    def run_for_universe(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Collects market data across all target assets."""
        results = {}
        for ticker in ASSET_UNIVERSE:
            res = self.run_for_ticker(ticker, force_refresh=force_refresh)
            results[ticker] = res
        return results
