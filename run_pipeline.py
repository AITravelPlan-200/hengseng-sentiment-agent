"""
HengSeng Sentiment Price Prediction Agent - Command Line Execution Entry Point.
Runs the complete multi-agent pipeline and outputs formatted institutional analysis.
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.orchestrator import orchestrator
import pandas as pd

def main():
    print("=" * 80)
    print("  HENG SENG SENTIMENT PRICE PREDICTION AGENT - LIVE EXECUTION")
    print("  CrowdWisdomTrading Quantitative Research Platform")
    print("=" * 80)

    def print_progress(msg: str, frac: float):
        bar = "█" * int(frac * 30) + "░" * (30 - int(frac * 30))
        print(f"[{bar}] {int(frac * 100):>3}% | {msg}")

    # Run full pipeline with progress callbacks
    results = orchestrator.run_full_pipeline(
        progress_callback=print_progress,
        force_market_refresh=False
    )

    stock_analyses = results.get("stock_analyses", [])
    if not stock_analyses:
        print("❌ Error: No stock intelligence generated.")
        return

    df = pd.DataFrame(stock_analyses)

    print("\n" + "=" * 80)
    print("  MULTI-ASSET INTELLIGENCE & NEXT-DAY PRICE PREDICTIONS")
    print("=" * 80)

    # Format table for terminal
    disp_cols = [
        "ticker", "company_name", "exchange", "current_price",
        "social_mention_count", "combined_sentiment_score", "predicted_next_period_price",
        "predicted_price_movement_pct", "suggested_position"
    ]
    summary_df = df[disp_cols].copy()
    summary_df.columns = [
        "Ticker", "Company", "Exch", "Last Close",
        "Mentions", "Sentiment", "Pred Close", "Pred %", "Position"
    ]
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("  EXPORTED ARTIFACTS")
    print("=" * 80)
    exported = results.get("exported_files", {})
    print(f"- HKEX Rankings CSV:  {exported.get('hk_csv')}")
    print(f"- KRX Rankings CSV:   {exported.get('kr_csv')}")
    print(f"- HTML Briefing:      {exported.get('html_report')}")

    print("\n" + "=" * 80)
    print("  STREAMLIT DASHBOARD IS ACTIVE")
    print("  Open in your web browser: http://localhost:8501")
    print("=" * 80)

if __name__ == "__main__":
    main()
