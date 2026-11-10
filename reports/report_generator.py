"""
Financial Intelligence Report Generator.
Generates auditable CSV and formatted PDF/HTML reports for HKEX and KRX rankings,
predictions, and sentiment distributions.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from config.settings import REPORTS_DIR

logger = logging.getLogger("hengseng.reports")

class ReportGenerator:
    """Generates structured CSV and HTML/PDF executive reports."""

    @classmethod
    def export_rankings_csv(cls, rankings_dict: Dict[str, pd.DataFrame], exchange: str) -> str:
        """Exports ranking dataframes to a consolidated CSV."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"hengseng_ranking_{exchange.lower()}_{timestamp}.csv"
        filepath = REPORTS_DIR / filename

        combined_records = []
        for rank_type, df in rankings_dict.items():
            if not df.empty:
                temp = df.copy()
                temp["Ranking_Category"] = rank_type
                combined_records.append(temp)

        if combined_records:
            final_df = pd.concat(combined_records, ignore_index=True)
            final_df.to_csv(filepath, index=False, encoding="utf-8-sig")
            logger.info(f"Ranking CSV generated: {filepath}")
            return str(filepath)
        return ""

    @classmethod
    def generate_html_report(
        cls,
        stock_analyses: List[Dict[str, Any]],
        exchange_summaries: Dict[str, Any],
        title: str = "HengSeng Sentiment Price Prediction Agent - Executive Briefing"
    ) -> str:
        """
        Generates a polished financial terminal themed HTML report,
        which can be rendered directly or printed/saved as PDF.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        report_file = REPORTS_DIR / f"hengseng_executive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        rows_html = ""
        for s in stock_analyses:
            direction_color = "#00e676" if s.get("sentiment_direction") == "Bullish" else ("#ff5252" if s.get("sentiment_direction") == "Bearish" else "#90caf9")
            pos_color = "#00e676" if s.get("suggested_position") == "Long" else ("#ff5252" if s.get("suggested_position") == "Short" else "#e0e0e0")
            pred_change = s.get("predicted_price_movement_pct", 0.0)
            change_color = "#00e676" if pred_change >= 0 else "#ff5252"

            rows_html += f"""
            <tr>
                <td style="font-weight:bold; color:#00e5ff;">{s.get('ticker')}</td>
                <td>{s.get('company_name')}</td>
                <td>{s.get('exchange')}</td>
                <td style="text-align:right;">{s.get('currency')} {s.get('current_price', 0):,.2f}</td>
                <td style="text-align:right;">{s.get('social_mention_count', 0)}</td>
                <td style="text-align:right; color:{direction_color}; font-weight:bold;">{s.get('combined_sentiment_score', 0):+.2f}</td>
                <td style="text-align:right; font-weight:bold;">{s.get('predicted_next_period_price', 0):,.2f}</td>
                <td style="text-align:right; color:{change_color}; font-weight:bold;">{pred_change:+.2f}%</td>
                <td style="text-align:center; font-weight:bold; color:{pos_color};">{s.get('suggested_position')}</td>
                <td style="text-align:right;">{s.get('support_level', 0):,.2f} / {s.get('resistance_level', 0):,.2f}</td>
            </tr>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    background-color: #0b0f19;
                    color: #e2e8f0;
                    margin: 0;
                    padding: 30px;
                }}
                .header {{
                    border-bottom: 2px solid #1e293b;
                    padding-bottom: 20px;
                    margin-bottom: 25px;
                }}
                h1 {{
                    color: #f8fafc;
                    margin: 0 0 10px 0;
                    font-size: 26px;
                    letter-spacing: -0.5px;
                }}
                .meta {{
                    color: #94a3b8;
                    font-size: 13px;
                }}
                .badge {{
                    background: #1e293b;
                    border: 1px solid #334155;
                    padding: 4px 10px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-family: monospace;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                    font-size: 13px;
                }}
                th {{
                    background-color: #111827;
                    color: #94a3b8;
                    text-align: left;
                    padding: 12px 10px;
                    border-bottom: 2px solid #1f2937;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                td {{
                    padding: 12px 10px;
                    border-bottom: 1px solid #1e293b;
                }}
                tr:hover {{
                    background-color: #162032;
                }}
                .disclaimer {{
                    margin-top: 35px;
                    padding: 15px;
                    border-left: 3px solid #f59e0b;
                    background: #1e1b13;
                    color: #d97706;
                    font-size: 12px;
                    line-height: 1.5;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <div class="meta">
                    Report Timestamp: <span class="badge">{timestamp}</span> &nbsp;|&nbsp; 
                    Scope: <span class="badge">HKEX & KRX (10 Target Equities)</span> &nbsp;|&nbsp;
                    AI Assessment: <span class="badge">CrowdWisdomTrading</span>
                </div>
            </div>

            <h2>Comprehensive Multi-Asset Intelligence Table</h2>
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Asset Name</th>
                        <th>Exch</th>
                        <th style="text-align:right;">Last Price</th>
                        <th style="text-align:right;">Mentions</th>
                        <th style="text-align:right;">Sentiment</th>
                        <th style="text-align:right;">Pred Next Close</th>
                        <th style="text-align:right;">Pred %</th>
                        <th style="text-align:center;">Position</th>
                        <th style="text-align:right;">Support / Resist</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>

            <div class="disclaimer">
                <strong>DISCLAIMER & METHODOLOGICAL DISCLOSURE:</strong><br>
                This document is generated by the HengSeng Sentiment Price Prediction Agent for institutional AI research, quantitative evaluation, and the CrowdWisdomTrading developer assessment. 
                Predictions are derived via supervised machine learning models trained strictly on chronological time-series data without future-period leakage. All outputs are probabilistic estimates and do not represent financial or investment advice.
            </div>
        </body>
        </html>
        """

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Executive HTML report written to {report_file}")
        return str(report_file)
