"""
PDF report export section.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

from portfolio_analyzer.core.report_builder import generate_pdf_report


def render_report_section(pf_data: dict, comparison_df: pd.DataFrame | None,
                          active_return=None, tracking_error=None, information_ratio=None):
    """Render the PDF export block: AI commentary + download button."""
    st.markdown("### Report export")
    st.markdown("""
        <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
            Generate a downloadable PDF summary of the backtest results
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Generating PDF"):
        try:
            # Build returns DataFrame
            results_df = pd.DataFrame({"Strategy": pf_data["portfolio_returns"]})
            if comparison_df is not None and not comparison_df.empty:
                results_df["Benchmark"] = comparison_df["Benchmark"]

            # Relative metrics for the report
            metrics_dict: dict = {}
            if comparison_df is not None and not comparison_df.empty:
                metrics_dict.update({
                    "Active Return": active_return,
                    "Tracking Error": tracking_error,
                    "Information Ratio": information_ratio,
                })
            metrics_dict = {k: v for k, v in metrics_dict.items() if pd.notna(v)}
            metrics_df = (
                pd.DataFrame.from_dict(metrics_dict, orient="index", columns=["Value"])
                if metrics_dict else pd.DataFrame(columns=["Value"])
            )

            # ── AI commentary (optional) ──────────────────────────
            _has_key = (
                bool(st.secrets.get("OPENAI_API_KEY", "").strip())
                and not st.secrets.get("OPENAI_API_KEY", "").startswith("sk-...")
            )
            ai_metrics_comment = ""

            if _has_key:
                from portfolio_analyzer.core.ai_analysis import (
                    generate_metrics_commentary,
                )

                _ret = pf_data["portfolio_returns"]
                _total_ret = (1 + _ret).cumprod().iloc[-1] - 1
                _n_years = len(_ret) / 252
                _cagr = (1 + _total_ret) ** (1 / _n_years) - 1 if _n_years > 0 else 0
                _vol = _ret.std() * np.sqrt(252)
                _sharpe = (_ret.mean() * 252) / _vol if _vol != 0 else 0
                _dd = ((1 + _ret).cumprod() / (1 + _ret).cumprod().cummax() - 1).min()
                _win_rate = (_ret > 0).sum() / len(_ret)
                _best_day = _ret.max()
                _worst_day = _ret.min()
                _pdf_metrics = {
                    "CAGR": f"{_cagr:.2%}",
                    "Volatility": f"{_vol:.2%}",
                    "Sharpe": f"{_sharpe:.2f}",
                    "Max Drawdown": f"{_dd:.2%}",
                    "Win Rate": f"{_win_rate:.2%}",
                    "Best Day": f"{_best_day:.2%}",
                    "Worst Day": f"{_worst_day:.2%}"
                }

                ai_metrics_comment = generate_metrics_commentary(
                    metrics=_pdf_metrics,
                    benchmark_name=pf_data.get("benchmark", "benchmark"),
                    tracking_error=tracking_error,
                    information_ratio=information_ratio,
                )

                # Only metrics commentary is generated now.

            pdf_bytes = generate_pdf_report(
                results_df=results_df,
                metrics_df=metrics_df,
                tickers=pf_data["tickers"],
                weights=pf_data["weights"],
                metadata=pf_data.get("metadata", None),
                pdf_context={
                    "strategy": pf_data.get("strategy", "Unknown"),
                    "benchmark": pf_data.get("benchmark", "Unknown"),
                    "date_range": f"{pf_data['portfolio_returns'].index.min().strftime('%b %Y')} - {pf_data['portfolio_returns'].index.max().strftime('%b %Y')}" if "portfolio_returns" in pf_data else "Unknown",
                    "rebalancing": "Monthly"
                },
                ai_metrics_comment=ai_metrics_comment,
            )

            st.download_button(
                label="Generate report",
                data=pdf_bytes,
                file_name=f"tear_sheet_{datetime.today().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating export files: {e}")
