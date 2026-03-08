"""
KPI cards and metrics comparison table.
"""
import streamlit as st
import pandas as pd
import numpy as np

from portfolio_analyzer.core.data import load_price_data
from portfolio_analyzer.core.metrics import (
    compute_returns,
    calculate_annualized_metrics,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_skewness,
    calculate_excess_kurtosis,
)


def _kpi_card(title: str, subtitle: str, value: str):
    """Render a single KPI card using raw HTML."""
    st.markdown(f"""
    <div style="
        border: 1px solid #dadce0;
        padding: 15px;
        border-radius: 0px;
        margin-bottom: 15px;
    ">
        <div style="font-size: 16px; font-weight: 600; margin-bottom: 2px;">{title}</div>
        <div style="font-size: 12px; opacity: 0.6; margin-bottom: 8px;">{subtitle}</div>
        <div style="font-size: 28px; font-weight: 500;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpis(portfolio_returns: pd.Series, benchmark_ticker: str, start_date, end_date):
    """
    Compute and render the full KPI section:
    1. Benchmark-relative KPI cards (Active Return, TE, IR)
    2. Comparison metrics table (Portfolio vs Benchmark with deltas)

    Returns
    -------
    tuple: (comparison_df, drawdown, active_return, tracking_error, information_ratio)
        So the caller can pass them to chart renderers.
    """
    # ── Portfolio metrics ─────────────────────────────────────────
    cum_ret_series = (1 + portfolio_returns).cumprod()
    total_cum_return = cum_ret_series.iloc[-1] - 1

    ann_metrics = calculate_annualized_metrics(portfolio_returns)
    annualized_return = ann_metrics["ann_ret"]
    annualized_vol = ann_metrics["ann_vol"]
    sharpe_ratio = ann_metrics["sharpe"]

    drawdown = cum_ret_series / cum_ret_series.cummax() - 1
    max_drawdown = drawdown.min()
    calmar = calculate_calmar_ratio(portfolio_returns)
    sortino = calculate_sortino_ratio(portfolio_returns)

    # ── Section header ────────────────────────────────────────────
    st.markdown('<h3 style="margin-top: -5px;">Key performance indicators</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        Core risk and return metrics for the backtest period
    </div>
    """, unsafe_allow_html=True)

    # ── Benchmark data ────────────────────────────────────────────
    bench_prices = load_price_data([benchmark_ticker], start_date, end_date)
    comparison_df = None
    active_return = tracking_error = information_ratio = None

    b_total_ret = b_ann_ret = b_vol = b_sharpe = b_sortino = b_calmar = b_max_dd = "–"

    if not bench_prices.empty:
        bench_returns = compute_returns(bench_prices)[benchmark_ticker]
        comparison_df = pd.DataFrame({
            "Portfolio": portfolio_returns,
            "Benchmark": bench_returns,
        }).dropna()

    if comparison_df is not None and not comparison_df.empty:
        b_ret_series = comparison_df["Benchmark"]
        p_ret_series = comparison_df["Portfolio"]

        b_cum = (1 + b_ret_series).cumprod()
        b_total_ret = f"{(b_cum.iloc[-1] - 1):.1%}"
        b_ann = calculate_annualized_metrics(b_ret_series)
        b_ann_ret = f"{b_ann['ann_ret']:.1%}"
        b_vol = f"{b_ann['ann_vol']:.1%}"
        b_sharpe = f"{b_ann['sharpe']:.2f}"
        b_sortino_val = calculate_sortino_ratio(b_ret_series)
        b_sortino = f"{b_sortino_val:.2f}"
        b_calmar_val = calculate_calmar_ratio(b_ret_series)
        b_calmar = f"{b_calmar_val:.2f}"
        b_dd = b_cum / b_cum.cummax() - 1
        b_max_dd = f"{b_dd.min():.1%}"

        # Relative metrics
        p_ann_return = p_ret_series.mean() * 252
        b_ann_return_val = b_ret_series.mean() * 252
        active_return = p_ann_return - b_ann_return_val
        diff_returns = p_ret_series - b_ret_series
        tracking_error = diff_returns.std() * np.sqrt(252)
        information_ratio = active_return / tracking_error if tracking_error != 0 else 0.0

    # ── Relative KPI cards ────────────────────────────────────────
    if active_return is not None:
        row_bench = st.columns(3)
        with row_bench[0]:
            _kpi_card("Active return", "Port Ann. − Bench Ann.", f"{active_return:.2%}")
        with row_bench[1]:
            _kpi_card("Tracking error", "Vol. of return difference", f"{tracking_error:.2%}")
        with row_bench[2]:
            _kpi_card("Information ratio", "Active Return / TE", f"{information_ratio:.2f}")

    # ── Comparison table ──────────────────────────────────────────
    p_vals = [total_cum_return, annualized_return, annualized_vol, max_drawdown, sharpe_ratio, sortino, calmar]

    if comparison_df is not None and not comparison_df.empty:
        b_ret_s = comparison_df["Benchmark"]
        b_cum_s = (1 + b_ret_s).cumprod()
        b_ann_m = calculate_annualized_metrics(b_ret_s)
        b_dd_s = b_cum_s / b_cum_s.cummax() - 1
        b_vals = [
            b_cum_s.iloc[-1] - 1, b_ann_m["ann_ret"], b_ann_m["ann_vol"],
            b_dd_s.min(), b_ann_m["sharpe"],
            calculate_sortino_ratio(b_ret_s), calculate_calmar_ratio(b_ret_s),
        ]
    else:
        b_vals = [None] * 7

    lower_is_better = {2}
    deltas_text: list[str] = []

    for i in range(7):
        p = p_vals[i]
        b = b_vals[i]
        if b is None:
            deltas_text.append("–")
            continue
        d = p - b
        is_pct = i in [0, 1, 2, 3]
        txt = f"{abs(d):.1%}" if is_pct else f"{abs(d):.2f}"
        sign = "+" if d > 0 else "-" if d < 0 else ""
        deltas_text.append(f"{sign}{txt}")

    metrics_data = {
        "Metric": [
            "Total return", "Annualized return", "Annualized volatility",
            "Max drawdown", "Sharpe ratio", "Sortino ratio", "Calmar ratio",
        ],
        "Portfolio": [
            f"{total_cum_return:.1%}", f"{annualized_return:.1%}", f"{annualized_vol:.1%}",
            f"{max_drawdown:.1%}", f"{sharpe_ratio:.2f}", f"{sortino:.2f}", f"{calmar:.2f}",
        ],
        f"Benchmark ({benchmark_ticker})": [
            b_total_ret, b_ann_ret, b_vol,
            b_max_dd, b_sharpe, b_sortino, b_calmar,
        ],
        "Delta": deltas_text,
    }

    metrics_table = pd.DataFrame(metrics_data)
    st.dataframe(
        metrics_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Portfolio": st.column_config.TextColumn("Portfolio", width="small"),
            f"Benchmark ({benchmark_ticker})": st.column_config.TextColumn(
                f"Benchmark ({benchmark_ticker})", width="small"
            ),
            "Delta": st.column_config.TextColumn("Delta", width="small"),
        },
    )

    return comparison_df, drawdown, active_return, tracking_error, information_ratio
