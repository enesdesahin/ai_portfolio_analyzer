"""
Portfolio Builder — Main page orchestrator.

This page coordinates the sidebar inputs, backtest execution,
and rendering of analytics components. All visual rendering is
delegated to focused component modules in app/components/.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from portfolio_analyzer.core.data import load_price_data, fetch_asset_metadata
from portfolio_analyzer.core.metrics import compute_returns
from portfolio_analyzer.core.engine import DailyBacktestEngine

from portfolio_analyzer.app.components.sidebar import render_sidebar
from portfolio_analyzer.app.components.kpis import render_kpis
from portfolio_analyzer.app.components.charts import (
    render_performance_tab,
    render_drawdown_tab,
    render_volatility_tab,
    render_correlation_tab,
)
from portfolio_analyzer.app.components.report import render_report_section


# ── Composition pie charts ────────────────────────────────────────────

def render_portfolio_composition(tickers, weights, meta_df=None, key_prefix="composition"):
    st.markdown('<h3 style="margin-top: -20px;">Overall composition</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        Breakdown of assets and their allocated weights
    </div>
    """, unsafe_allow_html=True)

    w_df = pd.DataFrame({"Asset": tickers, "Weight": weights})
    sector_weights, country_weights, asset_class_weights = None, None, None

    if meta_df is not None and not meta_df.empty:
        w_df = w_df.merge(meta_df, left_on="Asset", right_index=True, how="left")
        if "Sector" in w_df.columns:
            sector_weights = w_df.groupby("Sector")["Weight"].sum().reset_index()
        if "Country" in w_df.columns:
            country_weights = w_df.groupby("Country")["Weight"].sum().reset_index()
        if "Asset Class" in w_df.columns:
            asset_class_weights = w_df.groupby("Asset Class")["Weight"].sum().reset_index()

    w_df.rename(columns={"Asset": "Ticker"}, inplace=True)
    desired_order = ["Company Name", "Ticker"]
    base_cols = [c for c in w_df.columns if c not in desired_order and c != "Weight"]
    final_cols = [c for c in (desired_order + base_cols + ["Weight"]) if c in w_df.columns]
    w_df = w_df[final_cols]

    brand_palette = ["#6d46b8", "#9d22a5", "#7e9e31", "#009fbd", "#7ccfff", "#d35400"]
    cols = st.columns(4)

    chart_configs = [
        (cols[0], "Asset allocation", "Distribution by individual selection", w_df, "Weight", "Ticker", f"{key_prefix}_asset_alloc"),
        (cols[1], "Sector allocation", "Exposure across industry sectors", sector_weights, "Weight", "Sector", f"{key_prefix}_sec_alloc"),
        (cols[2], "Asset class", "Breakdown by broad asset class", asset_class_weights, "Weight", "Asset Class", f"{key_prefix}_ac_alloc"),
        (cols[3], "Country allocation", "Geographic distribution of assets", country_weights, "Weight", "Country", f"{key_prefix}_country_alloc"),
    ]

    for col, title, subtitle, data, val_col, name_col, chart_key in chart_configs:
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div style="margin-bottom: 5px;">'
                    f'<div style="font-size: 18px; font-weight: 500; margin-bottom: 2px;">{title}</div>'
                    f'<div style="font-size: 13px; opacity: 0.6;">{subtitle}</div></div>',
                    unsafe_allow_html=True,
                )
                if data is not None and not data.empty:
                    fig = px.pie(data, values=val_col, names=name_col, hole=0.4, color_discrete_sequence=brand_palette)
                    fig.update_layout(
                        margin=dict(t=10, b=10, l=10, r=10), height=280,
                        showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
                    )
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                else:
                    st.info(f"No {title.lower().replace(' allocation', '')} data.")


# ── Page layout ───────────────────────────────────────────────────────

st.divider()

# 1. Sidebar — collect all user inputs
cfg = render_sidebar()

# 2. Page header
st.title("Analytics", anchor=False)
st.markdown("""
<div style="margin-top: -10px; opacity: 0.6; margin-bottom: 5px; font-size: 0.9em;">
    Analysis of your constructed portfolio
</div>
""", unsafe_allow_html=True)
st.markdown('<hr style="margin-top: 15px; margin-bottom: 25px; border: 0; height: 1px; background-color: #e6e6e6;" />', unsafe_allow_html=True)

# Track whether the user has already attempted a backtest at least once.
if "backtest_attempted" not in st.session_state:
    st.session_state["backtest_attempted"] = False

# Update click state before rendering the intro warning.
if cfg["create_portfolio_clicked"]:
    st.session_state["backtest_attempted"] = True

if "portfolio" not in st.session_state and not st.session_state["backtest_attempted"]:
    st.warning("Choose your parameters and click on **Run backtest**", icon=":material/info:")

# 3. Resolve weights for equal-weight / dynamic strategies
weights_vector = cfg["weights_vector"]
strategy_source = cfg["strategy_source"]
strategy_func = cfg["strategy_func"]
tickers = cfg["tickers"]

if strategy_source == "Equal-weight":
    n = len(tickers)
    weights_vector = np.array([1.0 / n] * n)
elif strategy_source != "Manual weights":
    if strategy_source == "Upload script" and strategy_func is None:
        st.warning("Please upload a valid strategy script.", icon=":material/info:")
        cfg["valid_weights"] = False

# 4. Run backtest
if cfg["create_portfolio_clicked"] and cfg["valid_weights"]:
    with st.spinner("Running backtest"):
        if cfg["data_source"] == "Download":
            prices = load_price_data(tickers, cfg["start_date"], cfg["end_date"])
        else:
            prices = cfg["prices_from_csv"]

        if prices.empty:
            st.error("No data found for selected assets.")
        else:
            asset_returns = compute_returns(prices)
            available_tickers = [t for t in tickers if t in asset_returns.columns]
            if len(available_tickers) != len(tickers):
                st.warning("Some assets failed to load data.")

            asset_returns = asset_returns[tickers]

            if strategy_source == "Manual weights":
                eval_weights = np.array(weights_vector)
                strategy_func = lambda df, w=eval_weights, t=tickers: pd.Series(w, index=t).reindex(df.columns).fillna(0)

            engine = DailyBacktestEngine(asset_returns)
            portfolio_returns, weights_history = engine.run(
                strategy_func=strategy_func,
                train_window_months=cfg["train_window"],
                test_window_months=cfg["test_window"],
                rebalance_freq="ME",
                window_type=cfg["window_type"].lower(),
            )

            if portfolio_returns.empty:
                st.error("Backtest returned no results. Check if the training window is larger than available data.")
                st.stop()

            weights_vector = weights_history.mean(axis=0).values
            start_date = portfolio_returns.index.min().date()
            asset_returns = asset_returns.loc[portfolio_returns.index]
            prices = prices.loc[portfolio_returns.index]

            metadata = fetch_asset_metadata(tickers)

            st.session_state["portfolio"] = {
                "tickers": tickers,
                "weights": weights_vector,
                "prices": prices,
                "asset_returns": asset_returns,
                "portfolio_returns": portfolio_returns,
                "start_date": start_date,
                "end_date": cfg["end_date"],
                "metadata": metadata,
                "benchmark": cfg["benchmark_ticker"],
                "strategy": cfg["strategy_source"],
                "show_toast": True,
            }


# 5. Display results (persisted across reruns via session state)
if "portfolio" in st.session_state:
    pf_data = st.session_state["portfolio"]
    portfolio_returns = pf_data["portfolio_returns"]
    benchmark_ticker = pf_data.get("benchmark", "^GSPC")

    # Toast + auto-collapse sidebar on first render
    if pf_data.get("show_toast", False):
        st.toast("Backtest successfully run !")
        st.session_state["portfolio"]["show_toast"] = False
        import streamlit.components.v1 as components
        components.html("""
        <script>
            var collapseBtn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
            if (collapseBtn) { collapseBtn.click(); }
            var sidebar = window.parent.document.querySelector('button[kind="headerNoPadding"]');
            if (!collapseBtn && sidebar) { sidebar.click(); }
        </script>
        """, height=0)

    # Composition pie charts
    render_portfolio_composition(
        pf_data["tickers"], pf_data["weights"],
        pf_data.get("metadata"), key_prefix="analytics_view",
    )

    # KPIs + metrics table
    comparison_df, drawdown, active_return, tracking_error, information_ratio = render_kpis(
        portfolio_returns, benchmark_ticker,
        pf_data["start_date"], pf_data["end_date"],
    )

    # Chart tabs
    tabs = st.tabs(["Performance", "Drawdown", "Volatility", "Correlation"])
    with tabs[0]:
        render_performance_tab(portfolio_returns, comparison_df)
    with tabs[1]:
        render_drawdown_tab(drawdown)
    with tabs[2]:
        render_volatility_tab(portfolio_returns, comparison_df)
    with tabs[3]:
        render_correlation_tab(pf_data["asset_returns"])

    # PDF export
    render_report_section(pf_data, comparison_df, active_return, tracking_error, information_ratio)
