"""
Sidebar component — all user input controls for the portfolio builder.
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from portfolio_analyzer.core.optimization import (
    mvo_strategy,
    max_sharpe_strategy,
    risk_parity_strategy,
    black_litterman_strategy,
)
from portfolio_analyzer.app.components.social import render_social_links


# ── Default ticker universe ──────────────────────────────────────────

DEFAULT_TICKERS = [
    # US Equities
    "SPY", "QQQ", "IWM", "VTI",
    # International Equities
    "EFA", "EEM", "VEA", "VWO",
    # Fixed Income
    "AGG", "TLT", "IEF", "LQD", "HYG",
    # Commodities (ETFs & Futures)
    "GLD", "SLV", "GSG", "PDBC",
    "GC=F", "SI=F", "CL=F", "NG=F", "ZC=F",
    # Real Estate
    "VNQ", "IYR",
    # Cryptocurrencies
    "BTC-USD", "ETH-USD",
    # Top US Stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    # Top European Stocks
    "MC.PA", "ASML.AS", "SAP.DE", "TTE.PA", "NOVO-B.CO", "SAN.PA", "SIE.DE", "OR.PA",
]

STRATEGY_OPTIONS = {
    "Equal-weight": lambda df: pd.Series(1.0 / len(df.columns), index=df.columns),
    "Manual weights": None,
    "Mean-variance": mvo_strategy,
    "Max sharpe": max_sharpe_strategy,
    "Risk parity": risk_parity_strategy,
    "Black-Litterman": black_litterman_strategy,
    "Upload script": None,
}


# ── Helpers ───────────────────────────────────────────────────────────

def _init_session_state():
    """Initialise session-state keys used by the sidebar."""
    if "available_tickers" not in st.session_state:
        st.session_state.available_tickers = list(DEFAULT_TICKERS)
    if "selected_tickers" not in st.session_state:
        st.session_state.selected_tickers = ["AAPL", "MSFT", "AMZN"]


def _add_ticker():
    """Callback: validate and add a user-typed ticker."""
    if "new_ticker_input" not in st.session_state:
        return
    new_ticker = st.session_state.new_ticker_input.strip().upper()
    if not new_ticker:
        return
    if new_ticker not in st.session_state.available_tickers:
        try:
            tick = yf.Ticker(new_ticker)
            hist = tick.history(period="1d")
            if not hist.empty:
                st.session_state.available_tickers.append(new_ticker)
                st.session_state.selected_tickers.append(new_ticker)
                st.toast(f"Added {new_ticker} to options")
                st.session_state.new_ticker_input = ""
            else:
                st.toast(f"Could not find data for {new_ticker}")
        except Exception:
            st.toast(f"Error validating {new_ticker}")
    else:
        if new_ticker not in st.session_state.selected_tickers:
            st.session_state.selected_tickers.append(new_ticker)
            st.toast(f"{new_ticker} selected")
        st.session_state.new_ticker_input = ""


# ── Main render function ─────────────────────────────────────────────

def render_sidebar() -> dict:
    """
    Render the full sidebar and return a configuration dict with all inputs.

    Returns
    -------
    dict with keys:
        tickers, start_date, end_date, benchmark_ticker,
        data_source, prices_from_csv,
        strategy_source, strategy_func,
        weights_vector, valid_weights,
        train_window, test_window, window_type,
        create_portfolio_clicked
    """
    _init_session_state()

    # ── Data source ───────────────────────────────────────────────
    st.sidebar.header("Input data")
    data_source = st.sidebar.selectbox("Data source", ["Download", "Upload CSV"])

    tickers: list[str] = []
    start_date = None
    end_date = None
    prices_from_csv = pd.DataFrame()
    benchmark_ticker = "^GSPC"

    if data_source == "Download":
        tickers = st.sidebar.multiselect(
            "Tickers",
            options=st.session_state.available_tickers,
            default=st.session_state.selected_tickers,
            key="ticker_multiselect",
        )
        if tickers != st.session_state.selected_tickers:
            st.session_state.selected_tickers = tickers

        st.sidebar.text_input(
            "Add a ticker",
            key="new_ticker_input",
            on_change=_add_ticker,
            placeholder="e.g. BRK-B",
            help="Type a ticker and hit Enter to add it to the list.",
        )

        benchmark_ticker = st.sidebar.selectbox(
            "Select benchmark",
            options=["^GSPC", "^IXIC", "^RUT", "ACWI", "AGG", "SPY", "QQQ"],
            index=0,
        )

        if len(tickers) == 0:
            st.warning("Please select at least one ticker", icon=":material/info:")
            st.stop()

        default_start = pd.to_datetime("2020-01-01")
        default_end = pd.to_datetime("today")
        start_date = st.sidebar.date_input("Start date", value=default_start)
        end_date = st.sidebar.date_input("End date", value=default_end)

        if start_date >= end_date:
            st.sidebar.error("Start date must be before end date.")
            st.stop()
    else:
        uploaded_file = st.sidebar.file_uploader("Upload market data (CSV)", type=["csv"])
        if uploaded_file is None:
            st.info(
                "Please upload a CSV file with dates as index and tickers as columns.",
                icon=":material/info:",
            )
            st.stop()

        try:
            prices_from_csv = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            tickers = list(prices_from_csv.columns)
            prices_from_csv.index = pd.to_datetime(prices_from_csv.index)
            prices_from_csv.sort_index(inplace=True)
            start_date = prices_from_csv.index.min().date()
            end_date = prices_from_csv.index.max().date()

            if len(tickers) == 0:
                st.sidebar.error("CSV must contain at least one column of asset prices.")
                st.stop()

            benchmark_ticker = st.sidebar.selectbox(
                "Select benchmark",
                options=["^GSPC", "^IXIC", "^RUT", "ACWI", "AGG", "SPY", "QQQ"],
                index=0,
            )
            st.sidebar.success(f"Loaded {len(tickers)} assets from {start_date} to {end_date}")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")
            st.stop()

    # ── Strategy / weights ────────────────────────────────────────
    valid_weights = True
    strategy_func = None
    weights_vector: np.ndarray | list = []

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.header("Configuration")

    strategy_source = st.sidebar.selectbox("Strategy", list(STRATEGY_OPTIONS.keys()))
    strategy_func = STRATEGY_OPTIONS.get(strategy_source)

    if strategy_source == "Manual weights":
        st.sidebar.text("Custom allocation")
        n = len(tickers)
        base_w = round(100.0 / n, 2) if n > 0 else 0.0
        defaults = [base_w] * n
        if n > 0:
            defaults[-1] = round(100.0 - base_w * (n - 1), 2)
        manual_weights = []
        for i, ticker in enumerate(tickers):
            w = (
                st.sidebar.number_input(
                    f"{ticker} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=defaults[i],
                    step=0.5,
                    key=f"manual_w_{ticker}",
                )
                / 100
            )
            manual_weights.append(w)
        weights_vector = np.array(manual_weights)
        total_weight = weights_vector.sum()
        st.sidebar.write(f"Total weight: {total_weight:.2%}")
        if abs(total_weight - 1.0) <= 0.01:
            st.sidebar.success("Weights are valid", icon=":material/info:")
        else:
            st.sidebar.error("Weights must sum to 100% (±1%)", icon=":material/error:")
            valid_weights = False
            st.sidebar.warning(
                "Please adjust weights before running the backtest",
                icon=":material/info:",
            )

    elif strategy_source == "Upload script":
        uploaded_script = st.sidebar.file_uploader("Upload strategy script (.py)", type=["py"])
        st.sidebar.info("Script must contain a function `get_weights(df)` returning a pd.Series.")
        if uploaded_script:
            script_content = uploaded_script.read().decode("utf-8")
            local_scope: dict = {}
            try:
                exec(script_content, globals(), local_scope)
                if "get_weights" in local_scope:
                    strategy_func = local_scope["get_weights"]
                    st.sidebar.success("Strategy loaded: get_weights")
                else:
                    st.sidebar.error("Function `get_weights(df)` not found in script.")
            except Exception as e:
                st.sidebar.error(f"Error loading script: {e}")

    # ── Backtest parameters ───────────────────────────────────────
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.header("Backtest parameters")
    train_window = st.sidebar.number_input(
        "Training window (months)",
        min_value=1,
        value=12,
        help="Number of months of historical data used to calculate portfolio weights",
    )
    test_window = st.sidebar.number_input(
        "Test window (months)",
        min_value=1,
        value=1,
        help="Number of months to hold the portfolio before rebalancing",
    )
    window_type = st.sidebar.selectbox("Window type", ["Rolling", "Expanding"])

    render_social_links()

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    create_portfolio_clicked = st.sidebar.button("Run backtest", disabled=False)

    return {
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "benchmark_ticker": benchmark_ticker,
        "data_source": data_source,
        "prices_from_csv": prices_from_csv,
        "strategy_source": strategy_source,
        "strategy_func": strategy_func,
        "weights_vector": weights_vector,
        "valid_weights": valid_weights,
        "train_window": train_window,
        "test_window": test_window,
        "window_type": window_type,
        "create_portfolio_clicked": create_portfolio_clicked,
    }
