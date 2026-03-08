"""
Chart renderers for the analytics tabs.
Each function renders one Streamlit tab's worth of Plotly charts.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ── Performance tab ──────────────────────────────────────────────────

def render_performance_tab(portfolio_returns: pd.Series, comparison_df: pd.DataFrame | None):
    """Render the Performance tab: cumulative return + active return chart."""
    st.markdown("#### Performance")
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        Growth of $1 invested over the specific period
    </div>
    """, unsafe_allow_html=True)

    cum_ret_series = (1 + portfolio_returns).cumprod()

    if comparison_df is not None and not comparison_df.empty:
        cum_df = (1 + comparison_df).cumprod().reset_index()
        cum_melt = cum_df.melt(id_vars="Date", var_name="Asset", value_name="Cumulative return")

        fig_perf = px.line(
            cum_melt, x="Date", y="Cumulative return", color="Asset",
            color_discrete_map={"Portfolio": "#134aa5", "Benchmark": "#FBC356"},
        )
        fig_perf.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=400,
            xaxis_title="Date",
            yaxis_title="Growth of $1",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        )
    else:
        perf_df = pd.DataFrame(cum_ret_series, columns=["Cumulative return"])
        fig_perf = px.area(perf_df, x=perf_df.index, y="Cumulative return")
        final_val = cum_ret_series.iloc[-1]
        if final_val < 1.0:
            color, fill = "#d93025", "rgba(217, 48, 37, 0.2)"
        else:
            color, fill = "#00cc96", "rgba(0, 204, 150, 0.2)"
        fig_perf.update_traces(line_color=color, fillcolor=fill)
        fig_perf.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Date",
            yaxis_title="Cumulative return",
            showlegend=False,
            height=400,
        )

    st.plotly_chart(fig_perf, use_container_width=True, key="unified_perf_chart")

    # ── Active return chart ───────────────────────────────────────
    if comparison_df is not None and not comparison_df.empty:
        st.markdown("#### Active return")
        st.markdown("""
        <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
            Difference in cumulative returns
        </div>
        """, unsafe_allow_html=True)

        p_ret = comparison_df["Portfolio"]
        b_ret = comparison_df["Benchmark"]
        active_cum_series = (1 + p_ret).cumprod() - (1 + b_ret).cumprod()

        pos_series = active_cum_series.copy()
        neg_series = active_cum_series.copy()
        pos_series[active_cum_series < 0] = None
        neg_series[active_cum_series >= 0] = None

        for i in range(1, len(active_cum_series)):
            curr = active_cum_series.iloc[i]
            prev = active_cum_series.iloc[i - 1]
            if (curr >= 0 and prev < 0) or (curr < 0 and prev >= 0):
                pos_series.iloc[i] = 0
                pos_series.iloc[i - 1] = 0 if pd.isna(pos_series.iloc[i - 1]) else pos_series.iloc[i - 1]
                neg_series.iloc[i] = 0
                neg_series.iloc[i - 1] = 0 if pd.isna(neg_series.iloc[i - 1]) else neg_series.iloc[i - 1]

        fig_active = go.Figure()
        fig_active.add_trace(go.Scatter(
            x=active_cum_series.index, y=pos_series,
            mode="lines", line=dict(color="#134aa5", width=1.5),
            fill="tozeroy", fillcolor="rgba(19, 74, 165, 0.2)",
            name="Outperformance", showlegend=False,
        ))
        fig_active.add_trace(go.Scatter(
            x=active_cum_series.index, y=neg_series,
            mode="lines", line=dict(color="#d93025", width=1.5),
            fill="tozeroy", fillcolor="rgba(217, 48, 37, 0.2)",
            name="Underperformance", showlegend=False,
        ))
        fig_active.add_hline(y=0, line_dash="dash", line_color="#202124", opacity=0.5)
        fig_active.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=350,
            xaxis_title="Date",
            yaxis_title="Active return",
            showlegend=False,
        )
        st.plotly_chart(fig_active, use_container_width=True, key="unified_active_chart")


# ── Drawdown tab ─────────────────────────────────────────────────────

def render_drawdown_tab(drawdown: pd.Series):
    """Render the Drawdown tab."""
    st.markdown("#### Drawdown")
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        Decline from the historical peak value
    </div>
    """, unsafe_allow_html=True)

    dd_df = pd.DataFrame(drawdown, columns=["Drawdown"])
    fig_dd = px.area(dd_df, x=dd_df.index, y="Drawdown")
    fig_dd.update_traces(line_color="#d93025", fillcolor="rgba(217, 48, 37, 0.2)", line_width=1.5)
    avg_dd = drawdown.mean()
    fig_dd.add_hline(
        y=avg_dd, line_dash="dash", line_color="#d93025", line_width=1.5,
        annotation_text=f"Avg: {avg_dd:.1%}", annotation_position="top left",
    )
    fig_dd.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Date",
        yaxis_title="Drawdown",
        showlegend=False,
        height=400,
    )
    st.plotly_chart(fig_dd, use_container_width=True, key="unified_dd_chart")


# ── Volatility tab ───────────────────────────────────────────────────

def render_volatility_tab(portfolio_returns: pd.Series, comparison_df: pd.DataFrame | None):
    """Render the Volatility tab: rolling vol, rolling Sharpe, rolling Beta."""
    # ── Rolling Volatility ────────────────────────────────────────
    st.markdown("#### Rolling volatility")
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        21-day annualized rolling volatility
    </div>
    """, unsafe_allow_html=True)

    rolling_vol = portfolio_returns.rolling(window=21).std() * np.sqrt(252)
    vol_df = pd.DataFrame(rolling_vol, columns=["Volatility"])

    fig_vol = px.area(vol_df, x=vol_df.index, y="Volatility")
    fig_vol.update_traces(line_color="#134aa5", fillcolor="rgba(19, 74, 165, 0.2)", line_width=1.5)
    avg_vol = rolling_vol.mean()
    fig_vol.add_hline(
        y=avg_vol, line_dash="dash", line_color="#d93025", line_width=1.5,
        annotation_text=f"Avg: {avg_vol:.1%}", annotation_position="top left",
    )
    fig_vol.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Date",
        yaxis_title="Annualized volatility",
        showlegend=False,
        height=300,
    )
    st.plotly_chart(fig_vol, use_container_width=True, key="unified_vol_chart")

    # ── Rolling Sharpe ────────────────────────────────────────────
    st.markdown("#### Rolling Sharpe")
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        6-month (126 trading days) annualized Sharpe ratio
    </div>
    """, unsafe_allow_html=True)

    window = 126
    rolling_ret = portfolio_returns.rolling(window=window).mean() * 252
    rolling_std = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = rolling_ret / rolling_std

    sharpe_df = pd.DataFrame(rolling_sharpe, columns=["Sharpe ratio"])
    fig_sharpe = px.line(sharpe_df, x=sharpe_df.index, y="Sharpe ratio")
    fig_sharpe.update_traces(line_color="#2B75B5", line_width=1.5)
    fig_sharpe.add_hline(
        y=1.0, line_dash="dash", line_color="#00cc96", opacity=0.8,
        annotation_text="Good (>1.0)", annotation_position="top left",
    )
    fig_sharpe.add_hline(y=0.0, line_dash="dash", line_color="#d93025", opacity=0.8)
    fig_sharpe.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="",
        yaxis_title="Sharpe ratio",
        showlegend=False,
        height=300,
    )
    st.plotly_chart(fig_sharpe, use_container_width=True, key="unified_sharpe_chart")

    # ── Rolling Beta ──────────────────────────────────────────────
    has_benchmark = comparison_df is not None and not comparison_df.empty
    if has_benchmark:
        st.markdown("#### Rolling Beta to Benchmark")
        st.markdown("""
        <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
            6-month (126 trading days) rolling beta to benchmark
        </div>
        """, unsafe_allow_html=True)

        p_ret = comparison_df["Portfolio"]
        b_ret = comparison_df["Benchmark"]
        rolling_cov = p_ret.rolling(window=window).cov(b_ret)
        rolling_bm_var = b_ret.rolling(window=window).var()
        rolling_beta = rolling_cov / rolling_bm_var

        beta_df = pd.DataFrame(rolling_beta, columns=["Beta"])
        fig_beta = px.line(beta_df, x=beta_df.index, y="Beta")
        fig_beta.update_traces(line_color="#8E44AD", line_width=1.5)
        fig_beta.add_hline(
            y=1.0, line_dash="dash", line_color="#202124", opacity=0.5,
            annotation_text="Market Beta (1.0)", annotation_position="top left",
        )
        fig_beta.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="",
            yaxis_title="Beta",
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig_beta, use_container_width=True, key="unified_beta_chart")


# ── Correlation tab ──────────────────────────────────────────────────

def render_correlation_tab(asset_returns: pd.DataFrame):
    """Render the Correlation tab: heatmap of asset return correlations."""
    st.markdown("#### Correlation matrix")
    st.markdown("""
    <div style="margin-top: -10px; opacity: 0.6; margin-bottom: 20px; font-size: 0.9em;">
        Linear correlation between asset returns over the testing period
    </div>
    """, unsafe_allow_html=True)

    corr_matrix = asset_returns.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="1",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
    )

    fig_corr.update_coloraxes(colorbar=dict(
        lenmode="fraction",
        len=1.0,
        yanchor="middle",
        y=0.5,
    ))

    fig_corr.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=400,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.plotly_chart(fig_corr, use_container_width=True, key="unified_corr_chart")
