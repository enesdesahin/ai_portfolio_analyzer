import os
import io
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import nest_asyncio

# To prevent asyncio loop errors with Streamlit and Playwright
nest_asyncio.apply()
from playwright.sync_api import sync_playwright

BRAND_COLORS = ["#3B2A22", "#705F57", "#000000", "#F1ECE1", "#D93025", "#34A853"]
BENCHMARK_COLOR = "#FBC356"
DONUT_PALETTE = ["#134aa5", "#7B2D8E", "#3B7A57", "#D4A017", "#D93025", "#705F57", "#8B7D74", "#A69B93", "#C1B8B1", "#34A853"]

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_cumulative_returns(results_df):
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
    has_benchmark = 'Benchmark' in results_df.columns
    fig, ax_cum = plt.subplots(figsize=(6, 2.8))
    r_daily = results_df['Strategy']
    cum_ret = (1 + r_daily).cumprod() - 1
    ax_cum.plot(cum_ret.index, cum_ret.values * 100, color='#004BAB', linewidth=0.8, label='Portfolio')
    if has_benchmark:
        bench_cum = (1 + results_df['Benchmark']).cumprod() - 1
        ax_cum.plot(bench_cum.index, bench_cum.values * 100, color=BENCHMARK_COLOR, linewidth=0.7, label='Benchmark')
        ax_cum.legend(loc='upper left', frameon=False, fontsize=7)
    ax_cum.set_ylabel('Cumulative Return', fontsize=7, color=BRAND_COLORS[1])
    ax_cum.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f%%'))
    ax_cum.spines['top'].set_visible(False)
    ax_cum.spines['right'].set_visible(False)
    ax_cum.spines['left'].set_visible(False)
    ax_cum.spines['bottom'].set_color(BRAND_COLORS[1])
    ax_cum.grid(axis='y', linestyle='--', alpha=0.4)
    ax_cum.tick_params(axis='both', which='both', length=0, colors=BRAND_COLORS[1], labelsize=7)
    ax_cum.xaxis.set_major_locator(mdates.YearLocator())
    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_cum.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_cum.set_xlabel('Year', fontsize=7, color=BRAND_COLORS[1])
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_drawdown(results_df):
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
    fig, ax_dd = plt.subplots(figsize=(6, 2.2))
    r_daily = results_df['Strategy']
    wealth = (1 + r_daily).cumprod()
    dd = (wealth - wealth.cummax()) / wealth.cummax()
    ax_dd.fill_between(dd.index, dd.values * 100, 0, color=BRAND_COLORS[4], alpha=0.25)
    ax_dd.plot(dd.index, dd.values * 100, color=BRAND_COLORS[4], linewidth=0.7)
    avg_dd = dd.mean() * 100
    ax_dd.axhline(y=avg_dd, color=BRAND_COLORS[4], linestyle='--', linewidth=0.8, alpha=0.6)
    ax_dd.set_ylabel('Drawdown', fontsize=7, color=BRAND_COLORS[1])
    ax_dd.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f%%'))
    ax_dd.spines['top'].set_visible(False)
    ax_dd.spines['right'].set_visible(False)
    ax_dd.spines['left'].set_visible(False)
    ax_dd.spines['bottom'].set_color(BRAND_COLORS[1])
    ax_dd.grid(axis='y', linestyle='--', alpha=0.4)
    ax_dd.tick_params(axis='both', which='both', length=0, colors=BRAND_COLORS[1], labelsize=7)
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_dd.set_xlabel('Year', fontsize=7, color=BRAND_COLORS[1])
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_distribution(results_df):
    import scipy.stats as stats
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
    has_benchmark = 'Benchmark' in results_df.columns
    fig, ax_dist = plt.subplots(figsize=(6, 2.2))
    r_daily = results_df['Strategy']
    r_mo = r_daily.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
    if has_benchmark:
        rb_monthly = results_df['Benchmark'].resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
        min_val = min(r_mo.min(), rb_monthly.min())
        max_val = max(r_mo.max(), rb_monthly.max())
    else:
        min_val, max_val = r_mo.min(), r_mo.max()
        
    common_bins = np.linspace(min_val, max_val, 20)
    ax_dist.hist(r_mo, bins=common_bins, color='#004BAB', alpha=0.7, density=True, edgecolor='none', rwidth=0.88, label='Portfolio')
    mu, sigma = r_mo.mean(), r_mo.std()
    x_dist = np.linspace(min(r_mo.min(), mu - 3*sigma), max(r_mo.max(), mu + 3*sigma), 200)
    ax_dist.plot(x_dist, stats.norm.pdf(x_dist, mu, sigma), color='#004BAB', linewidth=1.0)
    
    if has_benchmark:
        ax_dist.hist(rb_monthly, bins=common_bins, color=BENCHMARK_COLOR, alpha=0.5, density=True, edgecolor='none', rwidth=0.88, label='Benchmark')
        mu_b, sigma_b = rb_monthly.mean(), rb_monthly.std()
        ax_dist.plot(x_dist, stats.norm.pdf(x_dist, mu_b, sigma_b), color=BENCHMARK_COLOR, linewidth=1.0)
        ax_dist.legend(loc='upper right', frameon=False, fontsize=6)
        
    ax_dist.set_ylabel("Dist. Density", fontsize=7, color=BRAND_COLORS[1])
    ax_dist.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f%%'))
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    ax_dist.spines['left'].set_visible(False)
    ax_dist.spines['bottom'].set_color(BRAND_COLORS[1])
    ax_dist.grid(axis='y', linestyle='--', alpha=0.4)
    ax_dist.tick_params(axis='both', which='both', length=0, colors=BRAND_COLORS[1], labelsize=7)
    ax_dist.set_xlabel('Monthly Return', fontsize=7, color=BRAND_COLORS[1], labelpad=6)
    fig.tight_layout(pad=1.2)
    return _fig_to_base64(fig)


def plot_rolling_vol(results_df):
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
    has_benchmark = 'Benchmark' in results_df.columns
    fig, ax_vol = plt.subplots(figsize=(6, 2.2))
    r_daily = results_df['Strategy']
    window_6m = 126
    roll_vol = r_daily.rolling(window_6m).std() * np.sqrt(252)
    ax_vol.plot(roll_vol.index, roll_vol.values, color='#134aa5', linewidth=1.2, label='Strategy')
    if has_benchmark:
        rb_daily = results_df['Benchmark']
        roll_vol_b = rb_daily.rolling(window_6m).std() * np.sqrt(252)
        ax_vol.plot(roll_vol_b.index, roll_vol_b.values, color=BENCHMARK_COLOR, linewidth=1.0, alpha=0.8, label='Benchmark')
        ax_vol.legend(loc='upper right', frameon=False, fontsize=6)
    avg_vol = roll_vol.mean()
    ax_vol.axhline(y=avg_vol, color='#d93025', linestyle='--', linewidth=1.2)
    ax_vol.axhline(y=0.0, color='#000000', linestyle='--', linewidth=0.8)
    ax_vol.set_ylabel("Rolling Vol (%)", fontsize=7, color=BRAND_COLORS[1])
    ax_vol.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax_vol.spines['top'].set_visible(False)
    ax_vol.spines['right'].set_visible(False)
    ax_vol.spines['left'].set_visible(False)
    ax_vol.spines['bottom'].set_color(BRAND_COLORS[1])
    ax_vol.grid(axis='y', linestyle='--', alpha=0.4)
    ax_vol.tick_params(axis='both', which='both', length=0, colors=BRAND_COLORS[1], labelsize=7)
    ax_vol.xaxis.set_major_locator(mdates.YearLocator())
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_vol.set_xlabel('Year', fontsize=7, color=BRAND_COLORS[1])
    fig.tight_layout()
    return _fig_to_base64(fig)

def plot_monthly_heatmap(results_df):
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
    df = results_df[['Strategy']].copy()
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    monthly_ret = df.groupby(['Year', 'Month'])['Strategy'].apply(lambda x: (1 + x).prod() - 1).unstack()
    for m in range(1, 13):
        if m not in monthly_ret.columns:
            monthly_ret[m] = np.nan
    cols = list(range(1, 13))
    monthly_ret = monthly_ret[[c for c in cols if c in monthly_ret.columns]]
    month_names = {1:'JAN', 2:'FEB', 3:'MAR', 4:'APR', 5:'MAY', 6:'JUN',
                   7:'JUL', 8:'AUG', 9:'SEP', 10:'OCT', 11:'NOV', 12:'DEC'}
    monthly_ret.rename(columns=month_names, inplace=True)
    n_years = len(monthly_ret)
    from matplotlib.colors import LinearSegmentedColormap
    fig, ax = plt.subplots(figsize=(6, max(2.0, n_years * 0.28)))
    
    # Custom colormap matching requested hex codes: Red -> Yellow -> Green
    cmap_colors = ['#B5001F', '#ECA38A', '#FEFFB6', '#ACDF84', '#62C958']
    cmap = LinearSegmentedColormap.from_list('custom_ryg', cmap_colors)
    
    data = monthly_ret * 100
    vmax = max(abs(data.min().min()), abs(data.max().max())) if not data.empty else 10
    sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, center=0,
                vmin=-vmax, vmax=vmax,
                cbar=False, linewidths=1.0, ax=ax, linecolor='white',
                annot_kws={"size": 7, "color": "#000000"})
    ax.set_ylabel("Monthly Returns (%)", fontsize=7, color=BRAND_COLORS[1], labelpad=8)
    ax.set_xlabel("")
    # Month labels at bottom
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', rotation=0, labelsize=7, colors='#555', length=0)
    ax.tick_params(axis='y', rotation=0, labelsize=7, colors='#555', length=0)
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.5)
    return _fig_to_base64(fig)


def generate_pdf_report(results_df: pd.DataFrame, metrics_df: pd.DataFrame,
                        tickers=None, weights=None, metadata=None, pdf_context=None,
                        ai_commentary: str = "",
                        ai_metrics_comment: str = "") -> bytes:
    """Generates an A4 institutional tear sheet."""
    
    # --- 1. Allocation tables (Top Holdings, Industries, Countries) ---
    alloc_cards_html = ""
    holdings_table_html = ""
    sector_table_html = ""
    country_table_html = ""
    if tickers is not None and weights is not None:
        # Build name lookup: ticker -> company name (uppercase)
        name_lookup = {}
        if metadata is not None and not metadata.empty and "Company Name" in metadata.columns:
            for ticker in tickers:
                raw_name = metadata.loc[ticker, "Company Name"] if ticker in metadata.index else ticker
                name_lookup[ticker] = str(raw_name)
        else:
            for ticker in tickers:
                name_lookup[ticker] = str(ticker)

        # Sort by weight descending, use company name for display
        sorted_pairs = [(name_lookup.get(t, str(t)), w)
                        for t, w in sorted(zip(tickers, weights), key=lambda x: x[1], reverse=True)]
        
        def _alloc_table(title, rows, max_rows=10):
            """Build a compact HTML table for allocation breakdown."""
            rows = rows[:max_rows]
            html  = f'<div class="ac"><table class="alloc-tbl">'
            html += f'<colgroup><col style="width:80%"/><col style="width:20%"/></colgroup>'
            html += f'<thead><tr><td style="text-align:left">{title}</td><td style="text-align:right">Weight (%)</td></tr></thead><tbody>'
            for i, (name, w) in enumerate(rows):
                is_last = (i == len(rows) - 1)
                border = '1px solid #000000' if is_last else '1px solid #e8ecf2'
                cell   = f'border-bottom:{border};padding:6px 8px 6px 0;font-size:9px;color:#000000;'
                cell_r = f'border-bottom:{border};padding:6px 8px;font-size:9px;text-align:right;color:#000000;'
                html += f'<tr><td style="{cell}">{name}</td><td style="{cell_r}">{w*100:.2f}</td></tr>'
            html += '</tbody></table></div>'
            return html
        
        # Holdings table HTML + store data for alloc commentary
        holdings_table_html = _alloc_table("Holdings", sorted_pairs)
        sector_table_html = ""
        country_table_html = ""
        
        # Sector/Industry + Country tables
        if metadata is not None and not metadata.empty:
            w_df = pd.DataFrame({"Asset": tickers, "Weight": weights})
            w_df = w_df.merge(metadata, left_on="Asset", right_index=True, how="left")
            if "Sector" in w_df.columns:
                sector_data = w_df.groupby("Sector")["Weight"].sum().sort_values(ascending=False)
                sector_table_html = _alloc_table("Industries", list(zip(sector_data.index, sector_data.values)))
            if "Country" in w_df.columns:
                country_data = w_df.groupby("Country")["Weight"].sum().sort_values(ascending=False)
                country_table_html = _alloc_table("Country weights", list(zip(country_data.index, country_data.values)), max_rows=6)
        
        alloc_cards_html = holdings_table_html  # used below in layout

    # --- 2. Compute metrics ---
    strategy_returns = results_df['Strategy']
    has_benchmark = 'Benchmark' in results_df.columns

    def compute_metrics(returns):
        total_ret = (1 + returns).cumprod().iloc[-1] - 1
        n_years = len(returns) / 252
        cagr_val = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252) / ann_vol if ann_vol != 0 else 0
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = (returns.mean() * 252) / downside if downside != 0 else 0
        w = (1 + returns).cumprod()
        dd = (w - w.cummax()) / w.cummax()
        max_dd = dd.min()
        calmar = cagr_val / abs(max_dd) if max_dd != 0 else 0
        skew_val = returns.skew()
        kurt_val = returns.kurtosis()
        win_rate = (returns > 0).sum() / len(returns)
        best_day = returns.max()
        worst_day = returns.min()
        today = returns.index[-1]
        roll = {}
        for lbl, start in {'MTD': pd.Timestamp(today.year, today.month, 1),
                            'YTD': pd.Timestamp(today.year, 1, 1),
                            '1Y': today - pd.DateOffset(years=1)}.items():
            mask = returns.index >= start
            roll[lbl] = (1 + returns[mask]).prod() - 1 if mask.any() else np.nan
        return {'total_return': total_ret, 'cagr': cagr_val, 'ann_vol': ann_vol,
                'max_dd': max_dd, 'skew': skew_val, 'kurtosis': kurt_val,
                'win_rate': win_rate, 'best_day': best_day, 'worst_day': worst_day,
                'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'rolling': roll}

    s = compute_metrics(strategy_returns)
    bm = compute_metrics(results_df['Benchmark']) if has_benchmark else None

    fp = lambda v: f"{v:.2%}"
    fr = lambda v: f"{v:.2f}"
    na = "—"
    def bv(k, f): return f(bm[k]) if bm else na

    # Analysis period
    period_start = strategy_returns.index[0].strftime('%b %Y')
    period_end = strategy_returns.index[-1].strftime('%b %Y')
    gen_date = datetime.now().strftime('%B %d, %Y')

    groups = [
        [("Cumulative return", fp(s['total_return']), bv('total_return', fp)),
         ("CAGR%", fp(s['cagr']), bv('cagr', fp))],
        [("Sharpe", fr(s['sharpe']), bv('sharpe', fr)),
         ("Sortino", fr(s['sortino']), bv('sortino', fr)),
         ("Calmar", fr(s['calmar']), bv('calmar', fr))],
        [("Max drawdown", fp(s['max_dd']), bv('max_dd', fp)),
         ("Annualized volatility", fp(s['ann_vol']), bv('ann_vol', fp))],
        [("Win rate", fp(s['win_rate']), bv('win_rate', fp)),
         ("Best day", fp(s['best_day']), bv('best_day', fp)),
         ("Worst day", fp(s['worst_day']), bv('worst_day', fp))]
    ]

    # Rolling returns (MTD, YTD, 1Y only)
    rg = []
    for lbl in ['MTD', 'YTD', '1Y']:
        sv = s['rolling'].get(lbl, np.nan)
        bval = bm['rolling'].get(lbl, np.nan) if bm else np.nan
        if pd.notna(sv):
            rg.append((lbl, fp(sv), fp(bval) if pd.notna(bval) else na))
    if rg: groups.append(rg)

    metrics_rows = ""
    all_metrics = [(name, sv, bval) for g in groups for name, sv, bval in g]
    for i, (name, sv, bval) in enumerate(all_metrics):
        is_last = (i == len(all_metrics) - 1)
        border_style = f'border-bottom:1px solid #000000;' if is_last else ''
        td_extra = f' style="{border_style}"' if is_last else ''
        metrics_rows += f'<tr><td{td_extra}>{name}</td><td{td_extra}>{sv}</td><td{td_extra}>{bval}</td></tr>'

    # --- 3. Build chart images ---
    cum_img = plot_cumulative_returns(results_df)
    dd_img = plot_drawdown(results_df)
    dist_img = plot_distribution(results_df)
    vol_img = plot_rolling_vol(results_df)
    heat_img = plot_monthly_heatmap(results_df)

    # --- 4. HTML ---
    html_content = f"""
    <!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
    <title>Tear Sheet</title>
    <link href="https://fonts.googleapis.com" rel="preconnect"/>
    <link crossorigin="" href="https://fonts.gstatic.com" rel="preconnect"/>
    <link href="https://fonts.googleapis.com/css2?family=Rethink+Sans:wght@400;500;600;700&display=swap" rel="stylesheet"/>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ background:#fff; font-family:'Rethink Sans',sans-serif; color:#333; font-size:9px; }}
        .page {{ width:210mm; min-height:297mm; max-height:297mm; overflow:hidden; padding:16mm 14mm 10mm; }}
        
        .hdr {{ border-top:4px solid #000000; border-bottom:1.5px solid #000000; padding:10px 0 14px 0; margin-bottom:16px; text-align:left; }}
        .hdr-sub {{ font-size:8px; text-transform:uppercase; color:#000000; letter-spacing:1.5px; margin-bottom:3px; margin-left:1.5px; }}
        .hdr-title {{ font-size:32px; font-weight:600; color:#000000; line-height:1.1; margin-left:-2px; }}
        .hdr-meta {{ font-size:10px; color:#444; margin-top:4px; font-weight:400; }}
        .hdr-meta b {{ color:#000; font-weight:700; }}
        
        .main {{ display:flex; gap:24px; }}
        .charts {{ flex:0 0 54%; }}
        .sidebar {{ flex:1; display:flex; flex-direction:column; gap:0; }}
        .chart-box {{ margin-bottom:10px; }}
        .chart-box img {{ width:100%; height:auto; display:block; }}
        
        .alloc-row {{ display:flex; flex-direction:column; gap:0; }}
        .ac {{ margin-bottom:20px; }}
        
        table, .alloc-tbl {{ width:100%; border-collapse:collapse; font-size:9px; }}
        table thead td, .alloc-tbl thead td {{
            font-weight:700; color:#000000; padding:6px 8px; font-size:9px;
            border-top:1px solid #000000; border-bottom:1px solid #000000;
        }}
        table tbody td, .alloc-tbl tbody td {{
            padding:6px 8px; border-bottom:1px solid #c5cedf; color:#000000;
            text-align:right; white-space:nowrap; font-size:9px; font-weight:400;
        }}
        table tbody td:first-child, .alloc-tbl tbody td:first-child {{ text-align:left; padding-left:0; }}
        table thead td:first-child, .alloc-tbl thead td:first-child {{ padding-left:0; }}
        tr.sp td {{ padding:0; height:8px; border-bottom:none; }}
        
        .metrics-block {{ margin-top:0; }}
        .alloc-page {{ width:44%; }}
        .chart-label {{ font-size:10.5px; font-weight:700; color:#000000; margin-bottom:1px; letter-spacing:0.2px; }}
        .chart-sublabel {{ font-size:7.5px; font-weight:400; color:#5f6368; display:block; margin-bottom:6px; margin-top:1.5px; letter-spacing:0.1px; }}
        .ai-block {{
            background-color: #f8f9fa;
            padding: 12px 15px;
            font-size: 10px;
            line-height: 1.5;
            color: #333333;
            margin-top: 15px;
            margin-bottom: 10px; /* Kept from original */
            text-align: justify; /* Kept from original */
            hyphens: auto; /* Kept from original */
            border-radius: 0px;
            max-height: 485px;
            overflow: hidden;
        }}
        .ai-block-title {{ font-size:11.5px; font-weight:700; color:#000000; margin-bottom:8px; letter-spacing:0.2px; display:block; }}
        .ai-block p {{ margin:0 0 6px 0; }}
        .ai-block p:last-child {{ margin-bottom:0; }}
        .ai-rec {{ font-size:7.5px; color:#000000; line-height:1.6; padding:8px 12px;
                   border:1px solid #000000; margin-top:14px;
                   text-align:justify; hyphens:auto;
                   border-radius:0px; }}
        .ai-rec strong {{ font-size:8px; display:block; margin-bottom:3px; color:#000000; }}
        .p2-top {{ display:flex; gap:16px; align-items:flex-start; }}
        .p2-holdings {{ flex:0 0 44%; }}
        .p2-alloc-comment {{ flex:1; }}
        .p2-bottom {{ margin-top:8px; }}
    </style>
    </head>
    <body>
    <div class="page">
        <div class="hdr">
            <div class="hdr-sub">{gen_date}</div>
            <div class="hdr-title">Portfolio analysis</div>
            <div class="hdr-meta">
                <b>Strategy :</b> {pdf_context.get('strategy', 'Global Portfolio') if pdf_context else 'Global Portfolio'} <span style="margin:0 8px;color:#ccc;">|</span>
                <b>Benchmark :</b> {pdf_context.get('benchmark', 'S&P 500') if pdf_context else 'S&P 500'} <span style="margin:0 8px;color:#ccc;">|</span>
                <b>Period :</b> {pdf_context.get('date_range', 'Jan 2021 - Dec 2026') if pdf_context else 'Jan 2021 - Dec 2026'}
            </div>
        </div>
        {f'<div class="ai-block">{ai_commentary}</div>' if ai_commentary else ''}

        <div class="main">
            <div class="charts">
                <div class="chart-box">
                    <div class="chart-label">Cumulative return<span class="chart-sublabel">Historical performance of the portfolio vs benchmark</span></div>
                    <img src="data:image/png;base64,{cum_img}"/>
                </div>
                <div class="chart-box">
                    <div class="chart-label">Distribution of monthly returns<span class="chart-sublabel">Histogram of historical monthly returns</span></div>
                    <img src="data:image/png;base64,{dist_img}"/>
                </div>
                <div class="chart-box">
                    <div class="chart-label">Rolling volatility (6-months)<span class="chart-sublabel">Annualized standard deviation over 6 months</span></div>
                    <img src="data:image/png;base64,{vol_img}"/>
                </div>
                <div class="chart-box">
                    <div class="chart-label">Drawdown<span class="chart-sublabel">Historical drawdown of the portfolio</span></div>
                    <img src="data:image/png;base64,{dd_img}"/>
                </div>
                <div class="chart-box">
                    <div class="chart-label">Monthly returns<span class="chart-sublabel">Historical performance matrix (%)</span></div>
                    <img src="data:image/png;base64,{heat_img}"/>
                </div>
            </div>
            <div class="sidebar">
                <div class="metrics-block">
                    <table>
                        <colgroup><col style="width:50%"/><col style="width:25%"/><col style="width:25%"/></colgroup>
                        <thead><tr>
                            <td style="text-align:left">Metric</td>
                            <td style="text-align:right">Portfolio</td>
                            <td style="text-align:right">Benchmark</td>
                        </tr></thead>
                        <tbody>{metrics_rows}</tbody>
                    </table>
                </div>
                {('<div class="ai-block" style="margin-top:16px;"><span class="ai-block-title">Portfolio insights</span>' + "".join(f"<p>{p.strip()}</p>" for p in ai_metrics_comment.split(chr(10)+chr(10)) if p.strip()) + "</div>") if ai_metrics_comment else ""}
                {('<div style="font-size: 7.5px; color: #888888; text-align: justify; margin-top: 4px; padding: 0 4px; line-height: 1.3;">* This analysis has been automatically generated by our internal AI model based on the portfolio&#39;s historical metrics and is provided for informational purposes only.</div>') if ai_metrics_comment else ""}
            </div>
        </div>
    </div>

    </body></html>
    """

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_content(html_content, wait_until='networkidle')
            pdf_bytes = page.pdf(format='A4', print_background=True,
                                 margin={'top': '0px', 'right': '0px', 'bottom': '0px', 'left': '0px'})
            browser.close()
            return pdf_bytes
    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise e
