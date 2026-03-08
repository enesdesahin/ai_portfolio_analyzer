# AI Portfolio Analyzer

AI Portfolio Analyzer is a local, data-driven financial monitoring dashboard powered by Streamlit, Python, and OpenAI. It allows you to build, backtest, and generate fully institutional-grade PDF tear sheets for custom investment portfolios with integrated AI commentary.

## Features

- **Portfolio Building:** Interactively allocate weights across various financial assets (ETFs, Stocks, etc.) using Yahoo Finance data.
- **Backtesting Engine:** Compute historical metrics (CAGR, Sharpe, Max Drawdown, Volatility) and compare performance against benchmarks like the S&P 500.
- **Risk Analysis:** View dynamic charts (Drawdown, Asset distributions, Rolling Volatility, Returns Heatmaps).
- **Institutional PDF Export:** Generate gorgeous, single-page institutional tear sheets in PDF format.
- **AI-Powered Insights:** Automatically synthesize the portfolio's historical metrics into a clear, intelligent financial summary using OpenAI's `gpt-4o-mini`.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) (Extremely fast Python package manager)
- An OpenAI API Key

## Installation

1. **Clone the repository and set your API key:**
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. **Run the Dashboard:**
   Using `uv`, you don't even need to manually set up a virtual environment. Just run:
   ```bash
   uv run streamlit run portfolio_analyzer/app/main.py
   ```
   *This command will automatically resolve dependencies and launch the app in your local browser.*

## Architecture

- `portfolio_analyzer/app/`: Contains the Streamlit frontend pages and components.
- `portfolio_analyzer/core/`: The backend logic processing data fetching (`yfinance`), risk metric calculations, optimization, and PDF generation (`Playwright`).

---
*Disclaimer: The AI analysis feature requires an active OpenAI API key to function. The generated financial reports are for informational purposes only and do not constitute professional financial advice.*
