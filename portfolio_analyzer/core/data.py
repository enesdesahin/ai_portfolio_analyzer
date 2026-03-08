import yfinance as yf
import streamlit as st
import pandas as pd

@st.cache_data
def load_price_data(tickers, start_date, end_date):
    """
    Fetches historical price data for the given tickers.
    Returns Adjusted Close prices only.
    """
    if not tickers:
        return pd.DataFrame()
    
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )
        
        # Handle yfinance multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            # If we have multiple levels (Price, Ticker), we want 'Close' or 'Adj Close'
            # yf.download with auto_adjust=True usually returns 'Close' as the single accessible level if not flattened properly or multi-level.
            # Recent yfinance versions return MultiIndex (Price, Ticker).
            
            # Let's try to get 'Close' level
            try:
                return data['Close']
            except KeyError:
                # If 'Close' is not in the first level, maybe it's flattened? 
                # This depends heavily on yfinance version. 
                # Let's fallback to just returning data if it looks right, or specific check
                pass

        # Fallback simplistic check
        if 'Close' in data.columns:
             return data['Close']
             
        return data

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_asset_metadata(tickers):
    """
    Fetches metadata (Sector, Country) for the given tickers.
    Falls back to quoteType for ETFs/futures/indices that lack sector data.
    """
    # Known futures asset class mappings
    FUTURES_ASSET_CLASS = {
        "ES=F": "Equity Index", "NQ=F": "Equity Index", "YM=F": "Equity Index", "RTY=F": "Equity Index",
        "GC=F": "Commodities", "SI=F": "Commodities", "HG=F": "Commodities",
        "CL=F": "Energy", "NG=F": "Energy", "RB=F": "Energy",
        "ZC=F": "Agriculture", "ZW=F": "Agriculture", "ZS=F": "Agriculture",
        "ZB=F": "Fixed Income", "ZN=F": "Fixed Income", "ZT=F": "Fixed Income",
    }

    # Keywords in ETF category to map to broad asset classes
    CATEGORY_TO_ASSET_CLASS = {
        "bond": "Fixed Income", "income": "Fixed Income", "treasury": "Fixed Income",
        "government": "Fixed Income", "corporate": "Fixed Income", "debt": "Fixed Income",
        "commodit": "Commodities", "gold": "Commodities", "silver": "Commodities",
        "metal": "Commodities", "energy": "Commodities", "oil": "Commodities",
        "real estate": "Real Estate", "reit": "Real Estate",
        "crypto": "Crypto", "bitcoin": "Crypto", "digital": "Crypto",
    }

    def _derive_asset_class(ticker, quote_type, category, sector):
        """Derive a broad asset class label."""
        # Futures
        if ticker in FUTURES_ASSET_CLASS:
            ac = FUTURES_ASSET_CLASS[ticker]
            return "Commodities" if ac in ("Commodities", "Energy", "Agriculture") else ac
        if ticker.endswith("=F"):
            return "Commodities"
        # Crypto
        if ticker.endswith("-USD") or quote_type == "CRYPTOCURRENCY":
            return "Crypto"
        # ETFs — check category keywords
        if quote_type == "ETF" and category:
            cat_lower = category.lower()
            for keyword, ac in CATEGORY_TO_ASSET_CLASS.items():
                if keyword in cat_lower:
                    return ac
            return "Equity"  # default ETF = equity
        # Stocks
        if quote_type == "EQUITY":
            if sector and sector.lower() == "real estate":
                return "Real Estate"
            return "Equity"
        return "Other"

    metadata = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            name = info.get("longName") or info.get("shortName") or t
            sector = info.get("sector", "")
            country = info.get("country", "")
            quote_type = info.get("quoteType", "")
            category = info.get("category", "")

            # Determine sector/asset class
            if sector and sector != "N/A":
                # Stock with a real sector — keep it
                pass
            elif t in FUTURES_ASSET_CLASS:
                # Known futures mapping
                sector = FUTURES_ASSET_CLASS[t]
            elif t.endswith("=F"):
                # Unknown future
                sector = "Futures"
            elif category and category != "N/A":
                # ETF with a category from yfinance
                sector = category
            elif quote_type == "CRYPTOCURRENCY":
                sector = "Crypto"
            elif quote_type:
                sector = quote_type
            else:
                sector = "Other"

            # Fallback country: infer from exchange if country is missing
            if not country or country == "N/A":
                exchange = info.get("exchange", "")
                us_exchanges = {"NMS", "NYQ", "NGM", "PCX", "BTS", "CME", "CBT", "NYM", "CMX"}
                if exchange in us_exchanges or t.endswith("=F"):
                    country = "United States"
                elif exchange in {"PAR", "ENX"}:
                    country = "France"
                elif exchange in {"GER", "FRA"}:
                    country = "Germany"
                elif exchange in {"AMS"}:
                    country = "Netherlands"
                elif exchange in {"CPH"}:
                    country = "Denmark"
                else:
                    country = "Other"

            asset_class = _derive_asset_class(t, quote_type, category, sector)

            metadata[t] = {
                "Company Name": name,
                "Sector": sector,
                "Country": country,
                "Asset Class": asset_class
            }
        except Exception:
            metadata[t] = {
                "Company Name": t,
                "Sector": "Other",
                "Country": "Other",
                "Asset Class": "Other"
            }
            
    return pd.DataFrame(metadata).T

@st.cache_data
def fetch_benchmark_data(start_date, end_date, benchmark_ticker="^GSPC"):
    """
    Fetches historical data for the benchmark (default: S&P 500).
    """
    return load_price_data([benchmark_ticker], start_date, end_date)
