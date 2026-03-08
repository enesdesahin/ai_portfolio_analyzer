import pandas as pd
import requests
import zipfile
import io
import streamlit as st

@st.cache_data
def get_fama_french_factors():
    """
    Downloads and parses the daily Fama/French 3 Factors from Ken French's library.
    Returns a DataFrame with index as Datetime and columns: ['Mkt-RF', 'SMB', 'HML', 'RF']
    Factors are in decimal form (divided by 100).
    """
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # The file logic inside the zip usually has a specific CSV name
            # We list files to be safe or assume standard naming
            file_names = z.namelist()
            csv_file = [f for f in file_names if f.endswith('.csv')][0]
            
            with z.open(csv_file) as f:
                # Ken French CSVs usually have header info. We need to skip rows until we find the data.
                # Usually row 3 or 4. But robust way is to just read and clean.
                # The daily file format typically starts with a few lines of description.
                # We verify the structure manually or use a standard skip.
                # For this specific file: header is usually around line 3, then data.
                # And usually there's a footer for annual factors we need to cut off.
                
                df = pd.read_csv(f, skiprows=3, index_col=0)
                
                # Cleaning
                # The file might contain annual factors at the bottom, which have shorter index (e.g. just Year)
                # or empty lines.
                
                # Check index format. Daily is usually YYYYMMDD
                df.index.name = "Date"
                
                # Coerce index to string/numeric to filter out junk footer text
                df = df[pd.to_numeric(df.index, errors='coerce').notna()]
                
                # Parse Dates
                df.index = pd.to_datetime(df.index, format="%Y%m%d")
                
                # Rename standard cols
                df.columns = df.columns.str.strip()
                
                # Columns are generally: 'Mkt-RF', 'SMB', 'HML', 'RF'
                # Convert from percent to decimal
                df = df / 100.0
                
                return df[["Mkt-RF", "SMB", "HML", "RF"]]
                
    except Exception as e:
        st.error(f"Error fetching Fama-French data: {e}")
        return pd.DataFrame()

import numpy as np

def run_factor_regression(portfolio_returns, start_date=None, end_date=None):
    """
    Runs a Fama-French 3-Factor regression on the portfolio returns.
    
    Args:
        portfolio_returns (pd.Series): Daily portfolio returns.
        start_date (datetime, optional): Start date filter.
        end_date (datetime, optional): End date filter.
        
    Returns:
        dict: containing 'betas' (dict of factor betas), 'alpha' (daily), 'r2', and 'aligned_data' (df).
                Returns None if regression fails or insufficient data.
    """
    try:
        # Fetch FF3 Data
        ff_data = get_fama_french_factors()
        
        if ff_data.empty:
            return None
            
        # Align Data
        # Filter FF data by date if needed (optional optimization)
        # Assuming portfolio_returns index is datetime
        
        # Combine
        aligned_df = pd.DataFrame({"Port": portfolio_returns}).join(ff_data, how="inner").dropna()
        
        if aligned_df.empty or len(aligned_df) < 5:
            return None
            
        # Prepare Regression Variables
        # Y = Portfolio Excess Return = Port - RF
        Y = aligned_df["Port"] - aligned_df["RF"]
        
        # X = [Mkt-RF, SMB, HML]
        X = aligned_df[["Mkt-RF", "SMB", "HML"]]
        
        # Add constant for intercept manually
        A = np.column_stack([np.ones(len(X)), X["Mkt-RF"], X["SMB"], X["HML"]])
        
        # Run Regression (OLS)
        coeffs, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)
        
        alpha_daily = coeffs[0]
        betas = {
            "Market": coeffs[1],
            "Size (SMB)": coeffs[2],
            "Value (HML)": coeffs[3]
        }
        
        # Calculate R-squared
        y_pred = A @ coeffs
        ss_res = np.sum((Y - y_pred)**2)
        ss_tot = np.sum((Y - Y.mean())**2)
        if ss_tot == 0:
            r2 = 0.0
        else:
            r2 = 1 - (ss_res / ss_tot)
            
        return {
            "betas": betas,
            "alpha_daily": alpha_daily,
            "r2": r2,
            "aligned_data": aligned_df
        }
        
    except Exception as e:
        print(f"Regression error: {e}")
        return None
