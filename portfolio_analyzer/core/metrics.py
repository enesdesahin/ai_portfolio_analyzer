import pandas as pd

def compute_returns(prices):
    """
    Computes simple returns from prices.
    """
    return prices.pct_change().dropna()

def compute_portfolio_returns(asset_returns, weights):
    """
    Computes portfolio daily returns based on asset returns and weights.
    """
    return asset_returns.dot(weights)

def calculate_cumulative_returns(returns):
    """
    Computes cumulative returns series (Wealth Index - 1).
    """
    return (1 + returns).cumprod() - 1

def calculate_annualized_metrics(returns, risk_free_rate=0.0):
    """
    Computes annualized return and volatility.
    Returns a dictionary with 'ann_ret' and 'ann_vol'.
    """
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * (252 ** 0.5)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe
    }

def calculate_drawdown_series(returns):
    """
    Computes the drawdown series for a return series.
    """
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns
import numpy as np
import pandas as pd

def calculate_beta(portfolio_returns, benchmark_returns):
    """Calculates the Beta of the portfolio relative to the benchmark."""
    # Ensure alignment
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    p_rets = portfolio_returns.loc[common_idx]
    b_rets = benchmark_returns.loc[common_idx]
    
    covariance = np.cov(p_rets, b_rets)[0, 1]
    variance = np.var(b_rets)
    
    if variance == 0:
        return 0.0
    return covariance / variance

def calculate_alpha(portfolio_returns, benchmark_returns, risk_free_rate=0.0):
    """Calculates Jensen's Alpha."""
    beta = calculate_beta(portfolio_returns, benchmark_returns)
    
    # annualized returns
    rp = portfolio_returns.mean() * 252
    rm = benchmark_returns.mean() * 252
    rf = risk_free_rate
    
    alpha = (rp - rf) - beta * (rm - rf)
    return alpha

def calculate_tracking_error(portfolio_returns, benchmark_returns):
    """Calculates Tracking Error (Std Dev of Excess Returns)."""
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    excess_returns = portfolio_returns.loc[common_idx] - benchmark_returns.loc[common_idx]
    return excess_returns.std() * np.sqrt(252)

def calculate_information_ratio(portfolio_returns, benchmark_returns):
    """Calculates Information Ratio."""
    te = calculate_tracking_error(portfolio_returns, benchmark_returns)
    if te == 0:
        return 0.0
    
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    excess_returns = portfolio_returns.loc[common_idx] - benchmark_returns.loc[common_idx]
    active_return = excess_returns.mean() * 252
    
    return active_return / te

def calculate_historical_var(returns, confidence_level=0.95):
    """
    Calculates Historical Value at Risk (VaR).
    
    Args:
        returns (pd.Series): Daily returns.
        confidence_level (float): Confidence level (e.g., 0.95 for 95%).
        
    Returns:
        float: VaR value (absolute positive value representing loss).
    """
    if returns.empty:
        return 0.0
    
    # Percentile method
    # For 95% confidence, we look at the 5th percentile of returns
    cutoff = (1 - confidence_level) * 100
    var_value = np.percentile(returns, cutoff)
    
    # VaR is typically expressed as a positive loss
    return -var_value

def calculate_historical_cvar(returns, confidence_level=0.95):
    """
    Calculates Historical Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    Args:
        returns (pd.Series): Daily returns.
        confidence_level (float): Confidence level.
        
    Returns:
        float: CVaR value (absolute positive value).
    """
    if returns.empty:
        return 0.0
        
    cutoff = (1 - confidence_level) * 100
    var_threshold = np.percentile(returns, cutoff)
    
    # Filter returns worse than VaR
    tail_losses = returns[returns <= var_threshold]
    
    if tail_losses.empty:
        return 0.0
        
    cvar_value = tail_losses.mean()
    return -cvar_value

def simulate_monte_carlo(mean_ret, cov_matrix, days, simulations):
    """
    Runs a simple Monte Carlo simulation for portfolio returns.
    
    Args:
        mean_ret (float): Daily expected return (scalar).
        cov_matrix (float): Daily volatility (scalar) or full covariance? 
                            For simple projection, we typically use portfolio daily mean & vol.
        days (int): Number of trading days to simulate.
        simulations (int): Number of simulation runs.
        
    Returns:
        np.ndarray: Simulated cumulative return paths (simulations x days).
    """
    # Assuming independent, identically distributed (IID) Normal returns
    # This is a simplification ("Simple Monte Carlo")
    # For a portfolio, inputs are usually Portfolio Daily Mean & Portfolio Daily Vol
    
    # If inputs are portfolio-level scalar mean & vol:
    daily_mean = mean_ret
    daily_vol = cov_matrix # treating as scalar vol for simplicity if passed as such
    
    # Generate random Z-scores
    # Shape: (simulations, days)
    random_shocks = np.random.normal(0, 1, (simulations, days))
    
    # Simulate daily returns: r = mean + vol * Z
    daily_returns = daily_mean + daily_vol * random_shocks
    
    # Cumulative returns: (1+r).cumprod()
    # Prepend 1.0/0.0 for start? Usually we start at 1.0 (or 0% return)
    # Let's return cumulative growth factor
    cumulative_paths = np.cumprod(1 + daily_returns, axis=1)
    
    # Insert start value of 1.0
    cumulative_paths = np.insert(cumulative_paths, 0, 1.0, axis=1)
    
    return cumulative_paths

def calculate_max_drawdown(returns):
    """Calculates the Maximum Drawdown of a return series."""
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    return drawdown.min()

def compute_beta_alpha(portfolio_returns, benchmark_returns, risk_free_rate=0.0):
    """
    Computes Beta, Alpha, and R-squared.
    Returns:
        tuple: (beta, alpha, r_squared)
    """
    # Ensure alignment
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    p = portfolio_returns.loc[common]
    b = benchmark_returns.loc[common]
    
    if len(p) < 2:
        return 0.0, 0.0, 0.0
        
    # Beta
    cov = np.cov(p, b)[0, 1]
    var_b = np.var(b)
    beta = cov / var_b if var_b != 0 else 0.0
    
    # Alpha (Annualized)
    # Alpha = Rp - [Rf + Beta * (Rm - Rf)]
    rp = p.mean() * 252
    rm = b.mean() * 252
    alpha = (rp - risk_free_rate) - beta * (rm - risk_free_rate)
    
    # R-squared
    corr = p.corr(b)
    r2 = corr ** 2
    
    return beta, alpha, r2


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sortino Ratio (penalizes only downside volatility)."""
    excess_returns = (returns - risk_free_rate) / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.inf

    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    if downside_deviation == 0:
        return np.inf
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation


def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar Ratio = Annualised Return / Max Drawdown."""
    wealth = (1 + returns).cumprod()
    peaks = wealth.cummax()
    drawdown = (wealth - peaks) / peaks
    max_drawdown = drawdown.min()

    if max_drawdown == 0:
        return np.inf

    annualized_ret = returns.mean() * periods_per_year
    return annualized_ret / abs(max_drawdown)


def calculate_skewness(returns: pd.Series) -> float:
    """Skewness of the return distribution."""
    return returns.skew()


def calculate_excess_kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (normal distribution = 0)."""
    return returns.kurtosis()
