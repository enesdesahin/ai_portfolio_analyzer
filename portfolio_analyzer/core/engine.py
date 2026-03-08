import pandas as pd
import numpy as np
from typing import Callable

class DailyBacktestEngine:
    """
    Runs a walk-forward daily backtest with periodic rebalancing logic.
    """
    def __init__(self, asset_returns: pd.DataFrame):
        """
        Args:
            asset_returns (pd.DataFrame): Daily log returns or simple returns for all assets.
        """
        self.asset_returns = asset_returns

    def run(self, 
            strategy_func: Callable[[pd.DataFrame], pd.Series],
            train_window_months: int,
            test_window_months: int = 1,
            rebalance_freq: str = 'ME',
            window_type: str = 'rolling') -> tuple[pd.Series, pd.DataFrame]:
        """
        Runs the walk-forward simulation, generating a daily return path and average weights.
        
        Args:
            strategy_func: Function that takes historical daily returns and outputs a weight Series.
            train_window_months: How many months of history to look back at each rebalance step.
            rebalance_freq: Pandas frequency string (e.g., 'ME' for month-end).
            window_type: 'rolling' or 'expanding'.
            
        Returns:
            pd.Series: Daily portfolio returns.
            pd.DataFrame: DataFrame of daily portfolio weights.
        """
        data = self.asset_returns
        
        # Determine the sequence of rebalance dates based on the data's index
        rebalance_dates = data.resample(rebalance_freq).last().index
        
        if len(rebalance_dates) <= train_window_months:
             return pd.Series(dtype=float), pd.DataFrame(columns=data.columns)
             
        strategy_returns = []
        weights_history = []
        
        for i in range(train_window_months, len(rebalance_dates), test_window_months):
            # The date we calculate new weights
            rebalance_date = rebalance_dates[i-1] 
            
            # The window where these weights will be applied
            test_start = rebalance_date + pd.Timedelta(days=1)
            
            # The end of the test window, capped at the maximum available date
            end_idx = min(i - 1 + test_window_months, len(rebalance_dates) - 1)
            test_end = rebalance_dates[end_idx]
            
            # The training window
            if window_type == 'rolling':
                train_start_idx = i - 1 - train_window_months
                train_start = rebalance_dates[train_start_idx] + pd.Timedelta(days=1) if train_start_idx >= 0 else data.index[0]
                train_data = data.loc[train_start:rebalance_date]
            else: # expanding
                train_data = data.loc[:rebalance_date]
                
            try:
                weights = strategy_func(train_data)
            except Exception as e:
                print(f"Error in strategy at {rebalance_date}: {e}")
                weights = pd.Series(0, index=data.columns)
                
            weights = weights.reindex(data.columns).fillna(0)
            
            test_data = data.loc[test_start:test_end]
            if test_data.empty:
                 continue
                 
            # Daily portfolio return = Sum of (Weight * Asset Daily Return)
            daily_port_returns = (test_data * weights).sum(axis=1)
            
            strategy_returns.append(daily_port_returns)
            
            # Record weights 
            weight_df = pd.DataFrame([weights.values] * len(test_data), index=test_data.index, columns=data.columns)
            weights_history.append(weight_df)
            
        if not strategy_returns:
            return pd.Series(dtype=float), pd.DataFrame(columns=data.columns)
            
        full_returns = pd.concat(strategy_returns)
        full_weights = pd.concat(weights_history)
        
        full_returns = full_returns[~full_returns.index.duplicated(keep='first')]
        full_weights = full_weights[~full_weights.index.duplicated(keep='first')]
        
        return full_returns, full_weights
