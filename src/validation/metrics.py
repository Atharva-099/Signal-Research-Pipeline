"""
Performance metrics for signal evaluation.
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats


def calculate_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    method: str = 'spearman'
) -> float:
    """
    Calculate Information Coefficient.
    
    IC = Correlation between signal and forward returns
    
    Args:
        signal: Signal values
        forward_returns: Forward returns
        method: 'spearman' or 'pearson'
        
    Returns:
        IC value
    """
    # Align and drop NaN
    valid = pd.concat([signal, forward_returns], axis=1).dropna()
    
    if len(valid) < 5:
        return np.nan
    
    if method == 'spearman':
        ic, _ = stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    else:
        ic, _ = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
    
    return ic


def calculate_ic_series(
    signal_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    return_col: str = 'fwd_ret_1d',
    method: str = 'spearman'
) -> pd.Series:
    """
    Calculate IC time series.
    
    Args:
        signal_df: DataFrame with 'date', 'symbol', 'signal'
        returns_df: DataFrame with 'date', 'symbol', return_col
        return_col: Column name for returns
        method: Correlation method
        
    Returns:
        Series of daily IC values
    """
    # Merge
    merged = signal_df.merge(
        returns_df[['date', 'symbol', return_col]],
        on=['date', 'symbol'],
        how='inner'
    )
    
    # Calculate IC per date
    def daily_ic(group):
        if len(group) < 3:
            return np.nan
        if method == 'spearman':
            return group['signal'].corr(group[return_col], method='spearman')
        else:
            return group['signal'].corr(group[return_col])
    
    ic_series = merged.groupby('date').apply(daily_ic)
    
    return ic_series


def calculate_ir(ic_series: pd.Series) -> float:
    """
    Calculate Information Ratio.
    
    IR = mean(IC) / std(IC)
    
    Args:
        ic_series: Series of IC values
        
    Returns:
        Information Ratio
    """
    ic = ic_series.dropna()
    if len(ic) < 2 or ic.std() == 0:
        return np.nan
    
    return ic.mean() / ic.std()


def calculate_sharpe(
    returns: pd.Series,
    periods_per_year: int = 365,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe Ratio.
    
    Args:
        returns: Returns series
        periods_per_year: Annualization factor
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Annualized Sharpe Ratio
    """
    returns = returns.dropna()
    
    if len(returns) < 2 or returns.std() == 0:
        return np.nan
    
    excess_return = returns.mean() - risk_free_rate / periods_per_year
    annualized_sharpe = excess_return / returns.std() * np.sqrt(periods_per_year)
    
    return annualized_sharpe


def calculate_sortino(
    returns: pd.Series,
    periods_per_year: int = 365,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sortino Ratio.
    
    Uses downside deviation instead of total volatility.
    
    Args:
        returns: Returns series
        periods_per_year: Annualization factor
        risk_free_rate: Risk-free rate
        
    Returns:
        Annualized Sortino Ratio
    """
    returns = returns.dropna()
    
    if len(returns) < 2:
        return np.nan
    
    excess_return = returns.mean() - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.inf if excess_return > 0 else np.nan
    
    downside_std = downside_returns.std()
    annualized_sortino = excess_return / downside_std * np.sqrt(periods_per_year)
    
    return annualized_sortino


def calculate_max_drawdown(cumulative_returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown.
    
    Args:
        cumulative_returns: Cumulative returns (1 + r1)(1 + r2)...
        
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    """
    # Running maximum
    running_max = cumulative_returns.cummax()
    
    # Drawdown series
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Maximum drawdown
    max_dd = drawdown.min()
    
    # Find dates
    trough_date = drawdown.idxmin()
    peak_date = cumulative_returns[:trough_date].idxmax()
    
    return max_dd, peak_date, trough_date


def calculate_calmar(
    returns: pd.Series,
    periods_per_year: int = 365
) -> float:
    """
    Calculate Calmar Ratio.
    
    Calmar = Annualized Return / Max Drawdown
    
    Args:
        returns: Returns series
        periods_per_year: Annualization factor
        
    Returns:
        Calmar Ratio
    """
    cumulative = (1 + returns).cumprod()
    max_dd, _, _ = calculate_max_drawdown(cumulative)
    
    if max_dd == 0:
        return np.nan
    
    annualized_return = (1 + returns.mean()) ** periods_per_year - 1
    
    return annualized_return / abs(max_dd)


def calculate_hit_rate(signal: pd.Series, returns: pd.Series) -> float:
    """
    Calculate hit rate (accuracy of direction prediction).
    
    Args:
        signal: Signal values
        returns: Forward returns
        
    Returns:
        Hit rate (0 to 1)
    """
    # Align
    valid = pd.concat([signal, returns], axis=1).dropna()
    
    if len(valid) == 0:
        return np.nan
    
    signal_dir = np.sign(valid.iloc[:, 0])
    return_dir = np.sign(valid.iloc[:, 1])
    
    return (signal_dir == return_dir).mean()


def calculate_turnover(signal: pd.DataFrame) -> pd.Series:
    """
    Calculate signal turnover.
    
    Args:
        signal: Signal DataFrame with 'date', 'symbol', 'signal'
        
    Returns:
        Daily turnover series
    """
    # Pivot to wide format
    wide = signal.pivot(index='date', columns='symbol', values='signal')
    
    # Changes in signal
    turnover = wide.diff().abs().sum(axis=1) / 2
    
    return turnover


class PerformanceMetrics:
    """Container for comprehensive performance metrics."""
    
    def __init__(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        return_col: str = 'fwd_ret_1d'
    ):
        """
        Initialize performance metrics.
        
        Args:
            signal_df: Signal DataFrame
            returns_df: Returns DataFrame
            return_col: Return column to use
        """
        self.signal_df = signal_df
        self.returns_df = returns_df
        self.return_col = return_col
        
        # Calculate IC series
        self.ic_series = calculate_ic_series(signal_df, returns_df, return_col)
    
    def summary(self) -> Dict[str, float]:
        """
        Calculate all summary metrics.
        
        Returns:
            Dictionary of metrics
        """
        ic = self.ic_series.dropna()
        
        return {
            'ic_mean': ic.mean(),
            'ic_std': ic.std(),
            'ic_ir': calculate_ir(ic),
            'ic_t_stat': ic.mean() / (ic.std() / np.sqrt(len(ic))) if len(ic) > 0 else np.nan,
            'ic_hit_rate': (ic > 0).mean(),
            'ic_positive_pct': (ic > 0).mean(),
            'ic_p_value': stats.ttest_1samp(ic, 0).pvalue if len(ic) > 1 else np.nan,
            'n_days': len(ic),
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame."""
        metrics = self.summary()
        return pd.DataFrame([metrics])
