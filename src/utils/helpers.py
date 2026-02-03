"""
Common utility functions.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import pandas as pd
import numpy as np


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Defaults to config/config.yaml
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Find project root
        current = Path(__file__).resolve()
        project_root = current.parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def zscore_normalize(series: pd.Series, window: int = None) -> pd.Series:
    """
    Z-score normalize a series.
    
    Args:
        series: Input series
        window: Rolling window size (None for expanding)
        
    Returns:
        Z-scored series
    """
    if window:
        mean = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std()
    else:
        mean = series.expanding(min_periods=1).mean()
        std = series.expanding(min_periods=1).std()
    
    # Avoid division by zero
    std = std.replace(0, np.nan)
    
    return (series - mean) / std


def rank_normalize(series: pd.Series) -> pd.Series:
    """
    Rank normalize a series to [-1, 1] range.
    
    Args:
        series: Input series
        
    Returns:
        Rank-normalized series
    """
    ranks = series.rank(pct=True)
    return 2 * ranks - 1  # Scale from [0,1] to [-1,1]


def calculate_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        prices: Price DataFrame
        periods: Number of periods for return calculation
        
    Returns:
        Returns DataFrame
    """
    return prices.pct_change(periods=periods)


def calculate_log_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """
    Calculate log returns from price data.
    
    Args:
        prices: Price DataFrame
        periods: Number of periods for return calculation
        
    Returns:
        Log returns DataFrame
    """
    return np.log(prices / prices.shift(periods))


def winsorize(series: pd.Series, limits: tuple = (0.01, 0.99)) -> pd.Series:
    """
    Winsorize extreme values.
    
    Args:
        series: Input series
        limits: Lower and upper percentile limits
        
    Returns:
        Winsorized series
    """
    lower = series.quantile(limits[0])
    upper = series.quantile(limits[1])
    return series.clip(lower=lower, upper=upper)


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
