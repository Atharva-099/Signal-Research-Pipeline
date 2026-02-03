"""
Abstract base class for all trading signals.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from ..utils import log, load_config, zscore_normalize, rank_normalize


class BaseSignal(ABC):
    """
    Abstract base class for trading signals.
    
    All signals should inherit from this class and implement
    the compute() method.
    """
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any] = None,
        normalize_method: str = 'zscore'
    ):
        """
        Initialize the signal.
        
        Args:
            name: Signal name
            config: Configuration dictionary
            normalize_method: 'zscore', 'rank', or 'none'
        """
        self.name = name
        self.config = config or load_config()
        self.normalize_method = normalize_method
        
        log.debug(f"Initialized signal: {self.name}")
    
    @abstractmethod
    def compute_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute raw signal values.
        
        Args:
            data: Price/market data DataFrame
            
        Returns:
            DataFrame with 'date', 'symbol', and 'signal' columns
        """
        pass
    
    def normalize(self, signal: pd.Series, method: str = None) -> pd.Series:
        """
        Normalize signal values.
        
        Args:
            signal: Raw signal values
            method: Normalization method (overrides default)
            
        Returns:
            Normalized signal values
        """
        method = method or self.normalize_method
        
        if method == 'zscore':
            return zscore_normalize(signal)
        elif method == 'rank':
            return rank_normalize(signal)
        elif method == 'none':
            return signal
        else:
            log.warning(f"Unknown normalize method: {method}, using zscore")
            return zscore_normalize(signal)
    
    def cross_sectional_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectionally normalize signal (z-score across assets per date).
        
        Args:
            data: DataFrame with 'date', 'symbol', 'signal' columns
            
        Returns:
            DataFrame with cross-sectionally normalized signal
        """
        result = data.copy()
        
        # Group by date and z-score across assets
        result['signal'] = result.groupby('date')['signal'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        return result
    
    def compute(
        self,
        data: pd.DataFrame,
        cross_sectional: bool = True
    ) -> pd.DataFrame:
        """
        Compute and normalize signal values.
        
        Args:
            data: Price/market data DataFrame
            cross_sectional: Whether to apply cross-sectional normalization
            
        Returns:
            DataFrame with normalized signal values
        """
        # Compute raw signal
        signal_df = self.compute_raw(data)
        
        # Time-series normalization per asset
        if self.normalize_method != 'none':
            signal_df['signal'] = signal_df.groupby('symbol')['signal'].transform(
                lambda x: self.normalize(x)
            )
        
        # Cross-sectional normalization
        if cross_sectional:
            signal_df = self.cross_sectional_normalize(signal_df)
        
        # Add metadata
        signal_df['signal_name'] = self.name
        
        return signal_df
    
    def to_wide_format(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert signal to wide format (dates as index, symbols as columns).
        
        Args:
            signal_df: Long-format signal DataFrame
            
        Returns:
            Wide-format DataFrame
        """
        return signal_df.pivot(index='date', columns='symbol', values='signal')
    
    def calculate_forward_returns(
        self,
        price_data: pd.DataFrame,
        periods: List[int] = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        Calculate forward returns for IC computation.
        
        Args:
            price_data: Price DataFrame with 'date', 'symbol', 'close'
            periods: Forward return periods
            
        Returns:
            DataFrame with forward returns
        """
        result = price_data[['date', 'symbol', 'close']].copy()
        result = result.sort_values(['symbol', 'date'])
        
        for period in periods:
            col_name = f'fwd_ret_{period}d'
            result[col_name] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-period) / x - 1
            )
        
        return result
    
    def calculate_ic(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        return_col: str = 'fwd_ret_1d'
    ) -> pd.DataFrame:
        """
        Calculate Information Coefficient (IC) over time.
        
        IC = Spearman correlation between signal and forward returns
        
        Args:
            signal_df: Signal DataFrame
            returns_df: Forward returns DataFrame
            return_col: Which return column to use
            
        Returns:
            DataFrame with daily IC values
        """
        # Merge signal and returns
        merged = signal_df.merge(
            returns_df[['date', 'symbol', return_col]],
            on=['date', 'symbol'],
            how='inner'
        )
        
        # Calculate rank correlation per date
        ic_series = merged.groupby('date').apply(
            lambda x: x['signal'].corr(x[return_col], method='spearman')
        ).reset_index()
        
        ic_series.columns = ['date', 'ic']
        
        return ic_series
    
    def summary_stats(self, ic_series: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate summary statistics for IC series.
        
        Args:
            ic_series: DataFrame with 'date' and 'ic' columns
            
        Returns:
            Dictionary of statistics
        """
        ic = ic_series['ic'].dropna()
        
        return {
            'ic_mean': ic.mean(),
            'ic_std': ic.std(),
            'ic_ir': ic.mean() / ic.std() if ic.std() > 0 else 0,
            'ic_hit_rate': (ic > 0).mean(),
            'ic_t_stat': ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else 0,
            'n_observations': len(ic)
        }
