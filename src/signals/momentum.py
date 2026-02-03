"""
Momentum-based trading signals.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .base_signal import BaseSignal
from ..utils import log


class MomentumSignal(BaseSignal):
    """
    Momentum signal based on past returns.
    
    Hypothesis: Assets that have performed well recently
    will continue to perform well.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize momentum signal.
        
        Args:
            lookback: Lookback period in days
            config: Configuration dictionary
        """
        super().__init__(name=f"momentum_{lookback}d", config=config, **kwargs)
        self.lookback = lookback
    
    def compute_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum as N-day return.
        
        Args:
            data: Price DataFrame with 'date', 'symbol', 'close'
            
        Returns:
            DataFrame with momentum signal
        """
        df = data[['date', 'symbol', 'close']].copy()
        df = df.sort_values(['symbol', 'date'])
        
        # Calculate returns over lookback period
        df['signal'] = df.groupby('symbol')['close'].transform(
            lambda x: x / x.shift(self.lookback) - 1
        )
        
        return df[['date', 'symbol', 'signal']].dropna()


class MultiMomentumSignal(BaseSignal):
    """
    Multi-timeframe momentum signal.
    
    Combines momentum signals across multiple timeframes
    for a more robust signal.
    """
    
    def __init__(
        self,
        lookbacks: List[int] = [5, 10, 20, 60],
        weights: List[float] = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize multi-timeframe momentum.
        
        Args:
            lookbacks: List of lookback periods
            weights: Weights for each timeframe (equal if None)
            config: Configuration dictionary
        """
        super().__init__(name="momentum_multi", config=config, **kwargs)
        self.lookbacks = lookbacks
        self.weights = weights or [1.0 / len(lookbacks)] * len(lookbacks)
    
    def compute_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute weighted average of multiple momentum signals.
        """
        df = data[['date', 'symbol', 'close']].copy()
        df = df.sort_values(['symbol', 'date'])
        
        # Calculate momentum for each lookback
        for i, lookback in enumerate(self.lookbacks):
            col_name = f'mom_{lookback}'
            df[col_name] = df.groupby('symbol')['close'].transform(
                lambda x: x / x.shift(lookback) - 1
            )
        
        # Z-score each momentum component within symbol
        for lookback in self.lookbacks:
            col_name = f'mom_{lookback}'
            df[f'{col_name}_z'] = df.groupby('symbol')[col_name].transform(
                lambda x: (x - x.expanding().mean()) / x.expanding().std()
            )
        
        # Weighted combination
        df['signal'] = sum(
            self.weights[i] * df[f'mom_{lb}_z'] 
            for i, lb in enumerate(self.lookbacks)
        )
        
        return df[['date', 'symbol', 'signal']].dropna()


class MomentumAcceleration(BaseSignal):
    """
    Momentum of momentum (acceleration).
    
    Measures whether momentum itself is increasing or decreasing.
    """
    
    def __init__(
        self,
        mom_lookback: int = 20,
        accel_lookback: int = 5,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize momentum acceleration signal.
        
        Args:
            mom_lookback: Lookback for base momentum
            accel_lookback: Lookback for acceleration calculation
        """
        super().__init__(name=f"momentum_accel_{mom_lookback}_{accel_lookback}", 
                         config=config, **kwargs)
        self.mom_lookback = mom_lookback
        self.accel_lookback = accel_lookback
    
    def compute_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute momentum acceleration.
        """
        df = data[['date', 'symbol', 'close']].copy()
        df = df.sort_values(['symbol', 'date'])
        
        # Base momentum
        df['momentum'] = df.groupby('symbol')['close'].transform(
            lambda x: x / x.shift(self.mom_lookback) - 1
        )
        
        # Acceleration (change in momentum)
        df['signal'] = df.groupby('symbol')['momentum'].transform(
            lambda x: x - x.shift(self.accel_lookback)
        )
        
        return df[['date', 'symbol', 'signal']].dropna()
