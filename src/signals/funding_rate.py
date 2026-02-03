"""
Funding rate based trading signals.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_signal import BaseSignal
from ..utils import log


class FundingRateSignal(BaseSignal):
    """
    Funding rate signal from perpetual futures.
    
    Hypothesis: Extreme funding rates indicate crowded positioning
    and may predict reversals.
    """
    
    def __init__(
        self,
        lookback: int = 7,
        config: Dict[str, Any] = None,
        contrarian: bool = True,
        **kwargs
    ):
        """
        Initialize funding rate signal.
        
        Args:
            lookback: Days to aggregate funding
            contrarian: If True, negative funding = buy signal
        """
        super().__init__(name=f"funding_{lookback}d", config=config, **kwargs)
        self.lookback = lookback
        self.contrarian = contrarian
    
    def compute_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute funding rate signal.
        
        Data should have 'date', 'symbol', 'funding_rate' columns.
        
        Args:
            data: Funding rate DataFrame
            
        Returns:
            DataFrame with funding signal
        """
        df = data[['date', 'symbol', 'funding_rate']].copy()
        df = df.sort_values(['symbol', 'date'])
        
        # Aggregate daily if needed (funding rates are typically every 8 hours)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.groupby(['date', 'symbol'])['funding_rate'].sum().reset_index()
        df['date'] = pd.to_datetime(df['date'])
        
        # Cumulative funding over lookback
        df['cum_funding'] = df.groupby('symbol')['funding_rate'].transform(
            lambda x: x.rolling(window=self.lookback, min_periods=1).sum()
        )
        
        # Signal (contrarian or trend-following)
        if self.contrarian:
            # High funding = short signal, low funding = long signal
            df['signal'] = -df['cum_funding']
        else:
            df['signal'] = df['cum_funding']
        
        return df[['date', 'symbol', 'signal']].dropna()


class FundingMomentumSignal(BaseSignal):
    """
    Funding rate momentum signal.
    
    Measures change in funding rate sentiment.
    """
    
    def __init__(
        self,
        short_lookback: int = 3,
        long_lookback: int = 14,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize funding momentum signal.
        
        Args:
            short_lookback: Short-term funding window
            long_lookback: Long-term funding window
        """
        super().__init__(name=f"funding_momentum_{short_lookback}_{long_lookback}", 
                         config=config, **kwargs)
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback
    
    def compute_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute funding momentum.
        
        Signal = short_term_funding - long_term_funding
        Rising funding = bearish, falling = bullish
        """
        df = data[['date', 'symbol', 'funding_rate']].copy()
        df = df.sort_values(['symbol', 'date'])
        
        # Aggregate daily
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.groupby(['date', 'symbol'])['funding_rate'].sum().reset_index()
        df['date'] = pd.to_datetime(df['date'])
        
        # Short and long term average funding
        df['funding_short'] = df.groupby('symbol')['funding_rate'].transform(
            lambda x: x.rolling(window=self.short_lookback, min_periods=1).mean()
        )
        df['funding_long'] = df.groupby('symbol')['funding_rate'].transform(
            lambda x: x.rolling(window=self.long_lookback, min_periods=1).mean()
        )
        
        # Momentum (contrarian - rising funding is bearish)
        df['signal'] = -(df['funding_short'] - df['funding_long'])
        
        return df[['date', 'symbol', 'signal']].dropna()
