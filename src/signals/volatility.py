"""
Volatility-based trading signals.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_signal import BaseSignal
from ..utils import log


class VolatilitySignal(BaseSignal):
    """
    Realized volatility signal.
    
    Hypothesis: Lower volatility assets may offer better
    risk-adjusted returns (low volatility anomaly).
    """
    
    def __init__(
        self,
        lookback: int = 20,
        annualize: bool = True,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize volatility signal.
        
        Args:
            lookback: Volatility calculation window
            annualize: Whether to annualize volatility
        """
        super().__init__(name=f"volatility_{lookback}d", config=config, **kwargs)
        self.lookback = lookback
        self.annualize = annualize
        # Crypto trades 365 days
        self.annualization_factor = np.sqrt(365) if annualize else 1
    
    def compute_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility signal.
        
        Signal = -volatility (negative because we prefer low vol)
        """
        df = data[['date', 'symbol', 'close']].copy()
        df = df.sort_values(['symbol', 'date'])
        
        # Daily returns
        df['returns'] = df.groupby('symbol')['close'].pct_change()
        
        # Rolling volatility
        df['volatility'] = df.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(window=self.lookback, min_periods=1).std() 
            * self.annualization_factor
        )
        
        # Negative volatility as signal (prefer low vol)
        df['signal'] = -df['volatility']
        
        return df[['date', 'symbol', 'signal']].dropna()


class VolatilityRegimeSignal(BaseSignal):
    """
    Volatility regime signal.
    
    Identifies high/low volatility regimes relative to history.
    """
    
    def __init__(
        self,
        short_lookback: int = 20,
        long_lookback: int = 60,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize volatility regime signal.
        
        Args:
            short_lookback: Short-term vol window
            long_lookback: Long-term vol window
        """
        super().__init__(name=f"vol_regime_{short_lookback}_{long_lookback}", 
                         config=config, **kwargs)
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback
    
    def compute_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility regime signal.
        
        Signal = short_vol / long_vol - 1
        Positive = vol expansion (risk-off)
        Negative = vol contraction (risk-on)
        """
        df = data[['date', 'symbol', 'close']].copy()
        df = df.sort_values(['symbol', 'date'])
        
        # Daily returns
        df['returns'] = df.groupby('symbol')['close'].pct_change()
        
        # Short and long term volatility
        df['vol_short'] = df.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(window=self.short_lookback, min_periods=1).std()
        )
        df['vol_long'] = df.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(window=self.long_lookback, min_periods=1).std()
        )
        
        # Regime indicator
        df['signal'] = -1 * (df['vol_short'] / df['vol_long'] - 1)
        
        return df[['date', 'symbol', 'signal']].dropna()


class ATRSignal(BaseSignal):
    """
    Average True Range (ATR) based signal.
    
    ATR measures market volatility including gaps.
    """
    
    def __init__(
        self,
        lookback: int = 14,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize ATR signal.
        
        Args:
            lookback: ATR calculation period
        """
        super().__init__(name=f"atr_{lookback}d", config=config, **kwargs)
        self.lookback = lookback
    
    def compute_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ATR signal.
        
        Signal = -ATR% (negative because prefer low volatility)
        """
        df = data[['date', 'symbol', 'open', 'high', 'low', 'close']].copy()
        df = df.sort_values(['symbol', 'date'])
        
        # True Range components
        df['prev_close'] = df.groupby('symbol')['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # ATR
        df['atr'] = df.groupby('symbol')['true_range'].transform(
            lambda x: x.rolling(window=self.lookback, min_periods=1).mean()
        )
        
        # ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close']
        
        # Negative signal (prefer low ATR)
        df['signal'] = -df['atr_pct']
        
        return df[['date', 'symbol', 'signal']].dropna()
