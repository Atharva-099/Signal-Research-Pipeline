"""
Mean reversion trading signals.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from .base_signal import BaseSignal
from ..utils import log


class MeanReversionSignal(BaseSignal):
    """
    Mean reversion signal based on deviation from moving average.
    
    Hypothesis: Assets that have deviated significantly from their
    average will revert back.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        num_std: float = 2.0,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize mean reversion signal.
        
        Args:
            lookback: Moving average lookback period
            num_std: Number of standard deviations for bands
        """
        super().__init__(name=f"mean_reversion_{lookback}d", config=config, **kwargs)
        self.lookback = lookback
        self.num_std = num_std
    
    def compute_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean reversion signal.
        
        Signal = -(price - MA) / std
        Negative because we buy when price is below MA
        """
        df = data[['date', 'symbol', 'close']].copy()
        df = df.sort_values(['symbol', 'date'])
        
        # Moving average and standard deviation
        df['ma'] = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=self.lookback, min_periods=1).mean()
        )
        df['std'] = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=self.lookback, min_periods=1).std()
        )
        
        # Deviation from MA (negative for buying oversold)
        df['signal'] = -1 * (df['close'] - df['ma']) / df['std']
        
        return df[['date', 'symbol', 'signal']].dropna()


class RSISignal(BaseSignal):
    """
    Relative Strength Index (RSI) based signal.
    
    RSI measures overbought/oversold conditions.
    """
    
    def __init__(
        self,
        lookback: int = 14,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize RSI signal.
        
        Args:
            lookback: RSI calculation period
        """
        super().__init__(name=f"rsi_{lookback}d", config=config, **kwargs)
        self.lookback = lookback
    
    def compute_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RSI signal.
        
        Signal = 50 - RSI (so oversold = positive, overbought = negative)
        """
        df = data[['date', 'symbol', 'close']].copy()
        df = df.sort_values(['symbol', 'date'])
        
        # Daily returns
        df['change'] = df.groupby('symbol')['close'].diff()
        
        # Separate gains and losses
        df['gain'] = df['change'].clip(lower=0)
        df['loss'] = (-df['change']).clip(lower=0)
        
        # Average gains and losses (EMA)
        df['avg_gain'] = df.groupby('symbol')['gain'].transform(
            lambda x: x.ewm(span=self.lookback, adjust=False).mean()
        )
        df['avg_loss'] = df.groupby('symbol')['loss'].transform(
            lambda x: x.ewm(span=self.lookback, adjust=False).mean()
        )
        
        # RSI calculation
        df['rs'] = df['avg_gain'] / df['avg_loss'].replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # Signal: 50 - RSI (contrarian)
        df['signal'] = 50 - df['rsi']
        
        return df[['date', 'symbol', 'signal']].dropna()


class BollingerBandSignal(BaseSignal):
    """
    Bollinger Band mean reversion signal.
    
    Uses position within Bollinger Bands as signal.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        num_std: float = 2.0,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize Bollinger Band signal.
        
        Args:
            lookback: Moving average period
            num_std: Standard deviations for bands
        """
        super().__init__(name=f"bollinger_{lookback}d", config=config, **kwargs)
        self.lookback = lookback
        self.num_std = num_std
    
    def compute_raw(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Bollinger Band position.
        
        Signal = -(price - MA) / band_width
        -1 = at lower band, +1 = at upper band
        """
        df = data[['date', 'symbol', 'close']].copy()
        df = df.sort_values(['symbol', 'date'])
        
        # Calculate bands
        df['ma'] = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=self.lookback, min_periods=1).mean()
        )
        df['std'] = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=self.lookback, min_periods=1).std()
        )
        
        df['upper'] = df['ma'] + self.num_std * df['std']
        df['lower'] = df['ma'] - self.num_std * df['std']
        
        # Position within bands (-1 to +1) then negate for contrarian
        band_width = df['upper'] - df['lower']
        df['bb_position'] = (df['close'] - df['lower']) / band_width * 2 - 1
        
        # Contrarian signal
        df['signal'] = -df['bb_position']
        
        return df[['date', 'symbol', 'signal']].dropna()
