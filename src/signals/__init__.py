"""Signal generation modules."""

from .base_signal import BaseSignal
from .momentum import MomentumSignal, MultiMomentumSignal, MomentumAcceleration
from .mean_reversion import MeanReversionSignal, RSISignal, BollingerBandSignal
from .volatility import VolatilitySignal, VolatilityRegimeSignal, ATRSignal
from .funding_rate import FundingRateSignal, FundingMomentumSignal

__all__ = [
    # Base
    'BaseSignal',
    
    # Momentum
    'MomentumSignal',
    'MultiMomentumSignal',
    'MomentumAcceleration',
    
    # Mean Reversion
    'MeanReversionSignal',
    'RSISignal',
    'BollingerBandSignal',
    
    # Volatility
    'VolatilitySignal',
    'VolatilityRegimeSignal',
    'ATRSignal',
    
    # Funding Rate
    'FundingRateSignal',
    'FundingMomentumSignal',
]


# Signal registry for easy access
SIGNAL_REGISTRY = {
    'momentum': MomentumSignal,
    'momentum_multi': MultiMomentumSignal,
    'momentum_accel': MomentumAcceleration,
    'mean_reversion': MeanReversionSignal,
    'rsi': RSISignal,
    'bollinger': BollingerBandSignal,
    'volatility': VolatilitySignal,
    'vol_regime': VolatilityRegimeSignal,
    'atr': ATRSignal,
    'funding': FundingRateSignal,
    'funding_momentum': FundingMomentumSignal,
}


def get_signal(name: str, **kwargs) -> BaseSignal:
    """
    Get a signal instance by name.
    
    Args:
        name: Signal name from registry
        **kwargs: Signal-specific parameters
        
    Returns:
        Signal instance
    """
    if name not in SIGNAL_REGISTRY:
        raise ValueError(f"Unknown signal: {name}. Available: {list(SIGNAL_REGISTRY.keys())}")
    
    return SIGNAL_REGISTRY[name](**kwargs)
