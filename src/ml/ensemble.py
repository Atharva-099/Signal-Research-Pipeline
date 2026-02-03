"""
Ensemble methods for combining signals.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from ..utils import log, load_config


class SignalEnsemble:
    """
    Combine multiple signals into an ensemble.
    """
    
    def __init__(
        self,
        method: str = 'equal',
        config: Dict[str, Any] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            method: 'equal', 'ic_weighted', 'sharpe_weighted', or 'stacking'
            config: Configuration dictionary
        """
        self.method = method
        self.config = config or load_config()
        self.weights_ = None
        self.meta_model_ = None
        
        log.info(f"SignalEnsemble initialized with method: {method}")
    
    def fit(
        self,
        signals_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        return_col: str = 'fwd_ret_1d'
    ) -> 'SignalEnsemble':
        """
        Fit ensemble weights.
        
        Args:
            signals_df: DataFrame with 'date', 'symbol', and signal columns
            returns_df: DataFrame with 'date', 'symbol', and return column
            return_col: Return column name
            
        Returns:
            self
        """
        signal_cols = [c for c in signals_df.columns if c not in ['date', 'symbol']]
        
        if self.method == 'equal':
            self.weights_ = {col: 1.0 / len(signal_cols) for col in signal_cols}
        
        elif self.method == 'ic_weighted':
            self.weights_ = self._calculate_ic_weights(signals_df, returns_df, return_col)
        
        elif self.method == 'sharpe_weighted':
            self.weights_ = self._calculate_sharpe_weights(signals_df, returns_df, return_col)
        
        elif self.method == 'stacking':
            self._fit_stacking(signals_df, returns_df, return_col)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        log.info(f"Ensemble fitted. Weights: {self.weights_}")
        
        return self
    
    def _calculate_ic_weights(
        self,
        signals_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        return_col: str
    ) -> Dict[str, float]:
        """Calculate weights based on IC."""
        signal_cols = [c for c in signals_df.columns if c not in ['date', 'symbol']]
        
        # Merge
        merged = signals_df.merge(
            returns_df[['date', 'symbol', return_col]],
            on=['date', 'symbol'],
            how='inner'
        )
        
        # Calculate IC per signal
        ics = {}
        for col in signal_cols:
            ic_series = merged.groupby('date').apply(
                lambda x: x[col].corr(x[return_col], method='spearman')
            )
            ics[col] = ic_series.mean()
        
        # Convert to positive weights
        ic_series = pd.Series(ics)
        ic_positive = ic_series.clip(lower=0)
        
        if ic_positive.sum() == 0:
            return {col: 1.0 / len(signal_cols) for col in signal_cols}
        
        weights = (ic_positive / ic_positive.sum()).to_dict()
        
        return weights
    
    def _calculate_sharpe_weights(
        self,
        signals_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        return_col: str
    ) -> Dict[str, float]:
        """Calculate weights based on hypothetical Sharpe."""
        signal_cols = [c for c in signals_df.columns if c not in ['date', 'symbol']]
        
        merged = signals_df.merge(
            returns_df[['date', 'symbol', return_col]],
            on=['date', 'symbol'],
            how='inner'
        )
        
        sharpes = {}
        for col in signal_cols:
            # Simple: signal * return
            merged['strat_ret'] = merged[col] * merged[return_col]
            daily_ret = merged.groupby('date')['strat_ret'].mean()
            
            sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(365)
            sharpes[col] = sharpe
        
        sharpe_series = pd.Series(sharpes)
        sharpe_positive = sharpe_series.clip(lower=0)
        
        if sharpe_positive.sum() == 0:
            return {col: 1.0 / len(signal_cols) for col in signal_cols}
        
        weights = (sharpe_positive / sharpe_positive.sum()).to_dict()
        
        return weights
    
    def _fit_stacking(
        self,
        signals_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        return_col: str
    ):
        """Fit stacking meta-model."""
        signal_cols = [c for c in signals_df.columns if c not in ['date', 'symbol']]
        
        merged = signals_df.merge(
            returns_df[['date', 'symbol', return_col]],
            on=['date', 'symbol'],
            how='inner'
        )
        
        X = merged[signal_cols].values
        y = merged[return_col].values
        
        # Use Ridge for regularization
        self.meta_model_ = Ridge(alpha=1.0)
        self.meta_model_.fit(X, y)
        
        # Extract weights
        self.weights_ = dict(zip(signal_cols, self.meta_model_.coef_))
    
    def combine(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine signals using fitted weights.
        
        Args:
            signals_df: DataFrame with signal columns
            
        Returns:
            DataFrame with combined 'ensemble' signal
        """
        signal_cols = [c for c in signals_df.columns if c not in ['date', 'symbol']]
        
        result = signals_df[['date', 'symbol']].copy()
        
        if self.method == 'stacking' and self.meta_model_ is not None:
            X = signals_df[signal_cols].values
            result['signal'] = self.meta_model_.predict(X)
        else:
            if self.weights_ is None:
                raise ValueError("Ensemble not fitted. Call fit() first.")
            
            result['signal'] = sum(
                signals_df[col] * weight 
                for col, weight in self.weights_.items()
                if col in signals_df.columns
            )
        
        result['signal_name'] = f'ensemble_{self.method}'
        
        return result


class RegimeAdaptiveEnsemble(SignalEnsemble):
    """
    Ensemble with regime-dependent weights.
    """
    
    def __init__(self, regime_detector=None, config: Dict[str, Any] = None):
        """
        Initialize regime-adaptive ensemble.
        
        Args:
            regime_detector: RegimeDetector instance
            config: Configuration dictionary
        """
        super().__init__(method='regime_adaptive', config=config)
        self.regime_detector = regime_detector
        self.regime_weights_ = {}
    
    def fit(
        self,
        signals_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        price_df: pd.DataFrame,
        return_col: str = 'fwd_ret_1d'
    ) -> 'RegimeAdaptiveEnsemble':
        """
        Fit regime-dependent weights.
        
        Args:
            signals_df: Signal DataFrame
            returns_df: Returns DataFrame
            price_df: Price DataFrame for regime detection
            return_col: Return column
            
        Returns:
            self
        """
        if self.regime_detector is None:
            from .regime_detection import RegimeDetector
            self.regime_detector = RegimeDetector()
            self.regime_detector.fit(price_df)
        
        # Get regimes
        regimes = self.regime_detector.predict(price_df)
        
        # Merge with signals
        merged = signals_df.merge(
            regimes[['date', 'regime_label']],
            on='date',
            how='inner'
        )
        
        merged = merged.merge(
            returns_df[['date', 'symbol', return_col]],
            on=['date', 'symbol'],
            how='inner'
        )
        
        signal_cols = [c for c in signals_df.columns if c not in ['date', 'symbol']]
        
        # Fit weights per regime
        for regime in merged['regime_label'].unique():
            regime_data = merged[merged['regime_label'] == regime]
            
            ics = {}
            for col in signal_cols:
                ic_series = regime_data.groupby('date').apply(
                    lambda x: x[col].corr(x[return_col], method='spearman')
                )
                ics[col] = ic_series.mean()
            
            ic_series = pd.Series(ics)
            ic_positive = ic_series.clip(lower=0)
            
            if ic_positive.sum() > 0:
                self.regime_weights_[regime] = (ic_positive / ic_positive.sum()).to_dict()
            else:
                self.regime_weights_[regime] = {col: 1.0 / len(signal_cols) for col in signal_cols}
        
        log.info(f"Regime weights: {self.regime_weights_}")
        
        return self
    
    def combine(
        self,
        signals_df: pd.DataFrame,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine signals using regime-dependent weights.
        
        Args:
            signals_df: Signal DataFrame
            price_df: Price DataFrame for regime detection
            
        Returns:
            Combined signal DataFrame
        """
        # Detect current regime
        regimes = self.regime_detector.predict(price_df)
        
        # Merge with signals
        merged = signals_df.merge(
            regimes[['date', 'regime_label']],
            on='date',
            how='inner'
        )
        
        signal_cols = [c for c in signals_df.columns if c not in ['date', 'symbol']]
        
        # Apply regime-specific weights
        result = merged[['date', 'symbol', 'regime_label']].copy()
        result['signal'] = 0.0
        
        for regime, weights in self.regime_weights_.items():
            mask = merged['regime_label'] == regime
            for col, weight in weights.items():
                if col in merged.columns:
                    result.loc[mask, 'signal'] += merged.loc[mask, col] * weight
        
        result['signal_name'] = 'ensemble_regime_adaptive'
        
        return result[['date', 'symbol', 'signal', 'signal_name', 'regime_label']]
