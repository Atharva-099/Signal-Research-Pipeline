"""
Hidden Markov Model based regime detection.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from ..utils import log, load_config


class RegimeDetector:
    """
    Detect market regimes using Hidden Markov Models.
    
    Identifies latent states like bull/bear/sideways markets.
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        features: List[str] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of hidden states
            features: Features to use for regime detection
            config: Configuration dictionary
        """
        self.n_regimes = n_regimes
        self.features = features or ['returns', 'volatility', 'volume']
        self.config = config or load_config()
        
        self.model = None
        self.scaler = StandardScaler()
        self.regime_labels = {}
        
        log.info(f"RegimeDetector initialized with {n_regimes} regimes")
    
    def _create_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for regime detection.
        
        Args:
            price_df: Price DataFrame
            
        Returns:
            Feature DataFrame for HMM
        """
        df = price_df.sort_values(['symbol', 'date']).copy()
        
        # Calculate returns
        df['returns'] = df.groupby('symbol')['close'].pct_change()
        
        # Rolling volatility
        df['volatility'] = df.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )
        
        # Volume if available
        if 'volume' in df.columns:
            df['volume_z'] = df.groupby('symbol')['volume'].transform(
                lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()
            )
        else:
            df['volume_z'] = 0
        
        return df[['date', 'symbol', 'returns', 'volatility', 'volume_z']].dropna()
    
    def fit(self, price_df: pd.DataFrame) -> 'RegimeDetector':
        """
        Fit HMM to price data.
        
        Args:
            price_df: Price DataFrame
            
        Returns:
            self
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            log.error("hmmlearn not installed. Run: pip install hmmlearn")
            return self
        
        # Create features
        feature_df = self._create_features(price_df)
        
        # Aggregate across symbols for market-wide regime
        market_features = feature_df.groupby('date').agg({
            'returns': 'mean',
            'volatility': 'mean',
            'volume_z': 'mean'
        }).reset_index()
        
        # Prepare data
        X = market_features[['returns', 'volatility', 'volume_z']].values
        X = self.scaler.fit_transform(X)
        
        # Fit HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        
        self.model.fit(X)
        
        # Decode to get regimes
        regimes = self.model.predict(X)
        
        # Label regimes based on mean returns
        self._label_regimes(market_features, regimes)
        
        log.info(f"HMM fitted. Regimes: {self.regime_labels}")
        
        return self
    
    def _label_regimes(self, df: pd.DataFrame, regimes: np.ndarray):
        """Label regimes based on characteristics."""
        df = df.copy()
        df['regime'] = regimes
        
        # Calculate mean return per regime
        regime_stats = df.groupby('regime').agg({
            'returns': 'mean',
            'volatility': 'mean'
        })
        
        # Label based on returns
        sorted_by_return = regime_stats['returns'].sort_values()
        
        for i, (regime, _) in enumerate(sorted_by_return.items()):
            if i == 0:
                self.regime_labels[regime] = 'bear'
            elif i == len(sorted_by_return) - 1:
                self.regime_labels[regime] = 'bull'
            else:
                self.regime_labels[regime] = 'neutral'
    
    def predict(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regimes for new data.
        
        Args:
            price_df: Price DataFrame
            
        Returns:
            DataFrame with regime predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create features
        feature_df = self._create_features(price_df)
        
        # Aggregate for market-wide
        market_features = feature_df.groupby('date').agg({
            'returns': 'mean',
            'volatility': 'mean',
            'volume_z': 'mean'
        }).reset_index()
        
        # Predict
        X = market_features[['returns', 'volatility', 'volume_z']].values
        X = self.scaler.transform(X)
        
        regimes = self.model.predict(X)
        regime_probs = self.model.predict_proba(X)
        
        # Create result
        result = market_features[['date']].copy()
        result['regime'] = regimes
        result['regime_label'] = result['regime'].map(self.regime_labels)
        
        # Add probabilities
        for i in range(self.n_regimes):
            label = self.regime_labels.get(i, f'regime_{i}')
            result[f'prob_{label}'] = regime_probs[:, i]
        
        return result
    
    def get_regime_statistics(
        self,
        price_df: pd.DataFrame,
        signal_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Calculate statistics per regime.
        
        Args:
            price_df: Price DataFrame
            signal_df: Optional signal DataFrame for conditional analysis
            
        Returns:
            DataFrame with regime statistics
        """
        # Get regimes
        regime_df = self.predict(price_df)
        
        # Merge with returns
        price_df = price_df.sort_values(['symbol', 'date']).copy()
        price_df['returns'] = price_df.groupby('symbol')['close'].pct_change()
        
        # Aggregate daily returns
        daily_returns = price_df.groupby('date')['returns'].mean().reset_index()
        daily_returns = daily_returns.merge(regime_df, on='date', how='inner')
        
        # Stats per regime
        stats = daily_returns.groupby('regime_label').agg({
            'returns': ['mean', 'std', 'count'],
        })
        
        stats.columns = ['mean_return', 'volatility', 'n_days']
        stats['sharpe'] = stats['mean_return'] / stats['volatility'] * np.sqrt(365)
        stats['pct_time'] = stats['n_days'] / stats['n_days'].sum()
        
        return stats.reset_index()


class StructuralBreakDetector:
    """
    Detect structural breaks (regime changes) in time series.
    
    Uses PELT or Binseg algorithms.
    """
    
    def __init__(self, method: str = 'pelt', min_size: int = 30):
        """
        Initialize detector.
        
        Args:
            method: 'pelt' or 'binseg'
            min_size: Minimum segment size
        """
        self.method = method
        self.min_size = min_size
        self.breakpoints_ = None
    
    def detect(self, series: pd.Series) -> List[int]:
        """
        Detect structural breaks in series.
        
        Args:
            series: Time series to analyze
            
        Returns:
            List of breakpoint indices
        """
        try:
            import ruptures as rpt
        except ImportError:
            log.warning("ruptures not installed. Run: pip install ruptures")
            return []
        
        signal = series.dropna().values.reshape(-1, 1)
        
        if self.method == 'pelt':
            algo = rpt.Pelt(model='rbf', min_size=self.min_size)
        else:
            algo = rpt.Binseg(model='l2', min_size=self.min_size)
        
        algo.fit(signal)
        self.breakpoints_ = algo.predict(pen=1.0)
        
        return self.breakpoints_
