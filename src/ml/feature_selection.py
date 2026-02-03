"""
ML-based feature selection and signal discovery.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

from ..utils import log, load_config


class FeatureSelector:
    """
    ML-based feature selection using tree models and SHAP.
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        n_estimators: int = 100,
        config: Dict[str, Any] = None
    ):
        """
        Initialize feature selector.
        
        Args:
            model_type: 'xgboost', 'rf', or 'gbm'
            n_estimators: Number of trees
            config: Configuration dictionary
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.config = config or load_config()
        self.model = None
        self.feature_importance_ = None
        self.shap_values_ = None
        
        log.info(f"FeatureSelector initialized with {model_type}")
    
    def _create_model(self):
        """Create the tree model."""
        if self.model_type == 'xgboost':
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=5,
                    learning_rate=0.1,
                    verbosity=0,
                    n_jobs=-1
                )
            except ImportError:
                log.warning("XGBoost not available, falling back to RF")
                self.model_type = 'rf'
        
        if self.model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=5,
                n_jobs=-1,
                random_state=42
            )
        
        if self.model_type == 'gbm':
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str] = None
    ) -> 'FeatureSelector':
        """
        Fit model and calculate feature importance.
        
        Args:
            X: Feature matrix
            y: Target (forward returns)
            feature_names: Optional feature names
            
        Returns:
            self
        """
        # Handle missing values
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Align indices
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common]
        
        if len(X) < 50:
            log.warning("Insufficient data for feature selection")
            return self
        
        feature_names = feature_names or X.columns.tolist()
        
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X, y)
        
        # Get feature importance
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)
        
        log.info(f"Fit complete. Top 5 features: {self.feature_importance_.head().to_dict()}")
        
        return self
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.feature_importance_
    
    def select_top_features(self, n: int = 10) -> List[str]:
        """
        Select top N features by importance.
        
        Args:
            n: Number of features to select
            
        Returns:
            List of top feature names
        """
        return self.feature_importance_.head(n).index.tolist()
    
    def calculate_shap_values(
        self,
        X: pd.DataFrame,
        sample_size: int = 100
    ) -> pd.DataFrame:
        """
        Calculate SHAP values for interpretability.
        
        Args:
            X: Feature matrix
            sample_size: Number of samples for SHAP
            
        Returns:
            DataFrame of SHAP values
        """
        try:
            import shap
        except ImportError:
            log.warning("SHAP not available")
            return pd.DataFrame()
        
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = X.fillna(0)
        
        # Sample data for speed
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X
        
        # Create explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # Create DataFrame
        self.shap_values_ = pd.DataFrame(
            shap_values,
            columns=X.columns,
            index=X_sample.index
        )
        
        return self.shap_values_
    
    def get_shap_summary(self) -> pd.DataFrame:
        """
        Get SHAP summary (mean absolute SHAP per feature).
        
        Returns:
            DataFrame with feature and mean SHAP
        """
        if self.shap_values_ is None:
            raise ValueError("SHAP not calculated. Call calculate_shap_values() first.")
        
        mean_shap = self.shap_values_.abs().mean().sort_values(ascending=False)
        
        return pd.DataFrame({
            'feature': mean_shap.index,
            'mean_abs_shap': mean_shap.values,
            'importance_rank': range(1, len(mean_shap) + 1)
        })


class SignalDiscovery:
    """
    Discover new signals using ML feature importance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize signal discovery."""
        self.config = config or load_config()
        self.selector = FeatureSelector()
        self.discovered_features_ = None
    
    def create_feature_matrix(
        self,
        price_df: pd.DataFrame,
        max_lookbacks: List[int] = [5, 10, 20, 60]
    ) -> pd.DataFrame:
        """
        Create comprehensive feature matrix.
        
        Args:
            price_df: Price DataFrame with 'date', 'symbol', 'close', 'volume'
            max_lookbacks: Lookback periods to use
            
        Returns:
            Feature matrix
        """
        features = {}
        
        df = price_df.sort_values(['symbol', 'date']).copy()
        
        # Log returns
        df['log_return'] = df.groupby('symbol')['close'].transform(
            lambda x: np.log(x / x.shift(1))
        )
        
        for lb in max_lookbacks:
            # Momentum
            features[f'momentum_{lb}d'] = df.groupby('symbol')['close'].transform(
                lambda x: x / x.shift(lb) - 1
            )
            
            # Volatility
            features[f'volatility_{lb}d'] = df.groupby('symbol')['log_return'].transform(
                lambda x: x.rolling(lb, min_periods=1).std()
            )
            
            # Volume change
            if 'volume' in df.columns:
                features[f'volume_ma_{lb}d'] = df.groupby('symbol')['volume'].transform(
                    lambda x: x / x.rolling(lb, min_periods=1).mean() - 1
                )
            
            # Mean reversion (z-score)
            features[f'zscore_{lb}d'] = df.groupby('symbol')['close'].transform(
                lambda x: (x - x.rolling(lb, min_periods=1).mean()) / 
                          x.rolling(lb, min_periods=1).std()
            )
            
            # Skewness of returns
            features[f'skew_{lb}d'] = df.groupby('symbol')['log_return'].transform(
                lambda x: x.rolling(lb, min_periods=lb).skew()
            )
        
        # Combine into DataFrame
        feature_df = pd.DataFrame(features)
        feature_df['date'] = df['date'].values
        feature_df['symbol'] = df['symbol'].values
        
        return feature_df
    
    def discover(
        self,
        price_df: pd.DataFrame,
        forward_period: int = 1,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Discover important features for predicting returns.
        
        Args:
            price_df: Price DataFrame
            forward_period: Forward return period
            top_n: Number of top features to return
            
        Returns:
            Dictionary with discovered features and analysis
        """
        # Create features
        log.info("Creating feature matrix...")
        feature_df = self.create_feature_matrix(price_df)
        
        # Create target (forward returns)
        price_df = price_df.sort_values(['symbol', 'date'])
        target = price_df.groupby('symbol')['close'].transform(
            lambda x: x.shift(-forward_period) / x - 1
        )
        
        # Merge
        feature_cols = [c for c in feature_df.columns if c not in ['date', 'symbol']]
        
        # Combine all data
        merged = feature_df.copy()
        merged['target'] = target.values
        merged = merged.dropna()
        
        if len(merged) < 100:
            log.warning("Insufficient data for discovery")
            return {'features': [], 'importance': pd.Series()}
        
        # Fit selector
        log.info("Fitting feature selector...")
        X = merged[feature_cols]
        y = merged['target']
        
        self.selector.fit(X, y, feature_cols)
        
        # Get results
        self.discovered_features_ = self.selector.select_top_features(top_n)
        importance = self.selector.get_feature_importance()
        
        # SHAP analysis
        log.info("Calculating SHAP values...")
        try:
            self.selector.calculate_shap_values(X)
            shap_summary = self.selector.get_shap_summary()
        except Exception as e:
            log.warning(f"SHAP calculation failed: {e}")
            shap_summary = pd.DataFrame()
        
        return {
            'features': self.discovered_features_,
            'importance': importance,
            'shap_summary': shap_summary
        }
