"""
Signal decay tracking and monitoring.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..utils import log, load_config


class DecayTracker:
    """
    Track signal performance decay over time.
    """
    
    def __init__(
        self,
        rolling_window: int = 30,
        config: Dict[str, Any] = None
    ):
        """
        Initialize decay tracker.
        
        Args:
            rolling_window: Window for rolling metrics
            config: Configuration dictionary
        """
        self.rolling_window = rolling_window
        self.config = config or load_config()
        
        log.info(f"DecayTracker initialized with {rolling_window}-day window")
    
    def calculate_rolling_ic(
        self,
        ic_series: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate rolling IC statistics.
        
        Args:
            ic_series: IC time series
            
        Returns:
            DataFrame with rolling IC metrics
        """
        ic = ic_series.dropna()
        
        result = pd.DataFrame(index=ic.index)
        result['ic'] = ic
        result['ic_rolling_mean'] = ic.rolling(self.rolling_window, min_periods=5).mean()
        result['ic_rolling_std'] = ic.rolling(self.rolling_window, min_periods=5).std()
        result['ic_rolling_ir'] = result['ic_rolling_mean'] / result['ic_rolling_std']
        result['ic_cumulative'] = ic.expanding().mean()
        
        return result
    
    def estimate_decay_rate(
        self,
        ic_series: pd.Series
    ) -> Dict[str, float]:
        """
        Estimate signal decay rate.
        
        Uses exponential decay model: IC(t) = IC(0) * exp(-λt)
        
        Args:
            ic_series: IC time series
            
        Returns:
            Dict with decay parameters
        """
        ic = ic_series.dropna()
        
        if len(ic) < 30:
            return {'decay_rate': np.nan, 'half_life': np.nan}
        
        # Calculate cumulative IC over expanding windows
        periods = np.arange(1, len(ic) + 1)
        cumulative_ic = ic.expanding().mean().values
        
        # Fit exponential decay
        # log(IC) = log(IC0) - λ*t
        valid_mask = cumulative_ic > 0
        if valid_mask.sum() < 10:
            return {'decay_rate': np.nan, 'half_life': np.nan}
        
        log_ic = np.log(cumulative_ic[valid_mask])
        t = periods[valid_mask]
        
        # Linear regression
        slope, intercept = np.polyfit(t, log_ic, 1)
        
        decay_rate = -slope
        half_life = np.log(2) / decay_rate if decay_rate > 0 else np.inf
        
        return {
            'decay_rate': decay_rate,
            'half_life': half_life,
            'initial_ic': np.exp(intercept)
        }
    
    def detect_performance_drop(
        self,
        ic_series: pd.Series,
        threshold_pct: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect significant performance drops.
        
        Args:
            ic_series: IC time series
            threshold_pct: Threshold for drop detection
            
        Returns:
            Dict with drop detection results
        """
        rolling = self.calculate_rolling_ic(ic_series)
        
        # Peak rolling IC
        peak_ic = rolling['ic_rolling_mean'].expanding().max()
        
        # Drawdown from peak
        drawdown = (rolling['ic_rolling_mean'] - peak_ic) / peak_ic
        drawdown = drawdown.fillna(0)
        
        # Current status
        current_ic = rolling['ic_rolling_mean'].iloc[-1]
        current_drawdown = drawdown.iloc[-1]
        
        return {
            'current_ic': current_ic,
            'peak_ic': peak_ic.iloc[-1],
            'drawdown': current_drawdown,
            'significant_drop': abs(current_drawdown) > threshold_pct,
            'drop_date': drawdown.idxmin() if (drawdown < -threshold_pct).any() else None
        }


class HealthScore:
    """
    Calculate composite health score for signals.
    """
    
    def __init__(
        self,
        weights: Dict[str, float] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize health scorer.
        
        Args:
            weights: Component weights
            config: Configuration dictionary
        """
        self.config = config or load_config()
        
        monitoring_config = self.config.get('monitoring', {})
        health_weights = monitoring_config.get('health_weights', {})
        
        self.weights = weights or {
            'recent_ic': health_weights.get('recent_ic', 0.4),
            'stability': health_weights.get('stability', 0.2),
            'regime_robustness': health_weights.get('regime_robustness', 0.2),
            'decay_rate': health_weights.get('decay_rate', 0.2)
        }
        
        log.info(f"HealthScore initialized with weights: {self.weights}")
    
    def calculate(
        self,
        ic_series: pd.Series,
        regime_ics: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Calculate composite health score.
        
        Args:
            ic_series: IC time series
            regime_ics: ICs per regime (optional)
            
        Returns:
            Dict with health score and components
        """
        components = {}
        
        # Recent IC score (0-100)
        decay_tracker = DecayTracker()
        rolling = decay_tracker.calculate_rolling_ic(ic_series)
        recent_ic = rolling['ic_rolling_mean'].iloc[-1] if len(rolling) > 0 else 0
        
        # Scale IC to 0-100 (assuming IC range -0.1 to 0.1)
        components['recent_ic'] = max(0, min(100, (recent_ic + 0.1) / 0.2 * 100))
        
        # Stability score (inverse of IC volatility)
        ic_vol = ic_series.std()
        if ic_vol > 0:
            # Lower vol = higher score
            components['stability'] = max(0, min(100, (1 - ic_vol / 0.1) * 100))
        else:
            components['stability'] = 50
        
        # Regime robustness (min IC across regimes / max IC)
        if regime_ics and len(regime_ics) > 1:
            min_ic = min(regime_ics.values())
            max_ic = max(regime_ics.values())
            if max_ic > 0:
                components['regime_robustness'] = (min_ic / max_ic) * 100
            else:
                components['regime_robustness'] = 50
        else:
            components['regime_robustness'] = 50
        
        # Decay rate score (inverse of decay)
        decay_info = decay_tracker.estimate_decay_rate(ic_series)
        if decay_info['half_life'] == np.inf or np.isnan(decay_info['half_life']):
            components['decay_rate'] = 100
        elif decay_info['half_life'] > 0:
            # Longer half-life = better score
            components['decay_rate'] = min(100, decay_info['half_life'] / 365 * 100)
        else:
            components['decay_rate'] = 0
        
        # Composite score
        health_score = sum(
            components[k] * self.weights[k] for k in self.weights
        )
        
        return {
            'health_score': health_score,
            'components': components,
            'status': self._get_status(health_score)
        }
    
    def _get_status(self, score: float) -> str:
        """Get status label from score."""
        if score >= 80:
            return 'healthy'
        elif score >= 60:
            return 'good'
        elif score >= 40:
            return 'warning'
        else:
            return 'critical'


class KillCriteria:
    """
    Determine if signals should be killed (stopped).
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize kill criteria checker."""
        self.config = config or load_config()
        
        kill_config = self.config.get('monitoring', {}).get('kill_criteria', {})
        
        self.min_rolling_ic = kill_config.get('min_rolling_ic', 0.01)
        self.min_rolling_sharpe = kill_config.get('min_rolling_sharpe', 0.3)
        self.max_drawdown = kill_config.get('max_drawdown', 0.25)
        self.consecutive_loss_days = kill_config.get('consecutive_loss_days', 30)
    
    def check(
        self,
        ic_series: pd.Series,
        returns_series: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Check if signal should be killed.
        
        Args:
            ic_series: IC time series
            returns_series: Strategy returns (optional)
            
        Returns:
            Dict with kill decision and reasons
        """
        reasons = []
        
        decay_tracker = DecayTracker()
        rolling = decay_tracker.calculate_rolling_ic(ic_series)
        
        # Check rolling IC
        recent_ic = rolling['ic_rolling_mean'].iloc[-1] if len(rolling) > 0 else 0
        if recent_ic < self.min_rolling_ic:
            reasons.append(f"Rolling IC ({recent_ic:.4f}) below minimum ({self.min_rolling_ic})")
        
        # Check performance drop
        drop_info = decay_tracker.detect_performance_drop(ic_series)
        if abs(drop_info['drawdown']) > self.max_drawdown:
            reasons.append(f"IC drawdown ({drop_info['drawdown']:.1%}) exceeds maximum ({self.max_drawdown:.0%})")
        
        # Check consecutive negative IC
        if len(ic_series) >= self.consecutive_loss_days:
            recent = ic_series.tail(self.consecutive_loss_days)
            if (recent < 0).all():
                reasons.append(f"IC negative for {self.consecutive_loss_days} consecutive days")
        
        return {
            'kill': len(reasons) > 0,
            'reasons': reasons,
            'n_flags': len(reasons)
        }
