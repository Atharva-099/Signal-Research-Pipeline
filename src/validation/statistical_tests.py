"""
Statistical tests for signal validation.
"""

from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
from scipy import stats


def bootstrap_statistic(
    data: pd.Series,
    statistic_func,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Bootstrap confidence intervals for any statistic.
    
    Args:
        data: Data series
        statistic_func: Function to compute statistic
        n_iterations: Number of bootstrap samples
        confidence_level: Confidence level for interval
        random_state: Random seed
        
    Returns:
        Dict with point estimate and confidence interval
    """
    np.random.seed(random_state)
    
    data = data.dropna().values
    n = len(data)
    
    if n < 10:
        return {'estimate': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=n, replace=True)
        stat = statistic_func(sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return {
        'estimate': statistic_func(data),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std_error': bootstrap_stats.std()
    }


def bootstrap_sharpe(
    returns: pd.Series,
    n_iterations: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for Sharpe ratio.
    
    Args:
        returns: Returns series
        n_iterations: Bootstrap iterations
        confidence_level: Confidence level
        
    Returns:
        Dict with Sharpe and CI
    """
    def sharpe_func(r):
        return r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0
    
    return bootstrap_statistic(returns, sharpe_func, n_iterations, confidence_level)


def bootstrap_ic(
    ic_series: pd.Series,
    n_iterations: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for mean IC.
    
    Args:
        ic_series: IC time series
        n_iterations: Bootstrap iterations
        confidence_level: Confidence level
        
    Returns:
        Dict with IC mean and CI
    """
    return bootstrap_statistic(ic_series, np.mean, n_iterations, confidence_level)


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    returns_skew: float = 0,
    returns_kurtosis: float = 3,
    n_observations: int = 252,
    expected_sharpe: float = 0
) -> Tuple[float, float]:
    """
    Calculate Deflated Sharpe Ratio (Lopez de Prado).
    
    Adjusts Sharpe ratio for multiple testing.
    
    Args:
        sharpe: Observed Sharpe ratio
        n_trials: Number of strategies/signals tested
        returns_skew: Skewness of returns
        returns_kurtosis: Kurtosis of returns
        n_observations: Number of return observations
        expected_sharpe: Expected Sharpe under null hypothesis
        
    Returns:
        Tuple of (deflated_sharpe, p_value)
    """
    # Expected maximum Sharpe from random strategies
    from scipy.stats import norm
    
    # Euler-Mascheroni constant
    gamma = 0.5772156649
    
    # Expected max Sharpe from N random strategies
    z = (1 - gamma) * norm.ppf(1 - 1 / n_trials) + gamma * norm.ppf(1 - 1 / (n_trials * np.e))
    expected_max_sharpe = z
    
    # Adjust for non-normal returns
    sharpe_std = np.sqrt(
        (1 + 0.5 * sharpe**2 - returns_skew * sharpe + 
         (returns_kurtosis - 3) / 4 * sharpe**2) / n_observations
    )
    
    # Test statistic
    if sharpe_std > 0:
        t_stat = (sharpe - expected_max_sharpe) / sharpe_std
        p_value = norm.cdf(t_stat)
    else:
        t_stat = np.nan
        p_value = np.nan
    
    # Deflated Sharpe
    deflated = sharpe - expected_max_sharpe
    
    return deflated, p_value


def ic_significance_test(
    ic_series: pd.Series,
    null_ic: float = 0
) -> Dict[str, float]:
    """
    Test if IC is significantly different from null hypothesis.
    
    Args:
        ic_series: IC time series
        null_ic: Null hypothesis IC value
        
    Returns:
        Dict with t-statistic and p-value
    """
    ic = ic_series.dropna()
    
    if len(ic) < 3:
        return {'t_stat': np.nan, 'p_value': np.nan, 'significant': False}
    
    t_stat, p_value = stats.ttest_1samp(ic, null_ic)
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n': len(ic)
    }


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Benjamini-Hochberg FDR correction for multiple testing.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        Dict with adjusted p-values and significance
    """
    n = len(p_values)
    
    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # BH threshold
    thresholds = alpha * np.arange(1, n + 1) / n
    
    # Find significant
    significant = sorted_p <= thresholds
    
    # Adjusted p-values
    adjusted_p = np.minimum(1, sorted_p * n / np.arange(1, n + 1))
    
    # Reorder to original
    original_order_adjusted = np.empty(n)
    original_order_adjusted[sorted_indices] = adjusted_p
    
    original_order_significant = np.empty(n, dtype=bool)
    original_order_significant[sorted_indices] = significant
    
    return {
        'adjusted_p_values': original_order_adjusted.tolist(),
        'significant': original_order_significant.tolist(),
        'n_significant': sum(significant)
    }


class OverfitDetector:
    """
    Detects potential overfitting in signal backtests.
    """
    
    def __init__(
        self,
        min_sharpe: float = 0.5,
        min_ic: float = 0.02,
        significance_level: float = 0.05
    ):
        """
        Initialize detector.
        
        Args:
            min_sharpe: Minimum acceptable Sharpe
            min_ic: Minimum acceptable IC
            significance_level: Significance level for tests
        """
        self.min_sharpe = min_sharpe
        self.min_ic = min_ic
        self.significance_level = significance_level
    
    def check_train_test_gap(
        self,
        train_sharpe: float,
        test_sharpe: float,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Check for significant train/test performance gap.
        
        Large gaps indicate overfitting.
        """
        gap = train_sharpe - test_sharpe
        gap_ratio = gap / train_sharpe if train_sharpe != 0 else np.inf
        
        return {
            'gap': gap,
            'gap_ratio': gap_ratio,
            'overfit_flag': gap_ratio > threshold,
            'severity': 'high' if gap_ratio > 0.7 else 'medium' if gap_ratio > 0.5 else 'low'
        }
    
    def check_parameter_sensitivity(
        self,
        sharpes: List[float],
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Check if performance is sensitive to parameters.
        
        High sensitivity suggests overfitting.
        """
        sharpes = np.array(sharpes)
        
        cv = sharpes.std() / sharpes.mean() if sharpes.mean() != 0 else np.inf
        
        return {
            'cv': cv,
            'range': sharpes.max() - sharpes.min(),
            'sensitive': cv > threshold,
            'n_params': len(sharpes)
        }
    
    def comprehensive_check(
        self,
        ic_series: pd.Series,
        sharpe: float,
        n_trials: int,
        train_sharpe: float = None,
        test_sharpe: float = None
    ) -> Dict[str, any]:
        """
        Run comprehensive overfit detection.
        
        Returns:
            Dictionary of all checks and overall assessment
        """
        checks = {}
        flags = []
        
        # IC significance
        ic_test = ic_significance_test(ic_series)
        checks['ic_test'] = ic_test
        if not ic_test['significant']:
            flags.append('IC not statistically significant')
        
        # IC bootstrap
        ic_bootstrap = bootstrap_ic(ic_series)
        checks['ic_bootstrap'] = ic_bootstrap
        if ic_bootstrap['ci_lower'] < 0:
            flags.append('IC confidence interval includes zero')
        
        # Deflated Sharpe
        if n_trials > 1:
            deflated, p_val = deflated_sharpe_ratio(sharpe, n_trials)
            checks['deflated_sharpe'] = {'deflated': deflated, 'p_value': p_val}
            if deflated < self.min_sharpe:
                flags.append('Deflated Sharpe below threshold')
        
        # Train/test gap
        if train_sharpe is not None and test_sharpe is not None:
            gap_check = self.check_train_test_gap(train_sharpe, test_sharpe)
            checks['train_test_gap'] = gap_check
            if gap_check['overfit_flag']:
                flags.append('Large train/test performance gap')
        
        # Overall assessment
        checks['flags'] = flags
        checks['overfit_risk'] = 'high' if len(flags) >= 2 else 'medium' if len(flags) == 1 else 'low'
        
        return checks
