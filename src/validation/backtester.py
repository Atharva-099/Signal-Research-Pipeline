"""
Walk-forward backtesting engine.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .metrics import (
    calculate_ic_series,
    calculate_ir,
    calculate_sharpe,
    calculate_max_drawdown,
    PerformanceMetrics
)
from ..utils import log, load_config


@dataclass
class BacktestResult:
    """Container for backtest results."""
    
    signal_name: str
    start_date: datetime
    end_date: datetime
    
    # IC metrics
    ic_series: pd.Series
    ic_mean: float
    ic_std: float
    ic_ir: float
    
    # Return metrics
    strategy_returns: pd.Series
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
    # Other
    n_periods: int
    turnover: float
    
    def summary(self) -> Dict[str, Any]:
        """Return summary as dictionary."""
        return {
            'signal_name': self.signal_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'ic_mean': self.ic_mean,
            'ic_std': self.ic_std,
            'ic_ir': self.ic_ir,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'n_periods': self.n_periods,
        }
    
    def __repr__(self) -> str:
        return (
            f"BacktestResult({self.signal_name})\n"
            f"  Period: {self.start_date.date()} to {self.end_date.date()}\n"
            f"  IC Mean: {self.ic_mean:.4f}, IR: {self.ic_ir:.2f}\n"
            f"  Sharpe: {self.sharpe_ratio:.2f}, MaxDD: {self.max_drawdown:.2%}"
        )


class Backtester:
    """
    Walk-forward backtesting engine.
    
    Evaluates signals using proper time-series methodology
    to avoid look-ahead bias.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize backtester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.backtest_config = self.config.get('backtest', {})
        
        # Transaction costs
        self.tc_bps = self.backtest_config.get('transaction_cost_bps', 10)
        
        log.info("Backtester initialized")
    
    def run(
        self,
        signal_df: pd.DataFrame,
        price_df: pd.DataFrame,
        forward_period: int = 1,
        signal_name: str = None
    ) -> BacktestResult:
        """
        Run backtest on a signal.
        
        Args:
            signal_df: Signal DataFrame with 'date', 'symbol', 'signal'
            price_df: Price DataFrame with 'date', 'symbol', 'close'
            forward_period: Days ahead for return calculation
            signal_name: Name of signal being tested
            
        Returns:
            BacktestResult object
        """
        signal_name = signal_name or 'unnamed_signal'
        
        log.info(f"Running backtest for {signal_name}")
        
        # Calculate forward returns
        returns_df = self._calculate_forward_returns(price_df, forward_period)
        
        # Merge signal with returns
        merged = signal_df.merge(
            returns_df,
            on=['date', 'symbol'],
            how='inner'
        )
        
        if merged.empty:
            log.warning("No overlapping data between signal and returns")
            return self._empty_result(signal_name)
        
        # Calculate IC series
        return_col = f'fwd_ret_{forward_period}d'
        ic_series = calculate_ic_series(signal_df, returns_df, return_col)
        
        # Calculate strategy returns (simplified: signal-weighted)
        strategy_returns = self._calculate_strategy_returns(merged, return_col)
        
        # Calculate metrics
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = calculate_ir(ic_series)
        
        sharpe = calculate_sharpe(strategy_returns)
        cumulative = (1 + strategy_returns).cumprod()
        max_dd, _, _ = calculate_max_drawdown(cumulative)
        
        # Turnover
        turnover = self._calculate_turnover(signal_df)
        
        result = BacktestResult(
            signal_name=signal_name,
            start_date=merged['date'].min(),
            end_date=merged['date'].max(),
            ic_series=ic_series,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            strategy_returns=strategy_returns,
            sharpe_ratio=sharpe,
            sortino_ratio=np.nan,  # TODO: implement
            max_drawdown=max_dd,
            n_periods=len(ic_series),
            turnover=turnover
        )
        
        log.info(f"Backtest complete: IC={ic_mean:.4f}, Sharpe={sharpe:.2f}")
        
        return result
    
    def _calculate_forward_returns(
        self,
        price_df: pd.DataFrame,
        period: int
    ) -> pd.DataFrame:
        """Calculate forward returns."""
        df = price_df[['date', 'symbol', 'close']].copy()
        df = df.sort_values(['symbol', 'date'])
        
        col_name = f'fwd_ret_{period}d'
        df[col_name] = df.groupby('symbol')['close'].transform(
            lambda x: x.shift(-period) / x - 1
        )
        
        return df[['date', 'symbol', col_name]].dropna()
    
    def _calculate_strategy_returns(
        self,
        merged: pd.DataFrame,
        return_col: str
    ) -> pd.Series:
        """
        Calculate strategy returns from signal.
        
        Simple approach: Returns weighted by signal rank.
        """
        df = merged.copy()
        
        # Rank signals per date
        df['signal_rank'] = df.groupby('date')['signal'].transform(
            lambda x: x.rank(pct=True) * 2 - 1  # -1 to 1
        )
        
        # Weighted return per asset
        df['weighted_return'] = df['signal_rank'] * df[return_col]
        
        # Portfolio return per day (equal weight all assets)
        daily_returns = df.groupby('date')['weighted_return'].mean()
        
        # Apply transaction costs (approximate)
        turnover = df.groupby('date')['signal_rank'].apply(
            lambda x: x.diff().abs().mean()
        ).fillna(0)
        
        tc_adjustment = turnover * (self.tc_bps / 10000)
        daily_returns = daily_returns - tc_adjustment
        
        return daily_returns
    
    def _calculate_turnover(self, signal_df: pd.DataFrame) -> float:
        """Calculate average daily turnover."""
        wide = signal_df.pivot(index='date', columns='symbol', values='signal')
        changes = wide.diff().abs().sum(axis=1) / 2
        return changes.mean()
    
    def _empty_result(self, signal_name: str) -> BacktestResult:
        """Return empty result for failed backtest."""
        return BacktestResult(
            signal_name=signal_name,
            start_date=datetime.now(),
            end_date=datetime.now(),
            ic_series=pd.Series(),
            ic_mean=np.nan,
            ic_std=np.nan,
            ic_ir=np.nan,
            strategy_returns=pd.Series(),
            sharpe_ratio=np.nan,
            sortino_ratio=np.nan,
            max_drawdown=np.nan,
            n_periods=0,
            turnover=np.nan
        )


class WalkForwardBacktester(Backtester):
    """
    Walk-forward backtester with proper train/test splits.
    
    Avoids look-ahead bias by only using data available
    at the time of signal generation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        wf_config = self.backtest_config.get('walk_forward', {})
        self.window_size = wf_config.get('window_size_days', 180)
        self.step_size = wf_config.get('step_size_days', 30)
        self.min_train = wf_config.get('min_train_days', 90)
    
    def run_walk_forward(
        self,
        signal_generator,
        price_df: pd.DataFrame,
        forward_period: int = 1
    ) -> List[BacktestResult]:
        """
        Run walk-forward backtest.
        
        Args:
            signal_generator: Signal class instance
            price_df: Price DataFrame
            forward_period: Forward return period
            
        Returns:
            List of backtest results for each window
        """
        results = []
        
        dates = sorted(price_df['date'].unique())
        
        start_idx = self.min_train
        
        while start_idx < len(dates) - self.step_size:
            train_end = dates[start_idx]
            test_start = dates[start_idx + 1]
            test_end = dates[min(start_idx + self.step_size, len(dates) - 1)]
            
            # Train data
            train_data = price_df[price_df['date'] <= train_end]
            
            # Generate signal on train data
            signal_df = signal_generator.compute(train_data)
            
            # Test data
            test_data = price_df[
                (price_df['date'] >= test_start) & 
                (price_df['date'] <= test_end)
            ]
            
            # Run backtest on test period
            result = self.run(
                signal_df[signal_df['date'] >= test_start],
                test_data,
                forward_period,
                signal_name=f"{signal_generator.name}_wf_{start_idx}"
            )
            
            results.append(result)
            start_idx += self.step_size
        
        return results
