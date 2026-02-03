"""
Funding rate data fetcher for perpetual futures.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

from .base_fetcher import BaseFetcher
from ..utils import log


class FundingFetcher(BaseFetcher):
    """
    Fetches funding rate data from Binance Futures.
    
    Funding rates are a key indicator for market sentiment
    in perpetual futures markets.
    """
    
    BINANCE_FUTURES_URL = "https://fapi.binance.com"
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the funding rate fetcher."""
        super().__init__(config)
        log.info("FundingFetcher initialized")
    
    def fetch(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 30,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch funding rate history.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date
            end_date: End date
            days: Number of days (alternative to start_date)
            use_cache: Whether to use cache
            
        Returns:
            DataFrame with funding rate data
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days)
        
        # Check cache
        cache_key = self._get_cache_key(symbols, start_date, end_date)
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached
        
        all_data = []
        quote = self.config.get('assets', {}).get('quote_currency', 'USDT')
        
        for symbol in symbols:
            self._rate_limit(1200)  # Binance rate limit
            
            try:
                pair = f"{symbol.upper()}{quote}"
                
                url = f"{self.BINANCE_FUTURES_URL}/fapi/v1/fundingRate"
                params = {
                    'symbol': pair,
                    'startTime': int(start_date.timestamp() * 1000),
                    'endTime': int(end_date.timestamp() * 1000),
                    'limit': 1000
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    log.warning(f"No funding data for {symbol}")
                    continue
                
                df = pd.DataFrame(data)
                df['symbol'] = symbol.upper()
                df['date'] = pd.to_datetime(df['fundingTime'], unit='ms')
                df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
                
                df = df[['date', 'symbol', 'funding_rate']]
                all_data.append(df)
                
                log.debug(f"Fetched {len(df)} funding rates for {symbol}")
                
            except Exception as e:
                log.error(f"Failed to fetch funding for {symbol}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame(columns=['date', 'symbol', 'funding_rate'])
        
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        if use_cache:
            self._save_to_cache(cache_key, result)
        
        return result
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate funding rate data."""
        if data.empty:
            return False
        if 'funding_rate' not in data.columns:
            return False
        return True
    
    def aggregate_daily(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate funding rates to daily.
        
        Funding rates are typically recorded every 8 hours.
        
        Args:
            data: Funding rate DataFrame
            
        Returns:
            Daily aggregated funding rates
        """
        data = data.copy()
        data['date'] = pd.to_datetime(data['date']).dt.date
        
        aggregated = data.groupby(['date', 'symbol']).agg({
            'funding_rate': ['mean', 'sum', 'count']
        }).reset_index()
        
        aggregated.columns = ['date', 'symbol', 'funding_rate_mean', 
                              'funding_rate_sum', 'funding_count']
        aggregated['date'] = pd.to_datetime(aggregated['date'])
        
        return aggregated
    
    def calculate_cumulative_funding(
        self,
        data: pd.DataFrame,
        windows: List[int] = [1, 7, 14, 30]
    ) -> pd.DataFrame:
        """
        Calculate cumulative funding over various windows.
        
        Args:
            data: Funding rate DataFrame
            windows: Rolling windows in days
            
        Returns:
            DataFrame with cumulative funding columns
        """
        # First aggregate to daily
        daily = self.aggregate_daily(data)
        
        result = daily.copy()
        result = result.sort_values(['symbol', 'date'])
        
        for window in windows:
            col_name = f'cum_funding_{window}d'
            result[col_name] = result.groupby('symbol')['funding_rate_sum'].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )
        
        return result
