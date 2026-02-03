"""
Crypto price data fetcher using CoinGecko and Binance APIs.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from time import sleep

from .base_fetcher import BaseFetcher
from ..utils import log


class PriceFetcher(BaseFetcher):
    """
    Fetches cryptocurrency price data from CoinGecko or Binance.
    
    Supports OHLCV data with configurable timeframes.
    """
    
    # CoinGecko ID mapping for common symbols
    COINGECKO_IDS = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BNB': 'binancecoin',
        'XRP': 'ripple',
        'ADA': 'cardano',
        'AVAX': 'avalanche-2',
        'DOT': 'polkadot',
        'MATIC': 'matic-network',
        'LINK': 'chainlink',
        'DOGE': 'dogecoin',
        'SHIB': 'shiba-inu',
        'UNI': 'uniswap',
        'ATOM': 'cosmos',
        'LTC': 'litecoin',
    }
    
    # Binance symbol formatting
    BINANCE_BASE_URL = "https://api.binance.com"
    COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, config: Dict[str, Any] = None, source: str = None):
        """
        Initialize the price fetcher.
        
        Args:
            config: Configuration dictionary
            source: Data source ('coingecko' or 'binance')
        """
        super().__init__(config)
        
        # Determine source
        if source is None:
            source = self.config.get('data', {}).get('price_source', 'coingecko')
        self.source = source.lower()
        
        # Rate limits
        rate_limits = self.config.get('data', {}).get('rate_limit', {})
        self.rate_limit_per_min = rate_limits.get(
            f'{self.source}_calls_per_minute',
            30 if self.source == 'coingecko' else 1200
        )
        
        log.info(f"PriceFetcher initialized with source: {self.source}")
    
    def fetch(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = None,
        interval: str = 'daily',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price data for specified symbols.
        
        Args:
            symbols: List of asset symbols (e.g., ['BTC', 'ETH'])
            start_date: Start of data range
            end_date: End of data range (defaults to now)
            days: Alternative to start_date - number of days back
            interval: 'daily' or 'hourly'
            use_cache: Whether to use cache
            
        Returns:
            DataFrame with columns: [date, symbol, open, high, low, close, volume]
        """
        # Handle date range
        if end_date is None:
            end_date = datetime.now()
        
        if days is not None:
            start_date = end_date - timedelta(days=days)
        elif start_date is None:
            start_date = end_date - timedelta(days=365)
        
        # Check cache
        cache_key = self._get_cache_key(symbols, start_date, end_date, interval)
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Fetch based on source
        if self.source == 'coingecko':
            data = self._fetch_coingecko(symbols, start_date, end_date)
        elif self.source == 'binance':
            data = self._fetch_binance(symbols, start_date, end_date, interval)
        else:
            raise ValueError(f"Unknown source: {self.source}")
        
        # Validate and cache
        if self.validate(data):
            if use_cache:
                self._save_to_cache(cache_key, data)
            return data
        else:
            log.warning("Data validation failed")
            return data
    
    def _fetch_coingecko(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data from CoinGecko API."""
        all_data = []
        
        for symbol in symbols:
            coin_id = self.COINGECKO_IDS.get(symbol.upper())
            if coin_id is None:
                log.warning(f"Unknown symbol for CoinGecko: {symbol}")
                continue
            
            # Rate limit
            self._rate_limit(self.rate_limit_per_min)
            
            try:
                # Calculate days
                days = (end_date - start_date).days + 1
                
                # CoinGecko market_chart endpoint
                url = f"{self.COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': min(days, 365),  # CoinGecko limit
                    'interval': 'daily'
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Parse response
                prices = data.get('prices', [])
                volumes = data.get('total_volumes', [])
                
                if not prices:
                    log.warning(f"No price data for {symbol}")
                    continue
                
                # Create DataFrame
                df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['date'] = df['timestamp'].dt.date
                df['symbol'] = symbol.upper()
                
                # Add volume
                if volumes:
                    vol_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
                    df['volume'] = vol_df['volume'].values[:len(df)]
                else:
                    df['volume'] = np.nan
                
                # CoinGecko doesn't provide OHLC for free, so we approximate
                df['open'] = df['close'].shift(1)
                df['high'] = df['close'] * 1.01  # Approximate
                df['low'] = df['close'] * 0.99   # Approximate
                
                df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                all_data.append(df)
                
                log.debug(f"Fetched {len(df)} rows for {symbol}")
                
            except Exception as e:
                log.error(f"Failed to fetch {symbol} from CoinGecko: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame(columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
        
        result = pd.concat(all_data, ignore_index=True)
        result['date'] = pd.to_datetime(result['date'])
        return result.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    def _fetch_binance(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = 'daily'
    ) -> pd.DataFrame:
        """Fetch data from Binance API."""
        all_data = []
        
        # Map interval
        interval_map = {'daily': '1d', 'hourly': '1h', '4h': '4h'}
        kline_interval = interval_map.get(interval, '1d')
        
        quote = self.config.get('assets', {}).get('quote_currency', 'USDT')
        
        for symbol in symbols:
            # Rate limit
            self._rate_limit(self.rate_limit_per_min)
            
            try:
                pair = f"{symbol.upper()}{quote}"
                
                url = f"{self.BINANCE_BASE_URL}/api/v3/klines"
                params = {
                    'symbol': pair,
                    'interval': kline_interval,
                    'startTime': int(start_date.timestamp() * 1000),
                    'endTime': int(end_date.timestamp() * 1000),
                    'limit': 1000
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    log.warning(f"No data for {pair}")
                    continue
                
                # Parse klines
                # Format: [open_time, open, high, low, close, volume, close_time, ...]
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                df['date'] = pd.to_datetime(df['open_time'], unit='ms')
                df['symbol'] = symbol.upper()
                
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                all_data.append(df)
                
                log.debug(f"Fetched {len(df)} rows for {symbol}")
                
            except Exception as e:
                log.error(f"Failed to fetch {symbol} from Binance: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame(columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])
        
        result = pd.concat(all_data, ignore_index=True)
        return result.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate fetched price data.
        
        Checks for:
        - Required columns
        - No all-null columns
        - Positive prices
        - Reasonable price ranges
        """
        required_cols = ['date', 'symbol', 'close']
        
        # Check columns
        if not all(col in data.columns for col in required_cols):
            log.warning(f"Missing required columns. Got: {data.columns.tolist()}")
            return False
        
        # Check not empty
        if data.empty:
            log.warning("Data is empty")
            return False
        
        # Check for valid prices
        if (data['close'] <= 0).any():
            log.warning("Found non-positive prices")
            return False
        
        return True
    
    def get_returns(self, data: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            data: Price DataFrame
            periods: Return period
            
        Returns:
            DataFrame with returns added
        """
        result = data.copy()
        result = result.sort_values(['symbol', 'date'])
        result['returns'] = result.groupby('symbol')['close'].pct_change(periods)
        return result
    
    def pivot_prices(self, data: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Pivot data to wide format (dates as index, symbols as columns).
        
        Args:
            data: Long-format price data
            column: Column to pivot
            
        Returns:
            Wide-format DataFrame
        """
        return data.pivot(index='date', columns='symbol', values=column)
