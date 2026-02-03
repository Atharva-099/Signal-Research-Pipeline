"""
Abstract base class for all data fetchers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import pickle
import hashlib
import time

from ..utils import log, load_config, ensure_dir


class BaseFetcher(ABC):
    """
    Abstract base class for data fetchers.
    
    Provides common functionality for caching, rate limiting,
    and error handling.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the fetcher.
        
        Args:
            config: Configuration dictionary. Loads default if None.
        """
        self.config = config or load_config()
        self._setup_cache()
        self._last_request_time = 0
        
    def _setup_cache(self):
        """Set up the cache directory."""
        cache_config = self.config.get('data', {}).get('cache', {})
        self.cache_enabled = cache_config.get('enabled', True)
        
        if self.cache_enabled:
            cache_dir = cache_config.get('directory', 'data/cache')
            from ..utils import get_project_root
            self.cache_path = get_project_root() / cache_dir
            ensure_dir(self.cache_path)
            self.cache_expiry_hours = cache_config.get('expiry_hours', 24)
        else:
            self.cache_path = None
            
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments."""
        key_str = f"{self.__class__.__name__}_{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache if valid.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Cached DataFrame or None if not found/expired
        """
        if not self.cache_enabled:
            return None
            
        cache_file = self.cache_path / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
            
        # Check expiry
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age > timedelta(hours=self.cache_expiry_hours):
            log.debug(f"Cache expired for {cache_key}")
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            log.debug(f"Cache hit for {cache_key}")
            return data
        except Exception as e:
            log.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """
        Save data to cache.
        
        Args:
            cache_key: Unique cache key
            data: DataFrame to cache
        """
        if not self.cache_enabled:
            return
            
        cache_file = self.cache_path / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            log.debug(f"Cached data for {cache_key}")
        except Exception as e:
            log.warning(f"Failed to save cache: {e}")
    
    def _rate_limit(self, calls_per_minute: int):
        """
        Apply rate limiting between API calls.
        
        Args:
            calls_per_minute: Maximum calls per minute
        """
        min_interval = 60.0 / calls_per_minute
        elapsed = time.time() - self._last_request_time
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            log.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
            
        self._last_request_time = time.time()
    
    @abstractmethod
    def fetch(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data for the specified symbols.
        
        Args:
            symbols: List of asset symbols
            start_date: Start of data range
            end_date: End of data range
            **kwargs: Additional fetch parameters
            
        Returns:
            DataFrame with fetched data
        """
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate fetched data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid
        """
        pass
