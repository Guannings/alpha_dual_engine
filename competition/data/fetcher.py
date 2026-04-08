"""
TAIFEX OpenAPI client for historical and daily market data.
Base URL: https://openapi.taifex.com.tw/v1
"""
import logging
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = 'https://openapi.taifex.com.tw/v1'

# TAIFEX OpenAPI endpoints
ENDPOINTS = {
    'futures_daily': '/DailyMarketReportFut',
    'options_daily': '/DailyMarketReportOpt',
    'options_delta': '/DailyOptionsDelta',
    'institutional': '/MarketDataOfMajorInstitutionalTradersGeneralBytheDate',
    'put_call_ratio': '/PutCallRatio',
    'futures_large_traders': '/LargeTradersFutQry',
}

# Local cache directory
CACHE_DIR = Path(__file__).parent.parent / 'data_cache'


class TAIFEXFetcher:
    """Client for TAIFEX OpenAPI data retrieval."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'TAIFEX-CompBot/1.0',
        })

    def _fetch_json(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """Fetch JSON data from TAIFEX OpenAPI."""
        url = BASE_URL + endpoint
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"TAIFEX API error ({endpoint}): {e}")
            return []

    def fetch_futures_daily(self, symbol: str = 'TX') -> pd.DataFrame:
        """Fetch daily futures market report (latest available date)."""
        data = self._fetch_json(ENDPOINTS['futures_daily'])
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        # Filter for the requested symbol
        if 'ContractCode' in df.columns:
            df = df[df['ContractCode'].str.strip() == symbol]
        return df

    def fetch_options_daily(self, symbol: str = 'TXO') -> pd.DataFrame:
        """Fetch daily options market report."""
        data = self._fetch_json(ENDPOINTS['options_daily'])
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if 'ContractCode' in df.columns:
            df = df[df['ContractCode'].str.strip() == symbol]
        return df

    def fetch_institutional_data(self) -> pd.DataFrame:
        """Fetch institutional traders' buy/sell data."""
        data = self._fetch_json(ENDPOINTS['institutional'])
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def fetch_put_call_ratio(self) -> pd.DataFrame:
        """Fetch put/call ratio data."""
        data = self._fetch_json(ENDPOINTS['put_call_ratio'])
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    def fetch_options_delta(self) -> pd.DataFrame:
        """Fetch options delta data for Greeks tracking."""
        data = self._fetch_json(ENDPOINTS['options_delta'])
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    # --- Historical Data (CSV backfill from cache or TAIFEX downloads) ---

    def load_historical_futures(self, symbol: str = 'TX',
                                start_date: str = '2020-01-01',
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """Load historical futures data from local CSV cache.

        TAIFEX OpenAPI only provides the latest day's data.
        Historical data must be downloaded manually from:
        https://www.taifex.com.tw/cht/3/dlFutDataDown
        and saved to data_cache/{symbol}_daily.csv
        """
        cache_file = self.cache_dir / f'{symbol}_daily.csv'
        if not cache_file.exists():
            logger.warning(
                f"No historical data for {symbol}. "
                f"Download from https://www.taifex.com.tw/cht/3/dlFutDataDown "
                f"and save to {cache_file}"
            )
            return pd.DataFrame()

        df = pd.read_csv(cache_file, parse_dates=['Date'])
        df = df.set_index('Date').sort_index()

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df

    def load_historical_options(self, symbol: str = 'TXO',
                                start_date: str = '2020-01-01',
                                end_date: Optional[str] = None) -> pd.DataFrame:
        """Load historical options data from local CSV cache."""
        cache_file = self.cache_dir / f'{symbol}_daily.csv'
        if not cache_file.exists():
            logger.warning(f"No historical data for {symbol}. Save CSV to {cache_file}")
            return pd.DataFrame()

        df = pd.read_csv(cache_file, parse_dates=['Date'])
        df = df.set_index('Date').sort_index()

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df

    def load_historical_institutional(self, start_date: str = '2020-01-01',
                                      end_date: Optional[str] = None) -> pd.DataFrame:
        """Load historical institutional flow data from CSV cache."""
        cache_file = self.cache_dir / 'institutional_flow.csv'
        if not cache_file.exists():
            logger.warning(f"No institutional flow data. Save CSV to {cache_file}")
            return pd.DataFrame()

        df = pd.read_csv(cache_file, parse_dates=['Date'])
        df = df.set_index('Date').sort_index()

        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df

    def save_daily_snapshot(self, symbol: str = 'TX') -> None:
        """Fetch today's data from API and append to historical CSV cache."""
        df = self.fetch_futures_daily(symbol)
        if df.empty:
            logger.warning(f"No data to save for {symbol}")
            return

        cache_file = self.cache_dir / f'{symbol}_daily.csv'
        if cache_file.exists():
            existing = pd.read_csv(cache_file)
            combined = pd.concat([existing, df], ignore_index=True).drop_duplicates()
            combined.to_csv(cache_file, index=False)
        else:
            df.to_csv(cache_file, index=False)

        logger.info(f"Saved daily snapshot for {symbol} to {cache_file}")

    def backfill_from_api(self, symbols: Optional[List[str]] = None,
                          sleep_seconds: float = 1.0) -> None:
        """Fetch and save daily data for multiple symbols.

        Note: TAIFEX OpenAPI only returns the latest day, so this must be
        run daily to build up a historical dataset.
        """
        symbols = symbols or ['TX', 'MTX', 'TXO', 'GDF', 'UDF', 'XIF']
        for sym in symbols:
            self.save_daily_snapshot(sym)
            _time.sleep(sleep_seconds)  # Rate limiting
