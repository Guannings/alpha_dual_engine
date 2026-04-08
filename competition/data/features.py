"""
Taiwan-market feature engineering for regime classification.
Adapted from alpha_engine.py 7-feature pattern (line 207).
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..config import StrategyConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute 7 features for the TAIEX regime classifier.

    Features (adapted from US equity model):
        1. realized_vol          — TAIEX historical volatility (annualized)
        2. vol_momentum          — Change rate of realized vol over 21 days
        3. trend_score           — (TAIEX - 200-SMA) / 200-SMA, scaled
        4. momentum_21d          — 21-day TAIEX return
        5. institutional_flow    — Net buy/sell from three major institutional traders
        6. put_call_ratio        — TXO put/call volume ratio (contrarian signal)
        7. term_structure_slope  — Back month minus front month TX futures spread
    """

    FEATURE_NAMES = [
        'realized_vol',
        'vol_momentum',
        'trend_score',
        'momentum_21d',
        'institutional_flow',
        'put_call_ratio',
        'term_structure_slope',
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

    def compute_features(
        self,
        taiex_prices: pd.Series,
        institutional_net: Optional[pd.Series] = None,
        put_call_ratio: Optional[pd.Series] = None,
        front_month_price: Optional[pd.Series] = None,
        back_month_price: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Compute all 7 features from raw data.

        Args:
            taiex_prices: Daily TAIEX index close (or TX front-month close)
            institutional_net: Daily net buy/sell from institutions (NT$ billions)
            put_call_ratio: Daily TXO put/call volume ratio
            front_month_price: TX front-month settlement price
            back_month_price: TX back-month settlement price

        Returns:
            DataFrame with 7 feature columns, indexed by date.
        """
        features = pd.DataFrame(index=taiex_prices.index)

        # 1. Realized Volatility (annualized, 60-day window)
        log_returns = np.log(taiex_prices / taiex_prices.shift(1))
        features['realized_vol'] = log_returns.rolling(
            self.config.volatility_lookback
        ).std() * np.sqrt(252)

        # 2. Volatility Momentum (vol change over 21 days)
        vol_shifted = features['realized_vol'].shift(self.config.vol_momentum_lookback)
        features['vol_momentum'] = (
            (features['realized_vol'] / vol_shifted.replace(0, np.nan)) - 1
        ).clip(-0.5, 0.5)

        # 3. Trend Score: (price - 200-SMA) / 200-SMA * 100
        sma_200 = taiex_prices.rolling(self.config.sma_lookback).mean()
        features['trend_score'] = (
            (taiex_prices - sma_200) / sma_200
        ) * 100.0

        # 4. Momentum (21-day return)
        features['momentum_21d'] = taiex_prices.pct_change(
            self.config.momentum_lookback
        ).clip(-0.2, 0.2)

        # 5. Institutional Flow (normalized to rolling z-score)
        if institutional_net is not None:
            inst_mean = institutional_net.rolling(60).mean()
            inst_std = institutional_net.rolling(60).std().replace(0, np.nan)
            features['institutional_flow'] = (
                (institutional_net - inst_mean) / inst_std
            ).clip(-3, 3)
        else:
            features['institutional_flow'] = 0.0

        # 6. Put/Call Ratio (contrarian: high P/C = bearish crowd = potential bullish)
        if put_call_ratio is not None:
            pc_mean = put_call_ratio.rolling(60).mean()
            pc_std = put_call_ratio.rolling(60).std().replace(0, np.nan)
            features['put_call_ratio'] = (
                (put_call_ratio - pc_mean) / pc_std
            ).clip(-3, 3)
        else:
            features['put_call_ratio'] = 0.0

        # 7. Term Structure Slope (back month - front month, normalized)
        if front_month_price is not None and back_month_price is not None:
            spread = back_month_price - front_month_price
            spread_mean = spread.rolling(60).mean()
            spread_std = spread.rolling(60).std().replace(0, np.nan)
            features['term_structure_slope'] = (
                (spread - spread_mean) / spread_std
            ).clip(-3, 3)
        else:
            features['term_structure_slope'] = 0.0

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        logger.info(f"Computed {len(self.FEATURE_NAMES)} features, "
                    f"{len(features)} observations")
        return features

    def compute_sma_200(self, taiex_prices: pd.Series) -> pd.Series:
        """Compute 200-day SMA for trend regime signal."""
        return taiex_prices.rolling(self.config.sma_lookback).mean()

    def compute_above_sma(self, taiex_prices: pd.Series) -> pd.Series:
        """Boolean: is TAIEX above its 200-day SMA?"""
        sma = self.compute_sma_200(taiex_prices)
        return taiex_prices > sma

    def compute_returns(self, prices: pd.Series) -> pd.Series:
        """Daily log returns."""
        return np.log(prices / prices.shift(1)).dropna()
