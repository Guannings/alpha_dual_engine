"""
Backtesting engine for TAIFEX competition strategy.
Adapted from BacktestEngine in alpha_engine.py:1082-1540.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import StrategyConfig
from ..data.fetcher import TAIFEXFetcher
from ..data.features import FeatureEngineer
from ..data.contracts import CONTRACT_SPECS, should_roll
from ..strategy.regime import TAIEXRegimeClassifier
from ..strategy.signals import SignalGenerator, Signal, Direction
from ..strategy.optimizer import PositionOptimizer
from ..strategy.risk import RiskManager
from ..execution.portfolio import PortfolioTracker

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Day-by-day simulation of the competition strategy.

    Steps:
        1. Load historical TAIEX data
        2. Train regime classifier via walk-forward
        3. For each trading day:
            a. Determine regime
            b. Generate signals
            c. Optimize positions
            d. Apply risk checks
            e. Simulate fills with transaction costs
            f. Track P&L and category coverage
        4. Output performance metrics
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.fetcher = TAIFEXFetcher()
        self.feature_engineer = FeatureEngineer(self.config)

    def run(
        self,
        start_date: str = '2024-01-01',
        end_date: str = '2025-12-31',
        train_start: str = '2015-01-01',
    ) -> Optional[pd.DataFrame]:
        """Run full backtest.

        Args:
            start_date: Backtest start date (OOS period)
            end_date: Backtest end date
            train_start: Start of training data

        Returns:
            DataFrame with daily portfolio snapshots, or None if data missing.
        """
        logger.info(f"Backtest: {start_date} to {end_date}")

        # Load data
        taiex_data = self.fetcher.load_historical_futures('TX', start_date=train_start)
        if taiex_data.empty:
            logger.error(
                "No historical data. Download TX daily data and save to "
                "data_cache/TX_daily.csv with columns: Date, Open, High, Low, Close, Volume"
            )
            return None

        if 'Close' not in taiex_data.columns:
            logger.error("Data must have 'Close' column")
            return None

        prices = taiex_data['Close'].astype(float)

        # Compute features
        features = self.feature_engineer.compute_features(prices)
        returns = self.feature_engineer.compute_returns(prices)
        above_sma = self.feature_engineer.compute_above_sma(prices)

        # Train classifier
        classifier = TAIEXRegimeClassifier(self.config)
        ml_probs = classifier.walk_forward_train(features, returns)

        logger.info(f"Classifier trained. Stability: {classifier.model_stability}")

        # Backtest period
        bt_start = pd.Timestamp(start_date)
        bt_end = pd.Timestamp(end_date)

        bt_dates = prices.index[
            (prices.index >= bt_start) & (prices.index <= bt_end)
        ]
        if len(bt_dates) == 0:
            logger.error(f"No data in backtest period {start_date} to {end_date}")
            return None

        # Components
        signal_gen = SignalGenerator(self.config)
        optimizer = PositionOptimizer(self.config)
        risk_mgr = RiskManager(self.config)
        portfolio = PortfolioTracker(self.config)

        rebalance_interval = self.config.backtest_lookback_days // 12  # ~monthly
        days_since_rebalance = rebalance_interval  # Trigger on first day

        # Transaction costs
        fee_per_contract = self.config.futures_fee_per_contract
        tax_rate = self.config.futures_tax_rate

        daily_records = []

        for i, dt in enumerate(bt_dates):
            # Get current price
            taiex_price = float(prices.loc[dt])

            # Daily P&L from positions (mark-to-market)
            portfolio.update_prices({'TX': taiex_price, 'MTX': taiex_price})

            # Check if rebalance
            days_since_rebalance += 1
            if days_since_rebalance >= rebalance_interval:
                # Get features for this date
                if dt not in features.index:
                    continue

                feat_row = features.loc[dt]
                ml_prob = ml_probs.loc[dt] if dt in ml_probs.index else 0.5
                if np.isnan(ml_prob):
                    ml_prob = 0.5

                taiex_above = bool(above_sma.loc[dt]) if dt in above_sma.index else True
                vol = float(feat_row.get('realized_vol', 0.15))

                # Regime
                regime = classifier.get_regime(ml_prob, taiex_above, vol)

                # Signals
                signals = signal_gen.generate(
                    regime=regime,
                    ml_prob=ml_prob,
                    taiex_price=taiex_price,
                    momentum_21d=float(feat_row.get('momentum_21d', 0)),
                    realized_vol=vol,
                    institutional_flow=float(feat_row.get('institutional_flow', 0)),
                    put_call_ratio=float(feat_row.get('put_call_ratio', 0)),
                    capital=portfolio.equity,
                )

                # Optimize
                current_pos = portfolio.get_current_positions_dict()
                target_pos = optimizer.optimize(
                    signals=signals,
                    current_positions=current_pos,
                    capital=portfolio.equity,
                    regime=regime,
                    margin_used=portfolio.total_margin_used,
                )

                # Risk check
                risk_report = risk_mgr.check(
                    target_positions=target_pos,
                    current_prices={'TX': taiex_price, 'MTX': taiex_price},
                    capital=portfolio.equity,
                    equity=portfolio.equity,
                    margin_used=portfolio.total_margin_used,
                )

                if not risk_report.approved:
                    target_pos = risk_mgr.scale_down_positions(
                        target_pos, portfolio.equity,
                    )

                # Execute (simulate fills at current price + cost)
                date_str = dt.strftime('%Y-%m-%d')
                for sym, target_qty in target_pos.items():
                    current_qty = current_pos.get(sym, 0)
                    delta = target_qty - current_qty
                    if delta == 0:
                        continue

                    side = 'B' if delta > 0 else 'S'
                    qty = abs(delta)

                    # Transaction cost
                    spec = CONTRACT_SPECS.get(sym)
                    if spec:
                        notional = spec.multiplier * taiex_price * qty
                        fee = fee_per_contract * qty + notional * tax_rate
                    else:
                        fee = fee_per_contract * qty

                    portfolio.record_fill(
                        symbol=sym,
                        side=side,
                        quantity=qty,
                        fill_price=taiex_price,
                        fee=fee,
                        trade_date=date_str,
                    )

                days_since_rebalance = 0

            # Record daily snapshot
            daily_records.append({
                'date': dt,
                'taiex': taiex_price,
                'equity': portfolio.equity,
                'realized_pnl': portfolio.realized_pnl,
                'unrealized_pnl': sum(
                    p.unrealized_pnl for p in portfolio.positions.values()
                ),
                'margin_util': portfolio.margin_utilization,
                'risk_indicator': portfolio.risk_indicator,
                'num_positions': len(portfolio.positions),
                'categories': portfolio.num_categories_traded,
                'trading_days': portfolio.num_trading_days,
                'regime': regime if days_since_rebalance == 0 else '',
            })

        results = pd.DataFrame(daily_records).set_index('date')

        logger.info(
            f"Backtest complete. "
            f"Final equity: NT${portfolio.equity:,.0f}, "
            f"Categories: {portfolio.num_categories_traded}, "
            f"Trading days: {portfolio.num_trading_days}"
        )

        return results
