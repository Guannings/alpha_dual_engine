"""
Main bot loop for TAIFEX Competition Trading Bot.

Workflow:
    1. Connect to broker (TTB via pytaifex or simulated)
    2. Fetch latest market data
    3. Compute features → regime classification
    4. Generate signals per product category
    5. Optimize position sizes (SLSQP)
    6. Risk check (margin, daily loss, Greeks)
    7. Execute orders
    8. Sleep until next rebalance interval
    9. Repeat until market close
"""
import argparse
import logging
import sys
import time
from datetime import datetime, date, timedelta

from .config import StrategyConfig
from .data.fetcher import TAIFEXFetcher
from .data.features import FeatureEngineer
from .data.contracts import get_front_month_expiry, get_contract_month_code
from .strategy.regime import TAIEXRegimeClassifier
from .strategy.signals import SignalGenerator
from .strategy.optimizer import PositionOptimizer
from .strategy.risk import RiskManager, GreeksSnapshot
from .execution.broker import PytaifexBroker, SimulatedBroker
from .execution.order_manager import OrderManager
from .execution.portfolio import PortfolioTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('competition_bot.log'),
    ],
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot for TAIFEX competition."""

    def __init__(self, config: StrategyConfig, use_simulator: bool = True):
        self.config = config

        # Components
        self.fetcher = TAIFEXFetcher()
        self.feature_engineer = FeatureEngineer(config)
        self.classifier = TAIEXRegimeClassifier(config)
        self.signal_generator = SignalGenerator(config)
        self.optimizer = PositionOptimizer(config)
        self.risk_manager = RiskManager(config)
        self.portfolio = PortfolioTracker(config)

        # Broker
        if use_simulator:
            self.broker = SimulatedBroker()
        else:
            self.broker = PytaifexBroker()

        self.order_manager = OrderManager(self.broker)

        # State
        self.is_trained = False
        self.latest_regime = 'QUIET'
        self.latest_ml_prob = 0.5

    def train(self) -> None:
        """Train the regime classifier on historical data."""
        logger.info("Training regime classifier on historical data...")

        # Load historical TAIEX data
        taiex_data = self.fetcher.load_historical_futures('TX', start_date='2015-01-01')
        if taiex_data.empty:
            logger.error(
                "No historical data available for training. "
                "Please download TX daily data and save to data_cache/TX_daily.csv"
            )
            return

        # Need a 'Close' column
        if 'Close' not in taiex_data.columns:
            logger.error("Historical data must have a 'Close' column")
            return

        prices = taiex_data['Close'].astype(float)

        # Compute features
        features = self.feature_engineer.compute_features(prices)

        # Daily returns
        returns = self.feature_engineer.compute_returns(prices)

        # Walk-forward training
        ml_probs = self.classifier.walk_forward_train(features, returns)

        self.is_trained = True
        logger.info(
            f"Training complete. Model stability: {self.classifier.model_stability}. "
            f"Test scores: {[f'{s:.3f}' for s in self.classifier.test_scores]}"
        )

    def run_live(self) -> None:
        """Run the live trading loop during market hours."""
        if not self.is_trained:
            logger.error("Classifier not trained. Call train() first.")
            return

        # Connect to broker
        if not self.broker.connect():
            logger.error("Could not connect to broker")
            return

        self.risk_manager.start_of_day(self.portfolio.equity)

        logger.info(
            f"Bot started. Capital: NT${self.portfolio.equity:,.0f}. "
            f"Rebalance every {self.config.rebalance_interval_minutes} min."
        )

        try:
            while self._is_market_open():
                self._trading_cycle()
                sleep_seconds = self.config.rebalance_interval_minutes * 60
                logger.info(f"Sleeping {self.config.rebalance_interval_minutes} min until next cycle...")
                time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        finally:
            # End-of-day snapshot
            today = datetime.now().strftime('%Y-%m-%d')
            self.portfolio.record_daily_snapshot(today)

            summary = self.portfolio.get_summary()
            logger.info(f"End of day summary: {summary}")
            logger.info(f"Order summary: {self.order_manager.get_fill_summary()}")

            self.broker.disconnect()

    def _trading_cycle(self) -> None:
        """Single cycle: data → features → regime → signals → optimize → risk → execute."""
        logger.info("--- Trading cycle start ---")

        # 1. Fetch latest data
        futures_data = self.fetcher.fetch_futures_daily('TX')
        if futures_data.empty:
            logger.warning("No market data available, skipping cycle")
            return

        # Extract latest TAIEX price and features
        try:
            taiex_price = float(futures_data.iloc[-1].get('SettlementPrice',
                                futures_data.iloc[-1].get('Close', 0)))
        except (IndexError, ValueError):
            logger.warning("Could not parse TAIEX price")
            return

        if taiex_price <= 0:
            logger.warning(f"Invalid TAIEX price: {taiex_price}")
            return

        # 2. Get regime (use latest trained model)
        # In live mode we use the most recent features from real-time data
        account = self.broker.get_account_info()
        equity = account.get('equity', self.portfolio.equity)

        # Compute real-time feature approximations
        sma_200_above = taiex_price > 18000  # Simplified; real version uses rolling SMA
        regime = self.classifier.get_regime(
            self.latest_ml_prob, sma_200_above, 0.15
        )
        self.latest_regime = regime

        # 3. Generate signals
        signals = self.signal_generator.generate(
            regime=regime,
            ml_prob=self.latest_ml_prob,
            taiex_price=taiex_price,
            momentum_21d=0.0,      # Would be computed from recent data
            realized_vol=0.15,     # Would be computed from recent data
            institutional_flow=0.0,
            put_call_ratio=0.0,
            capital=equity,
        )

        # 4. Optimize positions
        current_positions = self.portfolio.get_current_positions_dict()
        margin_used = self.portfolio.total_margin_used

        target_positions = self.optimizer.optimize(
            signals=signals,
            current_positions=current_positions,
            capital=equity,
            regime=regime,
            margin_used=margin_used,
        )

        # 5. Risk check
        current_prices = {}
        for sym in target_positions:
            q = self.broker.get_quote(sym)
            if q:
                current_prices[sym] = q.last

        risk_report = self.risk_manager.check(
            target_positions=target_positions,
            current_prices=current_prices,
            capital=equity,
            equity=equity,
            margin_used=margin_used,
        )

        if not risk_report.approved:
            logger.warning(f"Risk check failed: {risk_report.violations}")
            # Try scaling down
            target_positions = self.risk_manager.scale_down_positions(
                target_positions, equity,
            )
            # Re-check
            risk_report = self.risk_manager.check(
                target_positions=target_positions,
                current_prices=current_prices,
                capital=equity,
                equity=equity,
                margin_used=margin_used,
            )
            if not risk_report.approved:
                logger.error("Risk check failed even after scaling — skipping cycle")
                return

        # 6. Execute orders
        today = date.today()
        responses = self.order_manager.reconcile_and_execute(
            target_positions=target_positions,
            current_positions=current_positions,
            as_of_date=today,
        )

        # Record fills in portfolio
        for resp in responses:
            if resp.status.value == 'filled':
                # Find the matching order details
                for oh in reversed(self.order_manager.order_history):
                    if oh['order_id'] == resp.order_id:
                        self.portfolio.record_fill(
                            symbol=oh['symbol'],
                            side=oh['side'],
                            quantity=resp.filled_quantity,
                            fill_price=resp.filled_price,
                            trade_date=today.strftime('%Y-%m-%d'),
                        )
                        break

        # Update portfolio prices
        self.portfolio.update_prices(current_prices)

        logger.info(
            f"Cycle complete: regime={regime}, "
            f"equity=NT${self.portfolio.equity:,.0f}, "
            f"margin_util={self.portfolio.margin_utilization:.1%}, "
            f"positions={len(self.portfolio.positions)}"
        )

    def _is_market_open(self) -> bool:
        """Check if Taiwan market is currently open."""
        now = datetime.now()
        # Simple check — production version should handle timezone (Asia/Taipei)
        current_time = now.time()
        if current_time < self.config.market_open or current_time > self.config.market_close:
            return False
        # Skip weekends
        if now.weekday() >= 5:
            return False
        return True


def main():
    parser = argparse.ArgumentParser(description='TAIFEX Competition Trading Bot')
    parser.add_argument('--mode', choices=['live', 'train', 'backtest'],
                        default='train', help='Run mode')
    parser.add_argument('--simulate', action='store_true', default=True,
                        help='Use simulated broker')
    parser.add_argument('--ttb-host', default='127.0.0.1',
                        help='TTB ZMQ host')
    parser.add_argument('--ttb-port', type=int, default=5555,
                        help='TTB ZMQ port')
    args = parser.parse_args()

    config = StrategyConfig()

    if args.mode == 'train':
        bot = TradingBot(config, use_simulator=True)
        bot.train()

    elif args.mode == 'live':
        bot = TradingBot(config, use_simulator=args.simulate)
        bot.train()
        bot.run_live()

    elif args.mode == 'backtest':
        from .backtest.engine import BacktestEngine
        engine = BacktestEngine(config)
        results = engine.run()
        if results is not None:
            from .backtest.evaluate import print_evaluation_report
            print_evaluation_report(results)


if __name__ == '__main__':
    main()
