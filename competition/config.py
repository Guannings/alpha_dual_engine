"""
Central configuration for TAIFEX Competition Trading Bot.
Adapted from alpha_engine.py StrategyConfig pattern.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import time


@dataclass
class StrategyConfig:
    """Configuration for TAIFEX competition strategy."""

    # --- Capital & Margin ---
    initial_capital: float = 2_000_000.0       # NT$2M virtual capital
    margin_safety_buffer: float = 0.30         # Keep risk indicator above 30% (25% = forced liquidation)
    max_margin_utilization: float = 0.75       # Never use more than 75% of available margin

    # --- Trading Hours (Taiwan local time) ---
    market_open: time = field(default_factory=lambda: time(8, 45))
    market_close: time = field(default_factory=lambda: time(13, 45))

    # --- Rebalancing ---
    rebalance_interval_minutes: int = 30       # Check signals every 30 min during session
    min_rebalance_threshold: float = 0.05      # Skip rebalance if turnover < 5%

    # --- Risk Limits ---
    max_daily_loss_pct: float = 0.03           # Circuit breaker: stop trading if down 3% intraday
    max_position_notional_pct: float = 0.25    # Max 25% of capital in any single contract
    max_portfolio_delta: float = 500.0         # Max absolute portfolio delta (index points)
    max_portfolio_gamma: float = 100.0         # Max absolute portfolio gamma
    max_portfolio_vega: float = 200_000.0      # Max absolute portfolio vega (NT$)

    # --- Regime Classifier ---
    ml_threshold: float = 0.55                 # Confidence threshold for regime classification
    xgb_n_estimators: int = 50
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.05
    walk_forward_initial_years: int = 5
    walk_forward_step_months: int = 6

    # --- Optimizer (SLSQP) ---
    target_volatility: float = 0.15            # 15% annualized vol target
    risk_aversion: float = 0.05
    entropy_lambda: float = 0.03               # Diversification incentive
    turnover_penalty: float = 0.5
    volatility_penalty: float = 50.0

    # --- Feature Engineering ---
    sma_lookback: int = 200
    momentum_lookback: int = 21
    volatility_lookback: int = 60
    vol_momentum_lookback: int = 21

    # --- Transaction Costs (TAIFEX fee schedule, per contract) ---
    futures_fee_per_contract: float = 20.0     # NT$ per futures contract (approx)
    options_fee_per_contract: float = 20.0     # NT$ per options contract (approx)
    futures_tax_rate: float = 0.00002          # Futures transaction tax
    options_tax_rate: float = 0.001            # Options transaction tax (on premium)
    slippage_ticks: int = 1                    # Assume 1 tick slippage per side

    # --- Backtest ---
    backtest_lookback_days: int = 252
    risk_free_rate: float = 0.015              # Taiwan 1-year rate (~1.5%)

    # --- Competition Requirements ---
    min_product_categories: int = 6            # Must trade 6+ of 10 categories
    min_trading_days: int = 28                 # Must trade 28+ days

    # --- Product Universe ---
    product_categories: Dict[str, List[str]] = field(default_factory=lambda: {
        'index_futures': ['TX', 'MTX', 'TE', 'TF'],       # TAIEX, Mini-TAIEX, Electronics, Finance
        'index_options': ['TXO'],                           # TAIEX options
        'equity_futures': ['STOCK_F'],                      # Individual stock futures (TSMC, MediaTek, etc.)
        'equity_options': ['STOCK_O'],                      # Individual stock options
        'gold_futures': ['GDF'],                            # Gold futures
        'fx_futures': ['UDF', 'RHF'],                      # USD/TWD, RMB/TWD
        'interest_rate': ['GBF'],                           # Government bond futures
        'etf_futures': ['ETF_F'],                           # ETF futures
        'commodity_futures': ['BTF'],                       # Brent oil futures
        'midcap_futures': ['XIF'],                          # Mid-cap 100 futures
    })

    @property
    def active_categories(self) -> List[str]:
        """Categories we plan to actively trade (6+ required)."""
        return [
            'index_futures',
            'index_options',
            'equity_futures',
            'gold_futures',
            'fx_futures',
            'midcap_futures',
        ]
