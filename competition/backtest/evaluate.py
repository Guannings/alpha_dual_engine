"""
Performance evaluation metrics for TAIFEX backtest results.
Reuses Sharpe/risk metric patterns from alpha_engine.py:1341-1377.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import StrategyConfig

logger = logging.getLogger(__name__)


def compute_sharpe_ratio(
    equity_series: pd.Series,
    risk_free_rate: float = 0.015,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio from equity curve.

    Sharpe = (mean(daily_returns) - rf_daily) / std(daily_returns) * sqrt(252)
    """
    daily_returns = equity_series.pct_change().dropna()
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        return 0.0

    rf_daily = risk_free_rate / periods_per_year
    excess_returns = daily_returns - rf_daily
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    return float(sharpe)


def compute_max_drawdown(equity_series: pd.Series) -> float:
    """Maximum drawdown as a negative percentage."""
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    return float(drawdown.min())


def compute_calmar_ratio(
    equity_series: pd.Series,
    risk_free_rate: float = 0.015,
) -> float:
    """Calmar ratio = annualized return / max drawdown."""
    if len(equity_series) < 2:
        return 0.0

    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    days = (equity_series.index[-1] - equity_series.index[0]).days
    if days <= 0:
        return 0.0
    ann_return = (1 + total_return) ** (365 / days) - 1

    max_dd = abs(compute_max_drawdown(equity_series))
    if max_dd == 0:
        return 0.0

    return float((ann_return - risk_free_rate) / max_dd)


def compute_sortino_ratio(
    equity_series: pd.Series,
    risk_free_rate: float = 0.015,
    periods_per_year: int = 252,
) -> float:
    """Sortino ratio (uses downside deviation only)."""
    daily_returns = equity_series.pct_change().dropna()
    if len(daily_returns) < 2:
        return 0.0

    rf_daily = risk_free_rate / periods_per_year
    excess = daily_returns - rf_daily
    downside = excess[excess < 0]

    if len(downside) == 0 or downside.std() == 0:
        return 0.0

    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def compute_win_rate(equity_series: pd.Series) -> float:
    """Percentage of positive-return days."""
    daily_returns = equity_series.pct_change().dropna()
    if len(daily_returns) == 0:
        return 0.0
    return float((daily_returns > 0).mean())


def evaluate_backtest(
    results: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
) -> Dict:
    """Full evaluation of backtest results.

    Args:
        results: DataFrame from BacktestEngine.run() with 'equity' column
        config: Strategy configuration

    Returns:
        Dict of performance metrics
    """
    config = config or StrategyConfig()
    equity = results['equity']

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    days = (equity.index[-1] - equity.index[0]).days
    ann_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

    sharpe = compute_sharpe_ratio(equity, config.risk_free_rate)
    max_dd = compute_max_drawdown(equity)
    calmar = compute_calmar_ratio(equity, config.risk_free_rate)
    sortino = compute_sortino_ratio(equity, config.risk_free_rate)
    win_rate = compute_win_rate(equity)

    # Competition-specific metrics
    categories_traded = int(results['categories'].max()) if 'categories' in results else 0
    trading_days = int(results['trading_days'].max()) if 'trading_days' in results else 0
    meets_category_req = categories_traded >= config.min_product_categories
    meets_day_req = trading_days >= config.min_trading_days

    # Margin safety
    if 'risk_indicator' in results:
        min_risk_indicator = results['risk_indicator'].replace(
            [np.inf, -np.inf], np.nan
        ).dropna().min()
        margin_safe = min_risk_indicator > 0.25 if not np.isnan(min_risk_indicator) else True
    else:
        min_risk_indicator = None
        margin_safe = True

    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'sortino_ratio': sortino,
        'win_rate': win_rate,
        'total_days': len(equity),
        'start_equity': equity.iloc[0],
        'end_equity': equity.iloc[-1],
        'profit_nt': equity.iloc[-1] - equity.iloc[0],
        # Competition requirements
        'categories_traded': categories_traded,
        'trading_days': trading_days,
        'meets_category_req': meets_category_req,
        'meets_day_req': meets_day_req,
        'min_risk_indicator': min_risk_indicator,
        'margin_safe': margin_safe,
    }


def print_evaluation_report(results: pd.DataFrame, config: Optional[StrategyConfig] = None) -> None:
    """Print a formatted evaluation report."""
    metrics = evaluate_backtest(results, config)

    print("\n" + "=" * 60)
    print("  TAIFEX COMPETITION BACKTEST REPORT")
    print("=" * 60)

    print(f"\n  Period:  {results.index[0].strftime('%Y-%m-%d')} to "
          f"{results.index[-1].strftime('%Y-%m-%d')} ({metrics['total_days']} days)")

    print(f"\n  --- Performance ---")
    print(f"  Total Return:       {metrics['total_return']:>10.2%}")
    print(f"  Annualized Return:  {metrics['annualized_return']:>10.2%}")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:      {metrics['sortino_ratio']:>10.2f}")
    print(f"  Calmar Ratio:       {metrics['calmar_ratio']:>10.2f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown']:>10.2%}")
    print(f"  Win Rate:           {metrics['win_rate']:>10.2%}")

    print(f"\n  --- P&L ---")
    print(f"  Start Equity:       NT${metrics['start_equity']:>12,.0f}")
    print(f"  End Equity:         NT${metrics['end_equity']:>12,.0f}")
    print(f"  Profit:             NT${metrics['profit_nt']:>12,.0f}")

    print(f"\n  --- Competition Requirements ---")
    cat_status = 'PASS' if metrics['meets_category_req'] else 'FAIL'
    day_status = 'PASS' if metrics['meets_day_req'] else 'FAIL'
    margin_status = 'PASS' if metrics['margin_safe'] else 'FAIL'

    print(f"  Categories Traded:  {metrics['categories_traded']:>3d}/6   [{cat_status}]")
    print(f"  Trading Days:       {metrics['trading_days']:>3d}/28  [{day_status}]")
    print(f"  Margin Safety:      {'OK' if metrics['margin_safe'] else 'BREACH'}       [{margin_status}]")
    if metrics['min_risk_indicator'] is not None:
        print(f"  Min Risk Indicator: {metrics['min_risk_indicator']:>10.1%}")

    print("\n" + "=" * 60)

    # Summary verdict
    all_pass = (metrics['meets_category_req'] and metrics['meets_day_req']
                and metrics['margin_safe'] and metrics['sharpe_ratio'] > 0)
    if all_pass:
        print("  VERDICT: Strategy meets all competition requirements")
    else:
        print("  VERDICT: Strategy has issues — review failures above")
    print("=" * 60 + "\n")
