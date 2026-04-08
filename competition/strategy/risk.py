"""
Real-time risk management for TAIFEX competition.
Monitors margin, daily P&L limits, Greeks, and position concentration.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import StrategyConfig
from ..data.contracts import CONTRACT_SPECS, get_margin_requirement, get_notional_value

logger = logging.getLogger(__name__)


@dataclass
class GreeksSnapshot:
    """Aggregate portfolio Greeks."""
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0


@dataclass
class RiskReport:
    """Risk assessment result."""
    approved: bool
    margin_utilization: float
    risk_indicator: float
    daily_pnl_pct: float
    total_notional: float
    greeks: GreeksSnapshot
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class RiskManager:
    """Portfolio risk manager with circuit breakers.

    Checks:
        1. Margin utilization < 75% (configurable)
        2. Risk indicator > 30% (25% = forced liquidation by exchange)
        3. Daily loss < 3% of capital (circuit breaker)
        4. Portfolio Greeks within limits (delta, gamma, vega)
        5. No single position > 25% of capital notional
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.daily_start_capital: float = 0.0
        self.circuit_breaker_triggered: bool = False

    def start_of_day(self, capital: float) -> None:
        """Record starting capital for daily loss tracking."""
        self.daily_start_capital = capital
        self.circuit_breaker_triggered = False
        logger.info(f"Risk manager: SOD capital = NT${capital:,.0f}")

    def check(
        self,
        target_positions: Dict[str, int],
        current_prices: Dict[str, float],
        capital: float,
        equity: float,
        margin_used: float,
        greeks: Optional[GreeksSnapshot] = None,
    ) -> RiskReport:
        """Run all risk checks on proposed target positions.

        Args:
            target_positions: {symbol: num_contracts} (signed)
            current_prices: {symbol: current_price}
            capital: Total account equity (NT$)
            equity: Net liquidation value (NT$)
            margin_used: Current margin requirement (NT$)
            greeks: Optional aggregate Greeks

        Returns:
            RiskReport with approval status and details.
        """
        violations = []
        warnings = []
        greeks = greeks or GreeksSnapshot()

        # 1. Calculate total margin for target positions
        total_margin = 0.0
        for sym, qty in target_positions.items():
            if qty == 0:
                continue
            spec = CONTRACT_SPECS.get(sym)
            if spec:
                total_margin += get_margin_requirement(sym, qty)

        margin_util = total_margin / capital if capital > 0 else 1.0
        # Risk indicator = equity / margin (TAIFEX definition)
        risk_indicator = equity / total_margin if total_margin > 0 else float('inf')

        if margin_util > self.config.max_margin_utilization:
            violations.append(
                f"Margin utilization {margin_util:.1%} exceeds "
                f"limit {self.config.max_margin_utilization:.1%}"
            )

        if risk_indicator < self.config.margin_safety_buffer:
            violations.append(
                f"Risk indicator {risk_indicator:.1%} below "
                f"safety buffer {self.config.margin_safety_buffer:.1%}"
            )

        if risk_indicator < 0.40:
            warnings.append(
                f"Risk indicator {risk_indicator:.1%} approaching danger zone"
            )

        # 2. Daily loss circuit breaker
        daily_pnl_pct = 0.0
        if self.daily_start_capital > 0:
            daily_pnl_pct = (equity - self.daily_start_capital) / self.daily_start_capital
            if daily_pnl_pct < -self.config.max_daily_loss_pct:
                violations.append(
                    f"Daily loss {daily_pnl_pct:.2%} exceeds circuit breaker "
                    f"{-self.config.max_daily_loss_pct:.2%}"
                )
                self.circuit_breaker_triggered = True

        # 3. Position concentration
        total_notional = 0.0
        for sym, qty in target_positions.items():
            if qty == 0:
                continue
            price = current_prices.get(sym, 0)
            spec = CONTRACT_SPECS.get(sym)
            if spec and price > 0:
                notional = get_notional_value(sym, price, qty)
                total_notional += notional
                concentration = notional / capital if capital > 0 else 1.0
                if concentration > self.config.max_position_notional_pct:
                    violations.append(
                        f"{sym}: notional concentration {concentration:.1%} "
                        f"exceeds {self.config.max_position_notional_pct:.1%}"
                    )

        # 4. Greeks limits
        if abs(greeks.delta) > self.config.max_portfolio_delta:
            violations.append(
                f"Portfolio delta {greeks.delta:.0f} exceeds "
                f"limit {self.config.max_portfolio_delta:.0f}"
            )
        if abs(greeks.gamma) > self.config.max_portfolio_gamma:
            warnings.append(
                f"Portfolio gamma {greeks.gamma:.0f} exceeds "
                f"limit {self.config.max_portfolio_gamma:.0f}"
            )
        if abs(greeks.vega) > self.config.max_portfolio_vega:
            warnings.append(
                f"Portfolio vega NT${greeks.vega:,.0f} exceeds "
                f"limit NT${self.config.max_portfolio_vega:,.0f}"
            )

        approved = len(violations) == 0 and not self.circuit_breaker_triggered

        report = RiskReport(
            approved=approved,
            margin_utilization=margin_util,
            risk_indicator=risk_indicator,
            daily_pnl_pct=daily_pnl_pct,
            total_notional=total_notional,
            greeks=greeks,
            violations=violations,
            warnings=warnings,
        )

        if not approved:
            logger.warning(f"RISK CHECK FAILED: {violations}")
        for w in warnings:
            logger.warning(f"Risk warning: {w}")

        return report

    def scale_down_positions(
        self,
        target_positions: Dict[str, int],
        capital: float,
        max_margin_pct: Optional[float] = None,
    ) -> Dict[str, int]:
        """Scale down all positions proportionally to fit within margin limit.

        Used when risk check fails due to margin.
        """
        max_margin_pct = max_margin_pct or self.config.max_margin_utilization
        max_margin = capital * max_margin_pct

        total_margin = 0.0
        for sym, qty in target_positions.items():
            if qty != 0:
                total_margin += get_margin_requirement(sym, qty)

        if total_margin <= max_margin:
            return target_positions

        scale = max_margin / total_margin if total_margin > 0 else 0
        logger.warning(f"Scaling positions by {scale:.2f} to fit margin limit")

        scaled = {}
        for sym, qty in target_positions.items():
            new_qty = int(qty * scale)
            scaled[sym] = new_qty

        return scaled
