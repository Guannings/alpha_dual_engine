"""
Live position tracker with P&L and margin calculation.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from ..config import StrategyConfig
from ..data.contracts import CONTRACT_SPECS, get_margin_requirement, get_notional_value

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A single open position."""
    symbol: str
    quantity: int            # Signed: positive = long, negative = short
    avg_entry_price: float
    current_price: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in NT$."""
        spec = CONTRACT_SPECS.get(self.symbol)
        if spec is None:
            return 0.0
        return (self.current_price - self.avg_entry_price) * self.quantity * spec.multiplier

    @property
    def notional(self) -> float:
        spec = CONTRACT_SPECS.get(self.symbol)
        if spec is None:
            return 0.0
        return spec.multiplier * self.current_price * abs(self.quantity)

    @property
    def margin_required(self) -> float:
        return get_margin_requirement(self.symbol, self.quantity)


class PortfolioTracker:
    """Real-time portfolio state tracking.

    Tracks:
        - Open positions with entry prices
        - Unrealized and realized P&L
        - Margin usage
        - Product category coverage (for competition requirement)
        - Trading day count
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.positions: Dict[str, Position] = {}
        self.realized_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.pnl_history: List[Dict] = []
        self.categories_traded: Set[str] = set()
        self.trading_days: Set[str] = set()  # Set of date strings

    @property
    def equity(self) -> float:
        """Net liquidation value = initial capital + realized + unrealized P&L - fees."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.config.initial_capital + self.realized_pnl + unrealized - self.total_fees

    @property
    def total_margin_used(self) -> float:
        """Total margin requirement for all open positions."""
        return sum(p.margin_required for p in self.positions.values())

    @property
    def margin_utilization(self) -> float:
        eq = self.equity
        if eq <= 0:
            return 1.0
        return self.total_margin_used / eq

    @property
    def risk_indicator(self) -> float:
        """TAIFEX risk indicator = equity / margin."""
        margin = self.total_margin_used
        if margin <= 0:
            return float('inf')
        return self.equity / margin

    @property
    def num_categories_traded(self) -> int:
        return len(self.categories_traded)

    @property
    def num_trading_days(self) -> int:
        return len(self.trading_days)

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all positions."""
        for sym, price in prices.items():
            if sym in self.positions:
                self.positions[sym].current_price = price

    def record_fill(
        self,
        symbol: str,
        side: str,
        quantity: int,
        fill_price: float,
        fee: float = 0.0,
        trade_date: Optional[str] = None,
    ) -> None:
        """Record a fill and update position.

        Args:
            symbol: Contract symbol
            side: 'B' (buy) or 'S' (sell)
            quantity: Unsigned quantity
            fill_price: Execution price
            fee: Transaction fee + tax
            trade_date: Date string for tracking trading days
        """
        signed_qty = quantity if side == 'B' else -quantity
        self.total_fees += fee

        # Track category and trading day
        spec = CONTRACT_SPECS.get(symbol)
        if spec:
            self.categories_traded.add(spec.category)
        if trade_date:
            self.trading_days.add(trade_date)

        if symbol in self.positions:
            pos = self.positions[symbol]
            old_qty = pos.quantity
            new_qty = old_qty + signed_qty

            if new_qty == 0:
                # Position closed — realize P&L
                pnl = (fill_price - pos.avg_entry_price) * old_qty
                if spec:
                    pnl *= spec.multiplier
                self.realized_pnl += pnl
                del self.positions[symbol]
                logger.info(
                    f"Closed {symbol}: realized P&L = NT${pnl:,.0f}"
                )
            elif (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
                # Adding to position — update average entry
                total_cost = pos.avg_entry_price * abs(old_qty) + fill_price * quantity
                pos.avg_entry_price = total_cost / abs(new_qty)
                pos.quantity = new_qty
            else:
                # Partial close + reversal
                close_qty = abs(old_qty)
                pnl = (fill_price - pos.avg_entry_price) * old_qty
                if spec:
                    pnl *= spec.multiplier
                self.realized_pnl += pnl

                remaining = abs(new_qty)
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=new_qty,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                )
                logger.info(
                    f"Reversed {symbol}: closed P&L = NT${pnl:,.0f}, "
                    f"new qty = {new_qty}"
                )
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=signed_qty,
                avg_entry_price=fill_price,
                current_price=fill_price,
            )
            logger.info(f"Opened {symbol}: {signed_qty} @ {fill_price}")

    def get_current_positions_dict(self) -> Dict[str, int]:
        """Get current positions as {symbol: quantity} dict."""
        return {sym: pos.quantity for sym, pos in self.positions.items()}

    def record_daily_snapshot(self, date_str: str) -> None:
        """Record end-of-day portfolio snapshot."""
        self.pnl_history.append({
            'date': date_str,
            'equity': self.equity,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
            'margin_used': self.total_margin_used,
            'risk_indicator': self.risk_indicator,
            'num_positions': len(self.positions),
            'categories': self.num_categories_traded,
            'trading_days': self.num_trading_days,
        })

    def get_summary(self) -> Dict:
        """Portfolio summary for logging/display."""
        return {
            'equity': self.equity,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
            'total_fees': self.total_fees,
            'margin_utilization': self.margin_utilization,
            'risk_indicator': self.risk_indicator,
            'num_positions': len(self.positions),
            'categories_traded': sorted(self.categories_traded),
            'trading_days': self.num_trading_days,
            'meets_category_req': self.num_categories_traded >= self.config.min_product_categories,
            'meets_day_req': self.num_trading_days >= self.config.min_trading_days,
        }
