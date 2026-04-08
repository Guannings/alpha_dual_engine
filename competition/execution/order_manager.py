"""
Order lifecycle management.
Translates optimizer target positions into broker orders.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple

from ..data.contracts import CONTRACT_SPECS, get_contract_month_code, get_front_month_expiry
from .broker import (
    BrokerInterface, OrderRequest, OrderResponse, OrderSide,
    OrderStatus, OrderType,
)

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages the order lifecycle: target → orders → fills.

    Workflow:
        1. Compare target positions with current positions
        2. Generate orders for the delta
        3. Submit limit orders with timeout
        4. If not filled, convert to market orders
        5. Track fill confirmations
    """

    def __init__(
        self,
        broker: BrokerInterface,
        limit_timeout_seconds: int = 30,
        slippage_ticks: int = 2,
    ):
        self.broker = broker
        self.limit_timeout = limit_timeout_seconds
        self.slippage_ticks = slippage_ticks
        self.order_history: List[Dict] = []

    def reconcile_and_execute(
        self,
        target_positions: Dict[str, int],
        current_positions: Dict[str, int],
        as_of_date=None,
    ) -> List[OrderResponse]:
        """Execute orders to move from current to target positions.

        Args:
            target_positions: {symbol: target_qty} (signed)
            current_positions: {symbol: current_qty} (signed)
            as_of_date: Date for contract month calculation

        Returns:
            List of OrderResponse for each order executed.
        """
        responses = []

        # Calculate deltas
        all_symbols = set(target_positions.keys()) | set(current_positions.keys())
        for symbol in all_symbols:
            target = target_positions.get(symbol, 0)
            current = current_positions.get(symbol, 0)
            delta = target - current

            if delta == 0:
                continue

            # Determine order side and quantity
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            quantity = abs(delta)

            # Get contract month
            if as_of_date:
                from datetime import date as date_type
                if isinstance(as_of_date, date_type):
                    expiry = get_front_month_expiry(as_of_date)
                    contract_month = get_contract_month_code(expiry)
                else:
                    contract_month = as_of_date.strftime('%Y%m')
            else:
                from datetime import date
                expiry = get_front_month_expiry(date.today())
                contract_month = get_contract_month_code(expiry)

            # Get current price for limit order
            quote = self.broker.get_quote(symbol)
            if quote:
                if side == OrderSide.BUY:
                    limit_price = quote.ask + self.slippage_ticks
                else:
                    limit_price = quote.bid - self.slippage_ticks
            else:
                limit_price = None

            # Place limit order
            order = OrderRequest(
                symbol=symbol,
                contract_month=contract_month,
                side=side,
                quantity=quantity,
                order_type=OrderType.LIMIT if limit_price else OrderType.MARKET,
                price=limit_price,
                timeout_seconds=self.limit_timeout,
            )

            logger.info(
                f"Placing order: {side.value} {quantity} {symbol} "
                f"{'@' + str(limit_price) if limit_price else 'MKT'}"
            )

            resp = self.broker.place_order(order)
            responses.append(resp)

            self.order_history.append({
                'symbol': symbol,
                'side': side.value,
                'quantity': quantity,
                'price': limit_price,
                'status': resp.status.value,
                'filled_qty': resp.filled_quantity,
                'filled_price': resp.filled_price,
                'order_id': resp.order_id,
            })

            # If limit order not fully filled, try market order fallback
            if (resp.status == OrderStatus.PENDING
                    and resp.filled_quantity < quantity
                    and limit_price is not None):
                remaining = quantity - resp.filled_quantity
                logger.warning(
                    f"Limit order partial fill ({resp.filled_quantity}/{quantity}), "
                    f"sending market order for {remaining}"
                )
                # Cancel the pending limit order
                self.broker.cancel_order(resp.order_id)

                # Market order for remaining
                mkt_order = OrderRequest(
                    symbol=symbol,
                    contract_month=contract_month,
                    side=side,
                    quantity=remaining,
                    order_type=OrderType.MARKET,
                )
                mkt_resp = self.broker.place_order(mkt_order)
                responses.append(mkt_resp)

            if resp.status == OrderStatus.REJECTED:
                logger.error(f"Order rejected: {resp.message}")

        return responses

    def get_fill_summary(self) -> Dict:
        """Summary statistics of order execution."""
        if not self.order_history:
            return {'total_orders': 0}

        filled = [o for o in self.order_history if o['status'] == 'filled']
        rejected = [o for o in self.order_history if o['status'] == 'rejected']

        return {
            'total_orders': len(self.order_history),
            'filled': len(filled),
            'rejected': len(rejected),
            'fill_rate': len(filled) / len(self.order_history) if self.order_history else 0,
        }
