"""
TTB / pytaifex broker integration layer.
Abstracts communication with TAIFEX trading terminal via ZMQ.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = 'B'
    SELL = 'S'


class OrderType(Enum):
    MARKET = 'MKT'
    LIMIT = 'LMT'


class OrderStatus(Enum):
    PENDING = 'pending'
    SUBMITTED = 'submitted'
    PARTIAL = 'partial'
    FILLED = 'filled'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'


@dataclass
class OrderRequest:
    """Request to place an order."""
    symbol: str
    contract_month: str    # e.g., '202604'
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    price: Optional[float] = None
    timeout_seconds: int = 30  # Timeout before converting to market order


@dataclass
class OrderResponse:
    """Response from broker after order submission."""
    order_id: str
    status: OrderStatus
    filled_quantity: int = 0
    filled_price: float = 0.0
    message: str = ''


@dataclass
class Quote:
    """Real-time quote data."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: str


class BrokerInterface(ABC):
    """Abstract broker interface for order execution."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to broker / trading terminal."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""

    @abstractmethod
    def place_order(self, order: OrderRequest) -> OrderResponse:
        """Submit an order."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest quote for a symbol."""

    @abstractmethod
    def get_positions(self) -> Dict[str, int]:
        """Get current positions {symbol: quantity}."""

    @abstractmethod
    def get_account_info(self) -> Dict:
        """Get account equity, margin, etc."""


class PytaifexBroker(BrokerInterface):
    """Broker implementation using pytaifex (ZMQ-based TTB wrapper).

    Requires TTB trading terminal running on Windows.
    pytaifex connects via ZMQ to the TTB API.

    Usage:
        broker = PytaifexBroker(host='127.0.0.1', port=5555)
        broker.connect()
        resp = broker.place_order(OrderRequest(...))
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 5555):
        self.host = host
        self.port = port
        self._connected = False
        self._zmq_socket = None

    def connect(self) -> bool:
        """Connect to TTB via ZMQ."""
        try:
            import zmq
            context = zmq.Context()
            self._zmq_socket = context.socket(zmq.REQ)
            self._zmq_socket.connect(f'tcp://{self.host}:{self.port}')
            self._zmq_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout
            self._connected = True
            logger.info(f"Connected to TTB at {self.host}:{self.port}")
            return True
        except ImportError:
            logger.error("zmq not installed: pip install pyzmq")
            return False
        except Exception as e:
            logger.error(f"TTB connection failed: {e}")
            return False

    def disconnect(self) -> None:
        if self._zmq_socket:
            self._zmq_socket.close()
            self._connected = False
            logger.info("Disconnected from TTB")

    def place_order(self, order: OrderRequest) -> OrderResponse:
        """Send order to TTB via ZMQ."""
        if not self._connected:
            return OrderResponse(
                order_id='', status=OrderStatus.REJECTED,
                message='Not connected to broker',
            )

        # Build ZMQ message for pytaifex
        msg = {
            'action': 'place_order',
            'symbol': order.symbol,
            'month': order.contract_month,
            'side': order.side.value,
            'qty': order.quantity,
            'type': order.order_type.value,
            'price': order.price,
        }

        try:
            import json
            self._zmq_socket.send_json(msg)
            response = self._zmq_socket.recv_json()

            return OrderResponse(
                order_id=response.get('order_id', ''),
                status=OrderStatus(response.get('status', 'rejected')),
                filled_quantity=response.get('filled_qty', 0),
                filled_price=response.get('filled_price', 0.0),
                message=response.get('message', ''),
            )
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return OrderResponse(
                order_id='', status=OrderStatus.REJECTED,
                message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        if not self._connected:
            return False
        try:
            import json
            self._zmq_socket.send_json({'action': 'cancel', 'order_id': order_id})
            resp = self._zmq_socket.recv_json()
            return resp.get('status') == 'cancelled'
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def get_quote(self, symbol: str) -> Optional[Quote]:
        if not self._connected:
            return None
        try:
            self._zmq_socket.send_json({'action': 'quote', 'symbol': symbol})
            resp = self._zmq_socket.recv_json()
            return Quote(
                symbol=symbol,
                bid=resp.get('bid', 0),
                ask=resp.get('ask', 0),
                last=resp.get('last', 0),
                volume=resp.get('volume', 0),
                timestamp=resp.get('timestamp', ''),
            )
        except Exception as e:
            logger.error(f"Quote fetch failed: {e}")
            return None

    def get_positions(self) -> Dict[str, int]:
        if not self._connected:
            return {}
        try:
            self._zmq_socket.send_json({'action': 'positions'})
            resp = self._zmq_socket.recv_json()
            return resp.get('positions', {})
        except Exception as e:
            logger.error(f"Position fetch failed: {e}")
            return {}

    def get_account_info(self) -> Dict:
        if not self._connected:
            return {}
        try:
            self._zmq_socket.send_json({'action': 'account'})
            return self._zmq_socket.recv_json()
        except Exception as e:
            logger.error(f"Account info failed: {e}")
            return {}


class SimulatedBroker(BrokerInterface):
    """Simulated broker for backtesting and paper trading.

    Fills orders immediately at last price + slippage.
    """

    def __init__(self):
        self._connected = False
        self._positions: Dict[str, int] = {}
        self._prices: Dict[str, float] = {}
        self._equity: float = 2_000_000.0
        self._next_order_id = 1

    def connect(self) -> bool:
        self._connected = True
        logger.info("SimulatedBroker connected")
        return True

    def disconnect(self) -> None:
        self._connected = False

    def set_prices(self, prices: Dict[str, float]) -> None:
        """Update simulated prices (for backtest engine)."""
        self._prices = prices

    def set_equity(self, equity: float) -> None:
        self._equity = equity

    def place_order(self, order: OrderRequest) -> OrderResponse:
        price = self._prices.get(order.symbol, 0)
        if price <= 0:
            return OrderResponse(
                order_id='', status=OrderStatus.REJECTED,
                message=f'No price for {order.symbol}',
            )

        # Apply slippage
        slippage = 1.0  # 1 tick
        fill_price = price + slippage if order.side == OrderSide.BUY else price - slippage

        # Update position
        current = self._positions.get(order.symbol, 0)
        if order.side == OrderSide.BUY:
            self._positions[order.symbol] = current + order.quantity
        else:
            self._positions[order.symbol] = current - order.quantity

        oid = str(self._next_order_id)
        self._next_order_id += 1

        return OrderResponse(
            order_id=oid,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            filled_price=fill_price,
        )

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_quote(self, symbol: str) -> Optional[Quote]:
        price = self._prices.get(symbol, 0)
        if price <= 0:
            return None
        return Quote(symbol=symbol, bid=price - 1, ask=price + 1,
                     last=price, volume=0, timestamp='')

    def get_positions(self) -> Dict[str, int]:
        return dict(self._positions)

    def get_account_info(self) -> Dict:
        return {
            'equity': self._equity,
            'margin_used': 0,
            'available_margin': self._equity,
        }
