"""
Signal generation for each TAIFEX product category.
Maps regime → trade direction and product for 6+ categories.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import StrategyConfig
from ..data.contracts import CONTRACT_SPECS, ContractSpec

logger = logging.getLogger(__name__)


class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Signal:
    """A trading signal for a specific contract."""
    symbol: str
    category: str
    direction: Direction
    strength: float          # 0.0 to 1.0
    target_contracts: int    # Desired number of contracts (signed)
    rationale: str

    @property
    def is_active(self) -> bool:
        return self.direction != Direction.FLAT and self.strength > 0


class SignalGenerator:
    """Generate trading signals across 6+ product categories based on regime.

    Strategy mapping:
        TRENDING  → directional momentum (long TX, momentum equity futures)
        VOLATILE  → hedged positions, premium selling (sell OTM options, long gold)
        QUIET     → theta harvesting (sell strangles), range trades
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

    def generate(
        self,
        regime: str,
        ml_prob: float,
        taiex_price: float,
        momentum_21d: float,
        realized_vol: float,
        institutional_flow: float,
        put_call_ratio: float,
        capital: float,
    ) -> List[Signal]:
        """Generate signals for all active categories.

        Args:
            regime: 'TRENDING', 'VOLATILE', or 'QUIET'
            ml_prob: Bull probability from regime classifier
            taiex_price: Current TAIEX index level
            momentum_21d: 21-day TAIEX return
            realized_vol: Current annualized volatility
            institutional_flow: Institutional net buy/sell z-score
            put_call_ratio: Put/call ratio z-score
            capital: Current portfolio value (NT$)

        Returns:
            List of Signal objects for each product category.
        """
        signals = []

        signals.extend(self._index_futures_signals(
            regime, ml_prob, momentum_21d, institutional_flow, capital))
        signals.extend(self._index_options_signals(
            regime, taiex_price, realized_vol, put_call_ratio, capital))
        signals.extend(self._equity_futures_signals(
            regime, momentum_21d, capital))
        signals.extend(self._gold_futures_signals(
            regime, realized_vol, capital))
        signals.extend(self._fx_futures_signals(
            regime, institutional_flow, capital))
        signals.extend(self._midcap_futures_signals(
            regime, momentum_21d, capital))

        active = [s for s in signals if s.is_active]
        logger.info(
            f"Generated {len(active)}/{len(signals)} active signals "
            f"(regime={regime})"
        )
        return signals

    def _index_futures_signals(
        self, regime: str, ml_prob: float, momentum: float,
        institutional_flow: float, capital: float,
    ) -> List[Signal]:
        """Index futures (TX/MTX): trend following with momentum."""
        spec = CONTRACT_SPECS['MTX']  # Use Mini-TAIEX for smaller position sizing
        max_contracts = self._max_contracts(spec, capital)

        if regime == 'TRENDING':
            direction = Direction.LONG if momentum > 0 else Direction.SHORT
            strength = min(1.0, abs(momentum) * 10 + max(0, institutional_flow * 0.2))
            n_contracts = max(1, int(max_contracts * strength * 0.5))
        elif regime == 'VOLATILE':
            # Reduce exposure, slight directional bias from institutional flow
            direction = Direction.LONG if institutional_flow > 0.5 else Direction.SHORT
            strength = 0.3
            n_contracts = max(1, int(max_contracts * 0.2))
        else:  # QUIET
            direction = Direction.LONG if ml_prob > 0.55 else Direction.FLAT
            strength = 0.2
            n_contracts = max(1, int(max_contracts * 0.15))

        signed = n_contracts * direction.value
        return [Signal(
            symbol='MTX', category='index_futures', direction=direction,
            strength=strength, target_contracts=signed,
            rationale=f'{regime}: mom={momentum:.3f}, inst={institutional_flow:.2f}',
        )]

    def _index_options_signals(
        self, regime: str, taiex_price: float, vol: float,
        put_call_ratio: float, capital: float,
    ) -> List[Signal]:
        """Index options (TXO): premium selling strategies.

        TRENDING → sell OTM puts (bullish) or calls (bearish)
        VOLATILE → sell strangles (collect premium from both sides)
        QUIET    → sell strangles with tight strikes (max theta)
        """
        if regime == 'TRENDING':
            # Sell OTM puts in uptrend (bullish premium collection)
            direction = Direction.SHORT  # Selling puts = short put
            strength = 0.6
            n_contracts = 2
            rationale = 'Sell OTM puts — bullish premium collection'
        elif regime == 'VOLATILE':
            # Sell strangles — collect rich premium
            direction = Direction.SHORT
            strength = 0.5
            n_contracts = 1  # Conservative in high vol
            rationale = 'Sell strangle — high IV premium harvest'
        else:  # QUIET
            # Sell strangles with tighter strikes
            direction = Direction.SHORT
            strength = 0.7
            n_contracts = 3
            rationale = 'Sell tight strangle — max theta in low vol'

        return [Signal(
            symbol='TXO', category='index_options', direction=direction,
            strength=strength, target_contracts=n_contracts * direction.value,
            rationale=rationale,
        )]

    def _equity_futures_signals(
        self, regime: str, momentum: float, capital: float,
    ) -> List[Signal]:
        """Equity futures: momentum on major stocks (TSMC, MediaTek, etc.)."""
        if regime == 'TRENDING':
            direction = Direction.LONG if momentum > 0 else Direction.SHORT
            strength = 0.5
            n_contracts = 2
        elif regime == 'VOLATILE':
            direction = Direction.FLAT
            strength = 0.0
            n_contracts = 0
        else:
            direction = Direction.LONG if momentum > 0 else Direction.FLAT
            strength = 0.3
            n_contracts = 1

        return [Signal(
            symbol='STOCK_F', category='equity_futures', direction=direction,
            strength=strength, target_contracts=n_contracts * direction.value,
            rationale=f'{regime}: equity momentum={momentum:.3f}',
        )]

    def _gold_futures_signals(
        self, regime: str, vol: float, capital: float,
    ) -> List[Signal]:
        """Gold futures (GDF): safe haven in volatile/defensive regimes."""
        spec = CONTRACT_SPECS['GDF']

        if regime == 'VOLATILE':
            direction = Direction.LONG
            strength = 0.6
            n_contracts = 2
        elif regime == 'QUIET':
            direction = Direction.LONG
            strength = 0.3
            n_contracts = 1
        else:  # TRENDING — less need for safe haven
            direction = Direction.FLAT
            strength = 0.0
            n_contracts = 0

        return [Signal(
            symbol='GDF', category='gold_futures', direction=direction,
            strength=strength, target_contracts=n_contracts * direction.value,
            rationale=f'{regime}: gold safe-haven, vol={vol:.3f}',
        )]

    def _fx_futures_signals(
        self, regime: str, institutional_flow: float, capital: float,
    ) -> List[Signal]:
        """FX futures (UDF — USD/TWD): institutional flow driven."""
        spec = CONTRACT_SPECS['UDF']

        if regime == 'VOLATILE':
            # Foreign outflow → TWD weakens → long USD/TWD
            direction = Direction.LONG if institutional_flow < -0.5 else Direction.SHORT
            strength = 0.4
            n_contracts = 1
        elif regime == 'TRENDING':
            direction = Direction.SHORT if institutional_flow > 0.5 else Direction.FLAT
            strength = 0.3
            n_contracts = 1
        else:
            direction = Direction.FLAT
            strength = 0.0
            n_contracts = 0

        return [Signal(
            symbol='UDF', category='fx_futures', direction=direction,
            strength=strength, target_contracts=n_contracts * direction.value,
            rationale=f'{regime}: FX inst_flow={institutional_flow:.2f}',
        )]

    def _midcap_futures_signals(
        self, regime: str, momentum: float, capital: float,
    ) -> List[Signal]:
        """Mid-cap 100 futures (XIF): relative value vs TAIEX."""
        spec = CONTRACT_SPECS['XIF']

        if regime == 'TRENDING':
            direction = Direction.LONG if momentum > 0 else Direction.SHORT
            strength = 0.4
            n_contracts = 2
        elif regime == 'QUIET':
            direction = Direction.LONG
            strength = 0.3
            n_contracts = 1
        else:
            direction = Direction.FLAT
            strength = 0.0
            n_contracts = 0

        return [Signal(
            symbol='XIF', category='midcap_futures', direction=direction,
            strength=strength, target_contracts=n_contracts * direction.value,
            rationale=f'{regime}: midcap mom={momentum:.3f}',
        )]

    @staticmethod
    def _max_contracts(spec: ContractSpec, capital: float, utilization: float = 0.1) -> int:
        """Maximum contracts affordable within margin utilization limit."""
        if spec.margin_initial <= 0:
            return 5
        return max(1, int(capital * utilization / spec.margin_initial))
