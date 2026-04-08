"""
SLSQP position optimizer for TAIFEX futures portfolio.
Adapted from AlphaDominatorOptimizer in alpha_engine.py:488-953.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..config import StrategyConfig
from .signals import Signal, Direction

logger = logging.getLogger(__name__)


class PositionOptimizer:
    """SLSQP-based position sizer for futures portfolio.

    Takes raw signals and produces optimal contract counts subject to:
        - Margin constraints (must stay below max utilization)
        - Notional exposure limits per contract
        - Portfolio volatility target
        - Diversification (entropy) incentive
        - Turnover penalty (reduce churn)

    Regime-dependent objectives:
        TRENDING  → maximize momentum exposure, directional bets
        VOLATILE  → minimize notional, hedged positions
        QUIET     → maximize theta, conservative sizing
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.current_weights: Optional[np.ndarray] = None

    def optimize(
        self,
        signals: List[Signal],
        current_positions: Dict[str, int],
        capital: float,
        regime: str,
        margin_used: float = 0.0,
    ) -> Dict[str, int]:
        """Optimize position sizes from signals.

        Args:
            signals: List of Signal objects from SignalGenerator
            current_positions: Current {symbol: num_contracts}
            capital: Current portfolio value (NT$)
            regime: Market regime string
            margin_used: Current margin already consumed (NT$)

        Returns:
            Dict of {symbol: target_contracts} (signed, negative = short)
        """
        active_signals = [s for s in signals if s.is_active]
        if not active_signals:
            logger.info("No active signals — flattening all positions")
            return {sym: 0 for sym in current_positions}

        n = len(active_signals)
        symbols = [s.symbol for s in active_signals]

        # Raw signal weights (strength * direction)
        raw_weights = np.array([
            s.strength * s.direction.value for s in active_signals
        ])

        # Normalize to [-1, 1] range
        max_abs = np.abs(raw_weights).max()
        if max_abs > 0:
            norm_weights = raw_weights / max_abs
        else:
            norm_weights = raw_weights

        # SLSQP optimization to find optimal weight allocation
        target_weights = self._slsqp_optimize(
            norm_weights, active_signals, regime, capital,
        )

        # Convert weights to contract counts
        from ..data.contracts import CONTRACT_SPECS, get_margin_requirement
        target_positions = {}
        available_margin = capital * self.config.max_margin_utilization - margin_used

        for i, sig in enumerate(active_signals):
            w = target_weights[i]
            spec = CONTRACT_SPECS.get(sig.symbol)
            if spec is None or spec.margin_initial <= 0:
                # Options or unknown — use signal's target directly, scaled by weight
                target_positions[sig.symbol] = int(sig.target_contracts * abs(w))
                continue

            # Allocate margin proportionally
            margin_share = available_margin * abs(w)
            max_contracts = int(margin_share / spec.margin_initial)
            direction = 1 if w >= 0 else -1
            target_positions[sig.symbol] = direction * min(
                max_contracts, abs(sig.target_contracts)
            )

        # Ensure positions not in active signals get flattened
        for sym in current_positions:
            if sym not in target_positions:
                target_positions[sym] = 0

        self.current_weights = target_weights
        logger.info(f"Optimized {len(target_positions)} positions (regime={regime})")
        return target_positions

    def _slsqp_optimize(
        self,
        raw_weights: np.ndarray,
        signals: List[Signal],
        regime: str,
        capital: float,
    ) -> np.ndarray:
        """Run SLSQP to find optimal weight vector.

        Objective varies by regime:
            TRENDING  → maximize weighted momentum (directional)
            VOLATILE  → minimize total exposure (hedged)
            QUIET     → balanced with diversification bonus
        """
        n = len(signals)
        config = self.config
        old_weights = (
            self.current_weights
            if self.current_weights is not None and len(self.current_weights) == n
            else np.zeros(n)
        )

        # Signal strengths as momentum proxy
        strengths = np.array([s.strength for s in signals])

        if regime == 'TRENDING':
            def objective(w):
                # Maximize momentum-weighted exposure
                momentum_reward = np.dot(np.abs(w), strengths)
                # Diversification via entropy
                w_abs = np.abs(w)
                w_pos = w_abs[w_abs > 1e-6]
                w_norm = w_pos / w_pos.sum() if w_pos.sum() > 0 else w_pos
                entropy = -np.sum(w_norm * np.log(w_norm + 1e-10)) if len(w_norm) > 0 else 0
                # Turnover penalty
                turnover = np.sum(np.abs(w - old_weights))
                return (
                    -momentum_reward
                    - config.entropy_lambda * entropy
                    + config.turnover_penalty * turnover
                )
        elif regime == 'VOLATILE':
            def objective(w):
                # Minimize total exposure while respecting signals
                total_exposure = np.sum(np.abs(w))
                signal_alignment = np.dot(w, raw_weights)
                turnover = np.sum(np.abs(w - old_weights))
                return (
                    total_exposure * 0.5
                    - signal_alignment * 0.3
                    + config.turnover_penalty * turnover
                )
        else:  # QUIET
            def objective(w):
                # Balanced: some exposure, good diversification
                signal_alignment = np.dot(w, raw_weights)
                w_abs = np.abs(w)
                w_pos = w_abs[w_abs > 1e-6]
                w_norm = w_pos / w_pos.sum() if w_pos.sum() > 0 else w_pos
                entropy = -np.sum(w_norm * np.log(w_norm + 1e-10)) if len(w_norm) > 0 else 0
                turnover = np.sum(np.abs(w - old_weights))
                return (
                    -signal_alignment * 0.5
                    - config.entropy_lambda * entropy * 2
                    + config.turnover_penalty * turnover
                )

        # Bounds: each weight between -1 and 1
        bounds = [(-1.0, 1.0)] * n

        # Constraint: total absolute weight <= 1 (capital allocation)
        constraints = [
            {'type': 'ineq', 'fun': lambda w: 1.0 - np.sum(np.abs(w))},
        ]

        # Max position concentration
        max_single = config.max_position_notional_pct
        for i in range(n):
            constraints.append(
                {'type': 'ineq', 'fun': lambda w, idx=i: max_single - abs(w[idx])}
            )

        # Starting point: signal-direction scaled
        init_w = raw_weights * 0.5 / max(1.0, np.sum(np.abs(raw_weights)))
        init_w = np.clip(init_w, -1.0, 1.0)

        try:
            result = minimize(
                objective,
                init_w,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10},
            )
            if result.success:
                return result.x
            else:
                logger.warning(f"SLSQP did not converge: {result.message}")
                return init_w
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return init_w
