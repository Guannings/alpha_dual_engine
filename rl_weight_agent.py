#!/usr/bin/env python3
"""
RL Weight Agent — Low-Level PPO for Portfolio Weight Allocation
================================================================
Step D of the RL Roadmap (A -> D -> C).

Replaces the scipy SLSQP optimizer with a trained PPO agent that learns
optimal portfolio weight allocations given the regime chosen by the
existing high-level PPO agent.

Flow (training — soft constraints):
    Macro obs -> HIGH-LEVEL PPO (frozen) -> regime -+
    Per-asset signals + portfolio state  -----------+-> LOW-LEVEL PPO -> raw weights -> portfolio sim
    reward = base_reward - soft_penalties + diversity_bonus

Flow (inference — safety net):
    LOW-LEVEL PPO -> raw weights -> clip at 35% per asset -> renormalize -> final weights

Uses Apple MLX for native Apple Silicon acceleration.

Classes:
    WeightObservationBuilder   - Builds 103-dim observation vectors
    ConstraintLayer            - Hard post-processing for regime-specific constraints
    ContinuousActorCritic      - MLX actor-critic for continuous actions (12-dim)
    ContinuousRolloutBuffer    - Experience buffer for continuous PPO
    WeightBacktestEnv          - Gymnasium environment for weight learning
    WeightRLTrainer            - Continuous PPO training manager
    RLWeightAgent              - Inference wrapper for trained weight model
    HierarchicalRLController   - Combined high-level + low-level controller
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

WEIGHT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rl_weight_ppo")

REGIME_MAP = {0: 'RISK_ON', 1: 'RISK_REDUCED', 2: 'DEFENSIVE'}


# =============================================================================
# 1A. WEIGHT OBSERVATION BUILDER (103-dim)
# =============================================================================

class WeightObservationBuilder:
    """
    Builds a 103-dimensional observation vector for the low-level weight agent.

    Dimensions:
        [0:3]     3   Regime one-hot (from high-level agent)
        [3:15]   12   Per-asset raw_momentum
        [15:27]  12   Per-asset volatilities
        [27:39]  12   Per-asset RSI-14 (/ 100)
        [39:51]  12   Per-asset above_sma (binary)
        [51:63]  12   Per-asset golden_cross (binary)
        [63:75]  12   Per-asset information_ratio
        [75:87]  12   Per-asset log_returns_30d
        [87:99]  12   Current portfolio weights
        [99]      1   ML probability
        [100]     1   Current drawdown
        [101]     1   Normalized days since rebalance
        [102]     1   Normalized portfolio value

    Online z-score normalization (Welford) on dims 3:103 (skip regime one-hot).
    """

    OBS_DIM = 103

    def __init__(self):
        self._norm_dim = self.OBS_DIM - 3  # dims 3:103
        self._running_mean = np.zeros(self._norm_dim)
        self._running_var = np.ones(self._norm_dim)
        self._count = 0

    def reset(self) -> None:
        self._running_mean = np.zeros(self._norm_dim)
        self._running_var = np.ones(self._norm_dim)
        self._count = 0

    def build(
        self,
        regime_action: int,
        assets: List[str],
        raw_momentum: pd.Series,
        asset_volatilities: pd.Series,
        rsi_14: pd.Series,
        above_sma: pd.Series,
        golden_cross: pd.Series,
        information_ratio: pd.Series,
        log_returns_30d: pd.Series,
        current_weights: np.ndarray,
        ml_prob: float,
        current_drawdown: float,
        days_since_rebalance: int,
        rebalance_period: int,
        portfolio_value: float,
        initial_capital: float,
        noise_sigma: float = 0.0,
    ) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # Block 0:3 — Regime one-hot
        regime_onehot = np.zeros(3, dtype=np.float32)
        regime_onehot[regime_action] = 1.0
        obs[0:3] = regime_onehot

        # Helper to extract per-asset values
        def _get_vals(series: pd.Series) -> np.ndarray:
            return np.array([float(series.get(a, 0.0)) if a in series.index else 0.0
                             for a in assets], dtype=np.float32)

        # Block 3:15 — Per-asset raw_momentum
        obs[3:15] = np.clip(_get_vals(raw_momentum), -1.0, 1.0)

        # Block 15:27 — Per-asset volatilities
        obs[15:27] = np.clip(_get_vals(asset_volatilities), 0.0, 2.0)

        # Block 27:39 — Per-asset RSI-14 (/ 100)
        obs[27:39] = np.clip(_get_vals(rsi_14) / 100.0, 0.0, 1.0)

        # Block 39:51 — Per-asset above_sma (binary)
        obs[39:51] = np.array([float(above_sma.get(a, False)) if a in above_sma.index else 0.0
                                for a in assets], dtype=np.float32)

        # Block 51:63 — Per-asset golden_cross (binary)
        obs[51:63] = np.array([float(golden_cross.get(a, False)) if a in golden_cross.index else 0.0
                                for a in assets], dtype=np.float32)

        # Block 63:75 — Per-asset information_ratio
        obs[63:75] = np.clip(_get_vals(information_ratio), -3.0, 3.0)

        # Block 75:87 — Per-asset log_returns_30d
        obs[75:87] = np.clip(_get_vals(log_returns_30d), -1.0, 1.0)

        # Block 87:99 — Current portfolio weights
        obs[87:99] = current_weights[:12] if len(current_weights) >= 12 else np.pad(
            current_weights, (0, 12 - len(current_weights)))

        # Block 99:103 — Scalars
        obs[99] = np.clip(ml_prob, 0.0, 1.0)
        obs[100] = np.clip(current_drawdown, -1.0, 0.0)
        obs[101] = days_since_rebalance / max(rebalance_period, 1)
        obs[102] = np.clip((portfolio_value / initial_capital - 1.0), -1.0, 5.0) / 5.0

        # Online z-score normalization (Welford) on dims 3:103
        raw_block = obs[3:].copy()
        self._count += 1
        delta = raw_block - self._running_mean
        self._running_mean += delta / self._count
        delta2 = raw_block - self._running_mean
        self._running_var += (delta * delta2 - self._running_var) / self._count

        std = np.sqrt(self._running_var + 1e-8)
        obs[3:] = np.clip((raw_block - self._running_mean) / std, -3.0, 3.0).astype(np.float32)

        if noise_sigma > 0:
            obs[3:] += np.random.normal(0, noise_sigma, size=self._norm_dim).astype(np.float32)

        return obs


# =============================================================================
# 1B. CONSTRAINT LAYER — Hard Post-Processing
# =============================================================================

class ConstraintLayer:
    """
    Constraint layer for portfolio weights (dual mode).

    apply() — Hard post-processing (kept as inference fallback):
        1. Zero out ineligible assets
        2. Per-asset cap: min(w[i], 0.30)
        3. Gold cap: 1% in RISK_ON/RISK_REDUCED, uncapped in DEFENSIVE
        4. Crypto floor: 5% total always
        5. Crypto cap: 15% if BTC>50-SMA in RISK_ON, else 5%
        6. Aggressive ceiling: growth_anchors + crypto <= 95%
        7. DEFENSIVE: all equities zeroed
        8. Renormalize to sum = 1

    compute_penalties() — Soft quadratic penalties (used during training):
        Returns penalties that are zero when satisfied, grow quadratically
        for violations. Agent lives with consequences of its own decisions.

    compute_diversity_bonus() — Shannon entropy reward [0, 0.1]:
        Breaks degenerate all-in-one-asset policies.
    """

    def __init__(self, assets: List[str], asset_categories: Dict[str, List[str]], config=None):
        from alpha_engine import StrategyConfig
        self.assets = assets
        self.n_assets = len(assets)
        self.config = config or StrategyConfig()

        self.equity_idx = [assets.index(a) for a in asset_categories.get('equities', []) if a in assets]
        self.gold_idx = [assets.index(a) for a in asset_categories.get('gold', []) if a in assets]
        self.safe_haven_idx = [assets.index(a) for a in asset_categories.get('safe_haven', []) if a in assets]
        self.crypto_idx = [assets.index(a) for a in asset_categories.get('crypto', []) if a in assets]
        self.growth_anchor_idx = []
        from alpha_engine import DataManager
        for a in DataManager.GROWTH_ANCHORS:
            if a in assets:
                self.growth_anchor_idx.append(assets.index(a))

    def apply(
        self,
        raw_weights: np.ndarray,
        regime: str,
        above_sma: pd.Series,
        golden_cross: pd.Series,
        rsi_14: pd.Series,
    ) -> np.ndarray:
        w = raw_weights.copy()

        # Step 1: Zero out ineligible assets (reuse _get_eligible_mask logic)
        eligible = self._get_eligible_mask(regime, above_sma, golden_cross)
        w[~eligible] = 0.0

        # Step 2: Per-asset cap
        for i in range(self.n_assets):
            if w[i] > self.config.max_single_weight:
                w[i] = self.config.max_single_weight

        # Step 3: Gold cap — 1% in RISK_ON/RISK_REDUCED
        if regime in ('RISK_ON', 'RISK_REDUCED'):
            for i in self.gold_idx:
                w[i] = min(w[i], self.config.gold_cap_risk_on)

        # Step 4 & 5: Crypto floor and cap
        eligible_crypto = [i for i in self.crypto_idx if eligible[i]]
        if eligible_crypto:
            crypto_floor = self.config.crypto_floor_risk_on  # 5% total
            crypto_total = sum(w[i] for i in eligible_crypto)

            # Determine crypto cap
            if regime == 'RISK_ON':
                btc_above_50 = False
                aligned_gc = golden_cross.reindex(pd.Index(self.assets)).fillna(False)
                btc_indices = [i for i in eligible_crypto if self.assets[i] == 'BTC-USD']
                if btc_indices:
                    btc_above_50 = bool(aligned_gc.values[btc_indices[0]])
                crypto_cap = self.config.total_crypto_cap if btc_above_50 else self.config.crypto_floor_risk_on
            else:
                crypto_cap = self.config.crypto_floor_risk_on

            # Enforce floor
            if crypto_total < crypto_floor:
                per_coin = crypto_floor / max(len(eligible_crypto), 1)
                for i in eligible_crypto:
                    w[i] = per_coin
            # Enforce cap
            elif crypto_total > crypto_cap:
                scale = crypto_cap / (crypto_total + 1e-8)
                for i in eligible_crypto:
                    w[i] *= scale

        # Step 6: Aggressive ceiling — growth_anchors + crypto <= 95%
        ga_crypto_sum = sum(w[i] for i in self.growth_anchor_idx) + sum(w[i] for i in self.crypto_idx)
        if ga_crypto_sum > self.config.aggressive_ceiling:
            excess = ga_crypto_sum - self.config.aggressive_ceiling
            aggressive_idx = self.growth_anchor_idx + self.crypto_idx
            aggressive_total = sum(w[i] for i in aggressive_idx)
            if aggressive_total > 0:
                scale = (aggressive_total - excess) / aggressive_total
                for i in aggressive_idx:
                    w[i] *= scale

        # Step 8: Iterative cap + renormalize (prevents cap violations after renorm)
        w = np.maximum(w, 0.0)
        max_cap = self.config.max_single_weight  # 0.30
        for _iteration in range(10):
            ws = w.sum()
            if ws <= 0:
                # Fallback: equal weight safe havens
                for i in self.safe_haven_idx:
                    w[i] = 1.0 / max(len(self.safe_haven_idx), 1)
                w = w / w.sum()
                break
            w = w / ws
            # Check if any asset exceeds cap after renormalization
            over_mask = w > max_cap
            if not np.any(over_mask):
                break
            # Clip overweight assets and redistribute
            w[over_mask] = max_cap

        return w

    def _get_eligible_mask(self, regime: str, above_sma: pd.Series, golden_cross: pd.Series) -> np.ndarray:
        aligned_sma = above_sma.reindex(pd.Index(self.assets)).fillna(False)
        eligible = np.zeros(self.n_assets, dtype=bool)

        # Equities: eligible if Price > 200-SMA
        for idx in self.equity_idx:
            if aligned_sma.values[idx]:
                eligible[idx] = True
        for idx in self.growth_anchor_idx:
            if aligned_sma.values[idx]:
                eligible[idx] = True

        # Crypto: ALWAYS eligible
        for idx in self.crypto_idx:
            eligible[idx] = True

        # Safe havens: ALWAYS eligible
        for idx in self.safe_haven_idx:
            eligible[idx] = True

        # DEFENSIVE: disable equities
        if regime == 'DEFENSIVE':
            for idx in self.equity_idx:
                eligible[idx] = False
            for idx in self.growth_anchor_idx:
                eligible[idx] = False

        # Fail-safe
        growth_eq_eligible = any(eligible[idx] for idx in self.equity_idx + self.growth_anchor_idx)
        if not growth_eq_eligible:
            tlt_candidates = [i for i, a in enumerate(self.assets) if a == 'TLT']
            if tlt_candidates:
                eligible[tlt_candidates[0]] = True
            for idx in self.safe_haven_idx:
                eligible[idx] = True

        return eligible

    def compute_penalties(
        self,
        weights: np.ndarray,
        regime: str,
        above_sma: pd.Series,
        golden_cross: pd.Series,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute quadratic soft-constraint penalties for training.

        Returns (total_penalty, breakdown_dict) where penalties are zero when
        satisfied and grow quadratically for violations.
        """
        w = weights.copy()
        breakdown = {}

        # 1. Per-asset concentration: scale=5.0, threshold=30%
        conc_penalty = 5.0 * np.sum(np.maximum(0.0, w - 0.30) ** 2)
        breakdown['concentration'] = float(conc_penalty)

        # 2. Gold cap in RISK_ON / RISK_REDUCED: scale=3.0, threshold=1%
        gold_total = sum(w[i] for i in self.gold_idx)
        if regime in ('RISK_ON', 'RISK_REDUCED'):
            gold_penalty = 3.0 * max(0.0, gold_total - 0.01) ** 2
        else:
            gold_penalty = 0.0
        breakdown['gold'] = float(gold_penalty)

        # 3. Crypto floor: scale=1.5, threshold=5%
        crypto_total = sum(w[i] for i in self.crypto_idx)
        crypto_floor_penalty = 1.5 * max(0.0, 0.05 - crypto_total) ** 2
        breakdown['crypto_floor'] = float(crypto_floor_penalty)

        # 4. Crypto cap (regime-dependent): scale=2.0
        if regime == 'RISK_ON':
            btc_above_50 = False
            aligned_gc = golden_cross.reindex(pd.Index(self.assets)).fillna(False)
            btc_indices = [i for i in self.crypto_idx if self.assets[i] == 'BTC-USD']
            if btc_indices:
                btc_above_50 = bool(aligned_gc.values[btc_indices[0]])
            crypto_cap = self.config.total_crypto_cap if btc_above_50 else self.config.crypto_floor_risk_on
        else:
            crypto_cap = self.config.crypto_floor_risk_on
        crypto_cap_penalty = 2.0 * max(0.0, crypto_total - crypto_cap) ** 2
        breakdown['crypto_cap'] = float(crypto_cap_penalty)

        # 5. Aggressive ceiling: scale=1.5, threshold=95%
        ga_total = sum(w[i] for i in self.growth_anchor_idx)
        aggressive_penalty = 1.5 * max(0.0, ga_total + crypto_total - 0.95) ** 2
        breakdown['aggressive'] = float(aggressive_penalty)

        # 6. Ineligible equities (below 200-SMA): scale=4.0
        eligible = self._get_eligible_mask(regime, above_sma, golden_cross)
        ineligible_weight = np.sum(w[~eligible])
        ineligible_penalty = 4.0 * ineligible_weight ** 2
        breakdown['ineligible'] = float(ineligible_penalty)

        # 7. Growth anchor floor: scale=3.0, threshold=40% in RISK_ON
        if regime == 'RISK_ON':
            eligible_ga_weight = sum(w[i] for i in self.growth_anchor_idx if eligible[i])
            ga_floor_penalty = 3.0 * max(0.0, 0.40 - eligible_ga_weight) ** 2
        else:
            ga_floor_penalty = 0.0
        breakdown['ga_floor'] = float(ga_floor_penalty)

        total = conc_penalty + gold_penalty + crypto_floor_penalty + crypto_cap_penalty + aggressive_penalty + ineligible_penalty + ga_floor_penalty
        return float(total), breakdown

    def compute_diversity_bonus(self, weights: np.ndarray) -> float:
        """
        Shannon entropy diversity bonus, normalized to [0, 0.2].

        Breaks the "all-in-one-asset" degenerate policy by rewarding spread.
        """
        w = weights.copy()
        w = w[w > 1e-8]  # filter near-zero
        if len(w) == 0:
            return 0.0
        w = w / w.sum()  # renormalize filtered weights
        entropy = -np.sum(w * np.log(w))
        max_entropy = np.log(len(weights))  # ln(12)
        if max_entropy < 1e-8:
            return 0.0
        return 0.2 * (entropy / max_entropy)


# =============================================================================
# 1C. CONTINUOUS ACTOR-CRITIC (MLX)
# =============================================================================

class ContinuousActorCritic(nn.Module):
    """
    MLX actor-critic network for continuous PPO (portfolio weights).

    Architecture: shared 2x128 MLP →
    - Policy head: outputs mu (12-dim) for Gaussian sampling + learnable log_std
    - Value head: scalar state value

    Action: sample z ~ N(mu, exp(log_std)), then weights = softmax(z)
    """

    def __init__(self, obs_dim: int = 103, n_assets: int = 12, hidden: int = 128):
        super().__init__()
        self.n_assets = n_assets
        self.shared1 = nn.Linear(obs_dim, hidden)
        self.shared2 = nn.Linear(hidden, hidden)
        self.mu_head = nn.Linear(hidden, n_assets)
        self.value_head = nn.Linear(hidden, 1)
        # State-independent log_std
        self._log_std = mx.zeros((n_assets,))

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Returns (mu, log_std, state_value)."""
        h = nn.tanh(self.shared1(x))
        h = nn.tanh(self.shared2(h))
        mu = self.mu_head(h)
        value = self.value_head(h).squeeze(-1)
        # Broadcast log_std to batch size
        log_std = mx.broadcast_to(self._log_std, mu.shape)
        return mu, log_std, value

    def get_action_and_value(self, obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Numpy in, numpy out — for env interaction.
        Returns (weights, z, log_prob, value).
        """
        x = mx.array(obs_np.reshape(1, -1))
        mu, log_std, value = self(x)
        mx.eval(mu, log_std, value)

        mu_np = np.array(mu[0])
        log_std_np = np.array(log_std[0])
        std_np = np.exp(log_std_np)

        # Sample z ~ N(mu, std)
        z = mu_np + std_np * np.random.randn(self.n_assets)

        # Log probability under Gaussian
        log_prob = -0.5 * np.sum(
            ((z - mu_np) / (std_np + 1e-8)) ** 2 + 2 * log_std_np + np.log(2 * np.pi)
        )

        # Weights = softmax(z)
        z_shifted = z - z.max()
        exp_z = np.exp(z_shifted)
        weights = exp_z / exp_z.sum()

        return weights.astype(np.float32), z.astype(np.float32), float(log_prob), float(np.array(value[0]))

    def predict_deterministic(self, obs_np: np.ndarray) -> np.ndarray:
        """Deterministic: weights = softmax(mu), no sampling."""
        x = mx.array(obs_np.reshape(1, -1))
        mu, _, _ = self(x)
        mx.eval(mu)
        mu_np = np.array(mu[0])
        mu_shifted = mu_np - mu_np.max()
        exp_mu = np.exp(mu_shifted)
        weights = exp_mu / exp_mu.sum()
        return weights.astype(np.float32)


# =============================================================================
# 1D. CONTINUOUS ROLLOUT BUFFER
# =============================================================================

class ContinuousRolloutBuffer:
    """Stores experience tuples for continuous PPO training (z vectors instead of int actions)."""

    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.z_actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(self, obs: np.ndarray, z: np.ndarray, log_prob: float,
            reward: float, value: float, done: bool):
        self.observations.append(obs.copy())
        self.z_actions.append(z.copy())
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.observations.clear()
        self.z_actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.observations)

    def compute_returns_and_advantages(self, last_value: float,
                                        gamma: float = 0.99,
                                        gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return returns, advantages


# =============================================================================
# 1E. WEIGHT BACKTEST ENVIRONMENT
# =============================================================================

class WeightBacktestEnv(gym.Env):
    """
    Gymnasium environment for RL weight allocation.

    Action space:  Box(low=0, high=1, shape=(12,)) — raw weights (post-softmax)
    Observation:   Box(low=-inf, high=inf, shape=(103,))
    Episode:       Full backtest period with pre-computed rebalance schedule

    Uses frozen high-level model for regime decisions.
    """

    metadata = {"render_modes": []}

    TURNOVER_PENALTY_SCALE = 0.5

    def __init__(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        ml_probs: pd.Series,
        sma_200: pd.DataFrame,
        above_sma: pd.DataFrame,
        raw_momentum: pd.DataFrame,
        relative_strength: pd.DataFrame,
        asset_volatilities: pd.DataFrame,
        information_ratio: pd.DataFrame,
        momentum_score: pd.DataFrame,
        golden_cross: pd.DataFrame,
        log_returns_30d: pd.DataFrame,
        rsi_14: pd.DataFrame,
        optimizer,
        config=None,
        train_end_date: Optional[str] = None,
        noise_sigma: float = 0.0,
        lookback_days: int = 252,
        high_level_model_path: Optional[str] = None,
    ):
        super().__init__()

        from alpha_engine import StrategyConfig, DataManager, BacktestEngine
        from rl_regime_agent import (
            ActorCritic as HighLevelActorCritic,
            ObservationBuilder as HighLevelObsBuilder,
            DifferentialSharpeReward,
            _load_mlx_model,
            MODEL_DIR as REGIME_MODEL_DIR,
        )

        self.config = config or StrategyConfig()
        self.optimizer = optimizer
        self.noise_sigma = noise_sigma
        self.lookback_days = lookback_days

        # Slice data to train window if specified
        if train_end_date is not None:
            cut = pd.Timestamp(train_end_date)
            prices = prices[prices.index <= cut]
            returns = returns[returns.index <= cut]
            features = features[features.index <= cut]
            ml_probs = ml_probs[ml_probs.index <= cut]
            sma_200 = sma_200[sma_200.index <= cut]
            above_sma = above_sma[above_sma.index <= cut]
            raw_momentum = raw_momentum[raw_momentum.index <= cut]
            relative_strength = relative_strength[relative_strength.index <= cut]
            asset_volatilities = asset_volatilities[asset_volatilities.index <= cut]
            information_ratio = information_ratio[information_ratio.index <= cut]
            momentum_score = momentum_score[momentum_score.index <= cut]
            golden_cross = golden_cross[golden_cross.index <= cut]
            log_returns_30d = log_returns_30d[log_returns_30d.index <= cut]
            rsi_14 = rsi_14[rsi_14.index <= cut]

        self.prices = prices
        self.returns = returns
        self.features = features
        self.ml_probs = ml_probs
        self.sma_200 = sma_200
        self.above_sma = above_sma
        self.raw_momentum = raw_momentum
        self.relative_strength = relative_strength
        self.asset_volatilities = asset_volatilities
        self.information_ratio = information_ratio
        self.momentum_score = momentum_score
        self.golden_cross = golden_cross
        self.log_returns_30d = log_returns_30d
        self.rsi_14 = rsi_14

        # Valid dates
        self.valid_dates = (
            ml_probs.dropna().index
            .intersection(prices.index)
            .intersection(returns.index)
            .intersection(above_sma.dropna().index)
            .intersection(raw_momentum.dropna().index)
            .intersection(information_ratio.dropna().index)
        )

        # Asset category indices
        self.equity_indices = list(optimizer.equity_idx)
        self.safe_haven_indices = list(optimizer.safe_haven_idx)
        self.crypto_indices = list(optimizer.crypto_idx)
        self.equity_tickers = [t for t in DataManager.EQUITIES if t in optimizer.assets]

        # SPY returns for benchmark
        spy_ticker = DataManager.BENCHMARK_TICKER
        self.spy_returns = returns[spy_ticker] if spy_ticker in returns.columns else pd.Series(0.0, index=returns.index)

        # Transaction costs
        self.asset_cost_bps = BacktestEngine._get_asset_cost_bps(optimizer.assets)
        # Rebalance schedule
        self.rebalance_period = self.config.rebalance_period
        self.start_idx = self.lookback_days
        self._precompute_rebalance_schedule()

        # Load frozen high-level model
        hl_path = high_level_model_path or os.path.join(REGIME_MODEL_DIR, "best_model")
        try:
            self.high_level_model = _load_mlx_model(hl_path)
            logger.info(f"Loaded frozen high-level model from {hl_path}")
        except FileNotFoundError:
            self.high_level_model = None
            logger.warning(f"No high-level model at {hl_path} — will use random regimes")

        # High-level observation builder
        self.hl_obs_builder = HighLevelObsBuilder()

        # Low-level observation builder
        self.weight_obs_builder = WeightObservationBuilder()

        # Constraint layer
        categories = {
            'equities': [a for a in DataManager.EQUITIES if a in optimizer.assets],
            'fixed_income': [a for a in DataManager.FIXED_INCOME if a in optimizer.assets],
            'alternatives': [a for a in DataManager.ALTERNATIVES if a in optimizer.assets],
            'crypto': [a for a in DataManager.CRYPTO if a in optimizer.assets],
            'safe_haven': [a for a in ['GLD', 'TLT', 'IEF', 'SHY'] if a in optimizer.assets],
            'gold': ['GLD'] if 'GLD' in optimizer.assets else [],
            'bonds_cash': [a for a in ['TLT', 'IEF', 'SHY'] if a in optimizer.assets],
        }
        self.constraint_layer = ConstraintLayer(optimizer.assets, categories, self.config)

        # Reward function
        self.reward_fn = DifferentialSharpeReward()

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(WeightObservationBuilder.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(optimizer.n_assets,),
            dtype=np.float32,
        )

        # Soft constraint settings
        self.use_soft_constraints = True
        self.diversity_lambda = 0.1
        self._penalty_history: List[Dict[str, float]] = []

        # Episode state
        self._episode_step = 0
        self._portfolio_value = 100000.0
        self._benchmark_value = 100000.0
        self._peak_value = 100000.0
        self._current_weights = np.zeros(optimizer.n_assets)
        self._portfolio_values_history: List[float] = []
        self._benchmark_values_history: List[float] = []

    def _precompute_rebalance_schedule(self):
        self.rebalance_indices = []
        idx = self.start_idx
        while idx < len(self.valid_dates):
            self.rebalance_indices.append(idx)
            idx += self.rebalance_period
        if len(self.rebalance_indices) < 2:
            logger.warning(f"Only {len(self.rebalance_indices)} rebalance points")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_step = 0
        self._portfolio_value = 100000.0
        self._benchmark_value = 100000.0
        self._peak_value = 100000.0
        self._current_weights = np.zeros(self.optimizer.n_assets)
        self._portfolio_values_history = [100000.0]
        self._benchmark_values_history = [100000.0]
        self.weight_obs_builder.reset()
        self.hl_obs_builder.reset()
        self.reward_fn.reset()

        obs = self._get_observation()
        return obs, {}

    def _get_regime_action(self, date) -> int:
        """Get regime from frozen high-level model (deterministic).

        FIX: Uses benchmark values (not portfolio values) for the regime agent's
        observation so the weight agent's tracking error doesn't bias regime selection.
        """
        if self.high_level_model is None:
            return np.random.randint(0, 3)

        idx = self.valid_dates.get_loc(date)

        ml_prob_val = self.ml_probs.loc[date]
        if np.isnan(ml_prob_val):
            ml_prob_val = 0.5

        bv_arr = np.array(self._benchmark_values_history)

        def _trailing_return(arr, n):
            if len(arr) < n + 1:
                return 0.0
            return (arr[-1] / arr[-n - 1]) - 1.0

        peak_bv = max(bv_arr) if len(bv_arr) > 0 else self._benchmark_value
        current_dd = (self._benchmark_value - peak_bv) / peak_bv if peak_bv > 0 else 0.0

        hl_obs = self.hl_obs_builder.build(
            features_row=self.features.loc[date],
            current_weights=self._current_weights,
            equity_indices=self.equity_indices,
            safe_haven_indices=self.safe_haven_indices,
            crypto_indices=self.crypto_indices,
            portfolio_value=self._benchmark_value,
            initial_capital=100000.0,
            days_since_rebalance=self.rebalance_period,
            rebalance_period=self.rebalance_period,
            current_drawdown=current_dd,
            above_sma=self.above_sma.loc[date],
            raw_momentum=self.raw_momentum.loc[date],
            asset_volatilities=self.asset_volatilities.loc[date],
            information_ratio=self.information_ratio.loc[date],
            ml_prob=ml_prob_val,
            golden_cross=self.golden_cross.loc[date],
            equity_tickers=self.equity_tickers,
            portfolio_returns_5d=_trailing_return(bv_arr, 5),
            portfolio_returns_21d=_trailing_return(bv_arr, 21),
            portfolio_returns_63d=_trailing_return(bv_arr, 63),
            benchmark_returns_21d=_trailing_return(bv_arr, 21),
            noise_sigma=0.0,
        )

        return self.high_level_model.predict_deterministic(hl_obs)

    def _get_observation(self) -> np.ndarray:
        if self._episode_step >= len(self.rebalance_indices):
            return np.zeros(WeightObservationBuilder.OBS_DIM, dtype=np.float32)

        idx = self.rebalance_indices[self._episode_step]
        date = self.valid_dates[idx]

        regime_action = self._get_regime_action(date)
        self._current_regime_action = regime_action

        ml_prob_val = self.ml_probs.loc[date]
        if np.isnan(ml_prob_val):
            ml_prob_val = 0.5

        current_dd = (self._portfolio_value - self._peak_value) / self._peak_value if self._peak_value > 0 else 0.0

        return self.weight_obs_builder.build(
            regime_action=regime_action,
            assets=self.optimizer.assets,
            raw_momentum=self.raw_momentum.loc[date],
            asset_volatilities=self.asset_volatilities.loc[date],
            rsi_14=self.rsi_14.loc[date],
            above_sma=self.above_sma.loc[date],
            golden_cross=self.golden_cross.loc[date],
            information_ratio=self.information_ratio.loc[date],
            log_returns_30d=self.log_returns_30d.loc[date],
            current_weights=self._current_weights,
            ml_prob=ml_prob_val,
            current_drawdown=current_dd,
            days_since_rebalance=self.rebalance_period,
            rebalance_period=self.rebalance_period,
            portfolio_value=self._portfolio_value,
            initial_capital=100000.0,
            noise_sigma=self.noise_sigma,
        )

    def step(self, action: np.ndarray):
        if self._episode_step >= len(self.rebalance_indices):
            return self._get_observation(), 0.0, True, False, {}

        rebal_idx = self.rebalance_indices[self._episode_step]
        date = self.valid_dates[rebal_idx]
        regime_action = getattr(self, '_current_regime_action', 0)
        regime = REGIME_MAP[regime_action]

        # --- 1. Apply constraint layer (hard constraints during training) ---
        new_weights = action.copy()
        new_weights = self.constraint_layer.apply(
            new_weights, regime,
            self.above_sma.loc[date],
            self.golden_cross.loc[date],
            self.rsi_14.loc[date],
        )

        # --- 2. Lazy drift ---
        if self._current_weights.sum() > 0:
            drift = np.abs(new_weights - self._current_weights)
            lazy_mask = drift < self.config.lazy_drift_threshold
            new_weights[lazy_mask] = self._current_weights[lazy_mask]
            ws = new_weights.sum()
            if ws > 0:
                new_weights = new_weights / ws

        # --- 3. Turnover gate + transaction costs ---
        turnover = np.sum(np.abs(new_weights - self._current_weights))
        if turnover >= self.config.min_rebalance_threshold:
            per_asset_turnover = np.abs(new_weights - self._current_weights)
            spread_cost = np.sum(per_asset_turnover * self.asset_cost_bps / 10000)
            cost = self._portfolio_value * spread_cost
            self._portfolio_value -= cost
            self._current_weights = new_weights

        # --- 3b. Simulation weights: momentum tilt + GA floor (match inference) ---
        sim_weights = self._current_weights.copy()
        if regime == 'RISK_ON':
            mom_vals = self.raw_momentum.loc[date]
            mom_scores_tilt = np.array([max(mom_vals.get(a, 0), 0) ** 2 for a in self.optimizer.assets])
            if mom_scores_tilt.sum() > 0:
                mom_scores_tilt = mom_scores_tilt / mom_scores_tilt.sum()
                sim_weights = 0.5 * sim_weights + 0.5 * mom_scores_tilt
                sim_weights = self.constraint_layer.apply(
                    sim_weights, regime,
                    self.above_sma.loc[date],
                    self.golden_cross.loc[date],
                    self.rsi_14.loc[date],
                )

            ga_idx = self.constraint_layer.growth_anchor_idx
            aligned_sma = self.above_sma.loc[date].reindex(pd.Index(self.optimizer.assets)).fillna(False)
            eligible_ga = [i for i in ga_idx if aligned_sma.values[i]]
            if eligible_ga:
                ga_weight = sum(sim_weights[i] for i in eligible_ga)
                if ga_weight < 0.40:
                    deficit = 0.40 - ga_weight
                    boost_each = deficit / len(eligible_ga)
                    for i in eligible_ga:
                        sim_weights[i] += boost_each
                    sim_weights = sim_weights / sim_weights.sum()

        # --- 4. Simulate daily returns forward ---
        next_step = self._episode_step + 1
        if next_step < len(self.rebalance_indices):
            end_idx = self.rebalance_indices[next_step]
        else:
            end_idx = len(self.valid_dates)

        window_port_returns = []
        window_bench_returns = []

        for day_idx in range(rebal_idx, min(end_idx, len(self.valid_dates))):
            day = self.valid_dates[day_idx]
            daily_ret = self.returns.loc[day][self.optimizer.assets].values
            daily_ret = np.nan_to_num(daily_ret, 0)
            port_daily_ret = np.dot(sim_weights, daily_ret)
            self._portfolio_value *= (1 + port_daily_ret)
            window_port_returns.append(port_daily_ret)

            spy_ret = self.spy_returns.loc[day] if day in self.spy_returns.index else 0.0
            self._benchmark_value *= (1 + spy_ret)
            window_bench_returns.append(spy_ret)

            self._peak_value = max(self._peak_value, self._portfolio_value)
            self._portfolio_values_history.append(self._portfolio_value)
            self._benchmark_values_history.append(self._benchmark_value)

        # --- 5. Compute reward ---
        base_reward = self.reward_fn.compute(
            np.array(window_port_returns),
            np.array(window_bench_returns),
            self._portfolio_value,
            self._peak_value,
        )

        # Soft constraint penalties (agent feels consequences of violations)
        constraint_penalty, penalty_breakdown = self.constraint_layer.compute_penalties(
            self._current_weights, regime,
            self.above_sma.loc[date],
            self.golden_cross.loc[date],
        )

        # Diversity bonus (breaks all-in-one-asset degeneracy)
        diversity_bonus = self.constraint_layer.compute_diversity_bonus(self._current_weights)

        reward = base_reward - self.TURNOVER_PENALTY_SCALE * turnover - constraint_penalty + diversity_bonus

        # Momentum alignment bonus: reward concentrating into momentum winners
        if regime == 'RISK_ON':
            mom_vals = self.raw_momentum.loc[date]
            mom_scores = np.array([max(mom_vals.get(a, 0), 0) ** 2 for a in self.optimizer.assets])
            if mom_scores.sum() > 0:
                mom_scores = mom_scores / mom_scores.sum()
                momentum_alignment = np.dot(new_weights, mom_scores)
                reward += 0.20 * momentum_alignment

        # Clip total reward to prevent exploitation of training-specific patterns
        reward = float(np.clip(reward, -5.0, 5.0))

        # Track penalties for logging
        penalty_breakdown['total'] = constraint_penalty
        penalty_breakdown['diversity_bonus'] = diversity_bonus
        penalty_breakdown['base_reward'] = base_reward
        self._penalty_history.append(penalty_breakdown)

        # --- 6. Advance step ---
        self._episode_step += 1
        terminated = self._episode_step >= len(self.rebalance_indices)
        truncated = False

        obs = self._get_observation()
        info = {
            'portfolio_value': self._portfolio_value,
            'benchmark_value': self._benchmark_value,
            'regime': regime,
            'turnover': turnover,
            'constraint_penalty': constraint_penalty,
            'diversity_bonus': diversity_bonus,
        }

        return obs, reward, terminated, truncated, info


# =============================================================================
# 1F. WEIGHT RL TRAINER (Continuous PPO)
# =============================================================================

class WeightRLTrainer:
    """
    Continuous PPO training manager for the low-level weight agent.

    Hyperparameters:
        lr=1e-4, n_steps=256, batch_size=64, gamma=0.99,
        ent_coef=0.05, hidden=128
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        ml_probs: pd.Series,
        sma_200: pd.DataFrame,
        above_sma: pd.DataFrame,
        raw_momentum: pd.DataFrame,
        relative_strength: pd.DataFrame,
        asset_volatilities: pd.DataFrame,
        information_ratio: pd.DataFrame,
        momentum_score: pd.DataFrame,
        golden_cross: pd.DataFrame,
        log_returns_30d: pd.DataFrame,
        rsi_14: pd.DataFrame,
        optimizer,
        config=None,
        train_end_date: str = "2023-12-31",
        total_timesteps: int = 200_000,
        model_dir: str = WEIGHT_MODEL_DIR,
    ):
        from alpha_engine import StrategyConfig
        self.config = config or StrategyConfig()
        self.train_end_date = train_end_date
        self.total_timesteps = total_timesteps
        self.model_dir = model_dir
        self.n_assets = optimizer.n_assets

        self._data_kwargs = dict(
            prices=prices, returns=returns, features=features,
            ml_probs=ml_probs, sma_200=sma_200, above_sma=above_sma,
            raw_momentum=raw_momentum, relative_strength=relative_strength,
            asset_volatilities=asset_volatilities,
            information_ratio=information_ratio,
            momentum_score=momentum_score, golden_cross=golden_cross,
            log_returns_30d=log_returns_30d, rsi_14=rsi_14,
            optimizer=optimizer, config=self.config,
        )

        self.model: Optional[ContinuousActorCritic] = None
        self.reward_history: List[float] = []

    def _make_env(self, train: bool = True) -> WeightBacktestEnv:
        if train:
            return WeightBacktestEnv(
                **self._data_kwargs,
                train_end_date='2022-12-31',
                noise_sigma=0.05,
            )
        else:
            val_kwargs = {}
            val_start = pd.Timestamp('2022-01-01')
            val_end = pd.Timestamp(self.train_end_date)
            for key, val in self._data_kwargs.items():
                if isinstance(val, (pd.DataFrame, pd.Series)):
                    val_kwargs[key] = val[(val.index >= val_start) & (val.index <= val_end)]
                else:
                    val_kwargs[key] = val
            return WeightBacktestEnv(
                **val_kwargs,
                train_end_date=None,
                noise_sigma=0.0,
            )

    def train(self) -> ContinuousActorCritic:
        """Run continuous PPO training with MLX."""
        lr = 1e-4
        n_steps = 256
        batch_size = 64
        gamma = 0.99
        gae_lambda = 0.95
        clip_range = 0.2
        ent_coef = 0.10
        vf_coef = 0.5
        max_grad_norm = 0.5
        n_epochs = 6
        eval_freq = max(self.total_timesteps // 30, 500)
        checkpoint_freq = 30_000  # Save checkpoints every 30K steps
        patience = 999  # Disabled; rely on checkpoints + best_model selection
        evals_without_improvement = 0

        logger.info(f"Starting weight PPO training: {self.total_timesteps} timesteps")
        print(f"Weight agent: MLX backend (continuous PPO)", flush=True)

        train_env = self._make_env(train=True)
        val_env = self._make_env(train=False)
        os.makedirs(self.model_dir, exist_ok=True)

        self.model = ContinuousActorCritic(
            obs_dim=WeightObservationBuilder.OBS_DIM,
            n_assets=self.n_assets,
            hidden=128,
        )
        mx.eval(self.model.parameters())
        optimizer_mlx = optim.Adam(learning_rate=lr)

        # Continuous PPO loss
        def ppo_loss(model, obs_batch, z_batch, old_logprob_batch,
                     advantage_batch, return_batch):
            mu, log_std, values = model(obs_batch)
            std = mx.exp(log_std)

            # Gaussian log probability of stored z
            new_log_probs = -0.5 * mx.sum(
                ((z_batch - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi),
                axis=-1
            )

            ratio = mx.exp(new_log_probs - old_logprob_batch)
            surr1 = ratio * advantage_batch
            surr2 = mx.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage_batch
            policy_loss = -mx.minimum(surr1, surr2).mean()

            value_loss = ((values - return_batch) ** 2).mean()

            # Gaussian entropy: 0.5 * n * (1 + log(2*pi)) + sum(log_std)
            entropy = 0.5 * self.n_assets * (1.0 + np.log(2 * np.pi)) + mx.sum(log_std, axis=-1)
            entropy = entropy.mean()

            total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            return total_loss

        loss_and_grad = nn.value_and_grad(self.model, ppo_loss)

        # Training loop
        buffer = ContinuousRolloutBuffer()
        obs, _ = train_env.reset()
        episode_reward = 0.0
        total_steps = 0
        best_eval_reward = -float('inf')
        episode_count = 0
        prev_checkpoint_step = 0
        early_stopped = False
        while total_steps < self.total_timesteps:
            buffer.clear()
            for _ in range(n_steps):
                weights, z, log_prob, value = self.model.get_action_and_value(obs)
                next_obs, reward, terminated, truncated, info = train_env.step(weights)
                done = terminated or truncated

                buffer.add(obs, z, log_prob, reward, value, done)
                obs = next_obs
                episode_reward += reward
                total_steps += 1

                if done:
                    self.reward_history.append(episode_reward)
                    episode_count += 1
                    episode_reward = 0.0
                    obs, _ = train_env.reset()

                if total_steps >= self.total_timesteps:
                    break

            # Compute advantages
            _, _, _, last_value = self.model.get_action_and_value(obs)
            returns, advantages = buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)

            adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            # PPO update
            obs_arr = np.array(buffer.observations, dtype=np.float32)
            z_arr = np.array(buffer.z_actions, dtype=np.float32)
            old_lp_arr = np.array(buffer.log_probs, dtype=np.float32)
            buf_len = len(buffer)

            for epoch in range(n_epochs):
                indices = np.random.permutation(buf_len)
                for start in range(0, buf_len, batch_size):
                    end = min(start + batch_size, buf_len)
                    mb_idx = indices[start:end]

                    mb_obs = mx.array(obs_arr[mb_idx])
                    mb_z = mx.array(z_arr[mb_idx])
                    mb_old_lp = mx.array(old_lp_arr[mb_idx])
                    mb_adv = mx.array(advantages[mb_idx])
                    mb_ret = mx.array(returns[mb_idx])

                    loss, grads = loss_and_grad(
                        self.model, mb_obs, mb_z, mb_old_lp, mb_adv, mb_ret
                    )

                    # Gradient clipping
                    import mlx.utils as mlx_utils
                    flat_grads = mlx_utils.tree_flatten(grads)
                    grad_norm = mx.sqrt(sum(
                        (g * g).sum() for _, g in flat_grads
                    ))
                    mx.eval(grad_norm)
                    gn = float(np.array(grad_norm))
                    if gn > max_grad_norm:
                        scale = max_grad_norm / (gn + 1e-8)
                        grads = mlx_utils.tree_map(
                            lambda g: g * scale if isinstance(g, mx.array) else g, grads
                        )

                    optimizer_mlx.update(self.model, grads)
                    mx.eval(self.model.parameters(), optimizer_mlx.state)

            # Logging
            if len(self.reward_history) > 0:
                recent = self.reward_history[-max(episode_count, 1):]
                mean_rew = np.mean(recent) if recent else 0.0
            else:
                mean_rew = 0.0

            iteration = total_steps // n_steps
            print(f"| iter {iteration:4d} | timesteps {total_steps:6d}/{self.total_timesteps} "
                  f"| episodes {episode_count:4d} | mean_reward {mean_rew:+7.2f} |", flush=True)

            # Penalty breakdown logging (every 10 iterations)
            if iteration % 10 == 0 and len(train_env._penalty_history) > 0:
                recent_penalties = train_env._penalty_history[-n_steps:]
                avg_total = np.mean([p['total'] for p in recent_penalties])
                avg_conc = np.mean([p.get('concentration', 0) for p in recent_penalties])
                avg_inelig = np.mean([p.get('ineligible', 0) for p in recent_penalties])
                avg_crypto = np.mean([p.get('crypto_cap', 0) + p.get('crypto_floor', 0) for p in recent_penalties])
                avg_div = np.mean([p.get('diversity_bonus', 0) for p in recent_penalties])
                print(f"  [PENALTIES] total={avg_total:.3f} conc={avg_conc:.3f} "
                      f"inelig={avg_inelig:.3f} crypto={avg_crypto:.3f} div_bonus={avg_div:.3f}", flush=True)

            # Linear LR decay: reduce to 50% of initial LR by end of training
            progress = total_steps / self.total_timesteps
            new_lr = lr * (1.0 - 0.5 * progress)
            optimizer_mlx.learning_rate = new_lr

            # Save periodic checkpoints (every 50K steps)
            if total_steps - prev_checkpoint_step >= checkpoint_freq:
                ckpt_name = f"checkpoint_{total_steps // 1000}k"
                self._save_model(os.path.join(self.model_dir, ckpt_name))
                print(f"  [CHECKPOINT] Saved {ckpt_name} at {total_steps} steps", flush=True)
                prev_checkpoint_step = total_steps

            # Periodic evaluation
            if total_steps % eval_freq < n_steps or total_steps >= self.total_timesteps:
                eval_reward = self._run_eval(val_env, n_episodes=5)
                print(f"  [EVAL] mean_reward={eval_reward:.3f} (best={best_eval_reward:.3f})", flush=True)

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    evals_without_improvement = 0
                    self._save_model(os.path.join(self.model_dir, "best_model"))
                    print(f"  [EVAL] New best weight model saved!", flush=True)
                else:
                    evals_without_improvement += 1
                    if evals_without_improvement >= patience:
                        print(f"  [EARLY STOP] No improvement for {patience} evals. Stopping.", flush=True)
                        early_stopped = True
                        break
        self._save_model(os.path.join(self.model_dir, "final_model"))
        stop_msg = " (early stopped)" if early_stopped else ""
        logger.info(f"Weight training complete{stop_msg}. {episode_count} episodes, {total_steps} timesteps.")
        return self.model

    def _run_eval(self, env: WeightBacktestEnv, n_episodes: int = 3) -> float:
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                weights = self.model.predict_deterministic(obs)
                obs, reward, terminated, truncated, info = env.step(weights)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
        return float(np.mean(rewards))

    def _save_model(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        weights_path = path + ".safetensors"
        config_path = path + ".json"
        import mlx.utils as mlx_utils
        mx.save_safetensors(weights_path, dict(mlx_utils.tree_flatten(self.model.parameters())))
        with open(config_path, 'w') as f:
            json.dump({'obs_dim': 103, 'n_assets': self.n_assets, 'hidden': 128}, f)
        logger.info(f"Weight model saved to {path}")

    def evaluate(self, n_episodes: int = 5) -> Dict:
        if self.model is None:
            self.model = _load_weight_model(os.path.join(self.model_dir, "best_model"))

        val_env = self._make_env(train=False)

        episode_returns = []
        episode_sharpes = []

        for _ in range(n_episodes):
            obs, _ = val_env.reset()
            total_reward = 0.0
            done = False
            while not done:
                weights = self.model.predict_deterministic(obs)
                obs, reward, terminated, truncated, info = val_env.step(weights)
                total_reward += reward
                done = terminated or truncated

            episode_returns.append(total_reward)

            pv = np.array(val_env._portfolio_values_history)
            daily_rets = np.diff(pv) / pv[:-1] if len(pv) > 1 else np.array([0.0])
            sharpe = (np.mean(daily_rets) * 252 - self.config.risk_free_rate) / (np.std(daily_rets) * np.sqrt(252) + 1e-8)
            episode_sharpes.append(sharpe)

        results = {
            'mean_reward': float(np.mean(episode_returns)),
            'std_reward': float(np.std(episode_returns)),
            'mean_sharpe': float(np.mean(episode_sharpes)),
            'final_portfolio_value': val_env._portfolio_value,
            'final_benchmark_value': val_env._benchmark_value,
        }

        logger.info(f"Weight eval: mean_reward={results['mean_reward']:.3f}, "
                     f"mean_sharpe={results['mean_sharpe']:.3f}")
        return results


# =============================================================================
# MODEL LOADING UTILITY
# =============================================================================

def _load_weight_model(path: str) -> ContinuousActorCritic:
    """Load a ContinuousActorCritic model from disk."""
    config_path = path + ".json"
    weights_path = path + ".safetensors"

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    else:
        cfg = {'obs_dim': 103, 'n_assets': 12, 'hidden': 128}

    model = ContinuousActorCritic(
        obs_dim=cfg['obs_dim'],
        n_assets=cfg['n_assets'],
        hidden=cfg['hidden'],
    )

    if os.path.exists(weights_path):
        weights = mx.load(weights_path)
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        logger.info(f"Loaded weight model from {path}")
    else:
        raise FileNotFoundError(f"No weight model at {weights_path}")

    return model


# =============================================================================
# 1G. RL WEIGHT AGENT — Inference Wrapper
# =============================================================================

class RLWeightAgent:
    """
    Inference wrapper for the trained low-level weight model.

    Loads the trained weight model and provides get_weights() for BacktestEngine.
    """

    def __init__(self, model_path: Optional[str] = None, config=None):
        from alpha_engine import StrategyConfig
        self.config = config or StrategyConfig()
        self.model_path = model_path or os.path.join(WEIGHT_MODEL_DIR, "best_model")

        try:
            self.model = _load_weight_model(self.model_path)
            logger.info(f"Loaded weight RL model from {self.model_path}")
        except FileNotFoundError:
            self.model = None
            logger.warning(f"No weight RL model at {self.model_path}")

        self.obs_builder = WeightObservationBuilder()
        self.constraint_layer: Optional[ConstraintLayer] = None
        self.assets: List[str] = []

        # Portfolio tracking state
        self._portfolio_value = 100000.0
        self._peak_value = 100000.0
        self._current_weights = np.zeros(0)
        self._days_since_rebalance = 0

        # Tracking for diagnostics
        self._weight_history: List[np.ndarray] = []
        self._weight_dates: List[pd.Timestamp] = []
        self._turnover_history: List[float] = []
        self._safety_net_drift_history: List[float] = []

    def set_backtest_context(self, optimizer, asset_categories: Dict[str, List[str]]):
        self.assets = optimizer.assets
        self._current_weights = np.zeros(optimizer.n_assets)
        self.constraint_layer = ConstraintLayer(optimizer.assets, asset_categories, self.config)

    def update_portfolio_state(self, portfolio_value: float, current_weights: np.ndarray):
        self._portfolio_value = portfolio_value
        self._peak_value = max(self._peak_value, portfolio_value)
        self._current_weights = current_weights.copy()
        self._days_since_rebalance += 1

    def get_weights(
        self,
        regime: str,
        regime_action: int,
        date: pd.Timestamp,
        ml_prob: float,
        above_sma: pd.Series,
        golden_cross: pd.Series,
        rsi_14: pd.Series,
        raw_momentum: pd.Series,
        asset_volatilities: pd.Series,
        information_ratio: pd.Series,
        log_returns_30d: pd.Series,
    ) -> np.ndarray:
        """Get constrained weight vector from the RL model."""
        if self.model is None or self.constraint_layer is None:
            # Fallback: equal weight safe havens
            w = np.zeros(len(self.assets))
            sh_idx = self.constraint_layer.safe_haven_idx if self.constraint_layer else []
            for i in sh_idx:
                w[i] = 1.0 / max(len(sh_idx), 1)
            return w / (w.sum() + 1e-8)

        current_dd = (self._portfolio_value - self._peak_value) / self._peak_value if self._peak_value > 0 else 0.0

        obs = self.obs_builder.build(
            regime_action=regime_action,
            assets=self.assets,
            raw_momentum=raw_momentum,
            asset_volatilities=asset_volatilities,
            rsi_14=rsi_14,
            above_sma=above_sma,
            golden_cross=golden_cross,
            information_ratio=information_ratio,
            log_returns_30d=log_returns_30d,
            current_weights=self._current_weights,
            ml_prob=ml_prob,
            current_drawdown=current_dd,
            days_since_rebalance=self.config.rebalance_period,  # FIX: match training env (always rebalance_period)
            rebalance_period=self.config.rebalance_period,
            portfolio_value=self._portfolio_value,
            initial_capital=100000.0,
            noise_sigma=0.0,
        )

        raw_weights = self.model.predict_deterministic(obs)

        # Apply full constraint layer (eligibility, gold cap, crypto bounds, per-asset cap)
        clipped = self.constraint_layer.apply(raw_weights, regime, above_sma, golden_cross, rsi_14)

        # Track safety net drift BEFORE momentum tilt/GA floor (measures model's constraint learning)
        safety_net_drift = float(np.sum(np.abs(clipped - raw_weights)))
        self._safety_net_drift_history.append(safety_net_drift)
        if safety_net_drift > 0.05:
            logger.debug(f"Safety net drift: {safety_net_drift:.4f} (raw max={raw_weights.max():.3f})")

        # Momentum tilt in RISK_ON: concentrate into winners like baseline optimizer
        if regime == 'RISK_ON':
            mom_scores = np.array([max(raw_momentum.get(a, 0), 0) ** 2 for a in self.assets])
            if mom_scores.sum() > 0:
                mom_scores = mom_scores / mom_scores.sum()
                # 50% RL allocation + 50% momentum-proportional
                clipped = 0.5 * clipped + 0.5 * mom_scores
                # Re-apply constraints after momentum tilt
                clipped = self.constraint_layer.apply(clipped, regime, above_sma, golden_cross, rsi_14)

            # Growth anchor floor: ensure >= 40% in SMH/XBI/TAN/IGV (like baseline's 200x penalty)
            ga_idx = self.constraint_layer.growth_anchor_idx
            aligned_sma = above_sma.reindex(pd.Index(self.assets)).fillna(False)
            eligible_ga = [i for i in ga_idx if aligned_sma.values[i]]
            if eligible_ga:
                ga_weight = sum(clipped[i] for i in eligible_ga)
                if ga_weight < 0.40:
                    deficit = 0.40 - ga_weight
                    boost_each = deficit / len(eligible_ga)
                    for i in eligible_ga:
                        clipped[i] += boost_each
                    clipped = clipped / clipped.sum()

        # --- Lazy drift: keep current weight if per-asset change is tiny ---
        if self._current_weights.sum() > 0:
            drift = np.abs(clipped - self._current_weights)
            lazy_mask = drift < self.config.lazy_drift_threshold
            clipped[lazy_mask] = self._current_weights[lazy_mask]
            ws = clipped.sum()
            if ws > 0:
                clipped = clipped / ws

        # --- Turnover gate: skip rebalance if total turnover too small ---
        turnover = np.sum(np.abs(clipped - self._current_weights))
        if turnover < self.config.min_rebalance_threshold:
            clipped = self._current_weights.copy()
            turnover = 0.0

        # Track for diagnostics
        self._weight_history.append(clipped.copy())
        self._weight_dates.append(date)
        self._turnover_history.append(turnover)
        self._days_since_rebalance = 0

        return clipped


# =============================================================================
# 1H. HIERARCHICAL RL CONTROLLER
# =============================================================================

class HierarchicalRLController:
    """
    Combined controller wrapping both high-level (regime) and low-level (weight) agents.

    Provides the same interface as RLRegimeClassifier, plus get_weights() for
    duck-typing in BacktestEngine.run().
    """

    def __init__(self, config=None):
        from alpha_engine import StrategyConfig
        from rl_regime_agent import RLRegimeClassifier, REGIME_TO_ACTION

        self.config = config or StrategyConfig()
        self.regime_agent = RLRegimeClassifier(config=self.config)
        self.weight_agent = RLWeightAgent(config=self.config)
        self._regime_to_action = REGIME_TO_ACTION
        self._last_regime = 'RISK_ON'

        # Stubs for Streamlit UI compatibility (delegate to regime agent)
        self.model = self.regime_agent.model
        self.shap_values = None
        self.shap_features = None
        self.train_scores: List[float] = []
        self.test_scores: List[float] = []
        self.feature_importances_history: List[Dict] = []
        self.model_stability = 'HIERARCHICAL_RL'

    def set_backtest_context(self, above_sma, raw_momentum, asset_volatilities,
                             information_ratio, golden_cross, features, optimizer):
        self.regime_agent.set_backtest_context(
            above_sma=above_sma,
            raw_momentum=raw_momentum,
            asset_volatilities=asset_volatilities,
            information_ratio=information_ratio,
            golden_cross=golden_cross,
            features=features,
            optimizer=optimizer,
        )

        categories = {
            'equities': [a for a in ['QQQ', 'IWM', 'SMH', 'XBI', 'TAN', 'IGV'] if a in optimizer.assets],
            'fixed_income': [a for a in ['TLT', 'IEF', 'SHY'] if a in optimizer.assets],
            'alternatives': [a for a in ['GLD'] if a in optimizer.assets],
            'crypto': [a for a in ['BTC-USD', 'ETH-USD'] if a in optimizer.assets],
            'safe_haven': [a for a in ['GLD', 'TLT', 'IEF', 'SHY'] if a in optimizer.assets],
            'gold': ['GLD'] if 'GLD' in optimizer.assets else [],
            'bonds_cash': [a for a in ['TLT', 'IEF', 'SHY'] if a in optimizer.assets],
        }
        self.weight_agent.set_backtest_context(optimizer, categories)

    def update_portfolio_state(self, portfolio_value: float, benchmark_value: float,
                               current_weights: np.ndarray, date: pd.Timestamp):
        # FIX: Feed benchmark_value as "portfolio_value" to the regime agent so it
        # doesn't see the weight agent's tracking error and spiral into DEFENSIVE.
        self.regime_agent.update_portfolio_state(benchmark_value, benchmark_value, current_weights, date)
        self.weight_agent.update_portfolio_state(portfolio_value, current_weights)

    def get_regime(self, ml_prob: float, spy_above_sma: bool, current_vol: float,
                   tlt_momentum: float = 0.0, equity_risk_premium: float = 0.0) -> str:
        """Baseline regime logic — SPY > 200-SMA = RISK_ON (master switch).

        The RL regime agent is bypassed because its 71% DEFENSIVE bias
        prevents equity participation in bull markets. The RL weight agent
        still controls allocation, preserving diversity and cost advantages.
        """
        if spy_above_sma:
            regime = 'RISK_ON'
        elif ml_prob > 0.55:
            regime = 'RISK_REDUCED'
        else:
            regime = 'DEFENSIVE'
        self._last_regime = regime
        return regime

    def get_weights(
        self,
        date: pd.Timestamp,
        ml_prob: float,
        above_sma: pd.Series,
        golden_cross: pd.Series,
        rsi_14: pd.Series,
        raw_momentum: pd.Series,
        asset_volatilities: pd.Series,
        information_ratio: pd.Series,
        log_returns_30d: pd.Series,
    ) -> np.ndarray:
        """Get weights from the low-level RL agent (called via duck-typing from BacktestEngine)."""
        # Use regime from get_regime() (baseline logic, not RL agent)
        regime_str = getattr(self, '_last_regime', 'RISK_ON')
        regime_action = self._regime_to_action.get(regime_str, 0)

        return self.weight_agent.get_weights(
            regime=regime_str,
            regime_action=regime_action,
            date=date,
            ml_prob=ml_prob,
            above_sma=above_sma,
            golden_cross=golden_cross,
            rsi_14=rsi_14,
            raw_momentum=raw_momentum,
            asset_volatilities=asset_volatilities,
            information_ratio=information_ratio,
            log_returns_30d=log_returns_30d,
        )

    def walk_forward_train(self, features, returns, initial_train_years=5, step_months=12):
        return pd.Series(0.5, index=features.index)

    def get_shap_figure(self):
        return self.regime_agent.get_shap_figure()

    def get_validation_curves_figure(self):
        return self.regime_agent.get_validation_curves_figure()

    def get_weight_diagnostics_figure(self) -> Optional[plt.Figure]:
        """Generate average weight bar chart for diagnostics."""
        wh = self.weight_agent._weight_history
        if not wh:
            return None

        avg_weights = np.mean(wh, axis=0)
        assets = self.weight_agent.assets

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = []
        for a in assets:
            if a in ('BTC-USD', 'ETH-USD'):
                colors.append('#f39c12')
            elif a in ('TLT', 'IEF', 'SHY', 'GLD'):
                colors.append('#3498db')
            else:
                colors.append('#2ecc71')

        bars = ax.bar(range(len(assets)), avg_weights, color=colors)
        ax.set_xticks(range(len(assets)))
        ax.set_xticklabels(assets, rotation=45, ha='right')
        ax.set_ylabel('Average Weight')
        ax.set_title('RL Weight Agent — Average Portfolio Allocation')
        ax.set_ylim(0, max(avg_weights) * 1.3 if max(avg_weights) > 0 else 0.5)

        for bar, w in zip(bars, avg_weights):
            if w > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{w:.1%}', ha='center', va='bottom', fontsize=9)

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig

    def get_weight_history_figure(self) -> Optional[plt.Figure]:
        """Generate stacked area chart of weight allocations over time."""
        wh = self.weight_agent._weight_history
        dates = self.weight_agent._weight_dates
        if not wh or len(wh) < 2:
            return None

        assets = self.weight_agent.assets
        weight_arr = np.array(wh)  # (n_rebalances, n_assets)

        # Distinct color per asset — maximally separable in stacked area
        asset_colors = {
            'QQQ': '#1f77b4',   # blue
            'IWM': '#ff7f0e',   # orange
            'SMH': '#2ca02c',   # green
            'XBI': '#d62728',   # red
            'TAN': '#9467bd',   # purple
            'IGV': '#8c564b',   # brown
            'TLT': '#e377c2',   # pink
            'IEF': '#7f7f7f',   # grey
            'SHY': '#bcbd22',   # yellow-green
            'GLD': '#FFD700',   # gold
            'BTC-USD': '#ff6600', # bright orange
            'ETH-USD': '#17becf', # cyan
        }
        colors = [asset_colors.get(a, '#333333') for a in assets]

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.stackplot(dates, weight_arr.T, labels=assets, colors=colors, alpha=0.85)
        ax.set_ylabel('Portfolio Weight')
        ax.set_ylim(0, 1)
        ax.set_title('RL Weight Agent — Allocation Over Time')
        ax.legend(loc='upper left', fontsize=8, ncol=4, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        fig.autofmt_xdate()
        plt.tight_layout()
        return fig

    def get_turnover_stats(self) -> Dict:
        """Get turnover statistics for diagnostics."""
        th = self.weight_agent._turnover_history
        if not th:
            return {}
        return {
            'mean_turnover': float(np.mean(th)),
            'median_turnover': float(np.median(th)),
            'max_turnover': float(np.max(th)),
            'total_rebalances': len(th),
        }

    def get_safety_net_stats(self) -> Dict:
        """Get safety net drift statistics for Streamlit diagnostics."""
        dh = self.weight_agent._safety_net_drift_history
        if not dh:
            return {}
        dh_arr = np.array(dh)
        return {
            'mean_drift': float(np.mean(dh_arr)),
            'max_drift': float(np.max(dh_arr)),
            'pct_adjusted': float(np.mean(dh_arr > 0.01) * 100),
            'total_rebalances': len(dh),
        }


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("=" * 70)
    print("RL Weight Agent — Standalone Test (MLX Backend)")
    print("=" * 70)

    print("\n[1/4] Loading market data...", flush=True)
    from alpha_engine import DataManager, StrategyConfig, AlphaDominatorOptimizer

    config = StrategyConfig()
    dm = DataManager(start_date='2010-01-01', config=config)
    dm.load_data()
    dm.engineer_features()

    (prices, returns, features, vix, sma_200, above_sma, raw_mom, rel_strength,
     vols, info_ratio, mom_score, golden_cross_df, log_ret_30d, rsi_14) = dm.get_aligned_data()
    categories = dm.get_asset_categories()
    optimizer = AlphaDominatorOptimizer(dm.all_tickers, categories, config)

    print(f"   Data: {len(prices)} days, {len(dm.all_tickers)} assets")

    print("\n[2/4] Building WeightBacktestEnv...", flush=True)
    env = WeightBacktestEnv(
        prices=prices, returns=returns, features=features,
        ml_probs=pd.Series(0.5, index=features.index),
        sma_200=sma_200, above_sma=above_sma, raw_momentum=raw_mom,
        relative_strength=rel_strength, asset_volatilities=vols,
        information_ratio=info_ratio, momentum_score=mom_score,
        golden_cross=golden_cross_df, log_returns_30d=log_ret_30d, rsi_14=rsi_14,
        optimizer=optimizer, config=config,
        train_end_date="2020-12-31", noise_sigma=0.02,
    )
    print(f"   Rebalance points: {len(env.rebalance_indices)}")

    print("\n[3/4] Running manual episode (random weights)...", flush=True)
    obs, info = env.reset()
    assert obs.shape == (103,), f"Bad obs shape: {obs.shape}"
    total_reward = 0.0
    step_count = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        pv = info.get('portfolio_value', 0)
        print(f"   Step {step_count:2d}: regime={info.get('regime', '?'):14s}, "
              f"reward={reward:+.3f}, PV=${pv:>10,.0f}")
        if terminated or truncated:
            break

    print(f"\n   Steps: {step_count}, Total reward: {total_reward:.3f}")
    print(f"   Final PV: ${env._portfolio_value:,.0f}")

    if '--train' in sys.argv:
        ts = 50_000
        idx = sys.argv.index('--train')
        if idx + 1 < len(sys.argv):
            try:
                ts = int(sys.argv[idx + 1])
            except ValueError:
                pass

        print(f"\n[4/4] Training weight PPO ({ts:,} timesteps)...", flush=True)
        trainer = WeightRLTrainer(
            prices=prices, returns=returns, features=features,
            ml_probs=pd.Series(0.5, index=features.index),
            sma_200=sma_200, above_sma=above_sma, raw_momentum=raw_mom,
            relative_strength=rel_strength, asset_volatilities=vols,
            information_ratio=info_ratio, momentum_score=mom_score,
            golden_cross=golden_cross_df, log_returns_30d=log_ret_30d, rsi_14=rsi_14,
            optimizer=optimizer, config=config,
            train_end_date="2020-12-31",
            total_timesteps=ts,
        )
        model = trainer.train()

        print("\nEvaluating...", flush=True)
        eval_results = trainer.evaluate(n_episodes=5)
        print(f"  Mean reward: {eval_results['mean_reward']:.3f}")
        print(f"  Mean Sharpe: {eval_results['mean_sharpe']:.3f}")
        print(f"  Final PV:    ${eval_results['final_portfolio_value']:,.0f}")
    else:
        print("\n[4/4] SKIPPED: pass --train to run training")

    print("\n" + "=" * 70)
    print("Test complete.")
    print("=" * 70)
