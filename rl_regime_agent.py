#!/usr/bin/env python3
"""
RL Regime Selector — PPO Agent for Adaptive Regime Classification
=================================================================
Step 1 of the RL Roadmap (A -> D -> C).

Replaces hand-crafted regime rules (SPY>200-SMA = RISK_ON) with a trained
PPO reinforcement learning agent that learns optimal regime-switching from data.

Uses Apple MLX for native Apple Silicon acceleration (no PyTorch dependency).

Classes:
    RewardFunction          - Abstract base for reward computation
    DifferentialSharpeReward - Excess-Sharpe reward with drawdown penalty
    ObservationBuilder      - Builds 25-dim normalized observation vectors
    RegimeBacktestEnv       - Gymnasium environment wrapping the backtest loop
    ActorCritic             - MLX actor-critic network for PPO
    RolloutBuffer           - Experience buffer for PPO rollout collection
    RLTrainer               - PPO training with walk-forward validation
    RLRegimeClassifier      - Drop-in replacement for AdaptiveRegimeClassifier
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# MLX imports
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Default model save directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rl_regime_ppo")

REGIME_MAP = {0: 'RISK_ON', 1: 'RISK_REDUCED', 2: 'DEFENSIVE'}

OBS_FEATURE_NAMES = [
    'realized_vol', 'vol_momentum', 'equity_risk_premium', 'trend_score',
    'momentum_21d', 'qqq_vs_spy', 'tlt_momentum',
    'equity_weight', 'safe_haven_weight', 'crypto_weight',
    'norm_portfolio_value', 'norm_days_since_rebal', 'current_drawdown',
    'mean_equity_momentum', 'pct_above_sma', 'mean_equity_vol',
    'btc_above_50d', 'tlt_above_sma', 'mean_equity_ir', 'ml_prob',
    'portfolio_ret_5d', 'portfolio_ret_21d', 'portfolio_ret_63d',
    'benchmark_ret_21d', 'excess_ret_21d',
]
REGIME_TO_ACTION = {v: k for k, v in REGIME_MAP.items()}


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

class RewardFunction(ABC):
    """Abstract base class for RL reward computation (swappable for Choice B/C/D)."""

    @abstractmethod
    def compute(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray,
                portfolio_value: float, peak_value: float) -> float:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class DifferentialSharpeReward(RewardFunction):
    """
    Reward = excess_sharpe_over_benchmark - drawdown_penalty.

    Excess Sharpe = (port_window_ret - bench_window_ret) / port_window_vol
    Drawdown penalty activates beyond -15% drawdown.
    Rewards clipped to [-3, 3].
    """

    DRAWDOWN_THRESHOLD = -0.15
    DRAWDOWN_PENALTY_SCALE = 5.0
    CLIP_LO, CLIP_HI = -3.0, 3.0

    def __init__(self):
        self._prev_peak = None

    def reset(self) -> None:
        self._prev_peak = None

    def compute(self, portfolio_returns: np.ndarray, benchmark_returns: np.ndarray,
                portfolio_value: float, peak_value: float) -> float:
        if len(portfolio_returns) < 2:
            return 0.0

        port_ret = np.nansum(portfolio_returns)
        bench_ret = np.nansum(benchmark_returns)
        port_vol = np.nanstd(portfolio_returns) * np.sqrt(252) + 1e-8

        excess_sharpe = (port_ret - bench_ret) / port_vol

        drawdown = (portfolio_value - peak_value) / peak_value if peak_value > 0 else 0.0
        dd_penalty = 0.0
        if drawdown < self.DRAWDOWN_THRESHOLD:
            dd_penalty = self.DRAWDOWN_PENALTY_SCALE * (self.DRAWDOWN_THRESHOLD - drawdown)

        reward = excess_sharpe - dd_penalty
        return float(np.clip(reward, self.CLIP_LO, self.CLIP_HI))


# =============================================================================
# OBSERVATION BUILDER
# =============================================================================

class ObservationBuilder:
    """
    Builds a 25-dimensional normalized observation vector per decision point.

    Dimensions (25 total):
        [0:7]   7 ML features (from engineer_features())
        [7:13]  6 portfolio-aggregate features
        [13:20] 7 cross-asset summaries
        [20:25] 5 recent performance features

    All normalized to ~[-1, 1] via online z-score standardization.
    """

    OBS_DIM = 25

    def __init__(self):
        self._running_mean = np.zeros(self.OBS_DIM)
        self._running_var = np.ones(self.OBS_DIM)
        self._count = 0

    def reset(self) -> None:
        self._running_mean = np.zeros(self.OBS_DIM)
        self._running_var = np.ones(self.OBS_DIM)
        self._count = 0

    def build(
        self,
        features_row: pd.Series,
        current_weights: np.ndarray,
        equity_indices: List[int],
        safe_haven_indices: List[int],
        crypto_indices: List[int],
        portfolio_value: float,
        initial_capital: float,
        days_since_rebalance: int,
        rebalance_period: int,
        current_drawdown: float,
        above_sma: pd.Series,
        raw_momentum: pd.Series,
        asset_volatilities: pd.Series,
        information_ratio: pd.Series,
        ml_prob: float,
        golden_cross: pd.Series,
        equity_tickers: List[str],
        portfolio_returns_5d: float,
        portfolio_returns_21d: float,
        portfolio_returns_63d: float,
        benchmark_returns_21d: float,
        noise_sigma: float = 0.0,
    ) -> np.ndarray:
        """Build and return the 25-dim observation vector."""

        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # --- Block 1: 7 ML features ---
        obs[0:7] = [
            features_row.get('realized_vol', 0.0),
            features_row.get('vol_momentum', 0.0),
            features_row.get('equity_risk_premium', 0.0),
            features_row.get('trend_score', 0.0) / 100.0,
            features_row.get('momentum_21d', 0.0),
            features_row.get('qqq_vs_spy', 0.0),
            features_row.get('tlt_momentum', 0.0),
        ]

        # --- Block 2: 6 portfolio-aggregate features ---
        total_w = current_weights.sum() if current_weights.sum() > 0 else 1.0
        eq_weight = sum(current_weights[i] for i in equity_indices) / total_w if equity_indices else 0.0
        sh_weight = sum(current_weights[i] for i in safe_haven_indices) / total_w if safe_haven_indices else 0.0
        cr_weight = sum(current_weights[i] for i in crypto_indices) / total_w if crypto_indices else 0.0

        norm_pv = np.clip((portfolio_value / initial_capital - 1.0), -1.0, 5.0) / 5.0
        norm_days = days_since_rebalance / max(rebalance_period, 1)
        norm_dd = np.clip(current_drawdown, -1.0, 0.0)

        obs[7:13] = [eq_weight, sh_weight, cr_weight, norm_pv, norm_days, norm_dd]

        # --- Block 3: 7 cross-asset summaries ---
        eq_mom_vals = [raw_momentum.get(t, 0.0) for t in equity_tickers if t in raw_momentum.index]
        mean_eq_mom = float(np.nanmean(eq_mom_vals)) if eq_mom_vals else 0.0

        eq_above = [float(above_sma.get(t, False)) for t in equity_tickers if t in above_sma.index]
        pct_above_sma = float(np.nanmean(eq_above)) if eq_above else 0.5

        eq_vols = [asset_volatilities.get(t, 0.0) for t in equity_tickers if t in asset_volatilities.index]
        mean_vol = float(np.nanmean(eq_vols)) if eq_vols else 0.2

        btc_above_50 = float(golden_cross.get('BTC-USD', False)) if 'BTC-USD' in golden_cross.index else 0.5
        tlt_above = float(above_sma.get('TLT', False)) if 'TLT' in above_sma.index else 0.5

        eq_ir = [information_ratio.get(t, 0.0) for t in equity_tickers if t in information_ratio.index]
        mean_ir = np.clip(float(np.nanmean(eq_ir)) if eq_ir else 0.0, -2, 2) / 2.0

        obs[13:20] = [
            np.clip(mean_eq_mom, -0.5, 0.5),
            pct_above_sma,
            np.clip(mean_vol, 0, 1),
            btc_above_50,
            tlt_above,
            mean_ir,
            np.clip(ml_prob, 0, 1),
        ]

        # --- Block 4: 5 recent performance features ---
        obs[20:25] = [
            np.clip(portfolio_returns_5d, -0.3, 0.3) / 0.3,
            np.clip(portfolio_returns_21d, -0.5, 0.5) / 0.5,
            np.clip(portfolio_returns_63d, -1.0, 1.0),
            np.clip(benchmark_returns_21d, -0.5, 0.5) / 0.5,
            np.clip(portfolio_returns_21d - benchmark_returns_21d, -0.3, 0.3) / 0.3,
        ]

        # Online z-score normalization (Welford's algorithm)
        self._count += 1
        delta = obs - self._running_mean
        self._running_mean += delta / self._count
        delta2 = obs - self._running_mean
        self._running_var += (delta * delta2 - self._running_var) / self._count

        std = np.sqrt(self._running_var + 1e-8)
        obs_normalized = np.clip((obs - self._running_mean) / std, -3.0, 3.0).astype(np.float32)

        if noise_sigma > 0:
            obs_normalized += np.random.normal(0, noise_sigma, size=self.OBS_DIM).astype(np.float32)

        return obs_normalized


# =============================================================================
# GYMNASIUM ENVIRONMENT
# =============================================================================

class RegimeBacktestEnv(gym.Env):
    """
    Gymnasium environment for RL regime selection.

    Action space:  Discrete(3) — {0: RISK_ON, 1: RISK_REDUCED, 2: DEFENSIVE}
    Observation:   Box(25,) — 25-dim normalized observation vector
    Episode:       Full backtest period (~40-50 regime decisions)
    """

    metadata = {"render_modes": []}

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
        reward_function: Optional[RewardFunction] = None,
        lookback_days: int = 252,
    ):
        super().__init__()

        from alpha_engine import StrategyConfig, DataManager, BacktestEngine

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

        # Store aligned data
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

        # Compute valid dates
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

        # SPY returns for benchmark tracking
        spy_ticker = DataManager.BENCHMARK_TICKER
        self.spy_returns = returns[spy_ticker] if spy_ticker in returns.columns else pd.Series(0.0, index=returns.index)

        # Transaction costs per asset
        self.asset_cost_bps = BacktestEngine._get_asset_cost_bps(optimizer.assets)

        # Pre-compute rebalance schedule
        self.rebalance_period = self.config.rebalance_period
        self.start_idx = self.lookback_days
        self._precompute_rebalance_schedule()

        # Reward function
        self.reward_fn = reward_function or DifferentialSharpeReward()

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(ObservationBuilder.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Episode state
        self.obs_builder = ObservationBuilder()
        self._episode_step = 0
        self._portfolio_value = 100000.0
        self._benchmark_value = 100000.0
        self._peak_value = 100000.0
        self._current_weights = np.zeros(optimizer.n_assets)
        self._current_day_idx = self.start_idx
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
        self._current_day_idx = self.start_idx
        self._portfolio_values_history = [100000.0]
        self._benchmark_values_history = [100000.0]
        self.obs_builder.reset()
        self.reward_fn.reset()

        obs = self._get_observation()
        return obs, {}

    def step(self, action: int):
        if self._episode_step >= len(self.rebalance_indices):
            return self._get_observation(), 0.0, True, False, {}

        rebal_idx = self.rebalance_indices[self._episode_step]
        regime = REGIME_MAP[action]

        # --- 1. Optimize with chosen regime ---
        date = self.valid_dates[rebal_idx]
        prev_date = self.valid_dates[rebal_idx - 1]
        lookback_start = self.valid_dates[rebal_idx - self.lookback_days]
        lookback_ret = self.returns.loc[lookback_start:prev_date][self.optimizer.assets]

        ml_prob_val = self.ml_probs.loc[date]
        if np.isnan(ml_prob_val):
            ml_prob_val = 0.5

        new_weights, success, method, diagnostics = self.optimizer.optimize(
            lookback_ret,
            self.raw_momentum.loc[date],
            self.information_ratio.loc[date],
            self.asset_volatilities.loc[date],
            regime,
            self.above_sma.loc[date],
            ml_prob_val,
            self.momentum_score.loc[date],
            self.golden_cross.loc[date],
            self.log_returns_30d.loc[date],
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

        # --- 3. Turnover gate + tiered transaction costs ---
        turnover = np.sum(np.abs(new_weights - self._current_weights))
        if turnover >= self.config.min_rebalance_threshold:
            per_asset_turnover = np.abs(new_weights - self._current_weights)
            cost = self._portfolio_value * np.sum(per_asset_turnover * self.asset_cost_bps / 10000)
            self._portfolio_value -= cost
            self._current_weights = new_weights

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
            port_daily_ret = np.dot(self._current_weights, daily_ret)
            self._portfolio_value *= (1 + port_daily_ret)
            window_port_returns.append(port_daily_ret)

            spy_ret = self.spy_returns.loc[day] if day in self.spy_returns.index else 0.0
            self._benchmark_value *= (1 + spy_ret)
            window_bench_returns.append(spy_ret)

            self._peak_value = max(self._peak_value, self._portfolio_value)
            self._portfolio_values_history.append(self._portfolio_value)
            self._benchmark_values_history.append(self._benchmark_value)

        # --- 5. Compute reward ---
        reward = self.reward_fn.compute(
            np.array(window_port_returns),
            np.array(window_bench_returns),
            self._portfolio_value,
            self._peak_value,
        )

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
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        if self._episode_step >= len(self.rebalance_indices):
            return np.zeros(ObservationBuilder.OBS_DIM, dtype=np.float32)

        idx = self.rebalance_indices[self._episode_step]
        date = self.valid_dates[idx]

        ml_prob_val = self.ml_probs.loc[date]
        if np.isnan(ml_prob_val):
            ml_prob_val = 0.5

        pv_arr = np.array(self._portfolio_values_history)
        bv_arr = np.array(self._benchmark_values_history)

        def _trailing_return(arr, n):
            if len(arr) < n + 1:
                return 0.0
            return (arr[-1] / arr[-n - 1]) - 1.0

        current_dd = (self._portfolio_value - self._peak_value) / self._peak_value if self._peak_value > 0 else 0.0

        return self.obs_builder.build(
            features_row=self.features.loc[date],
            current_weights=self._current_weights,
            equity_indices=self.equity_indices,
            safe_haven_indices=self.safe_haven_indices,
            crypto_indices=self.crypto_indices,
            portfolio_value=self._portfolio_value,
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
            portfolio_returns_5d=_trailing_return(pv_arr, 5),
            portfolio_returns_21d=_trailing_return(pv_arr, 21),
            portfolio_returns_63d=_trailing_return(pv_arr, 63),
            benchmark_returns_21d=_trailing_return(bv_arr, 21),
            noise_sigma=self.noise_sigma,
        )


# =============================================================================
# MLX ACTOR-CRITIC NETWORK
# =============================================================================

class ActorCritic(nn.Module):
    """
    MLX actor-critic network for PPO.

    Architecture: shared 2x64 MLP with separate policy and value heads.
    Policy head outputs logits for Discrete(3) action space.
    Value head outputs scalar state value.
    """

    def __init__(self, obs_dim: int = 25, n_actions: int = 3, hidden: int = 64):
        super().__init__()
        self.shared1 = nn.Linear(obs_dim, hidden)
        self.shared2 = nn.Linear(hidden, hidden)
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Returns (action_logits, state_value)."""
        h = nn.tanh(self.shared1(x))
        h = nn.tanh(self.shared2(h))
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs_np: np.ndarray) -> Tuple[int, float, float]:
        """Numpy in, numpy out — for env interaction."""
        x = mx.array(obs_np.reshape(1, -1))
        logits, value = self(x)
        mx.eval(logits, value)

        logits_np = np.array(logits[0])
        # Softmax for probabilities
        logits_np = logits_np - logits_np.max()
        probs = np.exp(logits_np) / np.exp(logits_np).sum()
        action = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action] + 1e-8)

        return int(action), float(log_prob), float(np.array(value[0]))

    def predict_deterministic(self, obs_np: np.ndarray) -> int:
        """Deterministic action selection (argmax policy)."""
        x = mx.array(obs_np.reshape(1, -1))
        logits, _ = self(x)
        mx.eval(logits)
        return int(np.array(logits[0]).argmax())


# =============================================================================
# ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    """Stores experience tuples for PPO training."""

    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(self, obs: np.ndarray, action: int, log_prob: float,
            reward: float, value: float, done: bool):
        self.observations.append(obs.copy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
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
# RL TRAINER (MLX PPO)
# =============================================================================

class RLTrainer:
    """
    PPO training manager using MLX.

    Hyperparameters:
        lr=3e-4, n_steps=128, batch_size=64, gamma=0.99,
        ent_coef=0.05, net_arch=[64, 64]

    Walk-forward split: train 2011-2020, validate 2020-present.
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
        total_timesteps: int = 100_000,
        model_dir: str = MODEL_DIR,
    ):
        from alpha_engine import StrategyConfig
        self.config = config or StrategyConfig()
        self.train_end_date = train_end_date
        self.total_timesteps = total_timesteps
        self.model_dir = model_dir

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

        self.model: Optional[ActorCritic] = None
        self.reward_history: List[float] = []

    def _make_env(self, train: bool = True) -> RegimeBacktestEnv:
        if train:
            return RegimeBacktestEnv(
                **self._data_kwargs,
                train_end_date='2022-12-31',
                noise_sigma=0.02,
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
            return RegimeBacktestEnv(
                **val_kwargs,
                train_end_date=None,
                noise_sigma=0.0,
            )

    def train(self) -> ActorCritic:
        """Run PPO training with MLX."""
        # Hyperparameters
        lr = 3e-4
        n_steps = 128
        batch_size = 64
        gamma = 0.99
        gae_lambda = 0.95
        clip_range = 0.2
        ent_coef = 0.05
        vf_coef = 0.5
        max_grad_norm = 0.5
        n_epochs = 10
        eval_freq = max(self.total_timesteps // 20, 500)

        logger.info(f"Starting MLX PPO training: {self.total_timesteps} timesteps, train_end={self.train_end_date}")
        print(f"Using MLX backend (Apple Silicon native)", flush=True)

        train_env = self._make_env(train=True)
        val_env = self._make_env(train=False)
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize network and optimizer
        self.model = ActorCritic(obs_dim=ObservationBuilder.OBS_DIM, n_actions=3, hidden=64)
        mx.eval(self.model.parameters())
        optimizer_mlx = optim.Adam(learning_rate=lr)

        # PPO loss function
        def ppo_loss(model, obs_batch, act_batch, old_logprob_batch,
                     advantage_batch, return_batch):
            logits, values = model(obs_batch)

            # Policy loss (clipped surrogate)
            log_probs_all = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            new_log_probs = mx.take_along_axis(log_probs_all, act_batch.reshape(-1, 1), axis=1).squeeze(-1)

            ratio = mx.exp(new_log_probs - old_logprob_batch)
            surr1 = ratio * advantage_batch
            surr2 = mx.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage_batch
            policy_loss = -mx.minimum(surr1, surr2).mean()

            # Value loss
            value_loss = ((values - return_batch) ** 2).mean()

            # Entropy bonus
            probs = mx.softmax(logits, axis=-1)
            entropy = -(probs * log_probs_all).sum(axis=-1).mean()

            total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            return total_loss

        loss_and_grad = nn.value_and_grad(self.model, ppo_loss)

        # Training loop
        buffer = RolloutBuffer()
        obs, _ = train_env.reset()
        episode_reward = 0.0
        total_steps = 0
        best_eval_reward = -float('inf')
        episode_count = 0

        while total_steps < self.total_timesteps:
            # --- Collect n_steps of experience ---
            buffer.clear()
            for _ in range(n_steps):
                action, log_prob, value = self.model.get_action_and_value(obs)
                next_obs, reward, terminated, truncated, info = train_env.step(action)
                done = terminated or truncated

                buffer.add(obs, action, log_prob, reward, value, done)
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

            # --- Compute advantages ---
            _, _, last_value = self.model.get_action_and_value(obs)
            returns, advantages = buffer.compute_returns_and_advantages(last_value, gamma, gae_lambda)

            # Normalize advantages
            adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

            # --- PPO update ---
            obs_arr = np.array(buffer.observations, dtype=np.float32)
            act_arr = np.array(buffer.actions, dtype=np.int32)
            old_lp_arr = np.array(buffer.log_probs, dtype=np.float32)
            buf_len = len(buffer)

            for epoch in range(n_epochs):
                indices = np.random.permutation(buf_len)
                for start in range(0, buf_len, batch_size):
                    end = min(start + batch_size, buf_len)
                    mb_idx = indices[start:end]

                    mb_obs = mx.array(obs_arr[mb_idx])
                    mb_act = mx.array(act_arr[mb_idx])
                    mb_old_lp = mx.array(old_lp_arr[mb_idx])
                    mb_adv = mx.array(advantages[mb_idx])
                    mb_ret = mx.array(returns[mb_idx])

                    loss, grads = loss_and_grad(
                        self.model, mb_obs, mb_act, mb_old_lp, mb_adv, mb_ret
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
                        grads = mlx_utils.tree_map(lambda g: g * scale if isinstance(g, mx.array) else g, grads)

                    optimizer_mlx.update(self.model, grads)
                    mx.eval(self.model.parameters(), optimizer_mlx.state)

            # --- Logging ---
            if len(self.reward_history) > 0:
                recent = self.reward_history[-max(episode_count, 1):]
                mean_rew = np.mean(recent) if recent else 0.0
            else:
                mean_rew = 0.0

            iteration = total_steps // n_steps
            print(f"| iter {iteration:4d} | timesteps {total_steps:6d}/{self.total_timesteps} "
                  f"| episodes {episode_count:4d} | mean_reward {mean_rew:+7.2f} |", flush=True)

            # --- Periodic evaluation ---
            if total_steps % eval_freq < n_steps or total_steps >= self.total_timesteps:
                eval_reward = self._run_eval(val_env, n_episodes=5)
                print(f"  [EVAL] mean_reward={eval_reward:.3f} (best={best_eval_reward:.3f})", flush=True)

                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self._save_model(os.path.join(self.model_dir, "best_model"))
                    print(f"  [EVAL] New best model saved!", flush=True)

        # Save final model
        self._save_model(os.path.join(self.model_dir, "final_model"))
        logger.info(f"Training complete. {episode_count} episodes, {total_steps} timesteps.")
        return self.model

    def _run_eval(self, env: RegimeBacktestEnv, n_episodes: int = 3) -> float:
        """Run deterministic evaluation episodes."""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.model.predict_deterministic(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            rewards.append(total_reward)
        return float(np.mean(rewards))

    def _save_model(self, path: str):
        """Save MLX model weights and architecture config."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        weights_path = path + ".safetensors"
        config_path = path + ".json"
        import mlx.utils as mlx_utils
        mx.save_safetensors(weights_path, dict(mlx_utils.tree_flatten(self.model.parameters())))
        with open(config_path, 'w') as f:
            json.dump({'obs_dim': 25, 'n_actions': 3, 'hidden': 64}, f)
        logger.info(f"Model saved to {path}")

    def evaluate(self, n_episodes: int = 5) -> Dict:
        """Run deterministic evaluation episodes and collect metrics."""
        if self.model is None:
            self.model = _load_mlx_model(os.path.join(self.model_dir, "best_model"))

        val_env = self._make_env(train=False)

        episode_returns = []
        episode_sharpes = []
        regime_counts = {r: 0 for r in REGIME_MAP.values()}

        for _ in range(n_episodes):
            obs, _ = val_env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action = self.model.predict_deterministic(obs)
                obs, reward, terminated, truncated, info = val_env.step(action)
                total_reward += reward
                if 'regime' in info:
                    regime_counts[info['regime']] += 1
                done = terminated or truncated

            episode_returns.append(total_reward)

            pv = np.array(val_env._portfolio_values_history)
            daily_rets = np.diff(pv) / pv[:-1] if len(pv) > 1 else np.array([0.0])
            sharpe = (np.mean(daily_rets) * 252 - self.config.risk_free_rate) / (np.std(daily_rets) * np.sqrt(252) + 1e-8)
            episode_sharpes.append(sharpe)

        total_regime_actions = max(sum(regime_counts.values()), 1)
        results = {
            'mean_reward': float(np.mean(episode_returns)),
            'std_reward': float(np.std(episode_returns)),
            'mean_sharpe': float(np.mean(episode_sharpes)),
            'regime_distribution': {k: v / total_regime_actions for k, v in regime_counts.items()},
            'final_portfolio_value': val_env._portfolio_value,
            'final_benchmark_value': val_env._benchmark_value,
        }

        logger.info(f"Evaluation: mean_reward={results['mean_reward']:.3f}, "
                     f"mean_sharpe={results['mean_sharpe']:.3f}, "
                     f"regime_dist={results['regime_distribution']}")
        return results


# =============================================================================
# MODEL LOADING UTILITY
# =============================================================================

def _load_mlx_model(path: str) -> ActorCritic:
    """Load an MLX ActorCritic model from disk."""
    config_path = path + ".json"
    weights_path = path + ".safetensors"

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    else:
        cfg = {'obs_dim': 25, 'n_actions': 3, 'hidden': 64}

    model = ActorCritic(obs_dim=cfg['obs_dim'], n_actions=cfg['n_actions'], hidden=cfg['hidden'])

    if os.path.exists(weights_path):
        weights = mx.load(weights_path)
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        logger.info(f"Loaded MLX model from {path}")
    else:
        raise FileNotFoundError(f"No model weights at {weights_path}")

    return model


# =============================================================================
# DROP-IN REGIME CLASSIFIER
# =============================================================================

class RLRegimeClassifier:
    """
    Drop-in replacement for AdaptiveRegimeClassifier.

    Same get_regime() interface. Loads a trained MLX PPO model and builds
    observations from internal state + arguments.

    Lifecycle:
        1. __init__()            — loads the MLX model from disk
        2. set_backtest_context() — called once before engine.run() to bind data
        3. update_portfolio_state() — called every day from BacktestEngine hook
        4. get_regime()          — called at rebalance points by BacktestEngine
    """

    def __init__(self, model_path: Optional[str] = None, config=None):
        from alpha_engine import StrategyConfig
        self.config = config or StrategyConfig()
        self.model_path = model_path or os.path.join(MODEL_DIR, "best_model")

        try:
            self.model = _load_mlx_model(self.model_path)
            logger.info(f"Loaded RL model from {self.model_path}")
        except FileNotFoundError:
            self.model = None
            logger.warning(f"No RL model found at {self.model_path} — will fall back to rule-based")

        self.obs_builder = ObservationBuilder()

        # Portfolio tracking state
        self._portfolio_value = 100000.0
        self._benchmark_value = 100000.0
        self._peak_value = 100000.0
        self._current_weights = np.zeros(0)
        self._portfolio_values_history: List[float] = [100000.0]
        self._benchmark_values_history: List[float] = [100000.0]
        self._days_since_rebalance = 0
        self._current_date: Optional[pd.Timestamp] = None

        # Data references (set via set_backtest_context)
        self._above_sma: Optional[pd.DataFrame] = None
        self._raw_momentum: Optional[pd.DataFrame] = None
        self._asset_volatilities: Optional[pd.DataFrame] = None
        self._information_ratio: Optional[pd.DataFrame] = None
        self._golden_cross: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
        self._equity_indices: List[int] = []
        self._safe_haven_indices: List[int] = []
        self._crypto_indices: List[int] = []
        self._equity_tickers: List[str] = []

        # Observation/action history for permutation importance
        self._obs_history: List[np.ndarray] = []
        self._action_history: List[int] = []

        # Stubs for Streamlit UI compatibility
        self.shap_values = None
        self.shap_features = None
        self.train_scores: List[float] = []
        self.test_scores: List[float] = []
        self.feature_importances_history: List[Dict] = []
        self.model_stability = 'RL_AGENT'

    def set_backtest_context(
        self,
        above_sma: pd.DataFrame,
        raw_momentum: pd.DataFrame,
        asset_volatilities: pd.DataFrame,
        information_ratio: pd.DataFrame,
        golden_cross: pd.DataFrame,
        features: pd.DataFrame,
        optimizer,
    ):
        from alpha_engine import DataManager
        self._above_sma = above_sma
        self._raw_momentum = raw_momentum
        self._asset_volatilities = asset_volatilities
        self._information_ratio = information_ratio
        self._golden_cross = golden_cross
        self._features = features
        self._current_weights = np.zeros(optimizer.n_assets)
        self._equity_indices = list(optimizer.equity_idx)
        self._safe_haven_indices = list(optimizer.safe_haven_idx)
        self._crypto_indices = list(optimizer.crypto_idx)
        self._equity_tickers = [t for t in DataManager.EQUITIES if t in optimizer.assets]

    def update_portfolio_state(self, portfolio_value: float, benchmark_value: float,
                               current_weights: np.ndarray, date: pd.Timestamp):
        self._portfolio_value = portfolio_value
        self._benchmark_value = benchmark_value
        self._current_weights = current_weights.copy()
        self._peak_value = max(self._peak_value, portfolio_value)
        self._portfolio_values_history.append(portfolio_value)
        self._benchmark_values_history.append(benchmark_value)
        self._days_since_rebalance += 1
        self._current_date = date

    def _lookup_row(self, df: Optional[pd.DataFrame], date: pd.Timestamp) -> pd.Series:
        if df is None:
            return pd.Series(dtype=float)
        if date in df.index:
            return df.loc[date]
        mask = df.index <= date
        if mask.any():
            return df.loc[df.index[mask][-1]]
        return pd.Series(dtype=float)

    def get_regime(self, ml_prob: float, spy_above_sma: bool, current_vol: float,
                   tlt_momentum: float = 0.0, equity_risk_premium: float = 0.0) -> str:
        if self.model is None:
            if spy_above_sma:
                return 'RISK_ON'
            if ml_prob > 0.55:
                return 'RISK_REDUCED'
            return 'DEFENSIVE'

        features_row = pd.Series({
            'realized_vol': current_vol, 'vol_momentum': 0.0,
            'equity_risk_premium': equity_risk_premium, 'trend_score': 0.0,
            'momentum_21d': 0.0, 'qqq_vs_spy': 0.0, 'tlt_momentum': tlt_momentum,
        })
        if self._features is not None and self._current_date is not None:
            full_row = self._lookup_row(self._features, self._current_date)
            if len(full_row) > 0:
                features_row = full_row

        date = self._current_date
        if date is None:
            above_sma_row = raw_mom_row = vols_row = ir_row = gc_row = pd.Series(dtype=float)
        else:
            above_sma_row = self._lookup_row(self._above_sma, date)
            raw_mom_row = self._lookup_row(self._raw_momentum, date)
            vols_row = self._lookup_row(self._asset_volatilities, date)
            ir_row = self._lookup_row(self._information_ratio, date)
            gc_row = self._lookup_row(self._golden_cross, date)

        pv_arr = np.array(self._portfolio_values_history)
        bv_arr = np.array(self._benchmark_values_history)

        def _trailing_return(arr, n):
            if len(arr) < n + 1:
                return 0.0
            return (arr[-1] / arr[-n - 1]) - 1.0

        current_dd = (self._portfolio_value - self._peak_value) / self._peak_value if self._peak_value > 0 else 0.0

        obs = self.obs_builder.build(
            features_row=features_row,
            current_weights=self._current_weights,
            equity_indices=self._equity_indices,
            safe_haven_indices=self._safe_haven_indices,
            crypto_indices=self._crypto_indices,
            portfolio_value=self._portfolio_value,
            initial_capital=100000.0,
            days_since_rebalance=self._days_since_rebalance,
            rebalance_period=self.config.rebalance_period,
            current_drawdown=current_dd,
            above_sma=above_sma_row,
            raw_momentum=raw_mom_row,
            asset_volatilities=vols_row,
            information_ratio=ir_row,
            ml_prob=ml_prob,
            golden_cross=gc_row,
            equity_tickers=self._equity_tickers,
            portfolio_returns_5d=_trailing_return(pv_arr, 5),
            portfolio_returns_21d=_trailing_return(pv_arr, 21),
            portfolio_returns_63d=_trailing_return(pv_arr, 63),
            benchmark_returns_21d=_trailing_return(bv_arr, 21),
            noise_sigma=0.0,
        )

        action = self.model.predict_deterministic(obs)
        regime = REGIME_MAP[action]
        self._days_since_rebalance = 0

        # Track for permutation importance
        self._obs_history.append(obs.copy())
        self._action_history.append(action)

        return regime

    # --- Stubs for Streamlit UI ---
    def walk_forward_train(self, features, returns, initial_train_years=5, step_months=12):
        logger.info("RLRegimeClassifier: walk_forward_train is a no-op (using PPO agent)")
        return pd.Series(0.5, index=features.index)

    def get_shap_figure(self):
        if len(self._obs_history) < 20 or self.model is None:
            return None
        try:
            obs_matrix = np.array(self._obs_history)        # (N, 25)
            actions = np.array(self._action_history)         # (N,)
            n_samples = min(200, len(obs_matrix))
            idx = np.random.choice(len(obs_matrix), n_samples, replace=False)
            obs_sample = obs_matrix[idx]
            act_sample = actions[idx]

            importances = np.zeros(obs_sample.shape[1])
            for feat_i in range(obs_sample.shape[1]):
                permuted = obs_sample.copy()
                permuted[:, feat_i] = np.random.permutation(permuted[:, feat_i])
                new_actions = np.array([self.model.predict_deterministic(row) for row in permuted])
                importances[feat_i] = (new_actions != act_sample).mean()

            order = np.argsort(importances)
            sorted_names = [OBS_FEATURE_NAMES[i] for i in order]
            sorted_imp = importances[order]

            fig, ax = plt.subplots(figsize=(10, 7))
            colors = ['#e74c3c' if v > 0.1 else '#3498db' for v in sorted_imp]
            ax.barh(sorted_names, sorted_imp, color=colors)
            ax.set_xlabel('Permutation Importance (action flip rate)')
            ax.set_title('RL Policy Feature Importance (Permutation-based)')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"RL SHAP Plot Error: {e}")
            return None

    def get_validation_curves_figure(self):
        eval_path = os.path.join(os.path.dirname(__file__), 'models', 'rl_regime_ppo', 'evaluations.npz')
        if not os.path.exists(eval_path):
            return None
        try:
            data = np.load(eval_path)
            timesteps = data['timesteps']
            results = data['results']       # (N, n_eval_episodes)
            ep_lengths = data['ep_lengths']

            mean_rewards = results.mean(axis=1)
            std_rewards = results.std(axis=1)
            mean_lengths = ep_lengths.mean(axis=1)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            ax1.plot(timesteps, mean_rewards, color='blue', linewidth=2, label='Mean Reward')
            ax1.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                             alpha=0.2, color='blue')
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_title('PPO Training Reward Curve')
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Episode Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(timesteps, mean_lengths, color='green', linewidth=2, label='Mean Ep Length')
            ax2.set_title('Episode Length Over Training')
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel('Steps per Episode')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"RL Health Plot Error: {e}")
            return None


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("=" * 70)
    print("RL Regime Agent — Standalone Test (MLX Backend)")
    print("=" * 70)

    # --- 1. Load data ---
    print("\n[1/5] Loading market data...", flush=True)
    from alpha_engine import DataManager, StrategyConfig, AlphaDominatorOptimizer

    config = StrategyConfig()
    dm = DataManager(start_date='2010-01-01', config=config)
    dm.load_data()
    dm.engineer_features()

    (prices, returns, features, vix, sma_200, above_sma, raw_mom, rel_strength,
     vols, info_ratio, mom_score, golden_cross_df, log_ret_30d, rsi_14) = dm.get_aligned_data()
    categories = dm.get_asset_categories()
    optimizer = AlphaDominatorOptimizer(dm.all_tickers, categories, config)

    print(f"   Data loaded: {len(prices)} trading days, {len(dm.all_tickers)} assets")
    print(f"   Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"   Equity indices: {optimizer.equity_idx}")

    # --- 2. Build environment ---
    print("\n[2/5] Building RegimeBacktestEnv (train window: 2010-2020)...", flush=True)
    train_env = RegimeBacktestEnv(
        prices=prices, returns=returns, features=features,
        ml_probs=pd.Series(0.5, index=features.index),
        sma_200=sma_200, above_sma=above_sma, raw_momentum=raw_mom,
        relative_strength=rel_strength, asset_volatilities=vols,
        information_ratio=info_ratio, momentum_score=mom_score,
        golden_cross=golden_cross_df, log_returns_30d=log_ret_30d, rsi_14=rsi_14,
        optimizer=optimizer, config=config,
        train_end_date="2020-12-31", noise_sigma=0.02,
    )
    print(f"   Rebalance points: {len(train_env.rebalance_indices)}")
    print(f"   Valid dates: {len(train_env.valid_dates)}")

    # --- 3. Validate environment ---
    print("\n[3/5] Running env validation...", flush=True)
    obs, info = train_env.reset()
    assert obs.shape == (25,), f"Bad obs shape: {obs.shape}"
    obs2, r, term, trunc, info2 = train_env.step(0)
    assert obs2.shape == (25,), f"Bad step obs shape"
    print("   Environment validation PASSED")

    # --- 4. Manual episode rollout ---
    print("\n[4/5] Running manual episode rollout (random actions)...", flush=True)
    obs, info = train_env.reset()
    print(f"   Initial obs shape: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]")
    total_reward = 0.0
    step_count = 0
    regime_counts = {'RISK_ON': 0, 'RISK_REDUCED': 0, 'DEFENSIVE': 0}
    while True:
        action = train_env.action_space.sample()
        obs, reward, terminated, truncated, info = train_env.step(action)
        total_reward += reward
        step_count += 1
        regime = info.get('regime', '?')
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        pv = info.get('portfolio_value', 0)
        bv = info.get('benchmark_value', 0)
        print(f"   Step {step_count:2d}: action={action} ({regime:14s}), "
              f"reward={reward:+.3f}, PV=${pv:>10,.0f}, BM=${bv:>10,.0f}")
        if terminated or truncated:
            break

    print(f"\n   Episode summary:")
    print(f"     Steps: {step_count}")
    print(f"     Total reward: {total_reward:.3f}")
    print(f"     Final portfolio: ${train_env._portfolio_value:,.0f}")
    print(f"     Final benchmark: ${train_env._benchmark_value:,.0f}")
    print(f"     Regime distribution: {regime_counts}")

    # --- 5. PPO Training ---
    if '--train' in sys.argv:
        # Parse timesteps from args: --train or --train 50000
        ts = 10_000
        idx = sys.argv.index('--train')
        if idx + 1 < len(sys.argv):
            try:
                ts = int(sys.argv[idx + 1])
            except ValueError:
                pass

        print(f"\n[5/5] Training PPO with MLX ({ts:,} timesteps)...", flush=True)
        trainer = RLTrainer(
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

        print("\n   Evaluating trained model...", flush=True)
        eval_results = trainer.evaluate(n_episodes=5)
        print(f"   Mean reward:   {eval_results['mean_reward']:.3f} +/- {eval_results['std_reward']:.3f}")
        print(f"   Mean Sharpe:   {eval_results['mean_sharpe']:.3f}")
        print(f"   Regime dist:   {eval_results['regime_distribution']}")
        print(f"   Final PV:      ${eval_results['final_portfolio_value']:,.0f}")
        print(f"   Final BM:      ${eval_results['final_benchmark_value']:,.0f}")
        print(f"   Training episodes logged: {len(trainer.reward_history)}")
    else:
        print("\n[5/5] SKIPPED: pass --train to run PPO training (e.g. --train 100000)")

    print("\n" + "=" * 70)
    print("Test complete.")
    print("=" * 70)
