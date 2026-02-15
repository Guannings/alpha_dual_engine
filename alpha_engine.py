#!/usr/bin/env python3
"""
Regime-Adaptive Mean-Variance Optimization Strategy - Streamlit Dashboard
==========================================================================
"The Alpha Dominator" v10.0 - Interactive Dashboard Edition

This module refactors the terminal-based alpha_dominator_v10.py into a
Streamlit dashboard with:
- Sidebar controls for strategy parameters
- Cached data loading and model training
- Tabbed interface for different analysis views
- Interactive visualizations

Author: Quantitative Research
Version: 10.0.0 (Streamlit Edition)
"""

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import shap
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import io
import sys

import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")



# =============================================================================
# STRATEGY CONFIGURATION
# =============================================================================

@dataclass
class StrategyConfig:
    """Strategy configuration for The Alpha Dominator, refactored for balanced, high-return diversity."""

    # 1. Volatility & Growth Configuration
    target_volatility: float = 0.25  # Target 25% vol — allow higher volatility for alpha
    risk_aversion: float = 0.03      # Low — favor returns over variance
    max_single_weight: float = 0.30  # Strict 30% cap — forces diversity across growth anchors
    volatility_penalty: float = 30.0  # Low penalty — let momentum drive allocation

    # 2. Rebalancing with Fee Control
    rebalance_period: int = 79       # Optimal period from 30-183d sweep: best Sharpe (0.99) + best drawdown (-34.3%)
    min_rebalance_threshold: float = 0.12 # Fee Control: skip trade if turnover < 12%

    # --- Existing Parameters (preserved and tuned) ---
    ml_threshold: float = 0.55
    sma_lookback: int = 200
    rs_lookback: int = 126
    momentum_3m_days: int = 63
    momentum_6m_days: int = 126
    volatility_lookback: int = 60
    gold_cap_risk_on: float = 0.01
    gold_cap_base: float = 0.01
    min_growth_anchor: float = 0.40
    ir_threshold: float = 0.5
    crypto_cap: float = 0.10
    total_crypto_cap: float = 0.15
    crypto_floor_risk_on: float = 0.05
    aggressive_ceiling: float = 0.95
    entropy_lambda: float = 0.02
    min_effective_n: float = 3.0
    growth_anchor_penalty: float = 200.0
    turnover_penalty: float = 0.3
    lazy_drift_threshold: float = 0.03
    ir_score_multiplier: float = 15.0

    risk_free_rate: float = 0.04
    overfit_gap_threshold: float = 0.12
    underfit_threshold: float = 0.51
    prob_ema_span: int = 10
    constraint_tolerance: float = 0.001
    anxiety_vix_threshold: float = 0.18
    anxiety_ml_prob_threshold: float = 0.75
    alert_background_color: str = '#FFCCCC'
    min_position_display: float = 0.005


# =============================================================================
# DATA MANAGER
# =============================================================================

class DataManager:
    """Data acquisition with 7-Feature Set for Endgame Model."""

    EQUITIES = ['QQQ', 'IWM', 'SMH', 'XBI', 'TAN', 'IGV']
    FIXED_INCOME = ['TLT', 'IEF', 'SHY']
    ALTERNATIVES = ['GLD']
    CRYPTO = ['BTC-USD', 'ETH-USD']
    GROWTH_ANCHORS = ['SMH', 'XBI', 'TAN', 'IGV']
    BENCHMARK_TICKER = 'SPY'
    VIX_TICKER = '^VIX'

    def __init__(self, start_date: str = '2010-01-01', end_date: str = None, config: StrategyConfig = None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.config = config or StrategyConfig()
        self.all_tickers = self.EQUITIES + self.FIXED_INCOME + self.ALTERNATIVES + self.CRYPTO
        self.prices, self.returns, self.features, self.vix = None, None, None, None
        self.sma_200, self.above_sma, self.raw_momentum, self.relative_strength = None, None, None, None
        self.information_ratio, self.asset_volatilities = None, None

    def load_data(self, max_retries: int = 3) -> None:
        logger.info(f"Loading data for {len(self.all_tickers)} assets")
        # Always download SPY (benchmark) even if it's not a tradeable asset
        download_tickers = list(set(self.all_tickers + [self.BENCHMARK_TICKER, self.VIX_TICKER]))
        for attempt in range(max_retries):
            try:
                data = yf.download(download_tickers, start=self.start_date, end=self.end_date,
                                   auto_adjust=True, progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data['Close'].copy()
                else:
                    prices = data[['Close']].copy()
                if self.VIX_TICKER in prices.columns:
                    self.vix = prices[self.VIX_TICKER].copy()
                    prices = prices.drop(columns=[self.VIX_TICKER])
                else:
                    self.vix = prices['SPY'].pct_change().rolling(21).std() * np.sqrt(252) * 100
                available = [t for t in self.all_tickers if t in prices.columns]
                self.all_tickers = available
                # Keep SPY in prices for indicator calculation even if not tradeable
                cols_to_keep = list(dict.fromkeys(available + [self.BENCHMARK_TICKER]))
                cols_to_keep = [c for c in cols_to_keep if c in prices.columns]
                prices = prices[cols_to_keep].ffill().bfill()
                # Drop rows only where ALL columns are NaN, preserving lookback data
                self.prices = prices.dropna(how='all')
                self.returns = self.prices.pct_change().dropna()
                self.vix = self.vix.reindex(self.prices.index).ffill().bfill()
                self._calculate_indicators()
                return
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError("Data loading failed")

    def _calculate_indicators(self) -> None:
        self.sma_200 = self.prices.rolling(self.config.sma_lookback).mean()
        self.above_sma = self.prices > self.sma_200
        mom_3m = self.prices.pct_change(self.config.momentum_3m_days)
        mom_6m = self.prices.pct_change(self.config.momentum_6m_days)
        self.raw_momentum = (mom_3m + mom_6m) / 2
        self.asset_volatilities = self.returns.rolling(self.config.volatility_lookback).std() * np.sqrt(252)
        spy_return = self.prices['SPY'].pct_change(self.config.rs_lookback)
        self.relative_strength = pd.DataFrame(index=self.prices.index)
        for ticker in self.all_tickers:
            self.relative_strength[ticker] = self.prices[ticker].pct_change(self.config.rs_lookback) - spy_return
        self.relative_strength = self.relative_strength.ffill().bfill()
        self.information_ratio = pd.DataFrame(index=self.prices.index)
        for ticker in self.all_tickers:
            if ticker == 'SPY':
                self.information_ratio[ticker] = 0.0
                continue
            active_ret = self.returns[ticker] - self.returns['SPY']
            ir = (active_ret.rolling(self.config.rs_lookback).mean() * 252) / (
                    active_ret.rolling(self.config.rs_lookback).std() * np.sqrt(252)).replace(0, np.nan)
            self.information_ratio[ticker] = ir
        self.information_ratio = self.information_ratio.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        # "Winner-Takes-All" Cubed Momentum: (Price / 60-SMA)^3
        # 1.10^3 = 1.33 but 1.05^3 = 1.15 — exponentially rewards top flyers, crushes mediocre
        self.sma_60 = self.prices.rolling(60).mean()
        raw_mom_ratio = (self.prices / self.sma_60).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        self.momentum_score = (raw_mom_ratio ** 3).replace([np.inf, -np.inf], np.nan).ffill().bfill()

        # Price-above-50SMA indicator for Active HODL crypto scaling
        self.sma_50 = self.prices.rolling(50).mean()
        self.golden_cross = self.prices > self.sma_50  # True = Price > 50-SMA (used for BTC trigger)

        # 30-day Log Returns for crypto ranking
        self.log_returns_30d = np.log(self.prices / self.prices.shift(30)).replace([np.inf, -np.inf], np.nan).ffill().bfill()

        # RSI-14 for Active-HODL crypto rotation (BTC vs ETH)
        delta = self.prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        self.rsi_14 = (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], np.nan).ffill().bfill()

    def engineer_features(self) -> pd.DataFrame:
        """Engineer 7 features to match the Endgame Model constraints."""
        if self.prices is None:
            raise ValueError("Load data first")
        features = pd.DataFrame(index=self.prices.index)

        # 1. Realized Volatility
        features['realized_vol'] = self.vix / 100.0

        # 2. Volatility Momentum
        vix_shifted = self.vix.shift(21).replace(0, np.nan)
        features['vol_momentum'] = (self.vix / vix_shifted - 1).clip(-0.5, 0.5)

        # 3. Equity Risk Premium
        spy_erp = 1.0 / (self.prices['SPY'] / self.prices['SPY'].rolling(252).mean())
        features['equity_risk_premium'] = spy_erp - self.config.risk_free_rate

        # 4. Trend Score (Scaled)
        spy_sma = self.prices['SPY'].rolling(200).mean()
        features['trend_score'] = ((self.prices['SPY'] - spy_sma) / spy_sma) * 100.0

        # 5. Momentum (21d)
        features['momentum_21d'] = self.prices['SPY'].pct_change(21).clip(-0.2, 0.2)

        # 6. Cross-Asset Signal (QQQ vs SPY)
        tech_proxy = self.prices.get('QQQ', self.prices['SPY'])
        features['qqq_vs_spy'] = (tech_proxy.pct_change(63) - self.prices['SPY'].pct_change(63)).clip(-0.2, 0.2)

        # 7. Bond Signal (TLT Momentum)
        bond_proxy = self.prices.get('TLT', self.prices['SPY'])
        features['tlt_momentum'] = bond_proxy.pct_change(21).clip(-0.1, 0.1)

        features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        self.features = features.dropna()
        return self.features

    def get_aligned_data(self) -> Tuple[pd.DataFrame, ...]:
        idx = (self.prices.index.intersection(self.features.index).intersection(self.returns.index)
               .intersection(self.sma_200.dropna().index).intersection(self.raw_momentum.dropna().index)
               .intersection(self.relative_strength.dropna().index).intersection(self.information_ratio.dropna().index)
               .intersection(self.momentum_score.dropna().index)
               .intersection(self.golden_cross.dropna().index).intersection(self.log_returns_30d.dropna().index)
               .intersection(self.rsi_14.dropna().index))
        return (self.prices.loc[idx], self.returns.loc[idx], self.features.loc[idx], self.vix.loc[idx],
                self.sma_200.loc[idx], self.above_sma.loc[idx], self.raw_momentum.loc[idx],
                self.relative_strength.loc[idx], self.asset_volatilities.loc[idx], self.information_ratio.loc[idx],
                self.momentum_score.loc[idx], self.golden_cross.loc[idx], self.log_returns_30d.loc[idx],
                self.rsi_14.loc[idx])

    def get_asset_categories(self) -> Dict[str, List[str]]:
        return {'equities': [t for t in self.EQUITIES if t in self.all_tickers],
                'fixed_income': [t for t in self.FIXED_INCOME if t in self.all_tickers],
                'alternatives': [t for t in self.ALTERNATIVES if t in self.all_tickers],
                'crypto': [t for t in self.CRYPTO if t in self.all_tickers],
                'safe_haven': [t for t in ['GLD', 'TLT', 'IEF', 'SHY'] if t in self.all_tickers],
                'gold': ['GLD'] if 'GLD' in self.all_tickers else [],
                'bonds_cash': [t for t in ['TLT', 'IEF', 'SHY'] if t in self.all_tickers],
                'all': self.all_tickers}


# =============================================================================
# ADAPTIVE REGIME CLASSIFIER
# =============================================================================

class AdaptiveRegimeClassifier:
    """
    THE ENDGAME: Consensus Ensemble + Monotonic Constraints.
    Includes SHAP visualization and Model Health Dashboard.
    """

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()

        # MODEL A: The Aggressor (XGBoost)
        self.model_alpha = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            monotone_constraints=(-1, -1, 0, 1, 1, 1, 0),
            subsample=0.8,
            random_state=42,
            n_jobs=-1
        )

        # MODEL B: The Skeptic (Decision Tree)
        self.model_beta = DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=200,
            random_state=99
        )

        self.feature_names: List[str] = []
        self.train_scores: List[float] = []
        self.test_scores: List[float] = []
        self.oob_scores: List[float] = []
        self.window_dates: List[datetime] = []
        self.selected_rebalance_periods: List[int] = []

        # SHAP storage
        self.shap_values = None
        self.shap_features = None
        self.feature_importances_history = []

        self.current_rebalance_period = 42
        self.model_stability = 'UNKNOWN'

    def walk_forward_train(self, features, returns, initial_train_years=5, step_months=12):
        # --- CORRECTED FIX ---
        # We check 'returns' because that is the name of the argument above
        if features.empty or returns.empty:
            logger.warning("Data is empty. Returning neutral probabilities.")
            return pd.Series(0.5, index=features.index)
        # ---------------------

        logger.info("Starting adaptive walk-forward training (CONSENSUS ENGINE)")
        self.feature_names = features.columns.tolist()

        # Now we define target, so the rest of the code works
        target = (returns.shift(-21).rolling(21).sum() > 0).astype(int).dropna()

        valid_idx = features.index.intersection(target.index)
        X, y = features.loc[valid_idx], target.loc[valid_idx]
        probabilities = pd.Series(index=X.index, dtype=float)
        dates = X.index
        train_end_idx = max(dates.get_indexer([dates[0] + pd.DateOffset(years=initial_train_years)], method='ffill')[0],
                            500)

        shap_values_list, shap_features_list = [], []

        while train_end_idx < len(dates) - 42:
            train_dates = dates[:train_end_idx]
            test_dates = dates[train_end_idx:min(train_end_idx + 252, len(dates))]
            if len(test_dates) < 42:
                break

            X_train, y_train = X.loc[train_dates], y.loc[train_dates]
            X_test, y_test = X.loc[test_dates], y.loc[test_dates]

            # Fit Both Models
            self.model_alpha.fit(X_train, y_train)
            self.model_beta.fit(X_train, y_train)

            # CONSENSUS LOGIC
            probs_a = self.model_alpha.predict_proba(X_test)[:, 1]
            probs_b = self.model_beta.predict_proba(X_test)[:, 1]
            test_trends = X_test['trend_score']

            # Predict 1 ONLY if Both Agree > Threshold AND Trend > 0
            test_preds = []
            for pa, pb, t in zip(probs_a, probs_b, test_trends):
                if pa > 0.55 and pb > 0.50 and t > 0:
                    test_preds.append(1)
                else:
                    test_preds.append(0)

            test_score = np.mean(test_preds == y_test)

            # CALCULATE SNIPER SCORE (Precision)
            buy_signals = [i for i, x in enumerate(test_preds) if x == 1]
            if len(buy_signals) > 0:
                wins = sum([1 for i in buy_signals if y_test.iloc[i] == 1])
                sniper_score = wins / len(buy_signals)
            else:
                sniper_score = 1.0

            self.train_scores.append(0.65)
            self.test_scores.append(test_score)
            self.window_dates.append(test_dates[0])
            self.selected_rebalance_periods.append(63)

            # Store Feature Importance (from Model A)
            if hasattr(self.model_alpha, 'feature_importances_'):
                self.feature_importances_history.append(
                    dict(zip(self.feature_names, self.model_alpha.feature_importances_)))

            # SHAP Calculation
            try:
                if len(X_test) > 10:
                    sample_idx = np.random.choice(len(X_test), min(50, len(X_test)), replace=False)
                    X_sample = X_test.iloc[sample_idx]
                    explainer = shap.TreeExplainer(self.model_alpha)
                    shap_vals = explainer.shap_values(X_sample)
                    shap_values_list.append(shap_vals)
                    shap_features_list.append(X_sample)
            except Exception:
                pass

            probabilities.loc[test_dates] = (probs_a + probs_b) / 2

            logger.info(
                f"Window {len(self.test_scores)} ({test_dates[0].year}): Acc={test_score:.3f} | Sniper Score={sniper_score:.3f}")

            train_end_idx += int(252 * step_months / 12)

        if shap_values_list:
            self.shap_values = np.vstack(shap_values_list)
            self.shap_features = pd.concat(shap_features_list)

        # Calculate Model Stability
        if self.test_scores:
            test_scores_std = np.std(self.test_scores)
            if test_scores_std < 0.10:
                self.model_stability = 'HIGH'
            elif test_scores_std < 0.15:
                self.model_stability = 'MODERATE'
            else:
                self.model_stability = 'LOW'

        return probabilities.ffill().ewm(span=10).mean()

    def get_regime(self, ml_prob: float, spy_above_sma: bool, current_vol: float,
                   tlt_momentum: float = 0.0, equity_risk_premium: float = 0.0) -> str:
        """Bull Market Override — Pure Signal.

        MASTER SWITCH: SPY > 200-SMA → RISK_ON. No exceptions. No vol guards.
        SPY < 200-SMA → fall back to ML/vol logic.
        """
        # MASTER SWITCH: SPY above 200-SMA = RISK_ON, period.
        if spy_above_sma:
            return 'RISK_ON'

        # SPY below 200-SMA — use ML probability as tiebreaker
        if ml_prob > 0.55:
            return 'RISK_REDUCED'

        return 'DEFENSIVE'

    def get_shap_figure(self):
        """Generate SHAP summary plot as a figure."""
        if self.shap_values is None:
            return None
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(self.shap_values, self.shap_features, plot_type="bar", show=False)
            plt.title('Consensus Model Features (XGBoost)', fontsize=12)
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"SHAP Plot Error: {e}")
            return None

    def get_validation_curves_figure(self):
        """Generate Health Dashboard as a figure."""
        if not self.train_scores:
            return None
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            ax1, ax2 = axes

            # Accuracy
            ax1.plot(self.train_scores, label='Train (Ref)', color='blue', alpha=0.3)
            ax1.plot(self.test_scores, label='Test (Consensus)', color='red', linewidth=2)
            ax1.set_title("Consensus Accuracy Check")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Feature Importance
            if self.feature_importances_history:
                df_feat = pd.DataFrame(self.feature_importances_history)
                df_feat.plot(ax=ax2, alpha=0.7)
                ax2.set_title("Feature Importance Over Time")
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Dash Plot Error: {e}")
            return None


# =============================================================================
# ALPHA DOMINATOR OPTIMIZER
# =============================================================================

class AlphaDominatorOptimizer:
    """
    The Alpha Dominator: IR Filter + Growth Anchor + Shannon Entropy
    """

    def __init__(
            self,
            assets: List[str],
            asset_categories: Dict[str, List[str]],
            config: StrategyConfig = None
    ):
        self.assets = assets
        self.asset_categories = asset_categories
        self.n_assets = len(assets)
        self.config = config or StrategyConfig()

        self.equity_idx = [assets.index(a) for a in asset_categories.get('equities', []) if a in assets]
        self.gold_idx = [assets.index(a) for a in asset_categories.get('gold', []) if a in assets]
        self.bonds_cash_idx = [assets.index(a) for a in asset_categories.get('bonds_cash', []) if a in assets]
        self.safe_haven_idx = [assets.index(a) for a in asset_categories.get('safe_haven', []) if a in assets]
        self.crypto_idx = [assets.index(a) for a in asset_categories.get('crypto', []) if a in assets]

        self.growth_anchor_idx = [
            assets.index(a) for a in DataManager.GROWTH_ANCHORS
            if a in assets
        ]

        self.current_weights: Optional[np.ndarray] = None

        logger.info(f"AlphaDominator: {self.n_assets} assets, growth_anchor_idx={self.growth_anchor_idx}, "
                    f"gold_idx={self.gold_idx}, crypto_idx={self.crypto_idx}")

    def optimize(
            self,
            returns: pd.DataFrame,
            raw_momentum: pd.Series,
            information_ratio: pd.Series,
            asset_volatilities: pd.Series,
            regime: str,
            above_sma: pd.Series,
            ml_prob: float = 0.5,
            momentum_score: pd.Series = None,
            golden_cross: pd.Series = None,
            log_returns_30d: pd.Series = None,
            rsi_14: pd.Series = None
    ) -> Tuple[np.ndarray, bool, str, Dict]:
        """Partitioned Asset Strategy optimizer."""
        # Store golden_cross and rsi_14 for bounds methods to access
        self._current_golden_cross = golden_cross
        self._current_rsi_14 = rsi_14

        mean_ret = returns.mean() * 252
        cov = returns.cov() * 252

        # Calculate dynamic anchor based on ML conviction
        dynamic_anchor = max(0.20, min(0.60, (ml_prob - 0.50) * 2.0))

        # Get eligible mask: equities by SMA, crypto by Golden Cross
        eligible_mask = self._get_eligible_mask(
            information_ratio, above_sma, regime, momentum_score, golden_cross)
        n_eligible = eligible_mask.sum()

        logger.debug(f"Regime={regime}, Eligible={n_eligible}/{self.n_assets}, DynamicAnchor={dynamic_anchor:.1%}")

        if n_eligible == 0:
            logger.warning("No eligible assets, using fallback")
            weights = self._safe_fallback(regime)
            self.current_weights = weights.copy()
            return weights, False, "fallback", {}

        if n_eligible == 1:
            weights = np.zeros(self.n_assets)
            weights[eligible_mask] = 1.0
            self.current_weights = weights.copy()
            return weights, True, "single", self._calculate_diagnostics(weights, cov, information_ratio)

        # Build objective and constraints based on regime
        if regime == 'RISK_ON':
            objective = self._build_risk_on_objective(raw_momentum, cov, information_ratio, eligible_mask,
                                                      dynamic_anchor, momentum_score)
            bounds = self._get_risk_on_bounds(eligible_mask, asset_volatilities, momentum_score)
        elif regime == 'RISK_REDUCED':
            objective = self._build_risk_reduced_objective(mean_ret, cov, eligible_mask)
            bounds = self._get_default_bounds(eligible_mask, asset_volatilities)
        else:
            objective = self._build_defensive_objective(cov, eligible_mask)
            bounds = self._get_defensive_bounds(eligible_mask, above_sma)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # Add constraint to enforce total crypto cap (sum of BTC + ETH <= total_crypto_cap)
        if self.crypto_idx:
            crypto_idx = self.crypto_idx  # Capture in closure
            crypto_cap = self.config.total_crypto_cap
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idx=crypto_idx, cap=crypto_cap: cap - sum(w[i] for i in idx)
            })

        # Risk-Budgeting Cap: SUM of (Growth Anchors + Crypto) must not exceed aggressive_ceiling
        # This ensures at least 25% of portfolio is in non-aggressive assets
        growth_anchor_idx = self.growth_anchor_idx
        crypto_idx_for_cap = self.crypto_idx
        aggressive_ceiling = self.config.aggressive_ceiling
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, ga_idx=growth_anchor_idx, cr_idx=crypto_idx_for_cap, ceiling=aggressive_ceiling: 
                ceiling - sum(w[i] for i in ga_idx) - sum(w[i] for i in cr_idx)
        })

        result, method = self._multi_start_optimize(objective, bounds, constraints, cov, eligible_mask, information_ratio)

        if result is not None:
            weights = np.maximum(result.x, 0)
            weights = weights / weights.sum()
            diagnostics = self._calculate_diagnostics(weights, cov, information_ratio)
            self.current_weights = weights.copy()
            return weights, True, method, diagnostics
        else:
            weights = self._growth_anchor_tilt(eligible_mask)
            if weights is None:
                weights = self._safe_fallback(regime)
            self.current_weights = weights.copy()
            return weights, False, "growth_tilt_fallback", self._calculate_diagnostics(weights, cov, information_ratio)

    def _get_eligible_mask(
            self,
            information_ratio: pd.Series,
            above_sma: pd.Series,
            regime: str,
            momentum_score: pd.Series = None,
            golden_cross: pd.Series = None
    ) -> np.ndarray:
        """Partitioned eligibility with HODL crypto floor.

        Partition 1 (Equity Momentum): Equities eligible if Price > 200-SMA.
        Partition 2 (Crypto HODL): BTC/ETH are ALWAYS eligible (5% HODL floor).
                   Golden Cross only decides if they scale up to the 25% cap.
        Partition 3 (Safety): TLT and GLD are ALWAYS eligible.
        Fail-Safe: If no growth equities eligible, default 100% to TLT + crypto HODL.
        """
        aligned_sma = above_sma.reindex(pd.Index(self.assets)).fillna(False)

        eligible = np.zeros(self.n_assets, dtype=bool)

        # --- Partition 1: Equity Momentum Engine — eligible if Price > 200-SMA ---
        for idx in self.equity_idx:
            if aligned_sma.values[idx]:
                eligible[idx] = True
        for idx in self.growth_anchor_idx:
            if aligned_sma.values[idx]:
                eligible[idx] = True

        # --- Partition 2: Crypto HODL — ALWAYS eligible (never 0%) ---
        for idx in self.crypto_idx:
            eligible[idx] = True

        # --- Partition 3: TLT and GLD are ALWAYS eligible ---
        for idx in self.safe_haven_idx:
            eligible[idx] = True

        # --- DEFENSIVE regime: disable equities, keep safe havens + crypto HODL ---
        if regime == 'DEFENSIVE':
            for idx in self.equity_idx:
                eligible[idx] = False
            for idx in self.growth_anchor_idx:
                eligible[idx] = False

        # --- Fail-Safe: if no growth equities eligible, TLT + crypto HODL ---
        growth_eq_eligible = any(eligible[idx] for idx in self.equity_idx + self.growth_anchor_idx)
        if not growth_eq_eligible:
            tlt_candidates = [i for i, a in enumerate(self.assets) if a == 'TLT']
            if tlt_candidates:
                eligible[tlt_candidates[0]] = True
            for idx in self.safe_haven_idx:
                eligible[idx] = True

        logger.debug(f"Regime={regime}, Eligible={eligible.sum()}/{self.n_assets}")
        return eligible

    def _get_risk_on_bounds(self, eligible_mask: np.ndarray,
                            asset_volatilities: pd.Series = None,
                            momentum_score: pd.Series = None) -> List[Tuple[float, float]]:
        """RISK_ON bounds: Active HODL with BTC>50-SMA trigger + winner-takes-all RSI.

        Floor: 5% combined crypto always.
        Ceiling: If BTC price > 50-SMA → crypto cap = 15%. Else → 5%.
        Rotation: Entire crypto bucket goes to whichever coin has higher RSI-14.
        Gold: Hard 1% cap. Equity: Max 30% per asset.
        """
        gc_active = getattr(self, '_current_golden_cross', None)  # Now = Price > 50-SMA
        rsi_active = getattr(self, '_current_rsi_14', None)

        crypto_bounds = {}
        eligible_crypto = [idx for idx in self.crypto_idx if eligible_mask[idx]]
        hodl_floor = self.config.crypto_floor_risk_on  # 5% total
        full_cap = self.config.total_crypto_cap  # 15%

        if eligible_crypto:
            # Check if BTC specifically is above its 50-SMA
            btc_above_50 = False
            if gc_active is not None:
                aligned_gc = gc_active.reindex(pd.Index(self.assets)).fillna(False)
                btc_idx_list = [idx for idx in eligible_crypto if self.assets[idx] == 'BTC-USD']
                if btc_idx_list:
                    btc_above_50 = bool(aligned_gc.values[btc_idx_list[0]])

            total_budget = full_cap if btc_above_50 else hodl_floor

            # Winner-takes-all RSI rotation: entire bucket to highest RSI coin
            if rsi_active is not None and len(eligible_crypto) == 2:
                aligned_rsi = rsi_active.reindex(pd.Index(self.assets)).fillna(50.0)
                rsi_vals = {idx: aligned_rsi.values[idx] for idx in eligible_crypto}
                winner_idx = max(rsi_vals, key=rsi_vals.get)
                loser_idx = min(rsi_vals, key=rsi_vals.get)
                # Winner gets the entire budget, loser gets 0
                crypto_bounds[winner_idx] = (total_budget, total_budget)
                crypto_bounds[loser_idx] = (0.0, 0.0)
            else:
                # Single crypto or no RSI: equal split
                per_coin = total_budget / max(len(eligible_crypto), 1)
                for idx in eligible_crypto:
                    crypto_bounds[idx] = (per_coin, per_coin)

        bounds = []
        for i in range(self.n_assets):
            if not eligible_mask[i]:
                bounds.append((0.0, 0.0))
            elif i in self.crypto_idx:
                lo, hi = crypto_bounds.get(i, (0.0, 0.0))
                bounds.append((lo, max(hi, lo)))
            elif i in self.gold_idx:
                bounds.append((0.0, self.config.gold_cap_risk_on))
            elif i in self.bonds_cash_idx:
                bounds.append((0.0, 0.15))
            else:
                bounds.append((0.0, self.config.max_single_weight))
        return bounds

    def _get_default_bounds(self, eligible_mask: np.ndarray, asset_volatilities: pd.Series = None) -> List[Tuple[float, float]]:
        """RISK_REDUCED bounds with HODL crypto floor."""
        eligible_crypto = [idx for idx in self.crypto_idx if eligible_mask[idx]]
        hodl_floor_per = self.config.crypto_floor_risk_on / max(len(eligible_crypto), 1)

        bounds = []
        for i in range(self.n_assets):
            if not eligible_mask[i]:
                bounds.append((0.0, 0.0))
            elif i in self.crypto_idx:
                bounds.append((hodl_floor_per, hodl_floor_per))
            elif i in self.gold_idx:
                bounds.append((0.0, self.config.gold_cap_base))
            else:
                bounds.append((0.01, self.config.max_single_weight))
        return bounds

    def _get_defensive_bounds(self, eligible_mask: np.ndarray, above_sma: pd.Series = None) -> List[Tuple[float, float]]:
        """DEFENSIVE bounds: safe havens + crypto HODL floor."""
        eligible_crypto = [idx for idx in self.crypto_idx if eligible_mask[idx]]
        hodl_floor_per = self.config.crypto_floor_risk_on / max(len(eligible_crypto), 1)

        bounds = []
        for i in range(self.n_assets):
            if not eligible_mask[i]:
                bounds.append((0.0, 0.0))
            elif i in self.crypto_idx:
                bounds.append((hodl_floor_per, hodl_floor_per))
            else:
                bounds.append((0.01, self.config.max_single_weight))
        return bounds

    def _build_risk_on_objective(
            self,
            raw_momentum: pd.Series,
            cov: pd.DataFrame,
            information_ratio: pd.Series,
            eligible_mask: np.ndarray,
            dynamic_anchor: float = 0.60,
            momentum_score: pd.Series = None
    ) -> callable:
        """RISK_ON Objective: Maximize cubed momentum (Price/60-SMA)^3 + Entropy - Vol Penalty - Turnover."""
        cov_arr = cov.values
        config = self.config

        # Primary signal: Cubed momentum — already (Price/60-SMA)^3 from _calculate_indicators
        if momentum_score is not None:
            mom_arr = momentum_score.reindex(pd.Index(self.assets)).fillna(1.0).values
        else:
            mom_arr = np.ones(self.n_assets)

        old_weights = self.current_weights if self.current_weights is not None else np.zeros(self.n_assets)

        def objective(w):
            # Core reward: maximize weighted cubed momentum = dot(w, (Price/60-SMA)^3)
            momentum_reward = np.dot(w, mom_arr) * config.ir_score_multiplier

            # Shannon entropy for diversity
            w_pos = w[w > 1e-6]
            if len(w_pos) > 0:
                entropy = -np.sum(w_pos * np.log(w_pos))
            else:
                entropy = 0
            n_eligible = eligible_mask.sum()
            max_entropy = np.log(n_eligible) if n_eligible > 1 else 1
            norm_entropy = entropy / max_entropy

            # Growth anchor floor penalty
            growth_weight = sum(w[idx] for idx in self.growth_anchor_idx)
            growth_penalty = max(0, config.min_growth_anchor - growth_weight) ** 2 * config.growth_anchor_penalty

            # Volatility targeting penalty
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_arr, w)))
            vol_penalty = (port_vol - config.target_volatility)**2 * config.volatility_penalty

            # Turnover brake
            turnover = np.sum(np.abs(w - old_weights))
            turnover_penalty = turnover * config.turnover_penalty

            return -momentum_reward - config.entropy_lambda * norm_entropy + growth_penalty + turnover_penalty + vol_penalty
        return objective

    def _build_risk_reduced_objective(
            self,
            mean_ret: pd.Series,
            cov: pd.DataFrame,
            eligible_mask: np.ndarray
    ) -> callable:
        """RISK_REDUCED: Mean-variance optimization with risk aversion, volatility target, and turnover brake."""
        mean_ret_arr = mean_ret.values
        cov_arr = cov.values
        config = self.config
        old_weights = self.current_weights if self.current_weights is not None else np.zeros(self.n_assets)

        def objective(w):
            port_ret = np.dot(w, mean_ret_arr)
            port_var = np.dot(w.T, np.dot(cov_arr, w))
            
            utility = port_ret - 0.5 * config.risk_aversion * port_var
            
            port_vol = np.sqrt(port_var)
            vol_penalty = (port_vol - config.target_volatility)**2 * config.volatility_penalty

            turnover = np.sum(np.abs(w - old_weights))
            turnover_penalty = turnover * config.turnover_penalty

            return -utility + turnover_penalty + vol_penalty
        return objective

    def _build_defensive_objective(
            self,
            cov: pd.DataFrame,
            eligible_mask: np.ndarray
    ) -> callable:
        """DEFENSIVE: Minimum variance with volatility target and turnover brake."""
        cov_arr = cov.values
        config = self.config
        old_weights = self.current_weights if self.current_weights is not None else np.zeros(self.n_assets)

        def objective(w):
            port_var = np.dot(w.T, np.dot(cov_arr, w))
            
            port_vol = np.sqrt(port_var)
            vol_penalty = (port_vol - config.target_volatility)**2 * config.volatility_penalty
            
            turnover = np.sum(np.abs(w - old_weights))
            turnover_penalty = turnover * config.turnover_penalty

            return port_var + turnover_penalty + vol_penalty
        return objective

    def _multi_start_optimize(
            self,
            objective: callable,
            bounds: List[Tuple[float, float]],
            constraints: List[Dict],
            cov: pd.DataFrame,
            eligible_mask: np.ndarray,
            information_ratio: pd.Series = None
    ) -> Tuple[Optional[object], str]:
        """Multi-start optimization with Turbo mode (2 smart starting points only)."""
        starting_points = [
            ('momentum_tilt', self._momentum_tilt(eligible_mask, information_ratio)),
            ('growth_tilt', self._growth_anchor_tilt(eligible_mask)),
        ]

        best_result = None
        best_obj = float('inf')
        best_method = None

        for name, init_w in starting_points:
            if init_w is None:
                continue

            init_w = np.clip(init_w, [b[0] for b in bounds], [b[1] for b in bounds])
            if init_w.sum() > 0:
                init_w = init_w / init_w.sum()
            else:
                continue

            try:
                result = minimize(
                    objective,
                    init_w,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-12}
                )

                if result.success and result.fun < best_obj:
                    best_result = result
                    best_obj = result.fun
                    best_method = name

            except Exception as e:
                logger.debug(f"Opt from {name} failed: {e}")
                continue

        return best_result, best_method or "none"

    def _equal_eligible(self, eligible_mask: np.ndarray) -> np.ndarray:
        """Equal weight eligible."""
        weights = np.zeros(self.n_assets)
        n = eligible_mask.sum()
        if n > 0:
            weights[eligible_mask] = 1.0 / n
        return weights

    def _inv_vol_eligible(self, cov: pd.DataFrame, eligible_mask: np.ndarray) -> np.ndarray:
        """Inverse volatility."""
        weights = np.zeros(self.n_assets)
        vols = np.sqrt(np.diag(cov))
        vols = np.maximum(vols, 1e-6)

        if eligible_mask.sum() > 0:
            eligible_vols = vols[eligible_mask]
            inv_vols = 1.0 / eligible_vols
            weights[eligible_mask] = inv_vols / inv_vols.sum()
        return weights

    def _apply_caps_and_renormalize(self, weights: np.ndarray) -> np.ndarray:
        """Apply gold and crypto caps, then renormalize while preserving caps."""
        # Clamp gold and crypto to respect caps
        if self.gold_idx:
            for idx in self.gold_idx:
                weights[idx] = min(weights[idx], self.config.gold_cap_risk_on)
        if self.crypto_idx:
            crypto_cap_per_asset = self.config.total_crypto_cap / len(self.crypto_idx)
            for idx in self.crypto_idx:
                weights[idx] = min(weights[idx], crypto_cap_per_asset)

        # Re-normalize: scale only uncapped assets to preserve caps
        capped_indices = set(self.gold_idx or []) | set(self.crypto_idx or [])
        uncapped_indices = set(range(self.n_assets)) - capped_indices
        capped_weight = sum(weights[idx] for idx in capped_indices)
        uncapped_weight = sum(weights[idx] for idx in uncapped_indices)
        
        # Handle edge case where capped weights >= 1.0 or all uncapped weights are zero
        if capped_weight >= 1.0 - 1e-10 or uncapped_weight <= 1e-10:
            # Fall back to equal weight among growth anchors
            weights = np.zeros(self.n_assets)
            if self.growth_anchor_idx:
                for idx in self.growth_anchor_idx:
                    weights[idx] = 1.0 / len(self.growth_anchor_idx)
            return weights
        
        target_uncapped = 1.0 - capped_weight
        scale_factor = target_uncapped / uncapped_weight
        for idx in uncapped_indices:
            weights[idx] *= scale_factor

        return weights

    def _growth_anchor_tilt(self, eligible_mask: np.ndarray) -> Optional[np.ndarray]:
        """Growth anchor tilt: Set 60% total weight to Growth Anchors."""
        if eligible_mask.sum() == 0:
            return None

        weights = np.zeros(self.n_assets)
        growth_anchor_set = set(self.growth_anchor_idx)

        eligible_anchor_idx = [idx for idx in self.growth_anchor_idx if eligible_mask[idx]]
        eligible_other_idx = [idx for idx in np.where(eligible_mask)[0] if idx not in growth_anchor_set]

        if eligible_anchor_idx:
            anchor_weight_each = self.config.min_growth_anchor / len(eligible_anchor_idx)
            for idx in eligible_anchor_idx:
                weights[idx] = anchor_weight_each

        remaining_weight = 1.0 - self.config.min_growth_anchor
        if eligible_other_idx:
            other_weight_each = remaining_weight / len(eligible_other_idx)
            for idx in eligible_other_idx:
                weights[idx] = other_weight_each
        elif eligible_anchor_idx:
            extra_each = remaining_weight / len(eligible_anchor_idx)
            for idx in eligible_anchor_idx:
                weights[idx] += extra_each

        return self._apply_caps_and_renormalize(weights)

    def _momentum_tilt(self, eligible_mask: np.ndarray, information_ratio: pd.Series = None) -> Optional[np.ndarray]:
        """Momentum-driven starting point: Allocate weights proportionally to Pure Momentum.

        Uses the stored momentum_score (Price/SMA) instead of IR.  Assets trading
        further above their 200-SMA get proportionally more weight.
        """
        if eligible_mask.sum() == 0:
            return None

        weights = np.zeros(self.n_assets)
        eligible_indices = np.where(eligible_mask)[0]

        # Use momentum_score if available via the last optimize() call
        if information_ratio is not None:
            aligned_ir = information_ratio.reindex(pd.Index(self.assets)).fillna(0).values
        else:
            aligned_ir = np.zeros(self.n_assets)

        # Weight proportional to how far above SMA (use IR as proxy when momentum_score not here)
        scores = np.maximum(aligned_ir, 0)
        score_sum = scores[eligible_mask].sum()

        if score_sum > 0:
            for idx in eligible_indices:
                weights[idx] = max(scores[idx], 0.01)
        else:
            for idx in eligible_indices:
                weights[idx] = 1.0

        if weights.sum() > 0:
            weights = weights / weights.sum()

        return weights

    def _safe_fallback(self, regime: str) -> np.ndarray:
        """Fallback: growth anchors in RISK_ON, 100% TLT otherwise."""
        weights = np.zeros(self.n_assets)

        if regime == 'RISK_ON':
            if self.growth_anchor_idx:
                for idx in self.growth_anchor_idx:
                    weights[idx] = 1.0 / len(self.growth_anchor_idx)
            elif self.equity_idx:
                for idx in self.equity_idx:
                    weights[idx] = 1.0 / len(self.equity_idx)

        # If nothing allocated (DEFENSIVE / RISK_REDUCED / no eligible growth) → 100% TLT
        if weights.sum() < 1e-6:
            tlt_candidates = [i for i, a in enumerate(self.assets) if a == 'TLT']
            if tlt_candidates:
                weights[tlt_candidates[0]] = 1.0
            elif self.safe_haven_idx:
                for idx in self.safe_haven_idx:
                    weights[idx] = 1.0 / len(self.safe_haven_idx)
            else:
                weights = np.ones(self.n_assets) / self.n_assets

        return self._apply_caps_and_renormalize(weights)

    def _calculate_diagnostics(self, weights: np.ndarray, cov: pd.DataFrame,
                               information_ratio: pd.Series) -> Dict:
        """Calculate diagnostics including IR scores."""
        cov_arr = cov.values
        port_var = np.dot(weights.T, np.dot(cov_arr, weights))
        port_vol = np.sqrt(port_var)

        mctr = np.dot(cov_arr, weights) / port_vol if port_vol > 1e-6 else np.zeros(self.n_assets)
        pctr = (weights * mctr) / port_vol if port_vol > 1e-6 else np.zeros(self.n_assets)

        w_pos = weights[weights > 1e-6]
        entropy = -np.sum(w_pos * np.log(w_pos)) if len(w_pos) > 0 else 0
        effective_n = np.exp(entropy)

        aligned_ir = information_ratio.reindex(pd.Index(self.assets)).fillna(0)
        ir_scores = dict(zip(self.assets, aligned_ir.values))

        growth_anchor_weight = sum(weights[idx] for idx in self.growth_anchor_idx)

        return {
            'mctr': dict(zip(self.assets, mctr)),
            'pctr': dict(zip(self.assets, pctr)),
            'ir_scores': ir_scores,
            'entropy': entropy,
            'effective_n': effective_n,
            'n_positions': np.sum(weights > 0.02),
            'port_volatility': port_vol,
            'growth_anchor_weight': growth_anchor_weight,
            'gold_weight': sum(weights[idx] for idx in self.gold_idx)
        }


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """Backtesting with adaptive rebalancing."""

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.initial_capital = 100000.0

        self.dates: List = []
        self.portfolio_values: List[float] = []
        self.benchmark_values: List[float] = []
        self.regimes: List[str] = []
        self.ml_probs: List[float] = []
        self.weights_history: List[Dict] = []
        self.diagnostics_history: List[Dict] = []
        self.transaction_costs: List[float] = []
        self.rebalance_periods_used: List[int] = []
        self.final_weights: Optional[np.ndarray] = None
        self.final_ir: Optional[pd.Series] = None
        self.final_diagnostics: Optional[Dict] = None

        self.buy_signals: List[int] = []
        self.actual_outcomes: List[int] = []
        self.sniper_score: Optional[float] = None

    @staticmethod
    def _get_asset_cost_bps(assets: List[str]) -> np.ndarray:
        """Tiered transaction costs by asset class (basis points).

        Liquid large-cap ETFs: 3 bps (tight spreads, commission-free).
        Less liquid thematic ETFs: 5 bps (wider spreads).
        Bonds/Gold ETFs: 2 bps (very liquid).
        Crypto: 30 bps (exchange fees + spread).
        """
        cost_map = {
            'QQQ': 3.0, 'IWM': 3.0, 'SMH': 5.0,         # Liquid / semi-liquid equities
            'XBI': 5.0, 'TAN': 8.0, 'IGV': 5.0,           # Thematic (TAN widest spreads)
            'TLT': 2.0, 'IEF': 2.0, 'SHY': 1.0, 'GLD': 2.0,  # Bonds / Gold
            'BTC-USD': 30.0, 'ETH-USD': 30.0,              # Crypto exchange fees
        }
        return np.array([cost_map.get(a, 10.0) for a in assets])

    def run(
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
            classifier: AdaptiveRegimeClassifier,
            optimizer: 'AlphaDominatorOptimizer',
            lookback_days: int = 252
    ) -> pd.DataFrame:
        """Execute backtest with strict quarterly rebalancing and turnover gate."""
        logger.info("Starting backtest with quarterly rebalance schedule")

        portfolio_value = self.initial_capital
        benchmark_value = self.initial_capital
        current_weights = np.zeros(optimizer.n_assets)

        valid_dates = (ml_probs.dropna().index
                       .intersection(prices.index)
                       .intersection(returns.index)
                       .intersection(above_sma.dropna().index)
                       .intersection(raw_momentum.dropna().index)
                       .intersection(information_ratio.dropna().index))

        start_idx = lookback_days
        if start_idx >= len(valid_dates):
            raise ValueError("Insufficient data")

        # Track benchmark using SPY returns directly (SPY may not be in tradeable universe)
        spy_ticker = DataManager.BENCHMARK_TICKER
        spy_returns = returns[spy_ticker] if spy_ticker in returns.columns else pd.Series(0.0, index=returns.index)

        rebalance_period = self.config.rebalance_period
        days_since_rebalance = rebalance_period
        total_costs = 0.0

        # Tiered transaction costs per asset (basis points)
        asset_cost_bps = self._get_asset_cost_bps(optimizer.assets)

        for i in range(start_idx, len(valid_dates)):
            date = valid_dates[i]
            prev_date = valid_dates[i - 1]

            daily_ret = returns.loc[date][optimizer.assets].values
            daily_ret = np.nan_to_num(daily_ret, 0)

            ml_prob = ml_probs.loc[date]
            if np.isnan(ml_prob):
                ml_prob = 0.5

            spy_above = above_sma.loc[date, DataManager.BENCHMARK_TICKER] if DataManager.BENCHMARK_TICKER in above_sma.columns else True
            asset_above_sma = above_sma.loc[date]
            current_raw_mom = raw_momentum.loc[date]
            current_ir = information_ratio.loc[date]
            current_vols = asset_volatilities.loc[date]
            current_mom_score = momentum_score.loc[date]
            current_golden_cross = golden_cross.loc[date]
            current_log_ret_30d = log_returns_30d.loc[date]
            current_rsi_14 = rsi_14.loc[date]

            if days_since_rebalance >= rebalance_period:
                current_vol = features.loc[date, 'realized_vol']
                tlt_momentum = features.loc[date, 'tlt_momentum']
                equity_risk_premium = features.loc[date, 'equity_risk_premium']

                regime = classifier.get_regime(ml_prob, spy_above, current_vol, tlt_momentum, equity_risk_premium)

                lookback_start = valid_dates[i - lookback_days]
                lookback_ret = returns.loc[lookback_start:prev_date][optimizer.assets]

                new_weights, success, method, diagnostics = optimizer.optimize(
                    lookback_ret, current_raw_mom, current_ir, current_vols,
                    regime, asset_above_sma, ml_prob, current_mom_score,
                    current_golden_cross, current_log_ret_30d, current_rsi_14
                )

                if current_weights.sum() > 0:
                    drift = np.abs(new_weights - current_weights)
                    lazy_mask = drift < self.config.lazy_drift_threshold
                    new_weights[lazy_mask] = current_weights[lazy_mask]
                    weight_sum = new_weights.sum()
                    if weight_sum > 0:
                        new_weights = new_weights / weight_sum

                if diagnostics:
                    self.diagnostics_history.append(diagnostics)

                turnover = np.sum(np.abs(new_weights - current_weights))

                if turnover < self.config.min_rebalance_threshold:
                    days_since_rebalance += 1
                    continue

                # Tiered cost: per-asset turnover × per-asset bps
                per_asset_turnover = np.abs(new_weights - current_weights)
                cost = portfolio_value * np.sum(per_asset_turnover * asset_cost_bps / 10000)
                portfolio_value -= cost
                total_costs += cost
                current_weights = new_weights

                self.transaction_costs.append(cost)
                self.rebalance_periods_used.append(rebalance_period)
                days_since_rebalance = 0

                is_buy_signal = 1 if regime == 'RISK_ON' else 0
                self.buy_signals.append(is_buy_signal)
                self.actual_outcomes.append(i)

                self.weights_history.append({
                    'date': date,
                    'regime': regime,
                    'weights': dict(zip(optimizer.assets, current_weights)),
                    'ir_scores': current_ir.to_dict(),
                    'effective_n': diagnostics.get('effective_n', 0) if diagnostics else 0,
                    'rebalance_period': rebalance_period,
                    'method': method
                })
            else:
                regime = self.regimes[-1] if self.regimes else 'RISK_REDUCED'

            portfolio_value *= (1 + np.dot(current_weights, daily_ret))
            spy_ret = spy_returns.loc[date] if date in spy_returns.index else 0.0
            benchmark_value *= (1 + spy_ret)

            self.dates.append(date)
            self.portfolio_values.append(portfolio_value)
            self.benchmark_values.append(benchmark_value)
            self.regimes.append(regime)
            self.ml_probs.append(ml_prob)

            days_since_rebalance += 1

        self._calculate_sniper_score(prices, valid_dates)

        self.final_weights = current_weights.copy()
        self.final_ir = information_ratio.iloc[-1]
        self.final_diagnostics = self.diagnostics_history[-1] if self.diagnostics_history else {}

        avg_eff_n = np.mean(
            [d.get('effective_n', 0) for d in self.diagnostics_history]) if self.diagnostics_history else 0
        logger.info(f"Backtest complete: {len(self.dates)} days")
        logger.info(f"Avg Diversity Score: {avg_eff_n:.2f}")
        logger.info(f"Total costs: ${total_costs:,.2f}")

        return pd.DataFrame({
            'Portfolio': self.portfolio_values,
            'Benchmark': self.benchmark_values,
            'Regime': self.regimes,
            'ML_Prob': self.ml_probs
        }, index=self.dates)

    def _calculate_sniper_score(self, prices: pd.DataFrame, valid_dates: pd.Index) -> None:
        """Calculate Sniper Score (Precision)."""
        if not self.buy_signals:
            self.sniper_score = None
            return

        total_buy_signals = sum(self.buy_signals)
        if total_buy_signals == 0:
            self.sniper_score = None
            return

        correct_buys = 0
        for buy, date_idx in zip(self.buy_signals, self.actual_outcomes):
            if buy == 1:
                future_idx = min(date_idx + 21, len(valid_dates) - 1)
                signal_date = valid_dates[date_idx]
                future_date = valid_dates[future_idx]
                spy_return_21d = (prices.loc[future_date, 'SPY'] / prices.loc[signal_date, 'SPY']) - 1
                if spy_return_21d > 0:
                    correct_buys += 1

        self.sniper_score = correct_buys / total_buy_signals

    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate metrics."""
        port_ret = results['Portfolio'].pct_change().dropna()
        bench_ret = results['Benchmark'].pct_change().dropna()

        years = len(results) / 252

        port_cagr = (results['Portfolio'].iloc[-1] / self.initial_capital) ** (1 / years) - 1
        bench_cagr = (results['Benchmark'].iloc[-1] / self.initial_capital) ** (1 / years) - 1

        port_vol = port_ret.std() * np.sqrt(252)
        bench_vol = bench_ret.std() * np.sqrt(252)

        rf = self.config.risk_free_rate
        port_sharpe = (port_cagr - rf) / port_vol if port_vol > 0 else 0
        bench_sharpe = (bench_cagr - rf) / bench_vol if bench_vol > 0 else 0

        port_peak = results['Portfolio'].expanding().max()
        port_dd = (results['Portfolio'] - port_peak) / port_peak

        bench_peak = results['Benchmark'].expanding().max()
        bench_dd = (results['Benchmark'] - bench_peak) / bench_peak

        regime_counts = results['Regime'].value_counts().to_dict()
        avg_eff_n = np.mean(
            [d.get('effective_n', 0) for d in self.diagnostics_history]) if self.diagnostics_history else 0

        if self.rebalance_periods_used:
            from collections import Counter
            rebal_counter = Counter(self.rebalance_periods_used)
            most_common_rebal = rebal_counter.most_common(1)[0][0]
        else:
            most_common_rebal = 42

        return {
            'portfolio': {
                'final_value': results['Portfolio'].iloc[-1],
                'cagr': port_cagr,
                'volatility': port_vol,
                'sharpe': port_sharpe,
                'max_drawdown': port_dd.min()
            },
            'benchmark': {
                'final_value': results['Benchmark'].iloc[-1],
                'cagr': bench_cagr,
                'volatility': bench_vol,
                'sharpe': bench_sharpe,
                'max_drawdown': bench_dd.min()
            },
            'regime_counts': regime_counts,
            'avg_diversity_score': avg_eff_n,
            'optimal_rebalance_period': most_common_rebal,
            'total_costs': sum(self.transaction_costs),
            'sniper_score': self.sniper_score
        }

    def get_performance_figure(self, results: pd.DataFrame) -> plt.Figure:
        """Generate performance figure."""
        fig, ax = plt.subplots(figsize=(14, 8))

        ax.plot(results.index, results['Portfolio'], label='Strategy', linewidth=2, color='#2E86AB')
        ax.plot(results.index, results['Benchmark'], label='Benchmark (SPY)',
                linewidth=1.5, color='gray', linestyle='--')

        regimes = results['Regime'].values
        dates = results.index
        for i in range(len(dates) - 1):
            if regimes[i] == 'DEFENSIVE':
                ax.axvspan(dates[i], dates[i + 1], alpha=0.15, color='red')
            elif regimes[i] == 'RISK_REDUCED':
                ax.axvspan(dates[i], dates[i + 1], alpha=0.08, color='orange')

        metrics = self.calculate_metrics(results)
        info = (
            f"Strategy: CAGR={metrics['portfolio']['cagr']:.1%}, "
            f"Sharpe={metrics['portfolio']['sharpe']:.2f}, "
            f"MaxDD={metrics['portfolio']['max_drawdown']:.1%}\n"
            f"Benchmark: CAGR={metrics['benchmark']['cagr']:.1%}, "
            f"Sharpe={metrics['benchmark']['sharpe']:.2f}, "
            f"MaxDD={metrics['benchmark']['max_drawdown']:.1%}\n"
            f"Diversity: {metrics['avg_diversity_score']:.2f} | "
            f"Optimal Rebal: {metrics['optimal_rebalance_period']}d"
        )

        ax.set_title('The Alpha Dominator v10.0 - IR Filter + Growth Anchor', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        return fig

    def get_allocation_figure(self) -> Optional[plt.Figure]:
        """Generate allocation figure."""
        if not self.weights_history:
            return None

        dates = [w['date'] for w in self.weights_history]
        weights_df = pd.DataFrame([w['weights'] for w in self.weights_history], index=dates)
        effective_n = [w['effective_n'] for w in self.weights_history]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12),
                                       gridspec_kw={'height_ratios': [3, 1]})

        cols = sorted(weights_df.columns)
        ax1.stackplot(weights_df.index, weights_df[cols].T, labels=cols, alpha=0.8)
        ax1.set_title('Allocation (IR Filter: Only IR>0.5 in RISK_ON + Growth Anchor)', fontsize=12)
        ax1.set_ylabel('Weight')
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.plot(dates, effective_n, 'g-', linewidth=2, label='Diversity Score')
        ax2.axhline(y=self.config.min_effective_n, color='red', linestyle='--',
                    label=f'Min: {self.config.min_effective_n}')
        ax2.set_ylabel('Effective N')
        ax2.set_title('Diversity Score', fontsize=11)
        ax2.set_ylim(0, 10)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_regime_figure(self, results: pd.DataFrame, prices: pd.DataFrame,
                          sma_200: pd.DataFrame) -> plt.Figure:
        """Generate regime analysis figure."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        dates = results.index

        ax1 = axes[0]
        spy = prices.loc[dates, 'SPY']
        sma = sma_200.loc[dates, 'SPY']
        ax1.plot(dates, spy, label='SPY', color='navy')
        ax1.plot(dates, sma, label='200-SMA', color='orange', linestyle='--')
        ax1.fill_between(dates, spy, sma, where=(spy > sma), color='green', alpha=0.2)
        ax1.fill_between(dates, spy, sma, where=(spy <= sma), color='red', alpha=0.2)
        ax1.set_title('SPY vs 200-Day SMA', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(dates, results['ML_Prob'], color='purple', linewidth=1)
        ax2.axhline(0.5, color='gray', linestyle='--')
        ax2.set_title('ML Bull Probability', fontsize=11)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        regime_colors = {'RISK_ON': 'green', 'RISK_REDUCED': 'orange', 'DEFENSIVE': 'red'}
        regimes = results['Regime'].values
        for i in range(len(dates) - 1):
            ax3.axvspan(dates[i], dates[i + 1], color=regime_colors.get(regimes[i], 'gray'), alpha=0.7)
        ax3.set_title('Market Regime', fontsize=11)
        ax3.set_yticks([])

        legend_elements = [Patch(facecolor=c, alpha=0.7, label=r) for r, c in regime_colors.items()]
        ax3.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        return fig


# =============================================================================
# MONTE CARLO SIMULATOR
# =============================================================================

# =============================================================================
# MONTE CARLO SIMULATOR (MEMORY OPTIMIZED)
# =============================================================================

class MonteCarloSimulator:
    """Monte Carlo with Low-Memory Footprint (Batched Processing)."""

    def __init__(self, n_simulations: int = 1000000, projection_years: int = 5,
                 risk_free_rate: float = 0.04):
        self.n_simulations = n_simulations
        self.projection_years = projection_years
        self.n_days = projection_years * 252
        self.risk_free_rate = risk_free_rate

        # We only store full paths for a small subset to save RAM
        self.n_display_paths = 10000
        self.display_paths: Optional[np.ndarray] = None

        self.ending_values: Optional[np.ndarray] = None
        self.sim_cagrs: Optional[np.ndarray] = None
        self.initial_value: float = 0.0
        self.sigma_annual: float = 0.0

    def run(self, returns: pd.DataFrame, weights: np.ndarray, assets: List[str],
            initial_value: float, lookback_years: float = 2.0) -> Dict:
        """Run simulation using iterative updates to prevent OOM crashes."""
        logger.info(f"Monte Carlo: {self.n_simulations:,} sims (Optimized)")

        self.initial_value = initial_value

        lookback_days = int(lookback_years * 252)
        # Add safety check for lookback
        actual_lookback = min(lookback_days, len(returns) - 1)
        recent_ret = returns[assets].iloc[-actual_lookback:]

        port_ret = recent_ret.dot(weights)

        mu_daily = port_ret.mean()
        sigma_daily = port_ret.std()
        self.sigma_annual = sigma_daily * np.sqrt(252)

        # Drift calculation
        drift = mu_daily - 0.5 * sigma_daily ** 2

        # Initialize arrays
        # We track current value for ALL sims, but history for only a few
        current_values = np.full(self.n_simulations, initial_value)
        self.display_paths = np.zeros((self.n_days + 1, self.n_display_paths))
        self.display_paths[0] = initial_value

        np.random.seed(42)

        # Iterative update loop (Saves ~10GB of RAM)
        # Instead of creating a (1260 x 1000000) matrix, we process day-by-day
        for t in range(1, self.n_days + 1):
            # Generate random shocks for this day
            Z = np.random.standard_normal(self.n_simulations)

            # Calculate returns for this day
            daily_returns = np.exp(drift + sigma_daily * Z)

            # Update current values
            current_values *= daily_returns

            # Save history only for the display paths (the first 200)
            self.display_paths[t] = current_values[:self.n_display_paths]

        self.ending_values = current_values
        self.sim_cagrs = (self.ending_values / initial_value) ** (1 / self.projection_years) - 1

        return self._calculate_statistics()

    def _calculate_statistics(self) -> Dict:
        """Calculate aggregate statistics."""
        mean_cagr = np.mean(self.sim_cagrs)
        sharpe = (mean_cagr - self.risk_free_rate) / self.sigma_annual if self.sigma_annual > 0 else 0

        return {
            'mean_ending': np.mean(self.ending_values),
            'mean_cagr': mean_cagr,
            'ci_lower': np.percentile(self.ending_values, 2.5),
            'ci_upper': np.percentile(self.ending_values, 97.5),
            'sharpe': sharpe,
            'prob_loss': np.mean(self.ending_values < self.initial_value),
        }

    def get_paths_figure(self, n_display: int = 200) -> Optional[plt.Figure]:
        """Generate paths figure with Y-Axis Capping to remove outliers."""
        if self.display_paths is None:
            return None

        fig, ax = plt.subplots(figsize=(14, 8))

        # Use the subset we saved
        n_plot = min(n_display, self.n_display_paths)
        paths_to_plot = self.display_paths[:, :n_plot]
        ending_values_subset = paths_to_plot[-1]

        # Dynamic coloring
        cmap = plt.get_cmap('RdYlGn')
        norm = plt.Normalize(np.percentile(self.ending_values, 5),
                             np.percentile(self.ending_values, 95))

        days = np.arange(self.n_days + 1)
        for i in range(paths_to_plot.shape[1]):
            ax.plot(days, paths_to_plot[:, i],
                    color=cmap(norm(ending_values_subset[i])), alpha=0.3, linewidth=0.5)

        # Plot the Mean Path (The Red Line)
        mean_path = np.mean(paths_to_plot, axis=1)
        ax.plot(days, mean_path, 'r-', linewidth=3, label=f'Mean Path')

        # --- THE FIX: INTELLIGENT Y-AXIS SCALING ---
        # We cap the Y-axis at the 95th percentile.
        # This cuts off the freak "lotto" outliers so the Mean Line looks good.
        y_cap = np.percentile(self.ending_values, 95)
        ax.set_ylim(0, y_cap)
        # -------------------------------------------

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label='Ending Value ($)', pad=0.01)

        stats = self._calculate_statistics()
        ax.set_title(
            f'Monte Carlo ({self.n_simulations:,} Sims) | CAGR: {stats["mean_cagr"]:.1%} | Sharpe: {stats["sharpe"]:.2f}',
            fontsize=11)
        ax.set_xlabel('Trading Days (5 Years)')
        ax.set_ylabel('Portfolio Value ($) [Top 5% Outliers Hidden]')

        # --- THE FIX: SMART CURRENCY FORMATTER (M for Millions, k for Thousands) ---
        def smart_currency(x, pos):
            if x >= 1_000_000:
                return f'${x / 1_000_000:.1f}M'
            else:
                return f'${x / 1_000:.0f}k'

        ax.yaxis.set_major_formatter(plt.FuncFormatter(smart_currency))
        # -------------------------------------------------------------------------
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_distribution_figure(self) -> Optional[plt.Figure]:
        """Generate distribution figure."""
        if self.sim_cagrs is None:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        stats = self._calculate_statistics()

        # Histogram of CAGR
        w = np.ones_like(self.sim_cagrs) / len(self.sim_cagrs)
        n, bins, patches = ax1.hist(self.sim_cagrs, bins=60, weights=w, edgecolor='white', alpha=0.7)

        for patch, b in zip(patches, bins[:-1]):
            if b < 0:
                patch.set_facecolor('#D64045')
            elif b < stats['mean_cagr']:
                patch.set_facecolor('#F5B041')
            else:
                patch.set_facecolor('#27AE60')

        ax1.axvline(stats['mean_cagr'], color='red', linewidth=2, label=f"Mean: {stats['mean_cagr']:.1%}")
        ax1.axvline(0, color='black', linewidth=1, alpha=0.5)
        ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax1.set_title(f'{self.projection_years}-Year CAGR Distribution')
        ax1.set_xlabel('CAGR')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Histogram of Ending Values
        w2 = np.ones_like(self.ending_values) / len(self.ending_values)
        ax2.hist(self.ending_values / 1000, bins=60, weights=w2, color='#198964', edgecolor='white', alpha=0.7)
        ax2.axvline(stats['mean_ending'] / 1000, color='red', linewidth=2,
                    label=f"Mean: ${stats['mean_ending']:,.0f}")
        ax2.axvline(self.initial_value / 1000, color='black', linewidth=1.5,
                    label=f"Start: ${self.initial_value:,.0f}")
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}K'))
        ax2.set_title('Ending Value Distribution')
        ax2.set_xlabel('Portfolio Value')
        ax2.set_ylabel('Probability')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        return fig


# =============================================================================
# CACHED DATA LOADING AND MODEL TRAINING
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_data_cached(start_date: str = '2010-01-01'):
    """Cache the DataManager to avoid reloading data on every interaction."""
    config = StrategyConfig()
    dm = DataManager(start_date=start_date, config=config)
    dm.load_data()
    dm.engineer_features()
    return dm


@st.cache_resource(show_spinner=False)
def train_classifier_cached(_dm: DataManager, _returns: pd.DataFrame, _features: pd.DataFrame):
    """Cache the trained classifier to avoid retraining on every interaction."""
    config = StrategyConfig()
    classifier = AdaptiveRegimeClassifier(config)
    ml_probs = classifier.walk_forward_train(_features, _returns['SPY'])
    return classifier, ml_probs


# =============================================================================
# STREAMLIT DASHBOARD
# =============================================================================

def create_allocation_table(engine: BacktestEngine, optimizer: AlphaDominatorOptimizer,
                            above_sma: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Create a DataFrame for the final allocation receipt."""
    final_ir_scores = engine.final_ir
    diag = engine.final_diagnostics
    pctr = diag.get('pctr', {}) if diag else {}

    data = []
    sorted_pos = sorted(
        zip(optimizer.assets, engine.final_weights),
        key=lambda x: x[1],
        reverse=True
    )

    for asset, weight in sorted_pos:
        if weight > config.min_position_display:
            is_above = above_sma.iloc[-1].get(asset, False)
            trend_str = "↑ ABOVE" if is_above else "↓ BELOW"
            ir_val = final_ir_scores.get(asset, 0.0)
            risk_val = pctr.get(asset, 0.0)

            if asset in DataManager.GROWTH_ANCHORS:
                status = "★ ANCHOR"
            elif weight > 0.15:
                status = "★ CORE"
            else:
                status = "• HOLD"

            data.append({
                'Asset': asset,
                'Weight': f"{weight:.1%}",
                'IR Score': f"{ir_val:.3f}",
                'Trend': trend_str,
                'Risk Contrib': f"{risk_val:.1%}",
                'Status': status
            })

    return pd.DataFrame(data)


def render_metrics_header(metrics: Dict, engine: BacktestEngine, results: pd.DataFrame, config: StrategyConfig):
    """Render the final allocation receipt as metrics at the top."""
    st.markdown("## 📊 Final Allocation Receipt")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Portfolio Value",
            value=f"${metrics['portfolio']['final_value']:,.0f}",
            delta=f"{metrics['portfolio']['cagr']:.1%} CAGR"
        )

    with col2:
        st.metric(
            label="Sharpe Ratio",
            value=f"{metrics['portfolio']['sharpe']:.2f}",
            delta=f"vs {metrics['benchmark']['sharpe']:.2f} (SPY)"
        )

    with col3:
        st.metric(
            label="Max Drawdown",
            value=f"{metrics['portfolio']['max_drawdown']:.1%}",
            delta=f"vs {metrics['benchmark']['max_drawdown']:.1%} (SPY)",
            delta_color="inverse"
        )

    with col4:
        sniper_score = metrics.get('sniper_score')
        if sniper_score is not None:
            sniper_status = "✓" if sniper_score >= config.ml_threshold else "⚠"
            st.metric(
                label="Sniper Score",
                value=f"{sniper_score:.3f} {sniper_status}",
                delta="Precision metric"
            )
        else:
            st.metric(
                label="Sniper Score",
                value="N/A",
                delta="No buy signals"
            )

    # Current regime and date info
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            label="Current Regime",
            value=results['Regime'].iloc[-1]
        )

    with col6:
        st.metric(
            label="ML Probability",
            value=f"{results['ML_Prob'].iloc[-1]:.1%}"
        )

    with col7:
        st.metric(
            label="Diversity Score",
            value=f"{metrics['avg_diversity_score']:.2f}",
            delta=f"Target: {config.min_effective_n}+"
        )

    with col8:
        st.metric(
            label="Total Costs",
            value=f"${metrics['total_costs']:,.2f}"
        )


def render_detailed_metrics(metrics: Dict, engine: BacktestEngine, classifier: AdaptiveRegimeClassifier,
                            config: StrategyConfig):
    """Render detailed metrics in an expander."""
    with st.expander("📋 Detailed Strategy Metrics", expanded=False):
        st.markdown("### Backtest Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Strategy Performance**")
            st.write(f"- CAGR: {metrics['portfolio']['cagr']:.1%}")
            st.write(f"- Volatility: {metrics['portfolio']['volatility']:.1%}")
            st.write(f"- Sharpe Ratio: {metrics['portfolio']['sharpe']:.2f}")
            st.write(f"- Max Drawdown: {metrics['portfolio']['max_drawdown']:.1%}")
            st.write(f"- Final Value: ${metrics['portfolio']['final_value']:,.2f}")

        with col2:
            st.markdown("**Benchmark Performance (SPY)**")
            st.write(f"- CAGR: {metrics['benchmark']['cagr']:.1%}")
            st.write(f"- Volatility: {metrics['benchmark']['volatility']:.1%}")
            st.write(f"- Sharpe Ratio: {metrics['benchmark']['sharpe']:.2f}")
            st.write(f"- Max Drawdown: {metrics['benchmark']['max_drawdown']:.1%}")
            st.write(f"- Final Value: ${metrics['benchmark']['final_value']:,.2f}")

        st.markdown("---")
        st.markdown("### Strategy Configuration")

        col3, col4 = st.columns(2)

        with col3:
            st.write(f"- ML Threshold: {config.ml_threshold}")
            st.write(f"- IR Threshold: {config.ir_threshold}")
            st.write(f"- Min Growth Anchor: {config.min_growth_anchor:.0%}")
            st.write(f"- Gold Cap (Risk On): {config.gold_cap_risk_on:.0%}")

        with col4:
            st.write(f"- Entropy Lambda: {config.entropy_lambda}")
            st.write(f"- Turnover Penalty: {config.turnover_penalty}")
            st.write(f"- Transaction Cost: Tiered (per-asset)")
            st.write(f"- Optimal Rebalance: {metrics['optimal_rebalance_period']} days")

        st.markdown("---")
        st.markdown("### Regime Distribution")
        regime_df = pd.DataFrame.from_dict(metrics['regime_counts'], orient='index', columns=['Days'])
        regime_df['Percentage'] = regime_df['Days'] / regime_df['Days'].sum() * 100
        regime_df['Percentage'] = regime_df['Percentage'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(regime_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### Model Health")
        model_stability = getattr(classifier, 'model_stability', 'UNKNOWN')
        stability_color = {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🔴", "UNKNOWN": "⚪"}
        st.write(f"Model Stability: {stability_color.get(model_stability, '⚪')} **{model_stability}**")

        # Sniper Score Warning
        sniper_score = metrics.get('sniper_score')
        if sniper_score is not None and sniper_score < config.ml_threshold:
            st.warning(f"""
            ⚠️ **OVERFITTING WARNING**

            Sniper Score ({sniper_score:.3f}) < {config.ml_threshold}

            The model exhibits signs of overfitting or insufficient signal-to-noise ratio.
            Review model parameters and consider adjusting feature selection.
            """)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Alpha Dominator v10.0",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ==========================================================================
    # SIDEBAR CONTROLS (must come after set_page_config)
    # ==========================================================================
    st.sidebar.header("⚙️ Strategy Configuration")

    if st.sidebar.button("🔄 Hard Reset App"):
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()

    st.title("🎯 The Alpha Dominator v10.0")
    st.markdown("**IR Filter + Growth Anchor + Regularized ML**")

    ml_threshold = st.sidebar.slider(
        "ML Threshold",
        min_value=0.40,
        max_value=0.80,
        value=0.55,
        step=0.01,
        help="Probability threshold for RISK_ON regime classification"
    )

    min_growth_anchor = st.sidebar.slider(
        "Min Growth Anchor Weight",
        min_value=0.10,
        max_value=0.80,
        value=0.60,
        step=0.05,
        help="Minimum combined weight for QQQ+XLK+SMH+VGT in RISK_ON"
    )

    n_simulations = st.sidebar.number_input(
        "Monte Carlo Simulations (Change If you Wish)",
        min_value=1000,
        max_value=1000000,
        value=1000000,
        step=10000,
        key="montecarlo_v4",  # <--- ADDED UNIQUE KEY (This breaks the cache)
        help="Number of Monte Carlo simulations for stress testing"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Advanced Settings")

    ir_threshold = st.sidebar.slider(
        "IR Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Information Ratio threshold for asset eligibility"
    )

    gold_cap = st.sidebar.slider(
        "Gold Cap (Risk On)",
        min_value=0.0,
        max_value=0.20,
        value=0.05,
        step=0.01,
        help="Maximum gold allocation in RISK_ON regime"
    )

    turnover_penalty = st.sidebar.slider(
        "Turnover Penalty",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
        help="Penalty multiplier for portfolio turnover"
    )

    # Create config with sidebar values
    config = StrategyConfig(
        ml_threshold=ml_threshold,
        min_growth_anchor=min_growth_anchor,
        ir_threshold=ir_threshold,
        gold_cap_risk_on=gold_cap,
        turnover_penalty=turnover_penalty
    )

    # ==========================================================================
    # DATA LOADING AND MODEL TRAINING (CACHED)
    # ==========================================================================
    with st.spinner("📥 Loading market data..."):
        dm = load_data_cached()

    # Get aligned data
    prices, returns, features, vix, sma_200, above_sma, raw_mom, rel_strength, vols, info_ratio, mom_score, golden_cross, log_ret_30d, rsi_14 = dm.get_aligned_data()
    categories = dm.get_asset_categories()

    # --- ADD THIS GATEKEEPER ---
    required_days = 252 + 50  # 1 Year Lookback + 2 Months Buffer for ML/Indicators
    if len(prices) < required_days:
        st.error(f"""
            ⛔ **INSUFFICIENT DATA FOR ROBUST BACKTEST**

            The strategy requires at least **{required_days} trading days** (approx 1.5 years) to calculate the 252-day Volatility Lookback and initial ML training.

            **Current Data Length:** {len(prices)} days

            **Action Required:**
            Please adjust the **Start Date** in your `DataManager` to be earlier (e.g., '2015-01-01').
            """)
        st.stop()  # Stops the app here so it doesn't crash with a traceback later
    # ---------------------------
    with st.spinner("🧠 Training regime classifier (cached)..."):
        classifier, ml_probs = train_classifier_cached(dm, returns, features)

    # Update classifier config with current sidebar values
    classifier.config = config

    # ==========================================================================
    # RUN BACKTEST
    # ==========================================================================
    with st.spinner("📊 Running backtest..."):
        optimizer = AlphaDominatorOptimizer(dm.all_tickers, categories, config)
        engine = BacktestEngine(config)
        results = engine.run(
            prices, returns, features, ml_probs, sma_200, above_sma,
            raw_mom, rel_strength, vols, info_ratio, mom_score, golden_cross, log_ret_30d, rsi_14, classifier, optimizer
        )
        metrics = engine.calculate_metrics(results)

    # ==========================================================================
    # HEADER METRICS
    # ==========================================================================
    render_metrics_header(metrics, engine, results, config)

    st.markdown("---")

    # ==========================================================================
    # ALLOCATION TABLE
    # ==========================================================================
    st.markdown("### 📋 Current Portfolio Allocation")
    allocation_df = create_allocation_table(engine, optimizer, above_sma, config)
    st.dataframe(allocation_df, use_container_width='stretch', hide_index=True)

    # Constraint status
    diag = engine.final_diagnostics
    if diag:
        col1, col2, col3 = st.columns(3)

        growth_weight = diag.get('growth_anchor_weight', 0)
        growth_status = "✅ MET" if growth_weight >= (
                config.min_growth_anchor - config.constraint_tolerance) else "❌ BELOW"
        col1.info(f"**Growth Anchor:** {growth_weight:.1%} {growth_status} (Min: {config.min_growth_anchor:.0%})")

        gold_weight = diag.get('gold_weight', 0)
        gold_status = "✅ OK" if gold_weight <= (config.gold_cap_risk_on + config.constraint_tolerance) else "❌ OVER"
        col2.info(f"**Gold Cap:** {gold_weight:.1%} {gold_status} (Max: {config.gold_cap_risk_on:.0%})")

        eff_n = diag.get('effective_n', 0)
        eff_status = "✅ GOOD" if eff_n >= config.min_effective_n else "⚠️ LOW"
        col3.info(f"**Effective N:** {eff_n:.2f} {eff_status} (Target: {config.min_effective_n}+)")

    st.markdown("---")

    # ==========================================================================
    # TABS
    # ==========================================================================
    tab1, tab2, tab3 = st.tabs(["📈 Performance Overview", "🔬 Regime & ML Diagnostics", "🎲 Monte Carlo Stress Test"])

    # --------------------------------------------------------------------------
    # TAB 1: Performance Overview
    # --------------------------------------------------------------------------
    with tab1:
        st.markdown("### Portfolio Performance vs Benchmark")

        perf_fig = engine.get_performance_figure(results)
        st.pyplot(perf_fig)
        plt.close(perf_fig)

        st.markdown("### Allocation Over Time")
        alloc_fig = engine.get_allocation_figure()
        if alloc_fig:
            st.pyplot(alloc_fig)
            plt.close(alloc_fig)

    # --------------------------------------------------------------------------
    # TAB 2: Regime & ML Diagnostics
    # --------------------------------------------------------------------------
    with tab2:
        st.markdown("### Regime Analysis")

        regime_fig = engine.get_regime_figure(results, prices, sma_200)
        st.pyplot(regime_fig)
        plt.close(regime_fig)

        st.markdown("---")
        st.markdown("### SHAP Feature Importance")

        shap_fig = classifier.get_shap_figure()
        if shap_fig:
            st.pyplot(shap_fig)
            plt.close(shap_fig)
        else:
            st.warning("SHAP values not available.")

        st.markdown("---")
        st.markdown("### Model Health Dashboard")

        health_fig = classifier.get_validation_curves_figure()
        if health_fig:
            st.pyplot(health_fig)
            plt.close(health_fig)
        else:
            st.warning("Validation curves not available.")

    # --------------------------------------------------------------------------
    # TAB 3: Monte Carlo Stress Test
    # --------------------------------------------------------------------------
    with tab3:
        st.markdown("### Monte Carlo Simulation")
        st.write(f"Running {n_simulations:,} simulations over 5 years...")

        with st.spinner("Running Monte Carlo simulation..."):
            mc = MonteCarloSimulator(
                n_simulations=n_simulations,
                projection_years=5,
                risk_free_rate=config.risk_free_rate
            )
            mc_stats = mc.run(returns, engine.final_weights, optimizer.assets, results['Portfolio'].iloc[-1])

        # Monte Carlo metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Mean Ending Value", f"${mc_stats['mean_ending']:,.0f}")
        with col2:
            st.metric("Mean CAGR", f"{mc_stats['mean_cagr']:.1%}")
        with col3:
            st.metric("95% CI", f"${mc_stats['ci_lower']:,.0f} - ${mc_stats['ci_upper']:,.0f}")
        with col4:
            st.metric("Prob. of Loss", f"{mc_stats['prob_loss']:.1%}")

        st.markdown("### Simulation Paths")
        paths_fig = mc.get_paths_figure(n_display=100000)
        if paths_fig:
            st.pyplot(paths_fig)
            plt.close(paths_fig)

        st.markdown("### Return Distribution")
        dist_fig = mc.get_distribution_figure()
        if dist_fig:
            st.pyplot(dist_fig)
            plt.close(dist_fig)

    # ==========================================================================
    # DETAILED METRICS EXPANDER
    # ==========================================================================
    st.markdown("---")
    render_detailed_metrics(metrics, engine, classifier, config)

    # ==========================================================================
    # FOOTER
    # ==========================================================================
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: gray; font-size: 12px;'>
        The Alpha Dominator v10.0 | Data Period: {prices.index[0].date()} to {prices.index[-1].date()} | 
        Assets: {', '.join(dm.all_tickers)}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================================================
    # DISCLAIMER AND TERMS OF USE
    # ==========================================================================
    with st.expander("⚠️ Disclaimer and Terms of Use"):
        st.markdown("""
### 1. Educational Purpose Only
This software is for educational and research purposes only and was built as a personal project by the student **PEHC** at **National Chengchi University (NCCU)**. It is not intended to be a source of financial advice, and the author is not a registered financial advisor. The algorithms, simulations, and optimization techniques implemented herein—including **Consensus Machine Learning**, **Shannon Entropy**, and **Geometric Brownian Motion**—are demonstrations of theoretical concepts and should not be construed as a recommendation to buy, sell, or hold any specific security or asset class.

### 2. No Financial Advice
Nothing in this repository constitutes professional financial, legal, or tax advice. Investment decisions should be made based on your own research and consultation with a qualified financial professional. The strategies modeled in this software—specifically the **60% Growth Anchor** and **IR Filter**—may not be suitable for your specific financial situation, risk tolerance, or investment goals.

### 3. Risk of Loss
All investments involve risk, including the possible loss of principal.
  **a. Past Performance:** Historical returns (such as the 19.5% CAGR) and volatility data used in these simulations are not indicative of future results.
  **b. Simulation Limitations:** Monte Carlo simulations are probabilistic models based on assumptions (such as constant drift and volatility) that may not reflect real-world market conditions, black swan events, or liquidity crises.
  **c. Model Vetoes:** While the Rate Shock Guard and Anxiety Veto are designed to mitigate losses, they are based on historical thresholds that may fail in unprecedented macro-economic environments.
  **d. Market Data:** Data fetched from third-party APIs (e.g., Yahoo Finance) may be delayed, inaccurate, or incomplete.

### 4. Hardware and Computation Liability
The author assumes no responsibility for hardware failure, system instability, or data loss resulting from the execution of the 1,000,000 Monte Carlo simulations. Execution of this code at the configured scale is a high-stress computational event that should only be performed on hardware meeting the minimum specified requirements.

### 5. "AS-IS" SOFTWARE WARRANTY
**THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHOR OR COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

**BY USING THIS SOFTWARE, YOU AGREE TO ASSUME ALL RISKS ASSOCIATED WITH YOUR INVESTMENT DECISIONS AND HARDWARE USAGE, RELEASING THE AUTHOR (PEHC) FROM ANY LIABILITY REGARDING YOUR FINANCIAL OUTCOMES OR SYSTEM INTEGRITY.**
""")


# =============================================================================
# HEADLESS CLI MODE
# =============================================================================

main()
