"""
XGBoost regime classifier for Taiwan market.
Adapted from AdaptiveRegimeClassifier in alpha_engine.py:271-451.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    import shap
except ImportError:
    shap = None

from ..config import StrategyConfig

logger = logging.getLogger(__name__)

# Three regimes for Taiwan market
REGIMES = ('TRENDING', 'VOLATILE', 'QUIET')


class TAIEXRegimeClassifier:
    """Consensus ensemble regime classifier for Taiwan futures.

    Architecture (same as alpha_engine.py):
        - Model A (XGBoost): aggressive learner with monotonic constraints
        - Model B (DecisionTree): conservative cross-check
        - Consensus: both must agree for directional regime

    Regimes:
        TRENDING  — strong directional move, trade momentum
        VOLATILE  — high vol / mean-reverting, trade premium selling / hedges
        QUIET     — low vol range-bound, trade theta / range strategies
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()

        if XGBClassifier is None:
            raise ImportError("xgboost is required: pip install xgboost")

        # Model A: The Aggressor (XGBoost)
        # Monotonic constraints for 7 Taiwan features:
        #   realized_vol(-1), vol_momentum(-1), trend_score(+1),
        #   momentum_21d(+1), institutional_flow(+1),
        #   put_call_ratio(0), term_structure_slope(0)
        self.model_alpha = XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            monotone_constraints=(-1, -1, 1, 1, 1, 0, 0),
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

        # Model B: The Skeptic (Decision Tree)
        self.model_beta = DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=200,
            random_state=99,
        )

        self.feature_names: List[str] = []
        self.train_scores: List[float] = []
        self.test_scores: List[float] = []
        self.window_dates: List = []
        self.shap_values = None
        self.shap_features = None
        self.model_stability = 'UNKNOWN'

    def walk_forward_train(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        initial_train_years: Optional[int] = None,
        step_months: Optional[int] = None,
    ) -> pd.Series:
        """Walk-forward training with consensus ensemble.

        Args:
            features: DataFrame with 7 feature columns
            returns: Daily returns of TAIEX (for label construction)
            initial_train_years: Years of initial training window
            step_months: Months to step forward each window

        Returns:
            Series of ML probabilities (bull probability), indexed by date.
        """
        initial_train_years = initial_train_years or self.config.walk_forward_initial_years
        step_months = step_months or self.config.walk_forward_step_months

        if features.empty or returns.empty:
            logger.warning("Empty data — returning neutral probabilities")
            return pd.Series(0.5, index=features.index)

        self.feature_names = features.columns.tolist()

        # Label: is 21-day forward return positive?
        target = (returns.shift(-21).rolling(21).sum() > 0).astype(int).dropna()

        valid_idx = features.index.intersection(target.index)
        X, y = features.loc[valid_idx], target.loc[valid_idx]
        probabilities = pd.Series(index=X.index, dtype=float)
        dates = X.index

        train_end_idx = max(
            dates.get_indexer(
                [dates[0] + pd.DateOffset(years=initial_train_years)],
                method='ffill'
            )[0],
            500,
        )

        shap_values_list, shap_features_list = [], []

        while train_end_idx < len(dates) - 42:
            train_dates = dates[:train_end_idx]
            test_dates = dates[train_end_idx:min(train_end_idx + 126, len(dates))]
            if len(test_dates) < 42:
                break

            X_train, y_train = X.loc[train_dates], y.loc[train_dates]
            X_test, y_test = X.loc[test_dates], y.loc[test_dates]

            # Fit both models
            self.model_alpha.fit(X_train, y_train)
            self.model_beta.fit(X_train, y_train)

            # Consensus probabilities
            probs_a = self.model_alpha.predict_proba(X_test)[:, 1]
            probs_b = self.model_beta.predict_proba(X_test)[:, 1]

            # Consensus: predict bullish only if both agree and trend positive
            test_trends = X_test['trend_score']
            test_preds = []
            for pa, pb, t in zip(probs_a, probs_b, test_trends):
                if pa > 0.55 and pb > 0.50 and t > 0:
                    test_preds.append(1)
                else:
                    test_preds.append(0)

            test_score = np.mean(np.array(test_preds) == y_test.values)
            self.test_scores.append(test_score)
            self.window_dates.append(test_dates[0])

            # SHAP (optional)
            if shap is not None:
                try:
                    sample_size = min(50, len(X_test))
                    if sample_size > 10:
                        sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
                        X_sample = X_test.iloc[sample_idx]
                        explainer = shap.TreeExplainer(self.model_alpha)
                        sv = explainer.shap_values(X_sample)
                        shap_values_list.append(sv)
                        shap_features_list.append(X_sample)
                except Exception:
                    pass

            probabilities.loc[test_dates] = (probs_a + probs_b) / 2

            logger.info(
                f"Window {len(self.test_scores)} "
                f"({test_dates[0].strftime('%Y-%m')}): "
                f"Acc={test_score:.3f}"
            )

            train_end_idx += int(252 * step_months / 12)

        if shap_values_list:
            self.shap_values = np.vstack(shap_values_list)
            self.shap_features = pd.concat(shap_features_list)

        # Model stability
        if self.test_scores:
            std = np.std(self.test_scores)
            self.model_stability = (
                'HIGH' if std < 0.10 else 'MODERATE' if std < 0.15 else 'LOW'
            )

        return probabilities.ffill().ewm(span=10).mean()

    def get_regime(
        self,
        ml_prob: float,
        taiex_above_sma: bool,
        current_vol: float,
    ) -> str:
        """Determine market regime from ML probability and indicators.

        Logic:
            - TAIEX above 200-SMA + high ML prob → TRENDING
            - High volatility or TAIEX below SMA → VOLATILE
            - Otherwise → QUIET

        Returns:
            One of 'TRENDING', 'VOLATILE', 'QUIET'
        """
        if taiex_above_sma and ml_prob > self.config.ml_threshold:
            return 'TRENDING'

        if current_vol > 0.25 or (not taiex_above_sma and ml_prob < 0.45):
            return 'VOLATILE'

        return 'QUIET'

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get consensus probability for new data (after training)."""
        probs_a = self.model_alpha.predict_proba(features)[:, 1]
        probs_b = self.model_beta.predict_proba(features)[:, 1]
        return (probs_a + probs_b) / 2
