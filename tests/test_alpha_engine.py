"""Regression tests for the Alpha Dual Engine.

Fast, hermetic (no network): market data is synthesized so these run in CI
and against future library versions (the weekly latest-deps job exists to
catch the next pandas-3.0-style break early).
"""
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alpha_engine as ae

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FILES = ["alpha_engine.py", "rl_regime_agent.py", "rl_weight_agent.py"]


# ---------------------------------------------------------------------------
# Synthetic market data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_dm():
    """A DataManager filled with 8 years of synthetic GBM prices — the whole
    downstream pipeline (features, classifier, backtest) runs for real."""
    rng = np.random.default_rng(7)
    dm = ae.DataManager(start_date="2012-01-02", config=ae.StrategyConfig())
    idx = pd.bdate_range("2012-01-02", "2019-12-31")
    tickers = dm.all_tickers + [dm.BENCHMARK_TICKER]
    drift = rng.uniform(0.0001, 0.0006, len(tickers))
    vol = rng.uniform(0.008, 0.03, len(tickers))
    shocks = rng.standard_normal((len(idx), len(tickers)))
    log_paths = np.cumsum(drift + vol * shocks, axis=0)
    dm.prices = pd.DataFrame(100 * np.exp(log_paths), index=idx, columns=tickers)
    dm.returns = dm.prices.pct_change().dropna()
    dm.vix = pd.Series(18 + 8 * np.abs(rng.standard_normal(len(idx))), index=idx)
    dm._calculate_indicators()
    dm.engineer_features()
    return dm


@pytest.fixture(scope="module")
def aligned(synthetic_dm):
    return synthetic_dm.get_aligned_data()


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def _mc_returns():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0.0005, 0.01, size=(600, 3)), columns=list("ABC"))


@pytest.mark.parametrize("n_sims", [1_000, 20_000])
def test_monte_carlo_shapes(n_sims):
    """display_paths must cap at min(10000, n_sims) — n_sims below 10k used
    to raise a broadcast ValueError."""
    mc = ae.MonteCarloSimulator(n_simulations=n_sims, projection_years=2)
    stats = mc.run(_mc_returns(), np.array([0.4, 0.3, 0.3]), list("ABC"), 100_000.0)
    assert mc.display_paths.shape == (mc.n_days + 1, min(10_000, n_sims))
    assert mc.ending_values.shape == (n_sims,)
    for key in ("mean_ending", "mean_cagr", "ci_lower", "ci_upper", "prob_loss"):
        assert np.isfinite(stats[key])
    assert stats["ci_lower"] < stats["ci_upper"]


def test_monte_carlo_paths_figure_caps_display():
    mc = ae.MonteCarloSimulator(n_simulations=1_000, projection_years=1)
    mc.run(_mc_returns(), np.array([0.4, 0.3, 0.3]), list("ABC"), 100_000.0)
    fig = mc.get_paths_figure(n_display=50_000)  # asks for more than stored
    assert fig is not None


def test_monte_carlo_bootstrap():
    rng = np.random.default_rng(1)
    equity = pd.Series(100_000 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 800))))
    mc = ae.MonteCarloSimulator(n_simulations=2_000, projection_years=1)
    stats = mc.run_bootstrap(equity, float(equity.iloc[-1]))
    assert mc.ending_values.shape == (2_000,)
    assert np.isfinite(stats["mean_cagr"])


# ---------------------------------------------------------------------------
# Library-compat regressions
# ---------------------------------------------------------------------------

def test_no_positional_nan_to_num():
    """np.nan_to_num(x, 0) passes 0 as `copy`, which crashes on the read-only
    arrays pandas 3.0 returns. Keyword form only."""
    pattern = re.compile(r"nan_to_num\([^,)]+,\s*0\s*\)")
    for fname in SOURCE_FILES:
        with open(os.path.join(REPO, fname)) as f:
            assert not pattern.search(f.read()), f"positional nan_to_num misuse in {fname}"


def test_nan_to_num_on_readonly_array():
    """The exact failure mode from the first cloud deploy."""
    a = np.array([1.0, np.nan, 2.0])
    a.setflags(write=False)
    out = np.nan_to_num(a, nan=0.0)
    assert out[1] == 0.0


# ---------------------------------------------------------------------------
# Platform / demo-mode plumbing
# ---------------------------------------------------------------------------

def test_mlx_probe_is_consistent():
    assert isinstance(ae.RL_BACKEND_AVAILABLE, bool)
    if ae.RL_BACKEND_AVAILABLE:
        import mlx.core  # noqa: F401 — probe said yes, import must work
    else:
        assert ae.RL_BACKEND_ERROR


def test_demo_mode_constants_react_to_environment():
    assert ae.MC_SIM_MAX == (100_000 if ae.IS_HOSTED_DEMO else 1_000_000)
    code = ("import alpha_engine as ae; "
            "assert ae.IS_HOSTED_DEMO and ae.MC_SIM_DEFAULT == 50_000")
    env = dict(os.environ, HOSTNAME="streamlit")
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-500:]


def test_config_defaults():
    cfg = ae.StrategyConfig()
    assert 0 < cfg.ml_threshold < 1
    assert cfg.min_growth_anchor <= cfg.aggressive_ceiling
    assert cfg.rebalance_period > 0


# ---------------------------------------------------------------------------
# Full pipeline on synthetic data (the dependency-drift canary)
# ---------------------------------------------------------------------------

def test_full_pipeline_synthetic(synthetic_dm, aligned):
    (prices, returns, features, vix, sma_200, above_sma, raw_mom, rel_strength,
     vols, info_ratio, mom_score, golden_cross, log_ret_30d, rsi_14) = aligned

    assert len(prices) > 1500 and not features.empty

    config = ae.StrategyConfig()
    classifier = ae.AdaptiveRegimeClassifier(config)
    ml_probs = classifier.walk_forward_train(features, returns["SPY"])
    assert ml_probs.notna().sum() > 200

    categories = synthetic_dm.get_asset_categories()
    optimizer = ae.AlphaDominatorOptimizer(synthetic_dm.all_tickers, categories, config)
    engine = ae.BacktestEngine(config)
    results = engine.run(prices, returns, features, ml_probs, sma_200, above_sma,
                         raw_mom, rel_strength, vols, info_ratio, mom_score,
                         golden_cross, log_ret_30d, rsi_14, classifier, optimizer)
    metrics = engine.calculate_metrics(results)

    assert results["Portfolio"].iloc[-1] > 0
    assert np.isfinite(metrics["portfolio"]["sharpe"])
    assert -1 <= metrics["portfolio"]["max_drawdown"] <= 0
    weights = engine.final_weights
    assert weights is not None and abs(weights.sum() - 1.0) < 0.05


# ---------------------------------------------------------------------------
# Artifact round-trip
# ---------------------------------------------------------------------------

def test_classifier_artifact_roundtrip(tmp_path, synthetic_dm, aligned, monkeypatch):
    (_, returns, features, *_rest) = aligned
    config = ae.StrategyConfig()
    classifier = ae.AdaptiveRegimeClassifier(config)
    ml_probs = classifier.walk_forward_train(features, returns["SPY"])

    monkeypatch.setattr(ae, "ARTIFACT_PATH", str(tmp_path / "clf.joblib"))
    ae.save_classifier_artifact(classifier, ml_probs)

    loaded = ae._load_classifier_artifact(features)
    assert loaded is not None
    clf2, probs2 = loaded
    pd.testing.assert_series_equal(
        probs2.dropna(), ml_probs.reindex(features.index).ffill().dropna())

    # Schema drift must force a retrain, not a wrong-shape crash
    renamed = features.rename(columns={features.columns[0]: "different"})
    assert ae._load_classifier_artifact(renamed) is None
