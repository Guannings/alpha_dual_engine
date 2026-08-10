#!/usr/bin/env python3
"""Regenerate artifacts/classifier_cache.joblib.

Downloads current market data, runs the full walk-forward classifier
training, and writes the artifact that alpha_engine.py loads on startup
(fast cold starts, especially on the Streamlit Cloud demo). Run manually
after big feature changes, or let the weekly CI job do it.

Usage: python tools/refresh_artifacts.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf  # noqa: E402

# yfinance keeps a shared SQLite timezone cache. On CI runners a stale or
# contended cache throws "database is locked" mid-download, which silently
# empties the batch. Point it at a fresh per-run directory so this job can
# never inherit a locked cache from a previous run.
yf.set_tz_cache_location(tempfile.mkdtemp(prefix="yf_tz_cache_"))

from alpha_engine import (  # noqa: E402
    StrategyConfig, DataManager, AdaptiveRegimeClassifier, save_classifier_artifact,
)


def main() -> None:
    config = StrategyConfig()
    dm = DataManager(start_date='2010-01-01', config=config)
    dm.load_data()
    dm.engineer_features()
    (prices, returns, features, *_rest) = dm.get_aligned_data()

    classifier = AdaptiveRegimeClassifier(config)
    ml_probs = classifier.walk_forward_train(features, returns['SPY'])

    trained = ml_probs.notna().sum()
    if trained == 0:
        raise SystemExit("Training produced no probabilities — refusing to save artifact.")

    path = save_classifier_artifact(classifier, ml_probs)
    size_kb = os.path.getsize(path) / 1024
    print(f"Artifact written: {path} ({size_kb:.0f} KB, "
          f"{trained} trained days through {ml_probs.dropna().index[-1].date()})")


if __name__ == '__main__':
    main()
