"""Headless evaluation: OOS Sharpe decay + Sniper Score."""
import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd

from alpha_engine import (
    StrategyConfig, DataManager, AdaptiveRegimeClassifier,
    AlphaDominatorOptimizer, BacktestEngine
)

def main():
    print("=" * 70)
    print("OOS Sharpe Decay & Sniper Score Evaluation")
    print("=" * 70)

    # 1. Load data
    print("\n[1/5] Loading market data...")
    config = StrategyConfig()
    dm = DataManager(start_date='2010-01-01', config=config)
    dm.load_data()
    dm.engineer_features()

    prices, returns, features, vix, sma_200, above_sma, raw_mom, \
        rel_strength, vols, info_ratio, mom_score, golden_cross, \
        log_ret_30d, rsi_14 = dm.get_aligned_data()
    categories = dm.get_asset_categories()

    # 2. Train baseline classifier (rule-based + XGBoost)
    print("[2/5] Training baseline classifier (walk-forward)...")
    baseline_classifier = AdaptiveRegimeClassifier(config)
    baseline_ml_probs = baseline_classifier.walk_forward_train(features, returns['SPY'])

    # 3. Run baseline backtest
    print("[3/5] Running baseline backtest...")
    optimizer = AlphaDominatorOptimizer(dm.all_tickers, categories, config)
    baseline_engine = BacktestEngine(config)
    baseline_results = baseline_engine.run(
        prices, returns, features, baseline_ml_probs, sma_200, above_sma,
        raw_mom, rel_strength, vols, info_ratio, mom_score, golden_cross,
        log_ret_30d, rsi_14, baseline_classifier, optimizer
    )
    baseline_metrics = baseline_engine.calculate_metrics(baseline_results)

    # 4. Run hierarchical RL backtest
    print("[4/5] Running Hierarchical RL backtest...")
    from rl_weight_agent import HierarchicalRLController
    hier_controller = HierarchicalRLController(config=config)

    if hier_controller.regime_agent.model is None:
        print("ERROR: No regime model found. Run train_100k.py first.")
        sys.exit(1)
    if hier_controller.weight_agent.model is None:
        print("ERROR: No weight model found. Run train_weight_agent.py first.")
        sys.exit(1)

    rl_ml_probs = baseline_ml_probs

    optimizer2 = AlphaDominatorOptimizer(dm.all_tickers, categories, config)
    if hasattr(hier_controller, 'set_backtest_context'):
        hier_controller.set_backtest_context(
            above_sma=above_sma,
            raw_momentum=raw_mom,
            asset_volatilities=vols,
            information_ratio=info_ratio,
            golden_cross=golden_cross,
            features=features,
            optimizer=optimizer2,
        )

    rl_engine = BacktestEngine(config)
    rl_results = rl_engine.run(
        prices, returns, features, rl_ml_probs, sma_200, above_sma,
        raw_mom, rel_strength, vols, info_ratio, mom_score, golden_cross,
        log_ret_30d, rsi_14, hier_controller, optimizer2
    )
    rl_metrics = rl_engine.calculate_metrics(rl_results)

    # 5. Compute OOS metrics
    print("[5/5] Computing OOS metrics...\n")

    rl_train_cutoff = pd.Timestamp('2024-01-01')
    rf = config.risk_free_rate

    rl_oos_mask = rl_results.index >= rl_train_cutoff
    rl_is_mask = ~rl_oos_mask
    bl_oos_mask = baseline_results.index >= rl_train_cutoff
    bl_is_mask = ~bl_oos_mask

    rl_is = BacktestEngine._period_metrics(rl_results[rl_is_mask], rf)['portfolio']
    rl_oos = BacktestEngine._period_metrics(rl_results[rl_oos_mask], rf)['portfolio']
    bl_is = BacktestEngine._period_metrics(baseline_results[bl_is_mask], rf)['portfolio']
    bl_oos = BacktestEngine._period_metrics(baseline_results[bl_oos_mask], rf)['portfolio']

    rl_sharpe_decay = rl_is['sharpe'] - rl_oos['sharpe']
    bl_sharpe_decay = bl_is['sharpe'] - bl_oos['sharpe']

    sniper_score = rl_metrics.get('sniper_score')
    bl_sniper_score = baseline_metrics.get('sniper_score')

    # Print results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'RL Agent':>12} {'Baseline':>12}")
    print("-" * 54)
    print(f"{'Full-Period Sharpe':<30} {rl_metrics['portfolio']['sharpe']:>12.3f} {baseline_metrics['portfolio']['sharpe']:>12.3f}")
    print(f"{'In-Sample Sharpe':<30} {rl_is['sharpe']:>12.3f} {bl_is['sharpe']:>12.3f}")
    print(f"{'Out-of-Sample Sharpe':<30} {rl_oos['sharpe']:>12.3f} {bl_oos['sharpe']:>12.3f}")
    print(f"{'Sharpe Decay (IS - OOS)':<30} {rl_sharpe_decay:>12.3f} {bl_sharpe_decay:>12.3f}")
    print(f"{'CAGR (Full)':<30} {rl_metrics['portfolio']['cagr']:>11.2%} {baseline_metrics['portfolio']['cagr']:>11.2%}")
    print(f"{'CAGR (OOS)':<30} {rl_oos['cagr']:>11.2%} {bl_oos['cagr']:>11.2%}")
    print(f"{'Max Drawdown (Full)':<30} {rl_metrics['portfolio']['max_drawdown']:>11.2%} {baseline_metrics['portfolio']['max_drawdown']:>11.2%}")

    sn_str = f"{sniper_score:.3f}" if sniper_score is not None else "N/A"
    bl_sn_str = f"{bl_sniper_score:.3f}" if bl_sniper_score is not None else "N/A"
    print(f"{'Sniper Score':<30} {sn_str:>12} {bl_sn_str:>12}")

    print(f"\n{'Final PV':<30} ${rl_metrics['portfolio']['final_value']:>11,.0f} ${baseline_metrics['portfolio']['final_value']:>11,.0f}")

    # Targets
    print("\n" + "=" * 70)
    print("TARGET CHECK")
    print("=" * 70)

    decay_ok = rl_sharpe_decay < 0.5
    sniper_ok = sniper_score is not None and sniper_score > 0.6

    print(f"  OOS Sharpe Decay: {rl_sharpe_decay:.3f}  (target < 0.5)  {'PASS' if decay_ok else 'FAIL'}")
    print(f"  Sniper Score:     {sn_str}  (target > 0.6)  {'PASS' if sniper_ok else 'FAIL'}")
    print(f"  Baseline Decay:   {bl_sharpe_decay:.3f}  (reference)")
    print()

if __name__ == '__main__':
    main()
