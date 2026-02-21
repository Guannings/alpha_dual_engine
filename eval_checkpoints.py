"""Evaluate all saved weight model checkpoints to find the best one."""
import sys, os, warnings, glob
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd

from alpha_engine import (
    StrategyConfig, DataManager, AdaptiveRegimeClassifier,
    AlphaDominatorOptimizer, BacktestEngine
)
from rl_weight_agent import HierarchicalRLController, WEIGHT_MODEL_DIR

def evaluate_checkpoint(checkpoint_path, all_tickers, prices, returns, features,
                        ml_probs, sma_200, above_sma, raw_mom, rel_strength,
                        vols, info_ratio, mom_score, golden_cross, log_ret_30d,
                        rsi_14, categories, config):
    """Run backtest with a specific weight model checkpoint."""
    from rl_weight_agent import _load_weight_model
    hier = HierarchicalRLController(config=config)
    hier.weight_agent.model = _load_weight_model(checkpoint_path)
    hier.weight_agent.model_path = checkpoint_path

    optimizer = AlphaDominatorOptimizer(all_tickers, categories, config)
    if hasattr(hier, 'set_backtest_context'):
        hier.set_backtest_context(
            above_sma=above_sma, raw_momentum=raw_mom,
            asset_volatilities=vols, information_ratio=info_ratio,
            golden_cross=golden_cross, features=features, optimizer=optimizer,
        )

    engine = BacktestEngine(config)
    results = engine.run(
        prices, returns, features, ml_probs, sma_200, above_sma,
        raw_mom, rel_strength, vols, info_ratio, mom_score, golden_cross,
        log_ret_30d, rsi_14, hier, optimizer
    )
    metrics = engine.calculate_metrics(results)

    rf = config.risk_free_rate
    oos_cutoff = pd.Timestamp('2024-01-01')
    oos_mask = results.index >= oos_cutoff
    is_mask = ~oos_mask
    is_m = BacktestEngine._period_metrics(results[is_mask], rf)['portfolio']
    oos_m = BacktestEngine._period_metrics(results[oos_mask], rf)['portfolio']

    return {
        'full_sharpe': metrics['portfolio']['sharpe'],
        'is_sharpe': is_m['sharpe'],
        'oos_sharpe': oos_m['sharpe'],
        'decay': is_m['sharpe'] - oos_m['sharpe'],
        'cagr': metrics['portfolio']['cagr'],
        'max_dd': metrics['portfolio']['max_drawdown'],
        'sniper': metrics.get('sniper_score'),
        'final_pv': metrics['portfolio']['final_value'],
    }

def main():
    print("=" * 70)
    print("Checkpoint Evaluation — Finding Best Weight Model")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    config = StrategyConfig()
    dm = DataManager(start_date='2010-01-01', config=config)
    dm.load_data()
    dm.engineer_features()

    prices, returns, features, vix, sma_200, above_sma, raw_mom, \
        rel_strength, vols, info_ratio, mom_score, golden_cross, \
        log_ret_30d, rsi_14 = dm.get_aligned_data()
    categories = dm.get_asset_categories()

    # Train baseline for ml_probs
    print("[2/3] Training baseline classifier...")
    baseline_classifier = AdaptiveRegimeClassifier(config)
    ml_probs = baseline_classifier.walk_forward_train(features, returns['SPY'])

    # Find all checkpoints
    print("[3/3] Evaluating checkpoints...\n")
    checkpoints = []
    for name in ['best_model', 'final_model']:
        path = os.path.join(WEIGHT_MODEL_DIR, name)
        cfg_path = path + '.json'
        if os.path.exists(cfg_path):
            checkpoints.append((name, path))

    for f in sorted(glob.glob(os.path.join(WEIGHT_MODEL_DIR, 'checkpoint_*'))):
        if f.endswith('.json'):
            name = os.path.basename(f).replace('.json', '')
            path = f.replace('.json', '')
            checkpoints.append((name, path))

    if not checkpoints:
        print("No checkpoints found!")
        return

    results = []
    for name, path in checkpoints:
        print(f"  Evaluating {name}...", end=" ", flush=True)
        try:
            m = evaluate_checkpoint(
                path, dm.all_tickers, prices, returns, features, ml_probs,
                sma_200, above_sma, raw_mom, rel_strength, vols,
                info_ratio, mom_score, golden_cross, log_ret_30d,
                rsi_14, categories, config
            )
            results.append((name, m))
            print(f"Sharpe={m['full_sharpe']:.3f} CAGR={m['cagr']:.2%} "
                  f"OOS={m['oos_sharpe']:.3f} Decay={m['decay']:.3f}", flush=True)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Checkpoint':<20} {'Sharpe':>8} {'CAGR':>8} {'OOS':>8} {'Decay':>8} {'MaxDD':>8} {'Sniper':>8}")
    print("-" * 70)
    for name, m in results:
        sn = f"{m['sniper']:.3f}" if m['sniper'] is not None else "N/A"
        print(f"{name:<20} {m['full_sharpe']:>8.3f} {m['cagr']:>7.2%} "
              f"{m['oos_sharpe']:>8.3f} {m['decay']:>8.3f} {m['max_dd']:>7.2%} {sn:>8}")

    # Find best by composite score: prioritize full Sharpe + bonus for low decay
    best_name, best_m = max(results, key=lambda x: x[1]['full_sharpe'] - 0.1 * max(x[1]['decay'] - 0.3, 0))
    print(f"\nBest checkpoint: {best_name}")
    print(f"  Full Sharpe: {best_m['full_sharpe']:.3f}, CAGR: {best_m['cagr']:.2%}, "
          f"OOS: {best_m['oos_sharpe']:.3f}, Decay: {best_m['decay']:.3f}")

if __name__ == '__main__':
    main()
