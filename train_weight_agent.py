#!/usr/bin/env python3
"""Standalone 300K timestep PPO training script for the low-level weight agent."""
import sys
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['PYTHONUNBUFFERED'] = '1'

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)

from alpha_engine import DataManager, StrategyConfig, AlphaDominatorOptimizer
from rl_weight_agent import WeightRLTrainer
import pandas as pd

print("Loading data...", flush=True)
config = StrategyConfig()
dm = DataManager(start_date='2010-01-01', config=config)
dm.load_data()
dm.engineer_features()

(prices, returns, features, vix, sma_200, above_sma, raw_mom, rel_strength,
 vols, info_ratio, mom_score, golden_cross_df, log_ret_30d, rsi_14) = dm.get_aligned_data()
categories = dm.get_asset_categories()
optimizer = AlphaDominatorOptimizer(dm.all_tickers, categories, config)

print(f'Data: {len(prices)} days, {len(dm.all_tickers)} assets', flush=True)
print(f'Starting 300K timestep weight PPO training...', flush=True)
print('=' * 70, flush=True)

trainer = WeightRLTrainer(
    prices=prices, returns=returns, features=features,
    ml_probs=pd.Series(0.5, index=features.index),
    sma_200=sma_200, above_sma=above_sma, raw_momentum=raw_mom,
    relative_strength=rel_strength, asset_volatilities=vols,
    information_ratio=info_ratio, momentum_score=mom_score,
    golden_cross=golden_cross_df, log_returns_30d=log_ret_30d, rsi_14=rsi_14,
    optimizer=optimizer, config=config,
    train_end_date='2023-12-31',
    total_timesteps=300_000,
)
model = trainer.train()

print('=' * 70, flush=True)
print('Evaluating trained weight model (5 episodes)...', flush=True)
eval_results = trainer.evaluate(n_episodes=5)
print(f'  Mean reward:   {eval_results["mean_reward"]:.3f} +/- {eval_results["std_reward"]:.3f}', flush=True)
print(f'  Mean Sharpe:   {eval_results["mean_sharpe"]:.3f}', flush=True)
print(f'  Final PV:      ${eval_results["final_portfolio_value"]:,.0f}', flush=True)
print(f'  Final BM:      ${eval_results["final_benchmark_value"]:,.0f}', flush=True)
print(f'  Episodes logged: {len(trainer.reward_history)}', flush=True)
print('=' * 70, flush=True)
print('Done.', flush=True)
