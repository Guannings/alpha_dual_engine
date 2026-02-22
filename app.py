"""Streamlit dashboard for Alpha Dual Engine v154.6 backtest results."""
import sys, os, warnings, glob
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alpha_engine import (
    StrategyConfig, DataManager, AdaptiveRegimeClassifier,
    AlphaDominatorOptimizer, BacktestEngine
)
from rl_weight_agent import HierarchicalRLController, WEIGHT_MODEL_DIR, _load_weight_model

st.set_page_config(page_title="Alpha Dual Engine v154.6", layout="wide")
st.title("Alpha Dual Engine v154.6 — Backtest Dashboard")


@st.cache_data(show_spinner="Loading market data...")
def load_data():
    config = StrategyConfig()
    dm = DataManager(start_date='2010-01-01', config=config)
    dm.load_data()
    dm.engineer_features()
    data = dm.get_aligned_data()
    categories = dm.get_asset_categories()
    return dm, config, data, categories


@st.cache_data(show_spinner="Training baseline classifier...")
def train_classifier(_config, _features, _spy_returns):
    clf = AdaptiveRegimeClassifier(_config)
    ml_probs = clf.walk_forward_train(_features, _spy_returns)
    return clf, ml_probs


def run_baseline_backtest(dm, config, data, categories, ml_probs):
    prices, returns, features, vix, sma_200, above_sma, raw_mom, \
        rel_strength, vols, info_ratio, mom_score, golden_cross, \
        log_ret_30d, rsi_14 = data
    optimizer = AlphaDominatorOptimizer(dm.all_tickers, categories, config)
    engine = BacktestEngine(config)
    results = engine.run(
        prices, returns, features, ml_probs, sma_200, above_sma,
        raw_mom, rel_strength, vols, info_ratio, mom_score, golden_cross,
        log_ret_30d, rsi_14, AdaptiveRegimeClassifier(config), optimizer
    )
    # retrain classifier inside since we need a fresh one for baseline
    metrics = engine.calculate_metrics(results)
    return results, metrics


def run_rl_backtest(dm, config, data, categories, ml_probs, weight_model_path=None):
    prices, returns, features, vix, sma_200, above_sma, raw_mom, \
        rel_strength, vols, info_ratio, mom_score, golden_cross, \
        log_ret_30d, rsi_14 = data

    hier = HierarchicalRLController(config=config)
    if weight_model_path:
        hier.weight_agent.model = _load_weight_model(weight_model_path)
        hier.weight_agent.model_path = weight_model_path

    optimizer = AlphaDominatorOptimizer(dm.all_tickers, categories, config)
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
    return results, metrics


def find_checkpoints():
    """Find all available weight model checkpoints."""
    checkpoints = []
    for name in ['best_model', 'final_model']:
        path = os.path.join(WEIGHT_MODEL_DIR, name)
        if os.path.exists(path + '.json'):
            checkpoints.append((name, path))
    for f in sorted(glob.glob(os.path.join(WEIGHT_MODEL_DIR, 'checkpoint_*'))):
        if f.endswith('.json'):
            name = os.path.basename(f).replace('.json', '')
            path = f.replace('.json', '')
            checkpoints.append((name, path))
    return checkpoints


def plot_equity_curves(rl_results, baseline_results, oos_cutoff):
    """Plot equity curves with regime shading."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.05,
                        subplot_titles=("Equity Curves", "Regime"))

    # Portfolio curves
    fig.add_trace(go.Scatter(
        x=rl_results.index, y=rl_results['Portfolio'],
        name='RL Agent', line=dict(color='#2196F3', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=baseline_results.index, y=baseline_results['Portfolio'],
        name='Baseline', line=dict(color='#FF9800', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rl_results.index, y=rl_results['Benchmark'],
        name='SPY Benchmark', line=dict(color='#9E9E9E', width=1.5, dash='dot')
    ), row=1, col=1)

    # OOS cutoff line
    fig.add_shape(type="line", x0=str(oos_cutoff.date()), x1=str(oos_cutoff.date()),
                  y0=0, y1=1, yref="y domain", line=dict(dash="dash", color="red"),
                  row=1, col=1)

    # Regime plot
    regime_map = {'RISK_ON': 1, 'RISK_OFF': -1, 'NEUTRAL': 0}
    regime_numeric = rl_results['Regime'].map(regime_map).fillna(0)

    colors = regime_numeric.map({1: '#4CAF50', -1: '#F44336', 0: '#FFC107'})
    fig.add_trace(go.Bar(
        x=rl_results.index, y=regime_numeric,
        marker_color=colors.tolist(), name='Regime', showlegend=False
    ), row=2, col=1)

    fig.update_layout(
        height=600, template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=60, r=20, t=40, b=20)
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Regime", tickvals=[-1, 0, 1],
                     ticktext=['RISK_OFF', 'NEUTRAL', 'RISK_ON'], row=2, col=1)
    return fig


def plot_drawdown(results, label):
    """Plot drawdown chart."""
    pv = results['Portfolio']
    running_max = pv.cummax()
    drawdown = (pv - running_max) / running_max

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results.index, y=drawdown * 100,
        fill='tozeroy', name=label,
        line=dict(color='#F44336', width=1),
        fillcolor='rgba(244, 67, 54, 0.3)'
    ))
    fig.update_layout(
        height=300, template='plotly_dark',
        yaxis_title='Drawdown (%)', margin=dict(l=60, r=20, t=20, b=20)
    )
    return fig


def plot_checkpoint_comparison(checkpoint_data):
    """Plot checkpoint metrics comparison."""
    names = [d['name'] for d in checkpoint_data]
    sharpes = [d['full_sharpe'] for d in checkpoint_data]
    oos_sharpes = [d['oos_sharpe'] for d in checkpoint_data]
    decays = [d['decay'] for d in checkpoint_data]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Sharpe Ratios by Checkpoint",
                                        "Sharpe Decay by Checkpoint"))

    fig.add_trace(go.Bar(name='Full Sharpe', x=names, y=sharpes,
                         marker_color='#2196F3'), row=1, col=1)
    fig.add_trace(go.Bar(name='OOS Sharpe', x=names, y=oos_sharpes,
                         marker_color='#4CAF50'), row=1, col=1)

    decay_colors = ['#4CAF50' if d < 0.3 else '#FF9800' if d < 0.5 else '#F44336'
                    for d in decays]
    fig.add_trace(go.Bar(name='Decay', x=names, y=decays,
                         marker_color=decay_colors, showlegend=False), row=1, col=2)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="Target < 0.5", row=1, col=2)

    fig.update_layout(
        height=400, template='plotly_dark', barmode='group',
        margin=dict(l=60, r=20, t=40, b=20)
    )
    return fig


# --- Main App ---

dm, config, data, categories = load_data()
prices, returns, features, vix, sma_200, above_sma, raw_mom, \
    rel_strength, vols, info_ratio, mom_score, golden_cross, \
    log_ret_30d, rsi_14 = data

clf, ml_probs = train_classifier(config, features, returns['SPY'])

oos_cutoff = pd.Timestamp('2024-01-01')
rf = config.risk_free_rate

# Sidebar
st.sidebar.header("Configuration")
checkpoints = find_checkpoints()
checkpoint_names = [name for name, _ in checkpoints]
selected_ckpt = st.sidebar.selectbox(
    "Weight Model Checkpoint",
    checkpoint_names,
    index=0  # best_model
)
selected_path = dict(checkpoints)[selected_ckpt]

run_comparison = st.sidebar.checkbox("Compare all checkpoints", value=False)

# Run backtests
with st.spinner("Running RL backtest..."):
    rl_results, rl_metrics = run_rl_backtest(
        dm, config, data, categories, ml_probs, selected_path
    )

with st.spinner("Running baseline backtest..."):
    bl_results, bl_metrics = run_baseline_backtest(
        dm, config, data, categories, ml_probs
    )

# Compute period metrics
rl_oos = rl_results.index >= oos_cutoff
rl_is_m = BacktestEngine._period_metrics(rl_results[~rl_oos], rf)['portfolio']
rl_oos_m = BacktestEngine._period_metrics(rl_results[rl_oos], rf)['portfolio']
bl_oos = bl_results.index >= oos_cutoff
bl_is_m = BacktestEngine._period_metrics(bl_results[~bl_oos], rf)['portfolio']
bl_oos_m = BacktestEngine._period_metrics(bl_results[bl_oos], rf)['portfolio']

decay = rl_is_m['sharpe'] - rl_oos_m['sharpe']
sniper = rl_metrics.get('sniper_score')

# Target Scorecard
st.header(f"Scorecard — {selected_ckpt}")
cols = st.columns(4)
with cols[0]:
    val = rl_metrics['portfolio']['sharpe']
    st.metric("Full Sharpe", f"{val:.3f}", delta="PASS" if val > 0.95 else "FAIL")
with cols[1]:
    val = rl_metrics['portfolio']['cagr']
    st.metric("CAGR", f"{val:.2%}", delta="PASS" if val >= 0.21 else "FAIL")
with cols[2]:
    st.metric("Sharpe Decay", f"{decay:.3f}", delta="PASS" if decay < 0.5 else "FAIL")
with cols[3]:
    sn = sniper if sniper is not None else 0
    st.metric("Sniper Score", f"{sn:.3f}", delta="PASS" if sn > 0.6 else "FAIL")

# Equity Curves
st.header("Equity Curves")
st.plotly_chart(plot_equity_curves(rl_results, bl_results, oos_cutoff),
                use_container_width=True)

# Metrics Table
st.header("Detailed Metrics")
metrics_df = pd.DataFrame({
    'Metric': ['Full-Period Sharpe', 'In-Sample Sharpe', 'Out-of-Sample Sharpe',
               'Sharpe Decay', 'CAGR (Full)', 'CAGR (OOS)',
               'Max Drawdown', 'Volatility', 'Sniper Score', 'Final PV'],
    'RL Agent': [
        f"{rl_metrics['portfolio']['sharpe']:.3f}",
        f"{rl_is_m['sharpe']:.3f}",
        f"{rl_oos_m['sharpe']:.3f}",
        f"{decay:.3f}",
        f"{rl_metrics['portfolio']['cagr']:.2%}",
        f"{rl_oos_m['cagr']:.2%}",
        f"{rl_metrics['portfolio']['max_drawdown']:.2%}",
        f"{rl_metrics['portfolio']['volatility']:.2%}",
        f"{sniper:.3f}" if sniper else "N/A",
        f"${rl_metrics['portfolio']['final_value']:,.0f}",
    ],
    'Baseline': [
        f"{bl_metrics['portfolio']['sharpe']:.3f}",
        f"{bl_is_m['sharpe']:.3f}",
        f"{bl_oos_m['sharpe']:.3f}",
        f"{bl_is_m['sharpe'] - bl_oos_m['sharpe']:.3f}",
        f"{bl_metrics['portfolio']['cagr']:.2%}",
        f"{bl_oos_m['cagr']:.2%}",
        f"{bl_metrics['portfolio']['max_drawdown']:.2%}",
        f"{bl_metrics['portfolio']['volatility']:.2%}",
        f"{bl_metrics.get('sniper_score', 0):.3f}" if bl_metrics.get('sniper_score') else "N/A",
        f"${bl_metrics['portfolio']['final_value']:,.0f}",
    ],
})
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# Drawdown
st.header("Drawdown")
col1, col2 = st.columns(2)
with col1:
    st.subheader("RL Agent")
    st.plotly_chart(plot_drawdown(rl_results, "RL Agent"), use_container_width=True)
with col2:
    st.subheader("Baseline")
    st.plotly_chart(plot_drawdown(bl_results, "Baseline"), use_container_width=True)

# Checkpoint Comparison
if run_comparison:
    st.header("Checkpoint Comparison")
    checkpoint_data = []
    progress = st.progress(0)
    for i, (name, path) in enumerate(checkpoints):
        with st.spinner(f"Evaluating {name}..."):
            try:
                res, met = run_rl_backtest(dm, config, data, categories, ml_probs, path)
                oos_mask = res.index >= oos_cutoff
                is_m = BacktestEngine._period_metrics(res[~oos_mask], rf)['portfolio']
                oos_m_ckpt = BacktestEngine._period_metrics(res[oos_mask], rf)['portfolio']
                checkpoint_data.append({
                    'name': name,
                    'full_sharpe': met['portfolio']['sharpe'],
                    'oos_sharpe': oos_m_ckpt['sharpe'],
                    'decay': is_m['sharpe'] - oos_m_ckpt['sharpe'],
                    'cagr': met['portfolio']['cagr'],
                    'max_dd': met['portfolio']['max_drawdown'],
                    'sniper': met.get('sniper_score'),
                })
            except Exception as e:
                st.warning(f"{name}: {e}")
        progress.progress((i + 1) / len(checkpoints))

    if checkpoint_data:
        st.plotly_chart(plot_checkpoint_comparison(checkpoint_data),
                        use_container_width=True)

        ckpt_df = pd.DataFrame(checkpoint_data)
        ckpt_df = ckpt_df.rename(columns={
            'name': 'Checkpoint', 'full_sharpe': 'Sharpe', 'oos_sharpe': 'OOS Sharpe',
            'decay': 'Decay', 'cagr': 'CAGR', 'max_dd': 'Max DD', 'sniper': 'Sniper'
        })
        ckpt_df['CAGR'] = ckpt_df['CAGR'].map('{:.2%}'.format)
        ckpt_df['Max DD'] = ckpt_df['Max DD'].map('{:.2%}'.format)
        for c in ['Sharpe', 'OOS Sharpe', 'Decay']:
            ckpt_df[c] = ckpt_df[c].map('{:.3f}'.format)
        ckpt_df['Sniper'] = ckpt_df['Sniper'].map(
            lambda x: f"{x:.3f}" if x is not None else "N/A"
        )
        st.dataframe(ckpt_df, use_container_width=True, hide_index=True)
