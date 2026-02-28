# Alpha Dual Engine — RL Development State

Last updated: 2026-02-28

## Current Best Model

- **Checkpoint**: `checkpoint_50k` (promoted to `best_model`)
- **Source**: Aggressive regularization run (reward clip [-3,3], noise 0.10, LR decay to 20%)
- **Training**: 50K steps of 300K total (early-stopped at 60K)

## Performance (All Targets Passing)

| Metric | Value | Target | Status |
|---|---|---|---|
| Full-Period Sharpe | 0.992 | >0.95 | PASS |
| CAGR | 21.67% | >=21% | PASS |
| OOS Sharpe | 1.060 | — | Best across all configs |
| Sharpe Decay (IS - OOS) | -0.076 | <0.5 | PASS (negative = OOS better than IS) |
| Sniper Score | 0.711 | >0.6 | PASS |
| Max Drawdown | -40.68% | — | Baseline is -34.40% |

## Training Configurations Tried

### Run 1: Baseline 300K (ent_coef=0.10)
- `total_timesteps=300_000`, no regularization
- Final reward: +1.23
- Result: Sharpe 0.814, OOS 0.486, Decay 0.419, CAGR 18.32%

### Run 2: Extended 500K (ent_coef=0.10)
- `total_timesteps=500_000`, no regularization
- Final reward: +3.42 (much higher = overfitting)
- Result: Sharpe 0.779, OOS 0.451, Decay 0.423, CAGR 19.00%
- **Conclusion**: More training = worse. Confirmed overfitting.

### Run 3: Aggressive anti-overfit (ent_coef=0.10)
- Reward clip [-3, +3], noise 0.10, LR decay to 20%, patience=5
- Early-stopped at 60K steps
- **checkpoint_50k from this run is the current best model**
- Result at 50K: Sharpe 0.992, OOS 1.060, Decay -0.076
- Result at 60K: Sharpe 0.978, OOS 0.831, Decay 0.191

### Run 4: Moderated anti-overfit (ent_coef=0.10)
- Reward clip [-5, +5], noise 0.05, LR decay to 50%, patience=999
- Checkpoints every 30K steps, completed full 300K
- Final reward: +1.53
- Best checkpoint (241k): Sharpe 0.914, CAGR 21.72%
- All checkpoints worse than Run 3's checkpoint_50k

## All Checkpoint Evaluation Results

| Checkpoint | Sharpe | CAGR | OOS Sharpe | Decay | Max DD | Sniper |
|---|---|---|---|---|---|---|
| **checkpoint_50k** | **0.992** | **21.67%** | **1.060** | **-0.076** | -40.68% | 0.711 |
| checkpoint_60k | 0.978 | 21.35% | 0.831 | 0.191 | -32.58% | 0.722 |
| checkpoint_241k | 0.914 | 21.72% | 0.620 | 0.378 | -40.01% | 0.686 |
| final_model | 0.909 | 21.06% | 0.671 | 0.301 | -38.97% | 0.750 |
| checkpoint_211k | 0.867 | 20.81% | 0.609 | 0.330 | -40.86% | 0.686 |
| checkpoint_181k | 0.857 | 20.58% | 0.621 | 0.303 | -40.12% | 0.686 |
| checkpoint_151k | 0.840 | 20.22% | 0.580 | 0.335 | -38.07% | 0.686 |
| checkpoint_271k | 0.837 | 20.29% | 0.593 | 0.312 | -43.65% | 0.686 |
| checkpoint_120k | 0.822 | 19.78% | 0.617 | 0.264 | -40.11% | 0.686 |
| checkpoint_90k | 0.793 | 19.04% | 0.577 | 0.283 | -37.34% | 0.686 |
| checkpoint_30k | 0.775 | 18.45% | 0.495 | 0.368 | -31.68% | 0.686 |
| best_model (eval) | 0.660 | 16.30% | 0.553 | 0.140 | -38.77% | 0.667 |

## Ablation Test Results

| Model | Sharpe | CAGR | OOS Sharpe | Max DD |
|---|---|---|---|---|
| RL Agent (trained) | 0.992 | 21.67% | 1.061 | -40.68% |
| Random + constraints | 0.761 | 17.27% | 0.456 | -45.46% |
| Equal (1/N) + constraints | 0.832 | 18.38% | 0.876 | -35.84% |
| Baseline (rule-based) | 0.986 | 22.17% | 0.922 | -34.40% |

**RL value-add**: +0.231 Sharpe over random, +0.160 over equal, +0.006 over baseline.
**OOS value-add**: +0.605 over random, +0.185 over equal, +0.139 over baseline.

## Key Files

| File | Purpose |
|---|---|
| `alpha_engine.py` | Main strategy + Streamlit dashboard |
| `rl_regime_agent.py` | High-level PPO regime agent (MLX) |
| `rl_weight_agent.py` | Low-level PPO weight agent (training env + inference) |
| `train_100k.py` | Train regime agent (100K steps) |
| `train_weight_agent.py` | Train weight agent (currently 300K steps) |
| `eval_oos.py` | Full OOS evaluation with baseline comparison |
| `eval_checkpoints.py` | Evaluate all saved checkpoints |
| `models/rl_regime_ppo/` | Regime model weights |
| `models/rl_weight_ppo/` | Weight model weights + checkpoints |

## Key Hyperparameters (current best config)

```python
# rl_weight_agent.py — WeightRLTrainer
ent_coef = 0.10          # Entropy coefficient (exploration)
lr = 3e-4                # Initial learning rate
gamma = 0.99             # Discount factor
clip_range = 0.2         # PPO clip
n_steps = 2048           # Rollout length
batch_size = 64          # Mini-batch size
n_epochs = 10            # PPO epochs per update
noise_sigma = 0.10       # Observation noise (aggressive run)
reward_clip = [-3, +3]   # Total reward clipping
lr_decay = 0.20          # Final LR as fraction of initial

# Constraint penalties (ConstraintLayer.compute_penalties)
concentration_scale = 5.0   # Per-asset cap at 30%
gold_scale = 3.0            # Gold cap at 1% in RISK_ON
crypto_floor_scale = 1.5    # Crypto floor at 5%
crypto_cap_scale = 2.0      # Crypto cap (regime-dependent)
ineligible_scale = 4.0      # Below-SMA penalty
ga_floor_scale = 3.0        # Growth anchor floor at 40%
```

## Known Tradeoffs

1. **RL has higher max drawdown** (-40.68%) vs baseline (-34.40%). The agent is more aggressive in RISK_ON, which helps OOS Sharpe but hurts worst-case.
2. **Constraint drift is ~0.61** — the constraint layer does meaningful correction. This is expected and healthy for a 50K-step model.
3. **The 50K model came from a "failed" run** — the aggressive anti-overfit run that early-stopped at 60K was initially dismissed, but its 50K checkpoint turned out to be the best.

## If You Want to Retrain

```bash
# 1. Train regime agent (fast, ~2 min)
python train_100k.py

# 2. Train weight agent (slower, ~15 min)
python train_weight_agent.py

# 3. Evaluate all checkpoints
python eval_checkpoints.py

# 4. Full OOS comparison
python eval_oos.py

# 5. Launch dashboard
streamlit run alpha_engine.py
```

## Potential Improvements to Explore

- **Reduce max drawdown**: Add drawdown penalty to reward function or increase defensive allocation during high-vol periods
- **Ensemble checkpoints**: Average predictions from top-3 checkpoints instead of using a single one
- **Curriculum learning**: Start with large noise/clip, gradually reduce as training progresses
- **Per-regime weight models**: Train separate weight agents for RISK_ON vs DEFENSIVE
- **Transaction cost awareness**: Add turnover penalty directly into the RL reward

## Streamlit Notes

- Safari HTTPS-only mode blocks localhost — disable with: `defaults write com.apple.Safari UseHTTPSOnly -bool false` then restart Safari
- Re-enable after: `defaults write com.apple.Safari UseHTTPSOnly -bool true`
- Dashboard runs at `http://localhost:8501`
- Enable RL via sidebar checkbox "Use Hierarchical RL (Regime + Weights)"
- Ablation test is in RL Diagnostics tab (checkbox to run)

---

## Session: 2026-02-27 — README Math Appendix & Polish

### What was added to README.md

**1. Math Appendix — "Appendix: Mathematical Foundations — Deep Dive"**

Added before the Development Methodology section. Includes:

- **Table of Contents** — formula overview table (with Formula Name column) + clickable anchor links to every subsection
- **Foundational Concepts (Section 0)** — loss functions (MSE, loss vs reward, the 3 losses in the project) and gradient descent (update rule, learning rate, worked example, connection table)
- **Section A: Objective Function & SLSQP** — 3-term objective breakdown, how SLSQP works, hand-solvability (Layers 1–4), Lagrange multipliers (buffet analogy, worked example, geometric intuition, shadow prices), Gaussian elimination (1/2/3-variable worked examples, general pattern), active set method, complete picture table
- **Section B: Shannon Entropy** — formula, why ln shows up, concrete examples, Effective N
- **Section C: GBM** — SDE, Ito's Lemma, volatility drag, simulation formula, worked example, 1M paths
- **Section D: PPO** — policy (discrete/continuous), value function, GAE, clipped surrogate, full loss, training loop, hierarchical architecture, why PPO vs alternatives

**2. Author's Design Decisions (bottom of README)**

Documents all decisions made by the author (not AI):
- Strategy architecture: bifurcated logic, hierarchical RL, 12-asset universe
- Risk & parameter engineering: 25% vol, 12% turnover gate, 1% gold cap, 40% growth floor, cubed momentum
- RL training: reward function, bypassing biased regime agent, 50K checkpoint selection, drift bug
- References Section X documented failures

**3. Professionalized all informal text (~15 instances)**

- Removed all "In Plain English:" headers
- Removed Q&A blocks ("What is PPO?", "What is XGBoost?", "What is RSI?", etc.)
- Replaced coaching analogies (exam/homework, jury, sculptor, "updates its brain")
- Converted to formal prose
- Rewrote OOS table annotations

**4. LaTeX fixes**

- `\;` → `~` (was rendering as semicolons)
- `\,` → `~` (was rendering as commas)
- Removed LaTeX from markdown link text (GitHub can't render it)
- Removed `\boxed{}` (inconsistent support)
- Fixed mixed `\$` currency / `$...$` LaTeX conflicts
- Simplified GBM formulas in TOC table
- Fixed complex inline formula in PPO Technical Summary

### What was added to INTERVIEW_PREP.md (private, gitignored)

- Loss Functions section (concept + 3 project losses)
- Gradient Descent section (analogy, update rule, learning rate, SLSQP vs PPO)
- Lagrange Multipliers section (buffet analogy, mechanics, geometric intuition, shadow prices)
- Gaussian Elimination section (two-phase method, example, portfolio connection)
- 4 new one-liner entries in quick-response table

### Git commits this session (run `git log --oneline -15` to see)

Key commits:
- `dafa39f` Fix broken LaTeX in PPO Technical Summary
- `867f4ff` Fix two remaining LaTeX rendering issues
- `d8968f9` Add Author's Design Decisions section
- `fa5539c` Rewrite all informal/interview-style text as professional documentation
- `c1e41e1` Fix broken LaTeX in TOC sub-section links
- `3ed442a` Add Formula Name column to TOC table
- `3e142ad` Fix TOC: replace plain text with actual formulas
- `f4ae4cf` Fix broken LaTeX in GBM table of contents row
- `6d3083e` Add table of contents to mathematical appendix
- `834dfe7` Add gradient descent explanation, reorganize foundational concepts
- `84e8ea4` Add loss function explanation as foundational concept
- `199c27f` Add Gaussian elimination explanation
- `0cf803c` Add Lagrange multiplier deep-dive

### Where to pick up next

- README math appendix and main body are complete and professional
- All LaTeX verified clean (no \; \, \boxed, no LaTeX in links)
- Could add more math concepts if desired
- Could refine the Streamlit dashboard or code

---

## Session: 2026-02-28 — LaTeX Fixes, Content Additions & Narrative Fix

### LaTeX fixes in README.md

Multiple rounds of broken inline LaTeX found and fixed:

1. **Bifurcated Signal Engine (line 294):** `$\text{SMA}_{60,i}$` and `$\frac{P_i}{\text{SMA}_{60,i}}$` → simplified to `$SMA_{60,i}$` and `$P_i / SMA_{60,i}$`
2. **Objective function terms (lines 443-453):** `$\lambda_{\text{risk}}$`, `$\mathbf{w}$` → simplified to `$\lambda_{risk}$`, `$w$`; removed `\text{}` and `\mathbf{}` wrappers
3. **Min/max equivalence:** `$\min(-x) \equiv \max(x)$` → plain text: "minimizing $-x$ is equivalent to maximizing $x$"
4. **Max Drawdown example (line 635):** `\$150,000` currency signs conflicting with LaTeX delimiters → "150,000 dollars"; `$\frac{...}$` → inline division
5. **Volatility drag (line 670):** Same `\$` currency conflict; `$\frac{1}{2}\sigma^2$` → `$\sigma^2 / 2$`
6. **Observation noise (line 838):** `$\pm$10%` → `$\pm 10\%$` (closing `$` was against `1`)
7. **PPO formula in TOC (line 1014):** Removed `~` rendering as extra comma; simplified `\text{CLIP}` → `CLIP` and `\mathbb{E}` → `E`
8. **Hessian matrix formula:** Removed `\\[6pt]` spacing that rendered as visible backslash
9. **Ito's Lemma worked example (line 674):** `$\mu_{\text{adj}}$` → `$\mu_{adj}$`; `$\frac{1}{2}(0.25)^2$` → `$(0.25)^2 / 2$`

**Pattern confirmed:** GitHub inline `$...$` breaks with `\text{}`, `\mathbf{}`, `\mathbb{}`, nested `\frac{}`, `\min`, `\max`, and `\$` currency adjacent to math delimiters. Block `$$...$$` handles these fine. Fix strategy: simplify inline or move to block.

### Content additions to README.md

1. **"Why minimization achieves three goals at once"** — new subsection in Appendix Section A explaining the negative sign convention (minimize $-x$ = maximize $x$)
2. **Expanded "Why you CANNOT solve this analytically"** — added active set combinatorial argument: ~20 inequality constraints → $2^{20} \approx 1{,}000{,}000$ combinations; SLSQP converges in 20-50 iterations instead
3. **Worked Example: One Complete SLSQP Sub-Problem** — 3-asset portfolio with concrete numbers: build Lagrangian → take derivatives → solve by Gaussian elimination → interpret weights [7/12, 1/12, 4/12] and multiplier
4. **Hessian matrix explanation** — what it is (table of second derivatives), step-by-step derivation from $f(x,y) = 3x^2 + 2xy + 5y^2$ showing where each number comes from, diagonal vs off-diagonal meaning, 12x12 portfolio case, BFGS approximation, one-sentence summary
5. **Hessian added to TOC** — both the formula overview table and the sub-sections navigation list
6. **Cross-reference links** — added clickable links to Appendix Section B (Shannon Entropy) from Term 3, Section C (GBM) from Ito's Lemma paragraph, and Section D (PPO) from RL architecture paragraph
7. **Fixed Foundational Concepts anchor** — `#foundational-concepts` → `#0-foundational-concepts`

### Narrative fix in README.md

- **Lesson 2 "Regularization Overcorrection"** contradicted the "Overfitting Prevention" section — both described the same settings (reward clip [-3,+3], noise 0.10, LR decay 20%) but one called them a failure and the other the production config. Rewrote Lesson 2 to explain the run *appeared* to fail but its 50K checkpoint was the breakthrough, connecting to Lesson 3 and acknowledging these are now the production settings.

### Text improvements

- Reworded "To make the data accessible for career-level presentations" → "To facilitate interpretation and communication of results"

### What was added to INTERVIEW_PREP.md (private, gitignored)

- Hessian matrix one-liner in quick-response table
- Hessian matrix cheat sheet section (gradient vs Hessian, diagonal/off-diagonal meaning, BFGS, interview-ready explanation)

### Where to pick up next

- README appendix now covers: loss functions, gradient descent, SLSQP (with Hessian, Lagrange multipliers, Gaussian elimination, worked sub-problem example, active set), Shannon Entropy, GBM, PPO
- All LaTeX verified clean through multiple passes
- Could add more math concepts (e.g., Taylor expansion, covariance matrix deep-dive)
- Could refine the Streamlit dashboard or code
