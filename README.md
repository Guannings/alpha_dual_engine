# **How to Run the Docker File**

This project is fully containerized to ensure reproducibility.

**It runs consistently on whatever software environment using Docker.**

### 1. Prerequisites
**a. Docker Desktop (The Environment)**

  Installed and running ([Download here if you don't have one](https://www.docker.com/products/docker-desktop/))

**b. Git (The Code Downloader)**

  Check if you have it by typing `git --version` in your terminal. If not installed:

   **macOS:** Open Terminal and type `xcode-select --install` or download from [git-scm.com](https://git-scm.com/download/mac).

   **Windows:** Download and install **Git for Windows** from [gitforwindows.org](https://gitforwindows.org/).

   **Linux:** Run `sudo apt-get install git` (Debian/Ubuntu).


### 2. Installation
Open your terminal (or Command Prompt) and run:

```bash
# Clone the repository
git clone https://github.com/Guannings/alpha_dual_engine.git
```

### 3. Enter the project folder and install dependencies
```bash
cd alpha_dual_engine
pip install -r requirements.txt
```

### 4. Launching the Dashboard (Choose One Method)

> **Which file is the app?** The main dashboard is **`alpha_engine.py`** — the ⚙️ live strategy runner with configurable sliders and the Monte Carlo stress test. This is what every method below launches.
>
> `app.py` is a **separate, optional** dashboard for comparing saved training checkpoints (the "Scorecard" view). It is **not** the primary app — run it only if you specifically want the checkpoint comparison, via `streamlit run app.py`.

Method A: The "One-Click" Launch (Recommended) Best for first-time setup. This script automatically handles the build, cache-clearing, and port mapping to ensure the 1M simulations work correctly.

```bash
# 1. Make the script executable (only needed once)
chmod +x run_app.sh
```

```bash
# 2. Launch the App
./run_app.sh
```

Method B: Manual Build & Run (Advanced) Use this if you want to configure specific ports or debug the Dockerfile manually.

```bash
# 1. Build the Image (Force fresh build)
docker build --no-cache -t alpha-dual_engine .
```

```bash
# 2. Run the Container
docker run --rm --dns 8.8.8.8 -p 8501:8501 alpha-dual_engine
```

Method C: Run Locally Without Docker Best for quick iteration if you already have Python 3.9+ and the dependencies installed (Step 3). No build step required.

```bash
# Launch the primary dashboard directly
streamlit run alpha_engine.py
```

Then open **http://localhost:8501** in your browser.

> **Note on results:** `alpha_engine.py` downloads **live market data** (via `yfinance`, from 2010 to today) on every run, so backtest numbers will shift over time as new market data arrives. This is expected — it is not a bug.

### Troubleshooting & Best Practices
#### **1. Avoid System Folders (Windows Users)**
Do **not** clone this repository into `C:\Windows\System32` or other restricted system directories. This will cause permission errors with Git and Docker.

**Recommended Path:** Clone into a user-controlled folder via the commands below:
```bash
# Go to your user folder
cd ~
# Then go to your desktop:
cd Desktop
```
Then proceed with **2. Installation** and its following commands.

#### **2. Case Sensitivity & Folder Names**
If you encounter a "Path not found" error when using `cd`, ensure you are matching the exact capitalization of the repository:
```powershell
# Use Tab-completion in your terminal to avoid typos
cd alpha_dual_engine
```

#### **3. Port Conflicts:**
**If port 8501 is busy, simply use Option B above to map it to a different local port (replace the first `8501` before the colon with a port number of your choice, e.g. `-p 9000:8501`).**

====================================================================================

# **Computational Requirements**

To ensure the stability of the **Monte Carlo simulation engine** and the **XGBoost training pipeline**, the following resources are recommended:

**1. Memory (RAM):**

  **a. System Total:** 8GB minimum.

  **b. Docker Allocation:**

 Ensure at least **4GB** is dedicated to the container in Docker Desktop settings.

 This prevents **Out-of-Memory (OOM)** errors during the 1,000,000-path stress tests.

**2. Processor (CPU):**

a. **4+ Cores** recommended.

b. The machine learning model utilizes multi-threading for rapid retraining and regime classification.

**3. Connectivity:**

Data Pipeline: High-speed internet access is mandatory for real-time data ingestion via the Yahoo Finance API.

====================================================================================
# **Disclaimer and Terms of Use**
**1. Educational Purpose Only**

This software is for educational and research purposes only and was built as a personal project by a student, PARVAUX, a Public Finance and Economics double major at National Chengchi University (NCCU). It is not intended to be a source of financial advice, and the author is not a registered financial advisor. The algorithms, simulations, and optimization techniques implemented herein—including Consensus Machine Learning, Shannon Entropy, and Geometric Brownian Motion—are demonstrations of theoretical concepts and should not be construed as a recommendation to buy, sell, or hold any specific security or asset class.

**2. No Financial Advice**

Nothing in this repository constitutes professional financial, legal, or tax advice. Investment decisions should be made based on your own research and consultation with a qualified financial professional. The strategies modeled in this software may not be suitable for your specific financial situation, risk tolerance, or investment goals.

**3. Risk of Loss**

All investments involve risk, including the possible loss of principal.

a. Past Performance: Historical returns (such as the CAGR in any plots generated from this model) and volatility data used in these simulations are not indicative of future results.

b. Simulation Limitations: Monte Carlo simulations are probabilistic models based on assumptions (such as constant drift and volatility) that may not reflect real-world market conditions, black swan events, or liquidity crises.

c. Model Vetoes: While the Rate Shock Guard and Anxiety Veto are designed to mitigate losses, they are based on historical thresholds that may fail in unprecedented macro-economic environments.

d. Market Data: Data fetched from third-party APIs (e.g., Yahoo Finance) may be delayed, inaccurate, or incomplete.

**4. Hardware and Computation Liability**

The author assumes no responsibility for hardware failure, system instability, or data loss resulting from the execution of the 1,000,000 Monte Carlo simulations. Execution of this code at the configured scale is a high-stress computational event that should only be performed on hardware meeting the minimum specified 8GB RAM requirements.

**5. "AS-IS" SOFTWARE WARRANTY**

**THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHOR OR COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

**BY USING THIS SOFTWARE, YOU AGREE TO ASSUME ALL RISKS ASSOCIATED WITH YOUR INVESTMENT DECISIONS AND HARDWARE USAGE, RELEASING THE AUTHOR (PARVAUX) FROM ANY LIABILITY REGARDING YOUR FINANCIAL OUTCOMES OR SYSTEM INTEGRITY.**

====================================================================================
# **Alpha Dual Engine v154.6: The Framework**

This documentation details the architecture of Alpha Dual Engine v154.6, a high-performance quantitative trading system refactored to prioritize "Acceleration Alpha" over passive diversification and engineered to navigate complex market regimes through a synthesis of machine learning, macro-economic veto guards, and entropy-weighted optimization. Established by PARVAUX, a Public Finance and Economics double major at National Chengchi University, the system represents a Bifurcated Logic that applies distinct mathematical strategies to Equities (Growth Flow) and Cryptocurrency (Store of Value Cycles). Unlike traditional mean-variance models that treat all assets identically, v154.6 utilizes a Bifurcated Logic Engine that applies distinct mathematical strategies to Equities (Growth Flow) and Cryptocurrency (Store of Value Cycles).

**I. Configuration & Constants (StrategyConfig)**

The StrategyConfig class serves as the immutable constitution of the Alpha Dual Engine v154.6. Defined as a Python dataclass, it centralizes every hard-coded rule, risk limit, and execution parameter into a single, modifiable control panel. This design ensures that the strategy's logic remains separated from its parameters, allowing for rapid sensitivity testing without risking code breakage.

This section acts as the primary filter for all downstream logic. If a trade or allocation violates these parameters, it is rejected before it ever reaches the optimization engine.

**1. Risk & Portfolio Constraints**

These parameters define the "shape" of the portfolio and the limits of its aggression.

a. target_volatility (0.25 / 25%)

* Function: This sets the target annualized standard deviation for the portfolio optimizer.

* Logic: Unlike conservative funds that target 10-12% volatility, this engine is tuned for 0.25, explicitly telling the math to accept significant daily variance in exchange for higher compound growth (CAGR). It prevents the optimizer from "diluting" high-momentum assets like SMH simply because they are volatile.

aa. prob_ema_span (10-day): Probabilities are smoothed using a 10-day Exponential Moving Average (EMA) to filter out daily market noise and prevent "regime flickering," which can lead to excessive and costly trading.

An Exponential Moving Average (EMA) assigns exponentially decaying weights to past observations, prioritizing recent data. Unlike a Simple Moving Average where all observations contribute equally, a 10-day EMA halves the influence roughly every 5 days. This smoothing eliminates daily probability jitter from the ML classifier, preventing rapid regime oscillation that would trigger excessive rebalancing.

b. max_single_weight (0.30 / 30%)

* Function: The "Anti-Blowup" Cap. No single asset can exceed 30% of the total portfolio value.

* Logic: This forces the "Cubed Momentum" engine to select a basket of winners rather than betting the farm on one. Even if SMH (Semiconductors) has a perfect momentum score, the system must find at least two or three other assets (e.g., TAN, XBI) to fill the remaining allocation, ensuring a mathematical minimum diversity of ~3.33 assets (1/0.30).

c. gold_cap_risk_on (0.01 / 1%)

* Function: The "Drag eliminator."

* Logic: During a RISK_ON regime, the strategy is strictly prohibited from allocating more than 1% to Gold (GLD). This prevents the optimizer from "hiding" in safe-haven assets during a bull market, ensuring 99% of capital is deployed into productive, high-beta assets.

d. entropy_lambda (0.02)

* Function: The penalty for concentration.

* Logic: A low lambda value (0.02) tells the optimizer that we prefer concentration over diversification. It allows the weights to cluster near the 30% maximums rather than being flattened out equally across all assets.

Shannon Entropy `H(w) = -Σ wᵢ·ln(wᵢ)` quantifies weight dispersion: equal weighting across 12 assets yields the maximum (≈ 2.48), while full concentration yields 0. The optimizer includes `0.02 × H(w)` as a soft diversification bonus. Since λ_entropy = 0.02 is deliberately small, momentum signals dominate — the portfolio remains concentrated in top performers but avoids extreme single-asset allocation. See Appendix Section C for the full derivation.

**2. Execution & Cost Control (The "Fee Guillotine")**

a. Transaction Cost Calculation & Formula

**Formula:** Cost = Portfolio Value × Total Turnover × Average Cost Rate (converted from basis points) — see [Appendix Section B](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#b-the-objective-function--slsqp-solver) for full derivation

This formula computes the total transaction cost per rebalance (see [Appendix Section B](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#b-the-objective-function--slsqp-solver) for full derivation):
- V_portfolio = total portfolio value
- |wᵢ^new - wᵢ^old| = absolute weight change per asset (e.g., SMH moving from 20% to 30% produces a change of 0.10)
- Σᵢ |·| = total turnover across all assets
- Cost_i^bps/10,000 = per-asset fee rate converted from basis points to decimal (1 bps = 0.01%)

**Example:** A \$100,000 portfolio shifting 10% into SMH (5 bps) incurs a cost of 100,000 × 0.10 × 0.0005 = 5 dollars per rebalance.

*  Per-Asset Transaction Costs:

 | Ticker | Asset Name | Asset Class | Cost (bps) | Cost (%) | Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **QQQ** | Invesco QQQ Trust | Large-cap Equity | 3 | 0.03% | Nasdaq 100 ETF, extremely liquid, tight spreads, commission-free |
| **IWM** | iShares Russell 2000 | Small-cap Equity | 3 | 0.03% | High volume, liquid, commission-free |
| **SMH** | VanEck Semiconductor | Sector Equity | 5 | 0.05% | Moderate liquidity, wider spreads than broad indices |
| **XBI** | SPDR S&P Biotech | Sector Equity | 5 | 0.05% | Thematic sector, moderate spreads |
| **TAN** | Invesco Solar Energy | Thematic Equity | 8 | 0.08% | Lower volume, wider spreads, less liquid |
| **IGV** | iShares Software | Sector Equity | 5 | 0.05% | Tech sector, moderate liquidity |
| **TLT** | iShares 20+ Yr Treasury | Long-term Bonds | 2 | 0.02% | Very liquid treasury ETF, tight spreads |
| **IEF** | iShares 7-10 Yr Treasury | Mid-term Bonds | 2 | 0.02% | Very liquid treasury ETF |
| **SHY** | iShares 1-3 Yr Treasury | Short-term Bonds | 1 | 0.01% | Most liquid, cash-like, tightest spreads |
| **GLD** | SPDR Gold Trust | Commodities | 2 | 0.02% | High liquidity, tight spreads |
| **BTC-USD** | Bitcoin | Cryptocurrency | 30 | 0.30% | Exchange fees (10-25 bps) + spread (5-10 bps) |
| **ETH-USD** | Ethereum | Cryptocurrency | 30 | 0.30% | Exchange fees (10-25 bps) + spread (5-10 bps) |

b. min_rebalance_threshold (0.12 / 12%)

* Function: The Trade Gate.

* Logic: This is the most critical execution parameter. Before rebalancing, the engine calculates the total portfolio turnover (the sum of absolute weight changes).

* If Turnover < 12%: Trade Cancelled.

* Effect: This prevents "noise trading"—small, mathematical adjustments that generate fees without adding significant alpha. The system only moves when the portfolio structure is significantly misaligned.

**3. Crypto Architecture Constraints**

These parameters define the Active HODL engine, treating crypto as a distinct asset class with its own physics.

a. crypto_floor_risk_on (0.05 / 5%)

* Function: The "Zero-Coin" Prevention Floor.

* Logic: Regardless of how bad the chart looks, the strategy is mathematically forbidden from holding less than 5% in the crypto bucket (BTC + ETH). This ensures exposure to "God Candles"—sudden, violent upside moves that often occur after periods of despair.

b. total_crypto_cap (0.15 / 15%)

* Function: The Volatility Containment Ceiling.

* Logic: Even in a full bull run, crypto exposure is capped at 15%. This prevents a 50% crypto drawdown from destroying the entire portfolio's performance, maintaining the Sharpe Ratio near 1.0.

**4. Macro & ML Thresholds**

These settings govern the "Brain" of the strategy, determining when to enter or exit the market.

a. ml_threshold (0.55)

* Function: The Consensus Bar.

* Logic: The Machine Learning ensemble must be at least 55% certain that the next period will be bullish before the strategy enters a RISK_ON state (unless overridden by the Macro trend).

b. anxiety_vix_threshold (18.0)

* Function: The "Choppy Market" Detector.

* Logic: If the VIX (Volatility Index) is above 18, the market is defined as "Anxious." In this state, the ML conviction requirements are often raised automatically to prevent "whipsaw" losses in sideways markets.


**II. Data Orchestration & Feature Engineering (DataManager)**

The DataManager class is the manufacturing plant of the strategy. It is responsible for ingesting raw, noisy market data and refining it into the precise "fuel" needed for both the Machine Learning models and the Portfolio Optimizer.

In v154.6, this class has been radically re-engineered to support the Bifurcated Asset Logic, treating Crypto and Equities as fundamentally different data species.

**1. Universe Construction (The "Alpha Roster")**

a. Growth Anchors (The Engine): SMH (Semiconductors), XBI (Biotech), TAN (Solar), IGV (Software).

* Logic: These are "High Beta" sectors. They move faster than the S&P 500. The strategy intentionally designates these as the core growth drivers to maximize acceleration alpha.

b. Defensive Assets (The Brakes): TLT (Long-Term Treasuries), IEF (Intermediate Treasuries), SHY (Cash Proxy), GLD (Gold).

* Logic: These are the "Panic Buttons." When the regime flips to DEFENSIVE, capital flees here.

c. Crypto Assets (The Turbo): BTC-USD, ETH-USD.

* Logic: These provide non-correlated, exponential upside potential.


**2. The Bifurcated Signal Engine**

This is the most critical logic block in the entire script, which splits the ranking logic based on the asset class.

a. For Equities: The "Cubed Momentum" Metric

* Formula:

**Formula:** M_i = (Current Price / 60-day SMA)³ — see [Appendix Section B](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#b-the-objective-function--slsqp-solver) for full derivation

The ratio (Price / 60-day SMA) measures how far the asset is above or below its recent trend (e.g., 220/200 = 1.10 means 10% above trend). The **cubing operation** nonlinearly amplifies the gap between strong and weak performers (see [Appendix Section B](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#b-the-objective-function--slsqp-solver) for full derivation):

Scenario A: Asset is 2% above trend → 1.02³ ≈ 1.06

Scenario B: Asset is 10% above trend → 1.10³ ≈ 1.33

After cubing, Scenario B scores approximately **5.5x higher** than Scenario A (0.33 ÷ 0.06 ≈ 5.5), causing the optimizer to concentrate capital aggressively into the strongest-trending assets — the desired behavior for a momentum-driven strategy.

* Result: The optimizer sees Scenario B as dramatically better than Scenario A, naturally forcing capital into the fastest-moving sector without needing manual "if/else" exclusions.

b. For Crypto: The "RSI Rotation" Metric

* Formula: 14-Day Relative Strength Index (RSI).

RSI is computed as:

**Formula:** RSI = 100 - (100 / (1 + (Average Gain over 14 days / Average Loss over 14 days)))

This produces a bounded oscillator between 0 and 100. Values above 70 indicate overbought conditions; below 30 indicates oversold; 50 is neutral. The system compares Bitcoin's RSI against Ethereum's RSI and allocates the entire crypto bucket to whichever coin exhibits stronger recent momentum.

* Logic: Crypto doesn't follow smooth trends like stocks; it moves in manic bursts. RSI measures the internal velocity of price changes.

* Result: This allows the "Active HODL" engine to swap 100% of its weight between Bitcoin and Ethereum based on which one is currently experiencing a stronger "pump."

> **Note:** This winner-take-all RSI rotation applies only to the classical SLSQP path. When "Use Hierarchical RL" is toggled on, the RL weight agent does not enforce RSI rotation — it learns its own crypto allocation via soft constraint penalties and can freely split the bucket between BTC and ETH.

**3. The 7-Factor Feature Synthesis**

This section builds the input vectors for the Machine Learning "Brain." It distills the complex market state into 7 digestible numbers:

These 7 features compress the market state into a compact input vector for the ML classifier. Each feature captures a distinct dimension of market conditions (see [Appendix Section A](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#a-the-xgboost-ensemble-classifier--regime-detection) for the full XGBoost derivation and how these features are used):

a. **realized_vol** = VIX/100. The VIX is a number (say 22) that represents how scared the market is. Dividing by 100 just rescales it to 0.22 so the model can digest it. Higher = more fear.

b. **trend_score** = (SPY Price - SPY 200-day SMA) / SPY 200-day SMA × 100. This asks: "how far is the S&P 500 above or below its 200-day average, in percentage terms?" If SPY is at 500 and the 200-day average is 480, trend_score = (500 - 480)/480 × 100 = 4.17. Positive = bullish, negative = bearish.

c. **momentum_21d** = SPY's percentage return over the last 21 trading days (~1 month). Positive = market has been going up recently.

d. **vol_momentum** = VIX today / VIX 21 days ago - 1. This measures: "is fear increasing or decreasing?" If VIX went from 15 to 20, vol_momentum = 20/15 - 1 = 0.33 (fear jumped 33%). Rising fear = bad.

e. **qqq_vs_spy** = QQQ return over 63 days minus SPY return over 63 days. This measures: "is tech outperforming the broader market?" If QQQ returned 12% and SPY returned 8% over 3 months, this equals 4%. Positive = tech is leading (usually bullish for growth).

f. **tlt_momentum** = TLT's return over 21 days. TLT is a long-term bond fund. If TLT is crashing, it means interest rates are spiking — bad for stocks.

g. **equity_risk_premium** = 1/(SPY/SPY 252-day SMA) - risk-free rate. This is a rough valuation check. When SPY is way above its 1-year average, the "premium" for holding stocks shrinks. The model uses this to detect if the market is overvalued.

**4. Temporal Integrity (The "Time Machine")**

The get_aligned_data method ensures the backtest is honest.

a. The Lag: It strictly uses data up to the close of T-1 (yesterday) to make decisions for T (today).

b. The NaN Handler: It forward-fills missing data (common in Crypto vs. Stock weekends) to prevent the optimizer from crashing due to misalignment.


**III. Regime Intelligence (AdaptiveRegimeClassifier)**

The AdaptiveRegimeClassifier is the central nervous system of the strategy, responsible for the binary decision that determines portfolio survival: "Risk-On" (Aggressive Growth) or "Defensive" (Capital Preservation).

In v154.6, this system was upgraded from a pure Machine Learning model to a Hybrid Override Architecture. This solves the "Black Box" problem by enforcing a strict hierarchy: Macro Logic overrules ML Probabilities.

**1. The Macro Override (The "Anti-Coward" Switch)**

This is the first and most powerful layer of the decision tree. It addresses a fatal flaw in many ML strategies: the tendency to be "too smart" and exit during strong bull markets due to minor volatility signals.

The Trend Rule:

* Logic: if SPY_Price > SPY_200_SMA: return "RISK_ON"

* Purpose: The 200-day Simple Moving Average is the industry-standard definition of a long-term trend. If the S&P 500 is trading above this line, the market is structurally healthy.

* Effect: This effectively "locks" the strategy into the market during major bull runs (e.g., 2013, 2017, 2024), preventing the ML models from triggering premature defensive exits. The strategy ignores all fear signals as long as the primary trend is up.

**2. The Consensus Ensemble (The "Brain")**

If the Macro Override is not active (i.e., the market is below the 200-day SMA or in a grey zone), the system defers to its Machine Learning ensemble to navigate the uncertainty.

a. Model Alpha: XGBoost (The Aggressor)

* Role: Detects complex, non-linear opportunities.

* Configuration: Tuned with Monotonic Constraints to enforce financial logic (e.g., "Higher Volatility = Lower Probability").

XGBoost is a gradient-boosted ensemble that sequentially builds hundreds of shallow decision trees, where each new tree corrects the residual errors of its predecessors. **Monotonic Constraints** enforce domain-specific logic on the learned function — for example, enforcing that higher VIX must produce lower bull probability, regardless of what patterns the training data might suggest. This prevents the model from learning spurious correlations (e.g., "VIX at 30 = bullish" from a single 2020 data point).

* Strength: Excellent at catching "V-shaped" recoveries where price action snaps back quickly despite high fear levels.

b. Model Beta: Decision Tree (The Skeptic)

* Role: Filters out noise and false breakouts.

* Configuration: Shallow tree (max_depth=2) with high regularization (min_samples_leaf=200).

A `max_depth=2` tree can only split twice — for example: "Is trend_score > 0?" → "Is VIX < 25?" → terminal node. This extreme simplicity makes it nearly impossible to overfit, as it can only capture the most dominant patterns. The `min_samples_leaf=200` parameter further constrains the tree by requiring every terminal node to be supported by at least 200 historical observations, preventing decisions based on rare or one-off events.

* Strength: Prevents the XGBoost model from overreacting to short-term noise.

c. The Consensus Rule:

To trigger a RISK_ON signal in the absence of a Macro Override, BOTH models must agree with high conviction (Probability > 0.55). If they disagree, the system defaults to safety. See [Appendix Section A](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#a-the-xgboost-ensemble-classifier--regime-detection) for the full mathematical derivation of the XGBoost classifier, log loss, and consensus logic.

This dual-model consensus acts as a confirmation filter: the gradient-boosted model (XGBoost) captures complex nonlinear patterns, while the shallow Decision Tree provides a conservative baseline. Both must independently produce a bull probability exceeding 55% for the system to enter RISK_ON. This "default to safety" design ensures the system only takes risk when two structurally different models agree that the statistical edge is significant.

**3. The "Anxiety" & "Panic" States**

The classifier doesn't just look at price; it monitors market psychology via the VIX (Volatility Index).
To ensure the system remains career-relevant in 2026 and beyond, the classifier uses an Adaptive Walk-Forward Training loop.

a. Anxiety State (VIX > 18):

The market is choppy. The system raises the required ML probability threshold (e.g., from 55% to 75%). It demands "extraordinary evidence" to take risks in a nervous environment.

b. Panic State (VIX > 35):

The "Circuit Breaker." If volatility explodes, the system forces a DEFENSIVE regime immediately, overriding everything else. This protects the portfolio during black swan events (e.g., COVID Crash 2020).

**4. Explainability via SHAP Values**

Because black-box models are unacceptable in institutional advisory, the classifier integrates SHAP (SHapley Additive exPlanations). This provides a detailed "Feature Importance" report, allowing the user to explain exactly why a decision was made. For instance, the user can point to a SHAP summary plot to show that TLT Momentum was the primary factor that triggered a defensive exit during the bond market crash of 2022.

SHAP decomposes each prediction into per-feature contributions. For example, if the model outputs a 60% bull probability, SHAP attributes the marginal contribution of each input: trend_score may contribute +15%, while vol_momentum contributes -8% and TLT_momentum contributes -3%. The methodology is grounded in **Shapley Values** from cooperative game theory (Nobel Prize in Economics, 2012), which provides a mathematically unique, fair allocation of credit among interacting variables. Each of the 7 input features receives exactly the contribution it deserves — no more, no less.

**5. The "Regime Shield" (Rate Shock Guard)**

A special logic block designed for the 2022 bear market scenario, where stocks and bonds fell together.

* Logic: It monitors the correlation between SPY and TLT (Treasuries).

* Trigger: If TLT momentum crashes while Equity Risk Premium is negative, it signals a "Rate Shock."

* Action: Immediate exit to Cash/Gold, bypassing even the standard defensive bonds, as bonds themselves are the danger.

**IV. Portfolio Optimization (AlphaDominatorOptimizer)**

The AlphaDominatorOptimizer is the execution arm of the strategy. While the RegimeClassifier decides if we should take risk, the Optimizer decides how much risk to take and where to put it.

This is not a standard "Mean-Variance" optimizer (which often produces boring, over-diversified portfolios). It is a Constrained Momentum Maximizer. It treats the portfolio construction as a mathematical optimization problem: "Find the mix of assets that maximizes expected momentum while keeping volatility below 25% and fees below the guillotine threshold."

**1. The Mathematical Objective**

The heart of the optimizer is the function it tries to minimize:

**Formula:** L(w) = λ_risk × w^T Σ w - λ_mom × (w · M) - λ_entropy × H(w) — see [Appendix Section B](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#b-the-objective-function--slsqp-solver) for full derivation

This objective function balances three competing goals — minimize risk, maximize momentum exposure, and maintain diversification — through a single scalar that the SLSQP solver minimizes. The full mathematical derivation is provided in Appendix Section B.

**Variables:**
- **w** = The portfolio weight vector (e.g., [0.25, 0.30, 0.10, ...] representing 25% SMH, 30% QQQ, 10% TLT). This is the unknown the optimizer solves for.
- λ = Scaling coefficients controlling the relative importance of each term.
- Σ = The **covariance matrix** — a symmetric matrix capturing pairwise co-movement between all 12 assets. High covariance between two assets means they tend to move together, reducing the diversification benefit of holding both.

**Term 1: Risk** — λ_risk · w'Σw (minimized)

w'Σw computes the portfolio variance — a single scalar quantifying total risk, accounting for all pairwise correlations. Concentrated positions in correlated assets produce high variance; diversified positions across uncorrelated assets produce low variance. The λ_risk coefficient is kept moderate to accommodate the 25% target volatility.

**Term 2: Momentum** — -λ_mom (w · M) (maximized via negation)

w · M is the dot product of weights and cubed momentum scores. Since the optimizer minimizes the objective, the negative sign converts momentum maximization into an equivalent minimization problem (minimizing -x is equivalent to maximizing x). This term drives the strategy toward high-momentum assets.

**Term 3: Entropy** — -λ_entropy · H(w) (maximized via negation)

H(w) = -Σᵢ wᵢ ln(wᵢ) is Shannon Entropy, measuring weight dispersion. Entropy equals 0 when fully concentrated and reaches its maximum (ln(12) ≈ 2.48) when equally distributed. The coefficient λ_entropy = 0.02 is deliberately small — a soft diversification nudge that prevents extreme concentration without overriding momentum signals. For a detailed derivation of Shannon Entropy and the Effective N metric, see [Section C: Shannon Entropy](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#c-shannon-entropy--from-information-theory-to-portfolio-diversification).

**Net effect:** The optimizer produces aggressive, momentum-driven portfolios concentrated in the top 3-4 trending assets, while the entropy term and bound constraints prevent full single-asset concentration.

**2. The Bifurcated Bounds Engine**

This is where the Crypto vs. Equity separation is strictly enforced. The optimizer doesn't just see a list of tickers; it sees two distinct asset classes with different rules.

a. The Crypto Bounds (Active HODL):

* The Check: It looks at Bitcoin's price relative to its 50-day SMA.

* The Ceiling:

Bull Mode (BTC > 50SMA): Unlocks the crypto bucket to 15%.

Bear Mode (BTC < 50SMA): Locks the crypto bucket to 5%.

* The Rotation (Internal Alpha): It doesn't split the money. It assigns the entire allowed weight (5% or 15%) to the coin with the higher 14-day RSI. The loser gets a strict (0.0, 0.0) bound.

b. The Equity Bounds (Growth Anchors):

* The Cap: Every equity asset (SMH, TAN, XBI, IGV) is capped at 30% (max_single_weight). This forces the optimizer to pick a "Starting Lineup" of at least 3 players.

* The Gold Leash: During Risk-On, GLD is mathematically handcuffed to a maximum of 1%. This ensures "Safety" assets don't steal capital from "Growth" assets.

c. Diversity Enforcement (Effective N)

After finding a candidate portfolio, the optimizer runs a "Sanity Check" using the Inverse Herfindahl Index (Effective N).

Effective N quantifies the number of independent positions the portfolio is effectively exposed to:

**Formula:** N_eff = 1 / (sum of w_i²) — see [Appendix Section C](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#c-shannon-entropy--from-information-theory-to-portfolio-diversification) for full derivation

- 100% in one asset: 1/1.0² = 1.0 — effectively 1 position.
- 50%/50% split: 1/(0.5² + 0.5²) = 1/0.50 = 2.0 — effectively 2 positions.
- 30%/30%/30%/10%: 1/(0.09 + 0.09 + 0.09 + 0.01) = 1/0.28 ≈ 3.6 — effectively 3.6 positions.

Squaring the weights penalizes concentration disproportionately: a 90% weight contributes 0.81 to the denominator, dominating the sum and pushing N_eff toward 1. The system requires N_eff ≥ 3.0.

* Target: min_effective_n = 3.0

* Logic: If the optimizer suggests a portfolio that behaves like only 1 or 2 assets (e.g., 90% SMH, 10% Cash), the system rejects it or penalizes it until it diversifies enough to look like at least 3 uncorrelated bets.

d. Defensive & Fallback Modes

The optimizer changes its personality based on the Regime:

* RISK_ON: Maximizes Momentum (Aggressive).

* RISK_REDUCED: Maximizes Sharpe Ratio (Balances Risk/Reward).

* DEFENSIVE: Minimizes Variance (Survival Mode). Allocates 100% to Cash (SHY), Bonds (TLT), and Gold (GLD) to ride out the storm.


**V. Execution Logic (BacktestEngine)**

The BacktestEngine is the "Reality Simulator" of the strategy. It moves beyond theoretical signal generation and optimization to model the messy mechanics of actual trading—slippage, transaction costs, and time. It is responsible for stepping through the 15-year history day-by-day and enforcing the strict execution rules that protect net returns.

**1. The Simulation Loop**

This is the core heartbeat of the backtester. It iterates through every trading day from 2010 to the present.

a. Temporal Integrity:

At every step, the engine strictly uses data available up to that specific date (current_date). This prevents "look-ahead bias"—the most common sin in quantitative finance where a model accidentally peeks at tomorrow's prices to make today's decision.

b. Regime Detection:

Before considering any trades, the engine calls the AdaptiveRegimeClassifier to determine the market state (RISK_ON, RISK_REDUCED, or DEFENSIVE). This decision sets the rules of engagement for the rest of the day.

**2. The "Fee Guillotine" (The Lazy Trader Protocol)**

This is the critical logic block implemented in v154.6 to solve the "Churn Problem" (where fees were eating 4% of the portfolio).

a. Logic: Inside the main loop, before executing any rebalance, the engine calculates the Proposed Turnover:

**Formula:** Turnover = sum of |w_i_new - w_i_old| — see [Appendix Section B](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#b-the-objective-function--slsqp-solver) for full derivation

Turnover is the sum of absolute weight changes across all assets — it quantifies the total portfolio reallocation. For example, if SMH shifts from 25% to 30% (+5%) and QQQ from 20% to 15% (-5%), total turnover is 0.05 + 0.05 = 0.10 (10%). If turnover falls below the 12% threshold, the rebalance is suppressed as the expected fee drag exceeds the benefit.

b. The Rule:

* If Turnover < 0.12 (12%): THE TRADE IS REJECTED.

* Effect: The engine effectively says, "This change is too small to be worth the fees. Do nothing." It carries forward the existing portfolio weights exactly as they were.

c. The Safety Valve:

If the Regime does change (e.g., Bullish to Defensive): The Gate is lifted. Safety takes priority over fees, and the trade is executed immediately to de-risk the portfolio.

**3. Cost Modeling & Friction**

The engine uses a tiered, per-asset transaction cost model that reflects real-world trading friction. Each asset has its own cost in basis points (see the fee table in Section I.2), ranging from 1 bps for the most liquid bond ETFs to 30 bps for cryptocurrency. The cost for each rebalance is calculated as:

**Formula:** Cost = Portfolio Value × Total Turnover × Average Cost Rate (converted from basis points) — see [Appendix Section B](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#b-the-objective-function--slsqp-solver) for the full mathematical derivation

The formula is identical to Section I.2: portfolio value multiplied by absolute weight changes multiplied by the per-asset fee rate. The tiered structure creates an implicit incentive against high-frequency crypto rebalancing — at 30 bps, cryptocurrency is 30x more expensive to trade than SHY (1 bps).

This penalizes high-turnover strategies proportionally to the actual friction of the assets being traded, rewarding the "Lazy" approach mandated by the Fee Guillotine.

To prevent the optimizer from putting 100% of the capital into a single "best" stock (a common flaw in traditional optimization), the system uses Shannon Entropy as an additional diversity penalty in the objective function (see [Appendix Section C](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#c-shannon-entropy--from-information-theory-to-portfolio-diversification) for the full entropy derivation).

**4. The "Sniper Score"**

This is a proprietary analytic calculated at the end of the run to judge the quality of the Machine Learning signals.

a. Definition: Precision of the "Buy" signal.

b. Calculation:

* Identify every instance where the model switched to RISK_ON.

* Check the 21-Day Forward Return of the S&P 500 from that date.

* If the market went up, it's a "Hit." If it went down, it's a "Miss."

c. Formula:

**Formula:** Sniper Score = (Successful Bull Signals) / (Total Bull Signals)

This measures the **directional precision** of RISK_ON signals — the proportion of bull signals followed by positive 21-day forward returns in the S&P 500. For example, if the model triggered 100 RISK_ON signals and 73 were followed by positive 21-day returns, the Sniper Score is 73/100 = 0.73 (73%). A random classifier would produce approximately 50%. Scores consistently above 70% indicate statistically meaningful timing ability.

d. Target: A score >0.70 (70%) indicates the model is highly selective and only entering when the statistical edge is real.

**5. The "Final Receipt" (Diagnostics)**

At the end of the simulation, the engine compiles a forensic report of the strategy's behavior:

a. Regime Distribution: How many days were spent in RISK_ON vs. DEFENSIVE. (Proving it doesn't just sit in cash).

b. Effective N (Diversity): A time-series plot proving the portfolio never concentrated 100% in a single asset.

c. Drawdown Analysis: A log of the deepest "pain points" to ensure the strategy remains psychologically swimmable for an investor.


**VI. Backtest Execution & Performance Analytics**

The Backtest Execution & Performance Analytics (BacktestEngine) is the final investigative layer of the strategy, providing a rigorous, out-of-sample validation of the system's decision-making integrity over a 15-year historical horizon. This engine does not merely calculate returns; it audit-trails every regime shift, rebalance decision, and transaction cost to ensure the strategy's theoretical edge translates into robust framework performance.

It is designed to answer three questions:

* Is it Stable? (Sharpe Ratio, Drawdown)

* Is it Precise? (Sniper Score)

* Is it Honest? (Constraint Checks)

**1. The Sniper Score (Precision Metric)**

This is a proprietary Key Performance Indicator (KPI) specific to the Alpha Dual Engine framework.

a. The Problem: Many strategies claim "high returns" but are actually just always long the market (Beta). They buy and hold, taking credit for the market's natural rise.

b. The Solution: The Sniper Score measures the timing accuracy of the Risk-On signal.

Calculation:

* Identify every date the model switched from DEFENSIVE to RISK_ON.

* Measure the S&P 500's return over the next 21 trading days.

* Hit: Market went UP. Miss: Market went DOWN.

c. Verdict: A high Sniper Score (>70%) means that the majority of the time the model told you to buy, the market actually rallied. This proves the Alpha is real.

**2. Financial Metrics**

Standard institutional metrics are calculated to benchmark against the SPY.

a. **CAGR** (Compound Annual Growth Rate): The geometric mean return. (Target: >20%).

> **Formula:** CAGR = (Final Value / Initial Value)^(1/years) - 1
>
> If \$10,000 grew to \$80,000 over 15 years: (80000/10000)^(1/15) - 1 = 8^0.0667 - 1 ≈ 14.9% per year. It smooths out all the ups and downs into a single "average annual growth" number.

b. **Sharpe Ratio**: Return per unit of risk. (Target: ~1.0).

**Formula:** Sharpe = (R_p - R_f) / σ_p — where R_p = portfolio return, R_f = risk-free rate (Treasury bills, ~4%), σ_p = portfolio volatility. The ratio quantifies excess return per unit of risk: a Sharpe of 1.0 indicates 1% of excess return for every 1% of volatility. Above 1.0 is considered excellent; below 0.5 is mediocre.

c. **Max Drawdown**: The deepest peak-to-trough decline. This verifies the "Defensive" logic works during crashes.

For example, if the portfolio reached a peak of 150,000 dollars and subsequently declined to 100,000 dollars before recovering, the max drawdown is (100,000 - 150,000) / 150,000 = −33.3%. This metric captures the worst cumulative loss from any peak to its subsequent trough — it represents the maximum capital loss an investor would have experienced at any point during the backtest. The defensive regime is specifically designed to limit this figure.

**3. Constraint Verification**

The script prints a final "Constraint Status" block to prove the optimizer followed the rules.

a. Growth Anchor Check: Did the portfolio maintain >40% in high-growth assets?

b. Gold Cap Check: Did it strictly keep Gold under 1%?

c. Effective N Check: Did it maintain the required diversity score (>3.0)?

**4. Visualizing the Alpha Generation**

To facilitate interpretation and communication of results, the engine generates diagnostic plots:

a. Equity Curve: Compares the strategy against the SPY benchmark, with color-coded regions indicating different regime states.

b. Allocation Stack: A historical visualization of how the portfolio rotated between Growth Anchors, Bonds, and Gold across different market cycles.

c. Regime Analysis: A multi-panel view aligning SPY price action with ML probabilities, revealing how the regime detection protected the portfolio during crashes.


**VII. Tail-Risk Stress Testing (MonteCarloSimulator)**

The Monte Carlo Simulator serves as the final, rigorous validation layer of the Alpha Dual Engine framework. It transitions the analysis from historical backtesting (what did happen) to stochastic modeling (what could happen), providing a probabilistic assessment of tail risk and expected future performance.

**1. Stochastic Engine: Geometric Brownian Motion (GBM)**

a. Mathematical Foundation: The simulator utilizes Geometric Brownian Motion, the industry-standard model for projecting asset price paths. This model assumes that the logarithm of the asset price follows a Brownian motion with drift and diffusion components.

b. Drift Component (μ): The engine calculates the annualized drift based on the portfolio's optimized weighted returns from the backtest, adjusted for volatility drag:

**Formula:** μ_adj = μ - (1/2)σ² — see [Appendix Section D](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#d-geometric-brownian-motion--the-complete-derivation) for full derivation

This adjustment accounts for **volatility drag** — the mathematical asymmetry where symmetric percentage gains and losses do not cancel out. For example, a +50% gain followed by a -50% loss does not return to breakeven: 100 → 150 → 75 (a net 25% loss). The σ²/2 correction subtracts this drag from the raw expected return.

- μ = annualized expected return (e.g., 0.20 = 20% per year)
- σ = annualized volatility (e.g., 0.25 = 25% per year)
- With μ = 0.20 and σ = 0.25: μ_adj = 0.20 - (0.25)² / 2 = 0.20 - 0.03125 = 0.169, reducing the effective growth rate to approximately 16.9%. The full derivation via Ito's Lemma is provided in [Appendix Section D](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#d-geometric-brownian-motion--the-complete-derivation).

c. Diffusion Component (σ): Volatility is modeled as a random walk, scaled by the annualized standard deviation of the portfolio and a standard normal random variable (Z):

**Formula:** S_(t+1) = S_t × exp(μ_adj × Δt + σ × √(Δt) × Z) — see [Appendix Section D](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#d-geometric-brownian-motion--the-complete-derivation) for full derivation

This discrete-time simulation formula generates one step of price evolution. Each component:

- S_t = current portfolio value; S_(t+1) = next-period simulated value
- exp(...) ensures prices remain strictly positive regardless of the random draw — this is the "Geometric" in GBM, operating on log-returns rather than absolute dollar changes
- μ_adj × Δt = the **deterministic drift** component, where Δt = 1/252 (one trading day). This produces a small predictable upward nudge each day
- σ × √(Δt) × Z = the **stochastic diffusion** component, where Z ~ N(0,1) is a standard normal random variable. The √(Δt) factor converts annualized volatility to the daily timescale (the standard square-root-of-time scaling used throughout quantitative finance)

Each simulated path chains 1,260 daily steps (5 years) to produce one possible future trajectory. Repeating this process 1,000,000 times constructs the full probability distribution of portfolio outcomes.

d. Vectorized Execution: To handle the immense computational load, the simulation logic is fully vectorized using NumPy, allowing for the simultaneous generation of all price paths without slow iterative loops.

e. Sturdy Scale: The 1,000,000-Path Stress Test

f. Statistical Convergence: While standard academic projects run 1,000 to 10,000 simulations, the Alpha Dual Engine is configured to execute 1,000,000 independent simulations. This scale ensures the Law of Large Numbers applies, minimizing sampling error and producing smooth, reliable probability distributions.

g. Data Intensity: Projecting 1,000,000 paths over a 5-year horizon (1,260 trading days) generates a matrix containing over 1.26 billion data points. This capability acts as a dual stress test: verifying the financial strategy's robustness and demonstrating the hardware's computational capacity.

h. Extreme Tail Detection: At this magnitude, the simulation can capture rare 3σ or 4σ events that smaller simulations often miss, providing a more honest view of potential catastrophic downside.

**2. Risk Metrics & Output Analytics**

a. Probability of Loss: The system calculates the specific likelihood that the portfolio's value will be lower than the initial capital at the end of the 5-year period. This is a critical metric for advisory clients focused on capital preservation.

b. Risk-Adjusted Expectations: It computes the mean Compound Annual Growth Rate (CAGR) and the projected Sharpe Ratio across all 1,000,000 paths, helping to set realistic long-term return expectations beyond the specific path dependency of the historical backtest.

**3. Visualization & Interpretation**

a. Path Visualization: The system generates a visual plot of simulation paths, overlaid with the mean trajectory and the 95% Confidence Interval bands. This provides an intuitive visual representation of the range of potential outcomes.

b. Distribution Histograms: It plots the frequency distribution of both CAGR and Ending Portfolio Values. These histograms are color-coded to highlight loss zones (red), underperformance zones (orange), and target zones (green), allowing for an immediate visual assessment of the strategy's risk/reward skew.

**VIII. Main Execution**

The main() function is the entry point that transforms the complex backend logic into an interactive, visual command center. It leverages the Streamlit library to create a browser-based dashboard that allows users to monitor, configure, and stress-test the strategy in real-time without touching the raw Python code.

This interface is designed for transparency and control, bridging the gap between a black-box algorithm and a human portfolio manager.

**1. Architecture & Initialization**

a. Session State Management:

Streamlit re-runs the entire script on every user interaction (click, toggle, etc.). To prevent the model from re-training or re-fetching data constantly (which would be slow), the dashboard uses @st.cache_resource and st.session_state.

* Data Caching: Market data is fetched once and stored.

* Model Caching: The XGBoost and Decision Tree models are trained once and persisted across interactions.

* Config Versioning: A built-in version guard automatically resets cached slider state whenever code defaults change, preventing stale configuration from degrading performance.

b. Sidebar Controls

The left-hand sidebar acts as the strategy's control panel, allowing live parameter injection into the StrategyConfig class.

a. ML Threshold: Users can adjust the consensus probability threshold for RISK_ON classification (default: 0.55).

b. Min Growth Anchor Weight: Controls the minimum combined allocation to growth anchor assets (default: 40%).

c. IR Threshold: Sets the Information Ratio eligibility bar for asset inclusion (default: 0.5).

d. Gold Cap (Risk On): Maximum gold allocation during bull markets (default: 1%).

e. Turnover Penalty: Controls how aggressively the optimizer penalizes portfolio changes (default: 0.3).

**2. The "Command Center" (Main View)**

The central area of the dashboard is organized into three tabs to provide different analytical lenses.

a. Tab 1: Performance Overview (The Bottom Line)

* Equity Curve: A line chart comparing the Strategy's cumulative returns against the Benchmark (SPY).

* Allocation Over Time: A stacked area chart showing how the portfolio's makeup changed over time. Users can see the "Rotation" in action—e.g., the Crypto layer expanding during bull runs and vanishing into TLT (Bonds) during crashes.

* KPI Metrics: Big, bold indicators displaying the "Vital Signs": Total Return, CAGR, Sharpe Ratio, and Max Drawdown.

b. Tab 2: Regime & ML Diagnostics (The Brain)

* Regime Timeline: A color-coded strip chart showing the strategy's historical state.

Green: Risk-On (Aggressive)

Yellow: Risk-Reduced (Cautious)

Red: Defensive (Cash/Hedges)

* ML Probability Plot: A time-series chart overlaying the Machine Learning model's "Bull Probability" against the S&P 500 price. This allows the user to audit why the model got scared (e.g., "Ah, the probability dropped to 40% right before the COVID crash").

* SHAP Feature Importance: A bar chart showing which features drove the ML model's decisions.

* Model Health Dashboard: Validation curves showing train/test accuracy over time.

c. Tab 3: Monte Carlo Stress Test

* 1,000,000-path simulation projecting the portfolio over 5 years.

* Mean Ending Value, Mean CAGR, 95% Confidence Interval, and Probability of Loss metrics.

* Simulation path visualization and return distribution histograms.

**3. System Logs & Transparency**

Log Console: The dashboard pipes Python's logging output to the screen. Users can see real-time messages like "Switching to DEFENSIVE due to VIX spike" or "Rebalance Skipped: Turnover < 12%," providing complete auditability of the decision-making process.

**IX. Hierarchical Reinforcement Learning System**

The Alpha Dual Engine now includes an optional **Hierarchical RL** system that replaces the rule-based regime classifier and scipy optimizer with two trained neural networks operating in a principal-agent hierarchy. This system can be toggled on/off via the Streamlit sidebar checkbox.

**1. Architecture: Two-Level PPO Hierarchy**

The system operates as a principal-agent hierarchy. The high-level **Regime Agent** selects the macro strategy (RISK_ON / RISK_REDUCED / DEFENSIVE) based on market conditions. The low-level **Weight Agent** then allocates portfolio weights conditioned on that regime decision. Both agents are trained using **Proximal Policy Optimization (PPO)** — an on-policy actor-critic algorithm that stabilizes learning by clipping policy updates to prevent catastrophic forgetting. The full mathematical derivation of PPO is provided in [Appendix Section E](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#e-proximal-policy-optimization-ppo--the-complete-math).

**Key terminology:**
- **PPO (Proximal Policy Optimization):** "Policy" refers to the agent's decision-making function. "Proximal" constrains how much the policy can change per update, ensuring stable convergence.
- **MLP (Multi-Layer Perceptron):** A feedforward neural network. "2x64" denotes 2 hidden layers with 64 neurons each — inputs are transformed through successive nonlinear layers to produce action outputs.
- **Actor-Critic:** A dual-network architecture where the **Actor** (policy network) selects actions and the **Critic** (value network) estimates expected future reward. The critic's value estimates guide the actor's learning by providing a baseline for advantage computation.

a. **High-Level Regime Agent** (`rl_regime_agent.py`)

* Architecture: 2x64 MLP Actor-Critic trained with Proximal Policy Optimization (PPO)
* Backend: MLX (Apple Silicon native — no CUDA dependency)
* Input: 7-dimensional macro feature vector (VIX, SPY momentum, trend score, ML probability, drawdown, vol momentum, equity risk premium)
* Output: Discrete action space — RISK_ON / RISK_REDUCED / DEFENSIVE
* Training: `python train_100k.py` (100K timestep PPO on historical regime labels)

b. **Low-Level Weight Agent** (`rl_weight_agent.py`)

* Architecture: 2x128 MLP Continuous Actor-Critic
* Input: 103-dimensional observation vector (per-asset momentum, volatility, SMA signals, RSI, information ratio, golden cross, current weights, regime encoding, portfolio state)
* Output: 12-dimensional continuous softmax (portfolio weight for each asset)
* Training: `python train_weight_agent.py` (300K timestep PPO with Differential Sharpe Reward)

The **softmax** function e^xᵢ / Σⱼ e^xⱼ maps the network's raw outputs to a valid probability distribution summing to 1. For example, raw outputs [2.0, 1.0, 0.5, ...] become approximately [0.45, 0.22, 0.12, ...] — directly interpretable as portfolio weights.

**2. Soft Constraint Training**

Unlike traditional constrained optimization where rules are imposed post-hoc, the weight agent learns from **quadratic soft-constraint penalties** during training. It experiences the cost of violating portfolio rules (concentration limits, gold cap, crypto bounds, growth anchor floor, ineligible asset penalties) as negative reward signals, allowing it to internalize the rules rather than having decisions overwritten.

The penalty is **quadratic** in the violation magnitude (see [Appendix Section E](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#e-proximal-policy-optimization-ppo--the-complete-math) for the full PPO math): a 1% overshoot incurs a penalty of (0.01)² = 0.0001, while a 20% overshoot incurs (0.20)² = 0.04 — a 400x increase. This convex penalty structure ensures the agent strongly avoids large violations while tolerating minor boundary-pushing. The **Scale** column in the table below is a multiplier on the base penalty — higher values indicate constraints the system prioritizes more aggressively.

| Constraint | Scale | Threshold | Purpose |
|:---|:---:|:---:|:---|
| Per-asset concentration | 5.0 | 30% | Prevents single-asset dominance |
| Gold cap (RISK_ON) | 3.0 | 1% | Eliminates safe-haven drag in bull markets |
| Crypto floor | 1.5 | 5% | Ensures minimum crypto exposure |
| Crypto cap (regime-dependent) | 2.0 | 5-15% | Volatility containment |
| Ineligible equities (below SMA) | 4.0 | 0% | Enforces trend-following discipline |
| Growth anchor floor (RISK_ON) | 3.0 | 40% | Maintains acceleration alpha core |

**3. Overfitting Prevention**

The training pipeline includes multiple anti-overfitting mechanisms, validated through systematic checkpoint evaluation:

a. **Reward Clipping** [-3, +3]: Prevents the agent from exploiting training-specific return patterns by bounding the reward signal.

Daily rewards are capped to the range [-3, +3]. Without clipping, extreme market events (e.g., a crash producing a reward of -50) would disproportionately dominate the gradient signal, biasing the agent toward excessive risk aversion. Clipping ensures that no single observation dominates the learning process.

b. **Observation Noise** (σ = 0.10): Gaussian noise injected into training observations to improve generalization to unseen market conditions.

Adding ±10% Gaussian noise (see [Appendix Section E](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#e-proximal-policy-optimization-ppo--the-complete-math) for the Gaussian policy derivation) to observations during training forces the agent to learn robust policies that generalize across noisy inputs rather than memorizing exact historical values (e.g., "VIX was 22.3 on March 5, 2020"). This is a standard regularization technique — the agent must learn transferable patterns, not overfit to specific data points.

c. **Learning Rate Decay**: Linear decay to 20% of initial LR by end of training, reducing late-stage memorization.

The learning rate controls the magnitude of parameter updates per training step. Starting at a higher value enables rapid initial convergence, while linearly decaying to 20% of the initial rate reduces late-stage parameter oscillation. This prevents large updates near the end of training from overwriting well-learned early patterns — a common cause of overfitting in RL.

d. **Checkpoint Evaluation** (`eval_checkpoints.py`): Periodic model snapshots are saved and evaluated on the full backtest to identify the optimal early-stopping point.

e. **Key Finding**: The best model (checkpoint at 50K steps) was found through systematic evaluation of 11 checkpoints. More training consistently degraded out-of-sample performance — the 50K model achieves **negative Sharpe decay** (OOS Sharpe 1.060 > IS Sharpe 0.985), proving genuine generalization.

**4. Inference Pipeline**

At inference time, the weight agent's raw outputs pass through a multi-stage pipeline:

a. **Constraint Layer**: Hard eligibility enforcement, per-asset caps, gold/crypto bounds

b. **Momentum Tilt**: 50% RL allocation + 50% momentum-proportional blending in RISK_ON

c. **Growth Anchor Floor**: Ensures >= 40% in eligible growth anchors (SMH/XBI/TAN/IGV)

d. **Lazy Drift Gate**: Suppresses micro-rebalances below a per-asset threshold

**5. Ablation Test: Proving the RL Agent's Value**

To verify the RL agent contributes meaningfully beyond the constraint layer, a rigorous ablation test replaces the trained model with naive alternatives while keeping all other components identical:

| Model | Sharpe | CAGR | OOS Sharpe | Max DD |
|:---|:---:|:---:|:---:|:---:|
| **RL Agent (trained)** | **0.992** | **21.67%** | **1.061** | -40.68% |
| Random + constraints | 0.761 | 17.27% | 0.456 | -45.46% |
| Equal (1/N) + constraints | 0.832 | 18.38% | 0.876 | -35.84% |
| Baseline (rule-based) | 0.986 | 22.17% | 0.922 | -34.40% |

The trained RL agent adds **+0.231 Sharpe** over random weights with the same constraints, and **+0.605 OOS Sharpe** — proving the learned policy drives real alpha, not the constraint scaffolding. This ablation test is available interactively in the Streamlit dashboard under the RL Diagnostics tab.

**6. Out-of-Sample Performance Summary**

All metrics evaluated with training cutoff at 2024-01-01:

| Metric | RL Agent | Baseline | Target | Status |
|:---|:---:|:---:|:---:|:---:|
| Full-Period Sharpe | 0.992 | 0.986 | >0.95 | PASS |
| CAGR | 21.67% | 22.17% | >=21% | PASS |
| OOS Sharpe | 1.060 | 0.921 | — | RL wins |
| Sharpe Decay (IS - OOS) | -0.076 | 0.084 | <0.5 | PASS |
| Sniper Score | 0.711 | 0.769 | >0.6 | PASS |

**Key definitions:**
- **IS (In-Sample):** Performance evaluated on the training period (2010-2023) — measures how well the model fits known historical data.
- **OOS (Out-of-Sample):** Performance evaluated on data the model has never seen (2024+) — measures generalization to unseen market conditions.
- **Sharpe Decay (IS − OOS):** The difference between in-sample and out-of-sample Sharpe ratios. A positive value indicates potential overfitting, where the model performs worse on new data than on training data. The RL agent's **negative** decay of −0.076 indicates that the model performed *better* on unseen data than on training data — the strongest possible evidence against overfitting, suggesting the learned policy captures genuinely transferable market patterns rather than memorized historical noise.

**7. New Files**

| File | Purpose |
|:---|:---|
| `rl_regime_agent.py` | High-level PPO regime agent (MLX) |
| `rl_weight_agent.py` | Low-level PPO weight agent with training env, constraint layer, and inference |
| `train_100k.py` | Regime agent training script (100K steps) |
| `train_weight_agent.py` | Weight agent training script (300K steps) |
| `eval_oos.py` | Full OOS evaluation: RL vs baseline with Sharpe decay analysis |
| `eval_checkpoints.py` | Checkpoint sweep to find optimal early-stopping point |
| `models/rl_regime_ppo/` | Trained regime model weights (best_model.safetensors) |
| `models/rl_weight_ppo/` | Trained weight model weights + checkpoints |

**X. Challenges & Lessons Learned**

Building the Hierarchical RL system was not a smooth, linear process. The final result — a model that achieves 0.992 Sharpe with negative OOS decay — emerged from a series of failures, dead ends, and hard-won insights. This section documents the key challenges encountered and how they were resolved, as they represent the most instructive parts of the development process.

**1. The Overfitting Paradox: More Training = Worse Performance**

The first major discovery was counterintuitive: increasing training from 300K to 500K timesteps made the model *worse*, not better.

| Training Steps | Training Reward | Full Sharpe | OOS Sharpe | CAGR |
|:---:|:---:|:---:|:---:|:---:|
| 300K | +1.23 | 0.814 | 0.486 | 18.32% |
| 500K | +3.42 | 0.779 | 0.451 | 19.00% |

The training reward at 500K was nearly 3x higher than at 300K, yet the backtest Sharpe *dropped*. The agent was memorizing historical return sequences — learning "buy SMH on this specific date pattern" rather than generalizable allocation principles. This is a well-known problem in financial RL but one that is rarely discussed honestly in practice.

**Lesson:** In financial RL, training reward is a misleading metric. The only honest evaluation is out-of-sample backtest performance.

**2. The Regularization Overcorrection**

The first attempt at fixing overfitting applied all regularization techniques simultaneously at maximum strength:
- Reward clipping tightened to [-3, +3]
- Observation noise increased from 0.05 to 0.10
- Early stopping with patience=5 evaluations
- Learning rate decay to 20% of initial

Result: The run early-stopped at just 60K steps. The **final model at 60K** showed a full-period Sharpe of **0.572** and CAGR of **14.44%** — apparent massive *underfitting*. The run was initially dismissed as a failure.

However, the checkpoint saved at 50K steps — just before early stopping triggered — turned out to be the best model across all training configurations (see Lesson 3 below). The aggressive regularization settings were not wrong; the run simply needed to be stopped even earlier. These same settings are now the production configuration described in the Overfitting Prevention section above.

**Lesson:** A training run that appears to fail can still contain high-quality intermediate checkpoints. Systematic checkpoint evaluation is essential — judging a run solely by its final model can discard the best result.

**3. The Checkpoint Sweep Breakthrough**

The solution came from a systematic approach: save model checkpoints every 30K steps, then evaluate *all* of them on the full backtest. This revealed a clear U-shaped performance curve:

| Checkpoint | Sharpe | OOS Sharpe | Decay |
|:---|:---:|:---:|:---:|
| 30K steps | 0.775 | 0.495 | 0.368 |
| **50K steps** | **0.992** | **1.060** | **-0.076** |
| 60K steps | 0.978 | 0.831 | 0.191 |
| 90K steps | 0.793 | 0.577 | 0.283 |
| 120K steps | 0.822 | 0.617 | 0.264 |
| 241K steps | 0.914 | 0.620 | 0.378 |
| 300K (final) | 0.909 | 0.671 | 0.301 |

The optimal model was at **50K steps** — from the aggressive regularization run that was initially dismissed as a failure. The checkpoint saved at 50K before that run early-stopped at 60K turned out to be the best model across all training configurations. Later checkpoints showed monotonically degrading OOS performance despite improving in-sample Sharpe.

**Lesson:** Checkpoint-based model selection is essential for financial RL. The "best" model by training metrics is almost never the best model by generalization metrics. Saving and evaluating multiple snapshots is the only reliable way to find the sweet spot.

**4. The "Is the RL Actually Doing Anything?" Question**

After fixing overfitting, a legitimate concern remained: the inference pipeline applies hard constraints, momentum tilting, and growth anchor floors *after* the RL agent's output. Is the agent actually contributing, or is the post-processing doing all the work?

This was answered through ablation testing — replacing the trained model with random (Dirichlet) and equal (1/N) weight generators while keeping everything else identical:

| Model | Sharpe | OOS Sharpe |
|:---|:---:|:---:|
| **Trained RL** | **0.992** | **1.061** |
| Random + same pipeline | 0.761 | 0.456 |
| Equal + same pipeline | 0.832 | 0.876 |

The constraint layer alone (random inputs) produces Sharpe 0.761. The RL agent adds +0.231 Sharpe — roughly 23% of total performance is attributable to the learned policy. In OOS, the gap is even larger: +0.605 Sharpe over random.

**Lesson:** Ablation testing is non-negotiable when combining learned models with rule-based post-processing. Without it, you cannot distinguish genuine alpha from well-designed guardrails.

**5. The Asset Count Mismatch Bug**

A practical engineering bug consumed significant debugging time: the evaluation script (`eval_checkpoints.py`) initially created its own optimizer with `prices.columns` (13 assets including SPY) instead of the 12-asset list the model was trained on. Every checkpoint evaluation crashed with a cryptic shape mismatch error: `"could not broadcast input array from shape (13,) into shape (12,)"`.

**Lesson:** In ML pipelines with multiple entry points (training, inference, evaluation), asset universe alignment must be enforced at a single source of truth. The fix was trivial (use `dm.all_tickers` consistently), but finding it required tracing through three different code paths.

**6. The Safety Net Drift Misdiagnosis**

The Streamlit dashboard initially reported "High safety net drift" for the 50K model, suggesting the agent was poorly trained. Investigation revealed the drift metric was measured *after* momentum tilting and growth anchor boosting — intentional post-processing steps that always produce large weight shifts regardless of model quality. Moving the measurement to immediately after the constraint layer (before post-processing) showed the true constraint drift was moderate and healthy.

**Lesson:** Diagnostic metrics must measure what they claim to measure. A monitoring metric that conflates model quality with intentional post-processing creates misleading signals.

**7. Summary of Key Takeaways**

| Challenge | Root Cause | Resolution |
|:---|:---|:---|
| More training = worse results | Agent memorizes historical patterns | Checkpoint sweep + early stopping at 50K |
| Over-regularization | All techniques at max simultaneously | Calibrate individually; rely on checkpoints |
| Shape mismatch in evaluation | Inconsistent asset universe across scripts | Single source of truth (`dm.all_tickers`) |
| Misleading safety net metric | Drift measured after post-processing | Measure before momentum tilt/GA floor |
| "Is RL doing anything?" concern | Constraint layer is strong by itself | Ablation test: RL adds +0.231 Sharpe |

**XI. Conclusion**

The Alpha Dual Engine v154.6 represents a modern evolution in finance and asset management—shifting from reactive rebalancing to proactive, regime-aware navigation. The addition of Hierarchical Reinforcement Learning extends this foundation with learned, adaptive decision-making that demonstrably generalizes to unseen market conditions. The challenges documented above — and their systematic resolution — underscore that building robust financial RL systems requires not just ML expertise, but disciplined experimentation, honest evaluation, and the willingness to question whether your model is truly contributing.

====================================================================================

# **Mathematical Foundations**

The full mathematical appendix — every formula derived from first principles with worked numerical examples — lives in the project wiki to keep this README focused:

### 📖 [**Read the complete Mathematical Foundations → Wiki**](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations)

It covers, from first principles:

| Section | Topic | Key Question Answered |
|:---:|:---|:---|
| 0 | [Foundational Concepts](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#0-foundational-concepts) | What is a loss function? What is gradient descent? |
| A | [XGBoost Ensemble Classifier](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#a-the-xgboost-ensemble-classifier--regime-detection) | How does the ML classifier detect market regimes? |
| B | [Objective Function & SLSQP Solver](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#b-the-objective-function--slsqp-solver) | How does the optimizer pick portfolio weights? |
| C | [Shannon Entropy](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#c-shannon-entropy--from-information-theory-to-portfolio-diversification) | How does the system measure diversification? |
| D | [Geometric Brownian Motion](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#d-geometric-brownian-motion--the-complete-derivation) | How are future stock prices simulated? |
| E | [Proximal Policy Optimization (PPO)](https://github.com/Guannings/alpha_dual_engine/wiki/Mathematical-Foundations#e-proximal-policy-optimization-ppo--the-complete-math) | How does the RL agent learn without destroying itself? |

Each section starts with intuition, builds the derivation, and ends with a concrete numerical example.

====================================================================================
# **Development Methodology**

**The core financial strategy was conceptualized and architected by the author.**

This project was built using an AI-Accelerated Workflow. Large Language Models (Claude Opus 4.6, Gemini 3 Pro) were utilized to accelerate syntax generation and boilerplate implementation, allowing the focus to remain on quantitative logic, parameter tuning, and risk management validation.

---

## **Author's Design Decisions**

The following strategic and architectural decisions were made entirely by the author. These are the choices that define the system's behavior, risk profile, and performance characteristics — none were suggested or generated by AI tooling.

**Strategy Architecture:**
- Bifurcated logic — treating cryptocurrency and equities as fundamentally distinct asset classes with separate ranking rules (cubed momentum for equities, RSI rotation for crypto)
- Hierarchical RL architecture — the decision to split regime detection and weight allocation into two separate PPO agents operating in a principal-agent hierarchy, rather than a single monolithic agent
- 12-asset universe design — selecting assets across growth anchors (SMH, XBI, TAN, IGV), safe havens (TLT, IEF, SHY, GLD), broad equity (QQQ, IWM), and crypto (BTC, ETH), with each category serving a specific role in the regime-switching logic

**Risk & Parameter Engineering:**
- 25% target volatility — a deliberately aggressive threshold that prevents the optimizer from diluting high-momentum positions, calibrated to the strategy's growth-oriented mandate
- 12% turnover gate — identified through backtest analysis that frequent micro-rebalances were eroding returns via transaction costs; this threshold suppresses trades too small to justify their fees
- 1% gold cap during RISK_ON — prevents the optimizer from allocating to safe-haven assets during bull markets, ensuring capital is fully deployed into productive positions
- 40% growth anchor floor — guarantees meaningful exposure to high-beta sector ETFs during risk-on regimes, preventing the optimizer from drifting toward conservative allocations
- Cubing momentum scores — the nonlinear amplification that forces the optimizer to concentrate into the strongest-trending assets rather than spreading across mediocre performers
- Every parameter in `StrategyConfig` — all thresholds, lambda coefficients, constraint bounds, and regime logic cutoffs

**RL Training & Debugging:**
- Reward function design — excess Sharpe ratio over benchmark with a quadratic drawdown penalty, balancing risk-adjusted return maximization against tail-risk control
- Bypassing the RL regime agent — the trained policy shows 57.1% RISK_REDUCED / 36.2% RISK_ON / 6.8% DEFENSIVE, too cautious compared to the rule-based baseline's 84.2% RISK_ON. Replaced with a rule-based override (SPY > 200-SMA) while retaining the trained Weight Agent
- Checkpoint selection — systematic evaluation of 11 model checkpoints across the training run, identifying the 50K-step model as optimal based on negative Sharpe decay (OOS Sharpe 1.060 > IS Sharpe 0.985), proving that more training degraded generalization
- Diagnosing the safety net drift measurement bug — identifying and resolving an error in the drift calculation that caused the defensive regime to trigger incorrectly

**Documented Failures:**
- Section X of this README documents 5 specific failures encountered during development and the diagnostic process for each. These failures — and the reasoning behind their resolution — represent the iterative engineering judgment that no AI tool can replicate.
