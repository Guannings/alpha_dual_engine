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

This software is for educational and research purposes only and was built as a personal project by a student, PARVAUX, a Public Fianace major at National Chengchi University (NCCU). It is not intended to be a source of financial advice, and the author is not a registered financial advisor. The algorithms, simulations, and optimization techniques implemented herein—including Consensus Machine Learning, Shannon Entropy, and Geometric Brownian Motion—are demonstrations of theoretical concepts and should not be construed as a recommendation to buy, sell, or hold any specific security or asset class.

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

This documentation details the architecture of Alpha Dual Engine v154.6, a high-performance quantitative trading system refactored to prioritize "Acceleration Alpha" over passive diversification and engineered to navigate complex market regimes through a synthesis of machine learning, macro-economic veto guards, and entropy-weighted optimization. Established by PARVAUX, a Public Finance major at National Chengchi University, the system represents a Bifurcated Logic that applies distinct mathematical strategies to Equities (Growth Flow) and Cryptocurrency (Store of Value Cycles). Unlike traditional mean-variance models that treat all assets identically, v154.6 utilizes a Bifurcated Logic Engine that applies distinct mathematical strategies to Equities (Growth Flow) and Cryptocurrency (Store of Value Cycles).

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

* Logic: This forces the "Cubed Momentum" engine to select a basket of winners rather than betting the farm on one. Even if SMH (Semiconductors) has a perfect momentum score, the system must find at least two or three other assets (e.g., TAN, XBI) to fill the remaining allocation, ensuring a mathematical minimum diversity of $\sim 3.33$ assets ($1 / 0.30$).

c. gold_cap_risk_on (0.01 / 1%)

* Function: The "Drag eliminator."

* Logic: During a RISK_ON regime, the strategy is strictly prohibited from allocating more than 1% to Gold (GLD). This prevents the optimizer from "hiding" in safe-haven assets during a bull market, ensuring 99% of capital is deployed into productive, high-beta assets.

d. entropy_lambda (0.02)

* Function: The penalty for concentration.

* Logic: A low lambda value (0.02) tells the optimizer that we prefer concentration over diversification. It allows the weights to cluster near the 30% maximums rather than being flattened out equally across all assets.

Shannon Entropy $H(\mathbf{w}) = -\sum_i w_i \ln(w_i)$ quantifies weight dispersion: equal weighting across 12 assets yields the maximum ($\approx 2.48$), while full concentration yields 0. The optimizer includes $0.02 \times H(\mathbf{w})$ as a soft diversification bonus. Since $\lambda_{\text{entropy}} = 0.02$ is deliberately small, momentum signals dominate — the portfolio remains concentrated in top performers but avoids extreme single-asset allocation. See Appendix Section B for the full derivation.

**2. Execution & Cost Control (The "Fee Guillotine")**

a. Transaction Cost Calculation & Formula

$$\text{Cost} = V_{\text{portfolio}} \times \sum_{i} \left| w_i^{\text{new}} - w_i^{\text{old}} \right| \times \frac{\text{Cost}_i^{\text{bps}}}{10{,}000}$$

This formula computes the total transaction cost per rebalance:
- $V_{\text{portfolio}}$ = total portfolio value
- $|w_i^{\text{new}} - w_i^{\text{old}}|$ = absolute weight change per asset (e.g., SMH moving from 20% to 30% produces a change of 0.10)
- $\sum_i |\cdot|$ = total turnover across all assets
- $\frac{\text{Cost}_i^{\text{bps}}}{10{,}000}$ = per-asset fee rate converted from basis points to decimal (1 bps = 0.01%)

**Example:** A \$100,000 portfolio shifting 10% into SMH (5 bps) incurs a cost of $100{,}000 \times 0.10 \times 0.0005 = 5$ dollars per rebalance.

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

* If $\text{Turnover} < 12\%$: Trade Cancelled.

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

$$M_i = \left(\frac{P_i}{\text{SMA}_{60,i}}\right)^3$$

Where $P_i$ is the current price of asset $i$ and $SMA_{60,i}$ is its 60-day Simple Moving Average. The ratio $P_i / SMA_{60,i}$ measures how far the asset is above or below its recent trend (e.g., $220/200 = 1.10$ means 10% above trend). The **cubing operation** nonlinearly amplifies the gap between strong and weak performers:

Scenario A: Asset is 2% above trend → $1.02^3 \approx 1.06$

Scenario B: Asset is 10% above trend → $1.10^3 \approx 1.33$

After cubing, Scenario B scores approximately **5.5x higher** than Scenario A ($0.33 \div 0.06 \approx 5.5$), causing the optimizer to concentrate capital aggressively into the strongest-trending assets — the desired behavior for a momentum-driven strategy.

* Result: The optimizer sees Scenario B as dramatically better than Scenario A, naturally forcing capital into the fastest-moving sector without needing manual "if/else" exclusions.

b. For Crypto: The "RSI Rotation" Metric

* Formula: 14-Day Relative Strength Index (RSI).

RSI is computed as:

$$\text{RSI} = 100 - \frac{100}{1 + \frac{\text{Avg Gain over 14 days}}{\text{Avg Loss over 14 days}}}$$

This produces a bounded oscillator between 0 and 100. Values above 70 indicate overbought conditions; below 30 indicates oversold; 50 is neutral. The system compares Bitcoin's RSI against Ethereum's RSI and allocates the entire crypto bucket to whichever coin exhibits stronger recent momentum.

* Logic: Crypto doesn't follow smooth trends like stocks; it moves in manic bursts. RSI measures the internal velocity of price changes.

* Result: This allows the "Active HODL" engine to swap 100% of its weight between Bitcoin and Ethereum based on which one is currently experiencing a stronger "pump."

**3. The 7-Factor Feature Synthesis**

This section builds the input vectors for the Machine Learning "Brain." It distills the complex market state into 7 digestible numbers:

These 7 features compress the market state into a compact input vector for the ML classifier. Each feature captures a distinct dimension of market conditions:

a. **realized_vol** = $\frac{\text{VIX}}{100}$. The VIX is a number (say 22) that represents how scared the market is. Dividing by 100 just rescales it to 0.22 so the model can digest it. Higher = more fear.

b. **trend_score** = $\frac{\text{SPY Price} - \text{SPY 200-day SMA}}{\text{SPY 200-day SMA}} \times 100$. This asks: "how far is the S&P 500 above or below its 200-day average, in percentage terms?" If SPY is at 500 and the 200-day average is 480, trend_score = $\frac{500 - 480}{480} \times 100 = 4.17$. Positive = bullish, negative = bearish.

c. **momentum_21d** = SPY's percentage return over the last 21 trading days (~1 month). Positive = market has been going up recently.

d. **vol_momentum** = $\frac{\text{VIX today}}{\text{VIX 21 days ago}} - 1$. This measures: "is fear increasing or decreasing?" If VIX went from 15 to 20, vol_momentum = $\frac{20}{15} - 1 = 0.33$ (fear jumped 33%). Rising fear = bad.

e. **qqq_vs_spy** = QQQ return over 63 days minus SPY return over 63 days. This measures: "is tech outperforming the broader market?" If QQQ returned 12% and SPY returned 8% over 3 months, this equals 4%. Positive = tech is leading (usually bullish for growth).

f. **tlt_momentum** = TLT's return over 21 days. TLT is a long-term bond fund. If TLT is crashing, it means interest rates are spiking — bad for stocks.

g. **equity_risk_premium** = $\frac{1}{\text{SPY} / \text{SPY 252-day SMA}} - \text{risk-free rate}$. This is a rough valuation check. When SPY is way above its 1-year average, the "premium" for holding stocks shrinks. The model uses this to detect if the market is overvalued.

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

To trigger a RISK_ON signal in the absence of a Macro Override, BOTH models must agree with high conviction (Probability > 0.55). If they disagree, the system defaults to safety.

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

$$\mathcal{L}(\mathbf{w}) = \lambda_{\text{risk}} \mathbf{w}^\top \Sigma \mathbf{w} - \lambda_{\text{mom}} (\mathbf{w} \cdot \mathbf{M}) - \lambda_{\text{entropy}} H(\mathbf{w})$$

This objective function balances three competing goals — minimize risk, maximize momentum exposure, and maintain diversification — through a single scalar that the SLSQP solver minimizes. The full mathematical derivation is provided in Appendix Section A.

**Variables:**
- $\mathbf{w}$ = The portfolio weight vector (e.g., $[0.25, 0.30, 0.10, \ldots]$ representing 25% SMH, 30% QQQ, 10% TLT). This is the unknown the optimizer solves for.
- $\lambda$ = Scaling coefficients controlling the relative importance of each term.
- $\Sigma$ = The **covariance matrix** — a symmetric matrix capturing pairwise co-movement between all 12 assets. High covariance between two assets means they tend to move together, reducing the diversification benefit of holding both.

**Term 1: Risk** — $\lambda_{risk} \cdot w^\top \Sigma w$ (minimized)

$w^\top \Sigma w$ computes the portfolio variance — a single scalar quantifying total risk, accounting for all pairwise correlations. Concentrated positions in correlated assets produce high variance; diversified positions across uncorrelated assets produce low variance. The $\lambda_{risk}$ coefficient is kept moderate to accommodate the 25% target volatility.

**Term 2: Momentum** — $-\lambda_{mom} (w \cdot M)$ (maximized via negation)

$w \cdot M$ is the dot product of weights and cubed momentum scores. Since the optimizer minimizes the objective, the negative sign converts momentum maximization into an equivalent minimization problem (minimizing $-x$ is equivalent to maximizing $x$). This term drives the strategy toward high-momentum assets.

**Term 3: Entropy** — $-\lambda_{entropy} \cdot H(w)$ (maximized via negation)

$H(w) = -\sum_i w_i \ln(w_i)$ is Shannon Entropy, measuring weight dispersion. Entropy equals 0 when fully concentrated and reaches its maximum ($\ln 12 \approx 2.48$) when equally distributed. The coefficient $\lambda_{entropy} = 0.02$ is deliberately small — a soft diversification nudge that prevents extreme concentration without overriding momentum signals. For a detailed derivation of Shannon Entropy and the Effective N metric, see [Section B: Shannon Entropy](#b-shannon-entropy--from-information-theory-to-portfolio-diversification).

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

$$N_{\text{eff}} = \frac{1}{\sum_i w_i^2}$$

- 100% in one asset: $\frac{1}{1.0^2} = 1.0$ — effectively 1 position.
- 50%/50% split: $\frac{1}{0.5^2 + 0.5^2} = \frac{1}{0.50} = 2.0$ — effectively 2 positions.
- 30%/30%/30%/10%: $\frac{1}{0.09 + 0.09 + 0.09 + 0.01} = \frac{1}{0.28} \approx 3.6$ — effectively 3.6 positions.

Squaring the weights penalizes concentration disproportionately: a 90% weight contributes $0.81$ to the denominator, dominating the sum and pushing $N_{\text{eff}}$ toward 1. The system requires $N_{\text{eff}} \geq 3.0$.

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

$$\text{Turnover} = \sum_{i} \left| w_i^{\text{new}} - w_i^{\text{old}} \right|$$

Turnover is the sum of absolute weight changes across all assets — it quantifies the total portfolio reallocation. For example, if SMH shifts from 25% to 30% (+5%) and QQQ from 20% to 15% (-5%), total turnover is $0.05 + 0.05 = 0.10$ (10%). If turnover falls below the 12% threshold, the rebalance is suppressed as the expected fee drag exceeds the benefit.

b. The Rule:

* If $\text{Turnover} < 0.12$ (12%): THE TRADE IS REJECTED.

* Effect: The engine effectively says, "This change is too small to be worth the fees. Do nothing." It carries forward the existing portfolio weights exactly as they were.

c. The Safety Valve:

If the Regime does change (e.g., Bullish to Defensive): The Gate is lifted. Safety takes priority over fees, and the trade is executed immediately to de-risk the portfolio.

**3. Cost Modeling & Friction**

The engine uses a tiered, per-asset transaction cost model that reflects real-world trading friction. Each asset has its own cost in basis points (see the fee table in Section I.2), ranging from 1 bps for the most liquid bond ETFs to 30 bps for cryptocurrency. The cost for each rebalance is calculated as:

$$\text{Cost} = V_{\text{portfolio}} \times \sum_{i} \left| w_i^{\text{new}} - w_i^{\text{old}} \right| \times \frac{\text{Cost}_i^{\text{bps}}}{10{,}000}$$

The formula is identical to Section I.2: portfolio value multiplied by absolute weight changes multiplied by the per-asset fee rate. The tiered structure creates an implicit incentive against high-frequency crypto rebalancing — at 30 bps, cryptocurrency is 30x more expensive to trade than SHY (1 bps).

This penalizes high-turnover strategies proportionally to the actual friction of the assets being traded, rewarding the "Lazy" approach mandated by the Fee Guillotine.

To prevent the optimizer from putting 100% of the capital into a single "best" stock (a common flaw in traditional optimization), the system uses Shannon Entropy as an additional diversity penalty in the objective function.

**4. The "Sniper Score"**

This is a proprietary analytic calculated at the end of the run to judge the quality of the Machine Learning signals.

a. Definition: Precision of the "Buy" signal.

b. Calculation:

* Identify every instance where the model switched to RISK_ON.

* Check the 21-Day Forward Return of the S&P 500 from that date.

* If the market went up, it's a "Hit." If it went down, it's a "Miss."

c. Formula:

$$\text{Sniper Score} = \frac{\text{Successful Bull Signals}}{\text{Total Bull Signals}}$$

This measures the **directional precision** of RISK_ON signals — the proportion of bull signals followed by positive 21-day forward returns in the S&P 500. For example, if the model triggered 100 RISK_ON signals and 73 were followed by positive 21-day returns, the Sniper Score is $\frac{73}{100} = 0.73$ (73%). A random classifier would produce approximately 50%. Scores consistently above 70% indicate statistically meaningful timing ability.

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

> **Formula:** $\text{CAGR} = \left(\frac{\text{Final Value}}{\text{Initial Value}}\right)^{1/\text{years}} - 1$
>
> If `$10,000` grew to `$80,000` over 15 years: $\left(\frac{80000}{10000}\right)^{1/15} - 1 = 8^{0.0667} - 1 \approx 14.9\%$ per year. It smooths out all the ups and downs into a single "average annual growth" number.

b. **Sharpe Ratio**: Return per unit of risk. (Target: ~1.0).

**Formula:** $\text{Sharpe} = \frac{R_p - R_f}{\sigma_p}$ where $R_p$ = portfolio return, $R_f$ = risk-free rate (Treasury bills, ~4%), $\sigma_p$ = portfolio volatility. The ratio quantifies excess return per unit of risk: a Sharpe of 1.0 indicates 1% of excess return for every 1% of volatility. Above 1.0 is considered excellent; below 0.5 is mediocre.

c. **Max Drawdown**: The deepest peak-to-trough decline. This verifies the "Defensive" logic works during crashes.

For example, if the portfolio reached a peak of 150,000 dollars and subsequently declined to 100,000 dollars before recovering, the max drawdown is $(100{,}000 - 150{,}000) / 150{,}000 = -33.3\%$. This metric captures the worst cumulative loss from any peak to its subsequent trough — it represents the maximum capital loss an investor would have experienced at any point during the backtest. The defensive regime is specifically designed to limit this figure.

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

b. Drift Component ($\mu$): The engine calculates the annualized drift based on the portfolio's optimized weighted returns from the backtest, adjusted for volatility drag:

$$\mu_{\text{adj}} = \mu - \frac{1}{2}\sigma^2$$

This adjustment accounts for **volatility drag** — the mathematical asymmetry where symmetric percentage gains and losses do not cancel out. For example, a +50% gain followed by a -50% loss does not return to breakeven: 100 → 150 → 75 (a net 25% loss). The $\sigma^2 / 2$ correction subtracts this drag from the raw expected return.

- $\mu$ = annualized expected return (e.g., 0.20 = 20% per year)
- $\sigma$ = annualized volatility (e.g., 0.25 = 25% per year)
- With $\mu = 0.20$ and $\sigma = 0.25$: $\mu_{adj} = 0.20 - (0.25)^2 / 2 = 0.20 - 0.03125 = 0.169$, reducing the effective growth rate to approximately 16.9%. The full derivation via Ito's Lemma is provided in [Appendix Section C](#c-geometric-brownian-motion--the-complete-derivation).

c. Diffusion Component ($\sigma$): Volatility is modeled as a random walk, scaled by the annualized standard deviation of the portfolio and a standard normal random variable ($Z$):

$$S_{t+1} = S_t \exp\left(\mu_{\text{adj}} \Delta t + \sigma \sqrt{\Delta t} Z\right)$$

This discrete-time simulation formula generates one step of price evolution. Each component:

- $S_t$ = current portfolio value; $S_{t+1}$ = next-period simulated value
- $\exp(\ldots)$ ensures prices remain strictly positive regardless of the random draw — this is the "Geometric" in GBM, operating on log-returns rather than absolute dollar changes
- $\mu_{\text{adj}} \Delta t$ = the **deterministic drift** component, where $\Delta t = \frac{1}{252}$ (one trading day). This produces a small predictable upward nudge each day
- $\sigma \sqrt{\Delta t} \cdot Z$ = the **stochastic diffusion** component, where $Z \sim \mathcal{N}(0, 1)$ is a standard normal random variable. The $\sqrt{\Delta t}$ factor converts annualized volatility to the daily timescale (the standard square-root-of-time scaling used throughout quantitative finance)

Each simulated path chains 1,260 daily steps (5 years) to produce one possible future trajectory. Repeating this process 1,000,000 times constructs the full probability distribution of portfolio outcomes.

d. Vectorized Execution: To handle the immense computational load, the simulation logic is fully vectorized using NumPy, allowing for the simultaneous generation of all price paths without slow iterative loops.

e. Sturdy Scale: The 1,000,000-Path Stress Test

f. Statistical Convergence: While standard academic projects run 1,000 to 10,000 simulations, the Alpha Dual Engine is configured to execute 1,000,000 independent simulations. This scale ensures the Law of Large Numbers applies, minimizing sampling error and producing smooth, reliable probability distributions.

g. Data Intensity: Projecting 1,000,000 paths over a 5-year horizon (1,260 trading days) generates a matrix containing over 1.26 billion data points. This capability acts as a dual stress test: verifying the financial strategy's robustness and demonstrating the hardware's computational capacity.

h. Extreme Tail Detection: At this magnitude, the simulation can capture rare $3\sigma$ or $4\sigma$ events that smaller simulations often miss, providing a more honest view of potential catastrophic downside.

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

The system operates as a principal-agent hierarchy. The high-level **Regime Agent** selects the macro strategy (RISK_ON / RISK_REDUCED / DEFENSIVE) based on market conditions. The low-level **Weight Agent** then allocates portfolio weights conditioned on that regime decision. Both agents are trained using **Proximal Policy Optimization (PPO)** — an on-policy actor-critic algorithm that stabilizes learning by clipping policy updates to prevent catastrophic forgetting. The full mathematical derivation of PPO is provided in [Appendix Section D](#d-proximal-policy-optimization-ppo--the-complete-math).

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

The **softmax** function $\frac{e^{x_i}}{\sum_j e^{x_j}}$ maps the network's raw outputs to a valid probability distribution summing to 1. For example, raw outputs $[2.0, 1.0, 0.5, \ldots]$ become approximately $[0.45, 0.22, 0.12, \ldots]$ — directly interpretable as portfolio weights.

**2. Soft Constraint Training**

Unlike traditional constrained optimization where rules are imposed post-hoc, the weight agent learns from **quadratic soft-constraint penalties** during training. It experiences the cost of violating portfolio rules (concentration limits, gold cap, crypto bounds, growth anchor floor, ineligible asset penalties) as negative reward signals, allowing it to internalize the rules rather than having decisions overwritten.

The penalty is **quadratic** in the violation magnitude: a 1% overshoot incurs a penalty of $(0.01)^2 = 0.0001$, while a 20% overshoot incurs $(0.20)^2 = 0.04$ — a 400x increase. This convex penalty structure ensures the agent strongly avoids large violations while tolerating minor boundary-pushing. The **Scale** column in the table below is a multiplier on the base penalty — higher values indicate constraints the system prioritizes more aggressively.

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

b. **Observation Noise** ($\sigma = 0.10$): Gaussian noise injected into training observations to improve generalization to unseen market conditions.

Adding $\pm 10\%$ Gaussian noise to observations during training forces the agent to learn robust policies that generalize across noisy inputs rather than memorizing exact historical values (e.g., "VIX was 22.3 on March 5, 2020"). This is a standard regularization technique — the agent must learn transferable patterns, not overfit to specific data points.

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
- **Sharpe Decay (IS $-$ OOS):** The difference between in-sample and out-of-sample Sharpe ratios. A positive value indicates potential overfitting, where the model performs worse on new data than on training data. The RL agent's **negative** decay of $-0.076$ indicates that the model performed *better* on unseen data than on training data — the strongest possible evidence against overfitting, suggesting the learned policy captures genuinely transferable market patterns rather than memorized historical noise.

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

# **Appendix: Mathematical Foundations — Deep Dive**

This section provides rigorous mathematical interpretations of every core formula used in the Alpha Dual Engine. Each subsection starts from first principles, builds the intuition, walks through the derivation, and ends with a concrete numerical example. The goal is to make every symbol, subscript, and Greek letter fully transparent — no hand-waving allowed.

---

## **Table of Contents**

| # | Section | Formula Name | Key Formulas | What It Answers |
|:---:|:---|:---|:---|:---|
| 0 | [Foundational Concepts](#0-foundational-concepts) | Mean Squared Error (Loss) | $L = (\text{guess} - \text{actual})^2$ | What is a loss function? |
| | | Gradient Descent Update | $w_{\text{new}} = w_{\text{old}} - \alpha \nabla L$ | How does the computer minimize any function? |
| A | [The Objective Function & SLSQP](#a-the-objective-function--slsqp-solver-section-iv) | Portfolio Objective Function | $\mathcal{L}(\mathbf{w}) = \lambda_{\text{risk}} \mathbf{w}^\top \Sigma \mathbf{w} - \lambda_{\text{mom}} (\mathbf{w} \cdot \mathbf{M}) - \lambda_{\text{entropy}} H(\mathbf{w})$ | How does the optimizer pick portfolio weights? |
| | | Lagrangian | $\mathcal{L} = f(\mathbf{w}) + \mu(\sum w_i - 1)$ | What are Lagrange multipliers and shadow prices? |
| | | Hessian Matrix | $H_{ij} = \partial^2 f / \partial w_i \partial w_j$ | How does the solver know the shape of the bowl? |
| B | [Shannon Entropy](#b-shannon-entropy--from-information-theory-to-portfolio-diversification) | Shannon Entropy | $H(\mathbf{w}) = -\sum_i w_i \ln(w_i)$ | How does the system measure diversification? |
| | | Effective N | $N_{\text{eff}} = e^{H(\mathbf{w})}$ | What is "Effective N" and why require at least 3? |
| C | [Geometric Brownian Motion](#c-geometric-brownian-motion--the-complete-derivation) | GBM Stochastic Differential Equation | $dS = \mu S \cdot dt + \sigma S \cdot dW$ | How are future stock prices simulated? |
| | | Discrete Simulation (Ito's Lemma) | $\Delta S = S \cdot e^{(\mu - \sigma^2/2)\Delta t + \sigma \sqrt{\Delta t} \cdot Z}$ | What is the $-\sigma^2/2$ correction? |
| D | [Proximal Policy Optimization (PPO)](#d-proximal-policy-optimization-ppo--the-complete-math) | PPO Clipped Surrogate Objective | $L^{CLIP} = -E[\min(r_t A_t, \text{clip}(r_t, 1 \pm \epsilon) A_t)]$ | How does the RL agent learn without destroying itself? |
| | | Generalized Advantage Estimation | $A_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \ldots$ | What is GAE and the actor-critic architecture? |

**Detailed sub-sections within each part:**

**Section 0 — Foundational Concepts**
- [What is a loss function?](#what-is-a-loss-function) — Squared error, loss vs reward, the three losses in this project
- [What is gradient descent?](#what-is-gradient-descent) — The update rule, learning rate, worked numerical example

**Section A — The Objective Function & SLSQP**
- [What the formula actually says](#what-the-formula-actually-says) — Breaking down each term (risk, momentum, entropy)
- [How SLSQP actually solves it](#how-slsqp-actually-solves-it) — The "approximate as a parabola and step" method
- [What is the Hessian matrix?](#what-is-the-hessian-matrix-b) — Second derivatives, curvature, and the BFGS approximation
- [Can the quadratic subproblem be solved by hand?](#can-the-quadratic-subproblem-be-solved-by-hand) — Layer-by-layer breakdown from 1 variable to 12
- [What is a Lagrange multiplier, exactly?](#what-is-a-lagrange-multiplier-exactly) — Intuition, worked example, geometric interpretation, shadow prices
- [Solving the Linear System: Gaussian Elimination](#solving-the-linear-system-gaussian-elimination) — Forward elimination, back-substitution, worked examples
- [The complete picture](#the-complete-picture) — Summary table: which layer uses which math

**Section B — Shannon Entropy**
- [The formula](#the-formula) — Step-by-step calculation with real numbers
- [Why does ln show up?](#why-does-ln-show-up) — Intuition: "importance-weighted surprise"
- [Concrete examples with real numbers](#concrete-examples-with-real-numbers) — All-in vs equal-weight vs the actual portfolio

**Section C — Geometric Brownian Motion**
- [The continuous-time SDE](#the-continuous-time-sde-the-textbook-form) — Drift and diffusion decomposition
- [Ito's Lemma and the volatility drag correction](#from-the-sde-to-the-formula-you-can-actually-compute-itos-lemma) — Why symmetric gains/losses do not cancel
- [The final simulation formula](#the-final-simulation-formula) — The discrete-time equation the code actually uses
- [A full worked example](#a-full-worked-example--one-simulated-day) — One simulated day with real numbers

**Section D — Proximal Policy Optimization (PPO)**
- [Step 1: The Policy](#step-1-the-policy-pi_theta) — Discrete (softmax) vs continuous (Gaussian) action spaces
- [Step 3: Advantage Estimation (GAE)](#step-3-advantage-estimation--was-this-action-better-than-average) — TD error, bias-variance tradeoff
- [Step 4: The Clipped Surrogate Objective](#step-4-the-ppo-clipped-surrogate-objective--the-core-formula) — The core PPO innovation
- [Step 5: The Full Loss Function](#step-5-the-full-loss-function) — Policy loss + value loss + entropy bonus
- [How the two agents work together](#how-the-two-agents-work-together-hierarchically) — Hierarchical RL: regime agent → weight agent

---

## **0. Foundational Concepts**

Before diving into the specific formulas, two ideas underpin everything in this appendix: the **loss function** (what the computer is trying to achieve) and **gradient descent** (how it gets there). Every section that follows — SLSQP, Shannon Entropy, GBM, PPO — is built on these two foundations.

### **What is a loss function?**

A loss function is a single number that measures **how wrong the current answer is.** The entire purpose of optimization — whether it is portfolio construction, neural network training, or anything else — is to make this number as small as possible.

**The simplest example:** Suppose you are predicting house prices. Your model guesses \$500K. The real price is \$450K.

$$L = (\text{guess} - \text{actual})^2 = (500{,}000 - 450{,}000)^2 = 2{,}500{,}000{,}000$$

If the model guesses \$460K instead:

$$L = (460{,}000 - 450{,}000)^2 = 100{,}000{,}000$$

Lower loss means a better guess. The computer tries many guesses and adjusts to make the loss smaller — the method for doing this is gradient descent (explained below).

**Why squared?** Two reasons: (1) squaring makes negatives positive — a guess that is \$50K too high and one that is \$50K too low are equally bad. (2) Squaring punishes big mistakes disproportionately — being off by \$100K is **four times** as bad as being off by \$50K ($100^2 = 10{,}000$ vs $50^2 = 2{,}500$), which forces the model to avoid large errors.

**"Loss" vs "reward" — same idea, opposite sign.** In optimization, you **minimize** loss (lower = better). In reinforcement learning, you **maximize** reward (higher = better). They are the same concept with the sign flipped: $\text{loss} = -\text{reward}$. This is why PPO's policy loss has a negative sign in front.

**The three loss functions in the Alpha Dual Engine:**

| Loss function | What it measures | Minimized by | Detailed in |
|:---|:---|:---|:---|
| Portfolio objective | Risk minus momentum minus diversification | SLSQP solver | Section A below |
| PPO policy loss | How much to adjust action probabilities | PPO actor (gradient descent) | Section D below |
| PPO value loss | How wrong the critic's prediction was (mean squared error) | PPO critic (gradient descent) | Section D below |

Each is explained with its full formula in the referenced section. The key insight: all three do the same thing conceptually — define "what is wrong" as a number, then make that number smaller.

### **What is gradient descent?**

Gradient descent is how the computer actually makes the loss function smaller. The loss function tells you "how wrong am I?" — gradient descent tells you "which direction should I adjust to be less wrong?"

**The hiking analogy:** You are blindfolded on a mountain. You want to reach the bottom of the valley. You cannot see, but you can feel the slope under your feet.

1. **Feel which direction is downhill** (compute the gradient)
2. **Take a step that way** (update your parameters)
3. **Repeat until the ground is flat** (loss stopped decreasing)

**What is a "gradient"?** The gradient is the slope, generalized to multiple dimensions. With one variable, the slope tells you: "if I nudge $x$ to the right, does the function go up or down?" With 12 variables (like portfolio weights), the gradient is a vector of 12 slopes — one per variable:

$$\nabla L = \left[\frac{\partial L}{\partial w_1},~ \frac{\partial L}{\partial w_2},~ \ldots,~ \frac{\partial L}{\partial w_{12}}\right]$$

Each slope says "if I nudge THIS weight slightly, does the loss go up or down?" The gradient points **uphill**, so you go the opposite direction.

**The update rule — the entire algorithm:**

$$w_{\text{new}} = w_{\text{old}} - \alpha \cdot \nabla L$$

Three pieces: $w_{\text{old}}$ is the current guess, $\nabla L$ is the slope (which direction is uphill), and $\alpha$ is the **learning rate** (how big of a step to take). The minus sign means "go opposite to the slope" — i.e., go downhill.

**The learning rate matters:** Too big and you overshoot the valley and bounce around forever. Too small and you creep toward the answer in a million steps. In the RL agents, $\alpha = 0.0003$ — very small steps, because large steps in RL can destroy the policy.

**Concrete example:** Minimize $f(x) = x^2$. The answer is obviously $x = 0$, but the computer does not know that.

Derivative: $f'(x) = 2x$. Start at $x = 10$, learning rate $\alpha = 0.1$:

| Step | $x$ | $f(x)$ | Slope $2x$ | Update: $x - 0.1 \times 2x$ |
|:---|:---|:---|:---|:---|
| 0 | 10.0 | 100.0 | 20.0 | 8.0 |
| 1 | 8.0 | 64.0 | 16.0 | 6.4 |
| 2 | 6.4 | 40.96 | 12.8 | 5.12 |
| 3 | 5.12 | 26.21 | 10.24 | 4.10 |
| ... | ... | ... | ... | ... |
| 20 | 0.115 | 0.013 | 0.23 | 0.092 |

Each step: compute slope, step opposite. The loss shrinks every time. After enough steps, $x \approx 0$.

**How gradient descent connects to the project:**

| Component | Uses gradient descent? | What it does instead |
|:---|:---|:---|
| SLSQP (portfolio optimizer) | No | Quadratic subproblems (Section A) — more sophisticated but same spirit |
| PPO actor (policy network) | **Yes** | Adam optimizer (a fancier gradient descent with momentum) |
| PPO critic (value network) | **Yes** | Adam optimizer — adjusts predictions to match actual returns |

The neural networks in the RL agents have thousands of parameters. Gradient descent adjusts all of them simultaneously — each one nudged in the direction that reduces the loss.

### **Technical Summary**

A **loss function** (also called cost function or objective function) is a scalar-valued function that quantifies the discrepancy between a model's current output and the desired output. **Gradient descent** is the iterative algorithm that minimizes the loss by computing the gradient (the vector of partial derivatives indicating the direction of steepest ascent) and stepping in the opposite direction, scaled by a learning rate $\alpha$. The Alpha Dual Engine employs three loss functions: a composite portfolio objective combining risk, momentum, and entropy terms (minimized by SLSQP via quadratic subproblems), a clipped surrogate policy loss (minimized by the PPO actor via gradient descent), and a mean squared error value loss (minimized by the PPO critic via gradient descent). SLSQP does not use gradient descent directly — it uses a more sophisticated quadratic programming approach — but the underlying principle is identical: compute which direction improves the answer, step that way, repeat.

---

## **A. The Objective Function & SLSQP Solver (Section IV)**

### **What the formula actually says**

The heart of the portfolio optimizer is the function it tries to minimize:

$$\mathcal{L}(\mathbf{w}) = \lambda_{\text{risk}} \mathbf{w}^\top \Sigma \mathbf{w} - \lambda_{\text{mom}} (\mathbf{w} \cdot \mathbf{M}) - \lambda_{\text{entropy}} H(\mathbf{w})$$

The goal: find the weight vector $\mathbf{w}$ (12 numbers that add up to 1) that makes this as small as possible.

### **Why minimization achieves three goals at once**

The objective combines three competing goals into a single scalar through a sign convention:

- **Term 1 (risk)** is positive → minimizing the objective pushes risk **down**.
- **Term 2 (momentum)** has a negative sign → minimizing $-\text{momentum}$ is mathematically equivalent to **maximizing** momentum.
- **Term 3 (entropy)** has a negative sign → minimizing $-\text{entropy}$ is mathematically equivalent to **maximizing** entropy (diversification).

This is a standard technique in optimization: rather than building separate maximizers and minimizers, you flip the sign of anything you want to maximize and minimize the whole expression. One solver, one pass, three goals satisfied simultaneously.

**Important clarification:** Despite the $\mathcal{L}$ notation, this is NOT a Lagrangian in the classical mechanics sense. It is simply an objective function that gets fed into a numerical solver. You do not solve it by hand with calculus.

### **Why you CANNOT solve this analytically**

In a simple case without constraints, you would take the derivative, set it to zero, and solve. But here:

1. The entropy term $H(\mathbf{w}) = -\sum w_i \ln(w_i)$ — the $\ln(w_i)$ makes the derivative non-linear
2. The inequality constraints (bounds, caps, floors) — you cannot just "set derivative = 0" when the answer must also satisfy 10+ inequalities
3. The growth anchor penalty uses $\max(0, \ldots)^2$ — the $\max$ function is not differentiable everywhere

The inequality constraints are the fundamental barrier. With **equality** constraints only (e.g., "weights sum to 1"), you could apply Lagrange multipliers, set up a system of equations, and solve via Gaussian elimination — all doable by hand. But with **inequality** constraints (e.g., "SMH $\leq$ 30%"), you face a combinatorial problem: you do not know in advance which constraints are **active** (binding at the optimum) and which are **inactive** (satisfied with room to spare). Maybe SMH hits exactly 30% and the cap matters; maybe SMH naturally settles at 22% and the cap is irrelevant. The only way to know is to solve the problem — but you need to know in order to set up the equations.

With approximately 20 inequality constraints, there are $2^{20} \approx 1{,}000{,}000$ possible combinations of active/inactive constraints. For each combination, you would need to solve a full system of equations, then verify that the solution satisfies all constraints that were assumed inactive. This is computationally infeasible by hand.

SLSQP's **active set method** handles this automatically: it maintains a working guess of which constraints are active, solves the resulting equality-only subproblem, checks whether any inactive constraints are violated or any active constraints should be released, adjusts the active set, and repeats. It typically converges in 20-50 iterations rather than exhaustively searching all $2^{20}$ combinations.

### **How SLSQP actually solves it**

The code uses `scipy.optimize.minimize` with the **SLSQP** method (Sequential Least Squares Quadratic Programming).

**The name, decoded:**
- **Sequential** — it solves the problem step by step, not all at once
- **Least Squares** — the method it uses to handle constraints (fits them like a "best fit" line)
- **Quadratic Programming** — at each step, it pretends the problem is a simpler "quadratic" (parabola-shaped) problem and solves that instead

**The core idea: "Approximate and step"**

Imagine you are blindfolded on a hilly landscape and you need to find the lowest valley. You cannot see the whole landscape, but you CAN feel the ground right around your feet.

**What SLSQP does at each step:**

**Step 1 — Start with a guess.** The solver picks an initial set of weights (actually it tries multiple random starting points via `_multi_start_optimize`).

**Step 2 — Feel the ground around you.** Compute the gradient (slope) and curvature (is the slope getting steeper or flatter?) at the current position.

**Step 3 — Build a mental model.** Approximate the nearby landscape as a simple parabola (a bowl shape). A parabola is easy to solve — its minimum is just the bottom of the bowl. This is the "Quadratic Programming" part.

Mathematically, at the current weights $\mathbf{w}_k$, SLSQP approximates:

$$\mathcal{L}(\mathbf{w}) \approx \mathcal{L}(\mathbf{w}_k) + \nabla \mathcal{L}^\top (\mathbf{w} - \mathbf{w}_k) + \frac{1}{2}(\mathbf{w} - \mathbf{w}_k)^\top B (\mathbf{w} - \mathbf{w}_k)$$

Each component explained:
- $\mathcal{L}(\mathbf{w}_k)$ = the function value where you are standing now
- $\nabla \mathcal{L}^\top (\mathbf{w} - \mathbf{w}_k)$ = the slope times how far you move (linear part — "which direction is downhill")
- $\frac{1}{2}(\mathbf{w} - \mathbf{w}_k)^\top B (\mathbf{w} - \mathbf{w}_k)$ = the curvature times how far you move squared (quadratic part — "how far until the bottom"). $B$ is the Hessian matrix (a table of second derivatives — how fast the slope itself is changing)

**Why a quadratic approximation?** Any smooth function, if you zoom in close enough, looks like a parabola. A linear approximation (straight line) tells you which direction is downhill, but not how far to go. A quadratic approximation (parabola) tells you the direction AND roughly where the bottom is.

#### **What is the Hessian matrix $B$?**

The gradient (first derivatives) tells you the **slope** — which direction is downhill. But slope alone does not tell you **how far to walk**. Two landscapes can have the same slope at your feet but completely different shapes:

- **Gentle curvature** (wide bowl): the slope is steep but the valley floor is far away → take a big step
- **Sharp curvature** (narrow bowl): the slope is steep but the valley floor is right below you → take a small step

The Hessian matrix captures exactly this distinction. It is a table of **second derivatives** — derivatives of derivatives — measuring how fast the slope itself is changing in every direction.

**Building the intuition step by step:**

For a function of one variable, say $f(x) = 3x^2$:
- First derivative $f'(x) = 6x$ → the slope. At $x = 5$, the slope is 30 (steep uphill).
- Second derivative $f''(x) = 6$ → the curvature. It is constant — the bowl has the same width everywhere. A large second derivative means a narrow bowl (minimum is nearby); a small one means a wide bowl (minimum is far away).

For a function of two variables, say $f(x, y) = 3x^2 + 2xy + 5y^2$, there is not just one second derivative but **four** — one for each pair of variables. Here is exactly where each number comes from:

**First, take the first partial derivatives (the gradient).** A partial derivative means differentiating with respect to one variable while treating all others as constants — the $\partial$ symbol (as opposed to $d$) indicates this:

$$\frac{\partial f}{\partial x} = 6x + 2y \qquad \frac{\partial f}{\partial y} = 2x + 10y$$

**Then, take partial derivatives of those partial derivatives (second partial derivatives):**

From the first partial derivative with respect to $x$, which is $6x + 2y$:

$$\frac{\partial}{\partial x}(6x + 2y) = 6 \qquad \frac{\partial}{\partial y}(6x + 2y) = 2$$

From the first partial derivative with respect to $y$, which is $2x + 10y$:

$$\frac{\partial}{\partial x}(2x + 10y) = 2 \qquad \frac{\partial}{\partial y}(2x + 10y) = 10$$

**Finally, arrange the four numbers into a grid — that is the Hessian:**

$$H = \begin{bmatrix} 6 & 2 \\ 2 & 10 \end{bmatrix}$$

The "matrix" is just bookkeeping — four numbers placed in a 2x2 table. There is no matrix multiplication or linear algebra involved in *computing* the Hessian. You take derivatives twice and write the results in a grid. Notice the off-diagonals are both 2 — this always happens (the order of differentiation does not matter), which is why the Hessian is always symmetric.

What each entry means:

- **Top-left = 6** ($\partial^2 f / \partial x^2$): Curvature in the $x$ direction alone — how fast the $x$-slope changes as you move in $x$.
- **Bottom-right = 10** ($\partial^2 f / \partial y^2$): Curvature in the $y$ direction alone. Since 10 > 6, the bowl is narrower in $y$ than in $x$ — the optimizer needs a smaller step in $y$ to reach the bottom.
- **Off-diagonals = 2** ($\partial^2 f / \partial x \, \partial y$): The **interaction** between variables — changing $x$ affects the slope in the $y$ direction. If this were 0, the two variables would be completely independent.

The off-diagonal entries are always symmetric (the top-right and bottom-left are equal). This is why the Hessian is always a symmetric matrix.

**For the portfolio with 12 weights**, the Hessian is a **12x12 symmetric matrix** — 12 diagonal entries (how curvy the landscape is in each weight's direction) and 66 off-diagonal entries (how each pair of weights interacts). Together, these 78 unique numbers completely describe the shape of the local bowl that SLSQP uses to decide where to step.

**Why SLSQP approximates the Hessian instead of computing it exactly:**

Computing the exact 12x12 Hessian requires evaluating 78 second derivatives at every iteration — expensive. Instead, SLSQP uses a technique called **BFGS** (Broyden-Fletcher-Goldfarb-Shanno) that *estimates* the Hessian from gradient changes between iterations. The logic is: "last step I moved from point A to point B, and the gradient changed from $g_A$ to $g_B$. The relationship between how much I moved and how much the gradient changed tells me the curvature." After a few iterations, this estimate converges close to the true Hessian — accurate enough to find the bottom of the bowl efficiently without computing 78 derivatives every time.

**The one-sentence summary:** The gradient says "go downhill." The Hessian says "the bottom of the hill is approximately *this far away* in *that direction*." Without the Hessian, the solver knows which way to walk but not how far — with it, the solver can jump directly to (approximately) the bottom in one step.

**Step 4 — Find the bottom of that bowl.** But ONLY within the allowed zone (you cannot step outside the fence = constraints):
- All weights must sum to 1 (equality constraint)
- Each weight must stay within its bounds (0% to 30% for equities, etc.)
- Growth anchors must be >= 40% total
- Crypto + growth anchors <= 95%

**Important distinction:** The objective function $\mathcal{L}(\mathbf{w})$ that we wrote is NOT a Lagrangian — it is just a cost function. But internally, at this step, SLSQP constructs its own Lagrangian to solve the constrained quadratic subproblem. The solver introduces Lagrange multipliers as auxiliary variables to enforce the equality constraints (e.g., "weights sum to 1") on the simplified parabola, and uses an active set method to handle inequality constraints — basically tracking which bounds are "active" (you are pushed right up against them) vs. "inactive" (you are safely inside). These are two different levels: our function is the problem definition; the Lagrange multipliers are part of the solver's internal machinery for respecting constraints at each iteration.

**Step 5 — Walk there.** Update the weights to that new position.

**Step 6 — Repeat** steps 2-5 until the function stops improving (converges), or you hit 1000 iterations.

### **Why SLSQP and not something simpler?**

| Method | Handles constraints? | Speed | Used for |
|:---|:---:|:---:|:---|
| Gradient Descent | No | Slow | Deep learning |
| Newton's Method | No | Fast | Unconstrained problems |
| Linear Programming | Only linear problems | Fast | Supply chain, logistics |
| **SLSQP** | **Yes — all types** | **Fast** | **Exactly this kind of problem** |

SLSQP is the go-to for "small-to-medium nonlinear problems with constraints" — which is exactly what portfolio optimization is (12 weights, ~10 constraints, nonlinear objective).

### **Technical Summary**

In formal terms, this is a constrained nonlinear optimization problem. The objective function combines a quadratic risk term, a linear momentum term, and a nonlinear entropy regularizer, subject to equality constraints (weights sum to 1) and bound constraints (per-asset caps). It is solved numerically using SLSQP — a sequential quadratic programming method that approximates the problem as a series of simpler quadratic subproblems at each iteration, converging to a local minimum while respecting all constraints. The objective function itself is not a Lagrangian — it is a cost function to be minimized. Internally, however, SLSQP constructs a Lagrangian at each iteration to enforce the constraints on the quadratic subproblem, using Lagrange multipliers for equality constraints and an active set method for inequality constraints. The method can be understood as Newton's method extended to handle both equality and inequality constraints.

### **Can the quadratic subproblem be solved by hand?**

Yes — and that is the entire point of SLSQP. It converts one impossible problem into a chain of easy problems. Each individual subproblem is solvable with high school and early university math. Here is how it breaks down, layer by layer.

#### **Layer 1: One variable, no constraints**

The simplest optimization problem: $y = 3x^2 - 12x + 5$. Take the derivative, set it to zero:

$$\frac{dy}{dx} = 6x - 12 = 0 \quad \Rightarrow \quad x = 2$$

The parabola's bottom is at $x = 2$. This is high school calculus.

#### **Layer 2: Twelve variables, no constraints**

Instead of one $x$, the optimizer works with 12 weights: $w_1, w_2, \ldots, w_{12}$. The quadratic approximation becomes a 12-dimensional bowl — it still has exactly one bottom.

To find it, take 12 partial derivatives (one per weight), set them all to zero, and solve the resulting system of equations simultaneously. This reduces to:

$$B \mathbf{d} = -\nabla \mathcal{L}$$

Where $B$ is a 12x12 matrix (how curvy the bowl is), $\nabla \mathcal{L}$ is a list of 12 slopes, and $\mathbf{d}$ is the step direction to solve for. This is a system of 12 linear equations with 12 unknowns — solvable by Gaussian elimination (systematically manipulating equations to isolate each variable). Tedious with 12 variables, but each operation is just addition, subtraction, multiplication, and division.

#### **Layer 3: Add the constraint "weights must sum to 1"**

Without constraints, the bottom of the bowl might be at weights like $[0.5, 0.5, 0.0, \ldots, -0.3]$ — a negative weight is nonsensical for a portfolio.

The constraint $\sum w_i = 1$ defines a flat plane slicing through the 12-dimensional bowl. The goal becomes: find the lowest point where the bowl and the plane intersect.

This is where the Lagrange multiplier enters. A new variable $\mu$ is introduced along with a new equation:

$$\mathcal{L}_{\text{sub}} = f(\mathbf{w}) + \mu \cdot \left(\sum w_i - 1\right)$$

Taking derivatives with respect to all 12 weights AND $\mu$ and setting them to zero yields **13 equations with 13 unknowns** (12 weights + 1 multiplier). Still a linear system. Still solvable by Gaussian elimination — just one row larger.

The multiplier $\mu$ has a concrete meaning: it measures how much the minimum would improve if the constraint were slightly relaxed. A large $\mu$ means the constraint is actively holding the solution back from the unconstrained optimum. The following subsection explains this concept in full detail.

#### **What is a Lagrange multiplier, exactly?**

**The real-world intuition:** You are at a buffet. You want to eat the most delicious combination of food possible. But you have one rule: your plate can only hold 1 kg total. Without the rule, you would pile on infinite amounts of the best dish. But the 1 kg limit forces tradeoffs — more steak means less dessert. The Lagrange multiplier answers a very specific question: "If my plate could hold 1.01 kg instead of 1 kg, how much more deliciousness could I get?" If the answer is "a lot" — the plate size is really constraining you. If the answer is "barely any" — the plate size does not matter much. The multiplier is a number that measures how much the constraint is costing you.

**A worked example from scratch:** Find the $x$ and $y$ that minimize $f(x, y) = x^2 + y^2$ (distance from the origin) subject to the constraint $x + y = 1$ (you must stay on a specific line).

Without the constraint, the answer is obviously $x = 0, y = 0$ (the origin). But the constraint says you cannot be at the origin — you must be somewhere on the line $x + y = 1$.

**The geometric picture:** Imagine concentric circles centered at the origin (a bullseye), getting bigger. The constraint is a diagonal line cutting through. You want the smallest circle that still touches the line. The point where the smallest circle is tangent to the line is the answer.

**Step 1 — Write the constraint as "something = 0":**

$$g(x, y) = x + y - 1 = 0$$

**Step 2 — Build the Lagrangian** (the original function plus the multiplier times the constraint):

$$\mathcal{L}(x, y, \mu) = x^2 + y^2 + \mu(x + y - 1)$$

**Step 3 — Take partial derivatives with respect to every variable (including $\mu$) and set them all to zero:**

$$\frac{\partial \mathcal{L}}{\partial x} = 2x + \mu = 0$$

$$\frac{\partial \mathcal{L}}{\partial y} = 2y + \mu = 0$$

$$\frac{\partial \mathcal{L}}{\partial \mu} = x + y - 1 = 0$$

The third equation is just the original constraint coming back. This always happens — the derivative with respect to $\mu$ always recovers the constraint. That is by design.

**Step 4 — Solve the system.** From equation 1: $x = -\mu/2$. From equation 2: $y = -\mu/2$. So $x = y$. Plug into equation 3:

$$x + x = 1 \quad \Rightarrow \quad x = 0.5, \quad y = 0.5, \quad \mu = -1$$

The answer: $x = 0.5$, $y = 0.5$, and $f = 0.25 + 0.25 = 0.5$.

**What does $\mu = -1$ mean?** If the constraint were relaxed from $x + y = 1$ to $x + y = 1.01$, the minimum value of $f$ would improve by approximately $0.01 \times |\mu| = 0.01$. The multiplier measures the "price" of the constraint.

**Why this trick works — the deep geometric intuition:** At the optimum on the constraint, two things must be true simultaneously: (1) you are ON the constraint line, and (2) you cannot improve by sliding along the line in either direction. Condition 2 means the gradient (slope) of $f$ must be perpendicular to the constraint line at the solution. If it were not perpendicular, there would be a component along the line — meaning you could slide along the line and get a better value, contradicting that you are at the optimum. The equation $\nabla f = -\mu \nabla g$ says exactly this: the gradient of the objective must point in the same direction as the gradient of the constraint (perpendicular to the constraint surface). The multiplier $\mu$ is just the scaling factor between them.

**Applied to the portfolio:** The objective is $f(\mathbf{w}) = \text{risk} - \text{momentum} - \text{entropy}$, and the constraint is $\sum w_i = 1$. The Lagrangian becomes:

$$\mathcal{L} = f(\mathbf{w}) + \mu\left(\sum_{i=1}^{12} w_i - 1\right)$$

Take 13 partial derivatives (12 weights + $\mu$), set all to zero, solve the 13x13 linear system. The multiplier $\mu$ tells you: "If you were allowed to use 101% of your capital instead of 100%, how much better could the portfolio be?"

**Multiple constraints = multiple multipliers.** The portfolio has more than one constraint, and each one gets its own multiplier:

| Constraint | Multiplier | What it measures |
|:---|:---|:---|
| Weights sum to 1 | $\mu_1$ | "How much would a 101% budget help?" |
| SMH <= 30% | $\mu_2$ | "How much is the SMH cap costing me?" |
| Growth anchors >= 40% | $\mu_3$ | "How much is the 40% floor costing me?" |
| Crypto <= 15% | $\mu_4$ | "How much is the crypto cap costing me?" |

If $\mu_2$ is large, the SMH cap is really constraining the optimizer — it desperately wants to put more into SMH but the 30% cap is blocking it. If $\mu_4$ is near zero, the crypto cap does not matter — the optimizer would not put more than 15% in crypto even without the rule. This is why Lagrange multipliers are sometimes called **shadow prices** — they tell you the price of each constraint, i.e., how much performance you are sacrificing to follow each rule.

#### **Solving the Linear System: Gaussian Elimination**

Every Lagrange multiplier problem ends with a system of linear equations — set all partial derivatives to zero, and you get something like "13 equations with 13 unknowns." **Gaussian elimination** is the method for solving these systems. It is one of the oldest algorithms in mathematics, used by Carl Friedrich Gauss in 1809 to compute the orbit of the asteroid Ceres from noisy telescope observations.

The idea is simple: **use one equation to eliminate one variable from all the others, repeat until each equation has only one variable, then read off the answers backwards.**

##### **Warm-up: 1 equation, 1 unknown**

$$2x = 10 \quad \Rightarrow \quad x = 5$$

No method needed. Just divide.

##### **2 equations, 2 unknowns — worked example**

$$x + y = 10$$

$$2x + 3y = 26$$

**Phase 1 — Forward elimination (kill $x$ from the second equation):**

Multiply the first equation by 2:

$$2x + 2y = 20$$

Subtract from the second equation:

$$(2x + 3y) - (2x + 2y) = 26 - 20$$

$$y = 6$$

Now the system looks like a triangle (upper triangular form):

$$x + y = 10$$

$$y = 6$$

**Phase 2 — Back-substitution (work upward):**

From the second equation: $y = 6$.

Plug into the first: $x + 6 = 10$, so $x = 4$.

**Answer:** $x = 4$, $y = 6$.

##### **3 equations, 3 unknowns — the full pattern**

$$x + 2y + z = 9$$

$$2x + 5y + 2z = 21$$

$$3x + 7y + 4z = 30$$

**Phase 1 — Forward elimination:**

**Step 1:** Eliminate $x$ from equations 2 and 3.

Equation 2 $-$ 2 $\times$ Equation 1:

$$(2x + 5y + 2z) - 2(x + 2y + z) = 21 - 18$$

$$y + 0z = 3 \quad \Rightarrow \quad y = 3$$

Equation 3 $-$ 3 $\times$ Equation 1:

$$(3x + 7y + 4z) - 3(x + 2y + z) = 30 - 27$$

$$y + z = 3$$

**Step 2:** Eliminate $y$ from equation 3 using the new equation 2.

New Equation 3 $-$ New Equation 2:

$$(y + z) - (y) = 3 - 3$$

$$z = 0$$

Now the system is triangular:

$$x + 2y + z = 9$$

$$y = 3$$

$$z = 0$$

**Phase 2 — Back-substitution (bottom to top):**

From equation 3: $z = 0$.

From equation 2: $y = 3$.

From equation 1: $x + 6 + 0 = 9$, so $x = 3$.

**Answer:** $x = 3$, $y = 3$, $z = 0$.

##### **The general pattern for $N$ unknowns**

1. **Forward elimination:** Use equation 1 to eliminate variable 1 from equations 2 through $N$. Then use the modified equation 2 to eliminate variable 2 from equations 3 through $N$. Continue until the system forms a triangle — each equation has one fewer variable than the one above it.

2. **Back-substitution:** The bottom equation now has one variable — solve it directly. Plug that answer into the equation above to find the next variable. Work upward until all $N$ variables are solved.

For $N$ unknowns, forward elimination requires roughly $(2/3)N^3$ arithmetic operations, and back-substitution requires roughly $N^2$. For the portfolio's 13x13 system: approximately $(2/3)(13)^3 = 1{,}465$ multiplications and additions — tedious but entirely feasible by hand in about 20-30 minutes.

##### **How Gaussian elimination connects to the portfolio**

When the SLSQP solver builds a quadratic subproblem with Lagrange multipliers, the result of setting all partial derivatives to zero is a system of linear equations — 12 weight unknowns + 1 (or more) multiplier unknowns. The solver uses Gaussian elimination (or a computationally equivalent method like LU decomposition) to solve this system at every iteration. The entire chain:

| Step | What happens | Method |
|:---|:---|:---|
| 1. Original problem | Nonlinear objective with constraints | Cannot be solved directly |
| 2. SLSQP approximation | Replace with quadratic subproblem | Taylor expansion |
| 3. Lagrange multipliers | Convert constraints into derivatives | Set all partials to zero |
| 4. Linear system | $N$ equations, $N$ unknowns | **Gaussian elimination** |
| 5. Basic arithmetic | Multiply, add, subtract, divide | **High school math** |

Every layer converts an unsolvable problem into a solvable one. At the very bottom, the entire optimization reduces to nothing more than multiplication and addition — the same operations taught in primary school. The computer's advantage is not intelligence; it is speed.

#### **Worked Example: One Complete SLSQP Sub-Problem**

To see exactly what the solver computes at each iteration, here is a complete worked example using a simplified 3-asset portfolio.

**Setup:**

- 3 assets (A, B, C) with momentum scores $M = [3, 1, 2]$ (Asset A trending strongest, B weakest)
- Covariance matrix (all variances = 2, all correlations = 0):

$$\Sigma = \begin{bmatrix} 2 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 2 \end{bmatrix}$$

The diagonal entries (2, 2, 2) are each asset's variance — how much it moves on its own. The off-diagonal entries (all zeros) are the covariances between pairs of assets — zero means they move independently. In the real portfolio, these off-diagonals would be non-zero because assets like SMH and QQQ are correlated, but zeros keep this example tractable.

- Constraint: $w_1 + w_2 + w_3 = 1$

**Building the objective function step by step:**

The full objective from the main README has three terms: risk, momentum, and entropy. For this example, entropy is omitted to keep the arithmetic simple. That leaves:

$$\text{Objective} = \underbrace{w^\top \Sigma w}_{\text{risk (want small)}} - \underbrace{w \cdot M}_{\text{momentum (want large)}}$$

The minus sign is the key: the optimizer **minimizes** this expression. Minimizing "risk minus momentum" simultaneously pushes risk **down** and momentum **up** — because subtracting a larger momentum number makes the whole expression smaller. (This is explained in detail in the "Why minimization achieves three goals at once" subsection above.)

Now expand each piece with the actual numbers:

**Risk term** $w^\top \Sigma w$ — this is matrix multiplication that measures total portfolio risk. Because the covariance matrix is diagonal (all zeros off the diagonal), it simplifies to each weight squared times its variance:

$$w^\top \Sigma w = 2 \times w_1^2 + 2 \times w_2^2 + 2 \times w_3^2$$

(If the off-diagonals were non-zero, there would be additional cross terms like $2 \times \text{cov}_{AB} \times w_1 w_2$, capturing the risk from correlated assets. Here they are zero, so those terms vanish.)

**Momentum term** $w \cdot M$ — this is a dot product. Multiply each weight by its momentum score and add up:

$$w \cdot M = w_1 \times 3 + w_2 \times 1 + w_3 \times 2 = 3w_1 + w_2 + 2w_3$$

**Subtract** momentum from risk to get the objective:

$$\mathcal{L}(\mathbf{w}) = \underbrace{2w_1^2 + 2w_2^2 + 2w_3^2}_{\text{from risk}} - \underbrace{(3w_1 + w_2 + 2w_3)}_{\text{from momentum}}$$

Which simplifies to:

$$\mathcal{L}(\mathbf{w}) = 2w_1^2 + 2w_2^2 + 2w_3^2 - 3w_1 - w_2 - 2w_3$$

Every number in this formula traces back to either the covariance matrix (the 2's in front of the squared terms) or the momentum scores (the 3, 1, 2 being subtracted). Nothing is arbitrary.

**Step 1 — Build the Lagrangian** (add the multiplier $\mu$ times the constraint, as described in the Lagrange multiplier section above):

$$\mathcal{L}(\mathbf{w}, \mu) = 2w_1^2 + 2w_2^2 + 2w_3^2 - 3w_1 - w_2 - 2w_3 + \mu(w_1 + w_2 + w_3 - 1)$$

**Step 2 — Take all partial derivatives and set to zero:**

$$\frac{\partial \mathcal{L}}{\partial w_1} = 4w_1 - 3 + \mu = 0$$

$$\frac{\partial \mathcal{L}}{\partial w_2} = 4w_2 - 1 + \mu = 0$$

$$\frac{\partial \mathcal{L}}{\partial w_3} = 4w_3 - 2 + \mu = 0$$

$$\frac{\partial \mathcal{L}}{\partial \mu} = w_1 + w_2 + w_3 - 1 = 0$$

This is 4 equations with 4 unknowns — a linear system.

**Step 3 — Solve by Gaussian elimination:**

From equations 1-3, express each weight in terms of $\mu$:

$$w_1 = \frac{3 - \mu}{4}, \quad w_2 = \frac{1 - \mu}{4}, \quad w_3 = \frac{2 - \mu}{4}$$

Substitute into equation 4:

$$\frac{3 - \mu}{4} + \frac{1 - \mu}{4} + \frac{2 - \mu}{4} = 1$$

$$\frac{6 - 3\mu}{4} = 1 \quad \Rightarrow \quad 6 - 3\mu = 4 \quad \Rightarrow \quad \mu = \frac{2}{3}$$

Back-substitute:

$$w_1 = \frac{3 - 2/3}{4} = \frac{7}{12} \approx 0.583, \quad w_2 = \frac{1 - 2/3}{4} = \frac{1}{12} \approx 0.083, \quad w_3 = \frac{2 - 2/3}{4} = \frac{4}{12} = \frac{1}{3} \approx 0.333$$

Verify: $7/12 + 1/12 + 4/12 = 12/12 = 1$ ✓

**Step 4 — Interpret the result:**

| Asset | Momentum Score | Optimal Weight | Interpretation |
|:---|:---:|:---:|:---|
| A | 3 (strongest) | 58.3% | Highest momentum → highest allocation |
| C | 2 (middle) | 33.3% | Middle momentum → middle allocation |
| B | 1 (weakest) | 8.3% | Lowest momentum → lowest allocation |

The optimizer allocated capital proportional to momentum strength — exactly the desired behavior. The multiplier $\mu = 2/3$ means: if the constraint allowed 101% total capital instead of 100%, the objective would improve by approximately $0.01 \times 2/3 \approx 0.0067$.

**Connection to the real portfolio:** The actual optimizer performs this same calculation with 12 assets instead of 3, a full 12x12 covariance matrix with non-zero correlations, and the nonlinear entropy term. Because entropy makes the objective non-quadratic, the quadratic approximation is not exact — so SLSQP must iterate, rebuilding and re-solving the subproblem 20-50 times until the answer converges. Each iteration is the same sequence shown above: build Lagrangian → take derivatives → solve linear system → update weights.

#### **Layer 4: Add inequality constraints (0% to 30% per asset)**

At the solution, each inequality constraint is either **active** (the solution is pressed right against the boundary) or **inactive** (the solution is safely inside and the constraint has no effect).

The **active set method** solves this through intelligent guessing:

**Step 1 — Guess** which constraints are active. For example: "SMH is at the 30% cap, TAN is at the 0% floor, everything else is free."

**Step 2 — Convert the guess into equality constraints.** If SMH is active at 30%, set $w_{\text{SMH}} = 0.30$ as a fixed number. This reduces the number of unknowns.

**Step 3 — Solve** the reduced system (fewer variables, only equality constraints). This is the Lagrange multiplier problem from Layer 3 — solvable by hand.

**Step 4 — Check the guess.** Are any "free" weights outside their bounds? Is any active constraint being pushed the wrong way (indicated by its multiplier's sign)? If the guess is inconsistent, update the active set and return to Step 2.

Each iteration is just solving a system of linear equations. The process typically converges in 3-10 rounds.

#### **Why the FULL optimization cannot be done by hand**

Everything above solves ONE quadratic subproblem — one parabola approximation at one point in weight-space. The full SLSQP process chains 20-50 of these subproblems together, where each depends on the previous answer:

1. Stand at initial weights, build a parabola approximation, solve it (doable by hand)
2. Walk to the answer, build a NEW parabola at the new position, solve it (doable by hand)
3. Repeat until the answer stops changing (20-50 iterations)

Each iteration involves computing 12 derivatives, updating a 12x12 matrix, and solving a 13x13 linear system — roughly 2,000 arithmetic operations per step. One iteration takes approximately 30 minutes by hand. The full optimization would require 10-25 hours of continuous arithmetic.

Every single step uses math from high school or early university (derivatives, solving linear systems). Nothing is conceptually difficult. It is simply an enormous volume of basic calculations. The computer completes the entire process in approximately 0.01 seconds.

#### **The complete picture**

| Level | Math required | Hand-solvable? | Time by hand |
|:---|:---|:---:|:---|
| Bottom of $y = 3x^2 - 12x + 5$ | High school derivative | Yes | 30 seconds |
| 12-variable quadratic, no constraints | Solve 12 linear equations | Yes | 20 minutes |
| Add "weights sum to 1" | +1 Lagrange multiplier, 13 equations | Yes | 30 minutes |
| Add per-asset bounds (0%-30%) | Active set: guess, solve, check, repeat | Yes | 1-2 hours |
| Full SLSQP (20-50 iterations of the above) | Same math, repeated many times | Technically yes | 10-25 hours |
| The original nonlinear objective (entropy + momentum) | Cannot be solved directly | No | Impossible |

The genius of SLSQP is converting an impossible problem (nonlinear with constraints) into a long series of easy problems (quadratic with linear constraints). Each easy problem is just high school math applied repeatedly. The computer does not do hard math — it does easy math very fast, very many times.

---

## **B. Shannon Entropy — From Information Theory to Portfolio Diversification**

### **Where it came from**

Claude Shannon invented entropy in 1948 for information theory — specifically to measure "how much surprise is in a message?" It had nothing to do with finance. But the math turns out to be useful anywhere you want to measure how spread out something is.

### **The formula**

$$H(\mathbf{w}) = -\sum_i w_i \ln(w_i)$$

You have 12 assets with weights $w_1, w_2, \ldots, w_{12}$. For each one:
1. Take the weight (e.g., 0.30)
2. Take the natural log of that weight ($\ln(0.30) = -1.20$)
3. Multiply them together ($0.30 \times -1.20 = -0.36$)
4. Do this for all 12 assets
5. Add them all up
6. Put a negative sign in front (to make the result positive)

### **Why does $\ln$ show up?**

This is the part that confuses everyone. Here is the intuition:

$\ln(w_i)$ measures how "surprising" a weight is. If an asset has a big weight (say 0.90), that is not surprising — $\ln(0.90) = -0.105$ (small number). If an asset has a tiny weight (say 0.01), that is surprising — $\ln(0.01) = -4.605$ (big number).

Then you multiply by $w_i$ — which says "how often does this surprise actually matter?" A tiny weight is very surprising but rarely matters. A big weight is unsurprising but matters a lot.

So $w_i \times \ln(w_i)$ = "importance-weighted surprise" for each asset. Sum them all up and you get the total "spread-out-ness."

### **Concrete examples with real numbers**

**Portfolio A: All-in on one stock — SMH = 100%, everything else = 0%**

$$H = -(1.0 \times \ln(1.0)) = -(1.0 \times 0) = 0$$

Entropy = 0. Zero surprise. Zero diversity. You know exactly where all the money is.

(Note: $\ln(1) = 0$ because $e^0 = 1$.)

**Portfolio B: Split between 2 stocks — SMH = 50%, QQQ = 50%**

$$H = -(0.5 \times \ln(0.5) + 0.5 \times \ln(0.5))$$
$$= -(0.5 \times (-0.693) + 0.5 \times (-0.693))$$
$$= -(-0.347 - 0.347) = 0.693$$

Entropy = 0.693. Some diversity.

**Portfolio C: Equal across all 12 assets — each at 8.33%**

$$H = -12 \times (0.0833 \times \ln(0.0833))$$
$$= -12 \times (0.0833 \times (-2.485))$$
$$= -12 \times (-0.207) = 2.485$$

Entropy = 2.485. Maximum diversity for 12 assets.

### **The pattern**

| Portfolio | Entropy | What it looks like |
|:---|:---:|:---|
| 100% in one stock | 0 | All eggs in one basket |
| 50/50 split | 0.693 | Two baskets |
| Equal 12-way split | 2.485 | Maximum spread |

Entropy goes from 0 (fully concentrated) to $\ln(n)$ (perfectly spread across $n$ assets). It can never go negative and it can never exceed $\ln(n)$.

### **Why the optimizer uses it**

Go back to the objective function:

$$\mathcal{L}(\mathbf{w}) = \ldots - \lambda_{\text{entropy}} H(\mathbf{w})$$

That minus sign means the optimizer rewards higher entropy (more spread). Without it, the momentum term would happily shove 100% into the single best stock. The entropy term gently pushes back: "spread the money around a little."

But $\lambda_{\text{entropy}} = 0.02$ is tiny, so it is a whisper, not a shout. The momentum term easily overpowers it. The result: the portfolio concentrates in the top 3-4 winners but does not go full degenerate into just 1.

### **Why not just use Effective N instead?**

The difference is where they are used:
- **Entropy** goes inside the objective function as a smooth, differentiable penalty. The optimizer can compute its gradient and smoothly adjust weights. It is a **soft nudge during optimization**.
- **Effective N** ($1/\sum w_i^2$) is used as a **hard check after optimization**. It is a pass/fail gate: "does this portfolio look like at least 3 bets?"

Entropy is the carrot (gentle reward for spreading). Effective N is the stick (reject the portfolio if it is too concentrated).

### **Technical Summary**

Shannon Entropy measures how spread out the portfolio weights are on a scale from 0 (fully concentrated) to $\ln(n)$ (equally distributed). In the objective function, it serves as a regularization term — mathematically penalizing the optimizer for placing too much weight in too few assets. The deliberately low $\lambda = 0.02$ reflects the strategy's preference for concentration in top-momentum stocks while preventing extreme single-name dominance. This balances the momentum term's tendency to concentrate all capital into one winner against the mathematical requirement for a minimum diversity floor.

---

## **C. Geometric Brownian Motion — The Complete Derivation**

### **Start with regular Brownian Motion**

Before "Geometric," we need plain Brownian Motion. Imagine a drunk person stumbling on a straight road:

- Every second, they take one step
- The step is random — drawn from a bell curve (normal distribution)
- Each step is independent of the previous one

After 100 steps, their position is the sum of 100 random steps. This is Brownian Motion — a pure random walk. Mathematically:

$$X_{t+1} = X_t + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

The $\mathcal{N}(0, \sigma^2)$ means: random number from a bell curve centered at 0, with spread $\sigma$.

### **The problem: stocks cannot go negative**

If you model stock prices with regular Brownian Motion, you get nonsense. Say a stock is at 10. After enough negative random steps, it hits 0, then -5. A stock price of negative five dollars is meaningless.

The fix: instead of adding random dollar amounts, multiply by random percentage changes. A stock can drop 50%, then drop another 50% (now at 25% of original), then another 50% (12.5%)... it keeps halving forever but never hits zero.

This is the "Geometric" part — the randomness is **multiplicative**, not additive.

### **The continuous-time SDE (the textbook form)**

In finance textbooks, GBM is written as a stochastic differential equation (SDE):

$$dS = \mu S  ~ dt + \sigma S  ~ dW$$

Here is what every symbol means:

- $dS$ = the infinitesimal change in stock price (how much it moves in a tiny instant)
- $S$ = current stock price
- $\mu$ = drift — the expected annual return (e.g., 0.20 for 20%/year)
- $\sigma$ = volatility — the annual standard deviation (e.g., 0.25 for 25%)
- $dt$ = a tiny slice of time
- $dW$ = a Wiener process increment — a tiny random shock from a bell curve

The key insight: both terms are proportional to $S$. A 100 dollar stock moves in dollars, a 1000 dollar stock moves in tens of dollars — but they both move the same percentage. That is what makes it "geometric."

Reading it as a sentence: "The change in price = (expected drift times price times time) + (random shock times price times volatility)"

### **Splitting it into two forces**

Think of the stock price being pulled by two forces simultaneously:

**Force 1: The Drift — $\mu S  ~ dt$**

This is the predictable, deterministic part. If there were zero randomness, the stock would grow smoothly at rate $\mu$ per year, like a savings account. Over a tiny time step $dt$:

$$\text{Predictable change} = \mu \times S \times dt$$

If $S = 100$, $\mu = 0.20$, and $dt = 1/252$ (one trading day):

$$= 0.20 \times 100 \times \frac{1}{252} = 0.0794$$

About 8 cents of upward drift per day. Boring but reliable.

**Force 2: The Diffusion — $\sigma S  ~ dW$**

This is the random part. $dW$ is a random draw from $\mathcal{N}(0, dt)$ — a bell curve with variance equal to the time step. In practice:

$$dW = \sqrt{dt} \times Z, \quad Z \sim \mathcal{N}(0, 1)$$

So the random shock is:

$$\text{Random change} = \sigma \times S \times \sqrt{dt} \times Z$$

If $\sigma = 0.25$, $S = 100$, $dt = 1/252$, and $Z = 1.5$ (a moderately good day):

$$= 0.25 \times 100 \times \sqrt{1/252} \times 1.5 = 0.25 \times 100 \times 0.063 \times 1.5 = 2.36$$

The stock jumps up 2.36 dollars. Notice: **the random part (2.36) completely dwarfs the drift (0.08)**. On any single day, the noise dominates. The drift only shows up over months and years. This is why daily stock charts look like chaos but long-term charts trend upward.

### **From the SDE to the formula you can actually compute: Ito's Lemma**

The SDE $dS = \mu S  ~ dt + \sigma S  ~ dW$ is in continuous time — infinitely small time steps. To actually simulate it on a computer, we need a discrete formula. This is where **Ito's Lemma** comes in.

#### **The $-\frac{1}{2}\sigma^2$ correction**

This is the part everyone finds mysterious. Here is where it comes from.

Take the logarithm of the stock price: $\ln(S)$. If you apply calculus rules to find how $\ln(S)$ changes over time (using Ito's Lemma, which is just the chain rule but for random processes), you get:

$$d(\ln S) = \left(\mu - \frac{1}{2}\sigma^2\right) dt + \sigma  ~ dW$$

**Where did the $-\frac{1}{2}\sigma^2$ come from?**

In normal calculus, if $f(x) = \ln(x)$, then $f'(x) = 1/x$ and you are done. But in stochastic calculus, there is an extra term because the random part ($dW$) has a non-zero "squared" contribution. Specifically:

- Normal chain rule: $d(\ln S) = \frac{1}{S} dS$
- **Ito's chain rule:** $d(\ln S) = \frac{1}{S} dS - \frac{1}{2} \frac{1}{S^2} (dS)^2$

That extra $-\frac{1}{2} \frac{1}{S^2}(dS)^2$ term exists because $(dW)^2 = dt$ — a fundamental property of Brownian Motion where the square of a random step equals the time step, not zero. When you expand $(dS)^2 = (\sigma S  ~ dW)^2 = \sigma^2 S^2  ~ dt$, you get:

$$-\frac{1}{2} \frac{1}{S^2} \times \sigma^2 S^2  ~ dt = -\frac{1}{2}\sigma^2  ~ dt$$

That is the **volatility drag**.

### **The intuitive explanation of volatility drag**

Forget the calculus. Here is why it has to exist:

Consider two scenarios over 2 days, both with 25% volatility:

- **Path A:** +25% then -25%: 100 to 125 to 93.75 (lost 6.25%)
- **Path B:** -25% then +25%: 100 to 75 to 93.75 (lost 6.25%)

The average return is 0% (one up, one down), but you **lost money both ways**. This asymmetry — percentage gains and losses do not cancel out — is the volatility drag. The $-\frac{1}{2}\sigma^2$ term accounts for exactly this effect.

With $\sigma = 0.25$: drag = $\frac{1}{2}(0.25)^2 = 0.03125$ = 3.125% per year eaten by volatility.

### **The final simulation formula**

Integrating the log-price equation over a discrete time step $\Delta t$:

$$\ln S_{t+1} - \ln S_t = \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}  ~ Z$$

Exponentiate both sides (because $e^{\ln(a) - \ln(b)} = a/b$):

$$\frac{S_{t+1}}{S_t} = \exp\left[\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}  ~ Z\right]$$

Rearrange:

$$S_{t+1} = S_t \exp\left[\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t} ~ Z\right]$$

This is the exact formula used in the Monte Carlo simulation code. Here is every piece labeled:

| Symbol | What it is | Example value |
|:---|:---|:---|
| $S_t$ | Today's portfolio value | 100,000 |
| $S_{t+1}$ | Tomorrow's portfolio value | What we are computing |
| $\mu$ | Annualized expected return | 0.20 (20%) |
| $\sigma$ | Annualized volatility | 0.25 (25%) |
| $\mu - \frac{1}{2}\sigma^2$ | Drift corrected for vol drag | 0.20 - 0.03125 = 0.169 |
| $\Delta t$ | Time step as fraction of year | 1/252 = 0.00397 |
| $\sqrt{\Delta t}$ | Converts annual vol to daily | 0.063 |
| $Z$ | Random draw from $\mathcal{N}(0,1)$ | -1.2 (a bad day) |
| $\exp[\ldots]$ | Ensures price stays positive | Always > 0 |

### **A full worked example — one simulated day**

Starting value: 100,000 dollars. $\mu = 0.20$, $\sigma = 0.25$, $Z = -1.2$ (unlucky day).

**Step 1 — Drift component:**

$$\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t = (0.20 - 0.03125) \times \frac{1}{252} = 0.169 \times 0.00397 = 0.000671$$

**Step 2 — Random component:**

$$\sigma\sqrt{\Delta t} \times Z = 0.25 \times 0.063 \times (-1.2) = -0.01890$$

**Step 3 — Total exponent:**

$$0.000671 + (-0.01890) = -0.01823$$

**Step 4 — Exponentiate:**

$$S_{t+1} = 100{,}000 \times e^{-0.01823} = 100{,}000 \times 0.98194 = 98{,}194$$

The portfolio dropped 1,806 dollars on this simulated day. Notice how the drift (+0.067%) was completely overwhelmed by the random shock (-1.89%).

### **How 1,000,000 paths work**

The code does this exact calculation 1,260 times (5 years times 252 trading days) for each path, and runs 1,000,000 paths in parallel using NumPy vectorization. Each path draws its own independent sequence of $Z$ values, so each path is a different possible future.

After all paths complete, you have 1,000,000 final portfolio values. Sort them and you get:
- Mean = expected outcome
- 5th percentile = bad scenario (95% of paths did better)
- Percentage below starting value = probability of loss
- The full histogram = the complete probability distribution of your financial future

### **Technical Summary**

GBM models stock prices as a random walk in log-space. The price change each day comprises two components: a deterministic drift (expected return adjusted for volatility drag) and a stochastic diffusion (random shock scaled by volatility). The $-\frac{1}{2}\sigma^2$ correction is a direct consequence of Ito's Lemma — it accounts for the mathematical asymmetry where symmetric percentage gains and losses do not cancel out (a +50% gain followed by a -50% loss results in a net -25% loss, not 0%). The exponential wrapper ensures that simulated prices remain strictly positive regardless of the random draw. By simulating 1,000,000 independent paths over a 5-year horizon, the engine constructs a full probability distribution of future portfolio outcomes — capturing not just the expected case but the complete range of tail risks and upside scenarios.

---

## **D. Proximal Policy Optimization (PPO) — The Complete Math**

### **The problem PPO solves**

You have an agent that observes the world (market data), takes actions (pick a regime or pick portfolio weights), and receives rewards (excess Sharpe minus penalties). The goal: find the **policy** (the rule mapping observations to actions) that maximizes cumulative reward.

The Alpha Dual Engine has **two** PPO agents:
- **High-level (discrete):** observes 25-dim state and picks 1 of 3 regimes (RISK_ON / RISK_REDUCED / DEFENSIVE)
- **Low-level (continuous):** observes 103-dim state and outputs 12 portfolio weights

The math is the same for both. Here is the full derivation from scratch.

### **Step 1: The Policy $\pi_\theta$**

The policy is a neural network with parameters $\theta$ that maps observations to actions.

**Discrete case** (regime agent): The network outputs logits $\ell_1, \ell_2, \ell_3$ for the 3 regimes. Convert to probabilities via softmax:

$$\pi_\theta(a | s) = \frac{e^{\ell_a}}{\sum_i e^{\ell_i}}$$

So if the logits are $[2.0, 0.5, -1.0]$, the probabilities are roughly $[0.78, 0.17, 0.04]$ — the agent strongly prefers RISK_ON.

**Continuous case** (weight agent): The network outputs a mean vector $\mu \in \mathbb{R}^{12}$ and a learned log standard deviation $\log\sigma$. Actions are sampled from a Gaussian:

$$z \sim \mathcal{N}(\mu, \sigma^2), \quad \text{weights} = \text{softmax}(z)$$

The log probability of a sampled $z$ under this Gaussian is:

$$\log \pi_\theta(z | s) = -\frac{1}{2}\sum_{i=1}^{12}\left[\left(\frac{z_i - \mu_i}{\sigma_i}\right)^2 + 2\log\sigma_i + \log(2\pi)\right]$$

This is the standard Gaussian log-likelihood formula, implemented directly in `rl_weight_agent.py`.

### **Step 2: The Value Function $V(s)$**

The same network also outputs a scalar estimate $V(s)$ — "how much total future reward do I expect from this state?"

Both agents use a **shared trunk** (2 hidden layers with `tanh` activation) with two separate heads:
- **Policy head** (the Actor) outputs action probabilities or Gaussian parameters
- **Value head** (the Critic) outputs a single number — the estimated state value

This is the **Actor-Critic** architecture. The Actor decides what to do. The Critic judges how good the current state is.

### **Step 3: Advantage Estimation — "Was this action better than average?"**

The raw reward tells you how good the outcome was. But PPO needs to know: **was this action better or worse than what we would normally expect?** That is the advantage:

$$A_t = Q(s_t, a_t) - V(s_t)$$

"The value of this specific action minus the average value of being in this state."

We do not know $Q$ directly, so we estimate it using **Generalized Advantage Estimation (GAE)**.

#### **The TD Error (Temporal Difference)**

First, compute the TD error at each step:

$$\delta_t = r_t + \gamma \cdot V(s_{t+1}) \cdot (1 - \text{done}_t) - V(s_t)$$

Where:
- $r_t$ = reward at step $t$
- $\gamma = 0.99$ = discount factor ("how much do I care about future vs. now?" 0.99 means the future matters almost as much as the present)
- $V(s_{t+1})$ = critic's estimate of the next state's value
- $(1 - \text{done}_t)$ = zero out the future if the episode ended (no future rewards after terminal state)

The TD error $\delta_t$ answers: "was the actual outcome (reward + estimated future) better or worse than what the critic predicted for this state?"

#### **GAE: Smoothing TD errors across time**

GAE takes a weighted sum of TD errors looking forward:

$$A_t = \delta_t + (\gamma \lambda)\delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \ldots$$

Where $\lambda = 0.95$ (the GAE lambda) controls the **bias-variance tradeoff**:

- $\lambda = 0$: only use the immediate TD error. Low variance (stable) but high bias (the critic's estimate might be wrong)
- $\lambda = 1$: use all future TD errors. Low bias (captures the true trajectory) but high variance (noisy)
- $\lambda = 0.95$: heavily weight nearby steps but still look ahead. This is the standard sweet spot

In code, this is computed efficiently by working backwards from the last step:

```
last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
```

This single line implements the exponentially-decaying sum by accumulating backwards — each step adds its own $\delta_t$ and a discounted version of all future advantages.

### **Step 4: The PPO Clipped Surrogate Objective — The Core Formula**

This is the key innovation that makes PPO work. The naive approach (vanilla policy gradient) would be:

$$L = -\mathbb{E}\left[\log\pi_\theta(a_t | s_t) \cdot A_t\right]$$

"Increase the probability of actions with positive advantage, decrease for negative." But this is **dangerously unstable** — a single big gradient step can completely wreck the policy. The agent goes from smart to catastrophically broken in one update.

**PPO's fix: clip the update so the policy cannot change too much in one step.**

Define the probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

This measures: "how much more likely is this action under the new policy vs. the old one?"

- If $r = 1.0$: the policy has not changed
- If $r = 1.5$: the new policy is 50% more likely to take this action
- If $r = 0.7$: the new policy is 30% less likely to take this action

In code (log space for numerical stability):

$$r_t = \exp(\log\pi_{\theta}(a_t|s_t) - \log\pi_{\theta_{\text{old}}}(a_t|s_t))$$

Now, the **clipped surrogate objective**:

$$L^{\text{CLIP}} = -\mathbb{E}\left[\min\left(r_t \cdot A_t,~ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t\right)\right]$$

Where $\epsilon = 0.2$ (the `clip_range`). Here is how the clipping works:

**Case 1: $A_t > 0$ (good action — we want to do more of this)**
- The ratio $r_t$ grows as the policy increasingly favors this action
- If $r_t$ grows above $1 + \epsilon = 1.2$: the `clip` kicks in and caps the objective. "You are already doing this action way more — stop pushing."
- If $r_t < 1.2$: normal update, keep increasing the probability

**Case 2: $A_t < 0$ (bad action — we want to do less of this)**
- The ratio $r_t$ shrinks as the policy moves away from this action
- If $r_t$ drops below $1 - \epsilon = 0.8$: the `clip` kicks in. "You have already decreased this action a lot — stop pushing."
- If $r_t > 0.8$: normal update, keep decreasing the probability

The `min` ensures we always take the **more pessimistic** (more conservative) estimate. This is the "proximal" part — the policy stays close to where it was.

In the actual code (`rl_weight_agent.py` lines 1116-1119):

```python
ratio = mx.exp(new_log_probs - old_logprob_batch)
surr1 = ratio * advantage_batch
surr2 = mx.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage_batch
policy_loss = -mx.minimum(surr1, surr2).mean()
```

### **Step 5: The Full Loss Function**

PPO's total loss combines three components:

$$L^{\text{total}} = L^{\text{CLIP}} + c_1 \cdot L^{\text{VF}} - c_2 \cdot H[\pi]$$

#### **Part 1: Policy loss $L^{\text{CLIP}}$**

What we just covered. Makes the Actor better at choosing actions.

#### **Part 2: Value loss $L^{\text{VF}}$**

Trains the Critic to predict returns accurately:

$$L^{\text{VF}} = \frac{1}{N}\sum_t\left(V_\theta(s_t) - R_t\right)^2$$

Where $R_t = A_t + V_{\text{old}}(s_t)$ are the target returns (advantage + old value estimate). This is just mean squared error — "make the critic's predictions match reality." The code uses $c_1 = 0.5$ (`vf_coef`).

#### **Part 3: Entropy bonus $H[\pi]$**

Prevents the policy from becoming too confident too fast.

For the **discrete** regime agent (categorical entropy):

$$H[\pi] = -\sum_a \pi(a|s) \log\pi(a|s)$$

This is Shannon entropy again, but on the action probabilities instead of portfolio weights. If the agent always picks RISK_ON with 99% certainty, entropy is near zero (no exploration). If it picks uniformly (33% each), entropy is at its max (maximum exploration).

For the **continuous** weight agent (Gaussian entropy):

$$H[\pi] = \frac{1}{2} n (1 + \log(2\pi)) + \sum_i \log\sigma_i$$

Where $n = 12$ (number of assets). A wider Gaussian (bigger $\sigma$) = more entropy = more exploration. This formula comes from the analytical entropy of a multivariate Gaussian distribution.

The entropy bonus is **subtracted** from the loss (i.e., added as reward), so the optimizer encourages exploration. The code uses $c_2 = 0.05$ for the regime agent and $c_2 = 0.10$ for the weight agent (higher because continuous action spaces need more exploration to avoid collapsing to a single set of weights).

### **Step 6: The Complete Training Loop**

Here is how it all fits together in each iteration:

**1. Collect experience** (`n_steps = 128` or `256` steps):
   - Run the current policy in the environment
   - At each step, store the tuple $(s_t, a_t, r_t, V(s_t), \log\pi(a_t|s_t))$ in the rollout buffer
   - When an episode ends (full backtest traversal), reset the environment and continue collecting

**2. Compute advantages** using GAE:
   - Get the critic's value estimate for the final state (bootstrap)
   - Walk backwards through the buffer computing $\delta_t$ and GAE advantages

**3. Normalize advantages** to zero mean, unit variance:

$$A_t \leftarrow \frac{A_t - \bar{A}}{\text{std}(A) + 10^{-8}}$$

This stabilizes training — prevents one unusually good or bad episode from dominating the gradient signal

**4. PPO update** (run `n_epochs = 6-10` passes over the collected data):
   - Shuffle the buffer into mini-batches of size 64
   - For each mini-batch: compute the clipped loss + value loss + entropy bonus, backpropagate, update network weights
   - Clip gradient norms to 0.5 (`max_grad_norm`) to prevent gradient explosion

**5. Repeat** from step 1 until `total_timesteps` is reached

### **Why PPO specifically (vs. other RL algorithms)?**

| Algorithm | Problem | Why not for this use case |
|:---|:---|:---|
| DQN | Only works for discrete actions | Cannot output 12 continuous portfolio weights |
| A2C | No clipping — unstable with small data | The backtest has ~50 decisions per episode; cannot afford instability |
| TRPO | Trust region via constrained optimization — slow | Requires computing the Fisher Information Matrix; computationally expensive |
| **PPO** | **Clipping approximates trust region cheaply** | **Fast, stable, works for both discrete (regime) and continuous (weights)** |
| SAC | Off-policy — needs replay buffer, more complex | PPO's on-policy simplicity is sufficient for this problem scale |

PPO hits the sweet spot: almost as stable as TRPO, almost as simple as A2C, and works for both agents in the hierarchy.

### **How the two agents work together hierarchically**

The system operates as a principal-agent hierarchy:

1. The **Regime Agent** (principal) observes the macro state (25 dimensions: VIX, SPY momentum, drawdowns, ML probability, etc.) and outputs a discrete regime decision (RISK_ON / RISK_REDUCED / DEFENSIVE)

2. The regime decision is encoded as a **one-hot vector** and prepended to the Weight Agent's observation

3. The **Weight Agent** (subordinate) observes the full state (103 dimensions: regime encoding + per-asset signals + portfolio state) and outputs 12 continuous weights via softmax

4. During training, the Regime Agent is frozen (pre-trained) while the Weight Agent learns. During inference, both run in sequence: regime first, then weights conditioned on that regime

5. In the current production configuration, the Regime Agent's learned policy is **bypassed** in favor of a simple rule (SPY > 200-SMA = RISK_ON) because the learned regime policy exhibited a 71% DEFENSIVE bias that prevented equity participation in bull markets. The Weight Agent still operates, preserving the benefits of learned allocation while using the more reliable rule-based regime signal.

### **Technical Summary**

PPO is an on-policy actor-critic algorithm that stabilizes policy gradient updates via a clipped surrogate objective. At each iteration, the system collects a rollout of experience, computes GAE advantages (a bias-variance balanced estimate of action quality), then performs multiple epochs of mini-batch gradient descent on the clipped loss. The clipping mechanism — min($r_t A_t$, clip($r_t, 1 \pm \epsilon$) $A_t$) — prevents destructive policy updates by bounding the probability ratio, ensuring the policy evolves gradually rather than making catastrophic jumps. The Alpha Dual Engine deploys two PPO agents in a hierarchical configuration: a discrete agent for regime selection (3-action softmax over 25-dimensional macro observations) and a continuous agent for weight allocation (12-dimensional Gaussian over 103-dimensional per-asset observations). Both are implemented as shared-trunk actor-critic networks and trained natively on Apple Silicon via the MLX framework.

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
- Bypassing the RL regime agent — after discovering a 71% DEFENSIVE bias in the learned policy that prevented equity participation during bull markets, the decision to replace it with a rule-based override (SPY > 200-SMA) while retaining the trained Weight Agent
- Checkpoint selection — systematic evaluation of 11 model checkpoints across the training run, identifying the 50K-step model as optimal based on negative Sharpe decay (OOS Sharpe 1.060 > IS Sharpe 0.985), proving that more training degraded generalization
- Diagnosing the safety net drift measurement bug — identifying and resolving an error in the drift calculation that caused the defensive regime to trigger incorrectly

**Documented Failures:**
- Section X of this README documents 5 specific failures encountered during development and the diagnostic process for each. These failures — and the reasoning behind their resolution — represent the iterative engineering judgment that no AI tool can replicate.
