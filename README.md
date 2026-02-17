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

This software is for educational and research purposes only and was built as a personal project by a student, PEHCAUX, studying at National Chengchi University (NCCU). It is not intended to be a source of financial advice, and the author is not a registered financial advisor. The algorithms, simulations, and optimization techniques implemented herein—including Consensus Machine Learning, Shannon Entropy, and Geometric Brownian Motion—are demonstrations of theoretical concepts and should not be construed as a recommendation to buy, sell, or hold any specific security or asset class.

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

**BY USING THIS SOFTWARE, YOU AGREE TO ASSUME ALL RISKS ASSOCIATED WITH YOUR INVESTMENT DECISIONS AND HARDWARE USAGE, RELEASING THE AUTHOR (PEHCAUX) FROM ANY LIABILITY REGARDING YOUR FINANCIAL OUTCOMES OR SYSTEM INTEGRITY.**

====================================================================================
# **Alpha Dual Engine v100.0.0: The Framework**

This documentation details the architecture of Alpha Dual Engine v100.0.0, a high-performance quantitative trading system refactored to prioritize "Acceleration Alpha" over passive diversification and engineered to navigate complex market regimes through a synthesis of machine learning, macro-economic veto guards, and entropy-weighted optimization. Established by PEHCAUX, a Public Finance major at National Chengchi University, the system represents a Bifurcated Logic that applies distinct mathematical strategies to Equities (Growth Flow) and Cryptocurrency (Store of Value Cycles). Unlike traditional mean-variance models that treat all assets identically, v100.0.0 utilizes a Bifurcated Logic Engine that applies distinct mathematical strategies to Equities (Growth Flow) and Cryptocurrency (Store of Value Cycles).

**I. Configuration & Constants (StrategyConfig)**

The StrategyConfig class serves as the immutable constitution of the Alpha Dual Engine v100.0.0. Defined as a Python dataclass, it centralizes every hard-coded rule, risk limit, and execution parameter into a single, modifiable control panel. This design ensures that the strategy's logic remains separated from its parameters, allowing for rapid sensitivity testing without risking code breakage.

This section acts as the primary filter for all downstream logic. If a trade or allocation violates these parameters, it is rejected before it ever reaches the optimization engine.

**1. Risk & Portfolio Constraints**

These parameters define the "shape" of the portfolio and the limits of its aggression.

a. target_volatility (0.25 / 25%)

* Function: This sets the target annualized standard deviation for the portfolio optimizer.

* Logic: Unlike conservative funds that target 10-12% volatility, this engine is tuned for 0.25, explicitly telling the math to accept significant daily variance in exchange for higher compound growth (CAGR). It prevents the optimizer from "diluting" high-momentum assets like SMH simply because they are volatile.
b. prob_ema_span (10-day): Probabilities are smoothed using a 10-day Exponential Moving Average (EMA) to filter out daily market noise and prevent "regime flickering," which can lead to excessive and costly trading.

b. max_single_weight (0.30 / 30%)

* Function: The "Anti-Blowup" Cap. No single asset can exceed 30% of the total portfolio value.

* Logic: This forces the "Cubed Momentum" engine to select a basket of winners rather than betting the farm on one. Even if SMH (Semiconductors) has a perfect momentum score, the system must find at least two or three other assets (e.g., TAN, XBI) to fill the remaining allocation, ensuring a mathematical minimum diversity of $\sim 3.33$ assets ($1 / 0.30$).

c. gold_cap_risk_on (0.01 / 1%)

* Function: The "Drag eliminator."

* Logic: During a RISK_ON regime, the strategy is strictly prohibited from allocating more than 1% to Gold (GLD). This prevents the optimizer from "hiding" in safe-haven assets during a bull market, ensuring 99% of capital is deployed into productive, high-beta assets.

d. entropy_lambda (0.02)

* Function: The penalty for concentration.

* Logic: A low lambda value (0.02) tells the optimizer that we prefer concentration over diversification. It allows the weights to cluster near the 30% maximums rather than being flattened out equally across all assets.

**2. Execution & Cost Control (The "Fee Guillotine")**

a. Transaction Cost Calculation & Formula

$$\text{Cost} = V_{\text{portfolio}} \times \sum_{i} \left| w_i^{\text{new}} - w_i^{\text{old}} \right| \times \frac{\text{Cost}_i^{\text{bps}}}{10{,}000}$$

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

In v100.0.0, this class has been radically re-engineered to support the Bifurcated Asset Logic, treating Crypto and Equities as fundamentally different data species.

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

* The "Soft Filter": By raising the momentum ratio to the third power, the math creates an exponential gap between "Good" and "Great."

Scenario A: Asset is 2% above trend → $1.02^3 \approx 1.06$

Scenario B: Asset is 10% above trend → $1.10^3 \approx 1.33$

* Result: The optimizer sees Scenario B as 5x better than Scenario A, naturally forcing capital into the fastest-moving sector without needing manual "if/else" exclusions.

b. For Crypto: The "RSI Rotation" Metric

* Formula: 14-Day Relative Strength Index (RSI).

* Logic: Crypto doesn't follow smooth trends like stocks; it moves in manic bursts. RSI measures the internal velocity of price changes.

* Result: This allows the "Active HODL" engine to swap 100% of its weight between Bitcoin and Ethereum based on which one is currently experiencing a stronger "pump."

**3. The 7-Factor Feature Synthesis**

This section builds the input vectors for the Machine Learning "Brain." It distills the complex market state into 7 digestible numbers:

a. realized_vol: Realized volatility derived from the VIX (Market Fear).

b. trend_score: Distance of SPY from its 200-day SMA (The "Bull/Bear" Line).

c. momentum_21d: Short-term velocity of the market (Breakout detection).

d. vol_momentum: Rate of change in the VIX (Is fear rising or falling?).

e. qqq_vs_spy: Relative performance of QQQ vs SPY over 63 days (Tech leadership signal).

f. tlt_momentum: Momentum of TLT (Detects Interest Rate shocks).

g. equity_risk_premium: A valuation metric comparing SPY's mean-reversion ratio to the risk-free rate.

**4. Temporal Integrity (The "Time Machine")**

The get_aligned_data method ensures the backtest is honest.

a. The Lag: It strictly uses data up to the close of T-1 (yesterday) to make decisions for T (today).

b. The NaN Handler: It forward-fills missing data (common in Crypto vs. Stock weekends) to prevent the optimizer from crashing due to misalignment.


**III. Regime Intelligence (AdaptiveRegimeClassifier)**

The AdaptiveRegimeClassifier is the central nervous system of the strategy, responsible for the binary decision that determines portfolio survival: "Risk-On" (Aggressive Growth) or "Defensive" (Capital Preservation).

In v100.0.0, this system was upgraded from a pure Machine Learning model to a Hybrid Override Architecture. This solves the "Black Box" problem by enforcing a strict hierarchy: Macro Logic overrules ML Probabilities.

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

* Strength: Excellent at catching "V-shaped" recoveries where price action snaps back quickly despite high fear levels.

b. Model Beta: Decision Tree (The Skeptic)

* Role: Filters out noise and false breakouts.

* Configuration: Shallow tree (max_depth=2) with high regularization (min_samples_leaf=200).

* Strength: Prevents the XGBoost model from overreacting to short-term noise.

c. The Consensus Rule:

To trigger a RISK_ON signal in the absence of a Macro Override, BOTH models must agree with high conviction (Probability > 0.55). If they disagree, the system defaults to safety.

**3. The "Anxiety" & "Panic" States**

The classifier doesn't just look at price; it monitors market psychology via the VIX (Volatility Index).
To ensure the system remains career-relevant in 2026 and beyond, the classifier uses an Adaptive Walk-Forward Training loop.

a. Anxiety State (VIX > 18):

The market is choppy. The system raises the required ML probability threshold (e.g., from 55% to 75%). It demands "extraordinary evidence" to take risks in a nervous environment.

b. Panic State (VIX > 35):

The "Circuit Breaker." If volatility explodes, the system forces a DEFENSIVE regime immediately, overriding everything else. This protects the portfolio during black swan events (e.g., COVID Crash 2020).

**4. Explainability via SHAP Values**

Because black-box models are unacceptable in institutional advisory, the classifier integrates SHAP (SHapley Additive exPlanations). This provides a detailed "Feature Importance" report, allowing the user to explain exactly why a decision was made. For instance, the user can point to a SHAP summary plot to show that TLT Momentum was the primary factor that triggered a defensive exit during the bond market crash of 2022.

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

a. Minimizing Variance — $\mathbf{w}^\top \Sigma \mathbf{w}$: It tries to keep the portfolio stable. However, with target_volatility set to 0.25, the "leash" is loose, allowing for high-beta plays.

b. Maximizing Momentum — $-\mathbf{w} \cdot \mathbf{M}$: This is the alpha driver. It pushes weights toward assets with the highest "Cubed Momentum" scores (SMH, TAN).

c. Maximizing Entropy — $-H(\mathbf{w})$: The entropy_lambda (set to 0.02) is the "Anti-Concentration" term. It prevents the math from simply putting 100% into the single best stock. It forces a slight spread across the top 3-4 winners.

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

This is the critical logic block implemented in v100.0.0 to solve the "Churn Problem" (where fees were eating 4% of the portfolio).

a. Logic: Inside the main loop, before executing any rebalance, the engine calculates the Proposed Turnover:

$$\text{Turnover} = \sum_{i} \left| w_i^{\text{new}} - w_i^{\text{old}} \right|$$

b. The Rule:

* If $\text{Turnover} < 0.12$ (12%): THE TRADE IS REJECTED.

* Effect: The engine effectively says, "This change is too small to be worth the fees. Do nothing." It carries forward the existing portfolio weights exactly as they were.

c. The Safety Valve:

If the Regime does change (e.g., Bullish to Defensive): The Gate is lifted. Safety takes priority over fees, and the trade is executed immediately to de-risk the portfolio.

**3. Cost Modeling & Friction**

The engine uses a tiered, per-asset transaction cost model that reflects real-world trading friction. Each asset has its own cost in basis points (see the fee table in Section I.2), ranging from 1 bps for the most liquid bond ETFs to 30 bps for cryptocurrency. The cost for each rebalance is calculated as:

$$\text{Cost} = V_{\text{portfolio}} \times \sum_{i} \left| w_i^{\text{new}} - w_i^{\text{old}} \right| \times \frac{\text{Cost}_i^{\text{bps}}}{10{,}000}$$

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

a. CAGR (Compound Annual Growth Rate): The geometric mean return. (Target: >20%).

b. Sharpe Ratio: Return per unit of risk. (Target: ~1.0).

c. Max Drawdown: The deepest peak-to-trough decline. This verifies the "Defensive" logic works during crashes.

**3. Constraint Verification**

The script prints a final "Constraint Status" block to prove the optimizer followed the rules.

a. Growth Anchor Check: Did the portfolio maintain >40% in high-growth assets?

b. Gold Cap Check: Did it strictly keep Gold under 1%?

c. Effective N Check: Did it maintain the required diversity score (>3.0)?

**4. Visualizing the Alpha Generation**

To make the data accessible for career-level presentations, the engine generates diagnostic plots:

a. Equity Curve: Compares the strategy against the SPY benchmark, with color-coded regions indicating different regime states.

b. Allocation Stack: A historical visualization of how the portfolio rotated between Growth Anchors, Bonds, and Gold across different market cycles.

c. Regime Analysis: A multi-panel view aligning SPY price action with ML probabilities, revealing how the regime detection protected the portfolio during crashes.


**VII. Tail-Risk Stress Testing (MonteCarloSimulator)**

The Monte Carlo Simulator serves as the final, rigorous validation layer of the Alpha Dual Engine framework. It transitions the analysis from historical backtesting (what did happen) to stochastic modeling (what could happen), providing a probabilistic assessment of tail risk and expected future performance.

**1. Stochastic Engine: Geometric Brownian Motion (GBM)**

a. Mathematical Foundation: The simulator utilizes Geometric Brownian Motion, the industry-standard model for projecting asset price paths. This model assumes that the logarithm of the asset price follows a Brownian motion with drift and diffusion components.

b. Drift Component ($\mu$): The engine calculates the annualized drift based on the portfolio's optimized weighted returns from the backtest, adjusted for volatility drag:

$$\mu_{\text{adj}} = \mu - \frac{1}{2}\sigma^2$$

c. Diffusion Component ($\sigma$): Volatility is modeled as a random walk, scaled by the annualized standard deviation of the portfolio and a standard normal random variable ($Z$):

$$S_{t+1} = S_t \exp\left(\mu_{\text{adj}} \Delta t + \sigma \sqrt{\Delta t} Z\right)$$

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

**IX. Conclusion**

The Alpha Dual Engine v100.0.0 represents a modern evolution in finance and asset management—shifting from reactive rebalancing to proactive, regime-aware navigation. It is a robust asset for firms specializing in infrastructure advisory and hearty wealth preservation.

====================================================================================
# **Development Methodology**

**The core financial strategy was conceptualized and architected by the author.**

This project was built using an AI-Accelerated Workflow.

Large Language Models (Gemini, Claude Opus 4.6) were utilized to accelerate syntax generation and boilerplate implementation, allowing the focus to remain on quantitative logic, parameter tuning, and risk management validation.
