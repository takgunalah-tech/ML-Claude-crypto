# ML-Claude-crypto
Crypto market screener with ML develop by claudecode


🧠 PROJECT SENTINEL — v5.0 (Probabilistic Integrity Architecture)
🔷 1. Layer A — Data Architecture (“Snowball DB Engine”)
Core Philosophy
Local CSV = persistent time-series database
Data is incremental, validated, and never recomputed from scratch
✅ Pipeline
Load Base → Create Working DF → Fetch Update → Validate → Merge → Save → Serve
Structure
1. Base Layer (Storage)
CSV per ticker
Raw + integrity-processed data
NEVER used directly for ML
2. Working Layer (Computation)
Loaded from base
Used for:
feature engineering
labeling
training
🔒 Integrity Protocol
Macro Assets:
Forward-fill gaps (weekends valid)
Crypto Assets:
Fill small gaps (≤ few hours)
Preserve large gaps (signal structural risk)
🔑 Principle

Storage is immutable history
Working DF is the experimental layer

🔷 2. Layer B — Feature Intelligence System

This layer defines market state, not signals.

🔹 Model A — Macro Context

Output:

Macro_Risk_State ∈ [-1, 1]

Represents:

Risk-on / risk-off environment
🔹 Model B — Anchor Intelligence (BTC & ETH)
🧠 Objective:

Model market structure and transition probability

Feature Groups
1. Regime Detection
Trend vs Range
Trend Strength
2. Volatility State
Compression vs Expansion
Volatility Shift
3. Breakout / Transition Probability

Core idea:

P(range → trend)
P(trend → continuation)
P(trend → exhaustion)
4. Directional Bias
Bullish / Bearish / Neutral (continuous score)
✅ Output (per timestamp)
BTC_Regime
BTC_Volatility_State
BTC_Breakout_Prob
BTC_Continuation_Prob
BTC_Direction_Bias

ETH_* (same structure)
🔹 Model C — Target Layer (Altcoins)
Philosophy:

TA = feature space, not decision rules

Feature Categories
1. Standard Indicators (broad coverage)
RSI (multi-length)
MACD
Bollinger Bands
EMA/SMA spreads
ATR
Volume metrics
2. Derived Features
Momentum shifts
Volatility expansion
Mean reversion strength
3. Cross-Asset Injection

From Model B:

BTC/ETH regime
Volatility state
Breakout probabilities
🔑 Principle

Model C provides a high-dimensional feature matrix
ML decides what matters

🔷 3. Layer C — Strategy Engine (Probabilistic ML Core)
🎯 Core Target Definition

We DO NOT define trades directly.

We define an outcome question:

Will price hit +X% before -Y% ?
🔥 Volatility-Aware Labeling (Updated)

This is your key refinement.

Instead of:
Fixed %
OR pure ATR scaling
✅ Correct Formulation:
TP = +X%  + (k₁ × ATR)
SL = -Y%  - (k₂ × ATR)

Where:

X%, Y% = base expectation
ATR = volatility component
k₁, k₂ = sensitivity factors
🧠 Interpretation
% = structural move expectation
ATR = volatility condition filter

👉 This creates:

adaptive + contextual outcome boundaries

🔁 Label Construction

For each timestamp:

y = 1 → TP hit first
y = 0 → SL hit first
🔹 Training Process (3-Phase Confidence System)
Phase 0 — Full Historical Context
T0 = oldest → -85

Used for:

feature stability
regime diversity
Phase 1 — Training
Train on T0
Generate features
Generate labels
Train models / evaluate feature combinations
Phase 2 — Simulation (Test 1)
Window: -85 → -31

Purpose:

Internal validation
Filter weak feature combinations
Phase 3 — Final Validation (Test 2)
Window: -30 → -1

Purpose:

True out-of-sample validation
Confidence scoring
🔹 Model Output (Updated)

Per coin, model produces:

P(win | features)
Trade_Count
Profit_Factor
Feature_Importance / Contribution
✅ Selection Criteria

A model is valid ONLY if:

1. Minimum Activity
Trade_Count ≥ threshold

👉 Prevents:

Lucky models
Overfitting with few trades
2. Profitability
PF > 1.05 (or configurable)
3. Stability
PF(Test2) ≥ 70% of PF(Test1)
4. Feature Consistency
Feature importance stable across phases
No random spikes
🔑 Principle

A good model must be:

Active
Stable
Explainable
Repeatable
🔷 4. Execution Layer — Probabilistic Trade Engine
Live Pipeline
Load latest data (Integrity Protocol)
Build feature vector (same pipeline as training)
Run model inference
Signal Decision
If P(win) ≥ threshold AND conditions valid → Trade
Additional Constraints
Volume filter (liquidity confirmation)
Market condition alignment (optional)
🔥 Trade Construction (Updated)

From labeling logic:

TP = Entry + [X% + (k₁ × ATR)]
SL = Entry - [Y% + (k₂ × ATR)]
🧠 Key Difference
SL/TP are NOT arbitrary
They are derived from the trained outcome model
🔷 5. Anti-Cheating Mechanism (Critical Constraint)

You explicitly defined this — very important.

❌ What is NOT allowed:
Model with 1–2 trades showing high PF
“Perfect” but inactive strategy
✅ Enforcement

Minimum thresholds:

Trade_Count ≥ N (e.g., 20–50 depending on timeframe)
Interpretation

Probability must be supported by frequency

🔑 Principle

No activity = no validity

🔷 6. Feature Parity & Data Consistency
Requirement

The following MUST be identical:

Training Pipeline
Validation Pipeline
Live Pipeline
Includes:
Data source (Snowball only)
Feature engineering
Label logic (for consistency in evaluation)
❗ Important Clarification

Parity does NOT mean:

Same selected features

Parity DOES mean:

Same process
🔷 7. System Output (Explainable AI Layer)

Each selected model must provide:

Coin: SOL
P(win): 0.64
Trades: 48
PF: 1.32

Key Drivers:
- BTC breakout probability ↑
- RSI mean reversion zone
- Volatility expansion detected
Purpose
Transparency
Debugging
Trust in system decisions
🔷 8. Automation Layer
Hourly:
Update data
Recompute features
Run inference
Send signals (optional Telegram)
Every X hours:
Retrain models
Re-evaluate feature combinations
Update valid models
🔷 9. System Nature (Final Definition)
Multi-layer probabilistic trading AI
with volatility-aware outcome modeling
and statistical validation constraints
NOT:
Indicator bot
Rule-based system
Static strategy
IS:
Adaptive
Data-driven
Self-validating
Frequency-aware
✅ Final Alignment Summary

Your updated system now correctly enforces:

✔ Volatility-integrated outcome definition
✔ Multi-phase validation (true confidence filter)
✔ Feature-driven ML (not hardcoded TA logic)
✔ Anchor layer as probabilistic market state model
✔ Trade frequency as a validity constraint
✔ Explainable outputs (not black box)
