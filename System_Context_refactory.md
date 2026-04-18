System Context — PROJECT SENTINEL (v2)
0. System Identity

PROJECT SENTINEL is a probabilistic crypto market screening and signal system.

It is a model-driven decision system, not an indicator-based strategy.

Core function:

Estimate whether a volatility-adjusted TP will be reached before SL within a fixed horizon.

1. Core Design Philosophy
1.1 Outcome-Based Prediction (Non-Negotiable)

The system predicts:

LONG: TP hit before SL within horizon
SHORT: inverse

NOT:

price direction
returns regression
indicator signals

This framing defines:

labeling
evaluation
signal construction

Breaking this alignment invalidates the system.

1.2 Process Parity (Hard Constraint)

Training, validation, and inference MUST share:

identical feature definitions
identical labeling logic
identical TP/SL construction
identical thresholds
Enforcement (New)

All artifacts must carry:

pipeline_signature = hash(
    feature_logic +
    labeling_logic +
    config +
    data_schema_version
)

Validation rule:

if model.pipeline_signature != runtime.pipeline_signature:
    BLOCK inference
1.3 Reproducibility & Lineage

Every model must be reproducible via:

dataset version
feature version(s)
labeling config
training config

All must be stored with model metadata.

2. System Architecture
2.1 Layer Overview
Layer A: Data Layer (Base + Working)
Layer B: Feature Layer (Feature Store)
Layer C: Strategy Layer (Model + Validation)
Layer D: Execution Layer (Inference + Signals)
Layer E: Governance Layer (NEW)
3. Data Layer
3.1 Source of Truth (data/base)
Raw OHLCV only
Immutable
No derived values EVER
3.2 Working Layer (data/working)
Integrity-processed data
Feature-ready snapshots
Intermediate artifacts
3.3 Integrity Protocol

Rules unchanged, but now:

NEW: Data Quality Score

Each asset/time window must compute:

gap_ratio
fill_ratio
freshness_delay
volatility_jump_score

Output:

data_quality_score ∈ [0, 1]
Enforcement
If score < threshold:
block feature generation OR
flag signals as LOW_CONFIDENCE
4. Feature Layer (Feature Store Contract)
4.1 Feature Store Principle

Features are first-class versioned assets, not ad hoc calculations.

Motivation:

Prevent training-serving skew
Ensure reproducibility
Enable reuse
4.2 Feature Definition Contract

Each feature must define:

feature_name
version_id
input_sources
transformation_logic
output_schema
4.3 Feature Versioning

Use hybrid:

semantic version (MAJOR.MINOR.PATCH)
immutable hash ID

Any logic change → new version

4.4 Feature Groups
Macro features
Anchor features (BTC/ETH)
Altcoin features
4.5 Point-in-Time Correctness

Training must ONLY use features available at timestamp T.

No future leakage allowed.

5. Labeling Layer
5.1 Label Definition

Unchanged core:

horizon: default 48
TP/SL: ATR-adjusted
5.2 Label Validity Contract

Labels MUST include:

tp_logic_version
sl_logic_version
horizon
atr_config
5.3 Sample Filtering
ambiguous samples removed
MUST log discard ratio
6. Strategy Layer (Model + Validation)
6.1 Model Type
XGBoost classifier (default)
6.2 Model Metadata (MANDATORY)

Each model must store:

model_id
pipeline_signature
feature_versions
label_config
training_period
validation_period
metrics
6.3 Validation Framework (Upgraded)
Required Metrics
Profit Factor (PF)
Trade Count
Max Drawdown (NEW)
Expectancy (NEW)
Win Rate
Loss Streak (NEW)
Acceptance Gates
MIN_TRADE_COUNT
MIN_PF_TEST1
STABILITY_RATIO
MAX_DRAWDOWN (NEW)
6.4 Regime Segmentation (NEW)

Models may be segmented by:

volatility regime
macro risk state
trend regime

Inference must select model based on current regime.

7. Execution Layer
7.1 Inference Rules
enforce pipeline signature match
enforce data quality threshold
enforce feature availability
7.2 Signal Generation

Thresholds:

LONG: p ≥ 0.60
SHORT: p ≤ 0.40
7.3 Signal Lifecycle (Upgraded)

Each signal includes:

timestamp
expiry_time
decay_function
state
NEW: Signal Decay
p_adj = p * exp(-lambda * age)
Expiry
signals expire after TTL (e.g. 6–12h)
7.4 Position Sizing

Unchanged:

based on MAX_LOSS_USDT and SL distance
8. Governance Layer (NEW)
8.1 Model Registry

Tracks:

model versions
feature dependencies
pipeline signature
deployment status
8.2 Rollout Policy

Optional but recommended:

shadow testing
canary deployment
rollback support
8.3 Drift Monitoring (NEW)

Monitor:

feature distribution shift
prediction distribution shift
realized vs expected PF
8.4 Audit Trail

System must log:

model decisions
signal triggers
rejected signals
data quality issues
9. Automation Loop

Unchanged flow, but with gates:

Update data
Compute data quality
Build features (versioned)
Train or load model
Validate (strict gates)
Run inference (with checks)
Apply decay + state logic
Send signals
10. Hard Constraints (Updated)

Never:

break pipeline signature consistency
use unversioned features
train on different logic than inference
accept models without distribution metrics
ignore data quality issues
11. Known Risk Areas (Expanded)
silent feature version mismatch
data quality masking via fills
regime shift degrading models
stale signals without decay
notebook vs production divergence