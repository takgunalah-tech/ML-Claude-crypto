# System Context

## Purpose

This repository is a probabilistic crypto market screening and signal-generation system built around hourly data, volatility-aware labeling, XGBoost models, and explainable outputs.

The project is not a rule-based indicator bot. Technical indicators are used as feature inputs, while trade decisions are produced by trained models that estimate whether a target move is likely to hit take-profit before stop-loss.

Primary goals:

- Maintain an incremental, recoverable local market data store
- Build consistent features across training, validation, and live inference
- Train only on labels derived from volatility-aware TP/SL logic
- Accept models only if they are active, profitable, and stable out of sample
- Generate explainable LONG/SHORT signals with position sizing and Telegram delivery

## Project Identity

Internal project language in the repo refers to the system as `PROJECT SENTINEL`.

The architecture is split into three core layers plus an execution/automation layer:

- Layer A: data architecture / "Snowball DB Engine"
- Layer B: feature intelligence
- Layer C: strategy engine
- Execution layer: inference, signal state, notifications, retraining cadence

## Core System Contract

The most important invariant in this project is process parity.

Training, validation, and live inference must use the same:

- source data conventions
- integrity protocol behavior
- feature engineering logic
- threshold conventions
- TP/SL construction logic

Feature selection may differ by trained model, but the pipeline itself must stay aligned end to end.

## What The System Predicts

The model does not predict "price will go up" in the abstract.

It predicts an outcome of the form:

- LONG framing: will price hit `TP = +X% + k1 * ATR` before `SL = -Y% - k2 * ATR` within the label horizon?
- SHORT framing: symmetric inverse of the same logic

This outcome-based framing is the center of the system. If future changes break alignment between labeling, evaluation, and live trade construction, the project becomes invalid.

## Data Architecture

### Source Of Truth

Raw historical data lives in [`data/base`](./data/base) and is the long-term source of record.

- Crypto base OHLCV is stored under `data/base/crypto`
- Macro base OHLCV is stored under `data/base/macro`

These files are raw OHLCV only and should not be contaminated with derived indicators or integrity-filled values.

### Working Layer

The working layer lives under [`data/working`](./data/working).

It contains:

- hourly snapshot CSVs used as the recoverable working copy
- feature parquet files for incremental recompute
- macro and crypto snapshots used by the live/training pipeline

This layer is allowed to contain integrity-processed data and derived outputs.

### Integrity Protocol

Integrity handling is implemented in [`utils/data_utils.py`](./utils/data_utils.py).

Rules:

- Macro gaps are expected across weekends/holidays and may be forward-filled
- Crypto small gaps up to 3 hours may be linearly interpolated
- Larger crypto gaps are forward-filled in the working layer to preserve downstream indicator continuity
- Raw base CSVs remain unmodified by the integrity protocol

Implication:

- `data/base` preserves the record
- `data/working` preserves operability

## Market Universe

Universe configuration lives in [`config.py`](./config.py).

Current structure:

- Anchor tickers: `BTC/USDT:USDT`, `ETH/USDT:USDT`
- Altcoins: the rest of `CRYPTO_TICKERS`
- Macro context inputs: `SPX`, `QQQ`, `GLD`, `TLT`, `VIX`, `DXY`
- Primary timeframe: `1h`
- Primary exchange alias: Binance USD-M futures via `binanceusdm`

Important nuance:

- Initial long-history crypto download uses yfinance symbol mapping
- Incremental crypto updates use ccxt
- Macro download uses yfinance

Any symbol or naming change must preserve path consistency, feature store consistency, and model metadata consistency.

## Feature Intelligence

Feature logic lives in [`utils/features.py`](./utils/features.py).

### Model A: Macro Context

The macro model compresses multiple macro assets into `Macro_Risk_State` in `[-1, 1]`.

It uses:

- short and medium returns
- rolling volatility
- rolling z-score
- PCA-based compression after alignment/scaling

This is meant to represent risk-on vs risk-off context, not a direct trade signal.

### Model B: Anchor Intelligence

BTC and ETH are treated as anchor assets and transformed into contextual market-state features such as:

- regime
- volatility state
- breakout probability
- continuation probability
- directional bias

These are injected into altcoin models as context.

### Model C: Altcoin Feature Matrix

Altcoin features include:

- standard TA indicators
- derived momentum and volatility features
- pivot and VWAP-relative features
- time/session features
- injected BTC/ETH anchor features
- injected macro risk state

Feature store behavior:

- first run can compute full history
- later runs should prefer incremental recompute with warm-up overlap
- canonical feature parquet paths are defined centrally and must not be reimplemented ad hoc

## Labeling Logic

Label generation lives in [`utils/labeling.py`](./utils/labeling.py).

System assumptions:

- label horizon is `48` candles by default
- TP/SL grids and ATR multipliers are configurable in `config.py`
- ambiguous samples where neither TP nor SL hits inside the horizon are excluded from training
- vectorized NumPy labeling is the default fast path

The label is only valid if it uses the same volatility-aware construction that live TP/SL uses later.

## Modeling And Validation

Model training/evaluation utilities live in [`utils/model_utils.py`](./utils/model_utils.py).

Current modeling approach:

- XGBoost binary classifier
- lightweight estimator count for grid search ranking
- larger estimator count for final production model

The repo’s validity gates are critical and should be treated as policy, not convenience:

- minimum trade count: `MIN_TRADE_COUNT = 20`
- minimum test1 PF: `MIN_PF_TEST1 = 1.20`
- stability gate: `PF(test2) >= PF(test1) * 0.70`

Interpretation:

- high PF with tiny activity is not acceptable
- models must remain useful after the internal validation period
- the system prefers active and repeatable models over lucky backtests

## Signal Logic

Signal state and deduplication live in [`utils/signal_manager.py`](./utils/signal_manager.py).

Telegram delivery lives in [`utils/telegram_utils.py`](./utils/telegram_utils.py).

Operational behavior:

- LONG threshold defaults to `p(win) >= 0.60`
- SHORT threshold defaults to `p(win) <= 0.40`
- the middle zone is intentionally ignored
- signal state is persisted in `state/signals_state.json`
- repeat suppression and archival are part of the contract
- quantity is derived from `MAX_LOSS_USDT` and stop distance

The system is designed to avoid spammy duplicate alerts and to preserve a traceable lifecycle for each coin’s active signal.

## Automation Cadence

Cadence values live in [`config.py`](./config.py).

Current defaults:

- market update cadence: every `60` minutes
- retrain cadence: every `72` hours
- morning report hour: `7` UTC

Expected high-level loop:

1. Update raw/base data incrementally
2. Build or extend working snapshots and feature store
3. Load existing valid models or retrain when required
4. Run inference on latest feature row
5. Apply thresholds and state checks
6. Send Telegram signals/reports if configured

## Explainability

Explainability is part of the product, not an optional extra.

The system should be able to explain a selected signal in terms of top drivers, using:

- SHAP when available
- XGBoost feature importance fallback otherwise

Expected user-facing explanation style:

- coin
- direction
- `P(win)`
- TP / SL / qty
- key drivers

## Important Files

- [`README.md`](./README.md): high-level vision and design intent
- [`config.py`](./config.py): market universe, thresholds, cadence, paths
- [`utils/data_utils.py`](./utils/data_utils.py): data update pipeline and integrity protocol
- [`utils/features.py`](./utils/features.py): macro, anchor, and altcoin feature generation
- [`utils/labeling.py`](./utils/labeling.py): volatility-aware label generation and grid search
- [`utils/model_utils.py`](./utils/model_utils.py): training, PF evaluation, validity gates, model persistence
- [`utils/signal_manager.py`](./utils/signal_manager.py): signal lifecycle state
- [`utils/telegram_utils.py`](./utils/telegram_utils.py): notification delivery
- [`state/signals_state.json`](./state/signals_state.json): persisted signal state

## Design Principles For Future Changes

When changing this project, prefer decisions that preserve these rules:

- Never write derived features back into the raw base layer
- Never change labeling math without also changing evaluation and live TP/SL construction
- Never accept models solely because they have impressive PF with too few trades
- Never introduce separate feature pipelines for training and live inference
- Never construct feature parquet/model paths in multiple inconsistent ways
- Prefer incremental computation over full recomputation when the historical section is immutable
- Keep anchor features available before altcoin feature generation
- Preserve explainability and stateful signal lifecycle behavior

## Risks And Known Fragility Areas

Areas that deserve extra care:

- ticker naming consistency across yfinance, ccxt, file paths, and model metadata
- timezone normalization and timestamp alignment
- large filled gaps masking upstream data issues if logs are ignored
- stale or mismatched feature parquet paths
- silently breaking parity between notebook experiments and production code
- treating the current repo vision in `README.md` as implemented everywhere when some pieces may still be evolving

## Recommended Use Of This Document

Use this file as the shared context for:

- onboarding new contributors
- grounding AI coding assistants
- reviewing architectural changes
- checking whether an experiment still respects the system contract

If there is a conflict between ad hoc notebook experimentation and this document, preserve the production pipeline invariants unless the project owner intentionally updates the contract.
