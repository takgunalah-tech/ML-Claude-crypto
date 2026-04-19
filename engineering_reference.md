# Engineering Reference

## Purpose

This file is a practical build-and-debug reference for the current PROJECT SENTINEL codebase.

It documents:

- how data is handled today
- which files own which responsibilities
- what bugs were found during the recent refactor
- how those bugs were debugged and fixed
- what to preserve in a future rebuild
- what to improve next time

This is meant to reduce repeat mistakes when building the next system version.
<!--  this part to be ignore, as new system will be use different concept
## Current System Shape

The current repo is a layered ML trading workflow with a notebook-driven runner and utility modules underneath it.

High-level flow:

1. Update raw/base market data
2. Build working snapshots
3. Build features
4. Train / validate models
5. Save valid models
6. Run live inference
7. Manage signal lifecycle
8. Send Telegram notifications

Primary control points today:

- [notebooks/00_sentinel.ipynb](/c:/tmp/py/Git-Claude/ML-Claude-crypto/notebooks/00_sentinel.ipynb)
- [config.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/config.py)
- [utils/data_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_utils.py)
- [utils/features.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/features.py)
- [utils/labeling.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/labeling.py)
- [utils/model_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/model_utils.py)
- [utils/signal_manager.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/signal_manager.py)
- [utils/telegram_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/telegram_utils.py)

## File Ownership Map

### Configuration

- [config.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/config.py)

Owns:

- ticker universe
- data paths
- train/test split settings
- labeling grids
- model validity thresholds
- validation thresholds
- live signal thresholds
- signal TTL / decay
- scheduler cadence
- governance toggles

Important current settings:

- `MIN_TRADE_COUNT = 20`
- `MIN_TRADE_COUNT_TEST2 = 5`
- `MIN_PF_TEST1 = 1.05`
- `PF_STABILITY_RATIO = 0.65`
- `MAX_DRAWDOWN = 0.40`
- `VALIDATION_LONG_THRESHOLD = 0.60`
- `VALIDATION_SHORT_THRESHOLD = 0.40`
- `LONG_THRESHOLD = 0.58`
- `SHORT_THRESHOLD = 0.42`

### Data Layer

- [utils/data_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_utils.py)

Owns:

- base CSV load/save
- snapshot CSV load/save
- crypto incremental fetch via ccxt
- macro fetch via yfinance
- integrity protocol
- raw gap repair
- update orchestration per ticker

### Feature Layer

- [utils/features.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/features.py)

Owns:

- macro risk state
- BTC/ETH anchor features
- altcoin feature matrices
- feature parquet persistence
- incremental feature recompute

### Labeling Layer

- [utils/labeling.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/labeling.py)

Owns:

- volatility-aware TP/SL labeling
- NumPy vectorized label generation
- label parameter grid search

### Model / Validation Layer

- [utils/model_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/model_utils.py)

Owns:

- XGBoost training
- PF computation
- validation metrics
- validity gates
- model save/load
- backward-compatible model path lookup

### Signal Layer

- [utils/signal_manager.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/signal_manager.py)

Owns:

- signal deduplication
- signal registration
- active/archived signal state
- expiry handling
- probability decay
- repeat suppression

### Notification Layer

- [utils/telegram_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/telegram_utils.py)

Owns:

- Telegram send/retry behavior
- signal message formatting
- morning report formatting

### Governance / Metadata Layer

- [utils/governance_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/governance_utils.py)
- [utils/data_quality.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_quality.py)

Owns:

- pipeline signature generation
- model metadata shaping
- registry append logs
- audit log events
- data quality scoring

## How Data Is Handled Today -->

### Source Of Truth

Base data lives under:

- [data/base](/c:/tmp/py/Git-Claude/ML-Claude-crypto/data/base)

Rules:

- raw OHLCV only
- no feature columns
- no integrity-filled values
- intended to be the source of record

### Working Layer

Working data lives under:

- [data/working](/c:/tmp/py/Git-Claude/ML-Claude-crypto/data/working)

Rules:

- may contain integrity-filled values
- may contain feature parquet files
- used for downstream computation and recovery

### Current Integrity Behavior

Implemented in [utils/data_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_utils.py:521)

Macro:

- forward-fill expected market-hour gaps

Crypto:

- interpolate small gaps up to 3 hours
- forward-fill larger gaps in the working layer

Important nuance:

- forward-fill is a continuity fallback, not a true data repair
- raw gap repair is now attempted before working-layer smoothing

### Current Gap Repair Behavior

Implemented in:

- [utils/data_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_utils.py:444)

Behavior:

- detect internal missing ranges in raw crypto data
- fetch bounded historical ranges from ccxt
- merge repaired candles into the raw base layer
- only if repair fails does the working layer fall back to forward-fill

This was added because real historical crypto gaps were found in base CSVs.

## Recent Bugs Found And Fixed

### 1. Real Historical Crypto Gaps In Base Data

Symptom:

- warning like `Large data gap: ... missing candles -> forward-filled`
- some crypto base CSVs had true internal missing ranges

What was found:

- multiple crypto base files had historical holes
- one repeated pattern was around November 2025

Fix:

- added `_find_missing_ranges()`
- added `fetch_crypto_range_ccxt()`
- added `repair_base_gaps()`
- wired repair into `update_ticker()`

Relevant code:

- [utils/data_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_utils.py:136)
- [utils/data_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_utils.py:329)
- [utils/data_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_utils.py:444)
- [repair_data_gaps.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/repair_data_gaps.py:1)

Lesson:

- do not trust working-layer forward-fill as a substitute for raw data integrity

### 2. Windows Filename Bug For Futures Tickers

Symptom:

- model save logged success
- expected `_meta.json` file did not update
- an empty file like `ZEC_USDT` appeared

Root cause:

- model path sanitization handled `/` but not `:`
- ticker `ZEC/USDT:USDT` is unsafe as a Windows-derived filename if `:` is preserved

Fix:

- sanitize `:` in model/meta filenames
- add backward-compatible legacy model path lookup

Relevant code:

- [utils/model_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/model_utils.py:287)
- [utils/model_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/model_utils.py:301)

Lesson:

- all filename builders must share one canonical sanitizer
- Windows path behavior must be treated as a design input, not an afterthought

### 3. Legacy Signals Never Expired

Symptom:

- an old ZEC signal stayed open even after the TTL logic had been added

Root cause:

- older signals in [state/signals_state.json](/c:/tmp/py/Git-Claude/ML-Claude-crypto/state/signals_state.json) had no `expiry_at`
- new expiry logic initially only checked explicit `expiry_at`

Fix:

- infer expiry from `first_signal_at + ttl_hours` when `expiry_at` is missing

Relevant code:

- [utils/signal_manager.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/signal_manager.py:71)
- [utils/signal_manager.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/signal_manager.py:143)

Lesson:

- state schema migrations must be backward-compatible
- new lifecycle fields need legacy fallbacks

### 4. File Lock Contention During Writes

Symptom:

- `PermissionError` while saving CSVs or state JSON
- usually happened while notebooks or scheduler processes still held the file open

Fix:

- add retry loops around atomic save moves for:
  - base CSVs
  - snapshot CSVs
  - signal state file

Relevant code:

- [utils/data_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_utils.py:71)
- [utils/data_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_utils.py:95)
- [utils/signal_manager.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/signal_manager.py:49)

Lesson:

- notebook-driven systems often keep file handles alive longer than expected
- retry-safe writes are mandatory on Windows

### 5. Notebook Drift In Short Signal Logic

Symptom:

- one notebook still used an outdated short-threshold expression

Fix:

- aligned [notebooks/02_strategy_and_live.ipynb](/c:/tmp/py/Git-Claude/ML-Claude-crypto/notebooks/02_strategy_and_live.ipynb) with current `SHORT_THRESHOLD` logic

Lesson:

- notebooks easily drift from utilities
- reusable execution logic should move into shared modules where possible

### 6. Validation Was Too Permissive On Tiny Test2 Sample Sizes

Symptom:

- ZEC passed validation with only `1` trade in `Test2`

Root cause:

- minimum trade count was enforced on `Test1` only

Fix:

- added `MIN_TRADE_COUNT_TEST2`
- current value is `5`

Relevant code:

- [config.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/config.py:113)
- [utils/model_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/model_utils.py:207)

Lesson:

- out-of-sample validation needs both profitability and sample sufficiency

## Current Debugging Workflow

### Data Debugging

Useful files:

- [logs/sentinel.log](/c:/tmp/py/Git-Claude/ML-Claude-crypto/logs/sentinel.log)
- [detect_gap.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/detect_gap.py)
- [repair_data_gaps.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/repair_data_gaps.py)

Typical checks:

1. inspect the latest log tail
2. check base CSV row growth
3. audit missing timestamp ranges
4. confirm whether a gap is raw-data or working-layer only
5. repair base gaps before trusting forward-filled snapshots

### Training / Model Debugging

Useful files:

- [notebooks/00_sentinel.ipynb](/c:/tmp/py/Git-Claude/ML-Claude-crypto/notebooks/00_sentinel.ipynb)
- [utils/model_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/model_utils.py)
- [models/altcoins](/c:/tmp/py/Git-Claude/ML-Claude-crypto/models/altcoins)

Typical checks:

1. check whether `Model saved` appears in the log
2. check model file and meta file modified times
3. inspect `model_registry.jsonl`
4. compare `Test1` and `Test2` trade counts
5. check whether the model was rejected by gates or failed earlier

### Signal Debugging

Useful files:

- [state/signals_state.json](/c:/tmp/py/Git-Claude/ML-Claude-crypto/state/signals_state.json)
- [utils/signal_manager.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/signal_manager.py)

Typical checks:

1. verify if signal is active vs archived
2. inspect `archive_reason`
3. inspect `first_signal_at`, `expiry_at`, `repeat_count`
4. check whether the state file was successfully written

## Current Governance / Metadata Behavior

Added during recent refactor:

- pipeline signatures
- lightweight feature/version metadata
- model registry append log
- audit log append log
- data quality scoring

Relevant files:

- [utils/governance_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/governance_utils.py)
- [utils/data_quality.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_quality.py)

Current status:

- present in code
- not fully enforced yet
- designed to be backward-compatible

Important current toggles:

- `ENFORCE_PIPELINE_SIGNATURE = False`
- `ENFORCE_DATA_QUALITY = False`

Why they are still off:

- to avoid breaking current notebooks and old saved models mid-migration

## Fact-Checked Current Constraints

The following are true in the current repo as of this reference:

- raw crypto data repair exists in [utils/data_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_utils.py:444)
- data quality scoring exists in [utils/data_quality.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/data_quality.py:23)
- pipeline signature generation exists in [utils/governance_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/governance_utils.py:75)
- model metadata upgrade exists in [utils/governance_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/governance_utils.py:132)
- model path fallback exists in [utils/model_utils.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/model_utils.py:301)
- signal expiry inference for legacy signals exists in [utils/signal_manager.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/utils/signal_manager.py:71)
- live and validation thresholds are now split in [config.py](/c:/tmp/py/Git-Claude/ML-Claude-crypto/config.py:121)

## What To Preserve In A New System

- strict separation between raw/base data and working data
- bounded repair of raw historical gaps before working-layer smoothing
- one canonical path builder per artifact type
- backward-compatible load behavior during migrations
- retry-safe atomic writes
- explicit signal lifecycle state
- separate validation thresholds and live firing thresholds
- append-only registry/audit logs

## Improvement Areas For A New System

### 1. Move Notebook Orchestration Into Python Entry Points

Why:

- notebooks drift
- execution order becomes fragile
- debugging is slower

Recommendation:

- move training, inference, and scheduling into explicit Python runners
- keep notebooks for analysis only

### 2. Centralize All Artifact Naming

Why:

- path drift caused real bugs

Recommendation:

- one shared artifact naming module for:
  - model files
  - meta files
  - feature parquets
  - snapshots
  - registry entries

### 3. Track Fills Explicitly

Why:

- current data quality uses approximations for fill ratio

Recommendation:

- record fill events per ticker/timestamp
- store fill provenance in sidecar metadata

### 4. Enforce Pipeline Signature At Runtime

Why:

- current enforcement is off for compatibility

Recommendation:

- once all active models are regenerated, switch enforcement on

### 5. Add Real Feature Registry / Versioning

Why:

- current feature contract is lightweight only

Recommendation:

- define versioned feature groups
- include hash/version lineage in saved model metadata

### 6. Strengthen Validation Beyond PF

Why:

- PF alone can still be gamed by small samples

Recommendation:

- keep `MIN_TRADE_COUNT_TEST2`
- consider min wins/losses in `Test2`
- consider expectancy floor
- consider loss-streak guardrails

### 7. Split Model Selection From Signal Frequency

Why:

- loosening validation and loosening signal thresholds are different decisions

Recommendation:

- keep model validity statistically sane
- tune live firing thresholds separately for signal volume

### 8. Add Better Migration Tooling

Why:

- state/model metadata changes happen over time

Recommendation:

- add explicit migration scripts for:
  - model metadata upgrades
  - signal state upgrades
  - feature store path changes

### 9. Add Read-Only Health Commands

Why:

- debugging is repetitive

Recommendation:

- create simple scripts for:
  - data gap audit
  - model inventory
  - signal inventory
  - data quality report
  - registry consistency check

## Recommended Build Principles For The Next System

If rebuilding from scratch, prefer this order:

1. artifact naming + path conventions first
2. raw/base and working data split second
3. feature pipeline module third
4. labeling and validation contract fourth
5. model registry / governance fifth
6. live inference + signals sixth
7. notebook views only after the production path works

## Closing Note

The current repo is usable, but it has already demonstrated the main failure modes of notebook-first trading systems:

- path drift
- state drift
- schema drift
- file lock friction
- notebook logic drift
- weak migration boundaries

The recent fixes reduced those risks significantly, but the next system should treat these as first-class architecture concerns from day one.
