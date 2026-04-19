"""
Layer C — Strategy Engine utilities.

XGBoost training, monetary profit-factor evaluation, validity gates, SHAP drivers.
"""

import os
import json
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import xgboost as xgb
# shap is imported lazily inside get_shap_drivers() — saves ~2s on module load

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.governance_utils import (
    append_model_registry,
    audit_event,
    build_model_metadata,
    sanitize_for_json,
    validate_pipeline_signature,
)

logger = logging.getLogger(__name__)


# ── Training ──────────────────────────────────────────────────────────────────

def train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int | None = None,
    nthread: int | None = None,
) -> xgb.XGBClassifier:
    """Train an XGBoost binary classifier.

    Args:
        n_estimators: Override tree count. None → uses config.FINAL_ESTIMATORS (300).
                      Pass config.GRID_SEARCH_ESTIMATORS (50) for fast grid-search ranking.
        nthread:      Override XGBoost thread count. Set to 1 when running inside
                      joblib.Parallel to avoid CPU over-subscription.
    """
    if n_estimators is None:
        n_estimators = config.FINAL_ESTIMATORS
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
        nthread=nthread,
    )
    model.fit(X, y)
    return model


# ── Evaluation — MONETARY Profit Factor ──────────────────────────────────────

def compute_profit_factor(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = None,
    direction: str = 'long',
    tp_pct: float = 0.030,
    sl_pct: float = 0.015,
    k1: float = 0.5,
    k2: float = 0.3,
    atr_norm: np.ndarray = None,
) -> tuple[float, int]:
    """
    Simulate trades at the given P(win) threshold and compute MONETARY Profit Factor.

    PF = Gross Profit / Gross Loss

    For each triggered trade:
      - Win  (y=1): profit = tp_pct + k1 * atr_norm_i  (TP distance as % of price)
      - Loss (y=0): loss   = sl_pct + k2 * atr_norm_i  (SL distance as % of price)

    PF = sum(profits for all wins) / sum(losses for all losses)

    If atr_norm is None: uses fixed tp_pct/sl_pct per trade (symmetric).

    direction='long':  take trades where p_win >= threshold
    direction='short': take trades where p_win <= threshold  (SHORT_THRESHOLD = 0.40)

    Returns (profit_factor, trade_count).
    """
    if threshold is None:
        threshold = (
            config.VALIDATION_LONG_THRESHOLD
            if direction == 'long'
            else config.VALIDATION_SHORT_THRESHOLD
        )

    p_win = model.predict_proba(X)[:, 1]

    if direction == 'long':
        mask = p_win >= threshold
    else:
        mask = p_win <= threshold   # p_win <= SHORT_THRESHOLD (already the right value)

    if mask.sum() == 0:
        return 0.0, 0

    y_sel = np.array(y)[mask]

    if atr_norm is not None:
        atr_sel          = np.array(atr_norm)[mask]
        profits_per_trade = tp_pct + k1 * atr_sel
        losses_per_trade  = sl_pct + k2 * atr_sel
    else:
        profits_per_trade = np.full(len(y_sel), tp_pct)
        losses_per_trade  = np.full(len(y_sel), sl_pct)

    gross_profit = (y_sel       * profits_per_trade).sum()
    gross_loss   = ((1 - y_sel) * losses_per_trade).sum()

    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    return round(float(pf), 4), int(mask.sum())


def evaluate_model(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = None,
    direction: str = 'long',
    tp_pct: float = 0.030,
    sl_pct: float = 0.015,
    k1: float = 0.5,
    k2: float = 0.3,
    atr_norm: np.ndarray = None,
) -> dict:
    """
    Return full metrics dict: monetary PF, trade_count, win_rate, expectancy,
    max drawdown, and loss streak.
    Pass the same tp_pct/sl_pct/k1/k2/atr_norm used in labeling for correct PF.
    """
    if threshold is None:
        threshold = (
            config.VALIDATION_LONG_THRESHOLD
            if direction == 'long'
            else config.VALIDATION_SHORT_THRESHOLD
        )

    p_win = model.predict_proba(X)[:, 1]
    if direction == 'long':
        mask = p_win >= threshold
    else:
        mask = p_win <= threshold

    n_trades = int(mask.sum())
    if n_trades == 0:
        return {
            'PF': 0.0,
            'trade_count': 0,
            'win_rate': 0.0,
            'expectancy': 0.0,
            'max_drawdown': 0.0,
            'loss_streak': 0,
        }

    y_sel = np.array(y)[mask]
    wins  = int(y_sel.sum())

    if atr_norm is not None:
        atr_sel          = np.array(atr_norm)[mask]
        profits_per_trade = tp_pct + k1 * atr_sel
        losses_per_trade  = sl_pct + k2 * atr_sel
    else:
        profits_per_trade = np.full(len(y_sel), tp_pct)
        losses_per_trade  = np.full(len(y_sel), sl_pct)

    gross_profit = (y_sel       * profits_per_trade).sum()
    gross_loss   = ((1 - y_sel) * losses_per_trade).sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    pnl = np.where(y_sel == 1, profits_per_trade, -losses_per_trade)
    expectancy = float(np.mean(pnl)) if len(pnl) else 0.0
    equity = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(np.maximum(equity, 0))
    drawdown = running_peak - equity
    max_drawdown = float(drawdown.max()) if len(drawdown) else 0.0

    loss_streak = 0
    current_loss_streak = 0
    for outcome in y_sel:
        if outcome == 0:
            current_loss_streak += 1
            loss_streak = max(loss_streak, current_loss_streak)
        else:
            current_loss_streak = 0

    return {
        'PF':          round(float(pf), 4),
        'trade_count': n_trades,
        'win_rate':    round(float(wins / n_trades), 4),
        'expectancy':  round(expectancy, 6),
        'max_drawdown': round(max_drawdown, 6),
        'loss_streak': int(loss_streak),
    }


# ── Validity gate ─────────────────────────────────────────────────────────────

def check_validity(test1: dict, test2: dict) -> bool:
    """
    A model is valid if all core gates pass:
      1. test1 trade_count >= MIN_TRADE_COUNT
      2. test2 trade_count >= MIN_TRADE_COUNT_TEST2
      3. test1 PF >= MIN_PF_TEST1
      4. test2 PF >= test1 PF * PF_STABILITY_RATIO  (out-of-sample stability)
      5. test1 max_drawdown <= MAX_DRAWDOWN
    """
    if test1['trade_count'] < config.MIN_TRADE_COUNT:
        return False
    if test2['trade_count'] < config.MIN_TRADE_COUNT_TEST2:
        return False
    if test1['PF'] < config.MIN_PF_TEST1:
        return False
    if test1['PF'] > 0 and test2['PF'] < test1['PF'] * config.PF_STABILITY_RATIO:
        return False
    if test1.get('max_drawdown', 0.0) > config.MAX_DRAWDOWN:
        return False
    return True


# ── Feature importance (SHAP) ─────────────────────────────────────────────────

# Module-level SHAP explainer cache: {ticker: (model_path_mtime, explainer)}
# Invalidated automatically when the model file is updated (every 72h retrain).
_shap_cache: dict[str, tuple[float, object]] = {}


def get_shap_drivers(
    model: xgb.XGBClassifier,
    X_sample: pd.DataFrame,
    n: int = 5,
    ticker: str | None = None,
) -> list[str]:
    """Return top-n feature names by mean absolute SHAP value.

    If `ticker` is provided, the SHAP TreeExplainer is cached in memory and
    reused across hourly calls — it is only rebuilt when the model file changes
    (i.e., after a 72h retrain). This saves ~2s per coin per hourly pipeline run.
    """
    global _shap_cache
    try:
        import shap  # lazy import — shap takes ~2s to load on first call

        explainer = None

        if ticker is not None:
            model_path = _model_path(ticker)
            try:
                mtime = os.path.getmtime(model_path)
            except OSError:
                mtime = None

            cached = _shap_cache.get(ticker)
            if cached is not None and cached[0] == mtime:
                explainer = cached[1]  # cache hit — reuse existing explainer
            else:
                explainer = shap.TreeExplainer(model)
                _shap_cache[ticker] = (mtime, explainer)
        else:
            explainer = shap.TreeExplainer(model)

        shap_vals = explainer.shap_values(X_sample)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        top_idx   = np.argsort(mean_abs)[::-1][:n]
        return list(X_sample.columns[top_idx])
    except Exception as e:
        logger.warning(f'SHAP failed: {e}. Falling back to XGBoost importance.')
        imp     = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:n]
        return list(X_sample.columns[top_idx])


# ── Save / Load model ─────────────────────────────────────────────────────────

def _model_path(ticker: str) -> str:
    safe = ticker.replace('/', '_').replace(':', '_').replace('^', '')
    return os.path.join(config.MODELS_DIR, f'{safe}.json')


def _meta_path(ticker: str) -> str:
    safe = ticker.replace('/', '_').replace(':', '_').replace('^', '')
    return os.path.join(config.MODELS_DIR, f'{safe}_meta.json')


def _legacy_model_path_candidates(ticker: str) -> list[tuple[str, str]]:
    """
    Backward-compatible lookup for historical filename conventions.

    Older runs used spot-style names like ZEC_USDT.json even when the runtime
    ticker was ZEC/USDT:USDT. New canonical names keep the full futures suffix
    as ZEC_USDT_USDT.json.
    """
    candidates = [(_model_path(ticker), _meta_path(ticker))]

    legacy_ticker = ticker.replace(':USDT', '')
    if legacy_ticker != ticker:
        legacy_safe = legacy_ticker.replace('/', '_').replace(':', '_').replace('^', '')
        candidates.append((
            os.path.join(config.MODELS_DIR, f'{legacy_safe}.json'),
            os.path.join(config.MODELS_DIR, f'{legacy_safe}_meta.json'),
        ))

    return candidates


def save_model(model: xgb.XGBClassifier, ticker: str, meta: dict) -> None:
    """Save XGBoost model (native JSON) + metadata."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    meta = build_model_metadata(ticker, meta)
    model.save_model(_model_path(ticker))
    with open(_meta_path(ticker), 'w') as f:
        json.dump(sanitize_for_json(meta), f, indent=2)
    append_model_registry({
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'ticker': ticker,
        'model_id': meta.get('model_id'),
        'pipeline_signature': meta.get('pipeline_signature'),
        'deployment_status': 'saved',
        'metrics': {
            'test1': meta.get('test1'),
            'test2': meta.get('test2'),
        },
    })
    audit_event('model_saved', {
        'ticker': ticker,
        'model_id': meta.get('model_id'),
        'pipeline_signature': meta.get('pipeline_signature'),
    })
    logger.info(f'[{ticker}] Model saved.')


def load_model(ticker: str) -> tuple[xgb.XGBClassifier | None, dict | None]:
    """Load saved model + metadata. Returns (None, None) if not found."""
    mp = None
    ep = None
    for model_path, meta_path in _legacy_model_path_candidates(ticker):
        if os.path.exists(model_path) and os.path.exists(meta_path):
            mp, ep = model_path, meta_path
            break
    if mp is None or ep is None:
        return None, None
    model = xgb.XGBClassifier()
    model.load_model(mp)
    with open(ep) as f:
        meta = json.load(f)
    ok, reason = validate_pipeline_signature(meta.get('pipeline_signature'))
    if not ok:
        logger.warning(f'[{ticker}] Model blocked: {reason}')
        audit_event('model_blocked', {'ticker': ticker, 'reason': reason})
        return None, None
    meta.setdefault('pipeline_signature_status', reason)
    return model, meta


def list_valid_models() -> list[str]:
    """
    Return list of tickers that have a saved model + meta.
    Reads the original ticker from the 'ticker' field in the meta JSON
    (avoids fragile filename→ticker reconstruction which breaks for macro tickers).
    """
    if not os.path.exists(config.MODELS_DIR):
        return []
    tickers = []
    for fname in os.listdir(config.MODELS_DIR):
        if not fname.endswith('_meta.json'):
            continue
        path = os.path.join(config.MODELS_DIR, fname)
        try:
            with open(path) as fh:
                meta = json.load(fh)
            ok, _ = validate_pipeline_signature(meta.get('pipeline_signature'))
            if not ok:
                continue
            tickers.append(meta['ticker'])
        except (json.JSONDecodeError, KeyError):
            logger.warning(f'Could not read ticker from {fname} — skipping.')
    return tickers
