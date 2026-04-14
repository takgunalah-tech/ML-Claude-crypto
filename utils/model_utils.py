"""
Layer C — Strategy Engine utilities.

XGBoost training, monetary profit-factor evaluation, validity gates, SHAP drivers.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
# shap is imported lazily inside get_shap_drivers() — saves ~2s on module load

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ── Training ──────────────────────────────────────────────────────────────────

def train_xgboost(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """Train an XGBoost binary classifier."""
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
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
        threshold = config.LONG_THRESHOLD if direction == 'long' else config.SHORT_THRESHOLD

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
    Return full metrics dict: monetary PF, trade_count, win_rate.
    Pass the same tp_pct/sl_pct/k1/k2/atr_norm used in labeling for correct PF.
    """
    if threshold is None:
        threshold = config.LONG_THRESHOLD if direction == 'long' else config.SHORT_THRESHOLD

    p_win = model.predict_proba(X)[:, 1]
    if direction == 'long':
        mask = p_win >= threshold
    else:
        mask = p_win <= threshold

    n_trades = int(mask.sum())
    if n_trades == 0:
        return {'PF': 0.0, 'trade_count': 0, 'win_rate': 0.0}

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

    return {
        'PF':          round(float(pf), 4),
        'trade_count': n_trades,
        'win_rate':    round(float(wins / n_trades), 4),
    }


# ── Validity gate ─────────────────────────────────────────────────────────────

def check_validity(test1: dict, test2: dict) -> bool:
    """
    A model is valid if ALL three gates pass:
      1. test1 trade_count >= MIN_TRADE_COUNT
      2. test1 PF >= MIN_PF_TEST1
      3. test2 PF >= test1 PF * PF_STABILITY_RATIO  (out-of-sample stability)
    """
    if test1['trade_count'] < config.MIN_TRADE_COUNT:
        return False
    if test1['PF'] < config.MIN_PF_TEST1:
        return False
    if test1['PF'] > 0 and test2['PF'] < test1['PF'] * config.PF_STABILITY_RATIO:
        return False
    return True


# ── Feature importance (SHAP) ─────────────────────────────────────────────────

def get_shap_drivers(
    model: xgb.XGBClassifier,
    X_sample: pd.DataFrame,
    n: int = 5,
) -> list[str]:
    """Return top-n feature names by mean absolute SHAP value."""
    try:
        import shap  # lazy import — shap takes ~2s to load
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
    safe = ticker.replace('/', '_').replace('^', '')
    return os.path.join(config.MODELS_DIR, f'{safe}.json')


def _meta_path(ticker: str) -> str:
    safe = ticker.replace('/', '_').replace('^', '')
    return os.path.join(config.MODELS_DIR, f'{safe}_meta.json')


def save_model(model: xgb.XGBClassifier, ticker: str, meta: dict) -> None:
    """Save XGBoost model (native JSON) + metadata."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model.save_model(_model_path(ticker))
    with open(_meta_path(ticker), 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f'[{ticker}] Model saved.')


def load_model(ticker: str) -> tuple[xgb.XGBClassifier | None, dict | None]:
    """Load saved model + metadata. Returns (None, None) if not found."""
    mp = _model_path(ticker)
    ep = _meta_path(ticker)
    if not os.path.exists(mp) or not os.path.exists(ep):
        return None, None
    model = xgb.XGBClassifier()
    model.load_model(mp)
    with open(ep) as f:
        meta = json.load(f)
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
            tickers.append(meta['ticker'])
        except (json.JSONDecodeError, KeyError):
            logger.warning(f'Could not read ticker from {fname} — skipping.')
    return tickers
