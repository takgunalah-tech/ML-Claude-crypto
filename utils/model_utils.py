"""
Layer C — Strategy Engine utilities.

XGBoost training, profit-factor evaluation, validity gates, SHAP drivers.
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


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_profit_factor(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = None,
    direction: str = 'long',
) -> tuple[float, int]:
    """
    Simulate trades at the given P(win) threshold and compute Profit Factor.

    direction='long':  take trades where p_win >= threshold
    direction='short': take trades where p_win <= (1 - threshold)
                       (i.e. p_short = 1 - p_win >= threshold)

    Returns (profit_factor, trade_count).
    """
    if threshold is None:
        threshold = config.LONG_THRESHOLD if direction == 'long' else config.SHORT_THRESHOLD

    p_win = model.predict_proba(X)[:, 1]

    if direction == 'long':
        mask = p_win >= threshold
    else:
        mask = p_win <= (1 - threshold)

    if mask.sum() == 0:
        return 0.0, 0

    y_sel = np.array(y)[mask]
    wins   = y_sel.sum()
    losses = len(y_sel) - wins

    if losses == 0:
        pf = float('inf')
    else:
        pf = wins / losses

    return round(pf, 4), int(mask.sum())


def evaluate_model(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = None,
    direction: str = 'long',
) -> dict:
    """Return full metrics dict: PF, trade_count, win_rate."""
    if threshold is None:
        threshold = config.LONG_THRESHOLD if direction == 'long' else config.SHORT_THRESHOLD

    p_win = model.predict_proba(X)[:, 1]
    if direction == 'long':
        mask = p_win >= threshold
    else:
        mask = p_win <= (1 - threshold)

    n_trades = mask.sum()
    if n_trades == 0:
        return {'PF': 0.0, 'trade_count': 0, 'win_rate': 0.0}

    y_sel  = np.array(y)[mask]
    wins   = y_sel.sum()
    losses = len(y_sel) - wins
    pf     = wins / losses if losses > 0 else float('inf')

    return {
        'PF':          round(float(pf), 4),
        'trade_count': int(n_trades),
        'win_rate':    round(float(wins / n_trades), 4),
    }


# ── Validity gate ─────────────────────────────────────────────────────────────

def check_validity(test1: dict, test2: dict) -> bool:
    """
    A model is valid if ALL three gates pass:
      1. test1 trade_count >= MIN_TRADE_COUNT
      2. test1 PF >= MIN_PF_TEST1
      3. test2 PF >= test1 PF * PF_STABILITY_RATIO
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
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:n]
        return list(X_sample.columns[top_idx])


# ── Save / Load model ─────────────────────────────────────────────────────────

def _model_path(ticker: str) -> str:
    safe = ticker.replace('/', '_')
    return os.path.join(config.MODELS_DIR, f'{safe}.json')


def _meta_path(ticker: str) -> str:
    safe = ticker.replace('/', '_')
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
    """Return list of tickers that have a saved model + meta."""
    if not os.path.exists(config.MODELS_DIR):
        return []
    files = os.listdir(config.MODELS_DIR)
    tickers = []
    for f in files:
        if f.endswith('_meta.json'):
            safe = f.replace('_meta.json', '')
            ticker = safe.replace('_', '/', 1)   # BTC_USDT → BTC/USDT
            tickers.append(ticker)
    return tickers
