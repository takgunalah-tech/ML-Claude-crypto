"""
Volatility-aware label generation + adaptive grid search.

Label logic:
  TP = close * (1 + tp_pct + k1 * ATR_norm)
  SL = close * (1 - sl_pct - k2 * ATR_norm)
  y  = 1 if high of next N candles hits TP first, else 0
"""

import numpy as np
import pandas as pd
import logging
from itertools import product

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ── Core labeling ─────────────────────────────────────────────────────────────

def generate_labels(
    df: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    k1: float,
    k2: float,
    horizon: int = None,
    direction: str = 'long',
) -> pd.Series:
    """
    Generate binary labels for a given TP/SL combination.

    direction='long':
        TP_price = close * (1 + tp_pct + k1 * ATR_norm)
        SL_price = close * (1 - sl_pct - k2 * ATR_norm)
        y = 1 if high hits TP before low hits SL within `horizon` candles

    direction='short':
        TP_price = close * (1 - tp_pct - k1 * ATR_norm)
        SL_price = close * (1 + sl_pct + k2 * ATR_norm)
        y = 1 if low hits TP before high hits SL within `horizon` candles

    Returns a pd.Series with NaN for the last `horizon` rows (incomplete window).
    """
    if horizon is None:
        horizon = config.LABEL_HORIZON

    df = df.reset_index(drop=True)
    close  = df['close'].values
    high   = df['high'].values
    low    = df['low'].values
    atr    = df['ATR_14'].values if 'ATR_14' in df.columns else _fallback_atr(df)
    atr_norm = np.where(close > 0, atr / close, 0.0)

    n = len(df)
    labels = np.full(n, np.nan)

    if direction == 'long':
        tp_prices = close * (1 + tp_pct + k1 * atr_norm)
        sl_prices = close * (1 - sl_pct - k2 * atr_norm)
    else:
        tp_prices = close * (1 - tp_pct - k1 * atr_norm)
        sl_prices = close * (1 + sl_pct + k2 * atr_norm)

    for i in range(n - horizon):
        tp = tp_prices[i]
        sl = sl_prices[i]
        tp_hit = n  # default: never hit
        sl_hit = n

        for j in range(i + 1, i + 1 + horizon):
            h, l = high[j], low[j]
            if direction == 'long':
                if h >= tp and tp_hit == n:
                    tp_hit = j
                if l <= sl and sl_hit == n:
                    sl_hit = j
            else:
                if l <= tp and tp_hit == n:
                    tp_hit = j
                if h >= sl and sl_hit == n:
                    sl_hit = j
            if tp_hit < n and sl_hit < n:
                break

        labels[i] = 1 if tp_hit < sl_hit else 0

    return pd.Series(labels, index=df.index, name='label')


def _fallback_atr(df: pd.DataFrame, length: int = 14) -> np.ndarray:
    """Compute ATR from scratch if not in DataFrame."""
    high  = df['high'].values
    low   = df['low'].values
    close = df['close'].values
    n = len(df)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(length).mean().values
    return atr


# ── Adaptive grid search ──────────────────────────────────────────────────────

def _score(pf: float, n_trades: int) -> float:
    """Composite score: reward profitability AND frequency."""
    if n_trades < config.MIN_TRADE_COUNT:
        return -1.0
    return pf * np.log(n_trades + 1)


def find_optimal_label_params(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    direction: str = 'long',
    verbose: bool = False,
) -> dict:
    """
    Grid search over (tp_pct, sl_pct, k1, k2) to find the combination
    that maximises PF × log(trade_count) on the TRAINING SET ONLY.

    Returns dict with keys: tp_pct, sl_pct, k1, k2, best_score, best_pf, best_n.
    Falls back to config defaults if no valid combination found.
    """
    best_score  = -np.inf
    best_params = {
        'tp_pct': config.TP_GRID[2],
        'sl_pct': config.SL_GRID[1],
        'k1':     config.K1_GRID[1],
        'k2':     config.K2_GRID[1],
    }
    best_pf = 0.0
    best_n  = 0

    grid = list(product(config.TP_GRID, config.SL_GRID, config.K1_GRID, config.K2_GRID))
    if verbose:
        logger.info(f'Grid search over {len(grid)} combinations (direction={direction})')

    from utils.model_utils import train_xgboost, compute_profit_factor

    for tp, sl, k1, k2 in grid:
        labels = generate_labels(df_train, tp, sl, k1, k2, direction=direction)
        valid_mask = labels.notna()
        X = df_train.loc[valid_mask, feature_cols].copy()
        y = labels[valid_mask].astype(int)

        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            continue
        if len(y) < 30:
            continue

        try:
            model = train_xgboost(X, y)
        except Exception:
            continue

        threshold = config.LONG_THRESHOLD if direction == 'long' else config.SHORT_THRESHOLD
        pf, n_trades = compute_profit_factor(model, X, y, threshold, direction=direction)
        sc = _score(pf, n_trades)

        if verbose:
            logger.debug(f'tp={tp:.3f} sl={sl:.3f} k1={k1} k2={k2} → PF={pf:.3f} n={n_trades} score={sc:.3f}')

        if sc > best_score:
            best_score  = sc
            best_params = {'tp_pct': tp, 'sl_pct': sl, 'k1': k1, 'k2': k2}
            best_pf     = pf
            best_n      = n_trades

    best_params['best_score'] = best_score
    best_params['best_pf']    = best_pf
    best_params['best_n']     = best_n
    return best_params
