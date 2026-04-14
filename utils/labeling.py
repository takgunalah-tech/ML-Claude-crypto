"""
Volatility-aware label generation + adaptive grid search.

Label logic:
  direction='long':
    TP_price = close * (1 + tp_pct + k1 * ATR_norm)
    SL_price = close * (1 - sl_pct - k2 * ATR_norm)
    y = 1  if high of next N candles hits TP before low hits SL
    y = 0  if SL hit first
    y = NaN if neither hit within horizon (excluded from training)

  direction='short':
    TP_price = close * (1 - tp_pct - k1 * ATR_norm)
    SL_price = close * (1 + sl_pct + k2 * ATR_norm)
    y = 1  if low hits TP before high hits SL
    y = 0  if SL hit first
    y = NaN if neither hit within horizon
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

    Returns a pd.Series with:
      1   — TP hit before SL within `horizon` candles
      0   — SL hit before TP within `horizon` candles
      NaN — Neither hit (ambiguous; excluded from model training via dropna)

    The last `horizon` rows always get NaN (incomplete look-ahead window).
    """
    if horizon is None:
        horizon = config.LABEL_HORIZON

    df    = df.reset_index(drop=True)
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    atr   = df['ATR_14'].values if 'ATR_14' in df.columns else _fallback_atr(df)
    atr_norm = np.where(close > 0, atr / close, 0.0)

    n      = len(df)
    labels = np.full(n, np.nan)

    # Abort silently if horizon >= dataset length
    if horizon >= n:
        logger.warning(
            f'generate_labels: horizon ({horizon}) >= dataset length ({n}). '
            f'All labels will be NaN.'
        )
        return pd.Series(labels, index=df.index, name='label')

    if direction == 'long':
        tp_prices = close * (1 + tp_pct + k1 * atr_norm)
        sl_prices = close * (1 - sl_pct - k2 * atr_norm)
    else:
        tp_prices = close * (1 - tp_pct - k1 * atr_norm)
        sl_prices = close * (1 + sl_pct + k2 * atr_norm)

    for i in range(n - horizon):
        tp = tp_prices[i]
        sl = sl_prices[i]
        tp_hit = n  # sentinel: never hit
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
                break  # both resolved — no need to scan further

        # Assign label:
        if tp_hit < sl_hit:
            labels[i] = 1       # TP hit first → win
        elif sl_hit < tp_hit:
            labels[i] = 0       # SL hit first → loss
        # else: both == n → neither hit → leave as NaN (excluded from training)

    return pd.Series(labels, index=df.index, name='label')


def _fallback_atr(df: pd.DataFrame, length: int = 14) -> np.ndarray:
    """Compute ATR from scratch if ATR_14 column is not in DataFrame."""
    high  = df['high'].values
    low   = df['low'].values
    close = df['close'].values
    n  = len(df)
    tr = np.zeros(n)

    # First candle: use high-low range as best estimate (no previous close)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i]  - close[i - 1]),
        )

    atr = pd.Series(tr).rolling(length).mean().values
    return atr


# ── Adaptive grid search ──────────────────────────────────────────────────────

def _score(pf: float, n_trades: int) -> float:
    """
    Composite score: reward both profitability AND frequency.
    sqrt(n_trades) grows fast enough to prefer 100 trades over 20 trades
    for the same PF, without letting n_trades dominate.
    """
    if n_trades < config.MIN_TRADE_COUNT:
        return -1.0
    return pf * np.sqrt(n_trades)


def find_optimal_label_params(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    direction: str = 'long',
    verbose: bool = False,
) -> dict:
    """
    Grid search over (tp_pct, sl_pct, k1, k2) to find the combination that
    maximises PF × sqrt(trade_count) on the TRAINING SET ONLY.

    Anti look-ahead contract: the caller must ensure df_train ends at least
    LABEL_HORIZON candles before the test1 window starts:
        label_buffer = pd.Timedelta(hours=config.LABEL_HORIZON)
        train_df = feat[ts <= (train_end - label_buffer)]

    Returns dict with keys:
        tp_pct, sl_pct, k1, k2  — best parameters
        best_score, best_pf, best_n
        runner_ups               — list of top-3 dicts [{params, score, pf, n_trades}, ...]
    """
    from utils.model_utils import train_xgboost, compute_profit_factor

    # Default fallback if grid search finds nothing valid
    default_params = {
        'tp_pct': config.TP_GRID[2],
        'sl_pct': config.SL_GRID[1],
        'k1':     config.K1_GRID[1],
        'k2':     config.K2_GRID[1],
    }

    grid = list(product(config.TP_GRID, config.SL_GRID, config.K1_GRID, config.K2_GRID))
    if verbose:
        logger.info(f'Grid search: {len(grid)} combinations (direction={direction})')

    top_results: list[tuple] = []  # (score, params_dict, pf, n_trades)

    for tp, sl, k1, k2 in grid:
        labels     = generate_labels(df_train, tp, sl, k1, k2, direction=direction)
        valid_mask = labels.notna()
        X = df_train.loc[valid_mask, feature_cols].copy()
        y = labels[valid_mask].astype(int)

        # Require minimum class representation
        if len(y) < 30:
            continue
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            continue

        try:
            model = train_xgboost(X, y)
        except Exception:
            continue

        threshold = config.LONG_THRESHOLD if direction == 'long' else config.SHORT_THRESHOLD
        pf, n_trades = compute_profit_factor(
            model, X, y, threshold, direction=direction,
            tp_pct=tp, sl_pct=sl, k1=k1, k2=k2,
            atr_norm=(X['ATR_14_norm'].values if 'ATR_14_norm' in X.columns else None),
        )
        sc = _score(pf, n_trades)

        if verbose:
            logger.debug(
                f'  tp={tp:.3f} sl={sl:.3f} k1={k1} k2={k2} '
                f'→ PF={pf:.3f} n={n_trades} score={sc:.3f}'
            )

        top_results.append((sc, {'tp_pct': tp, 'sl_pct': sl, 'k1': k1, 'k2': k2}, pf, n_trades))

    if not top_results:
        logger.warning('Grid search: no valid combination found — using defaults.')
        return {
            **default_params,
            'best_score': -1.0,
            'best_pf':    0.0,
            'best_n':     0,
            'runner_ups': [],
        }

    # Sort descending by score, keep top 3
    top_results.sort(key=lambda x: x[0], reverse=True)
    top3 = top_results[:3]

    best_score, best_params, best_pf, best_n = top3[0]

    runner_ups = [
        {
            'params':   r[1],
            'score':    round(float(r[0]), 3),
            'pf':       round(float(r[2]), 3),
            'n_trades': int(r[3]),
        }
        for r in top3
    ]

    return {
        **best_params,
        'best_score': round(float(best_score), 3),
        'best_pf':    round(float(best_pf), 3),
        'best_n':     int(best_n),
        'runner_ups': runner_ups,   # includes the winner as runner_ups[0]
    }
