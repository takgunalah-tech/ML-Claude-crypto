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

# joblib is part of scikit-learn — always available in this project.
try:
    from joblib import Parallel, delayed as _delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    logger.warning('joblib not found — grid search will run sequentially.')


# ── Core labeling ─────────────────────────────────────────────────────────────

def _generate_labels_python(
    df: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    k1: float,
    k2: float,
    horizon: int = None,
    direction: str = 'long',
) -> pd.Series:
    """
    Original pure-Python label generator (reference implementation).
    Kept for parity testing against the vectorized version.
    ~840k iterations per call at default horizon=48 on 17,500-row datasets.
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
        tp_hit = n
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

        if tp_hit < sl_hit:
            labels[i] = 1
        elif sl_hit < tp_hit:
            labels[i] = 0

    return pd.Series(labels, index=df.index, name='label')


def _generate_labels_numpy(
    df: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    k1: float,
    k2: float,
    horizon: int = None,
    direction: str = 'long',
) -> pd.Series:
    """
    Vectorized NumPy label generator using sliding_window_view.

    Replaces the Python double-for-loop with NumPy C-level operations:
      - sliding_window_view creates a (n-horizon, horizon) view of high/low
      - argmax finds the first horizon index where TP/SL condition is met
      - Final label assigned by comparing tp_hit_idx vs sl_hit_idx

    Speedup: ~100-500x vs the Python version on typical dataset sizes.

    Edge cases handled identically to the Python version:
      - Simultaneous TP+SL hit in the same candle → NaN (same as Python)
      - Neither hit within horizon → NaN
      - Last `horizon` rows → NaN (incomplete look-ahead window)
    """
    if horizon is None:
        horizon = config.LABEL_HORIZON

    df    = df.reset_index(drop=True)
    close = df['close'].values.astype(np.float64)
    high  = df['high'].values.astype(np.float64)
    low   = df['low'].values.astype(np.float64)
    atr   = df['ATR_14'].values if 'ATR_14' in df.columns else _fallback_atr(df)
    atr_norm = np.where(close > 0, atr / close, 0.0)

    n      = len(df)
    labels = np.full(n, np.nan)

    if horizon >= n:
        logger.warning(
            f'generate_labels: horizon ({horizon}) >= dataset length ({n}). '
            f'All labels will be NaN.'
        )
        return pd.Series(labels, index=df.index, name='label')

    n_valid = n - horizon  # number of rows that can receive a label

    if direction == 'long':
        tp_prices = close * (1 + tp_pct + k1 * atr_norm)
        sl_prices = close * (1 - sl_pct - k2 * atr_norm)
    else:
        tp_prices = close * (1 - tp_pct - k1 * atr_norm)
        sl_prices = close * (1 + sl_pct + k2 * atr_norm)

    # Build (n_valid, horizon) sliding windows of high and low.
    # sliding_window_view returns a *view* (zero-copy) so memory overhead is minimal.
    from numpy.lib.stride_tricks import sliding_window_view

    # Windows start at index i+1 (one candle after entry candle i).
    # We skip index 0 of the full array and window over [1 .. n-1].
    high_win = sliding_window_view(high[1:], horizon)[:n_valid]  # (n_valid, horizon)
    low_win  = sliding_window_view(low[1:],  horizon)[:n_valid]  # (n_valid, horizon)

    # tp_prices/sl_prices for entries [0 .. n_valid-1]
    tp = tp_prices[:n_valid, np.newaxis]  # (n_valid, 1) for broadcasting
    sl = sl_prices[:n_valid, np.newaxis]

    if direction == 'long':
        tp_hit_mask = high_win >= tp   # (n_valid, horizon) bool
        sl_hit_mask = low_win  <= sl
    else:
        tp_hit_mask = low_win  <= tp
        sl_hit_mask = high_win >= sl

    # Find the first True position in each row.
    # argmax on a bool array returns 0 if all False (never hit) — we distinguish
    # "hit at index 0" from "never hit" using the any() check.
    tp_any = tp_hit_mask.any(axis=1)   # (n_valid,) bool
    sl_any = sl_hit_mask.any(axis=1)

    tp_first = np.where(tp_any, np.argmax(tp_hit_mask, axis=1), horizon)  # sentinel = horizon
    sl_first = np.where(sl_any, np.argmax(sl_hit_mask, axis=1), horizon)

    # Assign labels
    win  = (tp_first < sl_first)   # TP hit first
    loss = (sl_first < tp_first)   # SL hit first
    # Neither hit, or simultaneous hit (tp_first == sl_first) → NaN (already set)

    labels[:n_valid][win]  = 1
    labels[:n_valid][loss] = 0

    return pd.Series(labels, index=df.index, name='label')


def generate_labels(
    df: pd.DataFrame,
    tp_pct: float,
    sl_pct: float,
    k1: float,
    k2: float,
    horizon: int = None,
    direction: str = 'long',
    use_numpy: bool = True,
) -> pd.Series:
    """
    Generate binary labels for a given TP/SL combination.

    Returns a pd.Series with:
      1   — TP hit before SL within `horizon` candles
      0   — SL hit before TP within `horizon` candles
      NaN — Neither hit (ambiguous; excluded from model training via dropna)

    The last `horizon` rows always get NaN (incomplete look-ahead window).

    Args:
        use_numpy: If True (default), uses the vectorized NumPy implementation
                   (~100-500x faster). Set False to use the original Python version
                   for parity testing or debugging.
    """
    if use_numpy:
        return _generate_labels_numpy(df, tp_pct, sl_pct, k1, k2, horizon, direction)
    return _generate_labels_python(df, tp_pct, sl_pct, k1, k2, horizon, direction)


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


def _eval_one_combo(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    tp: float,
    sl: float,
    k1: float,
    k2: float,
    direction: str,
    verbose: bool,
) -> tuple | None:
    """Evaluate a single (tp, sl, k1, k2) combo. Returns (score, params, pf, n_trades) or None."""
    from utils.model_utils import train_xgboost, compute_profit_factor

    labels     = generate_labels(df_train, tp, sl, k1, k2, direction=direction)
    valid_mask = labels.notna()
    X = df_train.loc[valid_mask, feature_cols].copy()
    y = labels[valid_mask].astype(int)

    if len(y) < 30:
        return None
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None

    try:
        # nthread=1: avoid CPU over-subscription when called from joblib.Parallel
        model = train_xgboost(X, y, n_estimators=config.GRID_SEARCH_ESTIMATORS, nthread=1)
    except Exception:
        return None

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

    return (sc, {'tp_pct': tp, 'sl_pct': sl, 'k1': k1, 'k2': k2}, pf, n_trades)


def find_optimal_label_params(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    direction: str = 'long',
    verbose: bool = False,
    n_jobs: int = 4,
) -> dict:
    """
    Grid search over (tp_pct, sl_pct, k1, k2) to find the combination that
    maximises PF × sqrt(trade_count) on the TRAINING SET ONLY.

    Anti look-ahead contract: the caller must ensure df_train ends at least
    LABEL_HORIZON candles before the test1 window starts:
        label_buffer = pd.Timedelta(hours=config.LABEL_HORIZON)
        train_df = feat[ts <= (train_end - label_buffer)]

    Grid search runs in parallel via joblib (n_jobs workers). Each worker uses
    config.GRID_SEARCH_ESTIMATORS (default 50) trees for fast ranking; the
    final production model is retrained with config.FINAL_ESTIMATORS (300).

    Returns dict with keys:
        tp_pct, sl_pct, k1, k2  — best parameters
        best_score, best_pf, best_n
        runner_ups               — list of top-3 dicts [{params, score, pf, n_trades}, ...]
    """
    # Default fallback if grid search finds nothing valid
    default_params = {
        'tp_pct': config.TP_GRID[2],
        'sl_pct': config.SL_GRID[1],
        'k1':     config.K1_GRID[1],
        'k2':     config.K2_GRID[1],
    }

    grid = list(product(config.TP_GRID, config.SL_GRID, config.K1_GRID, config.K2_GRID))
    if verbose:
        logger.info(
            f'Grid search: {len(grid)} combinations (direction={direction}, '
            f'n_jobs={n_jobs if _JOBLIB_AVAILABLE else 1}, '
            f'estimators={config.GRID_SEARCH_ESTIMATORS}→{config.FINAL_ESTIMATORS})'
        )

    if _JOBLIB_AVAILABLE and n_jobs != 1:
        raw_results = Parallel(n_jobs=n_jobs, prefer='threads')(
            _delayed(_eval_one_combo)(df_train, feature_cols, tp, sl, k1, k2, direction, verbose)
            for tp, sl, k1, k2 in grid
        )
    else:
        raw_results = [
            _eval_one_combo(df_train, feature_cols, tp, sl, k1, k2, direction, verbose)
            for tp, sl, k1, k2 in grid
        ]

    top_results = [r for r in raw_results if r is not None]

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
