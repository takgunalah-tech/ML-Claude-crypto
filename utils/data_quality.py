"""
Lightweight data quality scoring for the v2 system contract.

This is intentionally conservative and additive:
  - computes quality metrics
  - returns a score and flags
  - does not block the pipeline unless the config toggle is enabled
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_data_quality(
    df: pd.DataFrame,
    asset_type: str = 'crypto',
    expected_freq: str = '1h',
    now: datetime | None = None,
) -> dict:
    if df is None or df.empty or 'timestamp' not in df.columns:
        return {
            'score': 0.0,
            'gap_ratio': 1.0,
            'fill_ratio': 0.0,
            'freshness_delay_hours': float('inf'),
            'volatility_jump_score': 1.0,
            'passes_threshold': False,
        }

    ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce').dropna().sort_values().reset_index(drop=True)
    if len(ts) < 2:
        freshness_delay = float('inf')
        return {
            'score': 0.0,
            'gap_ratio': 1.0,
            'fill_ratio': 0.0,
            'freshness_delay_hours': freshness_delay,
            'volatility_jump_score': 1.0,
            'passes_threshold': False,
        }

    expected_delta = pd.Timedelta(expected_freq)
    diffs = ts.diff().dropna()
    missing = sum(max(int(delta / expected_delta) - 1, 0) for delta in diffs if delta > expected_delta)
    expected_rows = len(ts) + missing
    gap_ratio = missing / expected_rows if expected_rows > 0 else 0.0

    # Approximation: explicit fills are not tracked historically yet, so use the
    # current null ratio as a conservative proxy until the pipeline stores fill events.
    fill_ratio = float(df.isna().sum().sum()) / float(df.size) if df.size else 0.0

    close = pd.to_numeric(df.get('close'), errors='coerce')
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) >= 20 and ret.std() > 0:
        z = ((ret - ret.mean()) / ret.std()).abs()
        volatility_jump_score = float(min(z.max() / 10.0, 1.0))
    else:
        volatility_jump_score = 0.0

    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    last_ts = ts.iloc[-1]
    freshness_delay = max((now_ts - last_ts).total_seconds() / 3600, 0.0)

    # Markets with defined trading closures tolerate longer freshness delays.
    freshness_limit = 6.0 if asset_type == 'crypto' else 72.0
    freshness_penalty = min(freshness_delay / freshness_limit, 1.0)

    penalties = (
        0.45 * gap_ratio +
        0.10 * fill_ratio +
        0.25 * freshness_penalty +
        0.20 * volatility_jump_score
    )
    score = max(0.0, min(1.0, 1.0 - penalties))

    return {
        'score': round(float(score), 4),
        'gap_ratio': round(float(gap_ratio), 4),
        'fill_ratio': round(float(fill_ratio), 4),
        'freshness_delay_hours': round(float(freshness_delay), 2),
        'volatility_jump_score': round(float(volatility_jump_score), 4),
        'passes_threshold': score >= getattr(config, 'DATA_QUALITY_THRESHOLD', 0.75),
    }
