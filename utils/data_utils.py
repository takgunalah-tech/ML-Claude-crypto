"""
Layer A — Snowball DB Engine  (3-layer architecture)

  Layer 0  data/base/{asset_type}/{ticker}.csv
           Raw OHLCV only — never modified with TA or ffill.
           Source of record; rollback point.

  Layer 1  data/working/{ticker}_snapshot.csv
           Integrity-protocol-applied working copy.
           Saved after every successful update so the model can recover
           without re-downloading raw data (debug / rollback).

  working_df (returned)
           In-memory Layer 1 slice handed to feature computation + model.
"""

import os
import time
import logging
import shutil
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
# ccxt and yfinance are lazy-imported inside their fetch functions
# to keep module-load time fast (~5-8s saved on ccxt alone)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# Tracks consecutive fetch failures per ticker across hourly calls
_fetch_failure_counts: dict[str, int] = {}


# ── Path helpers ──────────────────────────────────────────────────────────────

def _ticker_to_filename(ticker: str) -> str:
    """'BTC/USDT' → 'BTC_USDT_1h'  (safe for filesystem)"""
    return ticker.replace('/', '_').replace(':', '_').replace('^', '').replace('-', '_') + f'_{config.TIMEFRAME}'


def _base_csv_path(ticker: str, asset_type: str) -> str:
    folder = os.path.join(config.DATA_BASE, asset_type)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'{_ticker_to_filename(ticker)}.csv')


def _snapshot_csv_path(ticker: str, asset_type: str) -> str:
    folder = os.path.join(config.DATA_WORKING, asset_type)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'{_ticker_to_filename(ticker)}_snapshot.csv')


# ── Load / Save — Layer 0 (raw base CSV) ─────────────────────────────────────

def load_base_csv(ticker: str, asset_type: str) -> pd.DataFrame:
    """Load Layer 0 raw OHLCV CSV. Returns empty DataFrame if not found."""
    path = _base_csv_path(ticker, asset_type)
    if not os.path.exists(path):
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = pd.read_csv(path, parse_dates=['timestamp'])
    if df['timestamp'].dt.tz is not None:          # pandas 3.x reads 'Z' suffix as UTC-aware
        df['timestamp'] = df['timestamp'].dt.tz_convert(None)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def save_base_csv(df: pd.DataFrame, ticker: str, asset_type: str) -> None:
    """Atomically overwrite Layer 0 CSV (write .tmp → rename) — crash-safe."""
    path = _base_csv_path(ticker, asset_type)
    df_out = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df_out['timestamp'] = df_out['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    tmp = path + '.tmp'
    df_out.to_csv(tmp, index=False)
    shutil.move(tmp, path)


# ── Load / Save — Layer 1 (working snapshot CSV) ─────────────────────────────

def save_snapshot_csv(df: pd.DataFrame, ticker: str, asset_type: str) -> None:
    """Save enriched Layer 1 snapshot (all columns, not just OHLCV). Atomic write."""
    path = _snapshot_csv_path(ticker, asset_type)
    df_out = df.copy()
    if 'timestamp' in df_out.columns and hasattr(df_out['timestamp'], 'dt'):
        df_out['timestamp'] = df_out['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    tmp = path + '.tmp'
    df_out.to_csv(tmp, index=False)
    shutil.move(tmp, path)
    logger.debug(f'[{ticker}] Snapshot saved: {len(df_out)} rows × {len(df_out.columns)} cols')


def load_snapshot_csv(ticker: str, asset_type: str) -> pd.DataFrame | None:
    """Load Layer 1 snapshot. Returns None if not found."""
    path = _snapshot_csv_path(ticker, asset_type)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['timestamp'])
    if df['timestamp'].dt.tz is not None:          # pandas 3.x reads 'Z' suffix as UTC-aware
        df['timestamp'] = df['timestamp'].dt.tz_convert(None)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


# ── Merge helper ──────────────────────────────────────────────────────────────

def merge_incremental(base_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate, deduplicate on timestamp, sort ascending."""
    combined = pd.concat([base_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    return combined


# ── Validation ────────────────────────────────────────────────────────────────

def validate_ohlcv(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Run quality checks on a freshly fetched OHLCV DataFrame.
    Returns (is_valid, reason).
    """
    if df is None or df.empty:
        return False, 'Empty DataFrame'

    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        return False, f'Missing columns: {required - set(df.columns)}'

    ohlc = df[['open', 'high', 'low', 'close']]
    if ohlc.isnull().any().any():
        return False, 'NaN values in OHLC columns'

    if (df['close'] <= 0).any():
        return False, 'Non-positive close price detected'

    if (df['volume'] < 0).any():
        return False, 'Negative volume detected'

    if (df['high'] < df['low']).any():
        return False, 'high < low detected'

    if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
        return False, 'high < open or close detected'

    if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
        return False, 'low > open or close detected'

    if df['timestamp'].duplicated().any():
        return False, 'Duplicate timestamps detected'

    if not df['timestamp'].is_monotonic_increasing:
        return False, 'Timestamps not monotonically increasing'

    return True, 'OK'


# ── Exchange instance cache ───────────────────────────────────────────────────

_exchange_cache: dict = {}

def _get_exchange(exchange_id: str):
    """Return a cached ccxt exchange instance (avoids reloading markets per ticker)."""
    if exchange_id not in _exchange_cache:
        import ccxt  # lazy — ccxt takes 5-8s to import
        ex = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {'defaultType': 'spot'},
        })
        _exchange_cache[exchange_id] = ex
    return _exchange_cache[exchange_id]


def check_exchange_connectivity(exchange_id: str = None) -> tuple[bool, str]:
    """Quick connectivity test — tries to load markets. Returns (ok, message)."""
    exchange_id = exchange_id or config.EXCHANGE
    try:
        ex = _get_exchange(exchange_id)
        ex.load_markets()
        return True, f'{exchange_id} reachable ({len(ex.markets)} markets loaded)'
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


# ── Fetch: Crypto BASE (yfinance, first-run ~17k candles) ────────────────────

def fetch_crypto_base_yfinance(ticker: str) -> pd.DataFrame:
    """
    First-run only: download ~729 days of 1h OHLCV from yfinance.
    Maps ccxt symbol ('BTC/USDT') → yfinance symbol ('BTC-USD') via config.CRYPTO_YF_MAP.
    Handles yfinance >= 0.2 MultiIndex columns automatically.
    """
    import yfinance as yf  # lazy import

    yf_symbol = config.CRYPTO_YF_MAP.get(ticker)
    if not yf_symbol:
        raise ValueError(
            f'No yfinance mapping for {ticker!r}. '
            f'Add it to config.CRYPTO_YF_MAP.'
        )

    start = (datetime.utcnow() - timedelta(days=729)).strftime('%Y-%m-%d')
    end   = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')

    raw = yf.download(
        yf_symbol, start=start, end=end,
        interval='1h', auto_adjust=True, progress=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Flatten MultiIndex columns (yfinance >= 0.2 returns ('Close','BTC-USD') etc.)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    raw = raw.reset_index()
    # Normalise column names to lowercase
    raw.columns = [str(c).lower() for c in raw.columns]
    # Detect timestamp column (may be 'datetime', 'date', or 'index')
    ts_col = next(
        (c for c in raw.columns if c in ('datetime', 'date', 'index')),
        raw.columns[0]
    )
    raw = raw.rename(columns={ts_col: 'timestamp'})

    # Strip timezone → tz-naive UTC (works across pandas versions)
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    if raw['timestamp'].dt.tz is not None:
        raw['timestamp'] = raw['timestamp'].dt.tz_localize(None)

    raw['volume'] = raw.get('volume', pd.Series(0, index=raw.index)).fillna(0)

    df = raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    return df.reset_index(drop=True)


# ── Fetch: Crypto INCREMENTAL (ccxt/Binance, hourly updates) ─────────────────

def fetch_crypto_incremental_ccxt(ticker: str, since_ms: int) -> pd.DataFrame:
    """
    Incremental fetch from ccxt/Binance. Paginated (handles >1000 candles if
    multiple hours were missed). Used for every update AFTER base data exists.
    Exchange instance is cached — markets loaded only once.
    """
    # exchange = _get_exchange(config.EXCHANGE_SPOT)
    exchange = _get_exchange(config.EXCHANGE_FUTURES)
    all_rows = []
    current_since = since_ms

    while True:
        try:
            candles = exchange.fetch_ohlcv(
                ticker, config.TIMEFRAME, since=current_since, limit=1000
            )
        except Exception as e:
            raise RuntimeError(f'ccxt {type(e).__name__} for {ticker}: {e}') from e

        if not candles:
            break
        all_rows.extend(candles)
        last_ts = candles[-1][0]
        if len(candles) < 1000:
            break
        current_since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(all_rows, columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume'])
    # tz_localize(None) strips tz label → tz-naive UTC values (matches base CSV)
    df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.drop(columns='ts_ms')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


# ── Fetch: Macro (yfinance, both base and incremental) ───────────────────────

def fetch_macro_ohlcv(ticker: str, start_dt: datetime) -> pd.DataFrame:
    """
    Download hourly OHLCV from yfinance starting at start_dt.
    Used for both first-run (since 729 days ago) and incremental (since last_ts + 1h).
    yfinance 1h data is available for ~730 days.
    """
    import yfinance as yf  # lazy import

    yf_ticker = config.MACRO_TICKERS.get(ticker, ticker)
    end_dt = datetime.utcnow() + timedelta(days=1)

    _EMPTY = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    try:
        raw = yf.download(
            yf_ticker,
            start=start_dt.strftime('%Y-%m-%d'),
            end=end_dt.strftime('%Y-%m-%d'),
            interval='1h',
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        logger.warning(f'[{ticker}] yf.download raised: {e}')
        return _EMPTY

    if raw is None or raw.empty:
        return _EMPTY

    # Flatten MultiIndex columns (yfinance >= 0.2)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    raw = raw.reset_index()

    # Detect timestamp column
    ts_col = next(
        (c for c in raw.columns if str(c).lower() in ('datetime', 'date', 'index')),
        raw.columns[0]
    )

    # Normalise all column names to lowercase
    raw.columns = [str(c).lower() for c in raw.columns]
    ts_col = str(ts_col).lower()
    raw = raw.rename(columns={ts_col: 'timestamp'})

    # Strip timezone → tz-naive UTC (works across pandas versions)
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    if raw['timestamp'].dt.tz is not None:
        raw['timestamp'] = raw['timestamp'].dt.tz_localize(None)

    # Fill NaN volume with 0 (VIX and indices have no trading volume)
    raw['volume'] = raw.get('volume', pd.Series(0, index=raw.index)).fillna(0)

    df = raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df[df['timestamp'] >= pd.Timestamp(start_dt)].reset_index(drop=True)
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    return df


# ── Integrity Protocol — applied to working_df ONLY (not to base CSV) ────────

def apply_integrity_protocol(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """
    Fill gaps in the WORKING DataFrame to ensure downstream indicators have
    no NaN holes. Base CSV (Layer 0) is never modified by this function.

    Macro:  ffill up to 72h (3-day weekends/holidays are expected gaps)
    Crypto:
      - Gaps ≤ 3h : linear interpolation (minor API hiccup)
      - Gaps > 3h : forward-fill (prevents NaN propagation through RSI/MACD/BB)
                    A warning is logged so the user can investigate.
    """
    df = df.set_index('timestamp').sort_index()

    if asset_type == 'macro':
        df = df.ffill(limit=72)  # up to 3 days (weekend + holiday)
    else:
        if len(df) >= 2:
            full_idx = pd.date_range(df.index[0], df.index[-1], freq='1h')
            df = df.reindex(full_idx)

            null_mask  = df['close'].isnull()
            gap_groups = (null_mask != null_mask.shift()).cumsum()
            gap_sizes  = null_mask.groupby(gap_groups).transform('sum')

            small_gap = null_mask & (gap_sizes <= 3)
            large_gap = null_mask & (gap_sizes  > 3)

            # Small gaps: linear interpolation
            df_interp = df.interpolate(method='linear', limit=3)
            df[small_gap] = df_interp[small_gap]

            # Large gaps: forward-fill (NaN would break all downstream indicators)
            if large_gap.any():
                n_large = int(large_gap.sum())
                logger.warning(
                    f'Large data gap: {n_large} missing candles → forward-filled. '
                    f'Investigate source data if this recurs.'
                )
                df = df.ffill()

    df = df.reset_index().rename(columns={'index': 'timestamp'})
    return df


# ── Main update function — 3-layer Snowball flow ──────────────────────────────

def update_ticker(ticker: str, asset_type: str) -> pd.DataFrame | None:
    """
    Incremental self-healing update for a single ticker.

    Layer 0: data/base/{asset_type}/{ticker}.csv  — raw OHLCV (never touched by integrity)
    Layer 1: data/working/{asset_type}/{ticker}_snapshot.csv — enriched working copy
    Returns: working_df (in-memory Layer 1) ready for feature computation.
    Returns None only on catastrophic first-run failure.
    """
    # ── Load Layer 0 ─────────────────────────────────────────────────────────
    raw_base_df = load_base_csv(ticker, asset_type)

    # ── FIRST RUN: Layer 0 empty → download full history from yfinance ───────
    if raw_base_df.empty:
        logger.info(f'[{ticker}] First run — downloading base from yfinance...')
        try:
            if asset_type == 'crypto':
                new_df = fetch_crypto_base_yfinance(ticker)
            else:
                since_dt = datetime.utcnow() - timedelta(days=729)
                new_df = fetch_macro_ohlcv(ticker, since_dt)
        except Exception as e:
            logger.error(f'[{ticker}] Base download failed: {e}')
            return None

        if new_df is None or new_df.empty:
            logger.error(f'[{ticker}] Base download returned empty data.')
            return None

        valid, reason = validate_ohlcv(new_df)
        if not valid:
            logger.warning(f'[{ticker}] Base validation failed: {reason}')
            return None

        save_base_csv(new_df, ticker, asset_type)             # Layer 0
        logger.info(f'[{ticker}] Base saved: {len(new_df)} candles')

        working_df = apply_integrity_protocol(new_df.copy(), asset_type)
        save_snapshot_csv(working_df, ticker, asset_type)     # Layer 1
        _fetch_failure_counts[ticker] = 0
        return working_df

    # ── INCREMENTAL: fetch only new candles ───────────────────────────────────
    last_ts  = pd.Timestamp(raw_base_df['timestamp'].max())
    since_ms = None
    since_dt = None

    if asset_type == 'crypto':
        since_ms = int((last_ts + timedelta(hours=1)).timestamp() * 1000)
    else:
        since_dt = last_ts + timedelta(hours=1)

    new_df = None
    try:
        if asset_type == 'crypto':
            new_df = fetch_crypto_incremental_ccxt(ticker, since_ms)
        else:
            new_df = fetch_macro_ohlcv(ticker, since_dt)
        _fetch_failure_counts[ticker] = 0  # reset on success

    except Exception as e:
        fail_count = _fetch_failure_counts.get(ticker, 0) + 1
        _fetch_failure_counts[ticker] = fail_count
        logger.warning(f'[{ticker}] Fetch failed ({fail_count} consecutive): {e}')
        if fail_count >= 4:
            logger.error(
                f'[{ticker}] 4+ consecutive fetch failures — data gap ≥4h. '
                f'Working DF will ffill the gap via integrity protocol.'
            )

    # Fetch failed → fall back to existing snapshot (or rebuild from raw)
    if new_df is None or new_df.empty:
        if new_df is not None:
            logger.info(f'[{ticker}] No new candles.')
        snapshot = load_snapshot_csv(ticker, asset_type)
        if snapshot is not None:
            return snapshot
        return apply_integrity_protocol(raw_base_df.copy(), asset_type)

    # ── Continuity check: first new candle should be last_ts + 1h ────────────
    expected_next = last_ts + timedelta(hours=1)
    # actual_first  = pd.Timestamp(new_df['timestamp'].iloc[0])

    # ======change start========

    # 1. Clean the actual_first timestamp
    actual_first = pd.Timestamp(new_df['timestamp'].iloc[0]).tz_localize(None)
    # 2. Clean the entire timestamp column (Safer than checking the index)
    if pd.api.types.is_datetime64tz_dtype(new_df['timestamp']):
        new_df['timestamp'] = new_df['timestamp'].dt.tz_convert(None)
    # 3. Clean expected_next
    expected_next_naive = pd.Timestamp(expected_next).tz_localize(None)
    # 4. Now the calculation is safe
    gap_hours = (actual_first - expected_next_naive).total_seconds() / 3600

    # ======change end========

    # gap_hours = (actual_first - expected_next).total_seconds() / 3600
 
    if gap_hours > 1.5:
        logger.warning(
            f'[{ticker}] Gap detected: expected next candle at {expected_next}, '
            f'got {actual_first} ({gap_hours:.0f}h gap)'
        )

    # ── Validate new_df ───────────────────────────────────────────────────────
    valid, reason = validate_ohlcv(new_df)
    if not valid:
        logger.warning(f'[{ticker}] New data validation failed: {reason}. Using existing snapshot.')
        snapshot = load_snapshot_csv(ticker, asset_type)
        if snapshot is not None:
            return snapshot
        return apply_integrity_protocol(raw_base_df.copy(), asset_type)

    # ── Merge into Layer 0 and save (OHLCV only) ─────────────────────────────
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    new_ohlcv  = new_df[[c for c in ohlcv_cols if c in new_df.columns]]
    
    # === ADD THIS LINE TO FIX THE MERGE ERROR ===
    raw_base_df['timestamp'] = pd.to_datetime(raw_base_df['timestamp']).dt.tz_localize(None)

    merged_raw = merge_incremental(raw_base_df, new_ohlcv)
    save_base_csv(merged_raw, ticker, asset_type)            # Layer 0 updated
    logger.info(f'[{ticker}] +{len(new_df)} candles -> {len(merged_raw)} total raw rows')

    # ── Build and save Layer 1 snapshot ──────────────────────────────────────
    working_df = apply_integrity_protocol(merged_raw.copy(), asset_type)
    save_snapshot_csv(working_df, ticker, asset_type)        # Layer 1 updated
    return working_df


# ── Convenience: update all configured tickers ───────────────────────────────

def update_all_tickers() -> dict[str, pd.DataFrame]:
    """Update all crypto + macro tickers. Returns dict ticker → working_df."""
    results = {}
    for ticker in config.CRYPTO_TICKERS:
        df = update_ticker(ticker, 'crypto')
        if df is not None:
            results[ticker] = df

    for name in config.MACRO_TICKERS:
        df = update_ticker(name, 'macro')
        if df is not None:
            results[name] = df

    return results
