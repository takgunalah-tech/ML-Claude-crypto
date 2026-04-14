"""
Layer A — Snowball DB Engine
Handles all data fetching, validation, integrity, and incremental CSV storage.
"""

import os
import time
import logging
import tempfile
import shutil
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
# ccxt and yfinance are imported lazily inside their fetch functions
# to keep module-load time fast (~5-8s saved on ccxt alone)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ticker_to_filename(ticker: str) -> str:
    """'BTC/USDT' → 'BTC_USDT_1h'"""
    return ticker.replace('/', '_').replace('^', '').replace('-', '_') + f'_{config.TIMEFRAME}'


def _csv_path(ticker: str, asset_type: str) -> str:
    folder = os.path.join(config.DATA_BASE, asset_type)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'{_ticker_to_filename(ticker)}.csv')


# ── Load / Save ───────────────────────────────────────────────────────────────

def load_base_csv(ticker: str, asset_type: str) -> pd.DataFrame:
    """Load existing base CSV. Returns empty DataFrame if file does not exist."""
    path = _csv_path(ticker, asset_type)
    if not os.path.exists(path):
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def save_base_csv(df: pd.DataFrame, ticker: str, asset_type: str) -> None:
    """Atomically overwrite CSV (write to temp → rename) for crash safety."""
    path = _csv_path(ticker, asset_type)
    df_out = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df_out['timestamp'] = df_out['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    tmp = path + '.tmp'
    df_out.to_csv(tmp, index=False)
    shutil.move(tmp, path)


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


# ── Exchange instance cache (markets loaded once, shared across all tickers) ──

_exchange_cache: dict = {}

def _get_exchange(exchange_id: str):
    """Return a cached ccxt exchange instance (avoids reloading markets per ticker)."""
    if exchange_id not in _exchange_cache:
        import ccxt  # lazy — ccxt takes 5-8s to import
        ex = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'timeout': 30000,          # 30s (default 10s causes timeouts on slow connections)
            'options': {'defaultType': 'spot'},
        })
        _exchange_cache[exchange_id] = ex
    return _exchange_cache[exchange_id]


def check_exchange_connectivity(exchange_id: str = None) -> tuple[bool, str]:
    """Quick connectivity test — tries to ping the exchange. Returns (ok, message)."""
    exchange_id = exchange_id or config.EXCHANGE
    try:
        ex = _get_exchange(exchange_id)
        ex.load_markets()
        return True, f'{exchange_id} reachable ({len(ex.markets)} markets loaded)'
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


# ── Fetch: Crypto (ccxt / Binance) ───────────────────────────────────────────

def fetch_crypto_ohlcv(
    ticker: str,
    since_ms: int,
    exchange_id: str = None,
) -> pd.DataFrame:
    """
    Paginated fetch from Binance via ccxt.
    since_ms: epoch milliseconds (fetch from this timestamp forward).
    Exchange instance is cached so markets are loaded only once across all tickers.
    """
    exchange_id = exchange_id or config.EXCHANGE
    exchange = _get_exchange(exchange_id)
    all_rows = []
    limit = 1000
    current_since = since_ms

    while True:
        try:
            candles = exchange.fetch_ohlcv(
                ticker, config.TIMEFRAME, since=current_since, limit=limit
            )
        except Exception as e:
            raise RuntimeError(
                f'ccxt {type(e).__name__} for {ticker}: {e}'
            ) from e

        if not candles:
            break
        all_rows.extend(candles)
        last_ts = candles[-1][0]
        if len(candles) < limit:
            break
        current_since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(all_rows, columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume'])
    # tz_convert(None) strips UTC timezone → tz-naive UTC values
    df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True).dt.tz_convert(None)
    df = df.drop(columns='ts_ms')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


# ── Fetch: Macro (yfinance) ───────────────────────────────────────────────────

def fetch_macro_ohlcv(ticker: str, start_dt: datetime) -> pd.DataFrame:
    """
    Download hourly OHLCV from yfinance starting at start_dt.
    yfinance 1h data is available for ~730 days.

    Handles both old yfinance (flat columns) and new yfinance >= 0.2
    (MultiIndex columns like ('Close', 'SPY')).
    """
    import yfinance as yf  # lazy import
    yf_ticker = config.MACRO_TICKERS.get(ticker, ticker)
    end_dt = datetime.utcnow() + timedelta(days=1)
    raw = yf.download(
        yf_ticker,
        start=start_dt.strftime('%Y-%m-%d'),
        end=end_dt.strftime('%Y-%m-%d'),
        interval='1h',
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # ── Flatten MultiIndex columns (yfinance >= 0.2 returns ('Close','SPY') etc.) ──
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]  # keep price-type level only

    # ── Move DatetimeIndex into a plain column ────────────────────────────────
    raw = raw.reset_index()

    # Detect timestamp column (may be 'Datetime', 'Date', or 'index')
    ts_col = next(
        (c for c in raw.columns if str(c).lower() in ('datetime', 'date', 'index')),
        raw.columns[0]   # fallback: first column
    )

    # Normalise all column names to lowercase
    raw.columns = [str(c).lower() for c in raw.columns]
    ts_col = str(ts_col).lower()
    raw = raw.rename(columns={ts_col: 'timestamp'})

    # ── Strip timezone → tz-naive UTC ────────────────────────────────────────
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    if raw['timestamp'].dt.tz is not None:
        raw['timestamp'] = raw['timestamp'].dt.tz_convert(None)

    # Fill NaN volume with 0 (VIX and some indices have no trading volume)
    raw['volume'] = raw.get('volume', pd.Series(0, index=raw.index)).fillna(0)

    df = raw[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df[df['timestamp'] >= pd.Timestamp(start_dt)].reset_index(drop=True)
    return df


# ── Integrity Protocol ────────────────────────────────────────────────────────

def apply_integrity_protocol(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """
    Macro  → forward-fill gaps up to 4 candles (weekends/holidays are valid).
    Crypto → interpolate gaps ≤ 3 candles; preserve larger gaps as NaN.
    """
    df = df.set_index('timestamp').sort_index()

    if asset_type == 'macro':
        df = df.ffill(limit=4)
    else:
        # Build a full hour-by-hour index to expose missing rows
        if len(df) >= 2:
            full_idx = pd.date_range(df.index[0], df.index[-1], freq='1h')
            df = df.reindex(full_idx)
            # Find gap lengths
            null_mask = df['close'].isnull()
            gap_sizes = null_mask.groupby((null_mask != null_mask.shift()).cumsum()).transform('sum')
            small_gap = null_mask & (gap_sizes <= 3)
            # Interpolate only small gaps
            df_interp = df.interpolate(method='linear', limit=3)
            df[small_gap] = df_interp[small_gap]
            # Large gaps remain NaN

    df = df.reset_index().rename(columns={'index': 'timestamp'})
    return df


# ── Main update function ──────────────────────────────────────────────────────

def update_ticker(ticker: str, asset_type: str) -> pd.DataFrame | None:
    """
    Incremental self-healing update for a single ticker.

    1. Load existing CSV → determine last saved timestamp.
    2. Fetch only new candles (since last saved ts).
    3. Validate downloaded data.
    4. Merge, apply integrity protocol, save.
    5. Return merged in-memory DataFrame (ready to use immediately).

    Returns None if validation fails (caller should skip this ticker).
    """
    base_df = load_base_csv(ticker, asset_type)

    # Determine fetch start
    if base_df.empty:
        # First run: fetch full history (up to exchange/API limits)
        if asset_type == 'crypto':
            # ~2 years of 1h data
            since_dt = datetime.utcnow() - timedelta(days=730)
            since_ms = int(since_dt.timestamp() * 1000)
        else:
            since_dt = datetime.utcnow() - timedelta(days=729)
    else:
        last_ts = pd.Timestamp(base_df['timestamp'].max())
        if asset_type == 'crypto':
            since_ms = int((last_ts + timedelta(hours=1)).timestamp() * 1000)
        else:
            since_dt = last_ts + timedelta(hours=1)

    # Fetch
    try:
        if asset_type == 'crypto':
            new_df = fetch_crypto_ohlcv(ticker, since_ms)
        else:
            new_df = fetch_macro_ohlcv(ticker, since_dt)
    except Exception as e:
        logger.warning(f'[{ticker}] fetch failed: {e}. Will retry next run.')
        return None

    # If nothing new, just return existing data
    if new_df.empty:
        logger.info(f'[{ticker}] No new candles.')
        return base_df if not base_df.empty else None

    # Validate new data
    valid, reason = validate_ohlcv(new_df)
    if not valid:
        logger.warning(f'[{ticker}] Validation failed: {reason}. Skipping save.')
        return base_df if not base_df.empty else None

    # Merge + integrity + save
    merged = merge_incremental(base_df, new_df)
    merged = apply_integrity_protocol(merged, asset_type)
    save_base_csv(merged, ticker, asset_type)

    n_new = len(new_df)
    logger.info(f'[{ticker}] +{n_new} candles → {len(merged)} total rows saved.')
    return merged


def update_all_tickers() -> dict[str, pd.DataFrame]:
    """Update all crypto + macro tickers. Returns dict of loaded DataFrames."""
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
