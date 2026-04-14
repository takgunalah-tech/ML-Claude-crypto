"""
Layer B — Feature Intelligence System

Model A : Macro Context  → Macro_Risk_State ∈ [-1, 1]
Model B : Anchor (BTC/ETH) → regime, volatility, breakout, direction features
Model C : Altcoin feature matrix (standard TA + derived + cross-asset injection)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# pandas_ta is imported lazily on first use (~2s saved on module load)
_ta = None
def _get_ta():
    global _ta
    if _ta is None:
        import pandas_ta as _pandas_ta
        _ta = _pandas_ta
    return _ta

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    return a.div(b.replace(0, np.nan)).fillna(fill)


def _clip_norm(s: pd.Series, lo: float = -1.0, hi: float = 1.0) -> pd.Series:
    """Scale series to [lo, hi]."""
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(0.0, index=s.index)
    scaled = (s - s.min()) / rng * (hi - lo) + lo
    return scaled.clip(lo, hi)


def _ta_col(df: pd.DataFrame, prefix: str) -> pd.Series:
    """
    Return the first column whose name starts with `prefix`.
    Robust against pandas_ta column naming changes across versions.
    Example: _ta_col(bb, 'BBU') → Series for upper Bollinger Band
    """
    matches = [c for c in df.columns if c.startswith(prefix)]
    if not matches:
        raise ValueError(
            f"No column starting with {prefix!r} found in pandas_ta output. "
            f"Available columns: {list(df.columns)}"
        )
    return df[matches[0]]


# ── Model A — Macro Risk State ────────────────────────────────────────────────

def compute_macro_risk_state(macro_dfs: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Aggregate macro assets into a single Macro_Risk_State ∈ [-1, 1].

    Steps:
      1. Per asset: compute 1h/5h/20h returns, rolling vol (20), price z-score (60)
      2. Align on common hourly index (ffill ≤ 8 to bridge weekends/holidays)
      3. PCA(1) on the stacked feature matrix → first principal component
      4. Normalise to [-1, 1]
    """
    feature_blocks = []
    common_idx = None

    for name, df in macro_dfs.items():
        df = df.set_index('timestamp').sort_index()['close'].copy()

        ret1  = df.pct_change(1)
        ret5  = df.pct_change(5)
        ret20 = df.pct_change(20)
        rvol  = df.pct_change(1).rolling(20).std()
        zmean = df.rolling(60).mean()
        zstd  = df.rolling(60).std().replace(0, np.nan)
        zscore = (df - zmean) / zstd

        block = pd.DataFrame({
            f'{name}_ret1':   ret1,
            f'{name}_ret5':   ret5,
            f'{name}_ret20':  ret20,
            f'{name}_rvol20': rvol,
            f'{name}_z60':    zscore,
        })
        feature_blocks.append(block)
        common_idx = block.index if common_idx is None else common_idx.union(block.index)

    if not feature_blocks:
        return pd.Series(dtype=float, name='Macro_Risk_State')

    # ffill up to 8h BEFORE dropna to avoid dropping entire rows due to a single
    # asset's weekend/holiday gap (e.g. VIX has no Friday-evening data)
    aligned = [b.reindex(common_idx).ffill(limit=8) for b in feature_blocks]
    matrix  = pd.concat(aligned, axis=1).ffill(limit=8).dropna()

    if matrix.empty or matrix.shape[0] < 2:
        return pd.Series(0.0, index=common_idx, name='Macro_Risk_State')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(matrix)
    pca    = PCA(n_components=1)
    pc1    = pca.fit_transform(scaled).flatten()

    state = pd.Series(pc1, index=matrix.index, name='Macro_Risk_State')
    # Re-normalise to [-1, 1] in case PCA output exceeds it
    state = _clip_norm(state)
    # Reindex to full common_idx (fill any residual gaps)
    state = state.reindex(common_idx).ffill(limit=4).bfill(limit=1)
    state.name = 'Macro_Risk_State'
    return state


# ── Model B — Anchor Intelligence (BTC / ETH) ────────────────────────────────

def compute_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily classic pivot points and distances from close."""
    # Resample to daily to get previous-day OHLC
    daily = df.set_index('timestamp').resample('1D').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    ).shift(1)  # previous day

    daily['P']  = (daily['high'] + daily['low'] + daily['close']) / 3
    daily['R1'] = 2 * daily['P'] - daily['low']
    daily['S1'] = 2 * daily['P'] - daily['high']
    daily['R2'] = daily['P'] + (daily['high'] - daily['low'])
    daily['S2'] = daily['P'] - (daily['high'] - daily['low'])

    # Reindex to hourly using .ffill() — reindex(method='ffill') is deprecated
    df_idx  = df.set_index('timestamp')
    pivots  = daily[['P', 'R1', 'R2', 'S1', 'S2']].reindex(df_idx.index).ffill()

    close     = df_idx['close']
    nearest_R = pivots[['R1', 'R2']].min(axis=1)
    nearest_S = pivots[['S1', 'S2']].max(axis=1)

    out = pd.DataFrame(index=df_idx.index)
    out['pivot_P']            = _safe_div(close - pivots['P'], close)
    out['dist_to_resistance'] = _safe_div(nearest_R - close, close).clip(0)
    out['dist_to_support']    = _safe_div(close - nearest_S, close).clip(0)
    return out.reset_index()


def compute_anchor_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Full anchor feature set for BTC or ETH.
    Returns a DataFrame indexed by timestamp with columns prefixed by `prefix`.

    All pandas_ta outputs are accessed by column-name prefix (via _ta_col) to be
    robust against library version changes in column ordering.
    """
    df    = df.set_index('timestamp').sort_index().copy()
    close = df['close']
    high  = df['high']
    low   = df['low']
    vol   = df['volume']

    out = pd.DataFrame(index=df.index)

    # ── Regime ────────────────────────────────────────────────────────────────
    adx_df = _get_ta().adx(high, low, close, length=14)
    adx    = _ta_col(adx_df, 'ADX')
    out[f'{prefix}_ADX'] = adx / 100.0          # normalise to [0, 1]

    ema50  = _get_ta().ema(close, length=50)
    ema200 = _get_ta().ema(close, length=200)
    cross  = _safe_div(ema50 - ema200, close).clip(-0.05, 0.05)
    out[f'{prefix}_EMA_cross'] = cross / 0.05   # → [-1, 1]

    bb      = _get_ta().bbands(close, length=20)
    bb_upper = _ta_col(bb, 'BBU')
    bb_lower = _ta_col(bb, 'BBL')
    bb_mid   = _ta_col(bb, 'BBM')
    bb_width = _safe_div(bb_upper - bb_lower, bb_mid)
    out[f'{prefix}_BB_width'] = bb_width

    # Regime score: positive = trend, negative = range
    out[f'{prefix}_Regime'] = out[f'{prefix}_EMA_cross'] * (1 + out[f'{prefix}_ADX'])
    out[f'{prefix}_Regime'] = _clip_norm(out[f'{prefix}_Regime'])

    # ── Volatility State ──────────────────────────────────────────────────────
    atr14  = _get_ta().atr(high, low, close, length=14)
    atr_ma = atr14.rolling(50).mean()
    rel_atr = _safe_div(atr14, atr_ma.replace(0, np.nan))

    log_ret = np.log(close / close.shift(1))
    hv20    = log_ret.rolling(20).std() * np.sqrt(8760)
    hv60    = log_ret.rolling(60).std() * np.sqrt(8760)
    hv_ratio = _safe_div(hv20, hv60.replace(0, np.nan))

    out[f'{prefix}_ATR_rel']   = rel_atr
    out[f'{prefix}_HV_ratio']  = hv_ratio
    out[f'{prefix}_Vol_State'] = _clip_norm(rel_atr * hv_ratio)

    # ── Breakout / Transition Probability ─────────────────────────────────────
    roll_high   = high.rolling(20).max()
    roll_low    = low.rolling(20).min()
    dist_to_high = _safe_div(roll_high - close, close).clip(0)
    dist_to_low  = _safe_div(close - roll_low, close).clip(0)
    vol_surge    = _safe_div(vol, vol.rolling(20).mean())

    breakout_raw = vol_surge * (1 - dist_to_high * 10).clip(0, 1)
    out[f'{prefix}_Breakout_Prob']     = _clip_norm(breakout_raw, 0, 1)
    out[f'{prefix}_Continuation_Prob'] = (out[f'{prefix}_ADX'] * (1 - dist_to_high * 5).clip(0, 1))
    out[f'{prefix}_Continuation_Prob'] = _clip_norm(out[f'{prefix}_Continuation_Prob'], 0, 1)

    # ── Directional Bias ──────────────────────────────────────────────────────
    rsi14 = _get_ta().rsi(close, length=14)
    out[f'{prefix}_RSI_bias'] = ((rsi14 - 50) / 50).clip(-1, 1)

    macd_df   = _get_ta().macd(close, fast=12, slow=26, signal=9)
    macd_hist = _ta_col(macd_df, 'MACDh')
    macd_norm = _safe_div(macd_hist, close) * 1000
    out[f'{prefix}_MACD_bias'] = _clip_norm(macd_norm)

    ema9  = _get_ta().ema(close, length=9)
    ema21 = _get_ta().ema(close, length=21)
    out[f'{prefix}_EMA9_21_bias'] = _safe_div(ema9 - ema21, close).clip(-0.02, 0.02) / 0.02

    out[f'{prefix}_Direction_Bias'] = _clip_norm(
        out[f'{prefix}_RSI_bias'] + out[f'{prefix}_MACD_bias'] + out[f'{prefix}_EMA9_21_bias']
    )

    out = out.reset_index()
    return out


# ── Model C — Altcoin Feature Matrix ─────────────────────────────────────────

def compute_altcoin_features(
    df: pd.DataFrame,
    btc_anchor: pd.DataFrame,
    eth_anchor: pd.DataFrame,
    macro_state: pd.Series,
) -> pd.DataFrame:
    """
    Full feature matrix for one altcoin.
    Includes standard TA, derived features, pivot points, and cross-asset injection.
    """
    df    = df.set_index('timestamp').sort_index().copy()
    close = df['close']
    high  = df['high']
    low   = df['low']
    vol   = df['volume']

    out = pd.DataFrame(index=df.index)

    # ── Standard Indicators ───────────────────────────────────────────────────
    for length in [7, 14, 21]:
        out[f'RSI_{length}'] = _get_ta().rsi(close, length=length)

    macd_df = _get_ta().macd(close, fast=12, slow=26, signal=9)
    out['MACD']      = _ta_col(macd_df, 'MACD_')   # line (not MACDh / MACDs)
    out['MACD_sig']  = _ta_col(macd_df, 'MACDs')
    out['MACD_hist'] = _ta_col(macd_df, 'MACDh')

    bb = _get_ta().bbands(close, length=20)
    bb_upper = _ta_col(bb, 'BBU')
    bb_lower = _ta_col(bb, 'BBL')
    bb_mid   = _ta_col(bb, 'BBM')
    bb_pct   = _ta_col(bb, 'BBP')   # %b — some versions use BBP, others BBB_pct
    out['BB_upper'] = bb_upper
    out['BB_lower'] = bb_lower
    out['BB_mid']   = bb_mid
    out['BB_width'] = _safe_div(bb_upper - bb_lower, bb_mid)
    out['BB_pct']   = bb_pct

    for length in [9, 21, 50, 200]:
        out[f'EMA_{length}'] = _get_ta().ema(close, length=length)
    for length in [20, 50]:
        out[f'SMA_{length}'] = _get_ta().sma(close, length=length)

    atr14 = _get_ta().atr(high, low, close, length=14)
    out['ATR_14']      = atr14
    out['ATR_14_norm'] = _safe_div(atr14, close)

    adx_df = _get_ta().adx(high, low, close, length=14)
    out['ADX_14']  = _ta_col(adx_df, 'ADX')
    out['DI_plus'] = _ta_col(adx_df, 'DMP')
    out['DI_minus']= _ta_col(adx_df, 'DMN')

    # VWAP deviation (reset daily)
    typical    = (high + low + close) / 3
    cum_tp_vol = (typical * vol).groupby(df.index.date).cumsum()
    cum_vol    = vol.groupby(df.index.date).cumsum()
    vwap       = cum_tp_vol / cum_vol.replace(0, np.nan)
    out['VWAP_dev'] = _safe_div(close - vwap, close)

    out['Vol_ratio'] = _safe_div(vol, vol.rolling(20).mean())

    # ── Pivot Points ──────────────────────────────────────────────────────────
    pivot_df = compute_pivot_points(df.reset_index())
    pivot_df = pivot_df.set_index('timestamp')
    for col in ['pivot_P', 'dist_to_resistance', 'dist_to_support']:
        if col in pivot_df.columns:
            out[col] = pivot_df[col]

    # ── Derived Features ──────────────────────────────────────────────────────
    out['RSI_momentum']  = out['RSI_14'].diff(3)
    out['ATR_expansion'] = _safe_div(atr14, atr14.rolling(20).mean())
    out['MR_strength']   = _safe_div(
        close - out['SMA_20'],
        out['BB_width'].replace(0, np.nan)
    )
    out['EMA_stack'] = (
        (out['EMA_9']  > out['EMA_21']).astype(int) +
        (out['EMA_21'] > out['EMA_50']).astype(int) +
        (out['EMA_50'] > out['EMA_200']).astype(int)
    ).astype(float) - 1.5   # centre around 0

    # ── Time-of-day features (derived from tz-naive UTC index, no tz storage) ─
    # These capture intra-day and session patterns without storing raw timezone.
    hour = pd.Series(out.index.hour, index=out.index, dtype=float)
    out['Hour_UTC'] = hour   # 0–23

    def _session(h: float) -> float:
        if  0 <= h <  7: return 0.0   # Asia
        if  7 <= h < 13: return 1.0   # Europe
        if 13 <= h < 22: return 2.0   # US
        return 3.0                     # Off-hours / overnight

    out['Session'] = hour.map(_session)

    # ── Cross-Asset Injection ─────────────────────────────────────────────────
    btc_cols = [c for c in btc_anchor.columns if c != 'timestamp']
    eth_cols = [c for c in eth_anchor.columns if c != 'timestamp']

    btc_idx = btc_anchor.set_index('timestamp')[btc_cols]
    eth_idx = eth_anchor.set_index('timestamp')[eth_cols]

    out = out.join(btc_idx, how='left')
    out = out.join(eth_idx, how='left')

    # Forward-fill to handle minor timestamp misalignment (1-2 candle delay)
    btc_eth_cols = btc_cols + eth_cols
    out[btc_eth_cols] = out[btc_eth_cols].ffill(limit=2)

    if isinstance(macro_state, pd.Series):
        macro_aligned = macro_state.reindex(out.index).ffill(limit=4)
        out['Macro_Risk_State'] = macro_aligned

    out = out.reset_index()
    return out


# ── Build all feature matrices ─────────────────────────────────────────────

def build_all_features(
    crypto_dfs: dict[str, pd.DataFrame],
    macro_dfs:  dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Compute features for all altcoins.
    Returns dict: ticker → feature DataFrame.
    """
    macro_state = compute_macro_risk_state(macro_dfs)

    btc_df = crypto_dfs.get('BTC/USDT')
    eth_df = crypto_dfs.get('ETH/USDT')
    if btc_df is None or eth_df is None:
        raise ValueError('BTC/USDT and ETH/USDT are required anchor assets.')

    btc_anchor = compute_anchor_features(btc_df, 'BTC')
    eth_anchor = compute_anchor_features(eth_df, 'ETH')

    feature_dfs = {}
    import config
    for ticker in config.ALTCOIN_TICKERS:
        if ticker not in crypto_dfs:
            continue
        df   = crypto_dfs[ticker]
        feat = compute_altcoin_features(df, btc_anchor, eth_anchor, macro_state)
        feature_dfs[ticker] = feat

    return feature_dfs
