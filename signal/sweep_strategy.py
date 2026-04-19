import pandas as pd
import numpy as np
import ccxt
import requests
import time
import os
import logging

# Load configuration for this specific strategy
import strategies.sweep_config as config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CELL 3 — TELEGRAM FUNCTION (Self-contained for isolation) ---

_API_BASE = 'https://api.telegram.org/bot{token}/sendMessage'

def _send_telegram_raw(text: str) -> bool:
    """Send a plain-text message with retries and longer timeout."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning('Telegram not configured (missing BOT_TOKEN or CHAT_ID).')
        return False
        
    url = _API_BASE.format(token=config.TELEGRAM_BOT_TOKEN)
    payload = {
        'chat_id':    config.TELEGRAM_CHAT_ID,
        'text':       text,
        'parse_mode': 'HTML',
    }

    max_retries = 3
    initial_delay = 5 # seconds
    for attempt in range(max_retries):
        try:
            # Respect SSL verification setting from config
            verify_ssl = getattr(config, 'TELEGRAM_SSL_VERIFY', True)
            resp = requests.post(url, json=payload, timeout=30, verify=verify_ssl)
            
            if resp.status_code == 429:  # Rate Limited
                retry_after = resp.json().get('parameters', {}).get('retry_after', initial_delay)
                logger.warning(f"Telegram: Rate limited. Sleeping {retry_after}s...")
                time.sleep(retry_after)
                continue

            if not resp.ok:
                err_preview = resp.text[:200] if resp.text else ''
                logger.error(f'Telegram send failed: {resp.status_code} {err_preview}')
                return False
                
            return True

        except requests.exceptions.Timeout:
            logger.warning(f"Telegram: Timeout on attempt {attempt + 1}. Retrying...")
            time.sleep(initial_delay * (attempt + 1)) # Exponential backoff
        except requests.exceptions.RequestException as e:
            logger.error(f'Telegram: Request error: {e}')
            return False

    logger.error("Telegram send failed after max retries due to timeouts/rate limits.")
    return False

def send_telegram(message: str):
    return _send_telegram_raw(message)

# --- CELL 4 — DATA FETCHING (with retry for internet outages) ---

exchange = getattr(ccxt, config.EXCHANGE_ID)()

def fetch_ohlcv(symbol: str, max_retries: int = 5, initial_delay: int = 5) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for a symbol with retry logic to survive internet outages.
    """
    for attempt in range(max_retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, config.TIMEFRAME, limit=config.FETCH_LIMIT)
            df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True)
            return df
        except ccxt.NetworkError as e:
            logger.warning(f"Network error fetching {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(initial_delay * (attempt + 1)) # Exponential backoff
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    logger.error(f"Failed to fetch {symbol} after {max_retries} attempts.")
    return None

# --- CELL 5 — INDICATORS ---

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # ATR (manual calculation)
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close}).max(axis=1)
    df[f'ATR_{config.ATR_LEN}'] = tr.ewm(span=config.ATR_LEN, adjust=False).mean()

    # Rolling highest high (rangeHigh)
    df['rangeHigh'] = df['high'].rolling(window=config.RANGE_LOOKBACK).max()

    # Rolling lowest low (rangeLow)
    df['rangeLow'] = df['low'].rolling(window=config.RANGE_LOOKBACK).min()

    # Volume moving average
    df[f'vol_ma_{config.VOL_LEN}'] = df['volume'].rolling(window=config.VOL_LEN).mean()

    return df

# --- CELL 6 — STRATEGY LOGIC ---

def apply_strategy_logic(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Ensure all required columns are present
    required_cols = [
        'close', 'high', 'low', 'volume',
        f'ATR_{config.ATR_LEN}', 'rangeHigh', 'rangeLow', f'vol_ma_{config.VOL_LEN}'
    ]
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns for strategy logic. Has: {df.columns.tolist()}, Needs: {required_cols}")
        return df

    # Calculate range size and validity
    df['rangeSizePerc'] = (df['rangeHigh'] - df['rangeLow']) / df['close'] * 100
    df['validRange'] = df['rangeSizePerc'] > config.MIN_RANGE_PERC
    df['volSpike'] = df['volume'] > df[f'vol_ma_{config.VOL_LEN}'] * config.VOL_MULTIPLIER

    # Sweep detection (don't look ahead: compare against previous range boundaries)
    df['prev_rangeHigh'] = df['rangeHigh'].shift(1)
    df['prev_rangeLow'] = df['rangeLow'].shift(1)

    df['sweepHigh'] = (df['high'] > df['prev_rangeHigh']) & (df['close'] < df['rangeHigh'])
    df['sweepLow'] = (df['low'] < df['prev_rangeLow']) & (df['close'] > df['rangeLow'])

    # Entry conditions
    df['longCondition'] = df['validRange'] & df['sweepLow'] & df['volSpike']
    df['shortCondition'] = df['validRange'] & df['sweepHigh'] & df['volSpike']

    return df

def get_signal_details(df: pd.DataFrame, symbol: str) -> dict | None:
    if df.empty or len(df) < 2:
        return None

    # We evaluate the last completed candle (index -1)
    last = df.iloc[-1]
    
    # Indicators for display
    atr = last[f'ATR_{config.ATR_LEN}']
    price = last['close']
    
    signal = "NEUTRAL"
    tp, sl = 0.0, 0.0

    if last['longCondition']:
        signal = "LONG"
        sl = price - (atr * config.ATR_MULT)
        tp = (last['rangeHigh'] + last['rangeLow']) / 2 if config.TP_MODE == "Mid Range" else last['rangeHigh']
    elif last['shortCondition']:
        signal = "SHORT"
        sl = price + (atr * config.ATR_MULT)
        tp = (last['rangeHigh'] + last['rangeLow']) / 2 if config.TP_MODE == "Mid Range" else last['rangeLow']

    return {
        'symbol': symbol,
        'time': df.index[-1],
        'price': round(price, 4),
        'range_perc': round(last['rangeSizePerc'], 2),
        'valid_range': last['validRange'],
        'vol_spike': last['volSpike'],
        'signal': signal,
        'tp': round(tp, 4),
        'sl': round(sl, 4)
    }

# --- CELL 8 — MAIN ITERATION HELPER ---

def run_sweep_iteration(symbols: list) -> list:
    """Runs one full pass through symbols and returns a list of signal dicts."""
    results = []
    for symbol in symbols:
        try:
            df = fetch_ohlcv(symbol)
            if df is None:
                continue
                
            df = calculate_indicators(df)
            df = apply_strategy_logic(df)
            details = get_signal_details(df, symbol)
            
            if details:
                results.append(details)
        except Exception as e:
            logger.error(f"Iteration error for {symbol}: {e}")
            
    return results

if __name__ == "__main__":
    # Basic CLI mode if run directly
    print("Starting Sweep Strategy CLI...")
    send_telegram("🚀 Sweep Strategy Started (CLI Mode)")
    
    last_signals = {s: "NEUTRAL" for s in config.SYMBOLS}
    
    while True:
        logger.info("Starting market scan...")
        iteration_results = run_sweep_iteration(config.SYMBOLS)
        
        for res in iteration_results:
            symbol = res['symbol']
            current_sig = res['signal']
            
            if current_sig != "NEUTRAL" and current_sig != last_signals[symbol]:
                msg = (
                    f"🚨 <b>{current_sig} {symbol}</b>\n"
                    f"Price: {res['price']}\n"
                    f"TP: {res['tp']} | SL: {res['sl']}\n"
                    f"Range: {res['range_perc']}% | Vol Spike: {res['vol_spike']}"
                )
                send_telegram(msg)
                last_signals[symbol] = current_sig
                logger.info(f"SIGNAL: {msg.replace('<b>', '').replace('</b>', '')}")
            
        time.sleep(config.POLLING_INTERVAL_SECONDS)