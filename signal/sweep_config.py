import os

# ── General Settings ──────────────────────────────────────────────────────────
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BCH/USDT", "XRP/USDT",
    "CRV/USDT", "ASTER/USDT", "HYPE/USDT", "FIL/USDT", "XMR/USDT", "ONE/USDT"
]
TIMEFRAME = "1h"
EXCHANGE_ID = "binance"
FETCH_LIMIT = 200 # Number of candles to fetch for analysis
POLLING_INTERVAL_SECONDS = 300 # 5 minutes

# ── Strategy Parameters ───────────────────────────────────────────────────────
RANGE_LOOKBACK = 65
VOL_LEN = 21
VOL_MULTIPLIER = 1.5
ATR_LEN = 14
ATR_MULT = 2.0
MIN_RANGE_PERC = 3.0
TP_MODE = "Mid Range" # "Mid Range" or "Range High/Low"

# ── Telegram Configuration ────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv('SWEEP_TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('SWEEP_TELEGRAM_CHAT_ID', '')

# For testing the timeout fix, setting this to False might be useful
# if you're experiencing SSL certificate issues on your environment.
# It's recommended to keep it True in production if possible.
TELEGRAM_SSL_VERIFY = True 