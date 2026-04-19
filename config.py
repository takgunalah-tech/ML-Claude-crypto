from dotenv import load_dotenv
import os

load_dotenv()

# ── Tickers ───────────────────────────────────────────────────────────────────
# CRYPTO_TICKERS = [
#     'BTC/USDT', 'ETH/USDT', 'TRX/USDT', 'FIL/USDT', 'BCH/USDT', 'ZEC/USDT', 
#     "XRP/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT", "DOGE/USDT", "TRX/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", 
#     "NEAR/USDT", "APT/USDT", "SUI/USDT", "ARB/USDT", "LTC/USDT", "INJ/USDT", "HBAR/USDT", "AAVE/USDT", "ONDO/USDT",
# ]
CRYPTO_TICKERS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'TRX/USDT:USDT', 'FIL/USDT:USDT', 'BCH/USDT:USDT', 
    'ZEC/USDT:USDT', 'XRP/USDT:USDT', 'SOL/USDT:USDT', 'ADA/USDT:USDT', 'DOGE/USDT:USDT', 'AVAX/USDT:USDT', 
    'AAVE/USDT:USDT', 'LINK/USDT:USDT', 'DOT/USDT:USDT', 'NEAR/USDT:USDT', 'APT/USDT:USDT', 'SUI/USDT:USDT', 
    'ARB/USDT:USDT', 'LTC/USDT:USDT', 'INJ/USDT:USDT', 'HBAR/USDT:USDT',  'ONDO/USDT:USDT'
]

CRYPTO_YF_MAP = {
    'BTC/USDT:USDT':  'BTC-USD',
    'ETH/USDT:USDT':  'ETH-USD',
    'BNB/USDT:USDT':  'BNB-USD',
    'TRX/USDT:USDT':  'TRX-USD',
    'FIL/USDT:USDT':  'FIL-USD',
    'BCH/USDT:USDT':  'BCH-USD',
    'ZEC/USDT:USDT':  'ZEC-USD',
    'XRP/USDT:USDT':  'XRP-USD',
    'SOL/USDT:USDT':  'SOL-USD',
    'ADA/USDT:USDT':  'ADA-USD',
    'DOGE/USDT:USDT': 'DOGE-USD',
    'AVAX/USDT:USDT': 'AVAX-USD',
    'LINK/USDT:USDT': 'LINK-USD',
    'DOT/USDT:USDT':  'DOT-USD',
    'NEAR/USDT:USDT': 'NEAR-USD',
    'APT/USDT:USDT':  'APT21794-USD',
    'SUI/USDT:USDT':  'SUI20947-USD',
    'ARB/USDT:USDT':  'ARB11841-USD',
    'LTC/USDT:USDT':  'LTC-USD',
    'INJ/USDT:USDT':  'INJ-USD',
    'HBAR/USDT:USDT': 'HBAR-USD',
    'AAVE/USDT:USDT': 'AAVE-USD',
    'ONDO/USDT:USDT': 'ONDO-USD'
}

ANCHOR_TICKERS  = ['BTC/USDT:USDT', 'ETH/USDT:USDT']
ALTCOIN_TICKERS = [t for t in CRYPTO_TICKERS if t not in ANCHOR_TICKERS]

# Macro tickers: key = internal name, value = yfinance symbol
MACRO_TICKERS = {
    'SPX': '^GSPC',   # S&P 500 Index (not SPY ETF — ^GSPC is the actual index)
    'QQQ': 'QQQ',     # Nasdaq 100 ETF — works for 1h
    'GLD': 'GC=F',    # Gold Futures (not GLD ETF — GLD has market-hours gaps)
    'TLT': 'TLT',     # 20-Year Treasury ETF — works for 1h
    'VIX': '^VIX',    # Volatility Index — volume will be 0
    'DXY': 'DX-Y.NYB',    # US Dollar Index Futures (DX-Y.NYB was unreliable)
}

# yfinance symbols for BASE historical download of crypto (~729 days of 1h data)
# CRYPTO_YF_MAP = {
#     'BTC/USDT':  'BTC-USD', 'ETH/USDT':  'ETH-USD', 'XRP/USDT':  'XRP-USD',
#     'BNB/USDT':  'BNB-USD', 'SOL/USDT':  'SOL-USD', 'ADA/USDT':  'ADA-USD',
#     'DOGE/USDT': 'DOGE-USD', 'TRX/USDT':  'TRX-USD', 'AVAX/USDT': 'AVAX-USD',
#     'LINK/USDT': 'LINK-USD', 'DOT/USDT':  'DOT-USD', 'NEAR/USDT': 'NEAR-USD',
#     'APT/USDT':  'APT-USD', 'SUI/USDT':  'SUI-USD', 'ARB/USDT':  'ARB-USD',
#     'LTC/USDT':  'LTC-USD', 'INJ/USDT':  'INJ-USD', 'HBAR/USDT': 'HBAR-USD',
#     'AAVE/USDT': 'AAVE-USD', 'ONDO/USDT': 'ONDO-USD', 'FIL/USDT':  'FIL-USD',
#     'BCH/USDT':  'BCH-USD', 'ZEC/USDT':  'ZEC-USD'
# }

# Exchange config
EXCHANGE_SPOT    = 'binance'       # spot OHLCV incremental updates via ccxt
EXCHANGE_FUTURES = 'binanceusdm'   # USDT-margined futures (funding rate, OI — future use)
EXCHANGE         = EXCHANGE_FUTURES # Global alias for primary data source
TIMEFRAME        = '1h'

# ── Data paths ────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_BASE     = os.path.join(BASE_DIR, 'data', 'base')      # Layer 0: raw OHLCV CSVs
DATA_WORKING  = os.path.join(BASE_DIR, 'data', 'working')   # Layer 1: snapshot CSVs + parquets
MODELS_DIR    = os.path.join(BASE_DIR, 'models', 'altcoins')
STATE_DIR     = os.path.join(BASE_DIR, 'state')
LOGS_DIR      = os.path.join(BASE_DIR, 'logs')

# ── Data splits (days from today) ─────────────────────────────────────────────
TRAIN_END_DAYS = 85       # training data ends 85 days ago
TEST1          = (85, 31) # test set 1: 85→31 days ago  (exclusive start, inclusive end)
TEST2          = (31,  1) # test set 2: 31→1 days ago   (contiguous with TEST1 — no gap)

# ── Labeling ──────────────────────────────────────────────────────────────────
LABEL_HORIZON = 48        # candles to scan for TP/SL hit (48h look-ahead)

# Grid search space for adaptive label optimisation
TP_GRID  = [0.020, 0.025, 0.030, 0.035, 0.040, 0.050]
SL_GRID  = [0.010, 0.015, 0.020, 0.025]
K1_GRID  = [0.3, 0.5, 0.7]   # ATR multiplier for TP
K2_GRID  = [0.2, 0.3, 0.4]   # ATR multiplier for SL

# ── Model validity gates ──────────────────────────────────────────────────────
MIN_TRADE_COUNT    = 20
MIN_PF_TEST1       = 1.20   # raised from 1.05 — must clear breakeven + fees
PF_STABILITY_RATIO = 0.70   # PF(test2) / PF(test1) must be >= this

# ── Signal thresholds ─────────────────────────────────────────────────────────
LONG_THRESHOLD  = 0.60   # P(win) >= 0.60 → fire LONG
SHORT_THRESHOLD = 0.40   # P(win) <= 0.40 → fire SHORT  (dead zone: 0.40–0.60)

# ── Signal lifecycle ──────────────────────────────────────────────────────────
MAX_SIGNAL_REPEATS = 1   # how many times a signal can repeat before being archived

# ── Position sizing ───────────────────────────────────────────────────────────
MAX_LOSS_USDT = 5.0       # max loss per trade in USDT

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN  = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID    = os.getenv('TELEGRAM_CHAT_ID', '')
MORNING_REPORT_HOUR = 0   # 00:00 UTC is 8:00 AM GMT+8

# ── Scheduler ─────────────────────────────────────────────────────────────────
SCHEDULER_INTERVAL_MIN  = 60
RETRAIN_INTERVAL_HOURS  = 72

# ── Training performance ───────────────────────────────────────────────────────
# Grid search uses a lightweight model (fast ranking), final model uses full depth.
# Set to None to use FINAL_ESTIMATORS for both (disables two-stage training).
GRID_SEARCH_ESTIMATORS  = 50   # n_estimators used during grid search (ranking pass)
FINAL_ESTIMATORS        = 300  # n_estimators used for the production model
