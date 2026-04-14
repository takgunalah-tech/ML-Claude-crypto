from dotenv import load_dotenv
import os

load_dotenv()

# ── Tickers ───────────────────────────────────────────────────────────────────
CRYPTO_TICKERS = [
    'BTC/USDT', 'ETH/USDT',
    'SOL/USDT', 'ADA/USDT', 'AVAX/USDT',
    'LINK/USDT', 'DOT/USDT', 'MATIC/USDT',
]
ANCHOR_TICKERS  = ['BTC/USDT', 'ETH/USDT']
ALTCOIN_TICKERS = [t for t in CRYPTO_TICKERS if t not in ANCHOR_TICKERS]

MACRO_TICKERS = {
    'SPY': 'SPY',
    'QQQ': 'QQQ',
    'GLD': 'GLD',
    'TLT': 'TLT',
    'VIX': '^VIX',
    'DXY': 'DX-Y.NYB',
}

EXCHANGE  = 'binance'
TIMEFRAME = '1h'

# ── Data paths ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_BASE    = os.path.join(BASE_DIR, 'data', 'base')
DATA_WORKING = os.path.join(BASE_DIR, 'data', 'working')
MODELS_DIR   = os.path.join(BASE_DIR, 'models', 'altcoins')
STATE_DIR    = os.path.join(BASE_DIR, 'state')
LOGS_DIR     = os.path.join(BASE_DIR, 'logs')

# ── Data splits (days from today) ─────────────────────────────────────────────
TRAIN_END_DAYS = 85       # training data ends 85 days ago
TEST1          = (85, 31) # test set 1: 85→31 days ago
TEST2          = (30, 1)  # test set 2: 30→1 days ago (true OOS)

# ── Labeling ──────────────────────────────────────────────────────────────────
LABEL_HORIZON = 48        # candles to scan for TP/SL hit (48h)

# Grid search space for adaptive label optimisation
TP_GRID  = [0.020, 0.025, 0.030, 0.035, 0.040, 0.050]
SL_GRID  = [0.010, 0.015, 0.020, 0.025]
K1_GRID  = [0.3, 0.5, 0.7]   # ATR multiplier for TP
K2_GRID  = [0.2, 0.3, 0.4]   # ATR multiplier for SL

# ── Model validity gates ──────────────────────────────────────────────────────
MIN_TRADE_COUNT    = 20
MIN_PF_TEST1       = 1.05
PF_STABILITY_RATIO = 0.70   # PF(test2) / PF(test1) must be >= this

# ── Signal thresholds ─────────────────────────────────────────────────────────
LONG_THRESHOLD  = 0.60   # P(win) >= 0.60 → LONG
SHORT_THRESHOLD = 0.40   # P(win) <= 0.40 → SHORT (equiv. 1-P >= 0.60)

# ── Position sizing ───────────────────────────────────────────────────────────
MAX_LOSS_USDT = 5.0       # max loss per trade in USDT

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN  = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID    = os.getenv('TELEGRAM_CHAT_ID', '')
MORNING_REPORT_HOUR = 7   # 7am local time

# ── Scheduler ─────────────────────────────────────────────────────────────────
SCHEDULER_INTERVAL_MIN  = 60
RETRAIN_INTERVAL_HOURS  = 72
