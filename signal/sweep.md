Create a complete Python Jupyter Notebook (.ipynb) that implements a crypto trading signal system based on a TradingView Pine Script strategy.

The notebook must be structured cleanly with multiple cells.

---

### 🔹 CELL 1 — CONFIGURATION

Create a config section at the top with:

* List of symbols (example: ["BTC/USDT", "ETH/USDT"])

* Timeframe (example: "1h")

* Exchange (use ccxt, default Binance)

* Strategy parameters:

  * range_lookback = 65
  * vol_len = 21
  * vol_multiplier = 1.5
  * atr_len = 14
  * atr_mult = 2.0
  * min_range_perc = 3.0
  * tp_mode = "Mid Range"

* Telegram config:

  * bot_token
  * chat_id

* General settings:

  * fetch_limit (e.g. 200 candles)
  * polling interval in seconds

---

### 🔹 CELL 2 — IMPORTS

Import required libraries:

* pandas
* numpy
* ccxt
* requests
* time

---

### 🔹 CELL 3 — TELEGRAM FUNCTION

Create a function:

send_telegram(message)

Use requests.post() to send messages via Telegram bot API.

---

### 🔹 CELL 4 — DATA FETCHING

Create a function:

fetch_ohlcv(symbol)

* Use ccxt.binance()
* Fetch OHLCV data
* Convert to pandas DataFrame
* Columns: time, open, high, low, close, volume
* Convert timestamps properly

---

### 🔹 CELL 5 — INDICATORS

Implement:

* ATR (manual calculation using pandas)
* Rolling highest high (rangeHigh)
* Rolling lowest low (rangeLow)
* Volume moving average

---

### 🔹 CELL 6 — STRATEGY LOGIC

Translate the Pine Script logic exactly:

RANGE:

* rangeHigh = rolling max
* rangeLow = rolling min
* rangeSizePerc = (rangeHigh - rangeLow) / close * 100
* validRange = rangeSizePerc > min_range_perc

VOLUME:

* volSpike = volume > vol_ma * vol_multiplier

SWEEPS:

* sweepHigh = high > previous rangeHigh AND close < current rangeHigh
* sweepLow = low < previous rangeLow AND close > current rangeLow

ENTRY:

* longCondition = validRange AND sweepLow AND volSpike
* shortCondition = validRange AND sweepHigh AND volSpike

STOP LOSS:

* baseLongSL = close - ATR * atr_mult
* baseShortSL = close + ATR * atr_mult

TP:

* midRange = (rangeHigh + rangeLow) / 2
* longTP = midRange or rangeHigh
* shortTP = midRange or rangeLow

TRAILING:

* Maintain trailing SL:

  * Long: SL = max(previous SL, new baseLongSL)
  * Short: SL = min(previous SL, new baseShortSL)

---

### 🔹 CELL 7 — SIGNAL GENERATION

Only trigger signal when:

* New long appears (previous candle no position, now longCondition true)
* New short appears

Track state per symbol (dictionary)

---

### 🔹 CELL 8 — MAIN LOOP

Create loop:

while True:

```
for each symbol:
    fetch data
    compute indicators
    evaluate latest candle
    
    if long signal:
        send Telegram:
        "LONG BTC/USDT @ price | SL: X | TP: Y"

    if short signal:
        send Telegram:
        "SHORT BTC/USDT @ price | SL: X | TP: Y"

sleep(polling interval)
```

---

### 🔹 REQUIREMENTS

* Clean, readable code
* Use pandas vectorized operations
* Avoid lookahead bias (use previous candle values properly)
* Only evaluate latest closed candle
* Add basic error handling (try/except per symbol)

---

### 🔹 OPTIONAL (nice to have)

* Print logs in notebook
* Show last signal per symbol
* Prevent duplicate alerts

---

The final output must be a fully working notebook ready to run.
