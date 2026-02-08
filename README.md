# Alpaca Trader

Advanced algorithmic trading bot built for Alpaca Markets.

Designed for automated strategy execution, multi-layer technical analysis, and research-driven trading experimentation.

Supports configurable strategies, risk management automation, and detailed performance logging.

---

## 🚀 Quick Start

Clone and install:

```bash
git clone https://github.com/YOUR_REPO/alpaca-trader.git
cd alpaca-trader
pip install -r requirements.txt
```

Run:

```bash
python3 run.py
```

On first launch, the bot will create:

```
alpaca_trader/.env
```

Add your Alpaca API keys:

```
APCA_API_KEY_ID="your_key"
APCA_API_SECRET_KEY="your_secret"
APCA_API_BASE_URL="https://paper-api.alpaca.markets"
```

---

## 🎯 Features

### Core Trading Engine

- Automated signal evaluation loop
- Multi-strategy architecture
- Risk-aware position sizing
- Market regime detection
- Config-driven behavior (no code changes required)

### Technical Indicators

- SMA / EMA
- RSI
- MACD
- ADX
- ATR
- Bollinger Bands
- Multi-timeframe signal confirmation

### Strategy System

Supports multiple strategy modes:

- Moving Average crossover (default)
- Opening Range + Fair Value Gap (OR/FVG)
- Regime-filtered execution

### Risk Management

- ATR-based stop loss
- Trailing stop logic
- Multi-level take profits
- Risk-per-trade sizing
- Max drawdown protection
- Risk/reward validation
- Position hold-time limits

### Execution Controls

- Market or limit orders
- Slippage simulation
- Commission modeling
- Cash account compatibility
- T+1 settlement handling
- PDT rule awareness

### Market Filters

- Market regime classification
- Volume filters
- 200 SMA trend filter
- VIX volatility filter
- Candle confirmation
- MACD confirmation layer

### Analytics & Logging

Automatically generates:

```
logs/
├── trading.log
├── debug.log

data/
├── trades.csv
├── signals.csv
├── performance.csv
├── indicators.csv
├── session.csv
```

---

## 🧠 Strategy Overview

### Moving Average Strategy

Primary signal generated when:

- Short MA crosses long MA
- Trend filters confirm
- Risk/reward meets threshold
- Market regime supports trade

Optional confirmation:

- MACD alignment
- RSI thresholds
- Volume confirmation

---

### Opening Range + Fair Value Gap Strategy

Designed for intraday momentum:

1. Detect opening range window.
2. Identify Fair Value Gap structures.
3. Validate volume and direction.
4. Execute with ATR-based risk controls.

Configurable parameters:

- Opening range duration
- Minimum gap size
- Entry timeframe
- Risk/reward target
- Maximum entry window

---

## ⚙️ Configuration

All trading behavior controlled via:

```
alpaca_trader/config.json
```

Key sections:

### Strategy

```
STRATEGY_MODE
OR_FVG_ENABLED
OR_FVG_OPENING_RANGE_MINUTES
OR_FVG_MIN_GAP_SIZE
```

### Risk

```
RISK_PER_TRADE
ATR_STOP_MULTIPLIER
MAX_DRAWDOWN
MIN_RISK_REWARD
```

### Filters

```
REGIME_DETECTION
USE_200_SMA_FILTER
USE_VIX_FILTER
MULTIFRAME_FILTER
```

### Execution

```
USE_LIMIT_ORDERS
LIMIT_ORDER_TIMEOUT
SLIPPAGE_PCT
COMMISSION_PCT
```

---

## 🏗 Architecture

```
alpaca_trader/
├── api.py           # Alpaca API interface
├── engine.py        # Core trading loop
├── indicators.py    # Technical analysis
├── filters.py       # Market condition filters
├── risk.py          # Risk & position sizing
├── utils.py         # Helpers
├── cli.py           # CLI interface
├── config.json      # Main configuration
```

---

## 🔄 How It Works

1. Load configuration and API credentials
2. Fetch historical market data
3. Calculate indicators
4. Evaluate market regime
5. Generate trading signals
6. Validate risk constraints
7. Execute trades via Alpaca API
8. Log analytics data

---

## 📊 Design Philosophy

- Config-first architecture
- Strategy isolation
- Risk before execution
- Modular extensibility
- Research-friendly logging

---

## ⚠️ Important Notes

- Use paper trading first.
- Algorithmic trading involves financial risk.
- No strategy guarantees profit.

---

## 🛠 Roadmap (Example)

- Strategy plug-in system
- ML signal scoring
- Portfolio-level risk controls
- Multi-symbol scanning
- Performance dashboard

---

## Disclaimer

This software is provided for educational and research purposes only.

Not financial advice.
