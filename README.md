# Alpaca Trader

Automated day trading bot for Alpaca Markets using technical indicators and risk management.

## Features

- Multiple technical indicators (SMA/EMA, RSI, ADX, ATR, MACD, Bollinger Bands)
- Advanced risk management with trailing stops and profit targets
- Market regime detection (trend/range/high vol/low vol)
- Multi-timeframe confluence analysis
- Position sizing based on account equity and risk per trade
- Support for both long and short positions
- Configurable via JSON config file

## Requirements

- Python 3.8+
- Alpaca Markets account (paper or live)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Create `.env` file in `alpaca_trader/` directory:
```
APCA_API_KEY_ID="your_api_key"
APCA_API_SECRET_KEY="your_secret_key"
APCA_API_BASE_URL="https://paper-api.alpaca.markets"
```

2. Modify `config.json` to adjust trading parameters:
- `SYMBOL`: Stock to trade (default: SPY)
- `RISK_PER_TRADE`: Risk per trade as % of equity (default: 0.005)
- `MAX_TRADES_PER_DAY`: Daily trade limit (default: 5)
- `ENABLE_SHORT_SELLING`: Enable/disable short positions (default: true)

## Usage

```bash
python3 run.py
```

Or:

```bash
python3 -m alpaca_trader
```

## Key Parameters

- **SHORT_WINDOW**: Fast moving average period (default: 20)
- **LONG_WINDOW**: Slow moving average period (default: 50)
- **ADX_THRESHOLD**: Minimum ADX for trend detection (default: 20)
- **ATR_STOP_MULTIPLIER**: Stop loss distance in ATR units (default: 1.5)
- **USE_TRAILING_STOP**: Enable trailing stop loss (default: true)
- **PROFIT_TARGET_1**: First profit target in R (default: 1.5)
- **PROFIT_TARGET_2**: Second profit target in R (default: 3.0)

## Logging

- `trading.log`: Main trading activity log
- `debug.log`: Detailed debug information

## Warning

This is for educational purposes. Test thoroughly in paper trading before using real capital. Trading involves risk of loss.
