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
- `RISK_PER_TRADE`: Risk per trade as % of equity (default: 0.01)
- `MAX_TRADES_PER_DAY`: Daily trade limit (default: 5)
- `ENABLE_SHORT_SELLING`: Enable/disable short positions (default: true)
- `MIN_SIGNAL_STRENGTH`: Minimum signal strength to enter trades (default: 0.3)

## Usage

```bash
python3 run.py
```

Or:

```bash
python3 -m alpaca_trader
```

## Key Parameters

### Entry/Exit Signals
- **SHORT_WINDOW**: Fast moving average period (default: 10)
- **LONG_WINDOW**: Slow moving average period (default: 30)
- **RSI_BUY_MAX**: Maximum RSI for buy signals (default: 55)
- **RSI_SELL_MIN**: Minimum RSI for sell signals (default: 30)
- **ADX_THRESHOLD**: Minimum ADX for trend detection (default: 25)
- **MIN_SIGNAL_STRENGTH**: Minimum signal strength threshold (default: 0.3)

### Risk Management
- **ATR_STOP_MULTIPLIER**: Stop loss distance in ATR units (default: 2.0)
- **USE_TRAILING_STOP**: Enable trailing stop loss (default: true)
- **PROFIT_TARGET_1**: First profit target in R (default: 2.0)
- **PROFIT_TARGET_2**: Second profit target in R (default: 4.0)
- **MAX_DRAWDOWN**: Maximum account drawdown threshold (default: 0.08)

### Filters
- **MULTIFRAME_FILTER**: Enable hourly timeframe confirmation (default: false)
- **USE_VIX_FILTER**: Filter trades based on VIX (default: false)
- **USE_200_SMA_FILTER**: Filter based on 200-day SMA (default: false)
- **VOLUME_MULTIPLIER**: Minimum volume as multiple of average (default: 0.5)

## Logging

- `trading.log`: Main trading activity log
- `debug.log`: Detailed debug information

## Warning

**DISCLAIMER: This software is provided for educational and research purposes only. The author is not responsible for any financial losses, damages, or liabilities incurred from using this trading bot. Use at your own risk.**

- Test thoroughly in paper trading before using real capital
- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Never trade with money you cannot afford to lose
- This is NOT financial advice
- Consult a licensed financial advisor before making investment decisions
