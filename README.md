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
- Comprehensive debug logging

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

### General Settings
- **DEBUG_MODE**: Enable detailed debug logging (default: true)
- **BAR_TIMEFRAME**: Candlestick timeframe for analysis (default: "5Min")
- **POLL_INTERVAL**: Seconds between market checks (default: 60)
- **MIN_NOTIONAL**: Minimum position size in dollars (default: 1.0)
- **PDT_RULE**: Enforce pattern day trader rules (default: true)

### Entry/Exit Signals
- **SHORT_WINDOW**: Fast moving average period (default: 10)
- **LONG_WINDOW**: Slow moving average period (default: 30)
- **USE_EMA**: Use EMA instead of SMA (default: true)
- **REQUIRE_MA_CROSSOVER**: Require recent MA crossover for signals (default: true)
- **CROSSOVER_LOOKBACK**: Bars to look back for crossovers (default: 5)
- **RSI_BUY_MAX**: Maximum RSI for buy signals in trend (default: 55)
- **RSI_SELL_MIN**: Minimum RSI for sell signals in trend (default: 45)
- **RSI_SELL_MAX**: Maximum RSI for sell signals in trend (default: 70)
- **RSI_RANGE_OVERSOLD**: RSI threshold for range-bound buy (default: 30)
- **RSI_RANGE_OVERBOUGHT**: RSI threshold for range-bound sell (default: 70)
- **ADX_THRESHOLD**: Minimum ADX for trend detection (default: 25)
- **MIN_SIGNAL_STRENGTH**: Minimum signal strength threshold (default: 0.3)

### Risk Management
- **ATR_STOP_MULTIPLIER**: Stop loss distance in ATR units (default: 2.0)
- **USE_TRAILING_STOP**: Enable trailing stop loss (default: true)
- **PROFIT_TARGET_1**: First profit target in R (default: 2.0)
- **PROFIT_TARGET_2**: Second profit target in R (default: 4.0)
- **MAX_DRAWDOWN**: Maximum account drawdown threshold (default: 0.08)
- **MAX_HOLD_TIME**: Maximum position hold time in seconds (default: 3600)
- **MIN_RISK_REWARD**: Minimum risk/reward ratio required (default: 2.0)
- **VOLATILITY_ADJUSTMENT**: Adjust position size based on volatility (default: true)

### Technical Indicators
- **BB_WINDOW**: Bollinger Bands period (default: 20)
- **BB_STD**: Bollinger Bands standard deviation (default: 2.0)
- **REQUIRE_CANDLE_PATTERN**: Require bullish/bearish candle patterns (default: false)
- **REQUIRE_MACD_CONFIRMATION**: Require MACD crossover confirmation (default: false)

### Filters
- **REGIME_DETECTION**: Enable market regime detection (default: true)
- **MULTIFRAME_FILTER**: Enable hourly timeframe confirmation (default: false)
- **USE_VIX_FILTER**: Filter trades based on VIX (default: false)
- **VIX_THRESHOLD**: Maximum VIX level to allow trades (default: 30)
- **USE_200_SMA_FILTER**: Filter based on 200-day SMA (default: false)
- **VOLUME_MULTIPLIER**: Minimum volume as multiple of average (default: 0.5)
- **MARKET_HOURS_FILTER**: Only trade during specific hours (default: false)
- **SKIP_MONDAYS_FRIDAYS**: Skip trading on Mondays and Fridays (default: false)

### Order Execution
- **USE_LIMIT_ORDERS**: Use limit orders instead of market orders (default: false)
- **LIMIT_ORDER_TIMEOUT**: Seconds to wait for limit order fill (default: 60)
- **ENABLE_SLIPPAGE**: Account for slippage in backtesting (default: true)
- **SLIPPAGE_PCT**: Estimated slippage percentage (default: 0.0005)
- **COMMISSION_PCT**: Commission percentage per trade (default: 0.0005)

### Backtesting
- **BACKTEST_DAYS**: Days of historical data for backtesting (default: 90)

### Advanced Features
- **USE_PIVOT_POINTS**: Use pivot point analysis (default: false)
- **USE_FIBONACCI**: Use Fibonacci retracement levels (default: false)
- **PULLBACK_PERCENTAGE**: Fibonacci pullback level (default: 0.382)

## Logging

- `trading.log`: Main trading activity log
- `debug.log`: Detailed debug information (when DEBUG_MODE is enabled)

## Architecture

```
alpaca_trader/
├── __init__.py         # Package initialization
├── __main__.py         # Module entry point
├── api.py              # Alpaca API wrapper with retry logic
├── engine.py           # Main trading engine
├── indicators.py       # Technical indicator calculations
├── filters.py          # Market filters and regime detection
├── risk.py             # Risk management data structures
├── utils.py            # Utility functions
├── config.json         # Configuration parameters
└── .env                # API credentials (create this)
```

## Warning

**DISCLAIMER: This software is provided for educational and research purposes only. The author is not responsible for any financial losses, damages, or liabilities incurred from using this trading bot. Use at your own risk.**

- Test thoroughly in paper trading before using real capital
- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Never trade with money you cannot afford to lose
- This is NOT financial advice
- Consult a licensed financial advisor before making investment decisions
