import os
import sys
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import pytz

from .api import AlpacaClient
from .indicators import sma, ema, rsi, atr, adx, macd, bollinger
from .filters import check_volume, check_candle_pattern, check_macd_confirmation, check_200_sma_filter, detect_market_regime, get_vix
from .filters import check_multiframe_confluence
from .utils import EASTERN, seconds_to_human_readable

BARS_FOR_200_SMA = 210
BARS_FOR_SIGNAL = 200
BARS_FOR_REGIME = 50
BARS_FOR_ATR = 50
MIN_BARS_FOR_ATR = 14
VOLUME_LOOKBACK = 20
DEFAULT_STOP_LOSS_PCT = 0.02
VIX_LOOKBACK_DAYS = 5
SPY_VOLATILITY_LOOKBACK = 20
VOLATILITY_ANNUALIZATION_FACTOR = 252

SCRIPT_DIR = Path(__file__).parent
LOG_PATH = SCRIPT_DIR / "trading.log"
DEBUG_LOG_PATH = SCRIPT_DIR / "debug.log"
SESSION_STATE_PATH = SCRIPT_DIR / "session_state.csv"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
debug_handler = logging.FileHandler(DEBUG_LOG_PATH, mode='a')
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
debug_logger.addHandler(debug_handler)
debug_logger.propagate = False

CONFIG_PATH = SCRIPT_DIR / "config.json"
ENV_PATH = SCRIPT_DIR / ".env"

DEFAULT_CONFIG = {
    "DEBUG_MODE": True,
    "SYMBOL": "SPY",
    "BAR_TIMEFRAME": "5Min",
    "RISK_PER_TRADE": 0.01,
    "SHORT_WINDOW": 10,
    "LONG_WINDOW": 30,
    "MIN_NOTIONAL": 1.0,
    "POLL_INTERVAL": 60,
    "MAX_DRAWDOWN": 0.08,
    "PDT_RULE": False,
    "USE_TRAILING_STOP": True,
    "PROFIT_TARGET_1": 2.0,
    "PROFIT_TARGET_2": 4.0,
    "VOLATILITY_ADJUSTMENT": True,
    "MARKET_HOURS_FILTER": False,
    "ENABLE_SLIPPAGE": True,
    "SLIPPAGE_PCT": 0.0005,
    "COMMISSION_PCT": 0.0005,
    "MIN_SIGNAL_STRENGTH": 0.5,
    "BACKTEST_DAYS": 90,
    "USE_LIMIT_ORDERS": False,
    "LIMIT_ORDER_TIMEOUT": 60,
    "ADX_THRESHOLD": 30,
    "VOLUME_MULTIPLIER": 1.0,
    "ATR_STOP_MULTIPLIER": 2.0,
    "MAX_HOLD_TIME": 3600,
    "REGIME_DETECTION": True,
    "MULTIFRAME_FILTER": False,
    "BB_WINDOW": 20,
    "BB_STD": 2.0,
    "USE_EMA": True,
    "REQUIRE_CANDLE_PATTERN": False,
    "USE_PIVOT_POINTS": False,
    "VIX_THRESHOLD": 30,
    "USE_VIX_FILTER": False,
    "USE_FIBONACCI": False,
    "MAX_TRADES_PER_DAY": 1,
    "SKIP_MONDAYS_FRIDAYS": False,
    "USE_200_SMA_FILTER": False,
    "REQUIRE_MACD_CONFIRMATION": True,
    "MIN_RISK_REWARD": 2.0,
    "PULLBACK_PERCENTAGE": 0.382,
    "ENABLE_SHORT_SELLING": False,
    "RSI_BUY_MAX": 55,
    "RSI_SELL_MIN": 45,
    "RSI_SELL_MAX": 70,
    "RSI_RANGE_OVERSOLD": 30,
    "RSI_RANGE_OVERBOUGHT": 70,
    "REQUIRE_MA_CROSSOVER": True,
    "CROSSOVER_LOOKBACK": 3,
    "REQUIRE_CASH_ACCOUNT": True,
    "T1_SETTLEMENT_ENABLED": True,
    "CASH_RESERVE_PCT": 0.1
}

if not ENV_PATH.exists():
    placeholder = (
        'APCA_API_KEY_ID="YOUR_REAL_KEY_ID"\n'
        'APCA_API_SECRET_KEY="YOUR_REAL_SECRET_KEY"\n'
        'APCA_API_BASE_URL="https://paper-api.alpaca.markets"\n'
    )
    with open(ENV_PATH, "w") as f:
        f.write(placeholder)
    load_dotenv(ENV_PATH)
    logger.warning("⚠️  .env file was missing – a placeholder has been created at:")
    logger.warning(f"    {ENV_PATH}")
    logger.warning("   Edit this file and replace the placeholder values with your real Alpaca API credentials.")
    logger.warning('   Example lines to replace:')
    logger.warning('       APCA_API_KEY_ID="YOUR_REAL_KEY_ID"')
    logger.warning('       APCA_API_SECRET_KEY="YOUR_REAL_SECRET_KEY"')
    logger.warning('   After editing, restart the script.')
    sys.exit(1)
else:
    load_dotenv(ENV_PATH)

if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError:
        print("⚠️  config.json is invalid – recreating with defaults")
        config = DEFAULT_CONFIG.copy()
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
else:
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    config = DEFAULT_CONFIG.copy()
    print(f"✅ Created default config file at {CONFIG_PATH}")

DEBUG_MODE = bool(config.get("DEBUG_MODE", False))
SYMBOL = config["SYMBOL"]
BAR_TIMEFRAME = config.get("BAR_TIMEFRAME", "5Min")
RISK_PER_TRADE = float(config["RISK_PER_TRADE"])
SHORT_WINDOW = int(config["SHORT_WINDOW"])
LONG_WINDOW = int(config["LONG_WINDOW"])

if SHORT_WINDOW >= LONG_WINDOW:
    logger.error(f"⚠️  Configuration error: SHORT_WINDOW ({SHORT_WINDOW}) must be less than LONG_WINDOW ({LONG_WINDOW})")
    sys.exit(1)

REQUIRE_CASH_ACCOUNT = bool(config.get("REQUIRE_CASH_ACCOUNT", True))
T1_SETTLEMENT_ENABLED = bool(config.get("T1_SETTLEMENT_ENABLED", True))
CASH_RESERVE_PCT = float(config.get("CASH_RESERVE_PCT", 0.1))

try:
    test_client = AlpacaClient(
        os.getenv("APCA_API_KEY_ID"),
        os.getenv("APCA_API_SECRET_KEY"),
        os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets"),
        api_version="v2"
    )
    account = test_client.get_account()
    logger.info("✅  API credentials validated")
    
    equity = float(getattr(account, 'equity', 0))
    buying_power = float(getattr(account, 'buying_power', 0))
    cash = float(getattr(account, 'cash', 0))
    is_pattern_day_trader = getattr(account, 'pattern_day_trader', False)
    daytrade_count = getattr(account, 'daytrade_count', 0)
    account_status = getattr(account, 'status', 'UNKNOWN')
    
    is_paper_account = "paper-api.alpaca.markets" in os.getenv("APCA_API_BASE_URL", "")
    
    logger.info(f"💵  Account Info:")
    logger.info(f"    Type: {'PAPER' if is_paper_account else 'LIVE'}")
    logger.info(f"    Equity: ${equity:.2f}")
    logger.info(f"    Cash: ${cash:.2f}")
    logger.info(f"    Buying Power: ${buying_power:.2f}")
    logger.info(f"    PDT Status: {is_pattern_day_trader}")
    logger.info(f"    Daytrade Count: {daytrade_count}")
    
    if account_status != 'ACTIVE':
        logger.error(f"⚠️  Account status is {account_status}, must be ACTIVE")
        sys.exit(1)
    
    is_margin_account = buying_power > cash * 1.5
    has_minimum_equity = equity >= 25000
    
    if not is_paper_account and not has_minimum_equity:
        if is_margin_account:
            logger.warning("⚠️  WARNING: LIVE margin account with equity < $25,000")
            logger.warning("    You should be using a CASH account to avoid PDT restrictions")
            logger.warning("    Convert to cash account in your Alpaca dashboard")
        
        if ENABLE_SHORT_SELLING:
            logger.error("⚠️  SHORT SELLING DISABLED: Live account with equity < $25,000 cannot short")
            logger.error("    Set ENABLE_SHORT_SELLING to False in config.json")
            logger.error("    Or increase account equity to $25,000+")
            sys.exit(1)
        
        logger.info("✅  Short selling disabled for live account < $25k")
        
        if T1_SETTLEMENT_ENABLED:
            logger.info(f"✅  T+1 settlement tracking enabled")
            logger.info(f"    Keeping {CASH_RESERVE_PCT*100:.0f}% cash reserve for safety")
    
    elif REQUIRE_CASH_ACCOUNT:
        if is_margin_account:
            logger.warning("⚠️  WARNING: Margin account detected")
            logger.warning("    REQUIRE_CASH_ACCOUNT is True but buying power exceeds cash")
            logger.warning("    Set REQUIRE_CASH_ACCOUNT to False in config.json for margin accounts")
        
        logger.info("✅  Cash account mode enabled")
        
        if T1_SETTLEMENT_ENABLED:
            logger.info("✅  T+1 settlement tracking enabled")
            logger.info(f"    Keeping {CASH_RESERVE_PCT*100:.0f}% cash reserve for safety")
    
    if is_paper_account:
        logger.info("📝  Paper trading account - all restrictions relaxed")
    elif has_minimum_equity:
        logger.info(f"✅  Equity ${equity:.2f} >= $25,000 - full trading enabled")
    
except Exception as e:
    logger.error(f"⚠️  Invalid API credentials: {e}")
    logger.error("    Please check your .env file and ensure your Alpaca API keys are correct")
    sys.exit(1)

MIN_NOTIONAL = float(config["MIN_NOTIONAL"])
POLL_INTERVAL = int(config["POLL_INTERVAL"])
MAX_DRAWDOWN = float(config["MAX_DRAWDOWN"])
PDT_RULE = bool(config["PDT_RULE"])
USE_TRAILING_STOP = bool(config["USE_TRAILING_STOP"])
PROFIT_TARGET_1 = float(config["PROFIT_TARGET_1"])
PROFIT_TARGET_2 = float(config["PROFIT_TARGET_2"])
VOLATILITY_ADJUSTMENT = bool(config["VOLATILITY_ADJUSTMENT"])
MARKET_HOURS_FILTER = bool(config["MARKET_HOURS_FILTER"])
ENABLE_SLIPPAGE = bool(config["ENABLE_SLIPPAGE"])
SLIPPAGE_PCT = float(config["SLIPPAGE_PCT"])
COMMISSION_PCT = float(config["COMMISSION_PCT"])
MIN_SIGNAL_STRENGTH = float(config["MIN_SIGNAL_STRENGTH"])
BACKTEST_DAYS = int(config["BACKTEST_DAYS"])
USE_LIMIT_ORDERS = bool(config["USE_LIMIT_ORDERS"])
LIMIT_ORDER_TIMEOUT = int(config["LIMIT_ORDER_TIMEOUT"])
ADX_THRESHOLD = float(config["ADX_THRESHOLD"])
VOLUME_MULTIPLIER = float(config["VOLUME_MULTIPLIER"])
ATR_STOP_MULTIPLIER = float(config["ATR_STOP_MULTIPLIER"])
MAX_HOLD_TIME = int(config["MAX_HOLD_TIME"])
REGIME_DETECTION = bool(config["REGIME_DETECTION"])
MULTIFRAME_FILTER = bool(config["MULTIFRAME_FILTER"])
BB_WINDOW = int(config["BB_WINDOW"])
BB_STD = float(config["BB_STD"])
USE_EMA = bool(config["USE_EMA"])
REQUIRE_CANDLE_PATTERN = bool(config["REQUIRE_CANDLE_PATTERN"])
USE_PIVOT_POINTS = bool(config["USE_PIVOT_POINTS"])
VIX_THRESHOLD = float(config["VIX_THRESHOLD"])
USE_VIX_FILTER = bool(config["USE_VIX_FILTER"])
USE_FIBONACCI = bool(config["USE_FIBONACCI"])
MAX_TRADES_PER_DAY = int(config["MAX_TRADES_PER_DAY"])
SKIP_MONDAYS_FRIDAYS = bool(config["SKIP_MONDAYS_FRIDAYS"])
USE_200_SMA_FILTER = bool(config["USE_200_SMA_FILTER"])
REQUIRE_MACD_CONFIRMATION = bool(config["REQUIRE_MACD_CONFIRMATION"])
MIN_RISK_REWARD = float(config["MIN_RISK_REWARD"])
PULLBACK_PERCENTAGE = float(config["PULLBACK_PERCENTAGE"])
ENABLE_SHORT_SELLING = bool(config.get("ENABLE_SHORT_SELLING", False))
RSI_BUY_MAX = float(config.get("RSI_BUY_MAX", 55))
RSI_SELL_MIN = float(config.get("RSI_SELL_MIN", 45))
RSI_SELL_MAX = float(config.get("RSI_SELL_MAX", 70))
RSI_RANGE_OVERSOLD = float(config.get("RSI_RANGE_OVERSOLD", 30))
RSI_RANGE_OVERBOUGHT = float(config.get("RSI_RANGE_OVERBOUGHT", 70))
REQUIRE_MA_CROSSOVER = bool(config.get("REQUIRE_MA_CROSSOVER", True))
CROSSOVER_LOOKBACK = int(config.get("CROSSOVER_LOOKBACK", 5))

api = AlpacaClient(
    os.getenv('APCA_API_KEY_ID'),
    os.getenv('APCA_API_SECRET_KEY'),
    os.getenv('APCA_API_BASE_URL'),
    api_version='v2'
)

class SettlementTracker:
    def __init__(self):
        self.pending_settlements = {}
    
    def add_trade(self, trade_date, amount):
        settlement_date = self._get_next_trading_day(trade_date)
        if settlement_date not in self.pending_settlements:
            self.pending_settlements[settlement_date] = 0.0
        self.pending_settlements[settlement_date] += amount
        logger.info(f"💰  T+1: ${amount:.2f} settling on {settlement_date.strftime('%Y-%m-%d')}")
        debug_print(f"Added ${amount:.2f} to settle on {settlement_date}")
    
    def _get_next_trading_day(self, date):
        next_day = date + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day.date()
    
    def settle_funds(self, current_date):
        settled_amount = 0.0
        current_date_only = current_date.date()
        
        dates_to_remove = []
        for settlement_date, amount in self.pending_settlements.items():
            if settlement_date <= current_date_only:
                settled_amount += amount
                dates_to_remove.append(settlement_date)
        
        for date in dates_to_remove:
            del self.pending_settlements[date]
        
        if settled_amount > 0:
            logger.info(f"✅  Settled ${settled_amount:.2f} on {current_date_only}")
            debug_print(f"Settled ${settled_amount:.2f}")
        
        return settled_amount
    
    def get_pending_amount(self):
        return sum(self.pending_settlements.values())
    
    def reset(self):
        self.pending_settlements = {}


class SignalState:
    def __init__(self):
        self.last_bullish_crossover_bar = -999
        self.last_bearish_crossover_bar = -999
    
    def reset(self):
        self.last_bullish_crossover_bar = -999
        self.last_bearish_crossover_bar = -999

class PositionState:
    def __init__(self):
        self.target_1_hit = False
        self.trailing_stop = None
    
    def reset(self):
        self.target_1_hit = False
        self.trailing_stop = None

signal_state = SignalState()
position_state = PositionState()

def save_session_state(trades_today, opening_equity, last_bullish_crossover, last_bearish_crossover, session_date):
    try:
        state_data = {
            'timestamp': datetime.now(EASTERN).isoformat(),
            'session_date': session_date.strftime('%Y-%m-%d'),
            'trades_today': trades_today,
            'opening_equity': opening_equity,
            'last_bullish_crossover_bar': last_bullish_crossover,
            'last_bearish_crossover_bar': last_bearish_crossover
        }
        
        df = pd.DataFrame([state_data])
        
        if SESSION_STATE_PATH.exists():
            existing = pd.read_csv(SESSION_STATE_PATH)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.tail(100)
        
        df.to_csv(SESSION_STATE_PATH, index=False)
        debug_print(f"Session state saved: trades={trades_today}, equity=${opening_equity:.2f}")
    except Exception as e:
        debug_print(f"Failed to save session state: {e}")

def load_session_state():
    try:
        if not SESSION_STATE_PATH.exists():
            debug_print("No session state file found, starting fresh")
            return None
        
        df = pd.read_csv(SESSION_STATE_PATH)
        if len(df) == 0:
            debug_print("Session state file empty, starting fresh")
            return None
        
        last_state = df.iloc[-1]
        last_timestamp = datetime.fromisoformat(last_state['timestamp'])
        now = datetime.now(EASTERN)
        
        time_diff = (now - last_timestamp).total_seconds()
        
        if time_diff > 7200:
            debug_print(f"Last session state too old ({time_diff/3600:.1f}h ago), starting fresh")
            return None
        
        session_date = datetime.strptime(last_state['session_date'], '%Y-%m-%d').date()
        if session_date != now.date():
            debug_print(f"Last session was on different day ({session_date}), starting fresh")
            return None
        
        state = {
            'trades_today': int(last_state['trades_today']),
            'opening_equity': float(last_state['opening_equity']),
            'last_bullish_crossover_bar': int(last_state['last_bullish_crossover_bar']),
            'last_bearish_crossover_bar': int(last_state['last_bearish_crossover_bar']),
            'timestamp': last_timestamp
        }
        
        debug_print(f"Loaded session state from {time_diff/60:.1f}m ago: trades={state['trades_today']}")
        logger.info(f"🔄  Resumed session from {time_diff/60:.1f}m ago: {state['trades_today']} trades today")
        return state
        
    except Exception as e:
        debug_print(f"Failed to load session state: {e}")
        return None

def debug_print(message):
    if DEBUG_MODE:
        debug_logger.debug(f"🔎  {message}")
        print(f"{datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - DEBUG - 🔎  {message}", flush=True)

def fetch_equity():
    debug_print("Fetching account equity")
    account = api.get_account()
    equity = float(account.equity)
    debug_print(f"Current equity: ${equity:.2f}")
    return equity

def fetch_buying_power(settlement_tracker=None):
    debug_print("Fetching buying power")
    account = api.get_account()
    bp = float(account.buying_power)
    cash = float(account.cash)
    
    if T1_SETTLEMENT_ENABLED and settlement_tracker:
        pending = settlement_tracker.get_pending_amount()
        available_cash = cash - pending
        
        if CASH_RESERVE_PCT > 0:
            reserve = cash * CASH_RESERVE_PCT
            available_cash = max(0, available_cash - reserve)
        
        debug_print(f"Cash: ${cash:.2f}, Pending: ${pending:.2f}, Available: ${available_cash:.2f}")
        return available_cash
    
    debug_print(f"Buying power: ${bp:.2f}")
    return bp

def get_recent_bars(symbol, limit=100):
    debug_print(f"Fetching {limit} bars for {symbol} ({BAR_TIMEFRAME})")
    try:
        bars = api.get_bars(symbol, BAR_TIMEFRAME, limit=limit)
        if bars is None or len(bars) == 0:
            debug_print(f"No bars returned for {symbol}")
            return None
        debug_print(f"Retrieved {len(bars)} bars")
        return bars
    except Exception as e:
        logger.error(f"Error fetching bars: {e}")
        debug_print(f"Error fetching bars: {e}")
        return None

def current_position_qty(symbol):
    debug_print(f"Checking position for {symbol}")
    try:
        positions = api.list_positions()
        for pos in positions:
            if pos.symbol == symbol:
                qty = float(pos.qty)
                debug_print(f"Found position: {qty} shares")
                return qty
        debug_print("No position found")
        return 0
    except Exception as e:
        debug_print(f"Error checking position: {e}")
        return 0

def close_all_positions():
    debug_print("Closing all positions")
    try:
        api.close_all_positions()
        logger.info("✅  All positions closed")
        debug_print("All positions closed successfully")
    except Exception as e:
        logger.error(f"Error closing positions: {e}")
        debug_print(f"Error closing positions: {e}")

def get_bid_ask(symbol):
    debug_print(f"Getting bid/ask for {symbol}")
    try:
        quote = api.get_latest_quote(symbol)
        bid = float(quote.bid_price)
        ask = float(quote.ask_price)
        debug_print(f"Bid: ${bid:.2f}, Ask: ${ask:.2f}")
        return bid, ask
    except Exception as e:
        logger.error(f"Error getting quote: {e}")
        debug_print(f"Error getting quote: {e}")
        return None, None

def submit_market_buy(symbol, position_size):
    debug_print(f"Submitting market buy order: {symbol}, size=${position_size:.2f}")
    if position_size <= 0:
        debug_print(f"Invalid position size: ${position_size:.2f}")
        return None
    try:
        execution_price = api.place_order(symbol, "buy", position_size, None, LIMIT_ORDER_TIMEOUT)
        if execution_price:
            logger.info(f"🟢  BUY {symbol} @ ${execution_price:.2f}")
            debug_print(f"Buy order filled @ ${execution_price:.2f}")
            return execution_price
        else:
            logger.warning(f"Buy order returned no execution price")
            debug_print(f"Buy order returned None")
            return None
    except Exception as e:
        logger.error(f"Buy order failed: {e}")
        debug_print(f"Buy order failed: {e}")
        return None

def submit_market_sell(symbol, qty):
    debug_print(f"Submitting market sell order: {symbol}, qty={qty}")
    try:
        shares = int(qty)
        if shares <= 0:
            debug_print(f"Invalid quantity: {shares}")
            return None
        order = api.submit_order(symbol=symbol, qty=shares, side="sell", type="market", time_in_force="day")
        status = api.get_order(order.id)
        timeout = 30
        start_time = time.time()
        while status.status not in {"filled", "cancelled", "expired", "rejected"}:
            if time.time() - start_time > timeout:
                debug_print("Order status check timeout")
                return None
            time.sleep(0.5)
            status = api.get_order(order.id)
        if status.status == "filled":
            price = float(status.filled_avg_price)
            logger.info(f"🔴  SELL {symbol} @ ${price:.2f}")
            debug_print(f"Sell order filled @ ${price:.2f}")
            return price
    except Exception as e:
        logger.error(f"Sell order failed: {e}")
        debug_print(f"Sell order failed: {e}")
        return None

def submit_limit_buy(symbol, position_size, limit_price):
    debug_print(f"Submitting limit buy: {symbol}, size=${position_size:.2f}, limit=${limit_price:.2f}")
    if position_size <= 0:
        debug_print(f"Invalid position size: ${position_size:.2f}")
        return None
    try:
        execution_price = api.place_order(symbol, "buy", position_size, limit_price, LIMIT_ORDER_TIMEOUT)
        if execution_price:
            logger.info(f"🟢  BUY {symbol} @ ${execution_price:.2f}")
            debug_print(f"Limit buy filled @ ${execution_price:.2f}")
            return execution_price
        else:
            debug_print("Limit order timeout, attempting market order")
            execution_price = api.place_order(symbol, "buy", position_size, None, LIMIT_ORDER_TIMEOUT)
            if execution_price:
                logger.info(f"🟢  BUY {symbol} @ ${execution_price:.2f} (market)")
                debug_print(f"Market order filled @ ${execution_price:.2f}")
                return execution_price
            else:
                logger.warning(f"Market order fallback also failed")
                debug_print(f"Market order fallback returned None")
                return None
    except Exception as e:
        logger.error(f"Buy order failed: {e}")
        debug_print(f"Buy order failed: {e}")
        return None

def submit_short_sell(symbol, position_size):
    debug_print(f"Submitting short sell: {symbol}, size=${position_size:.2f}")
    if position_size <= 0:
        debug_print(f"Invalid position size: ${position_size:.2f}")
        return None
    try:
        execution_price = api.place_order(symbol, "sell", position_size, None, LIMIT_ORDER_TIMEOUT)
        if execution_price:
            logger.info(f"🔴  SHORT {symbol} @ ${execution_price:.2f}")
            debug_print(f"Short sell filled @ ${execution_price:.2f}")
            return execution_price
        else:
            logger.warning(f"Short sell returned no execution price")
            debug_print(f"Short sell returned None")
            return None
    except Exception as e:
        logger.error(f"Short sell failed: {e}")
        debug_print(f"Short sell failed: {e}")
        return None

def submit_limit_short_sell(symbol, position_size, limit_price):
    debug_print(f"Submitting limit short: {symbol}, size=${position_size:.2f}, limit=${limit_price:.2f}")
    if position_size <= 0:
        debug_print(f"Invalid position size: ${position_size:.2f}")
        return None
    try:
        execution_price = api.place_order(symbol, "sell", position_size, limit_price, LIMIT_ORDER_TIMEOUT)
        if execution_price:
            logger.info(f"🔴  SHORT {symbol} @ ${execution_price:.2f}")
            debug_print(f"Limit short filled @ ${execution_price:.2f}")
            return execution_price
        else:
            debug_print("Limit order timeout, attempting market order")
            execution_price = api.place_order(symbol, "sell", position_size, None, LIMIT_ORDER_TIMEOUT)
            if execution_price:
                logger.info(f"🔴  SHORT {symbol} @ ${execution_price:.2f} (market)")
                debug_print(f"Market short filled @ ${execution_price:.2f}")
                return execution_price
            else:
                logger.warning(f"Market order fallback also failed")
                debug_print(f"Market order fallback returned None")
                return None
    except Exception as e:
        logger.error(f"Short sell failed: {e}")
        debug_print(f"Short sell failed: {e}")
        return None

def submit_buy_to_cover(symbol, qty):
    debug_print(f"Submitting buy to cover: {symbol}, qty={qty}")
    try:
        shares = int(qty)
        if shares <= 0:
            debug_print(f"Invalid quantity: {shares}")
            return None
        order = api.submit_order(symbol=symbol, qty=shares, side="buy", type="market", time_in_force="day")
        status = api.get_order(order.id)
        timeout = 30
        start_time = time.time()
        while status.status not in {"filled", "cancelled", "expired", "rejected"}:
            if time.time() - start_time > timeout:
                debug_print("Order status check timeout")
                return None
            time.sleep(0.5)
            status = api.get_order(order.id)
        if status.status == "filled":
            price = float(status.filled_avg_price)
            logger.info(f"🟢  COVER {symbol} @ ${price:.2f}")
            debug_print(f"Buy to cover filled @ ${price:.2f}")
            return price
    except Exception as e:
        logger.error(f"Buy to cover failed: {e}")
        debug_print(f"Buy to cover failed: {e}")
        return None

def calculate_position_size(equity, stop_loss, current_price):
    debug_print(f"Calculating position size: equity=${equity:.2f}, stop=${stop_loss:.2f}, price=${current_price:.2f}")
    risk_amount = equity * RISK_PER_TRADE
    price_risk = abs(current_price - stop_loss)
    if price_risk == 0:
        debug_print("Price risk is zero, returning MIN_NOTIONAL")
        return MIN_NOTIONAL
    shares = risk_amount / price_risk
    position_value = shares * current_price
    max_position = equity * 0.25
    if position_value > max_position:
        position_value = max_position
        debug_print(f"Position capped at 25% equity: ${position_value:.2f}")
    if position_value < MIN_NOTIONAL:
        position_value = MIN_NOTIONAL
        debug_print(f"Position set to minimum: ${position_value:.2f}")
    debug_print(f"Calculated position size: ${position_value:.2f}")
    return position_value

def advanced_signal_generator(symbol):
    debug_print(f"Generating signal for {symbol}")
    bars = get_recent_bars(symbol, BARS_FOR_SIGNAL)
    if bars is None or len(bars) < LONG_WINDOW:
        debug_print("Insufficient data for signal generation")
        return None, 0, 0, None
    
    closes = bars['close']
    highs = bars['high']
    lows = bars['low']
    current_price = closes.iloc[-1]
    
    debug_print("Calculating indicators...")
    if USE_EMA:
        short_ma_series = ema(closes, SHORT_WINDOW)
        long_ma_series = ema(closes, LONG_WINDOW)
        short_ma = short_ma_series.iloc[-1]
        long_ma = long_ma_series.iloc[-1]
    else:
        short_ma_series = sma(closes, SHORT_WINDOW)
        long_ma_series = sma(closes, LONG_WINDOW)
        short_ma = short_ma_series.iloc[-1]
        long_ma = long_ma_series.iloc[-1]
    
    bullish_crossover = False
    bearish_crossover = False
    
    if REQUIRE_MA_CROSSOVER and len(bars) >= LONG_WINDOW + CROSSOVER_LOOKBACK:
        current_bar_index = len(bars) - 1
        
        for i in range(1, CROSSOVER_LOOKBACK + 1):
            bar_index = current_bar_index - i
            if bar_index >= 1 and bar_index < len(bars) and (bar_index + 1) < len(bars):
                idx_current = len(short_ma_series) - i
                idx_prev = len(short_ma_series) - i - 1
                if idx_prev >= 0 and idx_current < len(short_ma_series):
                    if short_ma_series.iloc[idx_prev] <= long_ma_series.iloc[idx_prev] and short_ma_series.iloc[idx_current] > long_ma_series.iloc[idx_current]:
                        if bar_index > signal_state.last_bullish_crossover_bar:
                            bullish_crossover = True
                            signal_state.last_bullish_crossover_bar = bar_index
                            debug_print(f"Bullish crossover detected {i} bars ago")
                        break
        
        for i in range(1, CROSSOVER_LOOKBACK + 1):
            bar_index = current_bar_index - i
            if bar_index >= 1 and bar_index < len(bars) and (bar_index + 1) < len(bars):
                idx_current = len(short_ma_series) - i
                idx_prev = len(short_ma_series) - i - 1
                if idx_prev >= 0 and idx_current < len(short_ma_series):
                    if short_ma_series.iloc[idx_prev] >= long_ma_series.iloc[idx_prev] and short_ma_series.iloc[idx_current] < long_ma_series.iloc[idx_current]:
                        if bar_index > signal_state.last_bearish_crossover_bar:
                            bearish_crossover = True
                            signal_state.last_bearish_crossover_bar = bar_index
                            debug_print(f"Bearish crossover detected {i} bars ago")
                        break
    
    rsi_val = rsi(closes, 14).iloc[-1]
    adx_val = adx(highs, lows, closes).iloc[-1]
    atr_val = atr(highs, lows, closes).iloc[-1]
    upper, middle, lower = bollinger(closes, BB_WINDOW, BB_STD)
    
    debug_print(f"Indicators: MA_short={short_ma:.2f}, MA_long={long_ma:.2f}, RSI={rsi_val:.1f}, ADX={adx_val:.1f}")
    
    vix_level = get_vix(api, SYMBOL, USE_VIX_FILTER)
    if USE_VIX_FILTER and vix_level > VIX_THRESHOLD:
        debug_print(f"VIX filter triggered: {vix_level:.1f} > {VIX_THRESHOLD}")
        return None, 0, 0, None
    
    if not check_volume(bars, VOLUME_MULTIPLIER):
        if len(bars) >= 20 and "volume" in bars.columns:
            avg_vol = bars["volume"].rolling(window=20).mean().iloc[-1]
            cur_vol = bars["volume"].iloc[-1]
            debug_print(f"Volume filter failed: current={cur_vol:,.0f}, avg={avg_vol:,.0f}, required={avg_vol*VOLUME_MULTIPLIER:,.0f} ({VOLUME_MULTIPLIER}x)")
        else:
            debug_print("Volume filter failed: insufficient data")
        return None, 0, 0, None
    
    bullish_pattern, bearish_pattern = check_candle_pattern(bars)
    macd_signal = check_macd_confirmation(bars)
    multiframe_trend = check_multiframe_confluence(SYMBOL, USE_EMA, api) if MULTIFRAME_FILTER else "neutral"
    regime = detect_market_regime(bars, ADX_THRESHOLD) if REGIME_DETECTION else "trend"
    
    debug_print(f"Filters: regime={regime}, multiframe={multiframe_trend}, macd={macd_signal}")
    
    signal = None
    strength = 0
    stop = 0
    position_type = None
    
    if regime == "trend":
        if short_ma > long_ma and rsi_val < RSI_BUY_MAX:
            if REQUIRE_MA_CROSSOVER and not bullish_crossover:
                debug_print("Bullish signal rejected: no recent crossover")
            elif REQUIRE_CANDLE_PATTERN and not bullish_pattern:
                debug_print("Bullish signal rejected: candle pattern required")
            elif REQUIRE_MACD_CONFIRMATION and macd_signal != "bullish":
                debug_print("Bullish signal rejected: MACD confirmation required")
            else:
                signal = "buy"
                strength = min(1.0, (adx_val / 40) * 0.7 + 0.3)
                stop = current_price - atr_val * ATR_STOP_MULTIPLIER
                position_type = "long"
                debug_print(f"BUY signal: strength={strength:.2f}, stop=${stop:.2f}")
        
        elif short_ma < long_ma and rsi_val > RSI_SELL_MIN and rsi_val < RSI_SELL_MAX:
            if REQUIRE_MA_CROSSOVER and not bearish_crossover:
                debug_print("Bearish signal rejected: no recent crossover")
            elif REQUIRE_CANDLE_PATTERN and not bearish_pattern:
                debug_print("Bearish signal rejected: candle pattern required")
            elif REQUIRE_MACD_CONFIRMATION and macd_signal != "bearish":
                debug_print("Bearish signal rejected: MACD confirmation required")
            else:
                signal = "sell"
                strength = min(1.0, (adx_val / 40) * 0.7 + 0.3)
                stop = current_price + atr_val * ATR_STOP_MULTIPLIER
                position_type = "short"
                debug_print(f"SELL signal: strength={strength:.2f}, stop=${stop:.2f}")
    
    elif regime == "range":
        if current_price <= lower.iloc[-1] and rsi_val < RSI_RANGE_OVERSOLD:
            if REQUIRE_CANDLE_PATTERN and not bullish_pattern:
                debug_print("Range buy rejected: candle pattern required")
            elif REQUIRE_MACD_CONFIRMATION and macd_signal != "bullish":
                debug_print("Range buy rejected: MACD confirmation required")
            else:
                signal = "buy"
                strength = 0.85
                stop = current_price - atr_val * ATR_STOP_MULTIPLIER
                position_type = "long"
                debug_print(f"Range BUY signal: strength={strength:.2f}, stop=${stop:.2f}")
        
        elif current_price >= upper.iloc[-1] and rsi_val > RSI_RANGE_OVERBOUGHT:
            if REQUIRE_CANDLE_PATTERN and not bearish_pattern:
                debug_print("Range sell rejected: candle pattern required")
            elif REQUIRE_MACD_CONFIRMATION and macd_signal != "bearish":
                debug_print("Range sell rejected: MACD confirmation required")
            else:
                signal = "sell"
                strength = 0.85
                stop = current_price + atr_val * ATR_STOP_MULTIPLIER
                position_type = "short"
                debug_print(f"Range SELL signal: strength={strength:.2f}, stop=${stop:.2f}")
    
    if strength < MIN_SIGNAL_STRENGTH:
        debug_print(f"Signal rejected: strength {strength:.2f} < {MIN_SIGNAL_STRENGTH}")
        return None, 0, 0, None
    
    return signal, strength, stop, position_type

def scale_out_profit_taking(symbol, entry_price, current_price, stop_loss, position_type):
    debug_print(f"Checking scale out: entry=${entry_price:.2f}, current=${current_price:.2f}")
    
    if entry_price <= 0:
        debug_print("Invalid entry_price, skipping scale out")
        return False
    
    if position_type == 'long':
        profit_pct = ((current_price - entry_price) / entry_price) * 100
    else:
        profit_pct = ((entry_price - current_price) / entry_price) * 100
    
    risk_pct = abs((entry_price - stop_loss) / entry_price) * 100
    
    target_1_pct = risk_pct * PROFIT_TARGET_1
    target_2_pct = risk_pct * PROFIT_TARGET_2
    
    if profit_pct >= target_1_pct and not position_state.target_1_hit:
        qty = current_position_qty(symbol)
        if qty != 0:
            half_qty = int(qty / 2)
            if half_qty > 0:
                debug_print(f"Target 1 hit ({target_1_pct:.2f}%), scaling out {half_qty} shares")
                if position_type == 'long':
                    submit_market_sell(symbol, half_qty)
                else:
                    submit_buy_to_cover(symbol, half_qty)
                position_state.target_1_hit = True
                logger.info(f"💰  Partial profit @ {profit_pct:.2f}% ({half_qty} shares)")
                debug_print(f"Partial profit taken: {half_qty} shares @ {profit_pct:.2f}%")
            else:
                position_state.target_1_hit = True
                logger.info(f"💰  Target 1 reached @ {profit_pct:.2f}% (position too small to scale)")
                debug_print(f"Position size {qty} too small for partial exit, holding for target 2")
    
    if profit_pct >= target_2_pct:
        qty = current_position_qty(symbol)
        if qty != 0:
            debug_print(f"Target 2 hit ({target_2_pct:.2f}%), closing remaining {qty} shares")
            if position_type == 'long':
                submit_market_sell(symbol, qty)
            else:
                submit_buy_to_cover(symbol, qty)
            logger.info(f"💰💰  Full profit @ {profit_pct:.2f}%")
            debug_print(f"Full profit target hit: closed @ {profit_pct:.2f}%")
            return True
    
    return False

def atr_based_trailing_stop(symbol, entry_price, current_price, initial_stop, position_type):
    debug_print(f"Checking trailing stop: entry=${entry_price:.2f}, current=${current_price:.2f}")
    
    if position_state.trailing_stop is None:
        position_state.trailing_stop = initial_stop
        debug_print(f"Initialized trailing stop: ${initial_stop:.2f}")
    
    bars = get_recent_bars(symbol, 50)
    if bars is None or len(bars) < 14:
        debug_print("Insufficient data for ATR calculation")
        return False
    
    current_atr = atr(bars['high'], bars['low'], bars['close']).iloc[-1]
    
    if current_atr <= 0 or np.isnan(current_atr):
        debug_print(f"Invalid ATR value: {current_atr}, using initial stop")
        return False
    
    if position_type == 'long':
        new_stop = current_price - (current_atr * ATR_STOP_MULTIPLIER)
        if new_stop > position_state.trailing_stop:
            debug_print(f"Updating trailing stop: ${position_state.trailing_stop:.2f} -> ${new_stop:.2f}")
            position_state.trailing_stop = new_stop
        
        if current_price <= position_state.trailing_stop:
            debug_print(f"Long stop hit: ${current_price:.2f} <= ${position_state.trailing_stop:.2f}")
            return True
    else:
        new_stop = current_price + (current_atr * ATR_STOP_MULTIPLIER)
        if new_stop < position_state.trailing_stop:
            debug_print(f"Updating trailing stop: ${position_state.trailing_stop:.2f} -> ${new_stop:.2f}")
            position_state.trailing_stop = new_stop
        
        if current_price >= position_state.trailing_stop:
            debug_print(f"Short stop hit: ${current_price:.2f} >= ${position_state.trailing_stop:.2f}")
            return True
    
    return False

def main():
    logger.info("🚀  Trading engine starting...")
    debug_print("Trading engine initialized")
    logger.info(f"📊  Symbol: {SYMBOL}, Timeframe: {BAR_TIMEFRAME}")
    logger.info(f"⚙️  Risk/Trade: {RISK_PER_TRADE*100:.2f}%, Stop Mult: {ATR_STOP_MULTIPLIER}x")
    
    try:
        while True:
            try:
                clock = api.get_clock()
                if not clock.is_open:
                    next_open = clock.next_open.astimezone(EASTERN)
                    wait_time = (next_open - datetime.now(EASTERN)).total_seconds()
                    logger.info(f"🌙  Market closed. Next open: {next_open.strftime('%I:%M %p ET on %A, %B %d')}")
                    debug_print(f"Market closed, waiting {seconds_to_human_readable(int(wait_time))} until next open")
                    time.sleep(min(wait_time, 3600))
                    continue
                
                logger.info("🔔  Market open - session starting")
                debug_print("Market open, starting trading session")
                
                current_date = datetime.now(EASTERN)
                session_date = current_date.date()
                
                opening_equity = fetch_equity()
                logger.info(f"💵  Starting equity: ${opening_equity:.2f}")
                
                settlement_tracker = SettlementTracker()
                
                if T1_SETTLEMENT_ENABLED:
                    settlement_tracker.settle_funds(current_date)
                
                position_active = False
                entry_price = 0
                entry_time = None
                stop_loss = 0
                position_type = None
                trade_count = 0
                trades_today = 0
                total_pnl = 0
                
                signal_state.reset()
                position_state.reset()
                
                restored_state = load_session_state()
                if restored_state:
                    trades_today = restored_state['trades_today']
                    signal_state.last_bullish_crossover_bar = restored_state['last_bullish_crossover_bar']
                    signal_state.last_bearish_crossover_bar = restored_state['last_bearish_crossover_bar']
                    if abs(restored_state['opening_equity'] - opening_equity) < opening_equity * 0.05:
                        opening_equity = restored_state['opening_equity']
                        debug_print(f"Restored opening equity: ${opening_equity:.2f}")
                    logger.info(f"📊  Session restored: {trades_today} trades today")
                
                try:
                    existing_position = api.get_position(SYMBOL)
                    qty = float(existing_position.qty)
                    if qty != 0:
                        position_active = True
                        entry_price = float(existing_position.avg_entry_price)
                        position_type = 'long' if qty > 0 else 'short'
                        bars_for_atr = get_recent_bars(SYMBOL, 50)
                        if bars_for_atr is not None and len(bars_for_atr) >= 14:
                            atr_val = atr(bars_for_atr['high'], bars_for_atr['low'], bars_for_atr['close']).iloc[-1]
                            if position_type == 'long':
                                stop_loss = entry_price - atr_val * ATR_STOP_MULTIPLIER
                            else:
                                stop_loss = entry_price + atr_val * ATR_STOP_MULTIPLIER
                        else:
                            if position_type == 'long':
                                stop_loss = entry_price * 0.98
                            else:
                                stop_loss = entry_price * 1.02
                        
                        logger.info(f"🔄  Recovered existing {position_type.upper()} position: {abs(qty)} shares @ ${entry_price:.2f}, stop=${stop_loss:.2f}")
                        debug_print(f"Position recovered from previous session")
                        
                        entry_time = datetime.now(EASTERN)
                        
                        unrealized_plpc = float(existing_position.unrealized_plpc) if hasattr(existing_position, 'unrealized_plpc') else 0
                        if unrealized_plpc > 0.01:
                            position_state.target_1_hit = True
                            debug_print("Assuming target 1 already hit based on positive P&L")
                        
                        if USE_TRAILING_STOP:
                            position_state.trailing_stop = stop_loss
                except Exception as e:
                    debug_print(f"No existing position found or error during recovery: {e}")
                
                retry_count = 0
                max_retries = 3
                
                while clock.is_open:
                    try:
                        clock = api.get_clock()
                    except Exception as e:
                        debug_print(f"Error fetching clock: {e}")
                        time.sleep(10)
                        continue
                    
                    current_equity = fetch_equity()
                    drawdown = (opening_equity - current_equity) / opening_equity if opening_equity > 0 else 0
                    
                    if drawdown > MAX_DRAWDOWN:
                        logger.warning(f"⚠️  Max drawdown reached: {drawdown:.2%}")
                        debug_print(f"Max drawdown triggered: {drawdown:.2%}")
                        close_all_positions()
                        logger.info("🛑  Trading halted for the day")
                        time.sleep(3600)
                        break
                    
                    bars = get_recent_bars(SYMBOL, 10)
                    if bars is None or len(bars) == 0:
                        retry_count += 1
                        debug_print(f"No bars available, retry {retry_count}/{max_retries}")
                        if retry_count >= max_retries:
                            debug_print("Max retries reached, continuing with next iteration")
                            retry_count = 0
                        time.sleep(30)
                        continue
                    
                    retry_count = 0
                    current_price = bars['close'].iloc[-1]
                    vix_level = get_vix(api, SYMBOL, USE_VIX_FILTER)
                    
                    if position_active:
                        debug_print(f"Managing active position: {position_type}, entry=${entry_price:.2f}")
                        
                        if MAX_HOLD_TIME > 0 and entry_time:
                            time_in_trade = (datetime.now(EASTERN) - entry_time).total_seconds()
                            if time_in_trade > MAX_HOLD_TIME:
                                logger.info(f"⏰  Max hold time ({MAX_HOLD_TIME//60} min)")
                                debug_print(f"Max hold time exceeded, closing position")
                                qty = current_position_qty(SYMBOL)
                                if qty != 0:
                                    if position_type == 'long':
                                        submit_market_sell(SYMBOL, qty)
                                    else:
                                        submit_buy_to_cover(SYMBOL, abs(qty))
                                    position_active = False
                                    trade_count += 1
                                    position_state.reset()
                                    debug_print(f"Sleeping {seconds_to_human_readable(POLL_INTERVAL)} after exit")
                                    time.sleep(POLL_INTERVAL)
                                    continue
                        
                        if scale_out_profit_taking(SYMBOL, entry_price, current_price, stop_loss, position_type):
                            remaining_qty = current_position_qty(SYMBOL)
                            if remaining_qty == 0:
                                position_active = False
                                position_state.reset()
                                debug_print(f"Sleeping {seconds_to_human_readable(POLL_INTERVAL)} after exit")
                                time.sleep(POLL_INTERVAL)
                                continue
                        
                        if atr_based_trailing_stop(SYMBOL, entry_price, current_price, stop_loss, position_type):
                            qty = current_position_qty(SYMBOL)
                            if qty != 0:
                                if position_type == 'long':
                                    submit_market_sell(SYMBOL, qty)
                                else:
                                    submit_buy_to_cover(SYMBOL, abs(qty))
                                position_active = False
                                trade_count += 1
                                logger.info("🛑  Stop hit")
                                debug_print("Stop hit, position closed")
                                position_state.reset()
                                debug_print(f"Sleeping {seconds_to_human_readable(POLL_INTERVAL)} after exit")
                                time.sleep(POLL_INTERVAL)
                                continue
                    
                    if trades_today >= MAX_TRADES_PER_DAY:
                        logger.info(f"📊  Daily limit ({MAX_TRADES_PER_DAY}) - monitoring only")
                        debug_print(f"Daily trade limit reached ({trades_today}/{MAX_TRADES_PER_DAY})")
                        time.sleep(POLL_INTERVAL)
                        continue
                    
                    signal, strength, signal_stop_loss, signal_position_type = advanced_signal_generator(SYMBOL)
                    
                    if signal == 'sell' and not ENABLE_SHORT_SELLING:
                        debug_print("Short selling disabled, ignoring sell signal")
                        signal = None
                        signal_position_type = None
                    
                    bars = get_recent_bars(SYMBOL, 50)
                    if bars is not None:
                        regime = detect_market_regime(bars, ADX_THRESHOLD)
                    else:
                        regime = 'unknown'
                    
                    if signal in ['buy', 'sell'] and not position_active:
                        debug_print(f"Signal detected: {signal}, executing trade...")
                        buying_power = fetch_buying_power(settlement_tracker)
                        position_size = calculate_position_size(current_equity, signal_stop_loss, current_price)
                        
                        if buying_power >= position_size:
                            execution_price = None
                            
                            if signal == 'buy':
                                if USE_LIMIT_ORDERS:
                                    bid, ask = get_bid_ask(SYMBOL)
                                    limit_price = bid
                                    execution_price = submit_limit_buy(SYMBOL, position_size, limit_price)
                                else:
                                    execution_price = submit_market_buy(SYMBOL, position_size)
                            elif signal == 'sell':
                                if USE_LIMIT_ORDERS:
                                    bid, ask = get_bid_ask(SYMBOL)
                                    limit_price = ask
                                    execution_price = submit_limit_short_sell(SYMBOL, position_size, limit_price)
                                else:
                                    execution_price = submit_short_sell(SYMBOL, position_size)
                            
                            if execution_price:
                                trade_count += 1
                                trades_today += 1
                                entry_price = execution_price
                                entry_time = datetime.now(EASTERN)
                                stop_loss = signal_stop_loss
                                position_active = True
                                position_type = signal_position_type
                                
                                if T1_SETTLEMENT_ENABLED and signal == 'buy':
                                    trade_amount = position_size
                                    settlement_tracker.add_trade(datetime.now(EASTERN), trade_amount)
                                
                                if entry_price > 0:
                                    risk_amount = abs(entry_price - stop_loss) / entry_price
                                else:
                                    risk_amount = 0
                                
                                logger.info(f"    Entry=${entry_price:.2f}, Stop=${stop_loss:.2f}, Risk={risk_amount:.2%}")
                                logger.info(f"    Regime={regime}, Strength={strength:.2f}, Trade {trade_count} ({trades_today}/{MAX_TRADES_PER_DAY})")
                                debug_print(f"Trade executed: entry=${entry_price:.2f}, stop=${stop_loss:.2f}, regime={regime}")
                                
                                position_state.trailing_stop = stop_loss
                                debug_print(f"Trailing stop initialized: ${stop_loss:.2f}")
                            else:
                                logger.error(f"❌  Order execution failed: {signal.upper()} ${position_size:.2f}")
                                logger.error(f"    Possible reasons: Order rejected, timeout, or market closed")
                                debug_print(f"Order execution returned None - order not filled")
                                signal = None
                        else:
                            logger.warning(f"⚠️  Insufficient buying power: ${buying_power:.2f} < ${position_size:.2f}")
                            debug_print(f"Insufficient buying power: ${buying_power:.2f} < ${position_size:.2f}")
                            
                            if T1_SETTLEMENT_ENABLED:
                                pending = settlement_tracker.get_pending_amount()
                                logger.info(f"    Pending settlement: ${pending:.2f}")
                                debug_print(f"Funds tied up in T+1 settlement: ${pending:.2f}")
                    
                    position_status = f"{position_type.upper()}" if position_active else "FLAT"
                    
                    try:
                        ts = clock.timestamp
                        if ts.tzinfo is None:
                            ts = EASTERN.localize(ts)
                        else:
                            ts = ts.astimezone(EASTERN)
                        current_time = ts.strftime("%I:%M:%S %p ET")
                    except Exception:
                        current_time = datetime.now(EASTERN).strftime("%I:%M:%S %p ET")
                    
                    hourly_trend = check_multiframe_confluence(SYMBOL, USE_EMA, api)
                    status_msg = f"⏱️  {current_time} | {position_status} | {regime.upper()}"
                    
                    if position_active:
                        if entry_price > 0 and current_price > 0:
                            pnl_pct = ((current_price - entry_price) / entry_price) * 100 if position_type == 'long' else ((entry_price - current_price) / entry_price) * 100
                        else:
                            pnl_pct = 0
                        status_msg += f" | PnL: {pnl_pct:+.2f}%"
                    
                    status_msg += f" | H:{hourly_trend} | VIX:{vix_level:.1f} | {trades_today}/{MAX_TRADES_PER_DAY}"
                    logger.info(status_msg)
                    
                    save_session_state(
                        trades_today,
                        opening_equity,
                        signal_state.last_bullish_crossover_bar,
                        signal_state.last_bearish_crossover_bar,
                        session_date
                    )
                    
                    debug_print(f"Sleeping {seconds_to_human_readable(POLL_INTERVAL)}...")
                    time.sleep(POLL_INTERVAL)
                
                logger.info("🔚  Session ending...")
                debug_print("Session ending, closing all positions...")
                close_all_positions()
                
                final_equity = fetch_equity()
                session_pnl = final_equity - opening_equity
                session_pnl_pct = (session_pnl / opening_equity) * 100 if opening_equity > 0 else 0
                
                logger.info(f"📊  Summary: {trade_count} trades")
                logger.info(f"💰  Final: ${final_equity:.2f} (PNL: ${session_pnl:+.2f}, {session_pnl_pct:+.2f}%)")
                logger.info("✅  Day complete. Waiting for next session...")
                debug_print(f"Day complete. Trades: {trade_count}, PnL: ${session_pnl:+.2f}")
                
                next_open = None
                next_close = None
                try:
                    clock = api.get_clock()
                    if clock.next_open and clock.next_close:
                        next_open = clock.next_open
                        next_close = clock.next_close
                        if next_open.tzinfo is None:
                            next_open = EASTERN.localize(next_open)
                        else:
                            next_open = next_open.astimezone(EASTERN)
                        if next_close.tzinfo is None:
                            next_close = EASTERN.localize(next_close)
                        else:
                            next_close = next_close.astimezone(EASTERN)
                except Exception as e:
                    debug_print(f"Could not fetch next open time: {e}")
                
                if next_open:
                    now = datetime.now(EASTERN)
                    wait_seconds = (next_open - now).total_seconds()
                    if wait_seconds > 0:
                        logger.info(f"⏰  Next session: {next_open.strftime('%Y-%m-%d %I:%M %p ET')}")
                        logger.info(f"⏳  Sleeping {seconds_to_human_readable(int(wait_seconds))}")
                        debug_print(f"Sleeping until next market open: {wait_seconds}s")
                        time.sleep(wait_seconds)
                    else:
                        time.sleep(60)
                else:
                    logger.info("⏳  Sleeping 1 hour before retry")
                    time.sleep(3600)
                
            except Exception as e:
                logger.error(f"💥  Session error: {e}")
                debug_print(f"Session error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.info("⏳  Waiting 5 min before retry...")
                time.sleep(300)
                
    except KeyboardInterrupt:
        logger.info("🛑  User interrupt")
        debug_print("User interrupt detected")
        close_all_positions()
    except Exception as e:
        logger.error(f"💥  Fatal error: {e}")
        debug_print(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("🔚  Shutdown")
        debug_print("Script shutdown")

def run():
    return main()

if __name__ == "__main__":
    main()
