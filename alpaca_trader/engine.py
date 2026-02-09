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
SESSION_STATE_PATH = SCRIPT_DIR / "session.csv"
TRADES_PATH = SCRIPT_DIR / "trades.csv"
SIGNALS_PATH = SCRIPT_DIR / "signals.csv"
PERFORMANCE_PATH = SCRIPT_DIR / "performance.csv"
INDICATORS_PATH = SCRIPT_DIR / "indicators.csv"

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
    "POLL_INTERVAL": 30,
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
    "MIN_SIGNAL_STRENGTH": 0.4,
    "BACKTEST_DAYS": 90,
    "USE_LIMIT_ORDERS": False,
    "LIMIT_ORDER_TIMEOUT": 60,
    "ADX_THRESHOLD": 30,
    "VOLUME_MULTIPLIER": 0.7,
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
    "MAX_TRADES_PER_DAY": 3,
    "SKIP_MONDAYS_FRIDAYS": False,
    "USE_200_SMA_FILTER": False,
    "REQUIRE_MACD_CONFIRMATION": False,
    "MIN_RISK_REWARD": 2.0,
    "PULLBACK_PERCENTAGE": 0.382,
    "ENABLE_SHORT_SELLING": False,
    "RSI_BUY_MAX": 65,
    "RSI_SELL_MIN": 35,
    "RSI_SELL_MAX": 70,
    "RSI_RANGE_OVERSOLD": 30,
    "RSI_RANGE_OVERBOUGHT": 70,
    "REQUIRE_MA_CROSSOVER": False,
    "CROSSOVER_LOOKBACK": 3,
    "REQUIRE_CASH_ACCOUNT": False,
    "T1_SETTLEMENT_ENABLED": False,
    "CASH_RESERVE_PCT": 0.0,
    "STRATEGY_MODE": "ma_crossover",
    "OR_FVG_ENABLED": False,
    "OR_FVG_OPENING_RANGE_MINUTES": 15,
    "OR_FVG_ENTRY_TIMEFRAME": "3Min",
    "OR_FVG_MIN_GAP_SIZE": 0.05,
    "OR_FVG_RISK_REWARD_RATIO": 2.0,
    "OR_FVG_MAX_ENTRY_TIME": "10:30",
    "OR_FVG_REQUIRE_VOLUME_CONFIRM": True
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

STRATEGY_MODE = config.get("STRATEGY_MODE", "ma_crossover")
OR_FVG_ENABLED = bool(config.get("OR_FVG_ENABLED", False))
OR_FVG_OPENING_RANGE_MINUTES = int(config.get("OR_FVG_OPENING_RANGE_MINUTES", 15))
OR_FVG_ENTRY_TIMEFRAME = config.get("OR_FVG_ENTRY_TIMEFRAME", "3Min")
OR_FVG_MIN_GAP_SIZE = float(config.get("OR_FVG_MIN_GAP_SIZE", 0.05))
OR_FVG_RISK_REWARD_RATIO = float(config.get("OR_FVG_RISK_REWARD_RATIO", 2.0))
OR_FVG_MAX_ENTRY_TIME = config.get("OR_FVG_MAX_ENTRY_TIME", "10:30")
OR_FVG_REQUIRE_VOLUME_CONFIRM = bool(config.get("OR_FVG_REQUIRE_VOLUME_CONFIRM", True))

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

def log_trade(entry_time, exit_time, symbol, side, entry_price, exit_price, shares, position_value, stop_loss, target_1, target_2, pnl_dollars, pnl_percent, hold_minutes, exit_reason, regime, signal_strength, rsi, adx, ma_spread, slippage):
    try:
        trade_data = {
            'entry_time': entry_time.isoformat(),
            'exit_time': exit_time.isoformat(),
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'position_value': position_value,
            'stop_loss': stop_loss,
            'target_1': target_1,
            'target_2': target_2,
            'pnl_dollars': pnl_dollars,
            'pnl_percent': pnl_percent,
            'hold_minutes': hold_minutes,
            'exit_reason': exit_reason,
            'regime': regime,
            'signal_strength': signal_strength,
            'rsi': rsi,
            'adx': adx,
            'ma_spread': ma_spread,
            'slippage': slippage
        }
        
        df = pd.DataFrame([trade_data])
        
        if TRADES_PATH.exists():
            existing = pd.read_csv(TRADES_PATH)
            df = pd.concat([existing, df], ignore_index=True)
            cutoff_date = datetime.now(EASTERN) - timedelta(days=90)
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df = df[df['entry_time'] > cutoff_date]
        
        df.to_csv(TRADES_PATH, index=False)
        debug_print(f"Trade logged: {side} {symbol} P&L=${pnl_dollars:.2f} ({pnl_percent:.2f}%)")
    except Exception as e:
        debug_print(f"Failed to log trade: {e}")

def log_missed_signal(timestamp, signal_type, reject_reason, price_at_signal, symbol, signal_strength, rsi, adx, regime):
    try:
        signal_data = {
            'timestamp': timestamp.isoformat(),
            'signal_type': signal_type,
            'reject_reason': reject_reason,
            'price_at_signal': price_at_signal,
            'symbol': symbol,
            'signal_strength': signal_strength,
            'rsi': rsi,
            'adx': adx,
            'regime': regime
        }
        
        df = pd.DataFrame([signal_data])
        
        if SIGNALS_PATH.exists():
            existing = pd.read_csv(SIGNALS_PATH)
            df = pd.concat([existing, df], ignore_index=True)
            cutoff_date = datetime.now(EASTERN) - timedelta(days=30)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
            df = df[df['timestamp'] > cutoff_date]
        
        df.to_csv(SIGNALS_PATH, index=False)
        debug_print(f"Missed signal logged: {signal_type} rejected due to {reject_reason}")
    except Exception as e:
        debug_print(f"Failed to log missed signal: {e}")

def log_daily_performance(date, opening_equity, closing_equity, total_trades, winners, losers, total_pnl, max_drawdown, avg_regime, avg_vix):
    try:
        perf_data = {
            'date': date.strftime('%Y-%m-%d'),
            'opening_equity': opening_equity,
            'closing_equity': closing_equity,
            'total_trades': total_trades,
            'winners': winners,
            'losers': losers,
            'win_rate': (winners / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'pnl_percent': (total_pnl / opening_equity * 100) if opening_equity > 0 else 0,
            'max_drawdown': max_drawdown,
            'avg_regime': avg_regime,
            'avg_vix': avg_vix
        }
        
        df = pd.DataFrame([perf_data])
        
        if PERFORMANCE_PATH.exists():
            existing = pd.read_csv(PERFORMANCE_PATH)
            df = pd.concat([existing, df], ignore_index=True)
            cutoff_date = datetime.now(EASTERN) - timedelta(days=180)
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] > cutoff_date]
        
        df.to_csv(PERFORMANCE_PATH, index=False)
        debug_print(f"Daily performance logged: {total_trades} trades, P&L=${total_pnl:.2f}")
    except Exception as e:
        debug_print(f"Failed to log daily performance: {e}")

def log_indicators(timestamp, symbol, price, volume, rsi_val, adx_val, atr_val, ma_spread, regime, position_status):
    try:
        indicator_data = {
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'rsi': rsi_val,
            'adx': adx_val,
            'atr': atr_val,
            'ma_spread': ma_spread,
            'regime': regime,
            'position_status': position_status
        }
        
        df = pd.DataFrame([indicator_data])
        
        if INDICATORS_PATH.exists():
            existing = pd.read_csv(INDICATORS_PATH)
            df = pd.concat([existing, df], ignore_index=True)
            cutoff_date = datetime.now(EASTERN) - timedelta(days=7)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
            df = df[df['timestamp'] > cutoff_date]
        
        df.to_csv(INDICATORS_PATH, index=False)
    except Exception as e:
        debug_print(f"Failed to log indicators: {e}")

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
    return position_value

class ORFVGState:
    def __init__(self):
        self.opening_range_high = None
        self.opening_range_low = None
        self.opening_range_set = False
        self.fvg_detected = False
        self.fvg_direction = None
        self.fvg_candle_index = None
        self.entry_triggered = False
        
    def reset(self):
        self.opening_range_high = None
        self.opening_range_low = None
        self.opening_range_set = False
        self.fvg_detected = False
        self.fvg_direction = None
        self.fvg_candle_index = None
        self.entry_triggered = False

or_fvg_state = ORFVGState()

def detect_fair_value_gap(bars, min_gap_pct=0.05):
    if bars is None or len(bars) < 3:
        return None, None
    
    for i in range(len(bars) - 3, max(len(bars) - 10, 0) - 1, -1):
        if i < 0 or i + 2 >= len(bars):
            continue
            
        candle_1_high = bars['high'].iloc[i]
        candle_1_low = bars['low'].iloc[i]
        candle_2_high = bars['high'].iloc[i + 1]
        candle_2_low = bars['low'].iloc[i + 1]
        candle_3_high = bars['high'].iloc[i + 2]
        candle_3_low = bars['low'].iloc[i + 2]
        
        bullish_gap = candle_3_low > candle_1_high
        if bullish_gap:
            gap_size = candle_3_low - candle_1_high
            if candle_2_high > 0:
                gap_pct = (gap_size / candle_2_high) * 100
                if gap_pct >= min_gap_pct:
                    debug_print(f"Bullish FVG detected: gap={gap_size:.2f} ({gap_pct:.2f}%)")
                    return "bullish", i + 2
        
        bearish_gap = candle_3_high < candle_1_low
        if bearish_gap:
            gap_size = candle_1_low - candle_3_high
            if candle_2_low > 0:
                gap_pct = (gap_size / candle_2_low) * 100
                if gap_pct >= min_gap_pct:
                    debug_print(f"Bearish FVG detected: gap={gap_size:.2f} ({gap_pct:.2f}%)")
                    return "bearish", i + 2
    
    return None, None

def or_fvg_signal_generator(symbol):
    debug_print("Checking OR-FVG strategy")
    
    now = datetime.now(EASTERN)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    opening_range_end = market_open + timedelta(minutes=OR_FVG_OPENING_RANGE_MINUTES)
    
    max_entry_time_parts = OR_FVG_MAX_ENTRY_TIME.split(":")
    max_entry_time = now.replace(
        hour=int(max_entry_time_parts[0]), 
        minute=int(max_entry_time_parts[1]), 
        second=0, 
        microsecond=0
    )
    
    if now > max_entry_time:
        debug_print(f"Past max entry time ({OR_FVG_MAX_ENTRY_TIME})")
        return None, 0, 0, None
    
    if not or_fvg_state.opening_range_set and now >= opening_range_end:
        start_time = market_open
        end_time = opening_range_end
        
        bars_or = api.get_bars(
            symbol, 
            "1Min",
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            limit=OR_FVG_OPENING_RANGE_MINUTES
        )
        
        if bars_or is not None and len(bars_or) > 0:
            or_fvg_state.opening_range_high = bars_or['high'].max()
            or_fvg_state.opening_range_low = bars_or['low'].min()
            
            if (pd.isna(or_fvg_state.opening_range_high) or 
                pd.isna(or_fvg_state.opening_range_low) or
                or_fvg_state.opening_range_high <= 0 or
                or_fvg_state.opening_range_low <= 0 or
                or_fvg_state.opening_range_low >= or_fvg_state.opening_range_high):
                logger.error(f"❌  Invalid opening range: High={or_fvg_state.opening_range_high}, Low={or_fvg_state.opening_range_low}")
                debug_print("Invalid opening range values detected")
                return None, 0, 0, None
            
            or_fvg_state.opening_range_set = True
            logger.info(f"📊  Opening Range set: High=${or_fvg_state.opening_range_high:.2f}, Low=${or_fvg_state.opening_range_low:.2f}")
            debug_print(f"OR set: H={or_fvg_state.opening_range_high:.2f}, L={or_fvg_state.opening_range_low:.2f}")
    
    if not or_fvg_state.opening_range_set:
        debug_print("Opening range not yet set")
        return None, 0, 0, None
    
    bars_1min = api.get_bars(symbol, OR_FVG_ENTRY_TIMEFRAME, limit=50)
    if bars_1min is None or len(bars_1min) == 0:
        debug_print("No 1-min bars available")
        return None, 0, 0, None
    
    bars_df = bars_1min.reset_index()
    
    bars_after_or = bars_df[bars_df['timestamp'] >= opening_range_end]
    if len(bars_after_or) < 3:
        debug_print("Not enough bars after opening range")
        return None, 0, 0, None
    
    current_price = bars_after_or['close'].iloc[-1]
    
    if not or_fvg_state.fvg_detected:
        fvg_direction, fvg_index = detect_fair_value_gap(bars_after_or, OR_FVG_MIN_GAP_SIZE)
        
        if fvg_direction:
            or_fvg_state.fvg_detected = True
            or_fvg_state.fvg_direction = fvg_direction
            or_fvg_state.fvg_candle_index = fvg_index
            logger.info(f"🎯  FVG detected: {fvg_direction.upper()}")
            debug_print(f"FVG set: direction={fvg_direction}")
    
    if not or_fvg_state.fvg_detected:
        debug_print("No FVG detected yet")
        return None, 0, 0, None
    
    if or_fvg_state.entry_triggered:
        debug_print("Entry already triggered today")
        return None, 0, 0, None
    
    breakout_detected = False
    position_type = None
    
    if or_fvg_state.fvg_direction == "bullish":
        if current_price > or_fvg_state.opening_range_high:
            breakout_detected = True
            position_type = "long"
            debug_print(f"Bullish breakout: ${current_price:.2f} > ${or_fvg_state.opening_range_high:.2f}")
    elif or_fvg_state.fvg_direction == "bearish":
        if current_price < or_fvg_state.opening_range_low:
            breakout_detected = True
            position_type = "short"
            debug_print(f"Bearish breakout: ${current_price:.2f} < ${or_fvg_state.opening_range_low:.2f}")
    
    if not breakout_detected:
        debug_print("No breakout detected")
        return None, 0, 0, None
    
    if OR_FVG_REQUIRE_VOLUME_CONFIRM:
        if len(bars_after_or) >= 20:
            avg_volume = bars_after_or['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = bars_after_or['volume'].iloc[-1]
            if current_volume < avg_volume * 1.2:
                debug_print(f"Volume confirmation failed: {current_volume:.0f} < {avg_volume*1.2:.0f}")
                return None, 0, 0, None
        else:
            debug_print(f"Volume confirmation skipped: only {len(bars_after_or)} bars available (need 20)")
    
    if position_type == "long":
        stop_loss = or_fvg_state.opening_range_low
        signal = "buy"
    else:
        stop_loss = or_fvg_state.opening_range_high
        signal = "sell"
    
    strength = 1.0
    
    logger.info(f"✅  OR-FVG Entry: {signal.upper()} @ ${current_price:.2f}, Stop=${stop_loss:.2f}")
    debug_print(f"OR-FVG signal generated: {signal}, stop={stop_loss:.2f}")
    
    return signal, strength, stop_loss, position_type

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
    
    if STRATEGY_MODE == "or_fvg" or OR_FVG_ENABLED:
        target_pct = risk_pct * OR_FVG_RISK_REWARD_RATIO
        
        if profit_pct >= target_pct:
            qty = current_position_qty(symbol)
            if qty != 0:
                debug_print(f"OR-FVG target hit ({target_pct:.2f}%), closing {qty} shares")
                exit_price = None
                if position_type == 'long':
                    exit_price = submit_market_sell(symbol, qty)
                else:
                    exit_price = submit_buy_to_cover(symbol, qty)
                logger.info(f"💰  OR-FVG Target @ {profit_pct:.2f}%")
                debug_print(f"OR-FVG profit target hit: closed @ {profit_pct:.2f}%")
                return True, exit_price if exit_price else current_price
        return False, None
    
    target_1_pct = risk_pct * PROFIT_TARGET_1
    target_2_pct = risk_pct * PROFIT_TARGET_2
    
    if profit_pct >= target_1_pct and not position_state.target_1_hit:
        qty = current_position_qty(symbol)
        if qty != 0:
            half_qty = int(qty / 2)
            if half_qty > 0:
                debug_print(f"Target 1 hit ({target_1_pct:.2f}%), scaling out {half_qty} shares")
                exit_price = None
                if position_type == 'long':
                    exit_price = submit_market_sell(symbol, half_qty)
                else:
                    exit_price = submit_buy_to_cover(symbol, half_qty)
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
            exit_price = None
            if position_type == 'long':
                exit_price = submit_market_sell(symbol, qty)
            else:
                exit_price = submit_buy_to_cover(symbol, qty)
            logger.info(f"💰💰  Full profit @ {profit_pct:.2f}%")
            debug_print(f"Full profit target hit: closed @ {profit_pct:.2f}%")
            return True, exit_price if exit_price else current_price
    
    return False, None

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
                or_fvg_state.reset()
                
                restored_state = load_session_state()
                if restored_state:
                    trades_today = restored_state['trades_today']
                    signal_state.last_bullish_crossover_bar = restored_state['last_bullish_crossover_bar']
                    signal_state.last_bearish_crossover_bar = restored_state['last_bearish_crossover_bar']
                    if abs(restored_state['opening_equity'] - opening_equity) < opening_equity * 0.05:
                        opening_equity = restored_state['opening_equity']
                        debug_print(f"Restored opening equity: ${opening_equity:.2f}")
                    logger.info(f"📊  Session restored: {trades_today} trades today")
                
                entry_strength = 0
                entry_rsi = 0
                entry_adx = 0
                entry_ma_spread = 0
                entry_regime = 'unknown'
                winners = 0
                losers = 0
                vix_readings = []
                regime_readings = []
                max_intraday_drawdown = 0
                last_indicator_log = datetime.now(EASTERN)
                
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
                                    exit_time = datetime.now(EASTERN)
                                    hold_minutes = time_in_trade / 60
                                    
                                    if position_type == 'long':
                                        exit_price = submit_market_sell(SYMBOL, qty)
                                        pnl_dollars = (exit_price - entry_price) * qty if exit_price else 0
                                    else:
                                        exit_price = submit_buy_to_cover(SYMBOL, abs(qty))
                                        pnl_dollars = (entry_price - exit_price) * abs(qty) if exit_price else 0
                                    
                                    pnl_percent = (pnl_dollars / (entry_price * abs(qty)) * 100) if entry_price > 0 and qty != 0 else 0
                                    
                                    if pnl_dollars > 0:
                                        winners += 1
                                    elif pnl_dollars < 0:
                                        losers += 1
                                    
                                    risk_pct = abs((entry_price - stop_loss) / entry_price) if entry_price > 0 else 0
                                    target_1 = entry_price + (entry_price - stop_loss) * PROFIT_TARGET_1 if position_type == 'long' else entry_price - (stop_loss - entry_price) * PROFIT_TARGET_1
                                    target_2 = entry_price + (entry_price - stop_loss) * PROFIT_TARGET_2 if position_type == 'long' else entry_price - (stop_loss - entry_price) * PROFIT_TARGET_2
                                    
                                    log_trade(
                                        entry_time,
                                        exit_time,
                                        SYMBOL,
                                        position_type,
                                        entry_price,
                                        exit_price if exit_price else current_price,
                                        abs(qty),
                                        entry_price * abs(qty),
                                        stop_loss,
                                        target_1,
                                        target_2,
                                        pnl_dollars,
                                        pnl_percent,
                                        hold_minutes,
                                        'max_hold_time',
                                        entry_regime,
                                        entry_strength,
                                        entry_rsi,
                                        entry_adx,
                                        entry_ma_spread,
                                        0
                                    )
                                    
                                    position_active = False
                                    trade_count += 1
                                    position_state.reset()
                                    debug_print(f"Sleeping {seconds_to_human_readable(POLL_INTERVAL)} after exit")
                                    time.sleep(POLL_INTERVAL)
                                    continue
                        
                        qty_before_scale = current_position_qty(SYMBOL)
                        target_hit, exit_price_target = scale_out_profit_taking(SYMBOL, entry_price, current_price, stop_loss, position_type)
                        if target_hit:
                            remaining_qty = current_position_qty(SYMBOL)
                            if remaining_qty == 0:
                                exit_time = datetime.now(EASTERN)
                                hold_minutes = (exit_time - entry_time).total_seconds() / 60 if entry_time else 0
                                
                                if position_type == 'long':
                                    pnl_dollars = (exit_price_target - entry_price) * abs(qty_before_scale) if exit_price_target and qty_before_scale != 0 else 0
                                else:
                                    pnl_dollars = (entry_price - exit_price_target) * abs(qty_before_scale) if exit_price_target and qty_before_scale != 0 else 0
                                
                                pnl_percent = (pnl_dollars / (entry_price * abs(qty_before_scale)) * 100) if entry_price > 0 and qty_before_scale != 0 else 0
                                
                                if pnl_dollars > 0:
                                    winners += 1
                                elif pnl_dollars < 0:
                                    losers += 1
                                
                                risk_pct = abs((entry_price - stop_loss) / entry_price) if entry_price > 0 else 0
                                target_1 = entry_price + (entry_price - stop_loss) * PROFIT_TARGET_1 if position_type == 'long' else entry_price - (stop_loss - entry_price) * PROFIT_TARGET_1
                                target_2 = entry_price + (entry_price - stop_loss) * PROFIT_TARGET_2 if position_type == 'long' else entry_price - (stop_loss - entry_price) * PROFIT_TARGET_2
                                
                                log_trade(
                                    entry_time,
                                    exit_time,
                                    SYMBOL,
                                    position_type,
                                    entry_price,
                                    exit_price_target if exit_price_target else current_price,
                                    abs(qty_before_scale),
                                    entry_price * abs(qty_before_scale),
                                    stop_loss,
                                    target_1,
                                    target_2,
                                    pnl_dollars,
                                    pnl_percent,
                                    hold_minutes,
                                    'target_2_hit',
                                    entry_regime,
                                    entry_strength,
                                    entry_rsi,
                                    entry_adx,
                                    entry_ma_spread,
                                    0
                                )
                                
                                position_active = False
                                position_state.reset()
                                debug_print(f"Sleeping {seconds_to_human_readable(POLL_INTERVAL)} after exit")
                                time.sleep(POLL_INTERVAL)
                                continue
                        
                        if STRATEGY_MODE == "or_fvg" or OR_FVG_ENABLED:
                            stop_hit = False
                            if position_type == 'long' and current_price <= stop_loss:
                                stop_hit = True
                                debug_print(f"OR-FVG long stop hit: ${current_price:.2f} <= ${stop_loss:.2f}")
                            elif position_type == 'short' and current_price >= stop_loss:
                                stop_hit = True
                                debug_print(f"OR-FVG short stop hit: ${current_price:.2f} >= ${stop_loss:.2f}")
                            
                            if stop_hit:
                                qty = current_position_qty(SYMBOL)
                                if qty != 0:
                                    exit_time = datetime.now(EASTERN)
                                    hold_minutes = (exit_time - entry_time).total_seconds() / 60 if entry_time else 0
                                    
                                    if position_type == 'long':
                                        exit_price = submit_market_sell(SYMBOL, qty)
                                        pnl_dollars = (exit_price - entry_price) * qty if exit_price else 0
                                    else:
                                        exit_price = submit_buy_to_cover(SYMBOL, abs(qty))
                                        pnl_dollars = (entry_price - exit_price) * abs(qty) if exit_price else 0
                                    
                                    pnl_percent = (pnl_dollars / (entry_price * abs(qty)) * 100) if entry_price > 0 and qty != 0 else 0
                                    
                                    if pnl_dollars > 0:
                                        winners += 1
                                    elif pnl_dollars < 0:
                                        losers += 1
                                    
                                    risk_pct = abs((entry_price - stop_loss) / entry_price) if entry_price > 0 else 0
                                    target_1 = entry_price + (entry_price - stop_loss) * PROFIT_TARGET_1 if position_type == 'long' else entry_price - (stop_loss - entry_price) * PROFIT_TARGET_1
                                    target_2 = entry_price + (entry_price - stop_loss) * PROFIT_TARGET_2 if position_type == 'long' else entry_price - (stop_loss - entry_price) * PROFIT_TARGET_2
                                    
                                    log_trade(
                                        entry_time,
                                        exit_time,
                                        SYMBOL,
                                        position_type,
                                        entry_price,
                                        exit_price if exit_price else current_price,
                                        abs(qty),
                                        entry_price * abs(qty),
                                        stop_loss,
                                        target_1,
                                        target_2,
                                        pnl_dollars,
                                        pnl_percent,
                                        hold_minutes,
                                        'stop_hit',
                                        entry_regime,
                                        entry_strength,
                                        entry_rsi,
                                        entry_adx,
                                        entry_ma_spread,
                                        0
                                    )
                                    
                                    position_active = False
                                    trade_count += 1
                                    logger.info("🛑  Stop hit")
                                    debug_print("Stop hit, position closed")
                                    position_state.reset()
                                    debug_print(f"Sleeping {seconds_to_human_readable(POLL_INTERVAL)} after exit")
                                    time.sleep(POLL_INTERVAL)
                                    continue
                        elif atr_based_trailing_stop(SYMBOL, entry_price, current_price, stop_loss, position_type):
                            qty = current_position_qty(SYMBOL)
                            if qty != 0:
                                exit_time = datetime.now(EASTERN)
                                hold_minutes = (exit_time - entry_time).total_seconds() / 60 if entry_time else 0
                                
                                if position_type == 'long':
                                    exit_price = submit_market_sell(SYMBOL, qty)
                                    pnl_dollars = (exit_price - entry_price) * qty if exit_price else 0
                                else:
                                    exit_price = submit_buy_to_cover(SYMBOL, abs(qty))
                                    pnl_dollars = (entry_price - exit_price) * abs(qty) if exit_price else 0
                                
                                pnl_percent = (pnl_dollars / (entry_price * abs(qty)) * 100) if entry_price > 0 and qty != 0 else 0
                                
                                if pnl_dollars > 0:
                                    winners += 1
                                elif pnl_dollars < 0:
                                    losers += 1
                                
                                risk_pct = abs((entry_price - stop_loss) / entry_price) if entry_price > 0 else 0
                                target_1 = entry_price + (entry_price - stop_loss) * PROFIT_TARGET_1 if position_type == 'long' else entry_price - (stop_loss - entry_price) * PROFIT_TARGET_1
                                target_2 = entry_price + (entry_price - stop_loss) * PROFIT_TARGET_2 if position_type == 'long' else entry_price - (stop_loss - entry_price) * PROFIT_TARGET_2
                                
                                log_trade(
                                    entry_time,
                                    exit_time,
                                    SYMBOL,
                                    position_type,
                                    entry_price,
                                    exit_price if exit_price else current_price,
                                    abs(qty),
                                    entry_price * abs(qty),
                                    stop_loss,
                                    target_1,
                                    target_2,
                                    pnl_dollars,
                                    pnl_percent,
                                    hold_minutes,
                                    'stop_hit',
                                    entry_regime,
                                    entry_strength,
                                    entry_rsi,
                                    entry_adx,
                                    entry_ma_spread,
                                    0
                                )
                                
                                position_active = False
                                trade_count += 1
                                logger.info("🛑  Stop hit")
                                debug_print("Stop hit, position closed")
                                position_state.reset()
                                debug_print(f"Sleeping {seconds_to_human_readable(POLL_INTERVAL)} after exit")
                                time.sleep(POLL_INTERVAL)
                                continue
                    
                    if STRATEGY_MODE == "or_fvg" or OR_FVG_ENABLED:
                        signal, strength, signal_stop_loss, signal_position_type = or_fvg_signal_generator(SYMBOL)
                    else:
                        signal, strength, signal_stop_loss, signal_position_type = advanced_signal_generator(SYMBOL)
                    
                    bars_for_signal = get_recent_bars(SYMBOL, 50)
                    signal_rsi = 0
                    signal_adx = 0
                    signal_ma_spread = 0
                    if bars_for_signal is not None and len(bars_for_signal) >= LONG_WINDOW:
                        closes = bars_for_signal['close']
                        highs = bars_for_signal['high']
                        lows = bars_for_signal['low']
                        signal_rsi = rsi(closes, 14).iloc[-1]
                        signal_adx = adx(highs, lows, closes).iloc[-1]
                        if USE_EMA:
                            short_ma = ema(closes, SHORT_WINDOW).iloc[-1]
                            long_ma = ema(closes, LONG_WINDOW).iloc[-1]
                        else:
                            short_ma = sma(closes, SHORT_WINDOW).iloc[-1]
                            long_ma = sma(closes, LONG_WINDOW).iloc[-1]
                        signal_ma_spread = short_ma - long_ma
                    
                    if trades_today >= MAX_TRADES_PER_DAY:
                        if signal in ['buy', 'sell'] and strength > 0:
                            log_missed_signal(datetime.now(EASTERN), signal, 'max_trades_per_day', current_price, SYMBOL, strength, signal_rsi, signal_adx, regime)
                        logger.info(f"📊  Daily limit ({MAX_TRADES_PER_DAY}) - monitoring only")
                        debug_print(f"Daily trade limit reached ({trades_today}/{MAX_TRADES_PER_DAY})")
                        time.sleep(POLL_INTERVAL)
                        continue
                    
                    if signal == 'sell' and not ENABLE_SHORT_SELLING:
                        debug_print("Short selling disabled, ignoring sell signal")
                        if signal and strength > 0:
                            log_missed_signal(datetime.now(EASTERN), signal, 'short_selling_disabled', current_price, SYMBOL, strength, signal_rsi, signal_adx, regime)
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
                                
                                entry_strength = strength
                                entry_rsi = signal_rsi
                                entry_adx = signal_adx
                                entry_ma_spread = signal_ma_spread
                                entry_regime = regime
                                
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
                                
                                if STRATEGY_MODE == "or_fvg" or OR_FVG_ENABLED:
                                    or_fvg_state.entry_triggered = True
                                    debug_print("OR-FVG entry_triggered flag set")
                                
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
                            log_missed_signal(datetime.now(EASTERN), signal, 'insufficient_buying_power', current_price, SYMBOL, strength, signal_rsi, signal_adx, regime)
                            
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
                    
                    vix_readings.append(vix_level)
                    regime_readings.append(regime)
                    
                    current_drawdown = (opening_equity - current_equity) / opening_equity if opening_equity > 0 else 0
                    if current_drawdown > max_intraday_drawdown:
                        max_intraday_drawdown = current_drawdown
                    
                    now = datetime.now(EASTERN)
                    if (now - last_indicator_log).total_seconds() >= 300:
                        if bars_for_signal is not None and len(bars_for_signal) > 0:
                            log_indicators(
                                now,
                                SYMBOL,
                                current_price,
                                bars_for_signal['volume'].iloc[-1] if 'volume' in bars_for_signal.columns else 0,
                                signal_rsi,
                                signal_adx,
                                atr(bars_for_signal['high'], bars_for_signal['low'], bars_for_signal['close']).iloc[-1] if len(bars_for_signal) >= 14 else 0,
                                signal_ma_spread,
                                regime,
                                position_status
                            )
                            last_indicator_log = now
                    
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
                
                if position_active and entry_time:
                    exit_time = datetime.now(EASTERN)
                    hold_minutes = (exit_time - entry_time).total_seconds() / 60
                    qty = current_position_qty(SYMBOL)
                    
                    if qty != 0:
                        bars_eod = get_recent_bars(SYMBOL, 10)
                        exit_price = bars_eod['close'].iloc[-1] if bars_eod is not None and len(bars_eod) > 0 else current_price
                        
                        if position_type == 'long':
                            pnl_dollars = (exit_price - entry_price) * qty
                        else:
                            pnl_dollars = (entry_price - exit_price) * abs(qty)
                        
                        pnl_percent = (pnl_dollars / (entry_price * abs(qty)) * 100) if entry_price > 0 and qty != 0 else 0
                        
                        if pnl_dollars > 0:
                            winners += 1
                        elif pnl_dollars < 0:
                            losers += 1
                        
                        risk_pct = abs((entry_price - stop_loss) / entry_price) if entry_price > 0 else 0
                        target_1 = entry_price + (entry_price - stop_loss) * PROFIT_TARGET_1 if position_type == 'long' else entry_price - (stop_loss - entry_price) * PROFIT_TARGET_1
                        target_2 = entry_price + (entry_price - stop_loss) * PROFIT_TARGET_2 if position_type == 'long' else entry_price - (stop_loss - entry_price) * PROFIT_TARGET_2
                        
                        log_trade(
                            entry_time,
                            exit_time,
                            SYMBOL,
                            position_type,
                            entry_price,
                            exit_price,
                            abs(qty),
                            entry_price * abs(qty),
                            stop_loss,
                            target_1,
                            target_2,
                            pnl_dollars,
                            pnl_percent,
                            hold_minutes,
                            'eod_close',
                            entry_regime,
                            entry_strength,
                            entry_rsi,
                            entry_adx,
                            entry_ma_spread,
                            0
                        )
                
                close_all_positions()
                
                final_equity = fetch_equity()
                session_pnl = final_equity - opening_equity
                session_pnl_pct = (session_pnl / opening_equity) * 100 if opening_equity > 0 else 0
                
                avg_vix = sum(vix_readings) / len(vix_readings) if vix_readings else 0
                most_common_regime = max(set(regime_readings), key=regime_readings.count) if regime_readings else 'unknown'
                
                log_daily_performance(
                    session_date,
                    opening_equity,
                    final_equity,
                    trade_count,
                    winners,
                    losers,
                    session_pnl,
                    max_intraday_drawdown,
                    most_common_regime,
                    avg_vix
                )
                
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
