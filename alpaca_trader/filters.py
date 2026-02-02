import pandas as pd
from datetime import datetime
from .indicators import ema, sma, rsi, adx, atr, bollinger, macd
from .api import AlpacaClient
from .utils import EASTERN

def check_volume(bars: pd.DataFrame, multiplier: float):
    if len(bars) < 20 or "volume" not in bars.columns:
        return True
    avg = bars["volume"].rolling(window=20).mean().iloc[-1]
    cur = bars["volume"].iloc[-1]
    return cur >= avg * multiplier

def check_candle_pattern(bars: pd.DataFrame):
    if len(bars) < 2:
        return False, False
    last = bars.iloc[-1]
    prev = bars.iloc[-2]
    bullish = last["close"] > last["open"] and prev["close"] < prev["open"] and last["close"] > prev["open"] and last["open"] < prev["close"]
    bearish = last["close"] < last["open"] and prev["close"] > prev["open"] and last["close"] < prev["open"] and last["open"] > prev["close"]
    return bullish, bearish

def check_macd_confirmation(bars: pd.DataFrame):
    if len(bars) < 35:
        return "neutral"
    macd_line, signal_line, _ = macd(bars["close"])
    if macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
        return "bullish"
    if macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
        return "bearish"
    return "neutral"

def check_200_sma_filter(symbol: str, client: AlpacaClient):
    daily = client.get_bars(symbol, "1Day", limit=210)
    if len(daily) < 200:
        return "neutral"
    sma_200 = sma(daily["close"], 200).iloc[-1]
    price = daily["close"].iloc[-1]
    if price > sma_200 * 1.01:
        return "bullish"
    if price < sma_200 * 0.99:
        return "bearish"
    return "neutral"

def check_multiframe_confluence(symbol: str, use_ema: bool, client: AlpacaClient = None):
    if client is None:
        from .engine import api as client
    hourly = client.get_bars(symbol, "1Hour", limit=50)
    if len(hourly) < 50:
        return "neutral"
    if use_ema:
        short = ema(hourly["close"], 20).iloc[-1]
        long = ema(hourly["close"], 50).iloc[-1]
    else:
        short = sma(hourly["close"], 20).iloc[-1]
        long = sma(hourly["close"], 50).iloc[-1]
    price = hourly["close"].iloc[-1]
    if short > long and price > short:
        return "bullish"
    if short < long and price < short:
        return "bearish"
    return "neutral"

def detect_market_regime(bars: pd.DataFrame, adx_threshold: float):
    if len(bars) < 50:
        return "unknown"
    current_adx = adx(bars["high"], bars["low"], bars["close"]).iloc[-1]
    current_atr = atr(bars["high"], bars["low"], bars["close"]).iloc[-1]
    atr_series = atr(bars["high"], bars["low"], bars["close"])
    percentile = (atr_series <= current_atr).mean() * 100
    if percentile > 70:
        return "high_vol"
    if percentile < 30:
        return "low_vol"
    if current_adx > adx_threshold:
        return "trend"
    return "range"

def get_vix(client: AlpacaClient, symbol: str, use_vix_filter: bool):
    if not use_vix_filter:
        return 0
    try:
        vix = client.get_bars("VIX", "1Day", limit=5)
        if len(vix) > 0:
            return vix["close"].iloc[-1]
    except Exception as e:
        print(f"Warning: VIX data unavailable: {e}")
    try:
        spy = client.get_bars(symbol, "1Day", limit=20)
        if len(spy) >= 20:
            returns = spy["close"].pct_change()
            calculated_vix = returns.std() * (252 ** 0.5) * 100
            print(f"Using calculated volatility as VIX proxy: {calculated_vix:.1f}")
            return calculated_vix
    except Exception as e:
        print(f"Warning: Could not calculate volatility: {e}")
    print("Warning: VIX data unavailable, skipping VIX filter for this iteration")
    return 0
