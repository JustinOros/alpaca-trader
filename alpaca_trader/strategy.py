import math
from datetime import timedelta
import pandas as pd
from .indicators import sma, ema, rsi, adx, atr, bollinger
from .filters import check_volume, check_candle_pattern, check_macd_confirmation, detect_market_regime


def timeframe_delta(timeframe):
    tf = str(timeframe).lower()
    for suffix, unit in (("min", "minutes"), ("hour", "hours"), ("day", "days"), ("week", "weeks")):
        if tf.endswith(suffix):
            num = tf[: -len(suffix)]
            return timedelta(**{unit: int(num) if num.isdigit() else 1})
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def completed_bars(bars, timeframe, now):
    if bars is None or len(bars) == 0:
        return bars
    delta = timeframe_delta(timeframe)
    ends = bars.index + delta
    return bars[ends <= now]


class StrategyConfig:
    def __init__(self, config):
        self.short_window = int(config.get("SHORT_WINDOW", 20))
        self.long_window = int(config.get("LONG_WINDOW", 50))
        self.use_ema = bool(config.get("USE_EMA", True))
        self.bb_window = int(config.get("BB_WINDOW", 20))
        self.bb_std = float(config.get("BB_STD", 2.0))
        self.atr_stop_multiplier = float(config.get("ATR_STOP_MULTIPLIER", 2.0))
        self.adx_threshold = float(config.get("ADX_THRESHOLD", 25))
        self.volume_multiplier = float(config.get("VOLUME_MULTIPLIER", 0.7))
        self.regime_detection = bool(config.get("REGIME_DETECTION", True))
        self.require_candle_pattern = bool(config.get("REQUIRE_CANDLE_PATTERN", False))
        self.require_macd_confirmation = bool(config.get("REQUIRE_MACD_CONFIRMATION", False))
        self.require_ma_crossover = bool(config.get("REQUIRE_MA_CROSSOVER", True))
        self.crossover_lookback = int(config.get("CROSSOVER_LOOKBACK", 5))
        self.use_200_sma_filter = bool(config.get("USE_200_SMA_FILTER", False))
        self.use_vix_filter = bool(config.get("USE_VIX_FILTER", False))
        self.vix_threshold = float(config.get("VIX_THRESHOLD", 30))
        self.rsi_buy_max = float(config.get("RSI_BUY_MAX", 55))
        self.rsi_sell_min = float(config.get("RSI_SELL_MIN", 45))
        self.rsi_sell_max = float(config.get("RSI_SELL_MAX", 70))
        self.rsi_range_oversold = float(config.get("RSI_RANGE_OVERSOLD", 30))
        self.rsi_range_overbought = float(config.get("RSI_RANGE_OVERBOUGHT", 70))
        self.min_signal_strength = float(config.get("MIN_SIGNAL_STRENGTH", 0.4))


def _ma(closes, window, use_ema):
    return ema(closes, window) if use_ema else sma(closes, window)


def recent_crossover(short_series, long_series, lookback, bullish):
    diff = (short_series - long_series).dropna()
    if len(diff) < lookback + 1:
        return False
    for i in range(1, lookback + 1):
        cur = diff.iloc[-i]
        prev = diff.iloc[-i - 1]
        if bullish and prev <= 0 < cur:
            return True
        if not bullish and prev >= 0 > cur:
            return True
    return False


def above_200_sma(daily):
    if daily is None or len(daily) < 200:
        return True
    sma_200 = sma(daily["close"], 200).iloc[-1]
    return daily["close"].iloc[-1] >= sma_200 * 0.99


def _result(signal=None, strength=0.0, stop=0.0, position_type=None, reason="", **metrics):
    out = {
        "signal": signal,
        "strength": strength,
        "stop": stop,
        "position_type": position_type,
        "reason": reason,
        "price": math.nan,
        "rsi": math.nan,
        "adx": math.nan,
        "atr": math.nan,
        "ma_spread": math.nan,
        "regime": "unknown",
    }
    out.update(metrics)
    return out


def trend_flipped(bars, cfg, position_type):
    if bars is None or len(bars) < cfg.long_window:
        return False
    short_ma = float(_ma(bars["close"], cfg.short_window, cfg.use_ema).iloc[-1])
    long_ma = float(_ma(bars["close"], cfg.long_window, cfg.use_ema).iloc[-1])
    if position_type == "long":
        return short_ma < long_ma
    return short_ma > long_ma


def evaluate_signal(bars, cfg, daily=None, vix=0.0, bars_completed=False):
    if bars is None or len(bars) < cfg.long_window:
        return _result(reason="insufficient_data")

    closes = bars["close"]
    highs = bars["high"]
    lows = bars["low"]
    current_price = float(closes.iloc[-1])

    short_series = _ma(closes, cfg.short_window, cfg.use_ema)
    long_series = _ma(closes, cfg.long_window, cfg.use_ema)
    short_ma = float(short_series.iloc[-1])
    long_ma = float(long_series.iloc[-1])

    rsi_val = float(rsi(closes, 14).iloc[-1])
    adx_val = float(adx(highs, lows, closes).iloc[-1])
    atr_val = float(atr(highs, lows, closes).iloc[-1])
    upper, middle, lower = bollinger(closes, cfg.bb_window, cfg.bb_std)
    regime = detect_market_regime(bars, cfg.adx_threshold) if cfg.regime_detection else "trend"

    metrics = {
        "price": current_price,
        "rsi": rsi_val,
        "adx": adx_val,
        "atr": atr_val,
        "ma_spread": short_ma - long_ma,
        "regime": regime,
    }

    if cfg.use_vix_filter and vix > cfg.vix_threshold:
        return _result(reason=f"vix_filter {vix:.1f}>{cfg.vix_threshold}", **metrics)

    volume_bars = bars if bars_completed else bars.iloc[:-1]
    if not check_volume(volume_bars, cfg.volume_multiplier):
        return _result(reason="volume_filter", **metrics)

    if cfg.use_200_sma_filter and not above_200_sma(daily):
        return _result(reason="below_200_sma", **metrics)

    if math.isnan(atr_val) or atr_val <= 0:
        return _result(reason="invalid_atr", **metrics)

    bullish_pattern, bearish_pattern = check_candle_pattern(bars)
    macd_signal = check_macd_confirmation(bars)
    effective_regime = "trend" if regime in ("high_vol", "low_vol") else regime

    if effective_regime == "trend":
        if short_ma > long_ma and rsi_val < cfg.rsi_buy_max:
            if cfg.require_ma_crossover and not recent_crossover(short_series, long_series, cfg.crossover_lookback, True):
                return _result(reason="no_recent_bullish_crossover", **metrics)
            if cfg.require_candle_pattern and not bullish_pattern:
                return _result(reason="no_bullish_candle", **metrics)
            if cfg.require_macd_confirmation and macd_signal != "bullish":
                return _result(reason="no_bullish_macd", **metrics)
            strength = min(1.0, (adx_val / 40) * 0.7 + 0.3)
            if strength < cfg.min_signal_strength:
                return _result(reason=f"weak_signal {strength:.2f}", **metrics)
            return _result("buy", strength, current_price - atr_val * cfg.atr_stop_multiplier, "long", "trend_buy", **metrics)
        if short_ma < long_ma and cfg.rsi_sell_min < rsi_val < cfg.rsi_sell_max:
            if cfg.require_ma_crossover and not recent_crossover(short_series, long_series, cfg.crossover_lookback, False):
                return _result(reason="no_recent_bearish_crossover", **metrics)
            if cfg.require_candle_pattern and not bearish_pattern:
                return _result(reason="no_bearish_candle", **metrics)
            if cfg.require_macd_confirmation and macd_signal != "bearish":
                return _result(reason="no_bearish_macd", **metrics)
            strength = min(1.0, (adx_val / 40) * 0.7 + 0.3)
            if strength < cfg.min_signal_strength:
                return _result(reason=f"weak_signal {strength:.2f}", **metrics)
            return _result("sell", strength, current_price + atr_val * cfg.atr_stop_multiplier, "short", "trend_sell", **metrics)
        return _result(reason="trend_conditions_not_met", **metrics)

    if effective_regime == "range":
        if current_price <= lower.iloc[-1] and rsi_val < cfg.rsi_range_oversold:
            if cfg.require_candle_pattern and not bullish_pattern:
                return _result(reason="no_bullish_candle", **metrics)
            if cfg.require_macd_confirmation and macd_signal != "bullish":
                return _result(reason="no_bullish_macd", **metrics)
            if 0.85 < cfg.min_signal_strength:
                return _result(reason="weak_signal 0.85", **metrics)
            return _result("buy", 0.85, current_price - atr_val * cfg.atr_stop_multiplier, "long", "range_buy", **metrics)
        if current_price >= upper.iloc[-1] and rsi_val > cfg.rsi_range_overbought:
            if cfg.require_candle_pattern and not bearish_pattern:
                return _result(reason="no_bearish_candle", **metrics)
            if cfg.require_macd_confirmation and macd_signal != "bearish":
                return _result(reason="no_bearish_macd", **metrics)
            if 0.85 < cfg.min_signal_strength:
                return _result(reason="weak_signal 0.85", **metrics)
            return _result("sell", 0.85, current_price + atr_val * cfg.atr_stop_multiplier, "short", "range_sell", **metrics)
        return _result(reason="range_conditions_not_met", **metrics)

    return _result(reason=f"regime_{regime}", **metrics)
