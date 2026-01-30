import pandas as pd

def sma(data, window):
    return data.rolling(window=window).mean()

def ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    loss = loss.replace(0, 0.0001)
    rs = gain / loss
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def atr(high, low, close, window=14):
    high_low = high - low
    high_close_prev = abs(high - close.shift())
    low_close_prev = abs(low - close.shift())
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr_val = true_range.rolling(window=window).mean()
    return atr_val

def adx(high, low, close, window=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window=window).mean()
    up_move = high - high.shift()
    down_move = low.shift() - low
    plus_dm = pd.Series(0.0, index=close.index)
    minus_dm = pd.Series(0.0, index=close.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr_val)
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, 0.0001)
    dx = 100 * abs(plus_di - minus_di) / di_sum
    adx_val = dx.rolling(window=window).mean()
    return adx_val

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger(close, window=20, num_std=2):
    middle = sma(close, window)
    std = close.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower
