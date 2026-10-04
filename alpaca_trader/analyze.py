import argparse
import sys
from datetime import datetime, timedelta, time as dtime

import numpy as np
import pandas as pd

from .backtest import EASTERN, load_config, fetch_bars, load_csv, prepare_base, timeframe_minutes
from .indicators import sma, rsi, bollinger

HORIZONS = (1, 3, 5, 10)


def daily_table(base):
    g = base.groupby("date")
    daily = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    daily.index = pd.to_datetime(daily.index)
    low_idx = base.groupby("date")["low"].idxmin()
    high_idx = base.groupby("date")["high"].idxmax()
    daily["low_time"] = [pd.Timestamp(ts).tz_convert(EASTERN).strftime("%H:%M") for ts in low_idx.tolist()]
    daily["high_time"] = [pd.Timestamp(ts).tz_convert(EASTERN).strftime("%H:%M") for ts in high_idx.tolist()]
    c = daily["close"]
    daily["ret"] = c.pct_change()
    daily["overnight"] = daily["open"] / c.shift(1) - 1
    daily["intraday"] = daily["close"] / daily["open"] - 1
    daily["rsi2"] = rsi(c, 2)
    daily["sma200"] = sma(c, 200)
    daily["sma5"] = sma(c, 5)
    _, _, lower = bollinger(c, 20, 2)
    daily["bb_lower"] = lower
    daily["high10"] = c.rolling(10).max()
    down = (c < c.shift(1)).astype(int)
    streak = down.groupby((down != down.shift()).cumsum()).cumsum() * down
    daily["down_streak"] = streak
    next_open = daily["open"].shift(-1)
    for h in HORIZONS:
        daily[f"fwd{h}"] = c.shift(-h) / next_open - 1
    return daily


def pct(x):
    return f"{x * 100:+.3f}%"


def section(title):
    print()
    print(title)
    print("-" * len(title))


def summarize(sub, base_sub, label, width=34):
    n = len(sub)
    if n == 0:
        return f"  {label:<{width}}{'n=0':>7}"
    parts = [f"  {label:<{width}}{('n=' + str(n)):>7}"]
    for h in HORIZONS:
        col = f"fwd{h}"
        vals = sub[col].dropna()
        basev = base_sub[col].dropna()
        if len(vals) == 0:
            parts.append(f"{'':>22}")
            continue
        edge = vals.mean() - basev.mean()
        win = (vals > 0).mean() * 100
        parts.append(f"  {h}d {vals.mean() * 100:+6.2f}% w{win:3.0f}% e{edge * 100:+5.2f}")
    return "".join(parts)


def run_conditions(daily, periods):
    conds = [
        ("All days (baseline)", lambda d: d.index == d.index),
        ("1 down day", lambda d: d["down_streak"] == 1),
        ("2 down days in a row", lambda d: d["down_streak"] == 2),
        ("3+ down days in a row", lambda d: d["down_streak"] >= 3),
        ("RSI(2) < 20", lambda d: d["rsi2"] < 20),
        ("RSI(2) < 10", lambda d: d["rsi2"] < 10),
        ("RSI(2) < 5", lambda d: d["rsi2"] < 5),
        ("Close below lower Bollinger", lambda d: d["close"] < d["bb_lower"]),
        ("3%+ below 10 day high", lambda d: d["close"] <= d["high10"] * 0.97),
        ("5%+ below 10 day high", lambda d: d["close"] <= d["high10"] * 0.95),
        ("RSI(2) < 10 and above 200 SMA", lambda d: (d["rsi2"] < 10) & (d["close"] > d["sma200"])),
        ("RSI(2) < 10 and below 200 SMA", lambda d: (d["rsi2"] < 10) & (d["close"] <= d["sma200"])),
        ("3+ down days and above 200 SMA", lambda d: (d["down_streak"] >= 3) & (d["close"] > d["sma200"])),
        ("Up day (for contrast)", lambda d: d["ret"] > 0),
        ("RSI(2) > 90 (overbought)", lambda d: d["rsi2"] > 90),
    ]
    for name, (start, end) in periods:
        p = daily[(daily.index >= start) & (daily.index <= end)].dropna(subset=["sma200"])
        if len(p) == 0:
            continue
        section(f"Dip signals, {name} {p.index[0].date()} to {p.index[-1].date()}  (buy next open, hold N days)")
        print("  avg = average return, w = win rate, e = edge over baseline")
        for label, fn in conds:
            print(summarize(p[fn(p)], p, label))


def run_calendar(daily, periods):
    for name, (start, end) in periods:
        p = daily[(daily.index >= start) & (daily.index <= end)].dropna(subset=["ret", "overnight"])
        if len(p) == 0:
            continue
        section(f"Overnight vs daytime, {name}")
        on = (1 + p["overnight"]).prod() - 1
        intra = (1 + p["intraday"]).prod() - 1
        print(f"  Overnight (close to open)  total {on * 100:+8.2f}%  avg {pct(p['overnight'].mean())}  up {(p['overnight'] > 0).mean() * 100:.0f}%")
        print(f"  Daytime (open to close)    total {intra * 100:+8.2f}%  avg {pct(p['intraday'].mean())}  up {(p['intraday'] > 0).mean() * 100:.0f}%")
        section(f"Day of week (close to close), {name}")
        names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        for dow in range(5):
            s = p[p.index.dayofweek == dow]["ret"]
            print(f"  {names[dow]}  n={len(s):<4} avg {pct(s.mean())}  up {(s > 0).mean() * 100:.0f}%")


def run_intraday(base, daily, periods):
    for name, (start, end) in periods:
        mask = (base.index >= pd.Timestamp(start).tz_localize(EASTERN)) & (base.index <= pd.Timestamp(end).tz_localize(EASTERN) + pd.Timedelta(days=1))
        b = base[mask]
        d = daily[(daily.index >= start) & (daily.index <= end)]
        if len(b) == 0:
            continue
        opens = b.groupby("date")["open"].transform("first")
        rel = b["close"] / opens - 1
        bucket = b.index.floor("30min").strftime("%H:%M")
        path = rel.groupby(bucket).mean()
        lows = d["low_time"].apply(lambda s: f"{s[:2]}:{'00' if int(s[3:]) < 30 else '30'}").value_counts(normalize=True)
        highs = d["high_time"].apply(lambda s: f"{s[:2]}:{'00' if int(s[3:]) < 30 else '30'}").value_counts(normalize=True)
        section(f"Time of day, {name}  (avg price vs open, and where the day's low and high happen)")
        cheapest = path.idxmin()
        for slot in sorted(path.index):
            marker = "  <- cheapest on average" if slot == cheapest else ""
            print(f"  {slot}  vs open {pct(path[slot])}   low here {lows.get(slot, 0) * 100:5.1f}%   high here {highs.get(slot, 0) * 100:5.1f}%{marker}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Look for buy low patterns in historical data")
    today = datetime.now(EASTERN).date()
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default=(today - timedelta(days=1)).isoformat())
    parser.add_argument("--split", default="2023-01-01", help="Discovery period ends here, validation starts here")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--base", default="5Min")
    parser.add_argument("--feed", default="sip", choices=["sip", "iex"])
    parser.add_argument("--data", default=None)
    args = parser.parse_args(argv)

    config = load_config()
    symbol = args.symbol or config.get("SYMBOL", "SPY")
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    split = datetime.strptime(args.split, "%Y-%m-%d")
    if not start < split < end:
        sys.exit("--split must fall between --start and --end")

    raw = load_csv(args.data) if args.data else fetch_bars(symbol, start - timedelta(days=330), end, args.base, args.feed)
    base = prepare_base(raw, timeframe_minutes(args.base))
    daily = daily_table(base)
    print(f"{symbol}: {len(daily)} trading days from {daily.index[0].date()} to {daily.index[-1].date()}")

    periods = [
        ("DISCOVERY", (pd.Timestamp(start), pd.Timestamp(split) - pd.Timedelta(days=1))),
        ("VALIDATION", (pd.Timestamp(split), pd.Timestamp(end))),
    ]
    run_calendar(daily, periods)
    run_intraday(base, daily, periods)
    run_conditions(daily, periods)
    print()
    print("A pattern is only worth trading if its edge has the same sign and similar size in BOTH periods.")


if __name__ == "__main__":
    main()
