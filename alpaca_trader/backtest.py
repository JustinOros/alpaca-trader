import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from .indicators import atr
from .strategy import StrategyConfig, evaluate_signal, completed_bars, trend_flipped, mean_reversion_exit, overnight_exit_due

EASTERN = pytz.timezone("US/Eastern")
PKG_DIR = Path(__file__).parent
CONFIG_PATH = PKG_DIR / "config.json"
ENV_PATH = PKG_DIR / ".env"
CACHE_DIR = PKG_DIR / "backtest_cache"
TRADES_OUT = PKG_DIR / "backtest_trades.csv"
EQUITY_OUT = PKG_DIR / "backtest_equity.csv"
WARMUP_CALENDAR_DAYS = 330
SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)
OHLCV = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def timeframe_minutes(timeframe):
    tf = str(timeframe).lower()
    for suffix, mult in (("min", 1), ("hour", 60), ("day", 390)):
        if tf.endswith(suffix):
            num = tf[: -len(suffix)]
            return (int(num) if num.isdigit() else 1) * mult
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def load_config():
    if not CONFIG_PATH.exists():
        sys.exit(f"Missing {CONFIG_PATH}, run the bot once to create it")
    with open(CONFIG_PATH) as f:
        return json.load(f)


def month_ranges(start, end):
    cur = datetime(start.year, start.month, 1)
    while cur <= end:
        nxt = datetime(cur.year + (cur.month // 12), cur.month % 12 + 1, 1)
        yield cur, nxt
        cur = nxt


def fetch_bars(symbol, start, end, base_tf, feed):
    from dotenv import load_dotenv
    import alpaca_trade_api as tradeapi

    load_dotenv(ENV_PATH)
    api = tradeapi.REST(
        os.getenv("APCA_API_KEY_ID"),
        os.getenv("APCA_API_SECRET_KEY"),
        os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets"),
        api_version="v2",
    )
    CACHE_DIR.mkdir(exist_ok=True)
    this_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    frames = []
    for m_start, m_end in month_ranges(start, end):
        cache_file = CACHE_DIR / f"{symbol}_{base_tf}_{feed}_{m_start:%Y-%m}.csv"
        if cache_file.exists() and m_start < this_month:
            df = pd.read_csv(cache_file, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            q_end = min(m_end, end + timedelta(days=1))
            print(f"Downloading {symbol} {base_tf} {m_start:%Y-%m} ({feed})...", flush=True)
            df = api.get_bars(
                symbol,
                base_tf,
                start=m_start.strftime("%Y-%m-%dT00:00:00Z"),
                end=q_end.strftime("%Y-%m-%dT00:00:00Z"),
                feed=feed,
                adjustment="raw",
            ).df
            if df is None:
                df = pd.DataFrame(columns=list(OHLCV))
            df = df[[c for c in OHLCV if c in df.columns]]
            if m_start < this_month:
                df.to_csv(cache_file)
        if len(df):
            frames.append(df)
    if not frames:
        sys.exit("No bars returned from Alpaca")
    return pd.concat(frames)


def load_csv(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def prepare_base(df, base_minutes):
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_convert(EASTERN)
    t = df.index.time
    df = df[(t >= SESSION_OPEN) & (t < SESSION_CLOSE)].copy()
    df["end"] = df.index + pd.Timedelta(minutes=base_minutes)
    df["date"] = df.index.date
    return df


def build_daily(base):
    daily = base.groupby("date").agg(OHLCV)
    daily.index = pd.to_datetime(daily.index).tz_localize(EASTERN)
    return daily


class SignalBarBuilder:
    def __init__(self, base, signal_tf, history_bars):
        self.base = base
        self.minutes = timeframe_minutes(signal_tf)
        self.daily_mode = self.minutes >= 390
        self.history_bars = history_bars
        self.daily = build_daily(base)
        self.daily_dates = np.array([d.date() for d in self.daily.index])
        self.base_ends = base["end"].values
        self.rule = f"{self.minutes}min"
        self.rows_needed = int(math.ceil(history_bars * self.minutes / max(1, base_minutes_of(base)))) + 200
        self.days_by_date = {d: g for d, g in base.groupby("date")}

    def daily_with_partial(self, day, upto):
        hist_count = int(np.searchsorted(self.daily_dates, day, side="left"))
        hist = self.daily.iloc[max(0, hist_count - self.history_bars - 5):hist_count]
        day_bars = self.days_by_date.get(day)
        today = day_bars[day_bars["end"] <= upto] if day_bars is not None else day_bars
        if today is None or len(today) == 0:
            return hist
        partial = pd.DataFrame(
            [{
                "open": today["open"].iloc[0],
                "high": today["high"].max(),
                "low": today["low"].min(),
                "close": today["close"].iloc[-1],
                "volume": today["volume"].sum(),
            }],
            index=[pd.Timestamp(day).tz_localize(EASTERN)],
        )
        return pd.concat([hist, partial])

    def bars_at(self, day, upto):
        if self.daily_mode:
            return self.daily_with_partial(day, upto).tail(self.history_bars)
        n = int(np.searchsorted(self.base_ends, np.datetime64(upto.astimezone(pytz.utc).replace(tzinfo=None)), side="right"))
        chunk = self.base.iloc[max(0, n - self.rows_needed):n]
        if len(chunk) == 0:
            return chunk
        out = chunk[list(OHLCV)].resample(self.rule, origin="start_day", offset="9h30min").agg(OHLCV).dropna()
        return out.tail(self.history_bars)


def base_minutes_of(base):
    if len(base) < 2:
        return 5
    return int((base["end"].iloc[0] - base.index[0]).total_seconds() // 60)


class Backtester:
    def __init__(self, config, base, start_date, end_date, capital):
        self.c = config
        self.cfg = StrategyConfig(config)
        self.base = base
        self.start_date = start_date
        self.end_date = end_date
        self.capital = capital
        self.signal_tf = config.get("BAR_TIMEFRAME", "15Min")
        self.builder = SignalBarBuilder(base, self.signal_tf, 250)
        self.poll = int(config.get("POLL_INTERVAL", 300))
        self.eod_minutes = int(config.get("EOD_CLOSE_MINUTES", 10))
        self.risk = float(config.get("RISK_PER_TRADE", 0.01))
        self.max_dd = float(config.get("MAX_DRAWDOWN", 0.08))
        self.max_trades = int(config.get("MAX_TRADES_PER_DAY", 3))
        self.max_hold = int(config.get("MAX_HOLD_TIME", 0))
        self.pt1 = float(config.get("PROFIT_TARGET_1", 2.0))
        self.pt2 = float(config.get("PROFIT_TARGET_2", 3.0))
        self.trailing = bool(config.get("USE_TRAILING_STOP", True))
        self.atr_mult = float(config.get("ATR_STOP_MULTIPLIER", 2.0))
        self.shorts = bool(config.get("ENABLE_SHORT_SELLING", False))
        self.slip = float(config.get("SLIPPAGE_PCT", 0.0)) if config.get("ENABLE_SLIPPAGE", False) else 0.0
        self.comm = float(config.get("COMMISSION_PCT", 0.0))
        self.min_notional = float(config.get("MIN_NOTIONAL", 1.0))
        self.use_200 = bool(config.get("USE_200_SMA_FILTER", False))
        self.sizing = str(config.get("POSITION_SIZING", "risk")).lower()
        self.max_position_pct = float(config.get("MAX_POSITION_PCT", 0.25))
        self.hold_overnight = bool(config.get("HOLD_OVERNIGHT", False))
        self.exit_on_flip = bool(config.get("EXIT_ON_TREND_FLIP", False))
        self.cash = capital
        self.pos = None
        self.trades = []
        self.equity_rows = []
        self.days_by_date = self.builder.days_by_date

    def fill_price(self, day_bars, t, side):
        nxt = day_bars[day_bars.index >= t]
        raw = float(nxt["open"].iloc[0]) if len(nxt) else float(day_bars["close"].iloc[-1])
        return raw * (1 + self.slip) if side == "buy" else raw * (1 - self.slip)

    def equity(self, price):
        if self.pos is None:
            return self.cash
        if self.pos["type"] == "long":
            return self.cash + self.pos["shares"] * price
        return self.cash - self.pos["shares"] * price

    def exposure(self, price):
        if self.pos is None:
            return 0.0
        eq = self.equity(price)
        return self.pos["shares"] * price / eq if eq > 0 else 0.0

    def open_position(self, sig, price, t, equity, day_bars):
        cap = equity * self.max_position_pct
        if self.sizing == "fixed":
            value = cap
        else:
            price_risk = abs(price - sig["stop"])
            value = self.min_notional if price_risk == 0 else min((equity * self.risk / price_risk) * price, cap)
        value = max(self.min_notional, value)
        side = "buy" if sig["position_type"] == "long" else "sell"
        fill = self.fill_price(day_bars, t, side)
        shares = int(value / fill)
        if shares <= 0 or value > self.cash:
            return False
        cost = shares * fill * self.comm
        if sig["position_type"] == "long":
            self.cash -= shares * fill + cost
        else:
            self.cash += shares * fill - cost
        self.pos = {
            "type": sig["position_type"],
            "entry_time": t,
            "entry": fill,
            "shares": shares,
            "initial_shares": shares,
            "stop": sig["stop"],
            "trail": sig["stop"],
            "t1_hit": False,
            "realized": -cost,
            "reason_in": sig["reason"],
            "strength": sig["strength"],
            "rsi": sig["rsi"],
            "adx": sig["adx"],
            "regime": sig["regime"],
        }
        return True

    def reduce(self, shares, t, day_bars):
        p = self.pos
        side = "sell" if p["type"] == "long" else "buy"
        fill = self.fill_price(day_bars, t, side)
        cost = shares * fill * self.comm
        if p["type"] == "long":
            self.cash += shares * fill - cost
            p["realized"] += (fill - p["entry"]) * shares - cost
        else:
            self.cash -= shares * fill + cost
            p["realized"] += (p["entry"] - fill) * shares - cost
        p["shares"] -= shares
        return fill

    def close(self, t, day_bars, reason):
        p = self.pos
        fill = self.reduce(p["shares"], t, day_bars)
        self.trades.append({
            "entry_time": p["entry_time"],
            "exit_time": t,
            "side": p["type"],
            "entry_price": round(p["entry"], 4),
            "exit_price": round(fill, 4),
            "shares": p["initial_shares"],
            "pnl_dollars": round(p["realized"], 2),
            "pnl_percent": round(p["realized"] / (p["entry"] * p["initial_shares"]) * 100, 4),
            "hold_minutes": round((t - p["entry_time"]).total_seconds() / 60, 1),
            "hold_days": (t.date() - p["entry_time"].date()).days,
            "exit_reason": reason,
            "scaled_out": p["t1_hit"],
            "entry_reason": p["reason_in"],
            "regime": p["regime"],
            "strength": round(p["strength"], 3),
            "rsi": round(p["rsi"], 2),
            "adx": round(p["adx"], 2),
        })
        self.pos = None

    def manage(self, t, price, bars, day_bars):
        p = self.pos
        if self.cfg.strategy_mode == "overnight" and overnight_exit_due(p["entry_time"], t):
            self.close(t, day_bars, "overnight_exit")
            return True
        if self.max_hold > 0 and (t - p["entry_time"]).total_seconds() > self.max_hold:
            self.close(t, day_bars, "max_hold_time")
            return True
        if p["type"] == "long":
            profit_pct = (price - p["entry"]) / p["entry"] * 100
        else:
            profit_pct = (p["entry"] - price) / p["entry"] * 100
        risk_pct = abs(p["entry"] - p["stop"]) / p["entry"] * 100
        if self.pt1 > 0 and profit_pct >= risk_pct * self.pt1 and not p["t1_hit"]:
            half = int(p["shares"] / 2)
            if half > 0:
                self.reduce(half, t, day_bars)
            p["t1_hit"] = True
        if self.pt2 > 0 and profit_pct >= risk_pct * self.pt2:
            self.close(t, day_bars, "target_2_hit")
            return True
        if not self.trailing:
            hit = price <= p["stop"] if p["type"] == "long" else price >= p["stop"]
        else:
            tail = bars.tail(50)
            cur_atr = float(atr(tail["high"], tail["low"], tail["close"]).iloc[-1]) if len(tail) >= 14 else float("nan")
            if not math.isnan(cur_atr) and cur_atr > 0:
                if p["type"] == "long":
                    p["trail"] = max(p["trail"], price - cur_atr * self.atr_mult)
                else:
                    p["trail"] = min(p["trail"], price + cur_atr * self.atr_mult)
            hit = price <= p["trail"] if p["type"] == "long" else price >= p["trail"]
        if hit:
            self.close(t, day_bars, "stop_hit")
            return True
        if self.cfg.strategy_mode == "mean_reversion":
            reason = mean_reversion_exit(completed_bars(bars, self.signal_tf, t), self.cfg, p["entry_time"])
            if reason:
                self.close(t, day_bars, reason)
                return True
        if self.exit_on_flip and trend_flipped(completed_bars(bars, self.signal_tf, t), self.cfg, p["type"]):
            self.close(t, day_bars, "trend_flip")
            return True
        return False

    def run(self):
        days = [d for d in sorted(self.days_by_date) if self.start_date <= d <= self.end_date]
        signal_counts = {}
        for i, day in enumerate(days):
            day_bars = self.days_by_date[day]
            open_t = EASTERN.localize(datetime.combine(day, SESSION_OPEN))
            close_t = day_bars["end"].iloc[-1]
            last_poll = close_t - pd.Timedelta(minutes=self.eod_minutes)
            opening_equity = self.equity(float(day_bars["open"].iloc[0]))
            trades_today = 0
            halted = False
            t = open_t
            last_price = float(day_bars["open"].iloc[0])
            while t < last_poll:
                done = day_bars[day_bars["end"] <= t]
                bars = self.builder.bars_at(day, t)
                if len(bars) == 0:
                    t += pd.Timedelta(seconds=self.poll)
                    continue
                price = float(done["close"].iloc[-1]) if len(done) else float(bars["close"].iloc[-1])
                last_price = price
                eq = self.equity(price)
                if not halted and opening_equity > 0 and (opening_equity - eq) / opening_equity > self.max_dd:
                    if self.pos:
                        self.close(t, day_bars, "max_drawdown")
                    halted = True
                if halted:
                    t += pd.Timedelta(seconds=self.poll)
                    continue
                if self.pos and self.manage(t, price, bars, day_bars):
                    t += pd.Timedelta(seconds=self.poll)
                    continue
                if self.pos is None and trades_today < self.max_trades:
                    daily = None
                    if self.use_200:
                        daily = self.builder.daily_with_partial(day, t).tail(210)
                    sig_bars = completed_bars(bars, self.signal_tf, t) if self.hold_overnight else bars
                    sig = evaluate_signal(sig_bars, self.cfg, daily=daily, bars_completed=self.hold_overnight)
                    key = sig["signal"] or sig["reason"].split(" ")[0]
                    signal_counts[key] = signal_counts.get(key, 0) + 1
                    if sig["signal"] == "sell" and not self.shorts:
                        sig = None
                    if sig and sig["signal"] in ("buy", "sell"):
                        if self.open_position(sig, price, t, self.equity(price), day_bars):
                            trades_today += 1
                t += pd.Timedelta(seconds=self.poll)
            if self.cfg.strategy_mode == "overnight" and self.pos is None and not halted and trades_today < self.max_trades and i < len(days) - 1:
                bars = self.builder.bars_at(day, last_poll)
                done = day_bars[day_bars["end"] <= last_poll]
                if len(bars) and len(done):
                    price = float(done["close"].iloc[-1])
                    sig_bars = completed_bars(bars, self.signal_tf, last_poll)
                    sig = evaluate_signal(sig_bars, self.cfg, entry_window=True)
                    key = sig["signal"] or sig["reason"].split(" ")[0]
                    signal_counts[key] = signal_counts.get(key, 0) + 1
                    if sig["signal"] == "buy" and self.open_position(sig, price, last_poll, self.equity(price), day_bars):
                        trades_today += 1
            if self.pos and (not self.hold_overnight or i == len(days) - 1):
                self.close(last_poll, day_bars, "eod_close" if not self.hold_overnight else "end_of_test")
            close_price = float(day_bars["close"].iloc[-1])
            self.equity_rows.append({"date": day, "equity": round(self.equity(close_price), 2), "close": close_price, "trades": trades_today, "in_market": int(self.pos is not None or trades_today > 0), "exposure": round(self.exposure(close_price), 4)})
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(days)} days simulated", flush=True)
        return signal_counts


def wilson(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom, (centre + margin) / denom


def report(bt, signal_counts):
    trades = pd.DataFrame(bt.trades)
    eq = pd.DataFrame(bt.equity_rows)
    trades.to_csv(TRADES_OUT, index=False)
    eq.to_csv(EQUITY_OUT, index=False)
    days = len(eq)
    print()
    print("=" * 56)
    print(f"Backtest {eq['date'].iloc[0]} to {eq['date'].iloc[-1]} ({days} trading days)")
    print(f"Symbol {bt.c.get('SYMBOL')}  Signal TF {bt.signal_tf}  Poll {bt.poll}s")
    print("=" * 56)
    start_eq = bt.capital
    end_eq = float(eq["equity"].iloc[-1])
    total_ret = (end_eq / start_eq - 1) * 100
    years = days / 252
    cagr = ((end_eq / start_eq) ** (1 / years) - 1) * 100 if years > 0 and end_eq > 0 else 0
    curve = eq["equity"]
    max_dd = ((curve.cummax() - curve) / curve.cummax()).max() * 100
    rets = curve.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0
    bh = (eq["close"].iloc[-1] / eq["close"].iloc[0] - 1) * 100
    print(f"Start equity        ${start_eq:,.2f}")
    print(f"End equity          ${end_eq:,.2f}")
    bh_curve = eq["close"] / eq["close"].iloc[0]
    bh_dd = ((bh_curve.cummax() - bh_curve) / bh_curve.cummax()).max() * 100
    bh_rets = bh_curve.pct_change().dropna()
    bh_sharpe = (bh_rets.mean() / bh_rets.std() * math.sqrt(252)) if bh_rets.std() > 0 else 0
    print(f"Total return        {total_ret:+.2f}%   (buy and hold {bh:+.2f}%)")
    print(f"CAGR                {cagr:+.2f}%")
    print(f"Max drawdown        {max_dd:.2f}%   (buy and hold {bh_dd:.2f}%)")
    print(f"Sharpe (daily)      {sharpe:.2f}   (buy and hold {bh_sharpe:.2f})")
    print()
    n = len(trades)
    if n == 0:
        print("No trades taken.")
    else:
        wins = trades[trades["pnl_dollars"] > 0]
        losses = trades[trades["pnl_dollars"] <= 0]
        lo, hi = wilson(len(wins), n)
        gross_win = wins["pnl_dollars"].sum()
        gross_loss = -losses["pnl_dollars"].sum()
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        streak = longest = 0
        for v in trades["pnl_dollars"]:
            streak = streak + 1 if v <= 0 else 0
            longest = max(longest, streak)
        print(f"Trades              {n}  ({n / days * 100:.1f}% of days had a trade)")
        print(f"Win rate            {len(wins) / n * 100:.1f}%  (95% range {lo * 100:.1f}% to {hi * 100:.1f}%)")
        print(f"Avg win             ${wins['pnl_dollars'].mean() if len(wins) else 0:,.2f}")
        print(f"Avg loss            ${losses['pnl_dollars'].mean() if len(losses) else 0:,.2f}")
        print(f"Expectancy/trade    ${trades['pnl_dollars'].mean():,.2f}")
        print(f"Profit factor       {pf:.2f}")
        print(f"Best / worst        ${trades['pnl_dollars'].max():,.2f} / ${trades['pnl_dollars'].min():,.2f}")
        print(f"Longest loss streak {longest}")
        print(f"Avg hold            {trades['hold_minutes'].mean():.0f} min ({trades['hold_days'].mean():.1f} days)")
        print(f"Time in market      {eq['in_market'].mean() * 100:.1f}% of days")
        print(f"Avg exposure        {eq['exposure'].mean() * 100:.1f}% of equity")
        print()
        print("Exit reasons:")
        for reason, count in trades["exit_reason"].value_counts().items():
            sub = trades[trades["exit_reason"] == reason]
            print(f"  {reason:<16}{count:>5}   avg ${sub['pnl_dollars'].mean():,.2f}")
    print()
    print("Signal checks (when flat):")
    for k, v in sorted(signal_counts.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {k:<32}{v:>7}")
    print()
    print(f"Trades written to {TRADES_OUT}")
    print(f"Equity curve written to {EQUITY_OUT}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Backtest the alpaca-trader strategy using config.json")
    today = datetime.now(EASTERN).date()
    parser.add_argument("--start", default=(today - timedelta(days=365 * 3)).isoformat())
    parser.add_argument("--end", default=(today - timedelta(days=1)).isoformat())
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--base", default="5Min", help="Execution bar size used to simulate intraday polling")
    parser.add_argument("--feed", default="sip", choices=["sip", "iex"])
    parser.add_argument("--data", default=None, help="Use a local CSV of base bars instead of downloading")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config.json value for this run, repeatable")
    args = parser.parse_args(argv)

    config = load_config()
    for item in args.set:
        if "=" not in item:
            sys.exit(f"--set expects KEY=VALUE, got {item}")
        key, value = item.split("=", 1)
        try:
            config[key] = json.loads(value)
        except json.JSONDecodeError:
            config[key] = value
        print(f"Override {key} = {config[key]}")
    if config.get("STRATEGY_MODE") == "or_fvg" or config.get("OR_FVG_ENABLED"):
        sys.exit("OR-FVG mode is not supported by the backtester yet")
    if str(config.get("STRATEGY_MODE", "")).lower() in ("mean_reversion", "overnight") and not config.get("HOLD_OVERNIGHT"):
        sys.exit(f"STRATEGY_MODE {config.get('STRATEGY_MODE')} requires HOLD_OVERNIGHT=true")
    symbol = args.symbol or config.get("SYMBOL", "SPY")
    config["SYMBOL"] = symbol
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    base_minutes = timeframe_minutes(args.base)
    if base_minutes >= 390:
        sys.exit("--base must be an intraday timeframe like 1Min or 5Min")
    signal_minutes = timeframe_minutes(config.get("BAR_TIMEFRAME", "15Min"))
    if signal_minutes < 390 and signal_minutes % base_minutes != 0:
        sys.exit(f"BAR_TIMEFRAME must be a multiple of --base ({args.base})")

    if args.data:
        raw = load_csv(args.data)
    else:
        fetch_start = datetime.combine(start_date, dtime()) - timedelta(days=WARMUP_CALENDAR_DAYS)
        fetch_end = datetime.combine(end_date, dtime())
        raw = fetch_bars(symbol, fetch_start, fetch_end, args.base, args.feed)

    base = prepare_base(raw, base_minutes)
    print(f"Loaded {len(base):,} {args.base} bars from {base.index[0].date()} to {base.index[-1].date()}")
    bt = Backtester(config, base, start_date, end_date, args.capital)
    counts = bt.run()
    if not bt.equity_rows:
        sys.exit("No trading days in the selected range")
    report(bt, counts)


if __name__ == "__main__":
    main()
