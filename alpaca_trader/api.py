import time
import math
import logging
from datetime import datetime, timedelta, timezone
import backoff
import concurrent.futures
import alpaca_trade_api as tradeapi
import requests.exceptions

logging.getLogger('backoff').setLevel(logging.CRITICAL)

BARS_REQUEST_TIMEOUT = 30  

_RETRYABLE_ERRORS = (tradeapi.rest.APIError, ConnectionError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, TimeoutError)


_TERMINAL_ORDER_STATES = {"filled", "canceled", "cancelled", "expired", "rejected"}


def _timeframe_minutes(timeframe):
    tf = str(timeframe).lower()
    for suffix, mult in (("min", 1), ("hour", 60), ("day", 390), ("week", 1950), ("month", 8190)):
        if tf.endswith(suffix):
            num = tf[: -len(suffix)]
            return (int(num) if num.isdigit() else 1) * mult
    return 390


def _is_position_not_found(e):
    return isinstance(e, tradeapi.rest.APIError) and "position does not exist" in str(e)


class AlpacaClient:
    def __init__(self, api_key_id, api_secret_key, base_url, api_version="v2"):
        self.api = tradeapi.REST(api_key_id, api_secret_key, base_url, api_version=api_version)
    
    @backoff.on_exception(backoff.expo, _RETRYABLE_ERRORS, max_tries=5, jitter=backoff.full_jitter)
    def get_account(self):
        return self.api.get_account()
    
    @backoff.on_exception(backoff.expo, _RETRYABLE_ERRORS, max_tries=5, jitter=backoff.full_jitter)
    def get_clock(self):
        return self.api.get_clock()
    
    @backoff.on_exception(backoff.expo, _RETRYABLE_ERRORS, max_tries=5, jitter=backoff.full_jitter)
    def get_bars(self, symbol, timeframe, **kwargs):
        def _fetch():
            bars = self.api.get_bars(symbol, timeframe, **kwargs)
            if bars is None:
                return None
            return bars.df

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch)
            try:
                return future.result(timeout=BARS_REQUEST_TIMEOUT)
            except concurrent.futures.TimeoutError:
                logging.warning(f"get_bars timed out after {BARS_REQUEST_TIMEOUT}s for {symbol} {timeframe}")
                raise TimeoutError(f"get_bars hung for {symbol} {timeframe}")
    
    def get_latest_bars(self, symbol, timeframe, count):
        minutes = _timeframe_minutes(timeframe)
        trading_days = math.ceil(count * minutes / 390)
        if minutes < 390:
            trading_days = trading_days * 2 + 3
        calendar_days = int(trading_days * 1.5) + 7
        start = (datetime.now(timezone.utc) - timedelta(days=calendar_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        df = self.get_bars(symbol, timeframe, start=start)
        if df is None or len(df) == 0:
            return df
        return df.sort_index().tail(count)

    @backoff.on_exception(backoff.expo, _RETRYABLE_ERRORS, max_tries=5, jitter=backoff.full_jitter)
    def get_latest_quote(self, symbol):
        return self.api.get_latest_quote(symbol)
    
    @backoff.on_exception(backoff.expo, _RETRYABLE_ERRORS, max_tries=5, jitter=backoff.full_jitter)
    def submit_order(self, **kwargs):
        return self.api.submit_order(**kwargs)
    
    @backoff.on_exception(backoff.expo, _RETRYABLE_ERRORS, max_tries=5, jitter=backoff.full_jitter)
    def get_order(self, order_id):
        return self.api.get_order(order_id)
    
    @backoff.on_exception(backoff.expo, _RETRYABLE_ERRORS, max_tries=5, jitter=backoff.full_jitter)
    def cancel_order(self, order_id):
        return self.api.cancel_order(order_id)
    
    @backoff.on_exception(backoff.expo, _RETRYABLE_ERRORS, max_tries=5, jitter=backoff.full_jitter)
    def list_positions(self):
        return self.api.list_positions()
    
    @backoff.on_exception(backoff.expo, _RETRYABLE_ERRORS, max_tries=5, jitter=backoff.full_jitter)
    def list_orders(self, **kwargs):
        return self.api.list_orders(**kwargs)
    
    @backoff.on_exception(backoff.expo, _RETRYABLE_ERRORS, max_tries=5, jitter=backoff.full_jitter)
    def close_all_positions(self):
        return self.api.close_all_positions()
    
    @backoff.on_exception(backoff.expo, _RETRYABLE_ERRORS, max_tries=5, jitter=backoff.full_jitter, giveup=_is_position_not_found)
    def get_position(self, symbol):
        return self.api.get_position(symbol)

    @backoff.on_exception(backoff.expo, _RETRYABLE_ERRORS, max_tries=5, jitter=backoff.full_jitter, giveup=_is_position_not_found)
    def close_position(self, symbol):
        return self.api.close_position(symbol)
    
    def wait_for_fill(self, order_id, timeout):
        start = time.time()
        status = self.get_order(order_id)
        while status.status not in _TERMINAL_ORDER_STATES:
            if time.time() - start > timeout:
                try:
                    self.cancel_order(order_id)
                except Exception as e:
                    logging.warning(f"Cancel after timeout failed for {order_id}: {e}")
                cancel_deadline = time.time() + 15
                status = self.get_order(order_id)
                while status.status not in _TERMINAL_ORDER_STATES and time.time() < cancel_deadline:
                    time.sleep(1)
                    status = self.get_order(order_id)
                if status.status not in _TERMINAL_ORDER_STATES:
                    logging.error(f"Order {order_id} still {status.status} after cancel request")
                break
            time.sleep(0.5)
            status = self.get_order(order_id)
        filled_qty = float(getattr(status, "filled_qty", 0) or 0)
        avg_price = getattr(status, "filled_avg_price", None)
        if filled_qty > 0 and avg_price:
            if status.status != "filled":
                logging.warning(f"Order {order_id} partially filled: {filled_qty} shares, status={status.status}")
            return float(avg_price)
        return None

    def place_order(self, symbol, side, notional, limit_price, limit_order_timeout):
        try:
            quote = self.get_latest_quote(symbol)
            if quote is None:
                return None
            bid_price = getattr(quote, 'bid_price', None)
            ask_price = getattr(quote, 'ask_price', None)
            
            if bid_price is None or ask_price is None:
                return None
            if bid_price <= 0 or ask_price <= 0:
                return None
            
            if limit_price:
                price_source = limit_price
            else:
                price_source = bid_price if side == "buy" else ask_price
            if price_source is None or price_source <= 0:
                return None
            shares = int(notional / price_source)
            if shares == 0:
                return None
            if limit_price:
                order = self.submit_order(symbol=symbol, qty=shares, side=side, type="limit", limit_price=round(limit_price, 2), time_in_force="day")
                return self.wait_for_fill(order.id, limit_order_timeout)
            order = self.submit_order(symbol=symbol, qty=shares, side=side, type="market", time_in_force="day")
            return self.wait_for_fill(order.id, 30)
        except Exception as e:
            logging.error(f"Order placement error: {e}")
            return None
