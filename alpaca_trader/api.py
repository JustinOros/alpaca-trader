import time
import logging
import backoff
import alpaca_trade_api as tradeapi
import requests.exceptions

logging.getLogger('backoff').setLevel(logging.CRITICAL)

_RETRYABLE_ERRORS = (tradeapi.rest.APIError, ConnectionError, requests.exceptions.ConnectionError)


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
        bars = self.api.get_bars(symbol, timeframe, **kwargs)
        if bars is None:
            return None
        return bars.df
    
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
                start = time.time()
                while time.time() - start < limit_order_timeout:
                    status = self.get_order(order.id)
                    if status.status == "filled":
                        return float(status.filled_avg_price)
                    if status.status in {"cancelled", "expired", "rejected"}:
                        return None
                    time.sleep(2)
                self.cancel_order(order.id)
                return None
            order = self.submit_order(symbol=symbol, qty=shares, side=side, type="market", time_in_force="day")
            status = self.get_order(order.id)
            timeout = 30
            start_time = time.time()
            while status.status not in {"filled", "cancelled", "expired", "rejected"}:
                if time.time() - start_time > timeout:
                    return None
                time.sleep(0.5)
                status = self.get_order(order.id)
            if status.status == "filled":
                return float(status.filled_avg_price)
            return None
        except Exception as e:
            logging.error(f"Order placement error: {e}")
            return None
