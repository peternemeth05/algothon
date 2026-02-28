"""Alpha-first CMI bot for the Algothon / IMCity exchange.

Core idea:
- Build direct fair values for TIDE_SPOT, TIDE_SWING, WX_SPOT, WX_SUM from free APIs.
- Infer the missing flight leg from market prices when no AeroDataBox key is available.
- Price LON_ETF and LON_FLY off those component theos.
- Trade only the best edge at any time to respect the 1 request / second exchange limit.

This script is intentionally conservative on exchange traffic:
- SSE market stream is used for market data.
- REST actions are rate-limited in-process.
- At most one product is actively quoted at once.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

try:
    from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Product, Side, Trade
except ModuleNotFoundError:
    import sys

    fallback = Path("/Users/hayden/Downloads/algothon")
    if fallback.exists():
        sys.path.append(str(fallback))
    from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Product, Side, Trade


LONDON_LAT = 51.5074
LONDON_LON = -0.1278
THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def call_payoff(spot: float, strike: float) -> float:
    return max(0.0, spot - strike)


def put_payoff(spot: float, strike: float) -> float:
    return max(0.0, strike - spot)


def fly_payoff(etf: float) -> float:
    return (
        2.0 * put_payoff(etf, 6200.0)
        + call_payoff(etf, 6200.0)
        - 2.0 * call_payoff(etf, 6600.0)
        + 3.0 * call_payoff(etf, 7000.0)
    )


@dataclass
class QuoteState:
    product: str
    bid_id: str | None = None
    ask_id: str | None = None
    bid_price: float | None = None
    ask_price: float | None = None


class AlphaPulseBot(BaseBot):
    """Event-driven alpha bot with simple cross-product inference."""

    WATCHLIST = ("TIDE_SPOT", "TIDE_SWING", "WX_SPOT", "WX_SUM", "LON_ETF", "LON_FLY")
    EXTERNAL_REFRESH_SECS = 300.0
    MIN_ACTION_GAP_SECS = 1.05
    EVAL_INTERVAL_SECS = 4.0
    PRODUCT_SCORE_BIAS = {
        "TIDE_SPOT": 1.0,
        "TIDE_SWING": 0.9,
        "WX_SPOT": 1.0,
        "WX_SUM": 0.9,
        "LON_ETF": 1.2,
        "LON_FLY": 1.35,
    }

    def __init__(
        self,
        cmi_url: str,
        username: str,
        password: str,
        *,
        aerodatabox_key: str | None = None,
        base_order_size: int = 4,
        max_position: int = 20,
        aggress_edge: float = 12.0,
        quote_edge: float = 6.0,
    ):
        super().__init__(cmi_url, username, password)
        self.aerodatabox_key = aerodatabox_key
        self.base_order_size = base_order_size
        self.max_position = max_position
        self.aggress_edge = aggress_edge
        self.quote_edge = quote_edge

        self.products: dict[str, Product] = {}
        self.books: dict[str, OrderBook] = {}
        self.positions: dict[str, int] = {}
        self.theos: dict[str, float] = {}
        self.external_cache: dict[str, Any] = {}
        self.external_updated_at = 0.0
        self.last_eval_at = 0.0
        self.last_rest_at = 0.0
        self.active_quote: QuoteState | None = None
        self._lock = threading.Lock()
        self._eval_lock = threading.Lock()

    def on_orderbook(self, orderbook: OrderBook) -> None:
        with self._lock:
            self.books[orderbook.product] = orderbook
        self._maybe_evaluate()

    def on_trades(self, trade: Trade) -> None:
        # Only our fills are streamed here.
        signed = trade.volume if trade.buyer == self.username else -trade.volume
        with self._lock:
            self.positions[trade.product] = self.positions.get(trade.product, 0) + signed
        side = "BOUGHT" if signed > 0 else "SOLD"
        print(f"FILL {side:>6} {trade.volume} {trade.product} @ {trade.price}")

    def run(self) -> None:
        loaded_products = self._paced(lambda: self.get_products(), "load products")
        self.products = {p.symbol: p for p in loaded_products}
        self.positions = self._paced(lambda: self.get_positions(), "load positions")
        self._refresh_external_data(force=True)
        self.start()
        print(f"Started alpha bot for {self.username}. Watching: {', '.join(self.WATCHLIST)}")
        try:
            while True:
                self._maybe_evaluate(force=True)
                time.sleep(1.0)
        except KeyboardInterrupt:
            self._cancel_active_quote()
            self.stop()
            print("Stopped.")

    def _maybe_evaluate(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_eval_at < self.EVAL_INTERVAL_SECS:
            return
        if not self._eval_lock.acquire(blocking=False):
            return
        self.last_eval_at = now
        try:
            self._evaluate()
        except Exception as exc:
            print(f"Evaluation error: {exc}")
        finally:
            self._eval_lock.release()

    def _evaluate(self) -> None:
        if time.monotonic() - self.external_updated_at > self.EXTERNAL_REFRESH_SECS:
            self._refresh_external_data()

        with self._lock:
            books = dict(self.books)
            positions = dict(self.positions)

        if not books:
            return

        self.theos = self._build_theos(books)
        candidates = self._rank_opportunities(books, positions)
        if not candidates:
            return

        best = candidates[0]
        symbol = best["product"]
        edge = best["edge"]
        book = books[symbol]
        fair = self.theos[symbol]

        if edge >= self.aggress_edge:
            self._cancel_active_quote(except_product=symbol)
            self._take_liquidity(best, book, fair, positions.get(symbol, 0))
            return

        if edge >= self.quote_edge:
            self._quote_product(best, book, fair, positions.get(symbol, 0))
            return

        self._cancel_active_quote()

    def _rank_opportunities(
        self,
        books: dict[str, OrderBook],
        positions: dict[str, int],
    ) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for symbol in self.WATCHLIST:
            if symbol not in books or symbol not in self.theos:
                continue
            signal = self._signal_for(symbol, books[symbol], self.theos[symbol], positions.get(symbol, 0))
            if signal:
                ranked.append(signal)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def _signal_for(
        self,
        symbol: str,
        book: OrderBook,
        fair: float,
        position: int,
    ) -> dict[str, Any] | None:
        best_bid = self._best_market_bid(book)
        best_ask = self._best_market_ask(book)
        if best_bid is None and best_ask is None:
            return None

        if best_ask is not None and position < self.max_position:
            buy_edge = fair - best_ask
        else:
            buy_edge = float("-inf")

        if best_bid is not None and position > -self.max_position:
            sell_edge = best_bid - fair
        else:
            sell_edge = float("-inf")

        edge = max(buy_edge, sell_edge, 0.0)
        if edge <= 0:
            return None

        structural_bonus = self._structural_bonus(symbol, fair)
        score = (edge + structural_bonus) * self.PRODUCT_SCORE_BIAS.get(symbol, 1.0)

        return {
            "product": symbol,
            "edge": edge,
            "score": score,
            "buy_edge": max(0.0, buy_edge),
            "sell_edge": max(0.0, sell_edge),
            "direction": "BUY" if buy_edge >= sell_edge else "SELL",
        }

    def _take_liquidity(self, signal: dict[str, Any], book: OrderBook, fair: float, position: int) -> None:
        symbol = signal["product"]
        best_bid = self._best_market_bid(book)
        best_ask = self._best_market_ask(book)
        size = self._size_for_position(position, signal["edge"])
        if size <= 0:
            return

        if best_ask is not None and signal["buy_edge"] >= self.aggress_edge and position < self.max_position:
            self._send_ioc(OrderRequest(symbol, best_ask, Side.BUY, size))
            print(f"HIT BUY  {size} {symbol} @ {best_ask:.0f}  theo={fair:.1f}")
            return

        if best_bid is not None and signal["sell_edge"] >= self.aggress_edge and position > -self.max_position:
            self._send_ioc(OrderRequest(symbol, best_bid, Side.SELL, size))
            print(f"HIT SELL {size} {symbol} @ {best_bid:.0f}  theo={fair:.1f}")

    def _quote_product(self, signal: dict[str, Any], book: OrderBook, fair: float, position: int) -> None:
        symbol = signal["product"]
        product = self.products.get(symbol)
        if not product:
            return

        best_bid = self._best_market_bid(book)
        best_ask = self._best_market_ask(book)
        tick = product.tickSize or 1.0
        size = self._size_for_position(position, signal["edge"])
        if size <= 0:
            self._cancel_active_quote()
            return

        inventory_skew = clamp(position / max(self.max_position, 1), -1.0, 1.0) * 5.0
        half_width = self._dynamic_width(symbol, book, fair)
        bid = math.floor((fair - half_width - inventory_skew) / tick) * tick
        ask = math.ceil((fair + half_width - inventory_skew) / tick) * tick

        if best_bid is not None:
            bid = min(bid, best_bid + tick)
        if best_ask is not None:
            ask = max(ask, best_ask - tick)
        if bid <= 0 or ask <= bid:
            return

        directional_imbalance = abs(signal["buy_edge"] - signal["sell_edge"])
        one_sided = directional_imbalance >= self.quote_edge

        target_bid = bid if signal["buy_edge"] >= self.quote_edge and position < self.max_position else None
        target_ask = ask if signal["sell_edge"] >= self.quote_edge and position > -self.max_position else None
        if not one_sided:
            if target_bid is None and position < self.max_position:
                target_bid = bid
            if target_ask is None and position > -self.max_position:
                target_ask = ask

        if target_bid is None and target_ask is None:
            self._cancel_active_quote()
            return

        if self.active_quote and self.active_quote.product == symbol:
            unchanged = self.active_quote.bid_price == target_bid and self.active_quote.ask_price == target_ask
            if unchanged:
                return

        self._cancel_active_quote(except_product=symbol)

        bid_resp = None
        ask_resp = None
        if target_bid is not None:
            bid_resp = self._paced(
                lambda: self.send_order(OrderRequest(symbol, target_bid, Side.BUY, size)),
                "quote bid",
            )
        if target_ask is not None:
            ask_resp = self._paced(
                lambda: self.send_order(OrderRequest(symbol, target_ask, Side.SELL, size)),
                "quote ask",
            )

        self.active_quote = QuoteState(
            product=symbol,
            bid_id=bid_resp.id if bid_resp else None,
            ask_id=ask_resp.id if ask_resp else None,
            bid_price=target_bid if bid_resp else None,
            ask_price=target_ask if ask_resp else None,
        )
        bid_text = f"{size}@{target_bid:.0f}" if target_bid is not None else "-"
        ask_text = f"{size}@{target_ask:.0f}" if target_ask is not None else "-"
        print(f"QUOTE {symbol:>9}  {bid_text} / {ask_text}  theo={fair:.1f}")

    def _cancel_active_quote(self, except_product: str | None = None) -> None:
        quote = self.active_quote
        if not quote:
            return
        if except_product and quote.product == except_product:
            return
        if quote.bid_id:
            self._paced(lambda: self.cancel_order(quote.bid_id), "cancel bid")
        if quote.ask_id:
            self._paced(lambda: self.cancel_order(quote.ask_id), "cancel ask")
        self.active_quote = None

    def _send_ioc(self, order: OrderRequest) -> OrderResponse | None:
        resp = self._paced(lambda: self.send_order(order), "send IOC")
        if resp and resp.volume > 0:
            self._paced(lambda: self.cancel_order(resp.id), "cancel IOC remainder")
        return resp

    def _dynamic_width(self, symbol: str, book: OrderBook, fair: float) -> float:
        best_bid = self._best_market_bid(book)
        best_ask = self._best_market_ask(book)
        spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else 8.0

        if symbol == "LON_FLY":
            return max(8.0, spread / 2.0)
        if symbol in {"TIDE_SWING", "WX_SUM"}:
            return max(6.0, spread / 2.0)
        return max(4.0, spread / 2.0)

    def _size_for_position(self, position: int, edge: float = 0.0) -> int:
        utilization = abs(position) / max(self.max_position, 1)
        scale = 1.0 - clamp(utilization, 0.0, 0.85)
        edge_boost = 1.0 + clamp(edge / max(self.aggress_edge, 1.0), 0.0, 1.0)
        return max(1, int(round(self.base_order_size * scale * edge_boost)))

    def _build_theos(self, books: dict[str, OrderBook]) -> dict[str, float]:
        tide_spot = self._theo_tide_spot(books)
        tide_swing = self._theo_tide_swing(books)
        wx_spot = self._theo_wx_spot(books)
        wx_sum = self._theo_wx_sum(books)

        lhr_count = self._infer_lhr_count(books, tide_spot, wx_spot)
        lon_etf = tide_spot + wx_spot + lhr_count
        lon_fly = fly_payoff(lon_etf)

        theos = {
            "TIDE_SPOT": tide_spot,
            "TIDE_SWING": tide_swing,
            "WX_SPOT": wx_spot,
            "WX_SUM": wx_sum,
            "LON_ETF": lon_etf,
            "LON_FLY": lon_fly,
        }

        etf_mid = self._mid(books.get("LON_ETF"))
        if etf_mid is not None:
            theos["LON_FLY"] = 0.8 * theos["LON_FLY"] + 0.2 * fly_payoff(etf_mid)

        # If the market in an unmodeled product is tighter than our model, blend lightly with the midpoint.
        for symbol in ("TIDE_SWING", "WX_SUM"):
            mid = self._mid(books.get(symbol))
            if mid is not None:
                theos[symbol] = 0.85 * theos[symbol] + 0.15 * mid
        return theos

    def _theo_tide_spot(self, books: dict[str, OrderBook]) -> float:
        cached = self.external_cache.get("thames")
        if cached and cached.get("settle_level_m") is not None:
            raw = abs(float(cached["settle_level_m"])) * 1000.0
            return max(0.0, raw)
        if cached and cached.get("latest_level_m") is not None:
            raw = abs(float(cached["latest_level_m"])) * 1000.0
            return max(0.0, raw)
        return self._fallback_mid_or_start("TIDE_SPOT", books)

    def _theo_tide_swing(self, books: dict[str, OrderBook]) -> float:
        cached = self.external_cache.get("thames")
        if cached and cached.get("swing_sum") is not None:
            return max(0.0, float(cached["swing_sum"]))
        return self._fallback_mid_or_start("TIDE_SWING", books)

    def _theo_wx_spot(self, books: dict[str, OrderBook]) -> float:
        cached = self.external_cache.get("weather")
        if cached and cached.get("wx_spot_settle") is not None:
            return max(0.0, float(cached["wx_spot_settle"]))
        if cached and cached.get("wx_spot") is not None:
            return max(0.0, float(cached["wx_spot"]))
        return self._fallback_mid_or_start("WX_SPOT", books)

    def _theo_wx_sum(self, books: dict[str, OrderBook]) -> float:
        cached = self.external_cache.get("weather")
        if cached and cached.get("wx_sum") is not None:
            return max(0.0, float(cached["wx_sum"]))
        return self._fallback_mid_or_start("WX_SUM", books)

    def _infer_lhr_count(self, books: dict[str, OrderBook], tide_spot: float, wx_spot: float) -> float:
        if self.aerodatabox_key:
            count = self.external_cache.get("flights", {}).get("count")
            if count is not None:
                return max(0.0, float(count))

        etf_mid = self._mid(books.get("LON_ETF"))
        if etf_mid is not None:
            return max(0.0, etf_mid - tide_spot - wx_spot)

        start = self.products.get("LHR_COUNT")
        if start:
            return float(start.startingPrice)
        return 1500.0

    def _fallback_mid_or_start(self, symbol: str, books: dict[str, OrderBook]) -> float:
        mid = self._mid(books.get(symbol))
        if mid is not None:
            return mid
        product = self.products.get(symbol)
        return float(product.startingPrice) if product else 0.0

    def _mid(self, book: OrderBook | None) -> float | None:
        if not book:
            return None
        best_bid = self._best_market_bid(book)
        best_ask = self._best_market_ask(book)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return best_bid if best_bid is not None else best_ask

    def _best_market_bid(self, book: OrderBook) -> float | None:
        prices = [o.price for o in book.buy_orders if o.volume > o.own_volume]
        return max(prices) if prices else None

    def _best_market_ask(self, book: OrderBook) -> float | None:
        prices = [o.price for o in book.sell_orders if o.volume > o.own_volume]
        return min(prices) if prices else None

    def _structural_bonus(self, symbol: str, fair: float) -> float:
        if symbol != "LON_FLY":
            return 0.0
        etf_mid = self._mid(self.books.get("LON_ETF"))
        if etf_mid is None:
            return 0.0
        synthetic = fly_payoff(etf_mid)
        return abs(fair - synthetic) * 0.25

    def _refresh_external_data(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.external_updated_at < self.EXTERNAL_REFRESH_SECS:
            return

        self.external_cache["weather"] = self._fetch_weather_snapshot()
        self.external_cache["thames"] = self._fetch_thames_snapshot()
        if self.aerodatabox_key:
            self.external_cache["flights"] = self._fetch_flight_snapshot()
        self.external_updated_at = now

    def _fetch_weather_snapshot(self) -> dict[str, float]:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": LONDON_LAT,
                "longitude": LONDON_LON,
                "minutely_15": "temperature_2m,relative_humidity_2m",
                "past_minutely_15": 96,
                "forecast_minutely_15": 96,
                "timezone": "Europe/London",
            },
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()["minutely_15"]
        times = []
        for value in raw["time"]:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=ZoneInfo("Europe/London"))
            else:
                parsed = parsed.astimezone(ZoneInfo("Europe/London"))
            times.append(parsed)
        temps_c = raw["temperature_2m"]
        humids = raw["relative_humidity_2m"]
        current_idx = len(temps_c) // 2
        settle = self._next_settlement_time()
        window_start = settle - timedelta(hours=24)
        settle_idx = min(range(len(times)), key=lambda idx: abs((times[idx] - settle).total_seconds()))

        current_temp_f = temps_c[current_idx] * 9.0 / 5.0 + 32.0
        current_humidity = humids[current_idx]
        current_spot = current_temp_f * current_humidity
        settle_temp_f = temps_c[settle_idx] * 9.0 / 5.0 + 32.0
        settle_humidity = humids[settle_idx]
        settle_spot = settle_temp_f * settle_humidity

        wx_sum = 0.0
        for stamp, temp_c, humidity in zip(times, temps_c, humids):
            if stamp < window_start or stamp > settle:
                continue
            temp_f = temp_c * 9.0 / 5.0 + 32.0
            wx_sum += temp_f * humidity
        wx_sum /= 100.0

        return {
            "wx_spot": current_spot,
            "wx_spot_settle": settle_spot,
            "wx_sum": wx_sum,
        }

    def _fetch_thames_snapshot(self) -> dict[str, float]:
        resp = requests.get(
            f"https://environment.data.gov.uk/flood-monitoring/id/measures/{THAMES_MEASURE}/readings",
            params={"_sorted": "", "_limit": 193},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            return {}

        times = [datetime.fromisoformat(item["dateTime"].replace("Z", "+00:00")).astimezone(ZoneInfo("Europe/London")) for item in items]
        levels = [float(item["value"]) for item in items]
        latest_level = levels[-1]
        settle = self._next_settlement_time()
        proxy_time = settle - timedelta(hours=24)
        settle_idx = min(range(len(times)), key=lambda idx: abs((times[idx] - proxy_time).total_seconds()))
        settle_level = 0.7 * levels[settle_idx] + 0.3 * latest_level
        swing_sum = 0.0
        window_start = settle - timedelta(hours=48)
        window_end = settle - timedelta(hours=24)
        for prev_t, curr_t, prev, curr in zip(times, times[1:], levels, levels[1:]):
            if prev_t < window_start or curr_t > window_end:
                continue
            diff_cm = abs(curr - prev) * 100.0
            swing_sum += max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)

        return {
            "latest_level_m": latest_level,
            "settle_level_m": settle_level,
            "swing_sum": swing_sum,
        }

    def _fetch_flight_snapshot(self) -> dict[str, float]:
        now = datetime.now().replace(second=0, microsecond=0)
        start = (now - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M")
        end = now.strftime("%Y-%m-%dT%H:%M")
        resp = requests.get(
            f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/LHR/{start}/{end}",
            params={"direction": "Both"},
            headers={
                "x-rapidapi-host": "aerodatabox.p.rapidapi.com",
                "x-rapidapi-key": self.aerodatabox_key,
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        arrivals = payload.get("arrivals", [])
        departures = payload.get("departures", [])
        return {
            "count": float(len(arrivals) + len(departures)),
        }

    def _next_settlement_time(self) -> datetime:
        now = datetime.now(ZoneInfo("Europe/London"))
        settle = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if now >= settle:
            settle += timedelta(days=1)
        return settle

    def _paced(self, func, label: str) -> Any:
        wait = self.MIN_ACTION_GAP_SECS - (time.monotonic() - self.last_rest_at)
        if wait > 0:
            time.sleep(wait)
        result = func()
        self.last_rest_at = time.monotonic()
        if label:
            pass
        return result


if __name__ == "__main__":
    EXCHANGE_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
    USERNAME = "out of our depth"
    PASSWORD = "123456789"
    AERODATABOX_KEY = None  # Optional. Improves the LHR_COUNT estimate.

    bot = AlphaPulseBot(
        EXCHANGE_URL,
        USERNAME,
        PASSWORD,
        aerodatabox_key=AERODATABOX_KEY,
        base_order_size=3,
        max_position=15,
        aggress_edge=14.0,
        quote_edge=7.0,
    )
    bot.run()
