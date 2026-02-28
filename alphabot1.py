"""Simpler alternative alpha bot for the Algothon / IMCity exchange.

This file is intentionally narrower than alphabot.py:
- simpler fair-value construction
- simpler opportunity ranking
- one active quoted product at a time
- hardcoded credential block preserved in __main__
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
LONDON_TZ = ZoneInfo("Europe/London")


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


class AlphaPulseBot1(BaseBot):
    WATCHLIST = ("TIDE_SPOT", "TIDE_SWING", "WX_SPOT", "WX_SUM", "LON_ETF", "LON_FLY")
    REFRESH_SECS = 300.0
    EVAL_SECS = 3.0
    MIN_REST_GAP = 1.05
    TAKE_EDGE = 14.0
    QUOTE_EDGE = 7.0
    SMOOTH_ALPHA = 0.25

    def __init__(
        self,
        cmi_url: str,
        username: str,
        password: str,
        *,
        aerodatabox_key: str | None = None,
        base_order_size: int = 2,
        max_position: int = 12,
    ):
        super().__init__(cmi_url, username, password)
        self.aerodatabox_key = aerodatabox_key
        self.base_order_size = base_order_size
        self.max_position = max_position

        self.products: dict[str, Product] = {}
        self.books: dict[str, OrderBook] = {}
        self.positions: dict[str, int] = {}
        self.external_cache: dict[str, Any] = {}
        self.theos: dict[str, float] = {}
        self.smoothed_theos: dict[str, float] = {}
        self.active_quote: QuoteState | None = None

        self.last_refresh_at = 0.0
        self.last_eval_at = 0.0
        self.last_rest_at = 0.0

        self._lock = threading.Lock()
        self._eval_lock = threading.Lock()

    def on_orderbook(self, orderbook: OrderBook) -> None:
        with self._lock:
            self.books[orderbook.product] = orderbook
        self._maybe_evaluate()

    def on_trades(self, trade: Trade) -> None:
        signed = trade.volume if trade.buyer == self.username else -trade.volume
        with self._lock:
            self.positions[trade.product] = self.positions.get(trade.product, 0) + signed
        side = "BOUGHT" if signed > 0 else "SOLD"
        print(f"FILL {side:>6} {trade.volume} {trade.product} @ {trade.price}")

    def run(self) -> None:
        loaded_products = self._paced(lambda: self.get_products())
        self.products = {p.symbol: p for p in loaded_products}
        self.positions = self._paced(lambda: self.get_positions())
        self._refresh_external(force=True)
        self.start()
        print(f"Started alphabot1 for {self.username}. Watching: {', '.join(self.WATCHLIST)}")
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
        if not force and now - self.last_eval_at < self.EVAL_SECS:
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
        if time.monotonic() - self.last_refresh_at > self.REFRESH_SECS:
            self._refresh_external()

        with self._lock:
            books = dict(self.books)
            positions = dict(self.positions)

        if not books:
            return

        raw_theos = self._build_theos(books)
        self.theos = self._smooth_theos(raw_theos)

        best_signal = None
        for symbol in self.WATCHLIST:
            book = books.get(symbol)
            fair = self.theos.get(symbol)
            if not book or fair is None:
                continue
            signal = self._signal_for(symbol, book, fair, positions.get(symbol, 0))
            if signal and (best_signal is None or signal["score"] > best_signal["score"]):
                best_signal = signal

        if not best_signal:
            self._cancel_active_quote()
            return

        symbol = best_signal["product"]
        book = books[symbol]
        fair = self.theos[symbol]
        position = positions.get(symbol, 0)

        if best_signal["edge"] >= self.TAKE_EDGE:
            self._cancel_active_quote(except_product=symbol)
            self._take_liquidity(best_signal, book, fair, position)
            return

        if best_signal["edge"] >= self.QUOTE_EDGE:
            self._quote(best_signal, book, fair, position)
            return

        self._cancel_active_quote()

    def _build_theos(self, books: dict[str, OrderBook]) -> dict[str, float]:
        tide = self.external_cache.get("thames", {})
        weather = self.external_cache.get("weather", {})
        flights = self.external_cache.get("flights", {})

        tide_spot = self._tide_spot_theo(books, tide)
        tide_swing = self._tide_swing_theo(books, tide)
        wx_spot = self._weather_spot_theo(books, weather)
        wx_sum = self._weather_sum_theo(books, weather)

        if flights.get("count") is not None:
            lhr_count = max(0.0, float(flights["count"]))
        else:
            etf_mid = self._mid(books.get("LON_ETF"))
            if etf_mid is not None:
                lhr_count = max(0.0, etf_mid - tide_spot - wx_spot)
            else:
                lhr_product = self.products.get("LHR_COUNT")
                lhr_count = float(lhr_product.startingPrice) if lhr_product else 1500.0

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

        for symbol in ("TIDE_SWING", "WX_SUM", "LON_FLY"):
            mid = self._mid(books.get(symbol))
            if mid is not None:
                theos[symbol] = 0.85 * theos[symbol] + 0.15 * mid

        return theos

    def _signal_for(self, symbol: str, book: OrderBook, fair: float, position: int) -> dict[str, float] | None:
        best_bid = self._best_bid(book)
        best_ask = self._best_ask(book)
        if best_bid is None and best_ask is None:
            return None

        buy_edge = fair - best_ask if best_ask is not None and position < self.max_position else float("-inf")
        sell_edge = best_bid - fair if best_bid is not None and position > -self.max_position else float("-inf")
        edge = max(buy_edge, sell_edge, 0.0)
        if edge <= 0.0:
            return None

        inventory_penalty = max(0.0, abs(position) / max(self.max_position, 1) - 0.4) * self.QUOTE_EDGE
        score = edge - inventory_penalty
        return {
            "product": symbol,
            "edge": edge,
            "score": score,
            "buy_edge": max(0.0, buy_edge),
            "sell_edge": max(0.0, sell_edge),
        }

    def _take_liquidity(self, signal: dict[str, float], book: OrderBook, fair: float, position: int) -> None:
        symbol = signal["product"]
        best_ask_order = self._best_ask_order(book)
        best_bid_order = self._best_bid_order(book)

        if best_ask_order and signal["buy_edge"] >= self.TAKE_EDGE and position < self.max_position:
            available = best_ask_order.volume - best_ask_order.own_volume
            size = min(self._size_for(position, signal["edge"]), available)
            if size > 0:
                self._send_ioc(OrderRequest(symbol, best_ask_order.price, Side.BUY, size))
                print(f"HIT BUY  {size} {symbol} @ {best_ask_order.price:.0f}  theo={fair:.1f}")
                return

        if best_bid_order and signal["sell_edge"] >= self.TAKE_EDGE and position > -self.max_position:
            available = best_bid_order.volume - best_bid_order.own_volume
            size = min(self._size_for(position, signal["edge"]), available)
            if size > 0:
                self._send_ioc(OrderRequest(symbol, best_bid_order.price, Side.SELL, size))
                print(f"HIT SELL {size} {symbol} @ {best_bid_order.price:.0f}  theo={fair:.1f}")

    def _quote(self, signal: dict[str, float], book: OrderBook, fair: float, position: int) -> None:
        symbol = signal["product"]
        product = self.products.get(symbol)
        if not product:
            return

        tick = product.tickSize or 1.0
        best_bid = self._best_bid(book)
        best_ask = self._best_ask(book)
        size = self._size_for(position, signal["edge"])
        if size <= 0:
            self._cancel_active_quote()
            return

        spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else 8.0
        half_width = max(5.0, spread / 2.0)
        skew = clamp(position / max(self.max_position, 1), -1.0, 1.0) * 4.0

        bid = math.floor((fair - half_width - skew) / tick) * tick
        ask = math.ceil((fair + half_width - skew) / tick) * tick
        if best_bid is not None:
            bid = min(bid, best_bid + tick)
        if best_ask is not None:
            ask = max(ask, best_ask - tick)
        if bid <= 0 or ask <= bid:
            return

        target_bid = bid if signal["buy_edge"] >= self.QUOTE_EDGE and position < self.max_position else None
        target_ask = ask if signal["sell_edge"] >= self.QUOTE_EDGE and position > -self.max_position else None

        if target_bid is None and target_ask is None:
            self._cancel_active_quote()
            return

        if self.active_quote and self.active_quote.product == symbol:
            if self.active_quote.bid_price == target_bid and self.active_quote.ask_price == target_ask:
                return

        self._cancel_active_quote(except_product=symbol)

        bid_resp = None
        ask_resp = None
        if target_bid is not None:
            bid_resp = self._paced(lambda: self.send_order(OrderRequest(symbol, target_bid, Side.BUY, size)))
        if target_ask is not None:
            ask_resp = self._paced(lambda: self.send_order(OrderRequest(symbol, target_ask, Side.SELL, size)))

        self.active_quote = QuoteState(
            product=symbol,
            bid_id=bid_resp.id if bid_resp else None,
            ask_id=ask_resp.id if ask_resp else None,
            bid_price=target_bid if bid_resp else None,
            ask_price=target_ask if ask_resp else None,
        )
        print(f"QUOTE {symbol:>9}  {target_bid or '-'} / {target_ask or '-'}  theo={fair:.1f}")

    def _cancel_active_quote(self, except_product: str | None = None) -> None:
        quote = self.active_quote
        if not quote:
            return
        if except_product and quote.product == except_product:
            return
        if quote.bid_id:
            self._paced(lambda: self.cancel_order(quote.bid_id))
        if quote.ask_id:
            self._paced(lambda: self.cancel_order(quote.ask_id))
        self.active_quote = None

    def _send_ioc(self, order: OrderRequest) -> OrderResponse | None:
        response = self._paced(lambda: self.send_order(order))
        if response and response.filled < response.volume:
            self.cancel_order(response.id)
            self.last_rest_at = time.monotonic()
        return response

    def _smooth_theos(self, fresh: dict[str, float]) -> dict[str, float]:
        if not self.smoothed_theos:
            self.smoothed_theos = dict(fresh)
            return dict(fresh)

        smoothed: dict[str, float] = {}
        for symbol, value in fresh.items():
            previous = self.smoothed_theos.get(symbol, value)
            smoothed[symbol] = previous + self.SMOOTH_ALPHA * (value - previous)
        self.smoothed_theos = smoothed
        return dict(smoothed)

    def _tide_spot_theo(self, books: dict[str, OrderBook], tide: dict[str, Any]) -> float:
        if tide.get("projected_level_m") is not None:
            return max(0.0, abs(float(tide["projected_level_m"])) * 1000.0)
        if tide.get("settle_level_m") is not None:
            return max(0.0, abs(float(tide["settle_level_m"])) * 1000.0)
        if tide.get("latest_level_m") is not None:
            return max(0.0, abs(float(tide["latest_level_m"])) * 1000.0)
        return self._fallback("TIDE_SPOT", books)

    def _tide_swing_theo(self, books: dict[str, OrderBook], tide: dict[str, Any]) -> float:
        if tide.get("projected_swing_sum") is not None:
            return max(0.0, float(tide["projected_swing_sum"]))
        if tide.get("swing_sum") is not None:
            return max(0.0, float(tide["swing_sum"]))
        return self._fallback("TIDE_SWING", books)

    def _weather_spot_theo(self, books: dict[str, OrderBook], weather: dict[str, Any]) -> float:
        if weather.get("wx_spot_settle") is not None:
            return max(0.0, float(weather["wx_spot_settle"]))
        if weather.get("wx_spot") is not None:
            return max(0.0, float(weather["wx_spot"]))
        return self._fallback("WX_SPOT", books)

    def _weather_sum_theo(self, books: dict[str, OrderBook], weather: dict[str, Any]) -> float:
        if weather.get("wx_sum") is not None:
            return max(0.0, float(weather["wx_sum"]))
        return self._fallback("WX_SUM", books)

    def _fallback(self, symbol: str, books: dict[str, OrderBook]) -> float:
        mid = self._mid(books.get(symbol))
        if mid is not None:
            return mid
        product = self.products.get(symbol)
        return float(product.startingPrice) if product else 0.0

    def _mid(self, book: OrderBook | None) -> float | None:
        if not book:
            return None
        best_bid = self._best_bid(book)
        best_ask = self._best_ask(book)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return best_bid if best_bid is not None else best_ask

    def _best_bid(self, book: OrderBook) -> float | None:
        order = self._best_bid_order(book)
        return order.price if order else None

    def _best_ask(self, book: OrderBook) -> float | None:
        order = self._best_ask_order(book)
        return order.price if order else None

    def _best_bid_order(self, book: OrderBook):
        for order in book.buy_orders:
            if order.volume > order.own_volume:
                return order
        return None

    def _best_ask_order(self, book: OrderBook):
        for order in book.sell_orders:
            if order.volume > order.own_volume:
                return order
        return None

    def _size_for(self, position: int, edge: float) -> int:
        utilization = abs(position) / max(self.max_position, 1)
        scale = 1.0 - clamp(utilization, 0.0, 0.85)
        edge_boost = 1.0 + 0.25 * clamp(edge / max(self.TAKE_EDGE, 1.0), 0.0, 1.0)
        return max(0, int(round(self.base_order_size * scale * edge_boost)))

    def _refresh_external(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_refresh_at < self.REFRESH_SECS:
            return
        self.external_cache["weather"] = self._fetch_weather()
        self.external_cache["thames"] = self._fetch_thames()
        if self.aerodatabox_key:
            self.external_cache["flights"] = self._fetch_flights()
        self.last_refresh_at = now

    def _fetch_weather(self) -> dict[str, float]:
        try:
            response = requests.get(
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
            response.raise_for_status()
            raw = response.json()["minutely_15"]
            times = []
            for value in raw["time"]:
                stamp = datetime.fromisoformat(value)
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=LONDON_TZ)
                else:
                    stamp = stamp.astimezone(LONDON_TZ)
                times.append(stamp)

            temps_c = raw["temperature_2m"]
            humids = raw["relative_humidity_2m"]
            current_idx = len(temps_c) // 2
            settle = self._next_settlement_time()
            settle_idx = min(range(len(times)), key=lambda i: abs((times[i] - settle).total_seconds()))

            current_temp_f = temps_c[current_idx] * 9.0 / 5.0 + 32.0
            settle_temp_f = temps_c[settle_idx] * 9.0 / 5.0 + 32.0
            current_spot = current_temp_f * humids[current_idx]
            settle_spot = settle_temp_f * humids[settle_idx]

            wx_sum = 0.0
            window_start = settle - timedelta(hours=24)
            for stamp, temp_c, humidity in zip(times, temps_c, humids):
                if window_start <= stamp <= settle:
                    wx_sum += (temp_c * 9.0 / 5.0 + 32.0) * humidity
            wx_sum /= 100.0

            return {
                "wx_spot": current_spot,
                "wx_spot_settle": settle_spot,
                "wx_sum": wx_sum,
            }
        except Exception as exc:
            print(f"Warning: weather fetch failed: {exc}")
            return self.external_cache.get("weather", {})

    def _fetch_thames(self) -> dict[str, float]:
        try:
            response = requests.get(
                f"https://environment.data.gov.uk/flood-monitoring/id/measures/{THAMES_MEASURE}/readings",
                params={"_sorted": "", "_limit": 193},
                timeout=10,
            )
            response.raise_for_status()
            items = response.json().get("items", [])
            if not items:
                return self.external_cache.get("thames", {})

            times = [
                datetime.fromisoformat(item["dateTime"].replace("Z", "+00:00")).astimezone(LONDON_TZ)
                for item in items
            ]
            levels = [float(item["value"]) for item in items]
            latest_level = levels[-1]
            settle = self._next_settlement_time()

            projected_level = self._project_tide_level(times, levels, settle)
            settle_proxy = settle - timedelta(hours=24)
            settle_idx = min(range(len(times)), key=lambda i: abs((times[i] - settle_proxy).total_seconds()))
            settle_level = 0.6 * levels[settle_idx] + 0.4 * latest_level

            window_start = settle - timedelta(hours=48)
            window_end = settle - timedelta(hours=24)
            swing_sum = 0.0
            for prev_t, curr_t, prev, curr in zip(times, times[1:], levels, levels[1:]):
                if prev_t < window_start or curr_t > window_end:
                    continue
                diff_cm = abs(curr - prev) * 100.0
                swing_sum += max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)

            projected_swing = self._project_tide_swing(times, levels, settle - timedelta(hours=24), settle)

            result = {
                "latest_level_m": latest_level,
                "settle_level_m": settle_level,
                "swing_sum": swing_sum,
            }
            if projected_level is not None:
                result["projected_level_m"] = 0.75 * projected_level + 0.25 * settle_level
            if projected_swing is not None:
                result["projected_swing_sum"] = projected_swing
            return result
        except Exception as exc:
            print(f"Warning: Thames fetch failed: {exc}")
            return self.external_cache.get("thames", {})

    def _fetch_flights(self) -> dict[str, float]:
        try:
            now = datetime.now(LONDON_TZ).replace(second=0, microsecond=0)
            start = (now - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M")
            end = now.strftime("%Y-%m-%dT%H:%M")
            response = requests.get(
                f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/LHR/{start}/{end}",
                params={"direction": "Both"},
                headers={
                    "x-rapidapi-host": "aerodatabox.p.rapidapi.com",
                    "x-rapidapi-key": self.aerodatabox_key,
                },
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            arrivals = payload.get("arrivals", [])
            departures = payload.get("departures", [])
            return {"count": float(len(arrivals) + len(departures))}
        except Exception as exc:
            print(f"Warning: flight fetch failed: {exc}")
            return self.external_cache.get("flights", {})

    def _project_tide_level(
        self,
        times: list[datetime],
        levels: list[float],
        target: datetime,
    ) -> float | None:
        sample_count = min(len(times), len(levels), 96)
        if sample_count < 12:
            return None

        fit_times = times[-sample_count:]
        fit_levels = levels[-sample_count:]
        anchor = fit_times[-1]
        cycle = 12 * 3600 + 25 * 60
        omega = 2.0 * math.pi / cycle
        phases = [omega * (stamp - anchor).total_seconds() for stamp in fit_times]

        sin_sum = sum(math.sin(p) for p in phases)
        cos_sum = sum(math.cos(p) for p in phases)
        y_sum = sum(fit_levels)
        n = float(sample_count)

        sin2_sum = sum(math.sin(p) ** 2 for p in phases)
        cos2_sum = sum(math.cos(p) ** 2 for p in phases)
        sin_cos_sum = sum(math.sin(p) * math.cos(p) for p in phases)
        y_sin_sum = sum(y * math.sin(p) for y, p in zip(fit_levels, phases))
        y_cos_sum = sum(y * math.cos(p) for y, p in zip(fit_levels, phases))

        matrix = [
            [n, sin_sum, cos_sum],
            [sin_sum, sin2_sum, sin_cos_sum],
            [cos_sum, sin_cos_sum, cos2_sum],
        ]
        vector = [y_sum, y_sin_sum, y_cos_sum]
        coeffs = self._solve_3x3(matrix, vector)
        if coeffs is None:
            return None

        phase = omega * (target - anchor).total_seconds()
        offset, sin_coeff, cos_coeff = coeffs
        return offset + sin_coeff * math.sin(phase) + cos_coeff * math.cos(phase)

    def _project_tide_swing(
        self,
        times: list[datetime],
        levels: list[float],
        start: datetime,
        end: datetime,
    ) -> float | None:
        if end <= start:
            return None

        points = [start]
        cursor = start
        while True:
            cursor += timedelta(minutes=15)
            if cursor >= end:
                break
            points.append(cursor)
        points.append(end)

        predicted = []
        for stamp in points:
            level = self._project_tide_level(times, levels, stamp)
            if level is None:
                return None
            predicted.append(level)

        swing_sum = 0.0
        for prev, curr in zip(predicted, predicted[1:]):
            diff_cm = abs(curr - prev) * 100.0
            swing_sum += max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)
        return swing_sum

    def _solve_3x3(self, matrix: list[list[float]], vector: list[float]) -> tuple[float, float, float] | None:
        a = [row[:] + [value] for row, value in zip(matrix, vector)]
        for pivot in range(3):
            best = max(range(pivot, 3), key=lambda row: abs(a[row][pivot]))
            if abs(a[best][pivot]) < 1e-9:
                return None
            if best != pivot:
                a[pivot], a[best] = a[best], a[pivot]

            pivot_value = a[pivot][pivot]
            for col in range(pivot, 4):
                a[pivot][col] /= pivot_value

            for row in range(3):
                if row == pivot:
                    continue
                factor = a[row][pivot]
                for col in range(pivot, 4):
                    a[row][col] -= factor * a[pivot][col]

        return (a[0][3], a[1][3], a[2][3])

    def _next_settlement_time(self) -> datetime:
        now = datetime.now(LONDON_TZ)
        settle = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if now >= settle:
            settle += timedelta(days=1)
        return settle

    def _paced(self, func) -> Any:
        wait = self.MIN_REST_GAP - (time.monotonic() - self.last_rest_at)
        if wait > 0:
            time.sleep(wait)
        result = func()
        self.last_rest_at = time.monotonic()
        return result


if __name__ == "__main__":
    EXCHANGE_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
    USERNAME = "out of our depth"
    PASSWORD = "123456789"
    AERODATABOX_KEY = None  # Optional. Improves the LHR_COUNT estimate.

    bot = AlphaPulseBot1(
        EXCHANGE_URL,
        USERNAME,
        PASSWORD,
        aerodatabox_key=AERODATABOX_KEY,
        base_order_size=2,
        max_position=12,
    )
    bot.run()
