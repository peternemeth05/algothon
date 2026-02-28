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
import os
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
TIDE_CYCLE_SECS = 12 * 3600 + 25 * 60


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE pairs into the process env without requiring python-dotenv."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


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
    THEO_SMOOTHING = 0.22
    PRODUCT_SCORE_BIAS = {
        "TIDE_SPOT": 1.0,
        "TIDE_SWING": 0.9,
        "WX_SPOT": 1.0,
        "WX_SUM": 0.9,
        "LON_ETF": 1.2,
        "LON_FLY": 0.9,
    }
    PRODUCT_MAX_POSITION = {
        "LON_FLY": 6,
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
        self.smoothed_theos: dict[str, float] = {}
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
        if trade.buyer == self.username:
            signed = trade.volume
        elif trade.seller == self.username:
            signed = -trade.volume
        else:
            return

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

        raw_theos = self._build_theos(books)
        self.theos = self._smooth_theos(raw_theos)
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
            signal = self._signal_for(symbol, books[symbol], self.theos[symbol], positions.get(symbol, 0), positions)
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
        positions: dict[str, int],
    ) -> dict[str, Any] | None:
        best_bid = self._best_market_bid(book)
        best_ask = self._best_market_ask(book)
        if best_bid is None and best_ask is None:
            return None

        position_limit = self._position_limit(symbol)

        if best_ask is not None and position < position_limit:
            buy_edge = fair - best_ask
        else:
            buy_edge = float("-inf")

        if best_bid is not None and position > -position_limit:
            sell_edge = best_bid - fair
        else:
            sell_edge = float("-inf")

        edge = max(buy_edge, sell_edge, 0.0)
        if edge <= 0:
            return None

        structural_bonus = self._structural_bonus(symbol, fair)
        exposure_penalty = self._exposure_penalty(symbol, position, positions)
        score = max(0.0, (edge + structural_bonus - exposure_penalty) * self.PRODUCT_SCORE_BIAS.get(symbol, 1.0))

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
        best_bid_order = self._best_market_bid_order(book)
        best_ask_order = self._best_market_ask_order(book)
        best_bid = best_bid_order.price if best_bid_order else None
        best_ask = best_ask_order.price if best_ask_order else None
        position_limit = self._position_limit(symbol)
        size = self._size_for_position(symbol, position, signal["edge"])
        if size <= 0:
            return

        if best_ask is not None and signal["buy_edge"] >= self.aggress_edge and position < position_limit:
            size = min(size, best_ask_order.volume - best_ask_order.own_volume)
            if size <= 0:
                return
            self._send_ioc(OrderRequest(symbol, best_ask, Side.BUY, size))
            print(f"HIT BUY  {size} {symbol} @ {best_ask:.0f}  theo={fair:.1f}")
            return

        if best_bid is not None and signal["sell_edge"] >= self.aggress_edge and position > -position_limit:
            size = min(size, best_bid_order.volume - best_bid_order.own_volume)
            if size <= 0:
                return
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
        position_limit = self._position_limit(symbol)
        size = self._size_for_position(symbol, position, signal["edge"])
        if size <= 0:
            self._cancel_active_quote()
            return

        inventory_skew = clamp(position / max(position_limit, 1), -1.0, 1.0) * 5.0
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
        inventory_heavy = abs(position) >= max(2, position_limit // 3)

        target_bid = bid if signal["buy_edge"] >= self.quote_edge and position < position_limit else None
        target_ask = ask if signal["sell_edge"] >= self.quote_edge and position > -position_limit else None
        if not one_sided and not inventory_heavy:
            if target_bid is None and position < position_limit:
                target_bid = bid
            if target_ask is None and position > -position_limit:
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
        active_orders = self._paced(lambda: self.get_orders(product=quote.product), "refresh active orders")
        active_ids = {order["id"] for order in active_orders}
        if quote.bid_id in active_ids or quote.ask_id in active_ids:
            self.active_quote = QuoteState(
                product=quote.product,
                bid_id=quote.bid_id if quote.bid_id in active_ids else None,
                ask_id=quote.ask_id if quote.ask_id in active_ids else None,
                bid_price=quote.bid_price if quote.bid_id in active_ids else None,
                ask_price=quote.ask_price if quote.ask_id in active_ids else None,
            )
            print(f"Cancel lag detected for {quote.product}; keeping local quote state in sync.")
            return
        self.active_quote = None

    def _send_ioc(self, order: OrderRequest) -> OrderResponse | None:
        resp = self._paced(lambda: self.send_order(order), "send IOC")
        if resp and resp.filled < resp.volume:
            # The exchange does not expose a true IOC flag, so cancel any remainder immediately.
            self.cancel_order(resp.id)
            self.last_rest_at = time.monotonic()
        return resp

    def _dynamic_width(self, symbol: str, book: OrderBook, fair: float) -> float:
        best_bid = self._best_market_bid(book)
        best_ask = self._best_market_ask(book)
        spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else 8.0

        if symbol == "LON_FLY":
            return max(10.0, spread / 2.0 + 1.5)
        if symbol in {"TIDE_SWING", "WX_SUM"}:
            return max(7.0, spread / 2.0 + 1.0)
        return max(5.0, spread / 2.0 + 0.5)

    def _size_for_position(self, symbol: str, position: int, edge: float = 0.0) -> int:
        position_limit = self._position_limit(symbol)
        utilization = abs(position) / max(position_limit, 1)
        scale = 1.0 - clamp(utilization, 0.0, 0.9)
        edge_boost = 1.0 + 0.35 * clamp(edge / max(self.aggress_edge, 1.0), 0.0, 1.0)
        symbol_scale = 0.6 if symbol == "LON_FLY" else 1.0
        raw_size = self.base_order_size * scale * edge_boost * symbol_scale
        return max(0, int(round(raw_size)))

    def _position_limit(self, symbol: str) -> int:
        return self.PRODUCT_MAX_POSITION.get(symbol, self.max_position)

    def _exposure_penalty(self, symbol: str, position: int, positions: dict[str, int]) -> float:
        total_abs_inventory = sum(abs(qty) for qty in positions.values())
        if total_abs_inventory <= 0:
            return 0.0

        position_limit = self._position_limit(symbol)
        utilization = abs(position) / max(position_limit, 1)
        concentration = abs(position) / total_abs_inventory
        penalty = max(0.0, utilization - 0.4) * self.quote_edge

        if symbol == "LON_FLY":
            penalty += max(0.0, concentration - 0.25) * self.aggress_edge

        return penalty

    def _smooth_theos(self, fresh_theos: dict[str, float]) -> dict[str, float]:
        if not self.smoothed_theos:
            self.smoothed_theos = dict(fresh_theos)
            return dict(fresh_theos)

        alpha = self.THEO_SMOOTHING
        smoothed: dict[str, float] = {}
        for symbol, fresh in fresh_theos.items():
            previous = self.smoothed_theos.get(symbol, fresh)
            smoothed[symbol] = previous + alpha * (fresh - previous)
        self.smoothed_theos = smoothed
        return dict(smoothed)

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
            # Shift trust towards the market midpoint of the underlying ETF
            theos["LON_FLY"] = 0.3 * theos["LON_FLY"] + 0.7 * fly_payoff(etf_mid)

        # If the market in an unmodeled product is tighter than our model, blend lightly with the midpoint.
        for symbol in ("TIDE_SWING", "WX_SUM"):
            mid = self._mid(books.get(symbol))
            if mid is not None:
                theos[symbol] = 0.85 * theos[symbol] + 0.15 * mid
        return theos

    def _theo_tide_spot(self, books: dict[str, OrderBook]) -> float:
        cached = self.external_cache.get("thames")
        if cached and cached.get("projected_settle_level_m") is not None:
            raw = abs(float(cached["projected_settle_level_m"])) * 1000.0
            return max(0.0, raw)
        if cached and cached.get("settle_level_m") is not None:
            raw = abs(float(cached["settle_level_m"])) * 1000.0
            return max(0.0, raw)
        if cached and cached.get("latest_level_m") is not None:
            raw = abs(float(cached["latest_level_m"])) * 1000.0
            return max(0.0, raw)
        return self._fallback_mid_or_start("TIDE_SPOT", books)

    def _theo_tide_swing(self, books: dict[str, OrderBook]) -> float:
        cached = self.external_cache.get("thames")
        if cached and cached.get("projected_swing_sum") is not None:
            return max(0.0, float(cached["projected_swing_sum"]))
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
        etf_mid = self._mid(books.get("LON_ETF"))
        market_implied = etf_mid - tide_spot - wx_spot if etf_mid is not None else None

        if self.aerodatabox_key:
            count = self.external_cache.get("flights", {}).get("count")
            if count is not None:
                api_val = float(count)
                if market_implied is not None:
                    # If API differs wildly from market implied, trust market more
                    if abs(api_val - market_implied) > 500:
                        return 0.4 * api_val + 0.6 * market_implied
                return api_val

        if market_implied is not None:
            return max(0.0, market_implied)

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
        order = self._best_market_bid_order(book)
        return order.price if order else None

    def _best_market_ask(self, book: OrderBook) -> float | None:
        order = self._best_market_ask_order(book)
        return order.price if order else None

    def _best_market_bid_order(self, book: OrderBook):
        for order in book.buy_orders:
            if order.volume > order.own_volume:
                return order
        return None

    def _best_market_ask_order(self, book: OrderBook):
        for order in book.sell_orders:
            if order.volume > order.own_volume:
                return order
        return None

    def _fit_tide_cycle(
        self,
        times: list[datetime],
        levels: list[float],
    ) -> dict[str, float] | None:
        sample_count = min(len(times), len(levels), 128)
        if sample_count < 16:
            return None

        fit_times = times[-sample_count:]
        fit_levels = levels[-sample_count:]
        anchor = fit_times[-1]
        omega = 2.0 * math.pi / TIDE_CYCLE_SECS

        n = float(sample_count)
        sum_sin = 0.0
        sum_cos = 0.0
        sum_sin2 = 0.0
        sum_cos2 = 0.0
        sum_sin_cos = 0.0
        sum_y = 0.0
        sum_y_sin = 0.0
        sum_y_cos = 0.0

        for stamp, level in zip(fit_times, fit_levels):
            phase = omega * (stamp - anchor).total_seconds()
            sin_v = math.sin(phase)
            cos_v = math.cos(phase)
            sum_sin += sin_v
            sum_cos += cos_v
            sum_sin2 += sin_v * sin_v
            sum_cos2 += cos_v * cos_v
            sum_sin_cos += sin_v * cos_v
            sum_y += level
            sum_y_sin += level * sin_v
            sum_y_cos += level * cos_v

        det = (
            n * (sum_sin2 * sum_cos2 - sum_sin_cos * sum_sin_cos)
            - sum_sin * (sum_sin * sum_cos2 - sum_sin_cos * sum_cos)
            + sum_cos * (sum_sin * sum_sin_cos - sum_sin2 * sum_cos)
        )
        if abs(det) < 1e-9:
            return None

        det_offset = (
            sum_y * (sum_sin2 * sum_cos2 - sum_sin_cos * sum_sin_cos)
            - sum_sin * (sum_y_sin * sum_cos2 - sum_sin_cos * sum_y_cos)
            + sum_cos * (sum_y_sin * sum_sin_cos - sum_sin2 * sum_y_cos)
        )
        det_sin = (
            n * (sum_y_sin * sum_cos2 - sum_sin_cos * sum_y_cos)
            - sum_y * (sum_sin * sum_cos2 - sum_sin_cos * sum_cos)
            + sum_cos * (sum_sin * sum_y_cos - sum_y_sin * sum_cos)
        )
        det_cos = (
            n * (sum_sin2 * sum_y_cos - sum_y_sin * sum_sin_cos)
            - sum_sin * (sum_sin * sum_y_cos - sum_y_sin * sum_cos)
            + sum_y * (sum_sin * sum_sin_cos - sum_sin2 * sum_cos)
        )

        return {
            "offset": det_offset / det,
            "sin_coeff": det_sin / det,
            "cos_coeff": det_cos / det,
            "omega": omega,
            "anchor_ts": anchor.timestamp(),
        }

    def _predict_tide_level(self, model: dict[str, float], target: datetime) -> float:
        anchor = datetime.fromtimestamp(model["anchor_ts"], tz=target.tzinfo)
        phase = model["omega"] * (target - anchor).total_seconds()
        return (
            model["offset"]
            + model["sin_coeff"] * math.sin(phase)
            + model["cos_coeff"] * math.cos(phase)
        )

    def _project_tide_swing(self, model: dict[str, float], start: datetime, end: datetime) -> float:
        if end <= start:
            return 0.0

        points = [start]
        stamp = start
        while True:
            stamp += timedelta(minutes=15)
            if stamp >= end:
                break
            points.append(stamp)
        if points[-1] != end:
            points.append(end)

        swing_sum = 0.0
        previous = self._predict_tide_level(model, points[0])
        for stamp in points[1:]:
            current = self._predict_tide_level(model, stamp)
            diff_cm = abs(current - previous) * 100.0
            swing_sum += max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)
            previous = current
        return swing_sum

    def _structural_bonus(self, symbol: str, fair: float) -> float:
        if symbol != "LON_FLY":
            return 0.0
        etf_mid = self._mid(self.books.get("LON_ETF"))
        if etf_mid is None:
            return 0.0
        synthetic = fly_payoff(etf_mid)
        return abs(fair - synthetic) * 0.08

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
        try:
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
            settle_idx = min(
                range(len(times)),
                key=lambda idx: abs((times[idx] - settle).total_seconds()),
            )

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
        except Exception as exc:
            print(f"Warning: Failed to fetch weather data: {exc}")
            return self.external_cache.get("weather", {})

    def _fetch_thames_snapshot(self) -> dict[str, float]:
        try:
            resp = requests.get(
                f"https://environment.data.gov.uk/flood-monitoring/id/measures/{THAMES_MEASURE}/readings",
                params={"_sorted": "", "_limit": 193},
                timeout=30,
            )
            resp.raise_for_status()
            items = [i for i in resp.json().get("items", []) if i.get("value") is not None]
            if not items:
                return self.external_cache.get("thames", {})

            # Ensure data is sorted by time ascending (latest last)
            items.sort(key=lambda x: x["dateTime"])

            times = [
                datetime.fromisoformat(item["dateTime"].replace("Z", "+00:00")).astimezone(ZoneInfo("Europe/London"))
                for item in items
            ]
            levels = [float(item["value"]) for item in items]
            latest_level = levels[-1]
            settle = self._next_settlement_time()
            proxy_time = settle - timedelta(hours=24)
            settle_idx = min(
                range(len(times)),
                key=lambda idx: abs((times[idx] - proxy_time).total_seconds()),
            )
            settle_level = 0.7 * levels[settle_idx] + 0.3 * latest_level
            tide_model = self._fit_tide_cycle(times, levels)
            projected_settle_level = None
            projected_swing_sum = None
            if tide_model:
                modeled_level = self._predict_tide_level(tide_model, settle)
                projected_settle_level = 0.75 * modeled_level + 0.25 * settle_level
                projected_swing_sum = self._project_tide_swing(
                    tide_model,
                    settle - timedelta(hours=24),
                    settle,
                )
            swing_sum = 0.0
            window_start = settle - timedelta(hours=48)
            window_end = settle - timedelta(hours=24)
            for prev_t, curr_t, prev, curr in zip(
                times, times[1:], levels, levels[1:]
            ):
                if prev_t < window_start or curr_t > window_end:
                    continue
                diff_cm = abs(curr - prev) * 100.0
                swing_sum += max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)

            snapshot = {
                "latest_level_m": latest_level,
                "settle_level_m": settle_level,
                "swing_sum": swing_sum,
            }
            if projected_settle_level is not None:
                snapshot["projected_settle_level_m"] = projected_settle_level
            if projected_swing_sum is not None:
                snapshot["projected_swing_sum"] = projected_swing_sum
            return snapshot
        except Exception as exc:
            print(f"Warning: Failed to fetch Thames tide data: {exc}")
            return self.external_cache.get("thames", {})

    def _fetch_flight_snapshot(self) -> dict[str, float]:
        try:
            now = datetime.now(ZoneInfo("Europe/London")).replace(second=0, microsecond=0)
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
        except Exception as exc:
            print(f"Warning: Failed to fetch flight data: {exc}")
            return self.external_cache.get("flights", {})

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
    try:
        import dotenv
    except ModuleNotFoundError:
        load_env_file()
    else:
        dotenv.load_dotenv()

    EXCHANGE_URL = os.getenv(
        "CMI_EXCHANGE_URL", "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com"
    )
    USERNAME = os.getenv("CMI_USERNAME")
    PASSWORD = os.getenv("CMI_PASSWORD")
    AERODATABOX_KEY = os.getenv("AERODATABOX_KEY")
    if not USERNAME or not PASSWORD:
        raise SystemExit(
            "Set CMI_USERNAME and CMI_PASSWORD in .env before running alphabot.py."
        )

    bot = AlphaPulseBot(
        EXCHANGE_URL,
        USERNAME,
        PASSWORD,
        aerodatabox_key=AERODATABOX_KEY,
        base_order_size=2,
        max_position=12,
        aggress_edge=16.0,
        quote_edge=8.0,
    )
    bot.run()
