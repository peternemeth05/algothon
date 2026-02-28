"""AlphaBot3 - product-aware trading bot for the IMCity London challenge.

Design principles:
- treat snapshot products and path products differently
- decompose path products into realized + remaining value
- derive LON_ETF from components and LON_FLY from ETF distribution, not just ETF spot
- use lightweight online ML only as a secondary microstructure adjustment
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

    fallback = Path(__file__).resolve().parent
    if fallback.exists():
        sys.path.append(str(fallback))
    from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Product, Side, Trade


LONDON_LAT = 51.5074
LONDON_LON = -0.1278
THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"
LONDON_TZ = ZoneInfo("Europe/London")

M2_PERIOD = 12 * 3600 + 25 * 60
S2_PERIOD = 12 * 3600
M4_PERIOD = M2_PERIOD / 2.0


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


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class QuoteState:
    product: str
    bid_id: str | None = None
    ask_id: str | None = None
    bid_price: float | None = None
    ask_price: float | None = None


@dataclass(frozen=True)
class ProductConfig:
    take_edge: float
    quote_edge: float
    max_position: int
    base_width: float
    ml_weight: float
    size_multiplier: float


class OnlineLinearModel:
    """Very small online linear regressor for next-mid change prediction."""

    def __init__(self, dimension: int, learning_rate: float = 0.025, l2: float = 0.0005):
        self.weights = [0.0] * dimension
        self.learning_rate = learning_rate
        self.l2 = l2
        self.samples = 0

    def predict(self, features: list[float]) -> float:
        return sum(weight * value for weight, value in zip(self.weights, features))

    def update(self, features: list[float], target: float) -> None:
        clipped_target = clamp(target, -40.0, 40.0)
        prediction = self.predict(features)
        error = prediction - clipped_target
        self.samples += 1
        lr = self.learning_rate / (1.0 + 0.002 * self.samples)
        for index, value in enumerate(features):
            gradient = error * value + self.l2 * self.weights[index]
            self.weights[index] -= lr * gradient


class TideModel:
    PERIODS = (M2_PERIOD, S2_PERIOD, M4_PERIOD)

    def __init__(self, offset: float, coeffs: list[tuple[float, float]], anchor_ts: float):
        self.offset = offset
        self.coeffs = coeffs
        self.anchor_ts = anchor_ts

    def predict(self, target: datetime) -> float:
        delta = target.timestamp() - self.anchor_ts
        level = self.offset
        for (sin_coeff, cos_coeff), period in zip(self.coeffs, self.PERIODS):
            omega = 2.0 * math.pi / period
            phase = omega * delta
            level += sin_coeff * math.sin(phase) + cos_coeff * math.cos(phase)
        return level

    @classmethod
    def fit(cls, times: list[datetime], levels: list[float], max_samples: int = 192) -> TideModel | None:
        sample_count = min(len(times), len(levels), max_samples)
        if sample_count < 24:
            return None

        fit_times = times[-sample_count:]
        fit_levels = levels[-sample_count:]
        anchor_ts = fit_times[-1].timestamp()
        parameter_count = 1 + 2 * len(cls.PERIODS)

        ata = [[0.0] * parameter_count for _ in range(parameter_count)]
        atb = [0.0] * parameter_count

        for stamp, level in zip(fit_times, fit_levels):
            delta = stamp.timestamp() - anchor_ts
            row = [1.0]
            for period in cls.PERIODS:
                omega = 2.0 * math.pi / period
                phase = omega * delta
                row.append(math.sin(phase))
                row.append(math.cos(phase))

            for i in range(parameter_count):
                atb[i] += row[i] * level
                for j in range(parameter_count):
                    ata[i][j] += row[i] * row[j]

        solution = solve_linear_system(ata, atb)
        if solution is None:
            return None

        offset = solution[0]
        coeffs = [(solution[1 + 2 * idx], solution[2 + 2 * idx]) for idx in range(len(cls.PERIODS))]
        return cls(offset, coeffs, anchor_ts)


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    size = len(vector)
    augmented = [matrix[row][:] + [vector[row]] for row in range(size)]

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot_row][column]) < 1e-12:
            return None
        if pivot_row != column:
            augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]

        pivot = augmented[column][column]
        for inner in range(column, size + 1):
            augmented[column][inner] /= pivot

        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for inner in range(column, size + 1):
                augmented[row][inner] -= factor * augmented[column][inner]

    return [augmented[row][size] for row in range(size)]


class AlphaBot3(BaseBot):
    WATCHLIST = (
        "TIDE_SPOT",
        "TIDE_SWING",
        "WX_SPOT",
        "WX_SUM",
        "LHR_COUNT",
        "LHR_INDEX",
        "LON_ETF",
        "LON_FLY",
    )

    PRODUCT_CONFIGS = {
        "TIDE_SPOT": ProductConfig(11.0, 5.2, 26, 5.5, 0.30, 1.05),
        "TIDE_SWING": ProductConfig(15.0, 7.0, 18, 9.0, 0.14, 0.85),
        "WX_SPOT": ProductConfig(10.0, 4.8, 28, 4.5, 0.30, 1.05),
        "WX_SUM": ProductConfig(12.0, 5.6, 22, 7.0, 0.18, 0.90),
        "LHR_COUNT": ProductConfig(11.0, 4.8, 28, 5.5, 0.22, 1.00),
        "LHR_INDEX": ProductConfig(18.0, 8.0, 8, 9.0, 0.06, 0.65),
        "LON_ETF": ProductConfig(12.0, 5.0, 36, 7.0, 0.15, 1.50),
        "LON_FLY": ProductConfig(18.0, 8.0, 16, 11.0, 0.03, 1.70),
    }

    # When enabled, structural derived-product dislocations dominate capital allocation.
    STRUCTURE_FIRST = True
    REFRESH_SECS = 240.0
    EVAL_SECS = 2.0
    MIN_REST_GAP = 1.05
    MAX_ACTIVE_QUOTES = 2
    SETTLEMENT_GUARD_MINUTES = 8
    SETTLEMENT_GUARD_MULTIPLIER = 4.0
    SMOOTH_ALPHA = 0.28
    TOP2_SCORE_CLOSE_RATIO = 0.93
    HIGH_CONVICTION_EDGE = 7.0
    STAND_DOWN_EDGE = 5.5
    BASKET_DISLOCATION_THRESHOLD = 45.0
    BASKET_AGGRESS_THRESHOLD = 90.0
    FLY_DISLOCATION_THRESHOLD = 55.0
    FLY_AGGRESS_THRESHOLD = 95.0
    LHR_INDEX_SUPER_EDGE_MULTIPLIER = 1.35
    DERIVED_STRENGTH_SCORE = 18.0
    COMPARABLE_SCORE_RATIO = 0.92
    STRUCTURE_DOMINANCE_RATIO = 0.85
    OVERNIGHT_START_HOUR = 0
    OVERNIGHT_END_HOUR = 5
    LOW_UNCERTAINTY_SIGMA = 85.0

    def __init__(
        self,
        cmi_url: str,
        username: str,
        password: str,
        *,
        aerodatabox_key: str | None = None,
        base_order_size: int = 4,
    ):
        super().__init__(cmi_url, username, password)
        self.aerodatabox_key = aerodatabox_key
        self.base_order_size = base_order_size

        self.products: dict[str, Product] = {}
        self.books: dict[str, OrderBook] = {}
        self.positions: dict[str, int] = {}
        self.external_cache: dict[str, Any] = {}
        self.theos: dict[str, float] = {}
        self.smoothed_theos: dict[str, float] = {}
        self.active_quotes: dict[str, QuoteState] = {}

        self.last_refresh_at = 0.0
        self.last_eval_at = 0.0
        self.last_rest_at = 0.0
        self.tide_model: TideModel | None = None

        self.ml_models = {symbol: OnlineLinearModel(6) for symbol in self.WATCHLIST}
        self.ml_pending: dict[str, tuple[list[float], float]] = {}
        self._last_theo_log_at = 0.0
        self.structure_state: dict[str, Any] = {}
        self.last_priority_reason = "IDLE"

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
        loaded_products = self._paced(lambda: self.get_products())
        self.products = {product.symbol: product for product in loaded_products}
        self.positions = self._paced(lambda: self.get_positions())
        self._refresh_external(force=True)
        self.start()
        print(f"AlphaBot3 started for {self.username}. Watching: {', '.join(self.WATCHLIST)}")
        try:
            while True:
                self._maybe_evaluate(force=True)
                time.sleep(1.0)
        except KeyboardInterrupt:
            self._cancel_all_quotes()
            self.stop()
            print("AlphaBot3 stopped.")

    def _maybe_evaluate(self, force: bool = False) -> None:
        now_monotonic = time.monotonic()
        if not force and now_monotonic - self.last_eval_at < self.EVAL_SECS:
            return
        if not self._eval_lock.acquire(blocking=False):
            return
        self.last_eval_at = now_monotonic
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

        fresh_theos = self._build_theos(books)
        self.theos = self._smooth_theos(fresh_theos)
        self._maybe_log_theos()

        signals: list[dict[str, Any]] = []
        for symbol in self.WATCHLIST:
            book = books.get(symbol)
            fair = self.theos.get(symbol)
            if not book or fair is None:
                continue
            signal = self._signal_for(symbol, book, fair, positions.get(symbol, 0))
            if signal:
                signals.append(signal)

        if not signals:
            self._cancel_all_quotes()
            return

        selected = self._select_priority_signals(signals)
        if not selected:
            self._cancel_all_quotes()
            return

        selected_symbols = {signal["product"] for signal in selected}
        highest = selected[0]
        if highest["priority_reason"] == "STRUCTURE":
            for symbol in list(self.active_quotes.keys()):
                if symbol not in selected_symbols:
                    self._cancel_quote(symbol)
        else:
            for symbol in list(self.active_quotes.keys()):
                if symbol not in selected_symbols:
                    self._cancel_quote(symbol)

        active_targets: set[str] = set()
        for signal in selected:
            symbol = signal["product"]
            book = books[symbol]
            fair = signal["effective_fair"]
            position = positions.get(symbol, 0)

            if signal["edge"] >= signal["take_threshold"]:
                self._cancel_quote(symbol)
                self._take_liquidity(signal, book, fair, position)
                continue

            if signal["edge"] >= signal["quote_threshold"]:
                self._quote(signal, book, fair, position)
                active_targets.add(symbol)
            else:
                self._cancel_quote(symbol)

        for symbol in list(self.active_quotes.keys()):
            if symbol not in active_targets:
                self._cancel_quote(symbol)

    def _build_theos(self, books: dict[str, OrderBook]) -> dict[str, float]:
        tide = self.external_cache.get("thames", {})
        weather = self.external_cache.get("weather", {})
        flights = self.external_cache.get("flights", {})

        tide_spot = self._theo_tide_spot(books, tide)
        tide_swing = self._theo_tide_swing(books, tide)
        wx_spot = self._theo_wx_spot(books, weather)
        wx_sum = self._theo_wx_sum(books, weather)
        lhr_count = self._theo_lhr_count(books, flights, tide_spot, wx_spot)
        lhr_index = self._theo_lhr_index(books, flights)
        lon_etf = tide_spot + wx_spot + lhr_count
        lon_fly = self._theo_lon_fly(books, lon_etf)
        etf_mid = self._mid(books.get("LON_ETF"))
        fly_mid = self._mid(books.get("LON_FLY"))
        basket_dislocation = None if etf_mid is None else etf_mid - lon_etf
        fly_dislocation = None if fly_mid is None else fly_mid - lon_fly

        theos = {
            "TIDE_SPOT": tide_spot,
            "TIDE_SWING": tide_swing,
            "WX_SPOT": wx_spot,
            "WX_SUM": wx_sum,
            "LHR_COUNT": lhr_count,
            "LHR_INDEX": lhr_index,
            "LON_ETF": lon_etf,
            "LON_FLY": lon_fly,
        }

        # Light midpoint anchoring only where the model is inherently noisier.
        for symbol, weight in (("TIDE_SWING", 0.08), ("WX_SUM", 0.08), ("LHR_INDEX", 0.10)):
            mid = self._mid(books.get(symbol))
            if mid is not None:
                theos[symbol] = (1.0 - weight) * theos[symbol] + weight * mid

        self.structure_state = {
            "basket_dislocation": basket_dislocation,
            "fly_dislocation": fly_dislocation,
            "etf_model_fair": lon_etf,
            "fly_model_fair": lon_fly,
            "etf_mid": etf_mid,
            "fly_mid": fly_mid,
        }

        return theos

    def _theo_tide_spot(self, books: dict[str, OrderBook], tide: dict[str, Any]) -> float:
        if tide.get("projected_level_m") is not None:
            return max(0.0, abs(float(tide["projected_level_m"])) * 1000.0)
        if tide.get("latest_level_m") is not None:
            return max(0.0, abs(float(tide["latest_level_m"])) * 1000.0)
        return self._fallback("TIDE_SPOT", books)

    def _theo_tide_swing(self, books: dict[str, OrderBook], tide: dict[str, Any]) -> float:
        if tide.get("swing_fair") is not None:
            return max(0.0, float(tide["swing_fair"]))
        return self._fallback("TIDE_SWING", books)

    def _theo_wx_spot(self, books: dict[str, OrderBook], weather: dict[str, Any]) -> float:
        if weather.get("wx_spot_settle") is not None:
            return max(0.0, float(weather["wx_spot_settle"]))
        return self._fallback("WX_SPOT", books)

    def _theo_wx_sum(self, books: dict[str, OrderBook], weather: dict[str, Any]) -> float:
        if weather.get("wx_sum_fair") is not None:
            return max(0.0, float(weather["wx_sum_fair"]))
        return self._fallback("WX_SUM", books)

    def _theo_lhr_count(
        self,
        books: dict[str, OrderBook],
        flights: dict[str, Any],
        tide_spot: float,
        wx_spot: float,
    ) -> float:
        if flights.get("count_fair") is not None:
            return max(0.0, float(flights["count_fair"]))
        etf_mid = self._mid(books.get("LON_ETF"))
        if etf_mid is not None:
            return max(0.0, etf_mid - tide_spot - wx_spot)
        product = self.products.get("LHR_COUNT")
        return float(product.startingPrice) if product else 1500.0

    def _theo_lhr_index(self, books: dict[str, OrderBook], flights: dict[str, Any]) -> float:
        if flights.get("index_fair") is not None:
            return max(0.0, float(flights["index_fair"]))
        return self._fallback("LHR_INDEX", books)

    def _theo_lon_fly(self, books: dict[str, OrderBook], lon_etf: float) -> float:
        sigma = self._estimate_etf_uncertainty()
        scenario_down = fly_payoff(max(0.0, lon_etf - sigma))
        scenario_mid = fly_payoff(lon_etf)
        scenario_up = fly_payoff(lon_etf + sigma)
        structural = 0.2 * scenario_down + 0.6 * scenario_mid + 0.2 * scenario_up

        market_mid = self._mid(books.get("LON_FLY"))
        if market_mid is not None and sigma < self.LOW_UNCERTAINTY_SIGMA:
            structural = 0.96 * structural + 0.04 * market_mid
        return structural

    def _estimate_etf_uncertainty(self) -> float:
        progress_tide = self._progress_for("TIDE_SPOT")
        progress_wx = self._progress_for("WX_SPOT")
        progress_lhr = self._progress_for("LHR_COUNT")

        tide_sigma = 220.0 * (1.0 - progress_tide)
        wx_sigma = 150.0 * (1.0 - progress_wx)
        lhr_sigma = 90.0 * (1.0 - progress_lhr)
        return math.sqrt(tide_sigma * tide_sigma + wx_sigma * wx_sigma + lhr_sigma * lhr_sigma)

    def _signal_for(
        self,
        symbol: str,
        book: OrderBook,
        structural_fair: float,
        position: int,
    ) -> dict[str, Any] | None:
        config = self.PRODUCT_CONFIGS[symbol]
        best_bid = self._best_bid(book)
        best_ask = self._best_ask(book)
        if best_bid is None and best_ask is None:
            return None

        progress = self._progress_for(symbol)
        mid = self._mid(book)
        spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else max(2.0, config.base_width)
        imbalance = self._top_level_imbalance(book)
        features = self._ml_features(structural_fair, mid, spread, imbalance, progress, position, config.max_position)
        ml_adjust = self._ml_adjust(symbol, features, mid)
        effective_fair = structural_fair + config.ml_weight * ml_adjust

        # Allow inventory-reducing trades even at position limits.
        # At +max_position: block new buys, but always allow sells.
        # At -max_position: block new sells, but always allow buys.
        can_buy = position < config.max_position
        can_sell = position > -config.max_position
        buy_edge = effective_fair - best_ask if best_ask is not None and can_buy else float("-inf")
        sell_edge = best_bid - effective_fair if best_bid is not None and can_sell else float("-inf")
        edge = max(buy_edge, sell_edge, 0.0)
        if edge <= 0.0:
            # Even with no edge, generate a signal when at extreme inventory
            # so the quoting logic can skew prices to encourage mean-reversion.
            inventory_ratio = abs(position) / max(config.max_position, 1)
            if inventory_ratio >= 0.85:
                # Synthetic signal to let _quote run with strong skew
                direction = -1.0 if position > 0 else 1.0
                return {
                    "product": symbol,
                    "edge": config.quote_edge,  # minimum passable edge
                    "score": config.quote_edge * 0.5,
                    "buy_edge": config.quote_edge if direction > 0 else 0.0,
                    "sell_edge": config.quote_edge if direction < 0 else 0.0,
                    "effective_fair": effective_fair,
                    "progress": progress,
                    "ml_adjust": ml_adjust,
                }
            return None

        direction = 1.0 if buy_edge >= sell_edge else -1.0
        aligned_ml = direction * ml_adjust
        inventory_penalty = max(0.0, abs(position) / max(config.max_position, 1) - 0.35) * config.quote_edge
        confidence_boost = 0.35 + 0.65 * progress
        score = edge * (1.0 + confidence_boost) + 0.18 * max(0.0, aligned_ml) - inventory_penalty

        take_threshold = config.take_edge
        quote_threshold = config.quote_edge
        size_boost = 1.0
        priority_reason = "ML_MICRO" if abs(ml_adjust) > 2.0 else "BASE"

        if symbol in {"TIDE_SWING", "WX_SUM", "LHR_COUNT", "LHR_INDEX"}:
            late_path = 1.0 + 1.15 * progress
            score *= late_path
            size_boost *= 0.7 + 1.6 * progress
            if progress > 0.7:
                priority_reason = "LATE_PATH"

        if symbol in {"TIDE_SPOT", "WX_SPOT"}:
            snapshot_reliability = 1.0 + 0.55 * progress
            score *= snapshot_reliability
            size_boost *= 0.85 + 0.85 * progress
            if progress > 0.65:
                quote_threshold += 0.5
                priority_reason = "SNAPSHOT_CONVERGENCE"

        if symbol == "LHR_INDEX":
            if edge < config.take_edge * self.LHR_INDEX_SUPER_EDGE_MULTIPLIER:
                score *= 0.45
                quote_threshold += 1.5
                take_threshold += 2.5
            else:
                priority_reason = "LATE_PATH"

        basket_dislocation = self.structure_state.get("basket_dislocation")
        fly_dislocation = self.structure_state.get("fly_dislocation")
        if symbol == "LON_ETF" and basket_dislocation is not None:
            abs_basket = abs(basket_dislocation)
            if abs_basket >= self.BASKET_DISLOCATION_THRESHOLD:
                score += 0.9 * abs_basket
                size_boost *= 1.35
                quote_threshold = max(3.5, quote_threshold - 1.0)
                if abs_basket >= self.BASKET_AGGRESS_THRESHOLD:
                    take_threshold = max(5.0, take_threshold - 2.0)
                    size_boost *= 1.25
                priority_reason = "STRUCTURE"
            elif self.STRUCTURE_FIRST:
                score *= 1.08

        if symbol == "LON_FLY" and fly_dislocation is not None:
            abs_fly = abs(fly_dislocation)
            if abs_fly < self.FLY_DISLOCATION_THRESHOLD and edge < config.take_edge:
                return None
            if abs_fly >= self.FLY_DISLOCATION_THRESHOLD:
                score += 0.95 * abs_fly
                quote_threshold = max(config.quote_edge + 0.5, quote_threshold)
                size_boost *= 1.45
                if abs_fly >= self.FLY_AGGRESS_THRESHOLD:
                    take_threshold = max(8.0, take_threshold - 2.5)
                    size_boost *= 1.35
                priority_reason = "STRUCTURE"
            else:
                score *= 0.75

        if symbol == "LHR_COUNT":
            count_remaining = float(self.external_cache.get("flights", {}).get("count_remaining_weighted", 0.0))
            market_mid = self._mid(book)
            if self._is_overnight_lhr_window() and market_mid is not None:
                remaining_overstatement = market_mid - structural_fair
                if remaining_overstatement > max(18.0, 0.35 * count_remaining) and sell_edge > 0.0:
                    score += min(30.0, 0.45 * remaining_overstatement)
                    size_boost *= 1.25
                    take_threshold = max(6.0, take_threshold - 1.0)
                    priority_reason = "LATE_PATH"

        return {
            "product": symbol,
            "edge": edge,
            "score": score,
            "buy_edge": max(0.0, buy_edge),
            "sell_edge": max(0.0, sell_edge),
            "effective_fair": effective_fair,
            "progress": progress,
            "ml_adjust": ml_adjust,
            "take_threshold": take_threshold,
            "quote_threshold": quote_threshold,
            "size_boost": size_boost,
            "priority_reason": priority_reason,
            "structural_fair": structural_fair,
        }

    def _ml_features(
        self,
        fair: float,
        mid: float | None,
        spread: float,
        imbalance: float,
        progress: float,
        position: int,
        max_position: int,
    ) -> list[float]:
        midpoint = fair if mid is None else mid
        deviation = (midpoint - fair) / max(spread, 1.0)
        minutes = self._minutes_to_settlement()
        time_scale = clamp(minutes / (24.0 * 60.0), 0.0, 1.0)
        inventory = clamp(position / max(max_position, 1), -1.0, 1.0)
        return [
            1.0,
            clamp(deviation, -5.0, 5.0),
            clamp(spread / 25.0, 0.0, 5.0),
            clamp(imbalance, -1.0, 1.0),
            progress,
            inventory + (1.0 - time_scale) * 0.2,
        ]

    def _ml_adjust(self, symbol: str, features: list[float], mid: float | None) -> float:
        model = self.ml_models[symbol]
        if mid is not None:
            pending = self.ml_pending.get(symbol)
            if pending is not None:
                old_features, old_mid = pending
                model.update(old_features, mid - old_mid)
            self.ml_pending[symbol] = (features, mid)

        if model.samples < 8:
            return 0.0
        prediction = model.predict(features)
        return clamp(prediction, -12.0, 12.0)

    def _select_priority_signals(self, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted(signals, key=lambda item: item["score"], reverse=True)
        if not ranked:
            return []

        best = ranked[0]
        strong_derived = max(
            (
                signal["score"]
                for signal in ranked
                if signal["product"] in {"LON_ETF", "LON_FLY"} and signal["priority_reason"] == "STRUCTURE"
            ),
            default=0.0,
        )
        if strong_derived >= self.DERIVED_STRENGTH_SCORE:
            for signal in ranked:
                if signal["product"] not in {"LON_ETF", "LON_FLY"} and signal["score"] < strong_derived:
                    signal["score"] *= 0.55 if signal["edge"] < signal["take_threshold"] else 0.78
            ranked.sort(key=lambda item: item["score"], reverse=True)
            best = ranked[0]

        if self.STRUCTURE_FIRST and best["product"] not in {"LON_ETF", "LON_FLY"}:
            for signal in ranked[1:]:
                if (
                    signal["product"] in {"LON_ETF", "LON_FLY"}
                    and signal["score"] >= best["score"] * self.COMPARABLE_SCORE_RATIO
                ):
                    best = signal
                    break
            ranked.sort(
                key=lambda item: (
                    0 if item["product"] == best["product"] else 1,
                    -item["score"],
                )
            )

        if best["edge"] < self.STAND_DOWN_EDGE:
            self.last_priority_reason = "STAND_DOWN"
            print(f"PRIORITY STAND_DOWN best={best['product']} edge={best['edge']:.1f}")
            return []

        chosen = [ranked[0]]
        allow_two = (
            best["edge"] >= self.HIGH_CONVICTION_EDGE
            and len(ranked) > 1
            and ranked[1]["score"] >= ranked[0]["score"] * self.TOP2_SCORE_CLOSE_RATIO
            and ranked[1]["edge"] >= self.HIGH_CONVICTION_EDGE
        )
        if allow_two:
            chosen.append(ranked[1])

        if best["priority_reason"] == "STRUCTURE":
            chosen = [best] + [
                signal
                for signal in chosen[1:]
                if signal["product"] in {"LON_ETF", "LON_FLY"}
                or signal["score"] >= best["score"] * self.STRUCTURE_DOMINANCE_RATIO
            ]

        self.last_priority_reason = best["priority_reason"]
        selected_text = ", ".join(
            f"{signal['product']}({signal['score']:.1f},{signal['priority_reason']})"
            for signal in chosen
        )
        print(f"PRIORITY {best['priority_reason']} selected={selected_text}")
        return chosen

    def _take_liquidity(
        self,
        signal: dict[str, Any],
        book: OrderBook,
        fair: float,
        position: int,
    ) -> None:
        symbol = signal["product"]
        config = self.PRODUCT_CONFIGS[symbol]
        best_ask_order = self._best_ask_order(book)
        best_bid_order = self._best_bid_order(book)

        if best_ask_order and signal["buy_edge"] >= signal["take_threshold"] and position < config.max_position:
            available = best_ask_order.volume - best_ask_order.own_volume
            size = min(self._size_for(symbol, position, signal), available, config.max_position - position)
            if size > 0:
                self._send_ioc(OrderRequest(symbol, best_ask_order.price, Side.BUY, size))
                print(f"HIT BUY  {size} {symbol} @ {best_ask_order.price:.0f}  theo={fair:.1f}  why={signal['priority_reason']}")
                return

        if best_bid_order and signal["sell_edge"] >= signal["take_threshold"] and position > -config.max_position:
            available = best_bid_order.volume - best_bid_order.own_volume
            size = min(self._size_for(symbol, position, signal), available, config.max_position + position)
            if size > 0:
                self._send_ioc(OrderRequest(symbol, best_bid_order.price, Side.SELL, size))
                print(f"HIT SELL {size} {symbol} @ {best_bid_order.price:.0f}  theo={fair:.1f}  why={signal['priority_reason']}")

    def _quote(
        self,
        signal: dict[str, Any],
        book: OrderBook,
        fair: float,
        position: int,
    ) -> None:
        symbol = signal["product"]
        config = self.PRODUCT_CONFIGS[symbol]
        product = self.products.get(symbol)
        if not product:
            return

        tick = product.tickSize or 1.0
        best_bid = self._best_bid(book)
        best_ask = self._best_ask(book)
        size = self._size_for(symbol, position, signal)
        if size <= 0:
            self._cancel_quote(symbol)
            return

        spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else config.base_width * 2.0
        half_width = max(config.base_width, spread / 2.0)
        if self._in_settlement_guard():
            half_width *= self.SETTLEMENT_GUARD_MULTIPLIER

        # Path products can quote tighter as more of their value is realized.
        progress = signal["progress"]
        half_width *= 1.15 - 0.25 * progress
        if symbol in {"TIDE_SPOT", "WX_SPOT"}:
            half_width *= 1.10 - 0.40 * progress
        if symbol == "LON_ETF" and signal["priority_reason"] == "STRUCTURE":
            half_width *= 0.75
        if symbol == "LON_FLY" and signal["priority_reason"] == "STRUCTURE":
            half_width *= 0.80

        # Inventory skewing: aggressively move prices to encourage
        # the market to trade us back toward a neutral position.
        inventory_ratio = clamp(position / max(config.max_position, 1), -1.0, 1.0)
        base_skew = 2.5 + 2.5 * (1.0 - progress)
        # At extreme positions (>85% utilized), add extra skew to
        # make the inventory-reducing side much more attractive.
        extreme_bonus = max(0.0, abs(inventory_ratio) - 0.85) * 15.0
        skew = inventory_ratio * (base_skew + extreme_bonus)
        bid_price = math.floor((fair - half_width - skew) / tick) * tick
        ask_price = math.ceil((fair + half_width - skew) / tick) * tick

        if best_bid is not None:
            bid_price = min(bid_price, best_bid + tick)
        if best_ask is not None:
            ask_price = max(ask_price, best_ask - tick)
        if bid_price <= 0 or ask_price <= bid_price:
            return

        target_bid = bid_price if signal["buy_edge"] >= config.quote_edge and position < config.max_position else None
        target_ask = ask_price if signal["sell_edge"] >= config.quote_edge and position > -config.max_position else None
        if target_bid is None and target_ask is None:
            self._cancel_quote(symbol)
            return

        existing = self.active_quotes.get(symbol)
        if existing:
            unchanged_bid = existing.bid_price == target_bid
            unchanged_ask = existing.ask_price == target_ask
            if unchanged_bid and unchanged_ask:
                return

        self._cancel_quote(symbol)

        bid_response = None
        ask_response = None
        if target_bid is not None:
            bid_response = self._paced(lambda: self.send_order(OrderRequest(symbol, target_bid, Side.BUY, size)))
        if target_ask is not None:
            ask_response = self._paced(lambda: self.send_order(OrderRequest(symbol, target_ask, Side.SELL, size)))

        self.active_quotes[symbol] = QuoteState(
            product=symbol,
            bid_id=bid_response.id if bid_response else None,
            ask_id=ask_response.id if ask_response else None,
            bid_price=target_bid if bid_response else None,
            ask_price=target_ask if ask_response else None,
        )
        print(f"QUOTE {symbol:>10}  {target_bid or '-'} / {target_ask or '-'}  theo={fair:.1f}  why={signal['priority_reason']}")

    def _cancel_quote(self, symbol: str) -> None:
        quote = self.active_quotes.pop(symbol, None)
        if not quote:
            return
        if quote.bid_id:
            self._paced(lambda: self.cancel_order(quote.bid_id))
        if quote.ask_id:
            self._paced(lambda: self.cancel_order(quote.ask_id))

    def _cancel_all_quotes(self) -> None:
        for symbol in list(self.active_quotes.keys()):
            self._cancel_quote(symbol)

    def _send_ioc(self, order: OrderRequest) -> OrderResponse | None:
        response = self._paced(lambda: self.send_order(order))
        if response and response.filled < response.volume:
            self.cancel_order(response.id)
            self.last_rest_at = time.monotonic()
        return response

    def _size_for(self, symbol: str, position: int, signal: dict[str, Any]) -> int:
        config = self.PRODUCT_CONFIGS[symbol]
        edge = signal["edge"]
        utilization = abs(position) / max(config.max_position, 1)
        scale = 1.0 - clamp(utilization, 0.0, 0.85)
        edge_boost = 1.0 + 0.30 * clamp(edge / max(config.take_edge, 1.0), 0.0, 1.5)
        progress = signal["progress"]
        if symbol in {"TIDE_SWING", "WX_SUM", "LHR_COUNT", "LHR_INDEX"}:
            timing_bias = 0.55 + 1.55 * progress
        else:
            timing_bias = 0.85 + 1.00 * progress
        if symbol == "LON_ETF" and signal["priority_reason"] == "STRUCTURE":
            timing_bias *= 1.25
        if symbol == "LON_FLY" and signal["priority_reason"] == "STRUCTURE":
            timing_bias *= 1.35
        raw = self.base_order_size * config.size_multiplier * scale * edge_boost * timing_bias * signal.get("size_boost", 1.0)
        return max(0, int(round(raw)))

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

    def _refresh_external(self, force: bool = False) -> None:
        now_monotonic = time.monotonic()
        if not force and now_monotonic - self.last_refresh_at < self.REFRESH_SECS:
            return
        print("Refreshing external data...")
        self.external_cache["weather"] = self._fetch_weather()
        self.external_cache["thames"] = self._fetch_thames()
        if self.aerodatabox_key:
            self.external_cache["flights"] = self._fetch_flights()
        self.last_refresh_at = now_monotonic
        self._log_refresh_decomposition()
        print("External data refresh complete.")

    def _fetch_weather(self) -> dict[str, Any]:
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

            times: list[datetime] = []
            for value in raw["time"]:
                stamp = datetime.fromisoformat(value)
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=LONDON_TZ)
                else:
                    stamp = stamp.astimezone(LONDON_TZ)
                times.append(stamp)

            temperatures = raw["temperature_2m"]
            humidities = raw["relative_humidity_2m"]
            settle = self._next_settlement_time()
            now_london = datetime.now(LONDON_TZ)
            window_start = settle - timedelta(hours=24)

            settle_index = min(range(len(times)), key=lambda index: abs((times[index] - settle).total_seconds()))
            settle_temp_f = temperatures[settle_index] * 9.0 / 5.0 + 32.0
            wx_spot_settle = settle_temp_f * humidities[settle_index]

            wx_sum_realized = 0.0
            wx_sum_remaining = 0.0
            realized_intervals = 0
            total_intervals = 0
            for stamp, temp_c, humidity in zip(times, temperatures, humidities):
                if not (window_start <= stamp <= settle):
                    continue
                contribution = (temp_c * 9.0 / 5.0 + 32.0) * humidity / 100.0
                total_intervals += 1
                if stamp <= now_london:
                    wx_sum_realized += contribution
                    realized_intervals += 1
                else:
                    wx_sum_remaining += contribution

            return {
                "wx_spot_settle": wx_spot_settle,
                "wx_sum_realized": wx_sum_realized,
                "wx_sum_remaining": wx_sum_remaining,
                "wx_sum_fair": wx_sum_realized + wx_sum_remaining,
                "wx_realized_intervals": realized_intervals,
                "wx_total_intervals": max(total_intervals, 1),
            }
        except Exception as exc:
            print(f"Warning: weather fetch failed: {exc}")
            return self.external_cache.get("weather", {})

    def _fetch_thames(self) -> dict[str, Any]:
        try:
            response = requests.get(
                f"https://environment.data.gov.uk/flood-monitoring/id/measures/{THAMES_MEASURE}/readings",
                params={"_sorted": "", "_limit": 193},
                timeout=20,
            )
            response.raise_for_status()
            items = [item for item in response.json().get("items", []) if item.get("value") is not None]
            if not items:
                return self.external_cache.get("thames", {})

            items.sort(key=lambda item: item["dateTime"])
            times = [
                datetime.fromisoformat(item["dateTime"].replace("Z", "+00:00")).astimezone(LONDON_TZ)
                for item in items
            ]
            levels = [float(item["value"]) for item in items]
            now_london = datetime.now(LONDON_TZ)
            settle = self._next_settlement_time()
            window_start = settle - timedelta(hours=24)

            self.tide_model = TideModel.fit(times, levels)
            projected_level = self.tide_model.predict(settle) if self.tide_model else None

            actual_swing = 0.0
            realized_intervals = 0
            last_observed = None
            for previous_time, current_time, previous_level, current_level in zip(times, times[1:], levels, levels[1:]):
                if previous_time < window_start or current_time > settle:
                    continue
                if current_time <= now_london:
                    diff_cm = abs(current_level - previous_level) * 100.0
                    actual_swing += max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)
                    realized_intervals += 1
                    last_observed = current_time

            projected_swing = 0.0
            total_intervals = 96
            if self.tide_model:
                anchor = last_observed if last_observed is not None else window_start
                previous_level = self.tide_model.predict(anchor)
                cursor = anchor
                while cursor < settle:
                    cursor += timedelta(minutes=15)
                    if cursor > settle:
                        cursor = settle
                    current_level = self.tide_model.predict(cursor)
                    if cursor > now_london:
                        diff_cm = abs(current_level - previous_level) * 100.0
                        projected_swing += max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)
                    previous_level = current_level

            return {
                "latest_level_m": levels[-1],
                "projected_level_m": projected_level,
                "swing_realized": actual_swing,
                "swing_remaining": projected_swing,
                "swing_fair": actual_swing + projected_swing,
                "swing_realized_intervals": realized_intervals,
                "swing_total_intervals": total_intervals,
            }
        except Exception as exc:
            print(f"Warning: Thames fetch failed: {exc}")
            return self.external_cache.get("thames", {})

    def _fetch_flights(self) -> dict[str, Any]:
        try:
            settle = self._next_settlement_time()
            window_start = settle - timedelta(hours=24)
            now_london = datetime.now(LONDON_TZ)

            records: list[dict[str, Any]] = []
            for chunk_start, chunk_end in (
                (window_start, window_start + timedelta(hours=12)),
                (window_start + timedelta(hours=12), settle),
            ):
                response = requests.get(
                    f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/LHR/{chunk_start:%Y-%m-%dT%H:%M}/{chunk_end:%Y-%m-%dT%H:%M}",
                    params={"direction": "Both"},
                    headers={
                        "x-rapidapi-host": "aerodatabox.p.rapidapi.com",
                        "x-rapidapi-key": self.aerodatabox_key,
                    },
                    timeout=15,
                )
                response.raise_for_status()
                payload = response.json()
                for side, flights in (("arrival", payload.get("arrivals", [])), ("departure", payload.get("departures", []))):
                    for flight in flights:
                        movement = (flight.get("movement", {}) or {})
                        when = self._parse_flight_time(flight)
                        if when is None:
                            continue
                        if not (window_start <= when <= settle):
                            continue
                        status = str(flight.get("status", "")).lower()
                        records.append(
                            {
                                "side": side,
                                "time": when,
                                "status": status,
                                "cancelled": status in {"cancelled", "canceled"},
                            }
                        )

            if not records:
                return self.external_cache.get("flights", {})

            count_realized = 0.0
            count_remaining = 0.0
            bucket_values: list[tuple[datetime, datetime, int, int]] = []
            current_bucket = window_start
            for _ in range(48):
                bucket_end = current_bucket + timedelta(minutes=30)
                arrivals = 0
                departures = 0
                for record in records:
                    if record["cancelled"]:
                        continue
                    if current_bucket <= record["time"] < bucket_end:
                        if record["side"] == "arrival":
                            arrivals += 1
                        else:
                            departures += 1
                bucket_values.append((current_bucket, bucket_end, arrivals, departures))
                current_bucket = bucket_end

            for record in records:
                if record["cancelled"]:
                    continue
                if record["time"] <= now_london:
                    count_realized += 1.0
                    continue
                hours_ahead = max(0.0, (record["time"] - now_london).total_seconds() / 3600.0)
                reliability = 0.995
                if hours_ahead > 18.0:
                    reliability = 0.92
                elif hours_ahead > 12.0:
                    reliability = 0.95
                elif hours_ahead > 6.0:
                    reliability = 0.97
                elif hours_ahead > 2.0:
                    reliability = 0.985
                count_remaining += reliability

            raw_realized = 0.0
            raw_projected = 0.0
            realized_buckets = 0
            for bucket_start, bucket_end, arrivals, departures in bucket_values:
                total = arrivals + departures
                if total <= 0:
                    contribution = 0.0
                else:
                    contribution = 100.0 * (arrivals - departures) / total
                if bucket_end <= now_london:
                    raw_realized += contribution
                    realized_buckets += 1
                else:
                    hours_ahead = max(0.0, (bucket_end - now_london).total_seconds() / 3600.0)
                    damp = 1.0
                    if hours_ahead > 12.0:
                        damp = 0.85
                    elif hours_ahead > 6.0:
                        damp = 0.90
                    elif hours_ahead > 2.0:
                        damp = 0.95
                    raw_projected += contribution * damp

            return {
                "count_realized": count_realized,
                "count_remaining_weighted": count_remaining,
                "count_fair": count_realized + count_remaining,
                "index_realized_raw": raw_realized,
                "index_projected_raw": raw_projected,
                "index_fair": abs(raw_realized + raw_projected),
                "index_realized_buckets": realized_buckets,
                "index_total_buckets": 48,
            }
        except Exception as exc:
            print(f"Warning: flight fetch failed: {exc}")
            return self.external_cache.get("flights", {})

    def _parse_flight_time(self, flight: dict[str, Any]) -> datetime | None:
        try:
            movement = (flight.get("movement", {}) or {})
            candidates = [
                (movement.get("actualTime") or {}).get("local"),
                flight.get("actualTimeLocal"),
                (movement.get("revisedTime") or {}).get("local"),
                movement.get("actualTimeLocal"),
                movement.get("scheduledTimeLocal"),
                (movement.get("scheduledTime") or {}).get("local"),
                flight.get("scheduledTimeLocal"),
            ]
            for candidate in candidates:
                if not candidate:
                    continue
                parsed = datetime.fromisoformat(candidate)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=LONDON_TZ)
                else:
                    parsed = parsed.astimezone(LONDON_TZ)
                return parsed
        except Exception:
            return None
        return None

    def _progress_for(self, symbol: str) -> float:
        weather = self.external_cache.get("weather", {})
        tide = self.external_cache.get("thames", {})
        flights = self.external_cache.get("flights", {})
        default = clamp(1.0 - self._minutes_to_settlement() / (24.0 * 60.0), 0.0, 1.0)

        if symbol == "TIDE_SWING":
            realized = float(tide.get("swing_realized_intervals", 0.0))
            total = max(float(tide.get("swing_total_intervals", 96.0)), 1.0)
            return clamp(realized / total, 0.0, 1.0)
        if symbol == "WX_SUM":
            realized = float(weather.get("wx_realized_intervals", 0.0))
            total = max(float(weather.get("wx_total_intervals", 96.0)), 1.0)
            return clamp(realized / total, 0.0, 1.0)
        if symbol == "LHR_COUNT":
            realized = float(flights.get("count_realized", 0.0))
            total = realized + float(flights.get("count_remaining_weighted", 0.0))
            return clamp(realized / total, 0.0, 1.0) if total > 0.0 else default
        if symbol == "LHR_INDEX":
            realized = float(flights.get("index_realized_buckets", 0.0))
            total = max(float(flights.get("index_total_buckets", 48.0)), 1.0)
            return clamp(realized / total, 0.0, 1.0)
        if symbol == "LON_ETF":
            return min(self._progress_for("TIDE_SPOT"), self._progress_for("WX_SPOT"), self._progress_for("LHR_COUNT"))
        if symbol == "LON_FLY":
            return self._progress_for("LON_ETF")
        return default

    def _is_overnight_lhr_window(self) -> bool:
        now_london = datetime.now(LONDON_TZ)
        return self.OVERNIGHT_START_HOUR <= now_london.hour <= self.OVERNIGHT_END_HOUR

    def _minutes_to_settlement(self) -> float:
        settle = self._next_settlement_time()
        now_london = datetime.now(LONDON_TZ)
        return max(0.0, (settle - now_london).total_seconds() / 60.0)

    def _in_settlement_guard(self) -> bool:
        minutes = self._minutes_to_settlement()
        return 0.0 < minutes <= self.SETTLEMENT_GUARD_MINUTES

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

    def _top_level_imbalance(self, book: OrderBook) -> float:
        bid_order = self._best_bid_order(book)
        ask_order = self._best_ask_order(book)
        bid_volume = max(0, bid_order.volume - bid_order.own_volume) if bid_order else 0
        ask_volume = max(0, ask_order.volume - ask_order.own_volume) if ask_order else 0
        total = bid_volume + ask_volume
        if total <= 0:
            return 0.0
        return (bid_volume - ask_volume) / total

    def _maybe_log_theos(self) -> None:
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_theo_log_at < 15.0:
            return
        self._last_theo_log_at = now_monotonic
        formatted = "  ".join(f"{symbol}={value:.0f}" for symbol, value in self.theos.items())
        basket = self.structure_state.get("basket_dislocation")
        fly = self.structure_state.get("fly_dislocation")
        extras: list[str] = []
        if basket is not None:
            extras.append(f"basket={basket:+.1f}")
        if fly is not None:
            extras.append(f"fly={fly:+.1f}")
        suffix = "" if not extras else "  " + "  ".join(extras)
        print(f"THEOS  {formatted}{suffix}")

    def _log_refresh_decomposition(self) -> None:
        weather = self.external_cache.get("weather", {})
        tide = self.external_cache.get("thames", {})
        flights = self.external_cache.get("flights", {})

        if weather:
            print(
                "WX_PATH "
                f"realized={float(weather.get('wx_sum_realized', 0.0)):.1f} "
                f"remaining={float(weather.get('wx_sum_remaining', 0.0)):.1f}"
            )
        if tide:
            print(
                "TIDE_PATH "
                f"realized={float(tide.get('swing_realized', 0.0)):.1f} "
                f"remaining={float(tide.get('swing_remaining', 0.0)):.1f}"
            )
        if flights:
            print(
                "LHR_PATH "
                f"count={float(flights.get('count_realized', 0.0)):.1f}+{float(flights.get('count_remaining_weighted', 0.0)):.1f} "
                f"index={float(flights.get('index_realized_raw', 0.0)):.1f}+{float(flights.get('index_projected_raw', 0.0)):.1f}"
            )

    def _next_settlement_time(self) -> datetime:
        now_london = datetime.now(LONDON_TZ)
        settle = now_london.replace(hour=12, minute=0, second=0, microsecond=0)
        if now_london >= settle:
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
    load_env_file()

    EXCHANGE_URL = os.getenv("CMI_EXCHANGE_URL", "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com")
    USERNAME = os.getenv("CMI_USERNAME")
    PASSWORD = os.getenv("CMI_PASSWORD")
    AERODATABOX_KEY = os.getenv("AERODATABOX_KEY")

    print(f"connecting to {USERNAME} at {EXCHANGE_URL}")

    if not USERNAME or not PASSWORD:
        raise SystemExit("Set CMI_USERNAME and CMI_PASSWORD in .env before running alphabot3.py.")

    bot = AlphaBot3(
        EXCHANGE_URL,
        USERNAME,
        PASSWORD,
        aerodatabox_key=AERODATABOX_KEY,
        base_order_size=4,
    )
    bot.run()
