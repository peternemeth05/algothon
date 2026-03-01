"""AlphaBot2 – Predictive-valuation trading bot for the IMCity London Challenge.

Computes settlement-formula-precise theos for all 8 products:
  TIDE_SPOT, TIDE_SWING, WX_SPOT, WX_SUM,
  LHR_COUNT, LHR_INDEX, LON_ETF, LON_FLY

Key improvements over v1:
  - Multi-harmonic (M2+S2+M4) tidal model
  - Exact settlement formulas (strangle, T×H, directional index)
  - All 8 products on the watchlist (including LHR_COUNT, LHR_INDEX)
  - Multi-product quoting (up to 3 simultaneous)
  - ±100 position limit, 2×tick order-refresh guard, 10× settlement spread
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
    from bot_template import (
        BaseBot,
        OrderBook,
        OrderRequest,
        OrderResponse,
        Product,
        Side,
        Trade,
    )
except ModuleNotFoundError:
    import sys

    fallback = Path(__file__).resolve().parent
    if fallback.exists():
        sys.path.append(str(fallback))
    from bot_template import (
        BaseBot,
        OrderBook,
        OrderRequest,
        OrderResponse,
        Product,
        Side,
        Trade,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LONDON_LAT = 51.5074
LONDON_LON = -0.1278
THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"
LONDON_TZ = ZoneInfo("Europe/London")

# Tidal constituent periods (seconds)
M2_PERIOD = 12 * 3600 + 25 * 60       # 12 h 25 min – principal lunar
S2_PERIOD = 12 * 3600                   # 12 h 00 min – principal solar
M4_PERIOD = (12 * 3600 + 25 * 60) / 2  # 6 h 12 min 30 s – first overtide


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def call_payoff(spot: float, strike: float) -> float:
    """Call option payoff: max(0, S - K)."""
    return max(0.0, spot - strike)


def put_payoff(spot: float, strike: float) -> float:
    """Put option payoff: max(0, K - S)."""
    return max(0.0, strike - spot)


def fly_payoff(etf: float) -> float:
    """LON_FLY package: +2 P(6200), +1 C(6200), -2 C(6600), +3 C(7000)."""
    return (
        2.0 * put_payoff(etf, 6200.0)
        + 1.0 * call_payoff(etf, 6200.0)
        - 2.0 * call_payoff(etf, 6600.0)
        + 3.0 * call_payoff(etf, 7000.0)
    )


def load_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE pairs into the process env without requiring python-dotenv."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QuoteState:
    product: str
    bid_id: str | None = None
    ask_id: str | None = None
    bid_price: float | None = None
    ask_price: float | None = None


# ---------------------------------------------------------------------------
# Multi-harmonic tidal model
# ---------------------------------------------------------------------------

class TideModel:
    """Least-squares fit of M2 + S2 + M4 harmonics to observed tidal data."""

    PERIODS = (M2_PERIOD, S2_PERIOD, M4_PERIOD)  # 3 constituents

    def __init__(self, offset: float, coeffs: list[tuple[float, float]], anchor_ts: float):
        """
        Parameters
        ----------
        offset   : mean water level
        coeffs   : [(A_k, B_k), ...] for each constituent  (sin, cos)
        anchor_ts: POSIX timestamp of reference time
        """
        self.offset = offset
        self.coeffs = coeffs
        self.anchor_ts = anchor_ts

    def predict(self, target: datetime) -> float:
        """Predict the tidal level at *target* datetime."""
        dt = target.timestamp() - self.anchor_ts
        level = self.offset
        for (a, b), period in zip(self.coeffs, self.PERIODS):
            omega = 2.0 * math.pi / period
            phase = omega * dt
            level += a * math.sin(phase) + b * math.cos(phase)
        return level

    @classmethod
    def fit(cls, times: list[datetime], levels: list[float], max_samples: int = 192) -> TideModel | None:
        """Fit a 3-constituent harmonic model via normal equations (AᵀA x = Aᵀb)."""
        n = min(len(times), len(levels), max_samples)
        if n < 20:
            return None

        fit_times = times[-n:]
        fit_levels = levels[-n:]
        anchor_ts = fit_times[-1].timestamp()

        # Number of parameters: 1 (offset) + 2 per constituent = 7
        num_params = 1 + 2 * len(cls.PERIODS)

        # Build AᵀA and Aᵀb directly (avoid huge matrix allocation)
        ata = [[0.0] * num_params for _ in range(num_params)]
        atb = [0.0] * num_params

        for stamp, level in zip(fit_times, fit_levels):
            dt = stamp.timestamp() - anchor_ts
            row = [1.0]  # offset term
            for period in cls.PERIODS:
                omega = 2.0 * math.pi / period
                phase = omega * dt
                row.append(math.sin(phase))
                row.append(math.cos(phase))

            for i in range(num_params):
                for j in range(num_params):
                    ata[i][j] += row[i] * row[j]
                atb[i] += row[i] * level

        # Solve via Gaussian elimination with partial pivoting
        solution = _solve_linear_system(ata, atb)
        if solution is None:
            return None

        offset = solution[0]
        coeffs = [(solution[1 + 2 * k], solution[2 + 2 * k]) for k in range(len(cls.PERIODS))]
        return cls(offset, coeffs, anchor_ts)


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Solve Ax = b via Gaussian elimination with partial pivoting."""
    n = len(vector)
    # Build augmented matrix
    aug = [matrix[i][:] + [vector[i]] for i in range(n)]

    for col in range(n):
        # Partial pivoting
        best_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[best_row][col]) < 1e-12:
            return None
        if best_row != col:
            aug[col], aug[best_row] = aug[best_row], aug[col]

        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


# ---------------------------------------------------------------------------
# The Bot
# ---------------------------------------------------------------------------

class AlphaBot2(BaseBot):
    """Predictive-valuation bot with precise settlement formulas for all 8 products."""

    WATCHLIST = (
        "TIDE_SPOT", "TIDE_SWING",
        "WX_SPOT", "WX_SUM",
        "LHR_COUNT", "LHR_INDEX",
        "LON_ETF", "LON_FLY",
    )

    # --- Tuning knobs ---
    REFRESH_SECS = 300.0       # External API refresh interval
    EVAL_SECS = 2.5            # Minimum seconds between evaluations
    MIN_REST_GAP = 1.05        # Rate-limit gap between exchange REST calls
    TAKE_EDGE = 12.0           # Edge required to aggressively take liquidity
    QUOTE_EDGE = 5.0           # Edge required to passively quote
    SMOOTH_ALPHA = 0.40        # EMA blending for theo smoothing
    MAX_SIMULTANEOUS_QUOTES = 3
    ETF_ARB_TAKE_EDGE = 24.0   # Package edge required to trade ETF vs constituent basket
    FLY_ARB_TAKE_EDGE = 20.0   # Kept for standalone relative-value helpers, not treated as true package arb

    # Settlement guard: last 10 minutes before settlement
    SETTLEMENT_GUARD_MINUTES = 10
    SETTLEMENT_SPREAD_MULTIPLIER = 10.0

    def __init__(
        self,
        cmi_url: str,
        username: str,
        password: str,
        *,
        aerodatabox_key: str | None = None,
        base_order_size: int = 5,
        max_position: int = 100,
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

        # Multi-product quoting state
        self.active_quotes: dict[str, QuoteState] = {}

        # Tide model (reused between evaluations)
        self.tide_model: TideModel | None = None

        self.last_refresh_at = 0.0
        self.last_eval_at = 0.0
        self.last_rest_at = 0.0
        self.last_book_poll_at = 0.0
        self.flight_backoff_until = 0.0
        self.flight_backoff_seconds = max(self.REFRESH_SECS * 2.0, 600.0)

        self._lock = threading.Lock()
        self._eval_lock = threading.Lock()

    # ------------------------------------------------------------------
    # SSE callbacks
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        loaded_products = self._paced(lambda: self.get_products())
        self.products = {p.symbol: p for p in loaded_products}
        self.positions = self._paced(lambda: self.get_positions())
        self._refresh_external(force=True)
        self.start()
        print(f"AlphaBot2 started for {self.username}. Watching: {', '.join(self.WATCHLIST)}")
        try:
            while True:
                self._maybe_evaluate(force=True)
                time.sleep(1.0)
        except KeyboardInterrupt:
            self._cancel_all_quotes()
            self.stop()
            print("AlphaBot2 stopped.")

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

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
        # Refresh external data if stale
        if time.monotonic() - self.last_refresh_at > self.REFRESH_SECS:
            self._refresh_external()

        with self._lock:
            books = dict(self.books)
            positions = dict(self.positions)

        if not books:
            books = self._poll_books_if_stale()
            with self._lock:
                positions = dict(self.positions)

        if not books:
            return

        # Build and smooth theos
        raw_theos = self._build_theos(books)
        self.theos = self._smooth_theos(raw_theos)

        # Log theos periodically
        theo_str = "  ".join(f"{s}={v:.0f}" for s, v in self.theos.items())
        print(f"THEOS  {theo_str}")

        best_arb = self._best_structural_arb(books, positions)
        if best_arb:
            print(
                f"ARBCHK {best_arb['label']} "
                f"edge={best_arb['edge']:.1f} "
                f"size={best_arb['size']} "
                f"detail={best_arb['detail']}"
            )
            if best_arb["edge"] >= best_arb["threshold"]:
                if self._execute_structural_arb(best_arb):
                    return

        # Rank opportunities across all 8 products
        signals = []
        for symbol in self.WATCHLIST:
            book = books.get(symbol)
            fair = self.theos.get(symbol)
            if not book or fair is None:
                continue
            sig = self._signal_for(symbol, book, fair, positions.get(symbol, 0))
            if sig:
                signals.append(sig)

        signals.sort(key=lambda s: s["score"], reverse=True)

        # Take the top opportunities (up to MAX_SIMULTANEOUS_QUOTES)
        quoted_products: set[str] = set()
        for sig in signals:
            symbol = sig["product"]
            book = books[symbol]
            fair = self.theos[symbol]
            position = positions.get(symbol, 0)

            if sig["edge"] >= self.TAKE_EDGE:
                self._cancel_quote(symbol)
                self._take_liquidity(sig, book, fair, position)
                continue

            if sig["edge"] >= self.QUOTE_EDGE and len(quoted_products) < self.MAX_SIMULTANEOUS_QUOTES:
                self._quote(sig, book, fair, position)
                quoted_products.add(symbol)

        # Cancel quotes on products no longer in the top-N
        for sym in list(self.active_quotes.keys()):
            if sym not in quoted_products:
                self._cancel_quote(sym)

    # ------------------------------------------------------------------
    # Theo construction
    # ------------------------------------------------------------------

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
        lon_fly = fly_payoff(lon_etf)

        theos: dict[str, float] = {
            "TIDE_SPOT": tide_spot,
            "TIDE_SWING": tide_swing,
            "WX_SPOT": wx_spot,
            "WX_SUM": wx_sum,
            "LHR_COUNT": lhr_count,
            "LHR_INDEX": lhr_index,
            "LON_ETF": lon_etf,
            "LON_FLY": lon_fly,
        }

        # Blend LON_FLY with market-implied value from ETF mid
        etf_mid = self._mid(books.get("LON_ETF"))
        if etf_mid is not None:
            theos["LON_FLY"] = 0.3 * theos["LON_FLY"] + 0.7 * fly_payoff(etf_mid)

        # For products with weaker models, blend slightly with market mid
        for symbol in ("TIDE_SWING", "WX_SUM", "LHR_INDEX"):
            mid = self._mid(books.get(symbol))
            if mid is not None:
                theos[symbol] = 0.80 * theos[symbol] + 0.20 * mid

        return theos

    # --- TIDE_SPOT ---

    def _theo_tide_spot(self, books: dict[str, OrderBook], tide: dict[str, Any]) -> float:
        """Settlement = |mAOD| × 1000."""
        if tide.get("projected_level_m") is not None:
            return max(0.0, abs(float(tide["projected_level_m"])) * 1000.0)
        if tide.get("latest_level_m") is not None:
            return max(0.0, abs(float(tide["latest_level_m"])) * 1000.0)
        return self._fallback("TIDE_SPOT", books)

    # --- TIDE_SWING ---

    def _theo_tide_swing(self, books: dict[str, OrderBook], tide: dict[str, Any]) -> float:
        """Settlement = Σ(strangle payoffs) × 100.

        Strangle payoff per 15-min interval:
            diff = |level(t) - level(t-1)| × 100  (in cm)
            payoff = max(0, 20 - diff) + max(0, diff - 25)
        """
        if tide.get("swing_theo") is not None:
            return max(0.0, float(tide["swing_theo"]))
        if tide.get("projected_swing_raw") is not None:
            return max(0.0, float(tide["projected_swing_raw"]) * 100.0)
        return self._fallback("TIDE_SWING", books)

    # --- WX_SPOT ---

    def _theo_wx_spot(self, books: dict[str, OrderBook], weather: dict[str, Any]) -> float:
        """Settlement = Temp(°F) × Humidity(%) at Sunday 12:00 PM."""
        if weather.get("wx_spot_settle") is not None:
            return max(0.0, float(weather["wx_spot_settle"]))
        return self._fallback("WX_SPOT", books)

    # --- WX_SUM ---

    def _theo_wx_sum(self, books: dict[str, OrderBook], weather: dict[str, Any]) -> float:
        """Settlement = Σ(T×H / 100) over 96 fifteen-minute intervals."""
        if weather.get("wx_sum") is not None:
            return max(0.0, float(weather["wx_sum"]))
        return self._fallback("WX_SUM", books)

    # --- LHR_COUNT ---

    def _theo_lhr_count(
        self,
        books: dict[str, OrderBook],
        flights: dict[str, Any],
        tide_spot: float,
        wx_spot: float,
    ) -> float:
        """Settlement = Total Arrivals + Total Departures."""
        if flights.get("count") is not None:
            api_count = float(flights["count"])
            cancellations = float(flights.get("cancellations", 0))
            return max(0.0, api_count - cancellations)

        product = self.products.get("LHR_COUNT")
        baseline = float(product.startingPrice) if product else 1280.0

        now_london = datetime.now(LONDON_TZ)
        hour = now_london.hour + now_london.minute / 60.0
        if 23.0 <= hour or hour < 5.0:
            baseline -= 20.0
        elif 5.0 <= hour < 8.0:
            baseline += 10.0

        return max(0.0, baseline)

    # --- LHR_INDEX ---

    def _theo_lhr_index(self, books: dict[str, OrderBook], flights: dict[str, Any]) -> float:
        """Settlement = |Σ(100 × (arrivals - departures) / (arrivals + departures))| over 48 30-min buckets."""
        if flights.get("index") is not None:
            return max(0.0, float(flights["index"]))
        return self._fallback("LHR_INDEX", books)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _signal_for(
        self, symbol: str, book: OrderBook, fair: float, position: int,
    ) -> dict[str, Any] | None:
        best_bid = self._best_bid(book)
        best_ask = self._best_ask(book)
        if best_bid is None and best_ask is None:
            return None

        limit = self.max_position
        buy_edge = (fair - best_ask) if best_ask is not None and position < limit else float("-inf")
        sell_edge = (best_bid - fair) if best_bid is not None and position > -limit else float("-inf")
        edge = max(buy_edge, sell_edge, 0.0)
        if edge <= 0.0:
            return None

        # Inventory penalty to discourage concentration
        utilization = abs(position) / max(limit, 1)
        inventory_penalty = max(0.0, utilization - 0.3) * self.QUOTE_EDGE
        score = edge - inventory_penalty

        return {
            "product": symbol,
            "edge": edge,
            "score": score,
            "buy_edge": max(0.0, buy_edge),
            "sell_edge": max(0.0, sell_edge),
        }

    # ------------------------------------------------------------------
    # Execution: aggressive (IOC)
    # ------------------------------------------------------------------

    def _take_liquidity(
        self, signal: dict[str, Any], book: OrderBook, fair: float, position: int,
    ) -> None:
        symbol = signal["product"]
        best_ask_order = self._best_ask_order(book)
        best_bid_order = self._best_bid_order(book)
        limit = self.max_position

        if best_ask_order and signal["buy_edge"] >= self.TAKE_EDGE and position < limit:
            available = best_ask_order.volume - best_ask_order.own_volume
            size = min(self._size_for(position, signal["edge"]), available, limit - position)
            if size > 0:
                self._send_ioc(OrderRequest(symbol, best_ask_order.price, Side.BUY, size))
                print(f"HIT BUY  {size} {symbol} @ {best_ask_order.price:.0f}  theo={fair:.1f}")
                return

        if best_bid_order and signal["sell_edge"] >= self.TAKE_EDGE and position > -limit:
            available = best_bid_order.volume - best_bid_order.own_volume
            size = min(self._size_for(position, signal["edge"]), available, limit + position)
            if size > 0:
                self._send_ioc(OrderRequest(symbol, best_bid_order.price, Side.SELL, size))
                print(f"HIT SELL {size} {symbol} @ {best_bid_order.price:.0f}  theo={fair:.1f}")

    def _best_structural_arb(
        self,
        books: dict[str, OrderBook],
        positions: dict[str, int],
    ) -> dict[str, Any] | None:
        candidates = [
            self._best_lon_etf_overprice_arb(books, positions),
            self._best_lon_etf_underprice_arb(books, positions),
        ]
        viable = [candidate for candidate in candidates if candidate and candidate["edge"] > 0.0 and candidate["size"] > 0]
        if not viable:
            return None
        return max(
            viable,
            key=lambda candidate: (candidate["edge"] / max(candidate["threshold"], 1.0), candidate["edge"]),
        )

    def _execute_structural_arb(self, arb: dict[str, Any]) -> bool:
        handler = arb.get("executor")
        if not callable(handler):
            return False
        return bool(handler(arb))

    def _best_lon_etf_overprice_arb(
        self,
        books: dict[str, OrderBook],
        positions: dict[str, int],
    ) -> dict[str, Any] | None:
        etf_book = books.get("LON_ETF")
        if not etf_book:
            return None

        etf_bid = self._best_bid_order(etf_book)
        if not etf_bid:
            return None
        etf_ask = self._best_ask_order(etf_book)

        leg_symbols = ("TIDE_SPOT", "WX_SPOT", "LHR_COUNT")
        legs: list[dict[str, Any]] = []
        basket_cost = 0.0
        max_size = None

        etf_headroom = self._headroom_for_side("LON_ETF", Side.SELL, positions)
        etf_available = max(0, etf_bid.volume - etf_bid.own_volume)
        max_size = min(etf_headroom, etf_available)
        if max_size <= 0:
            return None

        for symbol in leg_symbols:
            book = books.get(symbol)
            ask_order = self._best_ask_order(book) if book else None
            bid_order = self._best_bid_order(book) if book else None
            if not ask_order:
                return None

            available = max(0, ask_order.volume - ask_order.own_volume)
            headroom = self._headroom_for_side(symbol, Side.BUY, positions)
            leg_cap = min(available, headroom)
            if leg_cap <= 0:
                return None

            max_size = min(max_size, leg_cap)
            basket_cost += ask_order.price
            legs.append(
                {
                    "symbol": symbol,
                    "price": ask_order.price,
                    "side": Side.BUY,
                    "unwind_side": Side.SELL,
                    "unwind_price": bid_order.price if bid_order else ask_order.price,
                }
            )

        edge = etf_bid.price - basket_cost
        if edge <= 0.0 or max_size <= 0:
            return None

        size_cap = self._arb_size_cap(edge, self.ETF_ARB_TAKE_EDGE)

        return {
            "label": "SELL_ETF_BUY_BASKET",
            "kind": "etf_overprice",
            "edge": edge,
            "size": min(max_size, size_cap),
            "etf_price": etf_bid.price,
            "etf_unwind_price": etf_ask.price if etf_ask else etf_bid.price,
            "basket_cost": basket_cost,
            "legs": legs,
            "threshold": self.ETF_ARB_TAKE_EDGE,
            "detail": f"etf_bid={etf_bid.price:.0f},basket_ask={basket_cost:.0f}",
            "executor": self._execute_lon_etf_overprice_arb,
        }

    def _execute_lon_etf_overprice_arb(self, arb: dict[str, Any]) -> bool:
        size = int(arb["size"])
        if size <= 0:
            return False

        involved = ["LON_ETF", "TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
        for symbol in involved:
            self._cancel_quote(symbol)

        etf_resp = self._send_ioc(OrderRequest("LON_ETF", arb["etf_price"], Side.SELL, size))
        etf_filled = etf_resp.filled if etf_resp else 0
        if etf_filled <= 0:
            return False

        print(
            f"ARB SELL ETF {etf_filled} LON_ETF @ {arb['etf_price']:.0f}  "
            f"basket={arb['basket_cost']:.0f} edge={arb['edge']:.1f}"
        )

        hedge_target = etf_filled
        leg_fills: list[tuple[dict[str, Any], int]] = []
        for leg in arb["legs"]:
            if hedge_target <= 0:
                break
            resp = self._send_ioc(OrderRequest(leg["symbol"], leg["price"], leg["side"], hedge_target))
            filled = resp.filled if resp else 0
            leg_fills.append((leg, filled))
            if filled <= 0:
                print(f"ARB HEDGE MISS {leg['symbol']} BUY @ {leg['price']:.0f}")
            elif filled < hedge_target:
                print(
                    f"ARB HEDGE PARTIAL {leg['symbol']} BUY {filled}/{hedge_target} @ {leg['price']:.0f}"
                )
            else:
                print(f"ARB HEDGE BUY {filled} {leg['symbol']} @ {leg['price']:.0f}")

        self._flatten_package_residual(
            anchor_symbol="LON_ETF",
            anchor_side=Side.SELL,
            anchor_filled=etf_filled,
            anchor_unwind_price=float(arb["etf_unwind_price"]),
            leg_fills=leg_fills,
        )
        return True

    def _best_lon_etf_underprice_arb(
        self,
        books: dict[str, OrderBook],
        positions: dict[str, int],
    ) -> dict[str, Any] | None:
        etf_book = books.get("LON_ETF")
        if not etf_book:
            return None

        etf_ask = self._best_ask_order(etf_book)
        if not etf_ask:
            return None
        etf_bid = self._best_bid_order(etf_book)

        leg_symbols = ("TIDE_SPOT", "WX_SPOT", "LHR_COUNT")
        legs: list[dict[str, Any]] = []
        basket_proceeds = 0.0

        etf_headroom = self._headroom_for_side("LON_ETF", Side.BUY, positions)
        etf_available = max(0, etf_ask.volume - etf_ask.own_volume)
        max_size = min(etf_headroom, etf_available)
        if max_size <= 0:
            return None

        for symbol in leg_symbols:
            book = books.get(symbol)
            bid_order = self._best_bid_order(book) if book else None
            ask_order = self._best_ask_order(book) if book else None
            if not bid_order:
                return None

            available = max(0, bid_order.volume - bid_order.own_volume)
            headroom = self._headroom_for_side(symbol, Side.SELL, positions)
            leg_cap = min(available, headroom)
            if leg_cap <= 0:
                return None

            max_size = min(max_size, leg_cap)
            basket_proceeds += bid_order.price
            legs.append(
                {
                    "symbol": symbol,
                    "price": bid_order.price,
                    "side": Side.SELL,
                    "unwind_side": Side.BUY,
                    "unwind_price": ask_order.price if ask_order else bid_order.price,
                }
            )

        edge = basket_proceeds - etf_ask.price
        if edge <= 0.0 or max_size <= 0:
            return None

        size_cap = self._arb_size_cap(edge, self.ETF_ARB_TAKE_EDGE)

        return {
            "label": "BUY_ETF_SELL_BASKET",
            "kind": "etf_underprice",
            "edge": edge,
            "size": min(max_size, size_cap),
            "etf_price": etf_ask.price,
            "etf_unwind_price": etf_bid.price if etf_bid else etf_ask.price,
            "basket_proceeds": basket_proceeds,
            "legs": legs,
            "threshold": self.ETF_ARB_TAKE_EDGE,
            "detail": f"etf_ask={etf_ask.price:.0f},basket_bid={basket_proceeds:.0f}",
            "executor": self._execute_lon_etf_underprice_arb,
        }

    def _execute_lon_etf_underprice_arb(self, arb: dict[str, Any]) -> bool:
        size = int(arb["size"])
        if size <= 0:
            return False

        involved = ["LON_ETF", "TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
        for symbol in involved:
            self._cancel_quote(symbol)

        etf_resp = self._send_ioc(OrderRequest("LON_ETF", arb["etf_price"], Side.BUY, size))
        etf_filled = etf_resp.filled if etf_resp else 0
        if etf_filled <= 0:
            return False

        print(
            f"ARB BUY ETF  {etf_filled} LON_ETF @ {arb['etf_price']:.0f}  "
            f"basket={arb['basket_proceeds']:.0f} edge={arb['edge']:.1f}"
        )

        hedge_target = etf_filled
        leg_fills: list[tuple[dict[str, Any], int]] = []
        for leg in arb["legs"]:
            if hedge_target <= 0:
                break
            resp = self._send_ioc(OrderRequest(leg["symbol"], leg["price"], leg["side"], hedge_target))
            filled = resp.filled if resp else 0
            leg_fills.append((leg, filled))
            if filled <= 0:
                print(f"ARB HEDGE MISS {leg['symbol']} SELL @ {leg['price']:.0f}")
            elif filled < hedge_target:
                print(
                    f"ARB HEDGE PARTIAL {leg['symbol']} SELL {filled}/{hedge_target} @ {leg['price']:.0f}"
                )
            else:
                print(f"ARB HEDGE SELL {filled} {leg['symbol']} @ {leg['price']:.0f}")

        self._flatten_package_residual(
            anchor_symbol="LON_ETF",
            anchor_side=Side.BUY,
            anchor_filled=etf_filled,
            anchor_unwind_price=float(arb["etf_unwind_price"]),
            leg_fills=leg_fills,
        )
        return True

    def _best_lon_fly_arb(
        self,
        books: dict[str, OrderBook],
        positions: dict[str, int],
    ) -> dict[str, Any] | None:
        fly_book = books.get("LON_FLY")
        if not fly_book:
            return None

        structural_fair = fly_payoff(max(0.0, self.theos.get("LON_ETF", 0.0)))
        candidates: list[dict[str, Any]] = []

        fly_ask = self._best_ask_order(fly_book)
        if fly_ask:
            available = max(0, fly_ask.volume - fly_ask.own_volume)
            headroom = self._headroom_for_side("LON_FLY", Side.BUY, positions)
            size = min(available, headroom, self._arb_size_cap(structural_fair - fly_ask.price, self.FLY_ARB_TAKE_EDGE))
            edge = structural_fair - fly_ask.price
            if edge > 0.0 and size > 0:
                candidates.append(
                    {
                        "label": "BUY_FLY",
                        "kind": "fly_underprice",
                        "edge": edge,
                        "size": size,
                        "price": fly_ask.price,
                        "side": Side.BUY,
                        "threshold": self.FLY_ARB_TAKE_EDGE,
                        "detail": f"fly_ask={fly_ask.price:.0f},fair={structural_fair:.0f}",
                        "executor": self._execute_lon_fly_arb,
                    }
                )

        fly_bid = self._best_bid_order(fly_book)
        if fly_bid:
            available = max(0, fly_bid.volume - fly_bid.own_volume)
            headroom = self._headroom_for_side("LON_FLY", Side.SELL, positions)
            size = min(available, headroom, self._arb_size_cap(fly_bid.price - structural_fair, self.FLY_ARB_TAKE_EDGE))
            edge = fly_bid.price - structural_fair
            if edge > 0.0 and size > 0:
                candidates.append(
                    {
                        "label": "SELL_FLY",
                        "kind": "fly_overprice",
                        "edge": edge,
                        "size": size,
                        "price": fly_bid.price,
                        "side": Side.SELL,
                        "threshold": self.FLY_ARB_TAKE_EDGE,
                        "detail": f"fly_bid={fly_bid.price:.0f},fair={structural_fair:.0f}",
                        "executor": self._execute_lon_fly_arb,
                    }
                )

        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate["edge"])

    def _execute_lon_fly_arb(self, arb: dict[str, Any]) -> bool:
        size = int(arb["size"])
        if size <= 0:
            return False

        self._cancel_quote("LON_FLY")
        resp = self._send_ioc(OrderRequest("LON_FLY", arb["price"], arb["side"], size))
        filled = resp.filled if resp else 0
        if filled <= 0:
            return False

        print(
            f"ARB {arb['label']:<8} {filled} LON_FLY @ {arb['price']:.0f}  "
            f"edge={arb['edge']:.1f}"
        )
        return True

    # ------------------------------------------------------------------
    # Execution: passive quoting
    # ------------------------------------------------------------------

    def _quote(
        self, signal: dict[str, Any], book: OrderBook, fair: float, position: int,
    ) -> None:
        symbol = signal["product"]
        product = self.products.get(symbol)
        if not product:
            return

        tick = product.tickSize or 1.0
        limit = self.max_position
        best_bid = self._best_bid(book)
        best_ask = self._best_ask(book)
        size = self._size_for(position, signal["edge"])
        if size <= 0:
            self._cancel_quote(symbol)
            return

        # Width determination
        spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else 8.0
        half_width = max(5.0, spread / 2.0)

        # Settlement guard: widen spreads 10× in final 10 minutes
        if self._in_settlement_guard():
            half_width *= self.SETTLEMENT_SPREAD_MULTIPLIER
            print(f"SETTLEMENT GUARD active for {symbol}: spread widened to {half_width * 2:.0f}")

        # Inventory skew
        skew = clamp(position / max(limit, 1), -1.0, 1.0) * 4.0

        bid_price = math.floor((fair - half_width - skew) / tick) * tick
        ask_price = math.ceil((fair + half_width - skew) / tick) * tick
        if best_bid is not None:
            bid_price = min(bid_price, best_bid + tick)
        if best_ask is not None:
            ask_price = max(ask_price, best_ask - tick)
        if bid_price <= 0 or ask_price <= bid_price:
            return

        target_bid = bid_price if signal["buy_edge"] >= self.QUOTE_EDGE and position < limit else None
        target_ask = ask_price if signal["sell_edge"] >= self.QUOTE_EDGE and position > -limit else None
        if target_bid is None and target_ask is None:
            self._cancel_quote(symbol)
            return

        # ORDER REFRESH GUARD: only cancel/replace if theo moved by > 2×tick
        existing = self.active_quotes.get(symbol)
        if existing:
            bid_unchanged = (existing.bid_price is None and target_bid is None) or (
                existing.bid_price is not None
                and target_bid is not None
                and abs(existing.bid_price - target_bid) <= 2.0 * tick
            )
            ask_unchanged = (existing.ask_price is None and target_ask is None) or (
                existing.ask_price is not None
                and target_ask is not None
                and abs(existing.ask_price - target_ask) <= 2.0 * tick
            )
            if bid_unchanged and ask_unchanged:
                return  # Keep queue priority

        # Cancel old quote for this product and place new ones
        self._cancel_quote(symbol)

        bid_resp = None
        ask_resp = None
        if target_bid is not None:
            bid_resp = self._paced(lambda: self.send_order(OrderRequest(symbol, target_bid, Side.BUY, size)))
        if target_ask is not None:
            ask_resp = self._paced(lambda: self.send_order(OrderRequest(symbol, target_ask, Side.SELL, size)))

        self.active_quotes[symbol] = QuoteState(
            product=symbol,
            bid_id=bid_resp.id if bid_resp else None,
            ask_id=ask_resp.id if ask_resp else None,
            bid_price=target_bid if bid_resp else None,
            ask_price=target_ask if ask_resp else None,
        )
        print(f"QUOTE {symbol:>10}  {target_bid or '-'} / {target_ask or '-'}  theo={fair:.1f}")

    # ------------------------------------------------------------------
    # Quote management
    # ------------------------------------------------------------------

    def _cancel_quote(self, symbol: str) -> None:
        quote = self.active_quotes.pop(symbol, None)
        if not quote:
            return
        if quote.bid_id:
            self._paced(lambda: self.cancel_order(quote.bid_id))
        if quote.ask_id:
            self._paced(lambda: self.cancel_order(quote.ask_id))

    def _cancel_all_quotes(self) -> None:
        for sym in list(self.active_quotes.keys()):
            self._cancel_quote(sym)

    def _send_ioc(self, order: OrderRequest) -> OrderResponse | None:
        resp = self._paced(lambda: self.send_order(order))
        if resp and resp.filled < resp.volume:
            self.cancel_order(resp.id)
            self.last_rest_at = time.monotonic()
        return resp

    # ------------------------------------------------------------------
    # Settlement guard
    # ------------------------------------------------------------------

    def _in_settlement_guard(self) -> bool:
        """Return True if we are within the final SETTLEMENT_GUARD_MINUTES before settlement."""
        settle = self._next_settlement_time()
        now = datetime.now(LONDON_TZ)
        minutes_to_settle = (settle - now).total_seconds() / 60.0
        return 0 < minutes_to_settle <= self.SETTLEMENT_GUARD_MINUTES

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _size_for(self, position: int, edge: float) -> int:
        utilization = abs(position) / max(self.max_position, 1)
        scale = 1.0 - clamp(utilization, 0.0, 0.8)
        edge_boost = 1.0 + 0.4 * clamp(edge / max(self.TAKE_EDGE, 1.0), 0.0, 1.5)
        return max(0, int(round(self.base_order_size * scale * edge_boost)))

    def _arb_size_cap(self, edge: float, take_edge: float) -> int:
        return max(
            1,
            int(
                round(
                    self.base_order_size
                    * (1.4 + 0.25 * clamp(edge / max(take_edge, 1.0), 0.0, 3.0))
                )
            ),
        )

    def _flatten_package_residual(
        self,
        *,
        anchor_symbol: str,
        anchor_side: Side,
        anchor_filled: int,
        anchor_unwind_price: float,
        leg_fills: list[tuple[dict[str, Any], int]],
    ) -> None:
        if anchor_filled <= 0:
            return

        matched = min([anchor_filled] + [filled for _, filled in leg_fills]) if leg_fills else 0
        if matched >= anchor_filled and all(filled == matched for _, filled in leg_fills):
            return

        residual_anchor = anchor_filled - matched
        if residual_anchor > 0:
            unwind_side = Side.BUY if anchor_side == Side.SELL else Side.SELL
            resp = self._send_ioc(OrderRequest(anchor_symbol, anchor_unwind_price, unwind_side, residual_anchor))
            filled = resp.filled if resp else 0
            print(
                f"ARB UNWIND {anchor_symbol} {unwind_side.name} {filled}/{residual_anchor} @ {anchor_unwind_price:.0f}"
            )

        for leg, filled in leg_fills:
            excess = filled - matched
            if excess <= 0:
                continue
            resp = self._send_ioc(OrderRequest(leg["symbol"], leg["unwind_price"], leg["unwind_side"], excess))
            unwind_filled = resp.filled if resp else 0
            print(
                f"ARB UNWIND {leg['symbol']} {leg['unwind_side'].name} "
                f"{unwind_filled}/{excess} @ {leg['unwind_price']:.0f}"
            )

    def _headroom_for_side(self, symbol: str, side: Side, positions: dict[str, int]) -> int:
        position = positions.get(symbol, 0)
        if side == Side.BUY:
            return max(0, self.max_position - position)
        return max(0, self.max_position + position)

    # ------------------------------------------------------------------
    # Theo smoothing
    # ------------------------------------------------------------------

    def _smooth_theos(self, fresh: dict[str, float]) -> dict[str, float]:
        if not self.smoothed_theos:
            self.smoothed_theos = dict(fresh)
            return dict(fresh)

        smoothed: dict[str, float] = {}
        for symbol, value in fresh.items():
            prev = self.smoothed_theos.get(symbol, value)
            smoothed[symbol] = prev + self.SMOOTH_ALPHA * (value - prev)
        self.smoothed_theos = smoothed
        return dict(smoothed)

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------

    def _fallback(self, symbol: str, books: dict[str, OrderBook]) -> float:
        mid = self._mid(books.get(symbol))
        if mid is not None:
            return mid
        product = self.products.get(symbol)
        return float(product.startingPrice) if product else 0.0

    def _mid(self, book: OrderBook | None) -> float | None:
        if not book:
            return None
        bb = self._best_bid(book)
        ba = self._best_ask(book)
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return bb if bb is not None else ba

    def _best_bid(self, book: OrderBook) -> float | None:
        o = self._best_bid_order(book)
        return o.price if o else None

    def _best_ask(self, book: OrderBook) -> float | None:
        o = self._best_ask_order(book)
        return o.price if o else None

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

    def _poll_books_if_stale(self) -> dict[str, OrderBook]:
        now = time.monotonic()
        if now - self.last_book_poll_at < 5.0:
            with self._lock:
                return dict(self.books)

        polled: dict[str, OrderBook] = {}
        for symbol in self.WATCHLIST:
            try:
                book = self._paced(lambda sym=symbol: self.get_orderbook(sym))
            except Exception as exc:
                print(f"Book poll failed for {symbol}: {exc}")
                continue
            polled[symbol] = book

        self.last_book_poll_at = now
        if polled:
            with self._lock:
                self.books.update(polled)
                return dict(self.books)
        with self._lock:
            return dict(self.books)

    # ------------------------------------------------------------------
    # External data refresh
    # ------------------------------------------------------------------

    def _refresh_external(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_refresh_at < self.REFRESH_SECS:
            return
        print("Refreshing external data...")
        self.external_cache["weather"] = self._fetch_weather()
        self.external_cache["thames"] = self._fetch_thames()
        if self.aerodatabox_key:
            self.external_cache["flights"] = self._fetch_flights()
        self.last_refresh_at = now
        print("External data refresh complete.")

    # --- Weather ---

    def _fetch_weather(self) -> dict[str, float]:
        """Fetch 15-min weather data from Open-Meteo with past + forecast for the settlement window."""
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

            times: list[datetime] = []
            for value in raw["time"]:
                stamp = datetime.fromisoformat(value)
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=LONDON_TZ)
                else:
                    stamp = stamp.astimezone(LONDON_TZ)
                times.append(stamp)

            temps_c: list[float] = raw["temperature_2m"]
            humids: list[float] = raw["relative_humidity_2m"]
            settle = self._next_settlement_time()

            # --- WX_SPOT: forecast for exact settlement time ---
            settle_idx = min(range(len(times)), key=lambda i: abs((times[i] - settle).total_seconds()))
            settle_temp_f = temps_c[settle_idx] * 9.0 / 5.0 + 32.0
            settle_humidity = humids[settle_idx]
            wx_spot_settle = settle_temp_f * settle_humidity

            # --- WX_SUM: Σ(T×H/100) over the 24h window (96 intervals) ---
            window_start = settle - timedelta(hours=24)
            wx_sum = 0.0
            interval_count = 0
            for stamp, temp_c, humidity in zip(times, temps_c, humids):
                if window_start <= stamp <= settle:
                    temp_f = temp_c * 9.0 / 5.0 + 32.0
                    wx_sum += (temp_f * humidity) / 100.0
                    interval_count += 1

            return {
                "wx_spot_settle": wx_spot_settle,
                "wx_sum": wx_sum,
                "wx_intervals": interval_count,
            }
        except Exception as exc:
            print(f"Warning: weather fetch failed: {exc}")
            return self.external_cache.get("weather", {})

    # --- Thames tidal data ---

    def _fetch_thames(self) -> dict[str, Any]:
        """Fetch last 48h of Thames tidal readings and build harmonic projection."""
        try:
            resp = requests.get(
                f"https://environment.data.gov.uk/flood-monitoring/id/measures/{THAMES_MEASURE}/readings",
                params={"_sorted": "", "_limit": 193},
                timeout=20,
            )
            resp.raise_for_status()
            items = [i for i in resp.json().get("items", []) if i.get("value") is not None]
            if not items:
                return self.external_cache.get("thames", {})

            items.sort(key=lambda x: x["dateTime"])
            times = [
                datetime.fromisoformat(item["dateTime"].replace("Z", "+00:00")).astimezone(LONDON_TZ)
                for item in items
            ]
            levels = [float(item["value"]) for item in items]
            latest_level = levels[-1]
            settle = self._next_settlement_time()

            # --- Fit multi-harmonic model ---
            model = TideModel.fit(times, levels)
            self.tide_model = model

            projected_level = None
            if model:
                projected_level = model.predict(settle)

            result: dict[str, Any] = {
                "latest_level_m": latest_level,
            }

            if projected_level is not None:
                result["projected_level_m"] = projected_level

            # --- TIDE_SWING: compute actuals + project remaining ---
            # The settlement window is 24h ending at settle (Sat 12PM to Sun 12PM)
            window_start = settle - timedelta(hours=24)

            # Actual swing sum for observed intervals within the window
            actual_swing_sum = 0.0
            last_observed_time = window_start
            for prev_t, curr_t, prev_lev, curr_lev in zip(times, times[1:], levels, levels[1:]):
                if curr_t < window_start or prev_t > settle:
                    continue
                diff_cm = abs(curr_lev - prev_lev) * 100.0
                actual_swing_sum += max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)
                last_observed_time = max(last_observed_time, curr_t)

            # Projected swing for remaining intervals (from model)
            projected_swing_remainder = 0.0
            if model and last_observed_time < settle:
                cursor = last_observed_time
                prev_level = model.predict(cursor)
                while cursor < settle:
                    cursor += timedelta(minutes=15)
                    if cursor > settle:
                        cursor = settle
                    curr_level = model.predict(cursor)
                    diff_cm = abs(curr_level - prev_level) * 100.0
                    projected_swing_remainder += max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)
                    prev_level = curr_level

            total_swing_raw = actual_swing_sum + projected_swing_remainder
            result["projected_swing_raw"] = total_swing_raw
            result["swing_theo"] = total_swing_raw  # Already in cm-based strikes (20/25)

            return result

        except requests.exceptions.Timeout:
            print("Warning: Thames fetch timed out")
            return self.external_cache.get("thames", {})
        except requests.exceptions.RequestException as exc:
            print(f"Warning: Thames fetch failed: {exc}")
            return self.external_cache.get("thames", {})
        except Exception as exc:
            print(f"Warning: Thames fetch error: {exc}")
            return self.external_cache.get("thames", {})

    # --- Flights ---

    def _fetch_flights(self) -> dict[str, Any]:
        """Fetch Heathrow flight schedule for the settlement window (Sat 12PM – Sun 12PM)."""
        now_monotonic = time.monotonic()
        if now_monotonic < self.flight_backoff_until:
            remaining = int(self.flight_backoff_until - now_monotonic)
            print(f"Flight API backoff active ({remaining}s remaining); using cached flight data.")
            return self.external_cache.get("flights", {})

        try:
            settle = self._next_settlement_time()
            window_start = settle - timedelta(hours=24)

            # AeroDataBox allows max 12h windows, so we fetch in two chunks
            all_arrivals: list[dict] = []
            all_departures: list[dict] = []

            for chunk_start, chunk_end in [
                (window_start, window_start + timedelta(hours=12)),
                (window_start + timedelta(hours=12), settle),
            ]:
                start_str = chunk_start.strftime("%Y-%m-%dT%H:%M")
                end_str = chunk_end.strftime("%Y-%m-%dT%H:%M")
                resp = requests.get(
                    f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/LHR/{start_str}/{end_str}",
                    params={"direction": "Both"},
                    headers={
                        "x-rapidapi-host": "aerodatabox.p.rapidapi.com",
                        "x-rapidapi-key": self.aerodatabox_key,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                payload = resp.json()
                all_arrivals.extend(payload.get("arrivals", []))
                all_departures.extend(payload.get("departures", []))

            total_arrivals = len(all_arrivals)
            total_departures = len(all_departures)
            total_count = total_arrivals + total_departures

            # Count cancellations
            cancellations = sum(
                1
                for f in all_arrivals + all_departures
                if f.get("status", "").lower() in ("cancelled", "canceled")
            )

            # --- LHR_INDEX: |Σ(metric)| over 48 × 30-min buckets ---
            # Bucket flights into 30-min intervals
            index_sum = 0.0
            bucket_start = window_start
            for _ in range(48):
                bucket_end = bucket_start + timedelta(minutes=30)
                arr_count = 0
                dep_count = 0
                for flight in all_arrivals:
                    ft = self._parse_flight_time(flight, "arrival")
                    if ft and bucket_start <= ft < bucket_end:
                        arr_count += 1
                for flight in all_departures:
                    ft = self._parse_flight_time(flight, "departure")
                    if ft and bucket_start <= ft < bucket_end:
                        dep_count += 1
                total_bucket = arr_count + dep_count
                if total_bucket > 0:
                    metric = 100.0 * (arr_count - dep_count) / total_bucket
                    index_sum += metric
                bucket_start = bucket_end

            self.flight_backoff_until = 0.0
            self.flight_backoff_seconds = max(self.REFRESH_SECS * 2.0, 600.0)

            return {
                "count": float(total_count),
                "cancellations": float(cancellations),
                "arrivals": total_arrivals,
                "departures": total_departures,
                "index": abs(index_sum),
            }
        except requests.exceptions.HTTPError as exc:
            response = exc.response
            if response is not None and response.status_code == 429:
                retry_after = 0.0
                retry_after_raw = response.headers.get("Retry-After")
                if retry_after_raw:
                    try:
                        retry_after = float(retry_after_raw)
                    except ValueError:
                        retry_after = 0.0
                delay = retry_after if retry_after > 0.0 else min(self.flight_backoff_seconds, 3600.0)
                self.flight_backoff_until = time.monotonic() + delay
                self.flight_backoff_seconds = min(max(delay * 2.0, self.REFRESH_SECS * 2.0), 3600.0)
                print(
                    f"Warning: flight fetch rate-limited (429). "
                    f"Backing off for {int(delay)}s and using cached flight data."
                )
                return self.external_cache.get("flights", {})
            print(f"Warning: flight fetch failed: {exc}")
            return self.external_cache.get("flights", {})
        except Exception as exc:
            print(f"Warning: flight fetch failed: {exc}")
            return self.external_cache.get("flights", {})

    def _parse_flight_time(self, flight: dict, direction: str) -> datetime | None:
        """Extract the best time estimate from an AeroDataBox flight record."""
        try:
            movement = flight.get(direction, flight.get("movement", {}))
            # Prefer actual time, fall back to scheduled
            time_str = (
                movement.get("actualTimeLocal")
                or movement.get("revisedTime", {}).get("local")
                or movement.get("scheduledTimeLocal")
            )
            if not time_str:
                # Try top-level keys
                time_str = (
                    flight.get("actualTimeLocal")
                    or flight.get("scheduledTimeLocal")
                )
            if time_str:
                parsed = datetime.fromisoformat(time_str)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=LONDON_TZ)
                return parsed
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_env_file()

    EXCHANGE_URL = os.getenv(
        "CMI_EXCHANGE_URL", "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com"
    )
    USERNAME = os.getenv("CMI_USERNAME")
    PASSWORD = os.getenv("CMI_PASSWORD")
    AERODATABOX_KEY = os.getenv("AERODATABOX_KEY")

    print(f"connecting to {USERNAME} at {EXCHANGE_URL}")

    if not USERNAME or not PASSWORD:
        raise SystemExit("Set CMI_USERNAME and CMI_PASSWORD in .env before running alphabot2.py.")

    bot = AlphaBot2(
        EXCHANGE_URL,
        USERNAME,
        PASSWORD,
        aerodatabox_key=AERODATABOX_KEY,
        base_order_size=5,
        max_position=100,
    )
    bot.run()
