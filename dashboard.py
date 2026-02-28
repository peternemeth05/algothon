from __future__ import annotations

import math
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bot_template import BaseBot

try:
    from alphabot import LONDON_LAT, LONDON_LON, THAMES_MEASURE, TIDE_CYCLE_SECS, load_env_file
except ModuleNotFoundError:
    LONDON_LAT = 51.5074
    LONDON_LON = -0.1278
    THAMES_MEASURE = "0006-level-tidal_level-i-15_min-mAOD"
    TIDE_CYCLE_SECS = 12 * 3600 + 25 * 60

    def load_env_file(path: str = ".env") -> None:
        return None


LONDON_TZ = ZoneInfo("Europe/London")
load_env_file()
EXCHANGE_URL = os.getenv("CMI_EXCHANGE_URL", "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com")
USERNAME = os.getenv("CMI_USERNAME")
PASSWORD = os.getenv("CMI_PASSWORD")


def next_settlement_time(now: datetime | None = None) -> datetime:
    now = now or datetime.now(LONDON_TZ)
    settle = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if now >= settle:
        settle += timedelta(days=1)
    return settle


def normalize_london_times(values) -> pd.Series | pd.DatetimeIndex:
    times = pd.to_datetime(values)
    tz = times.dt.tz if hasattr(times, "dt") else times.tz
    if tz is None:
        return times.dt.tz_localize("Europe/London") if hasattr(times, "dt") else times.tz_localize("Europe/London")
    return times.dt.tz_convert("Europe/London") if hasattr(times, "dt") else times.tz_convert("Europe/London")


def fit_tide_cycle(times: list[datetime], levels: list[float]) -> dict[str, float] | None:
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


def predict_tide_level(model: dict[str, float], target: datetime) -> float:
    anchor = datetime.fromtimestamp(model["anchor_ts"], tz=target.tzinfo)
    phase = model["omega"] * (target - anchor).total_seconds()
    return (
        model["offset"]
        + model["sin_coeff"] * math.sin(phase)
        + model["cos_coeff"] * math.cos(phase)
    )


def project_tide_swing(model: dict[str, float], start: datetime, end: datetime) -> float:
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
    previous = predict_tide_level(model, points[0])
    for stamp in points[1:]:
        current = predict_tide_level(model, stamp)
        diff_cm = abs(current - previous) * 100.0
        swing_sum += max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)
        previous = current
    return swing_sum


def future_projection_points(start: datetime, end: datetime, step_minutes: int = 15) -> list[datetime]:
    if end <= start:
        return [start]

    points = [start]
    stamp = start
    while True:
        stamp += timedelta(minutes=step_minutes)
        if stamp >= end:
            break
        points.append(stamp)
    if points[-1] != end:
        points.append(end)
    return points


class DashboardBot(BaseBot):
    def on_orderbook(self, orderbook) -> None:
        return None

    def on_trades(self, trade) -> None:
        return None

    def refresh_trades(self) -> None:
        self.get_market_trades()

    def snapshot_price_history(self, product: str, limit: int = 1200) -> pd.DataFrame:
        rows = [
            {"time": pd.Timestamp(trade.timestamp), "mid": trade.price}
            for trade in self.trades
            if trade.product == product
        ]
        if not rows:
            return pd.DataFrame(columns=["time", "mid"])
        market_df = pd.DataFrame(rows).tail(limit)
        market_df["time"] = normalize_london_times(market_df["time"])
        return market_df


@st.cache_resource(show_spinner=False)
def get_market_bot() -> DashboardBot | None:
    if not USERNAME or not PASSWORD:
        return None
    return DashboardBot(EXCHANGE_URL, USERNAME, PASSWORD)


def fetch_weather_history() -> tuple[pd.DataFrame, dict[str, float | datetime]]:
    settle = next_settlement_time()
    window_start = settle - timedelta(hours=24)

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

    weather_df = pd.DataFrame(
        {
            "time": normalize_london_times(raw["time"]),
            "temp_c": raw["temperature_2m"],
            "humidity": raw["relative_humidity_2m"],
        }
    )
    weather_df["temp_f"] = weather_df["temp_c"] * 9.0 / 5.0 + 32.0
    weather_df["wx_stream"] = weather_df["temp_f"] * weather_df["humidity"]
    weather_df["in_window"] = weather_df["time"].between(window_start, settle)
    weather_df["wx_sum_contrib"] = weather_df["wx_stream"].where(weather_df["in_window"], 0.0) / 100.0
    weather_df["wx_sum_running"] = weather_df["wx_sum_contrib"].cumsum()

    settle_idx = (weather_df["time"] - settle).abs().idxmin()
    metrics = {
        "wx_spot_now": float(weather_df.iloc[len(weather_df) // 2]["wx_stream"]),
        "wx_spot_settle": float(weather_df.loc[settle_idx, "wx_stream"]),
        "wx_sum": float(weather_df["wx_sum_contrib"].sum()),
        "window_start": window_start,
        "settle": settle,
    }
    return weather_df, metrics


def fetch_tide_history() -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, float | datetime | None]]:
    settle = next_settlement_time()
    window_start = settle - timedelta(hours=24)

    resp = requests.get(
        f"https://environment.data.gov.uk/flood-monitoring/id/measures/{THAMES_MEASURE}/readings",
        params={"_sorted": "", "_limit": 193},
        timeout=20,
    )
    resp.raise_for_status()
    items = [item for item in resp.json().get("items", []) if item.get("value") is not None]
    items.sort(key=lambda item: item["dateTime"])

    tide_df = pd.DataFrame(
        {
            "time": [
                datetime.fromisoformat(item["dateTime"].replace("Z", "+00:00")).astimezone(LONDON_TZ)
                for item in items
            ],
            "level_m": [float(item["value"]) for item in items],
        }
    )
    tide_df["abs_level_mm"] = tide_df["level_m"].abs() * 1000.0
    tide_df["diff_cm"] = tide_df["level_m"].diff().abs() * 100.0
    tide_df["swing_payoff"] = tide_df["diff_cm"].apply(
        lambda diff_cm: 0.0 if pd.isna(diff_cm) else max(0.0, 20.0 - diff_cm) + max(0.0, diff_cm - 25.0)
    )
    tide_df["in_window"] = tide_df["time"].between(window_start, settle)
    tide_df["swing_payoff_window"] = tide_df["swing_payoff"].where(tide_df["in_window"], 0.0)
    tide_df["swing_running"] = tide_df["swing_payoff_window"].cumsum()

    settle_proxy = settle - timedelta(hours=24)
    settle_idx = (tide_df["time"] - settle_proxy).abs().idxmin()
    latest_level = float(tide_df.iloc[-1]["level_m"])
    settle_level = 0.7 * float(tide_df.loc[settle_idx, "level_m"]) + 0.3 * latest_level

    tide_model = fit_tide_cycle(tide_df["time"].tolist(), tide_df["level_m"].tolist())
    projection_df = None
    projected_settle_level = None
    projected_swing_sum = None
    if tide_model is not None:
        projected_settle_level = 0.75 * predict_tide_level(tide_model, settle) + 0.25 * settle_level
        projected_swing_sum = project_tide_swing(tide_model, window_start, settle)
        projection_points = future_projection_points(tide_df["time"].iloc[-1], settle)
        projection_df = pd.DataFrame({"time": projection_points})
        projection_df["level_m"] = projection_df["time"].apply(
            lambda stamp: predict_tide_level(tide_model, stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else stamp)
        )

    metrics = {
        "latest_level": latest_level,
        "tide_spot_raw": abs(settle_level) * 1000.0,
        "tide_spot_projected": None if projected_settle_level is None else abs(projected_settle_level) * 1000.0,
        "observed_swing_sum": float(tide_df["swing_payoff_window"].sum()),
        "projected_swing_sum": projected_swing_sum,
        "window_start": window_start,
        "settle": settle,
    }
    return tide_df, projection_df, metrics


def fetch_flight_snapshot() -> tuple[pd.DataFrame | None, dict[str, float | datetime] | None]:
    api_key = os.getenv("AERODATABOX_KEY")
    if not api_key:
        return None, None

    now = datetime.now(LONDON_TZ).replace(second=0, microsecond=0)
    start = now - timedelta(hours=12)
    resp = requests.get(
        f"https://aerodatabox.p.rapidapi.com/flights/airports/iata/LHR/{start:%Y-%m-%dT%H:%M}/{now:%Y-%m-%dT%H:%M}",
        params={"direction": "Both"},
        headers={
            "x-rapidapi-host": "aerodatabox.p.rapidapi.com",
            "x-rapidapi-key": api_key,
        },
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    rows: list[dict[str, object]] = []
    for side, flights in (("arrival", payload.get("arrivals", [])), ("departure", payload.get("departures", []))):
        for flight in flights:
            schedule = (flight.get("movement", {}) or {}).get("scheduledTime", {}) or {}
            local = schedule.get("local")
            if local:
                rows.append({"side": side, "time": pd.Timestamp(local)})

    flights_df = pd.DataFrame(rows)
    if not flights_df.empty:
        flights_df["time"] = normalize_london_times(flights_df["time"])
        flights_df["hour"] = flights_df["time"].dt.floor("1h")

    metrics = {
        "count": float(len(payload.get("arrivals", [])) + len(payload.get("departures", []))),
        "start": start,
        "end": now,
    }
    return flights_df, metrics


def build_weather_figure(
    weather_df: pd.DataFrame,
    metrics: dict[str, float | datetime],
    now: datetime,
    market_df: pd.DataFrame | None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weather_df["time"], y=weather_df["temp_f"], mode="lines", name="Temp (F)", line={"color": "#c2410c"}))
    fig.add_trace(go.Scatter(x=weather_df["time"], y=weather_df["humidity"], mode="lines", name="Humidity (%)", line={"color": "#0369a1"}))
    fig.add_trace(go.Scatter(x=weather_df["time"], y=weather_df["wx_stream"], mode="lines", name="WX stream", line={"color": "#15803d", "width": 3}))
    fig.add_trace(go.Scatter(x=weather_df["time"], y=weather_df["wx_sum_running"], mode="lines", name="Running WX_SUM", line={"color": "#7c3aed", "width": 3}))
    if market_df is not None and not market_df.empty:
        fig.add_trace(go.Scatter(x=market_df["time"], y=market_df["mid"], mode="lines", name="WX_SPOT trade price", line={"color": "#f59e0b", "width": 3}))
    fig.add_vrect(x0=metrics["window_start"], x1=metrics["settle"], fillcolor="lightgray", opacity=0.15, line_width=0)
    fig.add_vline(x=metrics["settle"], line_dash="dash", line_color="black")
    fig.add_vline(x=now, line_dash="dot", line_color="#2563eb")
    fig.update_layout(
        height=420,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        title=(
            f"Weather stream | WX_SPOT now={metrics['wx_spot_now']:.1f} | "
            f"settle proxy={metrics['wx_spot_settle']:.1f} | WX_SUM={metrics['wx_sum']:.1f} | cursor={now:%H:%M:%S}"
        ),
        legend={"orientation": "h"},
    )
    return fig


def build_tide_figure(
    tide_df: pd.DataFrame,
    projection_df: pd.DataFrame | None,
    metrics: dict[str, float | datetime | None],
    now: datetime,
    market_df: pd.DataFrame | None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tide_df["time"], y=tide_df["level_m"], mode="lines", name="Observed tide (m)", line={"color": "#0f172a", "width": 3}))
    if projection_df is not None and not projection_df.empty:
        fig.add_trace(go.Scatter(x=projection_df["time"], y=projection_df["level_m"], mode="lines", name="Projected tide", line={"color": "#dc2626", "dash": "dash"}))
    if market_df is not None and not market_df.empty:
        fig.add_trace(go.Scatter(x=market_df["time"], y=market_df["mid"], mode="lines", name="TIDE_SPOT trade price", line={"color": "#f59e0b", "width": 3}))
    fig.add_trace(go.Bar(x=tide_df["time"], y=tide_df["swing_payoff"], name="Swing payoff", marker={"color": "#16a34a", "opacity": 0.28}, yaxis="y2"))
    fig.add_trace(go.Scatter(x=tide_df["time"], y=tide_df["swing_running"], mode="lines", name="Running TIDE_SWING", line={"color": "#7c3aed", "width": 3}, yaxis="y2"))
    fig.add_vrect(x0=metrics["window_start"], x1=metrics["settle"], fillcolor="lightgray", opacity=0.15, line_width=0)
    fig.add_vline(x=metrics["settle"], line_dash="dash", line_color="black")
    fig.add_vline(x=now, line_dash="dot", line_color="#2563eb")
    projected_text = "n/a" if metrics["tide_spot_projected"] is None else f"{metrics['tide_spot_projected']:.1f}"
    fig.update_layout(
        height=420,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        title=(
            f"Tide stream | latest={metrics['latest_level']:.3f}m | raw spot={metrics['tide_spot_raw']:.1f} | "
            f"projected spot={projected_text} | observed swing={metrics['observed_swing_sum']:.1f} | cursor={now:%H:%M:%S}"
        ),
        yaxis={"title": "mAOD"},
        yaxis2={"title": "Swing", "overlaying": "y", "side": "right"},
        legend={"orientation": "h"},
        barmode="overlay",
    )
    return fig


def build_flight_figure(
    flights_df: pd.DataFrame | None,
    metrics: dict[str, float | datetime] | None,
    now: datetime,
    market_df: pd.DataFrame | None,
) -> go.Figure | None:
    if metrics is None and (market_df is None or market_df.empty):
        return None

    fig = go.Figure()
    if flights_df is not None and not flights_df.empty:
        hourly = flights_df.groupby(["hour", "side"]).size().unstack(fill_value=0)
        for column, color in (("arrival", "#2563eb"), ("departure", "#dc2626")):
            if column in hourly.columns:
                fig.add_trace(go.Scatter(x=hourly.index, y=hourly[column], mode="lines+markers", name=column.title(), line={"color": color, "width": 3}))
    if market_df is not None and not market_df.empty:
        fig.add_trace(go.Scatter(x=market_df["time"], y=market_df["mid"], mode="lines", name="LHR_COUNT trade price", line={"color": "#f59e0b", "width": 3}))

    title = f"Heathrow / LHR_COUNT | cursor={now:%H:%M:%S}"
    if metrics is not None:
        title = (
            f"Heathrow / LHR_COUNT | snapshot count={metrics['count']:.0f} | "
            f"window={metrics['start']:%H:%M} to {metrics['end']:%H:%M} | cursor={now:%H:%M:%S}"
        )

    fig.update_layout(
        height=320,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        title=title,
        legend={"orientation": "h"},
    )
    fig.add_vline(x=now, line_dash="dot", line_color="#2563eb")
    return fig


def build_lhr_count_figure(
    market_df: pd.DataFrame | None,
    now: datetime,
    snapshot_metrics: dict[str, float | datetime] | None,
) -> go.Figure | None:
    if market_df is None or market_df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=market_df["time"],
            y=market_df["mid"],
            mode="lines",
            name="LHR_COUNT trade price",
            line={"color": "#f59e0b", "width": 3},
        )
    )
    if snapshot_metrics is not None:
        fig.add_hline(
            y=snapshot_metrics["count"],
            line_dash="dash",
            line_color="#2563eb",
            annotation_text=f"API snapshot {snapshot_metrics['count']:.0f}",
            annotation_position="top left",
        )
    fig.add_vline(x=now, line_dash="dot", line_color="#2563eb")
    fig.update_layout(
        height=360,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        title=f"LHR_COUNT live trade stream | cursor={now:%H:%M:%S}",
        yaxis={"title": "LHR_COUNT price"},
        legend={"orientation": "h"},
    )
    return fig


st.set_page_config(layout="wide", page_title="AlphaBot Live Streams")
st.title("AlphaBot Live Data Streams")

refresh_secs = st.sidebar.slider("Refresh interval (seconds)", min_value=1, max_value=120, value=5, step=1)
st.sidebar.caption("This app polls the same external feeds used by alphabot.py and reruns itself on a timer.")
st.sidebar.info("Weather and Thames source data typically update on roughly 15-minute intervals. The orange overlays now come from incremental market trades, so they move when new trades print without relying on SSE.")

now = datetime.now(LONDON_TZ)
settle = next_settlement_time(now)
st.caption(f"Last render: {now:%Y-%m-%d %H:%M:%S %Z} | Next settlement: {settle:%Y-%m-%d %H:%M %Z}")

bot = get_market_bot()
market_error = None
if bot is not None:
    try:
        bot.refresh_trades()
    except Exception as exc:
        market_error = exc
wx_market_df = bot.snapshot_price_history("WX_SPOT") if bot is not None and market_error is None else pd.DataFrame(columns=["time", "mid"])
tide_market_df = bot.snapshot_price_history("TIDE_SPOT") if bot is not None and market_error is None else pd.DataFrame(columns=["time", "mid"])
lhr_market_df = bot.snapshot_price_history("LHR_COUNT") if bot is not None and market_error is None else pd.DataFrame(columns=["time", "mid"])

metric_cols = st.columns(3)
metric_cols[0].metric("Current time", f"{now:%H:%M:%S}")
metric_cols[1].metric("Seconds to next refresh", f"{refresh_secs}")
metric_cols[2].metric("Minutes to settlement", f"{max(0.0, (settle - now).total_seconds() / 60.0):.1f}")
if bot is None:
    st.warning("Set CMI_USERNAME and CMI_PASSWORD in .env to plot live exchange trade prices.")
elif market_error is not None:
    st.error(f"Market trade polling failed: {market_error}")
else:
    sse_cols = st.columns(3)
    sse_cols[0].metric("WX_SPOT trades captured", str(len(wx_market_df)))
    sse_cols[1].metric("TIDE_SPOT trades captured", str(len(tide_market_df)))
    sse_cols[2].metric("LHR_COUNT trades captured", str(len(lhr_market_df)))

weather_error = None
tide_error = None
flight_error = None

try:
    weather_df, weather_metrics = fetch_weather_history()
except Exception as exc:
    weather_df, weather_metrics = None, None
    weather_error = exc

try:
    tide_df, projection_df, tide_metrics = fetch_tide_history()
except Exception as exc:
    tide_df, projection_df, tide_metrics = None, None, None
    tide_error = exc

try:
    flights_df, flight_metrics = fetch_flight_snapshot()
except Exception as exc:
    flights_df, flight_metrics = None, None
    flight_error = exc

col1, col2 = st.columns(2)

with col1:
    if weather_error:
        st.error(f"Weather refresh failed: {weather_error}")
    elif weather_df is not None:
        st.plotly_chart(build_weather_figure(weather_df, weather_metrics, now, wx_market_df), use_container_width=True)

with col2:
    if tide_error:
        st.error(f"Tide refresh failed: {tide_error}")
    elif tide_df is not None:
        st.plotly_chart(build_tide_figure(tide_df, projection_df, tide_metrics, now, tide_market_df), use_container_width=True)

lhr_fig = build_lhr_count_figure(lhr_market_df, now, flight_metrics)
if lhr_fig is not None:
    st.plotly_chart(lhr_fig, use_container_width=True)
elif bot is not None:
    st.info("Waiting for live LHR_COUNT trades from the exchange.")

if flight_error:
    st.error(f"Heathrow refresh failed: {flight_error}")
else:
    fig = build_flight_figure(flights_df, flight_metrics, now, lhr_market_df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    elif not os.getenv("AERODATABOX_KEY"):
        st.info("Set AERODATABOX_KEY in .env if you want the Heathrow API snapshot as context for the LHR_COUNT trade stream.")

time.sleep(refresh_secs)
st.rerun()
