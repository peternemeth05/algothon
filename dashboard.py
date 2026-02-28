import streamlit as st
import pandas as pd
import time
import requests
import plotly.graph_objects as go
from datetime import datetime
from bot_template import BaseBot, OrderBook

st.set_page_config(layout="wide", page_title="IMCity Alpha Dashboard")
st.title("📈 IMCity Alpha Dashboard")

# --- 1. THE SSE BOT (Real-time Market Data) ---
class DashboardBot(BaseBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_books = {}
        self.positions = {}

    def on_orderbook(self, orderbook: OrderBook):
        self.live_books[orderbook.product] = orderbook

    def on_trades(self, trade):
        signed_vol = trade.volume if trade.buyer == self.username else -trade.volume
        self.positions[trade.product] = self.positions.get(trade.product, 0) + signed_vol

@st.cache_resource
def get_bot():
    bot = DashboardBot(
        "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/", 
        "out of our depth", 
        "123456789"
    )
    bot.positions = bot.get_positions()
    bot.start() 
    return bot

bot = get_bot()

# --- 2. SAFE EXTERNAL DATA FETCHERS (Cached for 5 mins) ---
@st.cache_data(ttl=300)
def fetch_weather_history():
    """Fetches Open-Meteo data and calculates true WX_SPOT value"""
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 51.5074, "longitude": -0.1278,
                "minutely_15": "temperature_2m,relative_humidity_2m",
                "past_minutely_15": 96, "forecast_minutely_15": 10,
                "timezone": "Europe/London",
            },
            timeout=10
        )
        resp.raise_for_status()
        m = resp.json()["minutely_15"]
        df = pd.DataFrame({
            "time": pd.to_datetime(m["time"]),
            "temp_c": m["temperature_2m"],
            "humidity": m["relative_humidity_2m"]
        })
        # Calculate WX_SPOT: (Temp_F * Humidity)
        df["temp_f"] = df["temp_c"] * 9/5 + 32
        df["true_wx_spot"] = df["temp_f"] * df["humidity"]
        return df.dropna()
    except Exception as e:
        st.error(f"Weather API Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_thames_history():
    """Fetches EA Flood data with built-in retries and optimized payload."""
    url = "https://environment.data.gov.uk/flood-monitoring/id/measures/0006-level-tidal_level-i-15_min-mAOD/readings"
    
    # REDUCED LIMIT: 100 intervals = 25 hours. Exactly enough for the 24h strangle rule.
    params = {"_sorted": "", "_limit": 100} 
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # DEFENSIVE LOOP: Try up to 3 times before giving up
    for attempt in range(3): 
        try:
            # We enforce a strict 10-second timeout so we don't hang forever
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            
            items = resp.json().get("items", [])
            df = pd.DataFrame(items)[["dateTime", "value"]]
            df["time"] = pd.to_datetime(df["dateTime"])
            df["level"] = df["value"].astype(float)
            
            # Calculate TIDE_SPOT: abs(level * 1000)
            df["true_tide_spot"] = df["level"].abs() * 1000
            
            return df.sort_values("time").dropna()
            
        except requests.exceptions.HTTPError as e:
            # If it's a server overload error (502, 503, 504), wait and retry
            if resp.status_code in [502, 503, 504]:
                time.sleep(2)  # Wait 2 seconds before hammering them again
                continue
            else:
                st.error(f"Thames HTTP Error: {e}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Thames Request Failed: {e}")
            return pd.DataFrame()
            
    st.warning("⚠️ Thames API is severely overloaded right now (504).")
    return pd.DataFrame()

# --- 3. RATE-LIMITED MARKET HISTORY FETCHER ---
if 'last_trade_fetch' not in st.session_state:
    st.session_state.last_trade_fetch = 0

# Only fetch market trades every 15 seconds so we don't break the 1 req/sec limit
if time.time() - st.session_state.last_trade_fetch > 15:
    bot.get_market_trades()
    st.session_state.last_trade_fetch = time.time()

# Convert bot's incremental trade history into a DataFrame
if bot.trades:
    market_df = pd.DataFrame([t.to_dict() for t in bot.trades])
    market_df['time'] = pd.to_datetime(market_df['timestamp'])
else:
    market_df = pd.DataFrame()


# --- 4. THE DASHBOARD UI ---

# Top Row: Positions
st.subheader("🛡️ Current Positions")
if bot.positions:
    pos_df = pd.DataFrame(list(bot.positions.items()), columns=["Product", "Net Position"])
    st.dataframe(pos_df.set_index("Product").T, use_container_width=True)

st.divider()

# Middle Row: The Alpha Charts!
st.subheader("🔬 Alpha Scanners (Market Price vs True Value)")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### WX_SPOT: Market vs Weather API")
    wx_true = fetch_weather_history()
    fig_wx = go.Figure()
    
    # Plot True Value
    if not wx_true.empty:
        fig_wx.add_trace(go.Scatter(x=wx_true['time'], y=wx_true['true_wx_spot'], 
                                    mode='lines', name='True Value (API)', line=dict(color='cyan', width=3)))
    
    # Plot Market Price
    if not market_df.empty and 'WX_SPOT' in market_df['product'].values:
        wx_market = market_df[market_df['product'] == 'WX_SPOT']
        fig_wx.add_trace(go.Scatter(x=wx_market['time'], y=wx_market['price'], 
                                    mode='markers+lines', name='Market Price', line=dict(color='orange')))
    
    fig_wx.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark")
    st.plotly_chart(fig_wx, use_container_width=True)

with col2:
    st.markdown("### TIDE_SPOT: Market vs Thames API")
    tide_true = fetch_thames_history()
    fig_tide = go.Figure()
    
    # Plot True Value
    if not tide_true.empty:
        fig_tide.add_trace(go.Scatter(x=tide_true['time'], y=tide_true['true_tide_spot'], 
                                      mode='lines', name='True Value (API)', line=dict(color='cyan', width=3)))
    
    # Plot Market Price
    if not market_df.empty and 'TIDE_SPOT' in market_df['product'].values:
        tide_market = market_df[market_df['product'] == 'TIDE_SPOT']
        fig_tide.add_trace(go.Scatter(x=tide_market['time'], y=tide_market['price'], 
                                      mode='markers+lines', name='Market Price', line=dict(color='orange')))
    
    fig_tide.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark")
    st.plotly_chart(fig_tide, use_container_width=True)

st.divider()

# Bottom Row: Order Books
st.subheader("📊 Live Order Books")
if bot.live_books:
    cols = st.columns(4)
    sorted_products = sorted(bot.live_books.keys())
    for i, symbol in enumerate(sorted_products):
        with cols[i % 4]:
            st.markdown(f"**{symbol}**")
            ob = bot.live_books[symbol]
            asks = [{"Price": o.price, "Vol": o.volume} for o in ob.sell_orders[:3]]
            bids = [{"Price": o.price, "Vol": o.volume} for o in ob.buy_orders[:3]]
            
            if asks: st.dataframe(pd.DataFrame(asks).set_index("Price").style.applymap(lambda _: 'color: red'), use_container_width=True)
            if bids: st.dataframe(pd.DataFrame(bids).set_index("Price").style.applymap(lambda _: 'color: green'), use_container_width=True)

# Auto-refresh loop
time.sleep(1.5)
st.rerun()