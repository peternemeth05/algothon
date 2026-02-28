import streamlit as st
import pandas as pd
import time
from bot_template import BaseBot, OrderBook

# 1. Setup Page
st.set_page_config(layout="wide", page_title="IMCity Dashboard")
st.title("📈 IMCity Live Market Dashboard")

# 2. Create a Custom Bot just for listening to the SSE Stream
class DashboardBot(BaseBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_books = {}  # Store the latest orderbook for each product
        self.positions = {}

    def on_orderbook(self, orderbook: OrderBook):
        self.live_books[orderbook.product] = orderbook

    def on_trades(self, trade):
        # Update position locally when a trade happens
        signed_vol = trade.volume if trade.buyer == self.username else -trade.volume
        self.positions[trade.product] = self.positions.get(trade.product, 0) + signed_vol

# 3. Cache the bot so it only connects ONCE when Streamlit starts
@st.cache_resource
def get_bot():
    bot = DashboardBot(
        "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/", 
        "out of our depth", 
        "123456789"
    )
    # Fetch initial positions safely once
    bot.positions = bot.get_positions()
    bot.start()  # This starts the free SSE stream thread!
    return bot

bot = get_bot()

# 4. Display Global Stats (Positions)
st.subheader("🛡️ Current Positions (Max ±100)")
if bot.positions:
    pos_df = pd.DataFrame(list(bot.positions.items()), columns=["Product", "Net Position"])
    st.dataframe(pos_df.set_index("Product").T, use_container_width=True)
else:
    st.info("No positions yet.")

st.divider()

# 5. Display Order Books in a Grid
st.subheader("📊 Live Order Books")
if not bot.live_books:
    st.warning("Waiting for SSE market data to arrive...")
else:
    # Create 4 columns for a clean grid layout
    cols = st.columns(4)
    
    # Sort the products so they always appear in the same order
    sorted_products = sorted(bot.live_books.keys())
    
    for i, symbol in enumerate(sorted_products):
        with cols[i % 4]:
            st.markdown(f"### {symbol}")
            ob = bot.live_books[symbol]
            
            # Extract top 5 Asks (Sellers)
            asks = [{"Price": o.price, "Vol": o.volume, "MyVol": o.own_volume} for o in ob.sell_orders[:5]]
            # Extract top 5 Bids (Buyers)
            bids = [{"Price": o.price, "Vol": o.volume, "MyVol": o.own_volume} for o in ob.buy_orders[:5]]
            
            # Format nicely
            st.caption("🔴 ASKS (Sellers)")
            if asks:
                st.dataframe(pd.DataFrame(asks).set_index("Price"), use_container_width=True)
            else:
                st.write("No Asks")
                
            st.caption("🟢 BIDS (Buyers)")
            if bids:
                st.dataframe(pd.DataFrame(bids).set_index("Price"), use_container_width=True)
            else:
                st.write("No Bids")

# 6. Auto-Refresh the Streamlit UI every 1 second
# (This just refreshes the UI reading from local memory, it does NOT hit the exchange API)
time.sleep(1)
st.rerun()