import os

import dotenv
import matplotlib.pyplot as plt
import pandas as pd

from alphabot import AlphaPulseBot

dotenv.load_dotenv()

# Initialize the bot (using environment variables from .env)
bot = AlphaPulseBot(
    os.getenv("CMI_EXCHANGE_URL"),
    os.getenv("CMI_USERNAME"),
    os.getenv("CMI_PASSWORD"),
)

# Fetch all historical trades from the exchange
all_trades = bot.get_market_trades()

# Convert the list of Trade objects into a Pandas DataFrame
df = pd.DataFrame([t.to_dict() for t in all_trades])

# Filter for a specific product, like the ETF
etf_trades = df[df['product'] == 'LON_ETF'].copy()

# Convert timestamps and plot!
etf_trades['timestamp'] = pd.to_datetime(etf_trades['timestamp'])
etf_trades.set_index('timestamp', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(etf_trades.index, etf_trades['price'], marker='.', linestyle='-')
plt.title('LON_ETF Historical Market Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.show()