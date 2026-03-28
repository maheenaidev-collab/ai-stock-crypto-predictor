---

## 📄 data_fetcher.py

```python
"""
Market data fetching module.
Author: Maheen Riaz
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class DataFetcher:
    """Fetches historical market data from Yahoo Finance."""

    POPULAR_ASSETS = {
        'stocks': ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA'],
        'crypto': ['BTC-USD', 'ETH-USD', 'DOGE-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD']
    }

    def __init__(self):
        """Initialize data fetcher."""
        pass

    def fetch(self, symbol, period_days=365):
        """
        Fetch historical data for a given symbol.

        Args:
            symbol (str): Stock ticker or crypto symbol (e.g., 'AAPL', 'BTC-USD')
            period_days (int): Number of days of historical data

        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                raise ValueError(f"No data found for {symbol}")

            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index.name = 'date'

            # Clean data
            df = df.dropna()
            df = df[df['volume'] > 0]

            print(f"✅ Fetched {len(df)} days of data for {symbol}")
            return df

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return None

    def get_current_price(self, symbol):
        """Get the latest price for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if not data.empty:
                return round(data['Close'].iloc[-1], 2)
        except Exception:
            pass
        return None

    def get_asset_info(self, symbol):
        """Get basic asset information."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'name': info.get('shortName', symbol),
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap', 'N/A'),
                'sector': info.get('sector', 'Crypto' if '-USD' in symbol else 'N/A')
            }
        except Exception:
            return {'symbol': symbol, 'name': symbol}
