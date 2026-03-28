"""
Technical indicators calculation module.
Author: Maheen Riaz
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Calculate technical indicators for financial data."""

    @staticmethod
    def add_all(df):
        """Add all technical indicators to dataframe."""
        df = df.copy()
        df = TechnicalIndicators.add_sma(df)
        df = TechnicalIndicators.add_ema(df)
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger(df)
        df = TechnicalIndicators.add_volume_sma(df)
        df = TechnicalIndicators.add_returns(df)
        df = TechnicalIndicators.add_atr(df)
        df = df.dropna()
        return df

    @staticmethod
    def add_sma(df, periods=[20, 50]):
        """Add Simple Moving Averages."""
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df

    @staticmethod
    def add_ema(df, periods=[12, 26]):
        """Add Exponential Moving Averages."""
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df, period=14):
        """Add Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        """Add MACD indicator."""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df

    @staticmethod
    def add_bollinger(df, period=20, std_dev=2):
        """Add Bollinger Bands."""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df['bb_upper'] = sma + (std * std_dev)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (std * std_dev)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        return df

    @staticmethod
    def add_volume_sma(df, period=20):
        """Add Volume Simple Moving Average."""
        df['volume_sma'] = df['volume'].rolling(window=period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        return df

    @staticmethod
    def add_returns(df):
        """Add daily and cumulative returns."""
        df['daily_return'] = df['close'].pct_change()
        df['return_volatility'] = df['daily_return'].rolling(window=20).std()
        return df

    @staticmethod
    def add_atr(df, period=14):
        """Add Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=period).mean()
        return df

    @staticmethod
    def get_signals(df):
        """Generate trading signals from indicators."""
        latest = df.iloc[-1]
        signals = {}

        # RSI signal
        rsi = latest['rsi']
        if rsi > 70:
            signals['rsi'] = {'value': round(rsi, 1), 'signal': 'overbought', 'emoji': '🔴'}
        elif rsi < 30:
            signals['rsi'] = {'value': round(rsi, 1), 'signal': 'oversold', 'emoji': '🟢'}
        else:
            signals['rsi'] = {'value': round(rsi, 1), 'signal': 'neutral', 'emoji': '🟡'}

        # MACD signal
        if latest['macd'] > latest['macd_signal']:
            signals['macd'] = {'signal': 'bullish', 'emoji': '🟢'}
        else:
            signals['macd'] = {'signal': 'bearish', 'emoji': '🔴'}

        # SMA crossover
        if latest['sma_20'] > latest['sma_50']:
            signals['sma_cross'] = {'signal': 'golden_cross', 'emoji': '🟢'}
        else:
            signals['sma_cross'] = {'signal': 'death_cross', 'emoji': '🔴'}

        # Bollinger Band position
        price = latest['close']
        if price > latest['bb_upper']:
            signals['bollinger'] = {'signal': 'above_upper', 'emoji': '🔴'}
        elif price < latest['bb_lower']:
            signals['bollinger'] = {'signal': 'below_lower', 'emoji': '🟢'}
        else:
            signals['bollinger'] = {'signal': 'within_bands', 'emoji': '🟡'}

        # Overall trend
        bullish = sum(1 for s in signals.values() if s.get('emoji') == '🟢')
        bearish = sum(1 for s in signals.values() if s.get('emoji') == '🔴')

        if bullish > bearish:
            signals['overall'] = {'trend': 'bullish', 'emoji': '🟢', 'strength': bullish}
        elif bearish > bullish:
            signals['overall'] = {'trend': 'bearish', 'emoji': '🔴', 'strength': bearish}
        else:
            signals['overall'] = {'trend': 'neutral', 'emoji': '🟡', 'strength': 0}

        return signals
