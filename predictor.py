"""
Core prediction engine — orchestrates data, indicators, and models.
Author: Maheen Riaz
"""

from data_fetcher import DataFetcher
from indicators import TechnicalIndicators
from models import LinearPredictor, ForestPredictor, LSTMPredictor


class StockPredictor:
    """Main prediction engine combining all components."""

    def __init__(self):
        self.fetcher = DataFetcher()
        self.indicators = TechnicalIndicators()
        self.models = {
            'linear': LinearPredictor(),
            'forest': ForestPredictor(),
            'lstm': LSTMPredictor()
        }

    def analyze(self, symbol, period_days=365):
        """Fetch data and compute indicators."""
        df = self.fetcher.fetch(symbol, period_days)
        if df is None:
            return None, None

        df_indicators = self.indicators.add_all(df)
        signals = self.indicators.get_signals(df_indicators)

        return df_indicators, signals

    def train_all(self, df):
        """Train all models and return metrics."""
        metrics = []

        print("\n🏋️ Training models...\n")

        # Linear Regression
        print("  📐 Training Linear Regression...")
        lr_metrics = self.models['linear'].train(df)
        metrics.append(lr_metrics)
        print(f"     MAE: {lr_metrics['mae']} | R²: {lr_metrics['r2']}")

        # Random Forest
        print("  🌲 Training Random Forest...")
        rf_metrics = self.models['forest'].train(df)
        metrics.append(rf_metrics)
        print(f"     MAE: {rf_metrics['mae']} | R²: {rf_metrics['r2']}")

        # LSTM
        print("  🧠 Training LSTM Network...")
        lstm_metrics = self.models['lstm'].train(df, epochs=50)
        metrics.append(lstm_metrics)
        if 'error' not in lstm_metrics:
            print(f"     MAE: {lstm_metrics['mae']} | R²: {lstm_metrics['r2']}")
        else:
            print(f"     ⚠️ {lstm_metrics['error']}")

        return metrics

    def predict(self, symbol, days=7, period_days=365):
        """Full prediction pipeline."""
        # Fetch and analyze
        df, signals = self.analyze(symbol, period_days)
        if df is None:
            return None

        current_price = df['close'].iloc[-1]
        asset_info = self.fetcher.get_asset_info(symbol)

        # Train models
        metrics = self.train_all(df)

        # Generate predictions
        predictions = {}
        ensemble_preds = []

        for name, model in self.models.items():
            try:
                if model.is_trained:
                    preds = model.predict(df, days)
                    model_name = metrics
