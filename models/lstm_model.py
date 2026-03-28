"""
LSTM Neural Network model for price prediction.
Author: Maheen Riaz
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LSTMPredictor:
    """LSTM Neural Network based price predictor."""

    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.is_trained = False

    def _build_model(self, input_shape):
        """Build LSTM model architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam

            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            return model

        except ImportError:
            print("⚠️  TensorFlow not installed. LSTM model unavailable.")
            print("   Install with: pip install tensorflow")
            return None

    def _create_sequences(self, data):
        """Create sequences for LSTM input."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(data[i, 0])  # Predict close price
        return np.array(X), np.array(y)

    def train(self, df, epochs=50, batch_size=32):
        """Train the LSTM model."""
        feature_cols = ['close', 'volume', 'rsi', 'macd', 'sma_20', 'daily_return']
        available_cols = [col for col in feature_cols if col in df.columns]

        data = df[available_cols].values
        scaled_data = self.scaler.fit_transform(data)

        X, y = self._create_sequences(scaled_data)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model = self._build_model((X_train.shape[1], X_train.shape[2]))

        if self.model is None:
            return {'model': 'LSTM', 'error': 'TensorFlow not available'}

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0
        )

        self.is_trained = True

        # Evaluate
        predictions_scaled = self.model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        pred_full = np.zeros((len(predictions_scaled), len(available_cols)))
        pred_full[:, 0] = predictions_scaled.flatten()
        predictions = self.scaler.inverse_transform(pred_full)[:, 0]

        actual_full = np.zeros((len(y_test), len(available_cols)))
        actual_full[:, 0] = y_test
        actuals = self.scaler.inverse_transform(actual_full)[:, 0]

        metrics = {
            'mae': round(mean_absolute_error(actuals, predictions), 2),
            'rmse': round(np.sqrt(mean_squared_error(actuals, predictions)), 2),
            'r2': round(r2_score(actuals, predictions), 4),
            'model': 'LSTM Neural Network',
            'epochs': epochs
        }

        return metrics

    def predict(self, df, days=7):
        """Predict future prices."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")

        feature_cols = ['close', 'volume', 'rsi', 'macd', 'sma_20', 'daily_return']
        available_cols = [col for col in feature_cols if col in df.columns]

        data = df[available_cols].values
        scaled_data = self.scaler.transform(data)

        predictions = []
        current_sequence = scaled_data[-self.sequence_length:]

        for _ in range(days):
            input_seq = current_sequence.reshape(1, self.sequence_length, len(available_cols))
            pred_scaled = self.model.predict(input_seq, verbose=0)[0][0]

            pred_full = np.zeros((1, len(available_cols)))
            pred_full[0, 0] = pred_scaled
            pred_price = self.scaler.inverse_transform(pred_full)[0, 0]
            predictions.append(round(pred_price, 2))

            new_row = current_sequence[-1].copy()
            new_row[0] = pred_scaled
            current_sequence = np.vstack([current_sequence[1:], new_row])

        return predictions
