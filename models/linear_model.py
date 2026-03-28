"""
Linear Regression model for price prediction.
Author: Maheen Riaz
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LinearPredictor:
    """Linear Regression based price predictor."""

    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, df, target_days=1):
        """Prepare feature matrix and target variable."""
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]

        X = df[feature_cols].values
        y = df['close'].shift(-target_days).values

        # Remove NaN rows
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        return X, y, feature_cols

    def train(self, df, target_days=1):
        """Train the model."""
        X, y, feature_cols = self.prepare_features(df, target_days)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        self.feature_cols = feature_cols

        # Evaluate
        predictions = self.model.predict(X_test_scaled)
        metrics = {
            'mae': round(mean_absolute_error(y_test, predictions), 2),
            'rmse': round(np.sqrt(mean_squared_error(y_test, predictions)), 2),
            'r2': round(r2_score(y_test, predictions), 4),
            'model': 'Linear Regression'
        }

        return metrics

    def predict(self, df, days=7):
        """Predict future prices."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")

        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        last_features = df[feature_cols].iloc[-1:].values
        last_features_scaled = self.scaler.transform(last_features)

        predictions = []
        current_price = df['close'].iloc[-1]

        for i in range(days):
            pred = self.model.predict(last_features_scaled)[0]
            predictions.append(round(pred, 2))

        return predictions
