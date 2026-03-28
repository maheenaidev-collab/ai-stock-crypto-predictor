"""
Backtesting engine for prediction models.
Author: Maheen Riaz
"""

import numpy as np
from datetime import timedelta


class Backtester:
    """Test model accuracy on historical data."""

    def __init__(self):
        self.results = []

    def run(self, df, model, model_name, test_days=30):
        """
        Backtest a model on historical data.

        Args:
            df: DataFrame with indicators
            model: Trained model instance
            model_name: Name of the model
            test_days: Number of days to backtest

        Returns:
            dict: Backtest results
        """
        actuals = []
        predictions = []

        total_rows = len(df)
        start_idx = total_rows - test_days

        for i in range(start_idx, total_rows - 1):
            train_data = df.iloc[:i]
            actual_price = df.iloc[i + 1]['close']

            try:
                pred = model.predict(train_data, days=1)
                if pred:
                    predictions.append(pred[0])
                    actuals.append(actual_price)
            except Exception:
                continue

        if not predictions:
            return {'model': model_name, 'error': 'No predictions generated'}

        actuals = np.array(actuals)
        predictions = np.array(predictions)

        # Direction accuracy
        actual_direction = np.diff(actuals) > 0
        pred_direction = predictions[1:] > actuals[:-1]
        direction_accuracy = np.mean(actual_direction == pred_direction) if len(actual_direction) > 0 else 0

        mae = np.mean(np.abs(actuals - predictions))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        return {
            'model': model_name,
            'test_days': len(predictions),
            'mae': round(mae, 2),
            'mape': round(mape, 2),
            'direction_accuracy': round(direction_accuracy * 100, 1),
            'avg_actual': round(np.mean(actuals), 2),
            'avg_predicted': round(np.mean(predictions), 2)
        }
