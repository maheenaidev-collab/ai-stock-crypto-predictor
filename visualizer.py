"""
Visualization module for stock/crypto predictions.
Author: Maheen Riaz
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import numpy as np
from datetime import timedelta


def plot_price_history(df, symbol, save_path='output/price_history.png'):
    """Plot historical price with moving averages."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Price chart
    ax1.plot(df.index, df['close'], label='Close Price', color='#2196F3', linewidth=1.5)

    if 'sma_20' in df.columns:
        ax1.plot(df.index, df['sma_20'], label='SMA 20', color='#FF9800', linewidth=1, alpha=0.8)
    if 'sma_50' in df.columns:
        ax1.plot(df.index, df['sma_50'], label='SMA 50', color='#E91E63', linewidth=1, alpha=0.8)
    if 'bb_upper' in df.columns:
        ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray', label='Bollinger Bands')

    ax1.set_title(f'{symbol} — Price History', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Volume chart
    colors = ['#4CAF50' if df['close'].iloc[i] >= df['open'].iloc[i] else '#F44336' for i in range(len(df))]
    ax2.bar(df.index, df['volume'], color=colors, alpha=0.7)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Chart saved to {save_path}")


def plot_predictions(df, predictions, symbol, days, save_path='output/predictions.png'):
    """Plot price history with future predictions."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Historical (last 60 days)
    recent = df.tail(60)
    ax.plot(recent.index, recent['close'], label='Historical Price', color='#2196F3', linewidth=2)

    # Predictions
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(days)]

    # Connect historical to prediction
    connect_dates = [last_date] + future_dates
    connect_prices = [df['close'].iloc[-1]] + list(predictions.values())[0] if isinstance(predictions, dict) else [df['close'].iloc[-1]] + predictions

    colors = {'Linear Regression': '#FF9800', 'Random Forest': '#4CAF50', 'LSTM Neural Network': '#E91E63'}

    if isinstance(predictions, dict):
        for model_name, preds in predictions.items():
            conn = [df['close'].iloc[-1]] + preds
            conn_dates = [last_date] + future_dates
            color = colors.get(model_name, '#9C27B0')
            ax.plot(conn_dates, conn, label=f'{model_name}', color=color, linewidth=2, linestyle='--', marker='o', markersize=4)
    else:
        ax.plot(connect_dates, connect_prices, label='Predicted', color='#FF9800', linewidth=2, linestyle='--', marker='o', markersize=4)

    ax.axvline(x=last_date, color='gray', linestyle=':', alpha=0.5, label='Prediction Start')
    ax.set_title(f'{symbol} — Price Prediction ({days} Days)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Prediction chart saved to {save_path}")


def plot_model_comparison(metrics_list, save_path='output/model_comparison.png'):
    """Plot model comparison chart."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    models = [m['model'] for m in metrics_list]
    mae_scores = [m['mae'] for m in metrics_list]
    r2_scores = [m['r2'] for m in metrics_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#2196F3', '#4CAF50', '#FF9800']

    ax1.barh(models, mae_scores, color=colors[:len(models)])
    ax1.set_xlabel('MAE (Lower is Better)')
    ax1.set_title('Mean Absolute Error', fontweight='bold')

    ax2.barh(models, r2_scores, color=colors[:len(models)])
    ax2.set_xlabel('R² Score (Higher is Better)')
    ax2.set_title('R² Score', fontweight='bold')

    plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Comparison chart saved to {save_path}")
