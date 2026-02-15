#!/usr/bin/env python3
"""
PyStock - Stock Price Prediction using LSTM Neural Network

Predict closing stock prices using a neural network with LSTM (Long Short-Term Memory) layers
and TensorFlow backend.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import sys
import datetime as dt
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from pandas_datareader import data as web
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
SEED_VALUE = 0
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

plt.style.use('fivethirtyeight')

# Configuration
LOOKBACK_DAYS = 60
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 1
EPOCHS = 1
MODEL_PATH = 'pystock.h5'


def main():
    """Main function to predict stock price."""
    if len(sys.argv) < 2:
        print("Usage: pystock <TICKER>")
        print("Example: pystock AAPL")
        sys.exit(1)

    ticker = sys.argv[1]

    try:
        # Fetch stock data
        print(f"Fetching historical data for {ticker}...")
        start = dt.datetime(2019, 1, 1)
        today = dt.date.today()
        df = web.DataReader(ticker, 'yahoo', start, today - dt.timedelta(days=1))

        if df.empty:
            print(f"Error: No data found for ticker {ticker}")
            sys.exit(1)

        print(f"Successfully retrieved {len(df)} records\n")
        print(df.head())

        # Visualize historical close price
        plt.figure(figsize=(16, 8))
        plt.title(f'{ticker} - Close Price History')
        plt.plot(df['Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.show()

        # Prepare data
        print("\nPreparing training data...")
        data = df.filter(['Close'])
        dataset = data.values
        training_data_len = math.ceil(len(dataset) * TRAIN_TEST_SPLIT)

        # Scale data to 0-1 range
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Create training dataset
        training_data = scaled_data[0:training_data_len, :]
        train_x = []
        train_y = []

        for i in range(LOOKBACK_DAYS, len(training_data)):
            train_x.append(training_data[i - LOOKBACK_DAYS:i, 0])
            train_y.append(training_data[i, 0])

        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

        # Build and train model
        print("Building LSTM model...")
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(train_x.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        print(f"Training model for {EPOCHS} epoch(s)...")
        model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

        # Create testing dataset
        test_data = scaled_data[training_data_len - LOOKBACK_DAYS:, :]
        test_x = []
        test_y = dataset[training_data_len:, :]

        for i in range(LOOKBACK_DAYS, len(test_data)):
            test_x.append(test_data[i - LOOKBACK_DAYS:i, 0])

        test_x = np.array(test_x)
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

        # Make predictions
        print("Making predictions...")
        predictions = model.predict(test_x)
        predictions = scaler.inverse_transform(predictions)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_y[0], predictions[0]))
        print(f'\nModel Performance - RMSE: {rmse:.2f}')

        # Visualize results
        train = data[:training_data_len]
        valid = data[training_data_len:].copy()
        valid['Predictions'] = predictions

        plt.figure(figsize=(16, 8))
        plt.title(f'{ticker} - Model Predictions')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'], label='Trained')
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Trained', 'Actual', 'Predictions'], loc='lower right')
        plt.show()

        print("\nValidation Results:")
        print(valid)

        # Predict future price
        print("\n" + "="*50)
        print("Predicting next trading day...")
        print("="*50)

        quote = web.DataReader(ticker, 'yahoo', start, today)
        new_df = quote.filter(['Close'])
        last_60_days = new_df[-LOOKBACK_DAYS:].values
        last_60_days_scaled = scaler.transform(last_60_days)

        test_x = np.array([last_60_days_scaled])
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

        pred_price = model.predict(test_x)
        pred_price = scaler.inverse_transform(pred_price)

        print(f"\nPredicted closing price for {ticker}: ${pred_price[0][0]:.2f}")

        # Save model
        print(f"\nSaving model to {MODEL_PATH}...")
        model.save(MODEL_PATH)
        print("Done!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
