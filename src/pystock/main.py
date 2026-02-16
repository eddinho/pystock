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
import json
import math
import random
import os
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
MODEL_DIR = 'models'
MODEL_PATH_TEMPLATE = 'models/pystock_{ticker_id}.h5'
MODEL_STATE_TEMPLATE = 'models/pystock_{ticker_id}.state.json'


def generate_demo_data(ticker, start_date, end_date):
    """Generate demo stock data for testing."""
    print(f"Generating demo data for {ticker}...")
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 150.0
    prices = [base_price]
    for _ in range(len(dates) - 1):
        change = np.random.normal(0.001, 0.02)
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({'Close': prices}, index=dates)
    return df


def normalize_ticker_for_filename(ticker):
    """Return a filesystem-safe ticker id for output filenames."""
    ticker_id = ''.join(char if char.isalnum() else '_' for char in ticker.upper())
    ticker_id = ticker_id.strip('_')
    return ticker_id or "TICKER"


def load_model_state(state_path):
    """Load model state from disk if available."""
    state_file = Path(state_path)
    if not state_file.exists():
        return {}

    try:
        with state_file.open('r', encoding='utf-8') as file:
            state = json.load(file)
    except Exception as exc:
        print(f"Warning: failed to read model state {state_file} ({exc})")
        return {}

    if not isinstance(state, dict):
        return {}

    return state


def save_model_state(state_path, ticker, last_trained_date):
    """Persist model state to disk."""
    state_file = Path(state_path)
    state = {
        "ticker": ticker,
        "last_trained_date": last_trained_date,
    }
    with state_file.open('w', encoding='utf-8') as file:
        json.dump(state, file, indent=2)


def build_lstm_training_data(scaled_data, start_index):
    """Build LSTM training windows starting from a target index."""
    train_x = []
    train_y = []

    for i in range(start_index, len(scaled_data)):
        train_x.append(scaled_data[i - LOOKBACK_DAYS:i, 0])
        train_y.append(scaled_data[i, 0])

    if not train_x:
        return np.empty((0, LOOKBACK_DAYS, 1)), np.empty((0,))

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    return train_x, train_y


def fetch_stock_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    yfinance_error = None

    try:
        import yfinance as yf

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='1d',
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)

        if df.empty:
            return df

        if 'Close' not in df.columns:
            raise ValueError("Yahoo Finance response did not include a 'Close' column")

        return df.dropna(subset=['Close'])
    except ModuleNotFoundError:
        pass
    except Exception as exc:
        yfinance_error = exc

    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp(end_date).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
    )
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except Exception as exc:
        if yfinance_error is not None:
            raise RuntimeError(
                f"Failed to fetch data with yfinance ({yfinance_error}) and Yahoo chart API ({exc})"
            ) from exc
        raise

    chart = payload.get('chart', {})
    error = chart.get('error')
    if error:
        description = error.get('description', 'unknown Yahoo Finance error')
        raise ValueError(description)

    results = chart.get('result') or []
    if not results:
        return pd.DataFrame()

    result = results[0]
    timestamps = result.get('timestamp') or []
    quote = (result.get('indicators', {}).get('quote') or [{}])[0]
    adj_close_data = (result.get('indicators', {}).get('adjclose') or [{}])[0]

    if not timestamps:
        return pd.DataFrame()

    df = pd.DataFrame({
        'Open': quote.get('open'),
        'High': quote.get('high'),
        'Low': quote.get('low'),
        'Close': quote.get('close'),
        'Volume': quote.get('volume'),
        'Adj Close': adj_close_data.get('adjclose'),
    })
    df.index = pd.to_datetime(timestamps, unit='s')
    df.index.name = 'Date'

    if 'Close' not in df.columns:
        raise ValueError("Yahoo Finance response did not include a 'Close' column")

    return df.dropna(subset=['Close'])


def main():
    """Main function to predict stock price."""
    if len(sys.argv) < 2:
        print("Usage: pystock <TICKER> [--days DAYS] [--demo] [--fresh-model]")
        print("Example: pystock AAPL")
        print("Example: pystock AAPL --days 30")
        print("Example: pystock AAPL --demo --days 30  # Use demo data")
        print("Example: pystock AAPL --fresh-model      # Ignore saved model")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    ticker_id = normalize_ticker_for_filename(ticker)
    model_path = MODEL_PATH_TEMPLATE.format(ticker_id=ticker_id)
    state_path = MODEL_STATE_TEMPLATE.format(ticker_id=ticker_id)
    model_dir = Path(MODEL_DIR)
    legacy_model_path = Path(f'pystock_{ticker_id}.h5')
    legacy_state_path = Path(f'pystock_{ticker_id}.state.json')
    
    # Parse optional arguments
    days_to_predict = 1
    use_demo = False
    use_existing_model = True
    
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '--days' and i + 1 < len(sys.argv):
            try:
                days_to_predict = int(sys.argv[i + 1])
                if days_to_predict < 1:
                    print("Error: days must be at least 1")
                    sys.exit(1)
            except ValueError:
                print(f"Error: invalid days value '{sys.argv[i + 1]}'")
                sys.exit(1)
        elif sys.argv[i] == '--demo':
            use_demo = True
        elif sys.argv[i] == '--fresh-model':
            use_existing_model = False

    try:
        # Create model folder
        model_dir.mkdir(exist_ok=True)
        model_path_obj = Path(model_path)
        state_path_obj = Path(state_path)

        # Migrate legacy files from project root to models/
        if not model_path_obj.exists() and legacy_model_path.exists():
            print(f"Migrating legacy model to {model_path_obj}...")
            legacy_model_path.replace(model_path_obj)
        if not state_path_obj.exists() and legacy_state_path.exists():
            print(f"Migrating legacy model state to {state_path_obj}...")
            legacy_state_path.replace(state_path_obj)

        # Create predictions folder
        prediction_dir = Path('predictions')
        prediction_dir.mkdir(exist_ok=True)
        
        # Fetch stock data
        print(f"Fetching historical data for {ticker}...")
        start = dt.datetime(2019, 1, 1)
        today = dt.date.today()
        
        if use_demo:
            df = generate_demo_data(ticker, start, today - dt.timedelta(days=1))
        else:
            df = fetch_stock_data(ticker, start, today)

        if df.empty:
            print(f"Error: No data found for ticker {ticker}")
            sys.exit(1)

        print(f"Successfully retrieved {len(df)} records\n")
        print(df.head())

        # Visualize historical close price
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='blue', width=2)))
        fig.update_layout(
            title=f'{ticker} - Close Price History',
            xaxis_title='Date',
            yaxis_title='Close Price USD ($)',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        history_html = prediction_dir / f'pystock_history_{ticker_id}.html'
        fig.write_html(history_html)
        print(f"Chart saved to {history_html}")

        # Prepare data
        print("\nPreparing training data...")
        data = df.filter(['Close'])
        dataset = data.values
        if len(dataset) <= LOOKBACK_DAYS:
            print(f"Error: Need at least {LOOKBACK_DAYS + 1} records, got {len(dataset)}")
            sys.exit(1)
        training_data_len = math.ceil(len(dataset) * TRAIN_TEST_SPLIT)

        # Scale data to 0-1 range
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Build or load model, then continue training
        model = None
        loaded_existing_model = False
        if use_existing_model and Path(model_path).exists():
            try:
                print(f"Loading existing model from {model_path}...")
                model = load_model(model_path)
                model.compile(optimizer='adam', loss='mean_squared_error')
                print("Continuing training from saved model...")
                loaded_existing_model = True
            except Exception as load_error:
                print(f"Warning: failed to load saved model ({load_error}). Building a new model.")

        if model is None:
            if use_existing_model:
                print("Building new LSTM model...")
            else:
                print("Building new LSTM model (--fresh-model enabled)...")
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(LOOKBACK_DAYS, 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

        # Determine incremental training window from saved state
        last_trained_date = None
        if loaded_existing_model:
            model_state = load_model_state(state_path)
            last_trained_date = model_state.get('last_trained_date')
            if last_trained_date:
                print(f"Last trained date: {last_trained_date}")
            else:
                print("No previous model state found. Training on full history.")

        train_start_index = LOOKBACK_DAYS
        if last_trained_date:
            try:
                last_trained_timestamp = pd.Timestamp(last_trained_date)
                unseen_indexes = np.where(data.index > last_trained_timestamp)[0]
                if unseen_indexes.size == 0:
                    train_start_index = None
                else:
                    train_start_index = max(LOOKBACK_DAYS, int(unseen_indexes[0]))
            except Exception as state_error:
                print(f"Warning: invalid state date '{last_trained_date}' ({state_error}). Training on full history.")
                train_start_index = LOOKBACK_DAYS

        if train_start_index is None:
            print("No unseen dates detected. Skipping training.")
        else:
            train_x, train_y = build_lstm_training_data(scaled_data, train_start_index)
            if train_x.shape[0] == 0:
                print("No trainable windows for unseen dates. Skipping training.")
            else:
                first_target_date = pd.Timestamp(data.index[train_start_index]).date()
                last_target_date = pd.Timestamp(data.index[-1]).date()
                print(
                    f"Training model for {EPOCHS} epoch(s) on {train_x.shape[0]} window(s) "
                    f"from {first_target_date} to {last_target_date}..."
                )
                model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
                try:
                    save_model_state(state_path, ticker, last_target_date.isoformat())
                    print(f"Updated model state: {state_path}")
                except Exception as state_error:
                    print(f"Warning: failed to save model state ({state_error})")

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
        rmse = np.sqrt(mean_squared_error(test_y, predictions))
        print(f'\nModel Performance - RMSE: {rmse:.2f}')

        # Visualize results
        train = data[:training_data_len]
        valid = data[training_data_len:].copy()
        valid['Predictions'] = predictions

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Training Data', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Actual', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions', line=dict(color='red', width=2, dash='dash')))
        fig.update_layout(
            title=f'{ticker} - Model Predictions',
            xaxis_title='Date',
            yaxis_title='Close Price USD ($)',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        validation_html = prediction_dir / f'pystock_validation_{ticker_id}.html'
        fig.write_html(validation_html)
        print(f"Chart saved to {validation_html}")

        print("\nValidation Results:")
        print(valid)

        # Predict future price(s)
        print("\n" + "="*50)
        print(f"Predicting next {days_to_predict} trading day(s)...")
        print("="*50)

        if use_demo:
            quote = df.copy()
        else:
            quote = fetch_stock_data(ticker, start, today + dt.timedelta(days=1))
        new_df = quote.filter(['Close'])
        last_60_days = new_df[-LOOKBACK_DAYS:].values
        last_60_days_scaled = scaler.transform(last_60_days)

        # Predictions for multiple days
        predictions_list = []
        current_sequence = last_60_days_scaled.copy()

        for day in range(days_to_predict):
            # Reshape for model input
            test_x = np.array([current_sequence])
            test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

            # Predict next price
            pred_price_scaled = model.predict(test_x, verbose=0)
            pred_price = scaler.inverse_transform(pred_price_scaled)
            predictions_list.append(pred_price[0][0])

            # Update sequence: remove first element, add prediction for next iteration
            current_sequence = np.append(current_sequence[1:], pred_price_scaled)

        # Display predictions
        last_price = new_df['Close'].iloc[-1]
        print(f"\nCurrent price: ${last_price:.2f}")
        print(f"\nForecast for next {days_to_predict} trading day(s):")
        print("-" * 40)

        for i, pred in enumerate(predictions_list, 1):
            change = pred - last_price
            pct_change = (change / last_price) * 100
            print(f"Day {i}: ${pred:.2f} ({change:+.2f}, {pct_change:+.2f}%)")

        if days_to_predict > 1:
            print("-" * 40)
            final_pred = predictions_list[-1]
            total_change = final_pred - last_price
            total_pct = (total_change / last_price) * 100
            print(f"Total change (Day 1-{days_to_predict}): {total_change:+.2f} ({total_pct:+.2f}%)")

        # Visualize forecast if predicting multiple days
        if days_to_predict > 1:
            print("\nGenerating forecast visualization...")
            
            # Create date range for forecast
            last_date = new_df.index[-1]
            forecast_dates = pd.date_range(start=last_date + dt.timedelta(days=1), periods=days_to_predict, freq='B')
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': predictions_list
            })
            forecast_df.set_index('Date', inplace=True)
            
            # Create plotly figure
            fig = go.Figure()
            
            # Plot historical data (last 60 days)
            historical_plot = new_df[-60:]
            fig.add_trace(go.Scatter(
                x=historical_plot.index,
                y=historical_plot['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            ))
            
            # Plot forecast
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['Forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=8)
            ))
            
            # Add point at current price
            fig.add_trace(go.Scatter(
                x=[last_date],
                y=[last_price],
                mode='markers',
                name='Current Price',
                marker=dict(color='green', size=12)
            ))
            
            fig.update_layout(
                title=f'{ticker} - Price Forecast (Next {days_to_predict} Trading Days)',
                xaxis_title='Date',
                yaxis_title='Close Price USD ($)',
                hovermode='x unified',
                template='plotly_white',
                height=600
            )
            
            forecast_html = prediction_dir / f'pystock_forecast_{ticker_id}.html'
            fig.write_html(forecast_html)
            print(f"Forecast chart saved to {forecast_html}")

        # Save model
        print(f"\nSaving model to {model_path}...")
        model.save(model_path)
        print("Done!")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
