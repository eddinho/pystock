# Development Guide

This document provides guidance for developers and maintainers of PyStock.

## Project Overview

PyStock is a Python application that uses LSTM neural networks to predict stock closing prices.

### Architecture

```
src/pystock/
├── __init__.py       # Package initialization
└── main.py          # Core prediction logic
    ├── Data Fetching   # Yahoo Finance integration
    ├── Data Prep       # Scaling and sequence creation
    ├── Model Training  # LSTM neural network
    ├── Prediction      # Make future price predictions
    └── Visualization   # Matplotlib charts
```

## Setting Up Development Environment

1. Clone and navigate to the repository:
   ```bash
   git clone https://github.com/yourusername/pystock.git
   cd pystock
   ```

2. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e .
   ```

4. Install linting tools:
   ```bash
   pip install flake8 pylint pytest
   ```

## Key Parameters and Constants

Located at the top of `src/pystock/main.py`:

- `SEED_VALUE`: Random seed for reproducibility (default: 0)
- `LOOKBACK_DAYS`: Number of days to use for predictions (default: 60)
- `TRAIN_TEST_SPLIT`: Data split ratio (default: 0.8 = 80/20)
- `BATCH_SIZE`: Training batch size (default: 1)
- `EPOCHS`: Number of training epochs (default: 1)
- `MODEL_PATH`: Where to save the model (default: 'pystock.h5')

## Code Quality

### Linting

Check code style:
```bash
flake8 src/pystock/
pylint src/pystock/
```

Fix common issues:
```bash
autopep8 --in-place --aggressive src/pystock/main.py
```

### Testing

Currently, manual testing with different tickers is recommended:

```bash
pystock AAPL
pystock MSFT
pystock GOOG
```

Test edge cases:
```bash
# Invalid ticker
pystock INVALID_TICKER

# Special characters
pystock ^GSPC
```

## Making Changes

### Adding New Features

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make changes with clear commits:
   ```bash
   git add .
   git commit -m "Feature: description of change"
   ```

3. Push and create a Pull Request:
   ```bash
   git push origin feature/your-feature-name
   ```

### Improving the Model

To experiment with different architectures:

1. Modify the model definition in the `main()` function
2. Test thoroughly with multiple tickers
3. Document changes in your commit message
4. Consider impact on training time

Example: Adding more LSTM layers:
```python
model.add(LSTM(50, return_sequences=True, input_shape=(train_x.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))  # Additional layer
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
```

## Debugging

### Enable verbose output during training:
```python
model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)
```

### Check data shapes:
```python
print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")
print(f"test_x shape: {test_x.shape}")
```

### Test with smaller dataset:
Temporarily reduce EPOCHS to quickly test changes:
```python
EPOCHS = 1  # Quick test
```

## Releasing New Versions

1. Update version in `pyproject.toml`
2. Update `doc/CHANGELOG.md` with changes
3. Create a git tag:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```
4. Create a GitHub Release

## Useful Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
- [Time Series Forecasting with LSTM](https://machinelearningmastery.com/time-series-forecasting-with-the-long-short-term-memory-network-in-python/)
- [Stock Data - Yahoo Finance](https://finance.yahoo.com)

## Troubleshooting Common Issues

### "No module named tensorflow"
```bash
pip install tensorflow>=2.10.0
```

### Model training is too slow
- Reduce `LOOKBACK_DAYS`
- Reduce LSTM units
- Use `BATCH_SIZE = 32` instead of 1

### Memory issues
- Train on GPU: Install tensorflow-gpu
- Reduce model size and batch size

## Contributing

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

For questions about development, open an issue on GitHub.
