# PyStock

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)

Interactive stock price prediction using LSTM neural networks and TensorFlow.

## Features

- **LSTM Neural Network**: Deep learning architecture for time-series forecasting
- **Yahoo Finance Integration**: Automatic historical data retrieval
- **Data Visualization**: Plots historical prices and prediction comparisons
- **Reproducible Results**: Deterministic predictions with seed management

## Installation

### From source

Clone the repository and install in development mode:

```bash
git clone https://github.com/yourusername/pystock.git
cd pystock
pip install -e .
```

## Quick Start

### Generate predictions

```bash
pystock AAPL
```

### Examples

```bash
pystock MSFT
pystock TSLA
pystock ^GSPC  # S&P 500
```

## Output

The tool will:

1. Fetch historical stock data from Yahoo Finance (since 2019)
2. Display raw data preview
3. Plot historical closing price chart
4. Train LSTM model (80% training, 20% testing)
5. Display prediction accuracy (RMSE)
6. Show actual vs predicted prices
7. Predict next trading day's closing price
8. Save model as `pystock.h5`

## Model Architecture

```
Input (60-day lookback)
    ↓
LSTM (50 units, return sequences)
    ↓
LSTM (50 units)
    ↓
Dense (25 neurons)
    ↓
Output (predicted price)
```

## Configuration

Edit parameters in `src/pystock/main.py`:

```python
LOOKBACK_DAYS = 60       # Historical days to analyze
TRAIN_TEST_SPLIT = 0.8   # 80/20 split
BATCH_SIZE = 1           # Training batch size
EPOCHS = 1               # Training iterations
```

## File Structure

```
pystock/
├── src/
│   └── pystock/
│       ├── __init__.py       # Package initialization
│       └── main.py           # Core prediction logic
├── pyproject.toml            # Modern packaging config
├── README.md                 # This file
├── LICENSE                   # MIT License
└── requirements.txt          # Dependencies
```

## Dependencies

- **tensorflow** (≥2.10.0): Deep learning framework
- **keras** (≥2.10.0): Neural network API
- **pandas** (≥1.3.0): Data manipulation
- **pandas-datareader** (≥0.10.0): Yahoo Finance integration
- **scikit-learn** (≥1.0.0): ML utilities
- **numpy** (≥1.21.0): Numerical computing
- **matplotlib** (≥3.5.0): Visualization

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Troubleshooting

### Module not found
```bash
pip install -r requirements.txt
```

### Invalid ticker symbol
- Verify the ticker is correct and traded on Yahoo Finance
- Check internet connection

### GPU/CUDA issues
For CPU-only TensorFlow:
```bash
pip install tensorflow-cpu
```

### Model file location
The model (`pystock.h5`) is generated in the current directory. It's ignored by git (.gitignore).

## Disclaimer

**For educational purposes only.** Stock predictions are inherently uncertain and should not be the sole basis for investment decisions. Always consult official sources and financial advisors.

## License

MIT License - See [LICENSE](LICENSE) file

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](doc/CONTRIBUTING.md) for guidelines.

Ideas for enhancements:

- Support multiple stock symbols
- Add technical indicators (RSI, MACD, Bollinger Bands)
- Web interface
- Hyperparameter optimization
- Ensemble methods
- Real-time predictions

## Documentation

- [Contributing Guide](doc/CONTRIBUTING.md) - How to contribute
- [Development Guide](doc/DEVELOPMENT.md) - Architecture and development setup
- [Changelog](doc/CHANGELOG.md) - Version history and roadmap
- [Publication Checklist](doc/PUBLICATION_CHECKLIST.md) - Pre-publication requirements

## Author

Created as an educational project in deep learning and time-series forecasting.



