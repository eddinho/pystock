# PyStock

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow 2.10+](https://img.shields.io/badge/tensorflow-2.10%2B-orange)

Interactive stock price prediction using LSTM neural networks and TensorFlow.

**⚠️ DISCLAIMER:** This is an educational project and **NOT a financial tool or investment advice.** See [disclaimer](#disclaimer) below.

## Features

- **LSTM Neural Network**: Deep learning architecture for time-series forecasting
- **Yahoo Finance Integration**: Automatic historical data retrieval
- **Data Visualization**: Plots historical prices and prediction comparisons
- **Reproducible Results**: Deterministic predictions with seed management

## Prerequisites

- **Python 3.8** or higher
- **pip** (Python package manager)
- **Internet connection** (to fetch stock data from Yahoo Finance)
- **4GB RAM** minimum (GPU optional, CPU works fine)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow 2.10+](https://img.shields.io/badge/tensorflow-2.10%2B-orange)

Interactive stock price prediction using LSTM neural networks and TensorFlow.

**⚠️ DISCLAIMER:** This is an educational project and **NOT a financial tool or investment advice.** See [disclaimer](#disclaimer) below.

## Features

- **LSTM Neural Network**: Deep learning architecture for time-series forecasting
- **Yahoo Finance Integration**: Automatic historical data retrieval
- **Data Visualization**: Plots historical prices and prediction comparisons
- **Reproducible Results**: Deterministic predictions with seed management

## Prerequisites

- **Python 3.8** or higher
- **pip** (Python package manager)
- **Internet connection** (to fetch stock data from Yahoo Finance)
- **4GB RAM** minimum (GPU optional, CPU works fine)

## Installation

### From source

Clone the repository and install in development mode:

```bash
git clone https://github.com/eddinho/pystock.git
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

## Project Structure

```
pystock/
├── src/
│   └── pystock/
│       ├── __init__.py              # Package initialization
│       └── main.py                  # Core prediction logic
├── doc/
│   ├── CONTRIBUTING.md              # Contribution guidelines
│   ├── DEVELOPMENT.md               # Developer documentation
│   ├── CHANGELOG.md                 # Version history
│   └── PUBLICATION_CHECKLIST.md     # Pre-publication requirements
├── pyproject.toml                   # Modern packaging configuration
├── README.md                        # This file
├── LICENSE                          # MIT License
└── requirements.txt                 # Project dependencies
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

## Branching Strategy

This project uses a **Git flow** branching strategy to maintain code quality and enable collaborative development.

### Branch Structure

- **`main`** - Production-ready code. Stable releases only.
- **`dev`** - Development/staging branch where features integrate before production.
- **`feature/*`** - Feature branches for new functionality (created from `dev`).

### Contributor Workflow

Follow these steps to contribute:

1. **Create feature branch from dev**
   ```bash
   git checkout -b feature/your-feature origin/dev
   ```

2. **Make changes and commit**
   ```bash
   git add .
   git commit -m "Feature: clear description of changes"
   ```

3. **Push to your fork**
   ```bash
   git push -u origin feature/your-feature
   ```

4. **Create Pull Request to dev** (NOT main)
   - Set PR target to `dev` branch
   - Request review from maintainers
   - Address any feedback
   - Wait for approval before merging

5. **After approval, merge to dev**
   - PR is merged by maintainer
   - Feature is now in the staging/development branch

6. **Release to main when ready**
   - Maintainer creates PR from `dev` → `main`
   - Tests and reviews are performed
   - After merge to main, create release tag: `git tag -a v1.x.x`
   - This triggers a new production release

### Branch Protection Rules

**Protected branches:**
- **`main`**: Requires pull request review before merge. All contributors must use PRs.
- **`dev`**: Requires pull request review before merge.

⚠️ **Direct pushes to `main` or `dev` are blocked.** All changes must go through pull requests.

## Troubleshooting

### Module not found errors
```bash
pip install -r requirements.txt
```

### Invalid ticker symbol
- Verify the ticker is correct and traded on Yahoo Finance
- Check your internet connection
- Try a well-known symbol like AAPL or MSFT to test

### No data retrieved
- Yahoo Finance may have data limitations or temporary issues
- Try again in a few minutes
- Test your internet connection

### GPU/CUDA issues
For CPU-only TensorFlow (faster installation):
```bash
pip install tensorflow-cpu
```

### Model file location
- The trained model is saved as `pystock.h5` in the current directory
- It's automatically ignored by git (see `.gitignore`)
- Each run creates a new model, overwriting the previous one

### Out of memory errors
- Reduce `BATCH_SIZE` from 1 to smaller values
- Reduce `LOOKBACK_DAYS` from 60 to 30-45
- Reduce LSTM units from 50 to 25-30

## Disclaimer {#disclaimer}

### Legal Notice

**THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.**

### Educational Purpose Only

PyStock is an **educational project** designed to teach LSTM neural networks and time-series forecasting. It is **NOT**:
- A financial tool
- An investment advisor
- Financial advice
- A recommendation to buy or sell securities

### Risk Warning

⚠️ **Stock price predictions are inherently uncertain and should NEVER be used as the sole basis for investment decisions.** Factors affecting stock prices include:
- Market conditions and volatility
- Economic indicators
- Company announcements and earnings
- Geopolitical events
- Regulatory changes
- Model limitations and errors

Past performance does not guarantee future results.

### Liability

The authors and contributors are **not responsible** for any:
- Financial losses or gains
- Investment decisions based on this tool
- Damages or consequences arising from using this software
- Data accuracy or completeness from external sources

**Always consult qualified financial professionals before making investment decisions.**

## Documentation

- [Contributing Guide](doc/CONTRIBUTING.md) - How to contribute to the project
- [Development Guide](doc/DEVELOPMENT.md) - Architecture, setup, and debugging
- [Changelog](doc/CHANGELOG.md) - Version history and future roadmap
- [Publication Checklist](doc/PUBLICATION_CHECKLIST.md) - Pre-publication requirements

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](doc/CONTRIBUTING.md) for guidelines.

### Ideas for Enhancement

- Support multiple stock symbols comparison
- Add technical indicators (RSI, MACD, Bollinger Bands)
- Web interface with Flask/Django
- Hyperparameter optimization and tuning
- Ensemble methods (combining multiple models)
- Real-time predictions with websockets
- Portfolio analysis features
- More advanced architectures (GRU, Transformer)

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Author

Created as an educational project in deep learning and time-series forecasting.

---

**Questions?** Open an issue on GitHub or check the [Development Guide](doc/DEVELOPMENT.md).



