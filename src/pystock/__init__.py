"""PyStock - Stock Price Prediction using LSTM Neural Networks."""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"


def main():
    """Console entry point proxy."""
    from .main import main as run_main

    return run_main()


__all__ = ["main"]
