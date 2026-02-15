# Contributing to PyStock

Thank you for your interest in contributing to PyStock! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/pystock.git
   cd pystock
   ```

3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

### Testing Your Changes

Before submitting a pull request:

1. Test your changes with different stock tickers
2. Verify no errors occur with edge cases (invalid tickers, network issues)
3. Check that the model still trains and makes predictions

### Commit Messages

Write clear, descriptive commit messages:

```
Fix: Corrected LSTM layer configuration
Feature: Added support for custom epochs parameter
Docs: Updated README with new examples
```

## Types of Contributions

### Bug Reports

- Check existing issues first
- Include stock ticker used and full error message
- Describe steps to reproduce

### Feature Requests

- Describe the feature clearly
- Explain the use case and benefits
- Provide examples if applicable

### Code Improvements

- Performance optimizations
- Better error handling
- Improved documentation
- Code refactoring

## Pull Request Process

1. Update documentation if needed
2. Add descriptive comments for complex logic
3. Keep commits atomic and focused
4. Push to your fork and submit a pull request
5. Link relevant issues in the PR description

## Questions?

Feel free to open an issue to discuss ideas or ask questions.

Happy coding!
