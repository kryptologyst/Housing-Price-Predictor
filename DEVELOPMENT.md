# Housing Price Predictor - Development Environment Setup

This document provides instructions for setting up a development environment for the Housing Price Predictor project.

## Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git

## Quick Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/housing-price-predictor.git
cd housing-price-predictor
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n housing-predictor python=3.9
conda activate housing-predictor
```

### 3. Install Dependencies
```bash
# Install core dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install all optional dependencies
pip install -e ".[full]"
```

### 4. Verify Installation
```bash
# Run quick start
python quick_start.py

# Run tests
pytest tests/ -v

# Start web interface
streamlit run web_app/app.py
```

## Development Tools

### Code Formatting
```bash
# Format code with black
black src/ tests/ web_app/

# Check code style with flake8
flake8 src/ tests/ web_app/
```

### Type Checking
```bash
# Run type checking with mypy
mypy src/
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/housing_predictor

# Run specific test file
pytest tests/test_housing_predictor.py -v
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Project Structure

```
housing-price-predictor/
├── src/
│   └── housing_predictor/
│       ├── __init__.py
│       ├── core/           # Core functionality
│       ├── utils/          # Utility functions
│       └── config/         # Configuration management
├── web_app/               # Web interfaces
├── tests/                 # Test suite
├── data/                  # Data files
├── models/                # Saved models
├── config/                # Configuration files
├── docs/                  # Documentation
└── archive/               # Legacy code
```

## Configuration

The project uses YAML configuration files. Main configuration is in `config/config.yaml`.

### Environment Variables
You can override configuration using environment variables:
- `HOUSING_DATA_PATH`: Path to data directory
- `HOUSING_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `HOUSING_CACHE_DIR`: Cache directory path

## Data Setup

### Zillow Data
1. Download Zillow data from https://www.zillow.com/research/data/
2. Place CSV files in `data/zillow_data/` directory
3. Ensure files follow the naming convention in `DataLoader.load_zillow_data()`

### Synthetic Data
Synthetic data is generated automatically when needed. No setup required.

## Running the Application

### Web Interface
```bash
streamlit run web_app/app.py
```
Open http://localhost:8501 in your browser.

### Command Line Interface
```bash
# Train models
python web_app/cli.py train --dataset california --models linear_regression random_forest

# Make prediction
python web_app/cli.py predict --model random_forest --features 8.3 41.0 6.9 1.0 322.0 2.5 37.9 -122.2

# Evaluate models
python web_app/cli.py evaluate --dataset synthetic --models all
```

### Python API
```python
from housing_predictor import HousingPredictor

# Initialize and run pipeline
predictor = HousingPredictor()
results = predictor.run_full_pipeline('california', ['random_forest'])

# Make prediction
prediction = predictor.predict(features_array)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you've installed the package in development mode with `pip install -e .`

2. **Missing Data**: Check that data files are in the correct directories and have proper permissions.

3. **Memory Issues**: For large datasets, consider reducing batch sizes or using data chunking.

4. **Model Training Fails**: Check that all required dependencies are installed, especially for advanced models like XGBoost.

### Getting Help

- Check the logs in `logs/` directory
- Run with debug logging: `HOUSING_LOG_LEVEL=DEBUG python your_script.py`
- Open an issue on GitHub with error details

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and add tests
4. Run tests and linting: `pytest && flake8 && black --check`
5. Commit changes: `git commit -m "Add your feature"`
6. Push to branch: `git push origin feature/your-feature`
7. Create a Pull Request

## Performance Tips

- Use `numba` for numerical computations
- Enable GPU support for deep learning models
- Use data caching for repeated operations
- Consider using `dask` for large datasets

## Security Notes

- Never commit API keys or sensitive data
- Use environment variables for configuration
- Validate all user inputs
- Keep dependencies updated
