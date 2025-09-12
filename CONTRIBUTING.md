# 🤝 Contributing to Housing Price Predictor

Thank you for your interest in contributing to the Housing Price Predictor project! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- Git
- Basic knowledge of machine learning and data science

### Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/yourusername/housing-price-predictor.git
cd housing-price-predictor
```

## 🔧 Development Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

### 3. Install Development Dependencies
```bash
pip install pytest black flake8 mypy jupyter
```

### 4. Verify Installation
```bash
python 0001.py  # Test basic functionality
python -m pytest  # Run tests (if available)
```

## 📝 Contributing Guidelines

### What We're Looking For
- **New Datasets**: Integration with additional housing data sources
- **Model Improvements**: New algorithms or feature engineering techniques
- **Visualizations**: Enhanced charts and interactive dashboards
- **Performance Optimizations**: Speed and memory improvements
- **Documentation**: Better examples, tutorials, and API docs
- **Bug Fixes**: Resolving issues and edge cases

### Areas for Contribution
1. **Data Sources**
   - Redfin data integration
   - Realtor.com API support
   - International housing markets
   - Economic indicators integration

2. **Machine Learning**
   - Deep learning models (LSTM, Neural Networks)
   - Ensemble methods
   - Feature selection algorithms
   - Hyperparameter optimization

3. **Visualization**
   - Interactive web dashboards
   - Geographic mapping
   - Time series animations
   - Mobile-responsive charts

4. **Infrastructure**
   - Docker containerization
   - CI/CD pipelines
   - Cloud deployment scripts
   - API development

## 🎨 Code Style

### Python Style Guide
We follow PEP 8 with some modifications:

```python
# Good: Clear function names and docstrings
def predict_metro_prices(metro_name: str, months_ahead: int = 6) -> List[float]:
    """
    Predict future housing prices for a specific metropolitan area.
    
    Args:
        metro_name: Name of the metropolitan area
        months_ahead: Number of months to predict ahead
        
    Returns:
        List of predicted prices
    """
    pass

# Good: Type hints and clear variable names
housing_data: pd.DataFrame = load_zillow_data()
prediction_results: Dict[str, float] = {}
```

### Code Formatting
Use `black` for automatic code formatting:
```bash
black *.py
```

### Linting
Use `flake8` for linting:
```bash
flake8 *.py
```

### Type Checking
Use `mypy` for type checking:
```bash
mypy *.py
```

## 🧪 Testing

### Writing Tests
Create tests in the `tests/` directory:

```python
import pytest
from housing_predictor_advanced import HousingPredictor

def test_california_housing_load():
    predictor = HousingPredictor()
    X, y = predictor.load_california_housing()
    
    assert X is not None
    assert y is not None
    assert X.shape[0] > 0
    assert len(y) == X.shape[0]

def test_model_training():
    predictor = HousingPredictor()
    X, y = predictor.load_california_housing()
    
    predictor.train_models()
    
    assert len(predictor.models) > 0
    assert 'Linear Regression' in predictor.models
```

### Running Tests
```bash
pytest tests/
pytest --cov=. tests/  # With coverage
```

## 📤 Submitting Changes

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes
- Write clear, documented code
- Add tests for new functionality
- Update documentation as needed
- Follow the code style guidelines

### 3. Commit Changes
```bash
git add .
git commit -m "feat: add XGBoost model support

- Implement XGBoost regressor in housing_predictor_advanced.py
- Add hyperparameter tuning for XGBoost
- Update documentation with XGBoost examples
- Add tests for XGBoost functionality

Closes #123"
```

### Commit Message Format
Use conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

### 4. Push and Create PR
```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference to related issues
- Screenshots (if applicable)
- Test results

## 🐛 Issue Reporting

### Before Reporting
1. Check existing issues
2. Try the latest version
3. Provide minimal reproduction case

### Issue Template
```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Load dataset '...'
2. Run command '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., macOS 12.0]
- Python version: [e.g., 3.9.7]
- Package versions: [run `pip list`]

**Additional Context**
Any other context about the problem.
```

## 🏷️ Release Process

### Version Numbering
We use Semantic Versioning (SemVer):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes (backward compatible)

### Release Checklist
- [ ] Update version in `setup.py`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in Git

## 🎯 Project Roadmap

### Short Term (1-3 months)
- [ ] Add more ML algorithms (XGBoost, LightGBM)
- [ ] Improve data preprocessing pipeline
- [ ] Add more comprehensive tests
- [ ] Create Jupyter notebook tutorials

### Medium Term (3-6 months)
- [ ] Web dashboard with Flask/Django
- [ ] Real-time data integration
- [ ] Docker containerization
- [ ] API development

### Long Term (6+ months)
- [ ] Deep learning models
- [ ] Mobile application
- [ ] Cloud deployment
- [ ] Multi-language support

## 💬 Community

### Getting Help
- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and ideas
- Email: [your.email@example.com]

### Code of Conduct
Please be respectful and inclusive. We want this to be a welcoming community for everyone.

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to the Housing Price Predictor project! 🏠✨
