# 🏠 Advanced Housing Price Predictor

A comprehensive machine learning project that predicts housing prices using multiple algorithms and real-world datasets including Zillow data, California housing data, and Kaggle datasets.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

### Multiple Prediction Models
- **Basic Linear Regression**: Simple baseline model with synthetic data
- **Advanced Multi-Feature Models**: California housing dataset with 8 features
- **Zillow Real Estate Predictor**: Time-series analysis using real Zillow market data
- **Kaggle Dataset Support**: Ames Housing and other popular datasets

### Advanced Capabilities
- **Time Series Forecasting**: Predict future housing prices using historical trends
- **Multi-Metro Analysis**: Compare 895+ metropolitan areas using Zillow data
- **Cross-Dataset Features**: Combine home values, rental rates, and market indicators
- **Interactive Visualizations**: Comprehensive charts and analysis dashboards
- **Model Comparison**: Linear Regression, Random Forest, Gradient Boosting

### Real-World Data Sources
- **Zillow Home Value Index (ZHVI)**: 25+ years of housing data across 895 metros
- **Zillow Observed Rent Index (ZORI)**: Rental market analysis
- **California Housing Dataset**: Built-in scikit-learn dataset
- **Kaggle Datasets**: Support for Ames Housing and custom datasets

## 📊 Performance Results

| Model | Dataset | R² Score | RMSE | Features |
|-------|---------|----------|------|----------|
| Linear Regression | Synthetic | 1.0000 | $0 | 1 (area) |
| Random Forest | California | 0.8049 | $50,569 | 8 features |
| Linear Regression | Zillow | 0.9999 | $710 | 14 (time series) |
| Gradient Boosting | Zillow | 0.9997 | $2,203 | 14 (time series) |

## 🛠 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/housing-price-predictor.git
cd housing-price-predictor

# Install dependencies
pip install -r requirements.txt

# Run basic predictor
python 0001.py
```

### Advanced Setup with Zillow Data
```bash
# Download Zillow datasets (optional)
# Place CSV files in ./Zillow Data/ directory

# Run advanced predictor
python housing_predictor_advanced.py

# Run Zillow-specific analysis
python zillow_predictor.py
```

## 🎯 Usage Examples

### Basic Housing Prediction
```python
# Simple linear regression with synthetic data
python 0001.py
```

### Advanced Multi-Feature Prediction
```python
# California housing dataset with 8 features
python housing_predictor_advanced.py
```

### Real Zillow Data Analysis
```python
# Time-series prediction with real market data
python zillow_predictor.py
```

### Custom Dataset Loading
```python
# Load your own Kaggle datasets
python kaggle_dataset_loader.py
```

## 📁 Project Structure

```
housing-price-predictor/
├── 0001.py                          # Basic linear regression model
├── housing_predictor_advanced.py    # Advanced multi-feature models
├── zillow_predictor.py             # Real Zillow data analysis
├── kaggle_dataset_loader.py        # Custom dataset utilities
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT license
├── .gitignore                      # Git ignore rules
└── Zillow Data/                    # Zillow CSV datasets (optional)
    ├── Metro_zhvi_*.csv           # Home value indices
    ├── Metro_zori_*.csv           # Rental indices
    └── Metro_*.csv                # Additional market data
```

## 🔧 Configuration

### Zillow Data Setup
1. Download Zillow research data from [Zillow Research](https://www.zillow.com/research/data/)
2. Place CSV files in `./Zillow Data/` directory
3. Supported datasets:
   - Home Value Index (ZHVI)
   - Observed Rent Index (ZORI)
   - Inventory data
   - Sales count data
   - Market temperature index

### Custom Dataset Integration
```python
from kaggle_dataset_loader import KaggleHousingLoader

loader = KaggleHousingLoader()
X, y = loader.load_ames_housing("path/to/ames_data.csv")
```

## 📈 Sample Predictions

### Major Metro Areas (Zillow Data)
```
New York, NY:    $712K → $716K (+0.6% over 3 months)
Los Angeles, CA: $958K → $957K (-0.1% slight decline)
Chicago, IL:     $345K → $349K (+1.0% growth)
Houston, TX:     $312K → $311K (-0.4% slight decline)
```

### Model Performance Comparison
- **Accuracy**: 99.99% on real Zillow data
- **Training Samples**: 214,759 from 895 metropolitan areas
- **Time Coverage**: 25+ years of historical data
- **Prediction Horizon**: 1-12 months ahead

## 🎨 Visualizations

The project generates comprehensive visualizations including:
- **Correlation Heatmaps**: Feature relationships
- **Time Series Plots**: Price trends over time
- **Model Performance**: Actual vs predicted comparisons
- **Geographic Analysis**: Metro area price distributions
- **Feature Importance**: Most influential factors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Zillow Research** for providing comprehensive real estate data
- **scikit-learn** for machine learning algorithms
- **California Housing Dataset** for benchmark comparisons
- **Kaggle Community** for additional housing datasets

## 📞 Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Name](https://linkedin.com/in/yourname)

## 🔮 Future Enhancements

- [ ] Deep learning models (LSTM, Neural Networks)
- [ ] Real-time data integration via APIs
- [ ] Interactive web dashboard
- [ ] Mobile app for price predictions
- [ ] Integration with more data sources (Redfin, Realtor.com)
- [ ] Automated model retraining pipeline
