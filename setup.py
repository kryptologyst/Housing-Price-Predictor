"""
Setup script for Housing Price Predictor v2.0

A modern, comprehensive housing price prediction system using state-of-the-art
machine learning techniques and real-world datasets.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
with open(requirements_path, "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Core requirements (without optional dependencies)
core_requirements = [
    "numpy>=1.24.0",
    "pandas>=2.0.0", 
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "pyyaml>=6.0",
    "loguru>=0.7.0",
    "python-dotenv>=1.0.0"
]

setup(
    name="housing-price-predictor",
    version="2.0.0",
    author="AI Projects Series",
    author_email="your.email@example.com",
    description="Modern housing price prediction system with ML models and real datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/housing-price-predictor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "full": requirements,
        "ml": [
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ],
        "web": [
            "streamlit>=1.25.0",
            "gradio>=3.40.0",
            "plotly>=5.15.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipython>=8.14.0",
            "notebook>=6.5.0",
        ],
        "performance": [
            "numba>=0.57.0",
            "joblib>=1.3.0",
            "dask>=2023.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "housing-predictor=housing_predictor.cli:main",
            "housing-predictor-web=web_app.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "housing_predictor": [
            "config/*.yaml",
            "data/*.csv",
            "models/*.pkl",
        ],
    },
    zip_safe=False,
    keywords=[
        "machine-learning",
        "housing-prices",
        "real-estate",
        "prediction",
        "regression",
        "data-science",
        "zillow",
        "california-housing",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/housing-price-predictor/issues",
        "Source": "https://github.com/yourusername/housing-price-predictor",
        "Documentation": "https://github.com/yourusername/housing-price-predictor/blob/main/README.md",
    },
)