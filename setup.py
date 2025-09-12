"""
Setup script for Housing Price Predictor
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="housing-price-predictor",
    version="1.0.0",
    author="AI Projects Series",
    author_email="your.email@example.com",
    description="Advanced housing price predictor using multiple ML algorithms and real-world datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/housing-price-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
            "notebook>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "housing-predictor=housing_predictor_advanced:main",
            "zillow-predictor=zillow_predictor:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
