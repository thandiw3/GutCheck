"""
Package setup for GutCheck microbiome-based BMI classification.

This module provides the setup configuration for the enhanced GutCheck package,
including all dependencies required for the new features.
"""

from setuptools import setup, find_packages

setup(
    name="gutcheck",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "joblib>=1.0.0",
        "flask>=2.0.0",
        "werkzeug>=2.0.0",
        "optuna>=2.10.0",
        "shap>=0.40.0",
        "lime>=0.2.0",
        "eli5>=0.11.0",
        "skbio>=0.5.6",
        "statsmodels>=0.13.0",
        "plotly>=5.3.0",
        "requests>=2.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
        "web": [
            "flask-cors>=3.0.10",
            "gunicorn>=20.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gutcheck=microbiome_bmi_classifier.cli:main",
        ],
    },
    author="GutCheck Team",
    author_email="gutcheck@example.com",
    description="A comprehensive tool for microbiome-based BMI classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Gouri117/GutCheck",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
