# Microbiome BMI Classifier
This Python package is designed to classify individuals as Healthy or Obese based on microbiome data, specifically using OTU (Operational Taxonomic Unit) data along with associated BMI (Body Mass Index) labels. The tool integrates various data preprocessing, feature extraction, and machine learning models to provide a comprehensive solution for microbiome-based disease classification.

## Features
Data Preprocessing: Load and preprocess microbiome data, merging OTU counts and BMI metadata.

Synthetic Data Generation: Create synthetic microbiome and metadata data for testing and experimentation.

Feature Extraction: Extract summary statistics (mean, std, min, max) from OTU data to use as features for classification.

Classification: Train a Random Forest classifier to predict whether a sample is Healthy or Obese based on its features.

Model Evaluation: Evaluate the model's performance using accuracy, precision, recall, and F1-score.

## Installation
To install the package and its dependencies, follow these steps:
git clone https://github.com/yourusername/microbiome-bmi-classifier.git
cd microbiome-bmi-classifier

