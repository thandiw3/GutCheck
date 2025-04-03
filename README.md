# Microbiome BMI Classifier
This Python package is designed to classify individuals as Healthy or Obese based on microbiome data, specifically using OTU (Operational Taxonomic Unit) data along with associated BMI (Body Mass Index) labels. The tool integrates various data preprocessing, feature extraction, and machine learning models to provide a comprehensive solution for microbiome-based disease classification.


### What is it?

The **Microbiome BMI Classifier** is a Python tool designed to analyze microbiome data (which is the collection of microorganisms like bacteria in our body) and use it to predict whether someone is **Healthy** or **Obese** based on their **BMI (Body Mass Index)**. It does this by taking microbiome data, processing it, extracting useful information (features), and then using a machine learning model to classify people into two groups: **Healthy** or **Obese**.

### Steps:

1. **Generate Synthetic Data**:  
   The tool can create fake (simulated) microbiome data and metadata (like BMI) for testing. You don't need real data to start – it can create some for you.

2. **Preprocess Data**:  
   The tool will load and clean up the data so it’s in the right format. For example, it combines the microbiome data and BMI values into a single file that the machine learning model can work with.

3. **Extract Features**:  
   It looks at the microbiome data (called OTUs) and calculates important statistics (like the average, highest, and lowest values) for each sample. These statistics help the model understand the data better.

4. **Train a Model**:  
   Using the processed data, the tool trains a machine learning model to learn patterns that separate healthy people from obese ones based on their microbiome data.

5. **Evaluate the Model**:  
   After training, it tests the model to see how well it predicts health status. It checks things like **accuracy** (how often the model is correct), **precision**, and **recall**.

### Example:

1. The tool generates some fake microbiome data and BMI values (you can also use real data if you have it).
2. It combines this data into one file and extracts useful features (like the average count of bacteria).
3. It trains a machine learning model using this data to predict whether someone is **Healthy** or **Obese** based on their microbiome.
4. Finally, it tells you how well the model did by showing you numbers like accuracy and precision.

### Why is it useful?

This tool helps scientists and health researchers analyze how the **microbiome** (the bacteria living in our bodies) might be connected to obesity. It can be used to test hypotheses or analyze real data to see if the microbiome can predict whether someone is healthy or obese.

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

