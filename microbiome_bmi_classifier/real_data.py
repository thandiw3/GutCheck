"""
Real data examples module for microbiome-based BMI classification.

This module provides functionality to download, process, and analyze real microbiome datasets
for demonstrating the performance of the GutCheck algorithm on actual data.
"""

import os
import pandas as pd
import numpy as np
import requests
import tarfile
import zipfile
import gzip
import shutil
from io import BytesIO
import biom
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class MicrobiomeDataLoader:
    """
    Class for downloading and loading real microbiome datasets.
    """
    
    def __init__(self, data_dir='./data'):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        data_dir : str
            Directory to store downloaded data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Create subdirectories
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def download_american_gut_data(self, n_samples=1000):
        """
        Download data from the American Gut Project.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to download
            
        Returns:
        --------
        dict
            Dictionary with paths to downloaded files
        """
        print(f"Downloading American Gut Project data (n={n_samples})...")
        
        # Create directory for American Gut data
        ag_dir = os.path.join(self.raw_dir, 'american_gut')
        os.makedirs(ag_dir, exist_ok=True)
        
        # URLs for American Gut data
        otu_url = "ftp://ftp.microbio.me/AmericanGut/latest/gg-13-8-otus/1_American_Gut_Project.biom"
        metadata_url = "ftp://ftp.microbio.me/AmericanGut/latest/AG_100nt_even10k-md.txt"
        
        # Download OTU table
        otu_path = os.path.join(ag_dir, 'american_gut_otus.biom')
        if not os.path.exists(otu_path):
            print("Downloading OTU table...")
            response = requests.get(otu_url)
            with open(otu_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded OTU table to {otu_path}")
        else:
            print(f"OTU table already exists at {otu_path}")
        
        # Download metadata
        metadata_path = os.path.join(ag_dir, 'american_gut_metadata.txt')
        if not os.path.exists(metadata_path):
            print("Downloading metadata...")
            response = requests.get(metadata_url)
            with open(metadata_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded metadata to {metadata_path}")
        else:
            print(f"Metadata already exists at {metadata_path}")
        
        return {
            'otu_path': otu_path,
            'metadata_path': metadata_path,
            'data_dir': ag_dir
        }
    
    def download_hmp_data(self):
        """
        Download data from the Human Microbiome Project.
        
        Returns:
        --------
        dict
            Dictionary with paths to downloaded files
        """
        print("Downloading Human Microbiome Project data...")
        
        # Create directory for HMP data
        hmp_dir = os.path.join(self.raw_dir, 'hmp')
        os.makedirs(hmp_dir, exist_ok=True)
        
        # URLs for HMP data
        otu_url = "https://www.hmpdacc.org/hmp/HMQCP/all/otu_table_psn_v35.txt.gz"
        metadata_url = "https://www.hmpdacc.org/hmp/HMQCP/all/v35_map_uniquebyPSN.txt.bz2"
        
        # Download OTU table
        otu_path = os.path.join(hmp_dir, 'hmp_otus.txt.gz')
        if not os.path.exists(otu_path):
            print("Downloading OTU table...")
            response = requests.get(otu_url)
            with open(otu_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded OTU table to {otu_path}")
        else:
            print(f"OTU table already exists at {otu_path}")
        
        # Download metadata
        metadata_path = os.path.join(hmp_dir, 'hmp_metadata.txt.bz2')
        if not os.path.exists(metadata_path):
            print("Downloading metadata...")
            response = requests.get(metadata_url)
            with open(metadata_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded metadata to {metadata_path}")
        else:
            print(f"Metadata already exists at {metadata_path}")
        
        return {
            'otu_path': otu_path,
            'metadata_path': metadata_path,
            'data_dir': hmp_dir
        }
    
    def download_qiita_obesity_data(self):
        """
        Download obesity-related microbiome data from Qiita.
        
        Returns:
        --------
        dict
            Dictionary with paths to downloaded files
        """
        print("Downloading obesity-related microbiome data from Qiita...")
        
        # Create directory for Qiita data
        qiita_dir = os.path.join(self.raw_dir, 'qiita_obesity')
        os.makedirs(qiita_dir, exist_ok=True)
        
        # URLs for Qiita obesity data (study ID: 10317)
        otu_url = "https://qiita.ucsd.edu/public_download/?data=10317_processed_16S_data.biom"
        metadata_url = "https://qiita.ucsd.edu/public_download/?data=10317_20181010-080413.txt"
        
        # Download OTU table
        otu_path = os.path.join(qiita_dir, 'qiita_obesity_otus.biom')
        if not os.path.exists(otu_path):
            print("Downloading OTU table...")
            response = requests.get(otu_url)
            with open(otu_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded OTU table to {otu_path}")
        else:
            print(f"OTU table already exists at {otu_path}")
        
        # Download metadata
        metadata_path = os.path.join(qiita_dir, 'qiita_obesity_metadata.txt')
        if not os.path.exists(metadata_path):
            print("Downloading metadata...")
            response = requests.get(metadata_url)
            with open(metadata_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded metadata to {metadata_path}")
        else:
            print(f"Metadata already exists at {metadata_path}")
        
        return {
            'otu_path': otu_path,
            'metadata_path': metadata_path,
            'data_dir': qiita_dir
        }
    
    def load_american_gut_data(self, processed=True, filter_bmi=True, min_bmi=18.5, max_bmi=40):
        """
        Load and process American Gut Project data.
        
        Parameters:
        -----------
        processed : bool
            Whether to return processed data or raw data
        filter_bmi : bool
            Whether to filter samples by BMI
        min_bmi : float
            Minimum BMI for filtering
        max_bmi : float
            Maximum BMI for filtering
            
        Returns:
        --------
        tuple
            (otu_data, metadata) DataFrames
        """
        # Check if processed data already exists
        processed_otu_path = os.path.join(self.processed_dir, 'american_gut_otus_processed.csv')
        processed_meta_path = os.path.join(self.processed_dir, 'american_gut_metadata_processed.csv')
        
        if processed and os.path.exists(processed_otu_path) and os.path.exists(processed_meta_path):
            print("Loading processed American Gut data...")
            otu_data = pd.read_csv(processed_otu_path, index_col=0)
            metadata = pd.read_csv(processed_meta_path, index_col=0)
            
            # Filter by BMI if requested
            if filter_bmi:
                valid_samples = metadata[(metadata['BMI'] >= min_bmi) & (metadata['BMI'] <= max_bmi)].index
                otu_data = otu_data.loc[otu_data.index.isin(valid_samples)]
                metadata = metadata.loc[metadata.index.isin(otu_data.index)]
                
                print(f"Filtered to {len(otu_data)} samples with BMI between {min_bmi} and {max_bmi}")
            
            return otu_data, metadata
        
        # Download data if not already downloaded
        files = self.download_american_gut_data()
        
        # Load OTU table
        print("Loading OTU table...")
        otu_table = biom.load_table(files['otu_path'])
        
        # Convert to pandas DataFrame
        otu_data = pd.DataFrame(
            np.array(otu_table.matrix_data.todense()),
            index=otu_table.ids('sample'),
            columns=otu_table.ids('observation')
        )
        
        # Load metadata
        print("Loading metadata...")
        metadata = pd.read_csv(files['metadata_path'], sep='\t', index_col=0)
        
        # Process data
        if processed:
            print("Processing data...")
            
            # Rename OTU columns to standard format
            otu_data.columns = [f"OTU_{i+1}" for i in range(len(otu_data.columns))]
            
            # Extract BMI from metadata
            if 'BMI' in metadata.columns:
                # BMI column already exists
                pass
            elif 'HEIGHT_CM' in metadata.columns and 'WEIGHT_KG' in metadata.columns:
                # Calculate BMI from height and weight
                metadata['BMI'] = metadata['WEIGHT_KG'] / ((metadata['HEIGHT_CM'] / 100) ** 2)
            else:
                # No BMI information available
                print("Warning: No BMI information found in metadata")
                metadata['BMI'] = np.nan
            
            # Create binary label based on BMI
            metadata['Label'] = np.where(metadata['BMI'] >= 30, 1, 0)  # 1 for obese (BMI >= 30), 0 for healthy
            
            # Filter samples with valid BMI
            if filter_bmi:
                valid_samples = metadata[(metadata['BMI'] >= min_bmi) & (metadata['BMI'] <= max_bmi)].index
                otu_data = otu_data.loc[otu_data.index.isin(valid_samples)]
                metadata = metadata.loc[metadata.index.isin(otu_data.index)]
                
                print(f"Filtered to {len(otu_data)} samples with BMI between {min_bmi} and {max_bmi}")
            
            # Ensure OTU data and metadata have the same samples
            common_samples = set(otu_data.index).intersection(set(metadata.index))
            otu_data = otu_data.loc[list(common_samples)]
            metadata = metadata.loc[list(common_samples)]
            
            print(f"Final dataset: {len(otu_data)} samples, {len(otu_data.columns)} OTUs")
            
            # Save processed data
            otu_data.to_csv(processed_otu_path)
            metadata.to_csv(processed_meta_path)
            
            print(f"Saved processed data to {self.processed_dir}")
        
        return otu_data, metadata
    
    def load_synthetic_obesity_data(self, n_samples=1000, n_features=100, random_state=42):
        """
        Generate synthetic obesity microbiome data.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        n_features : int
            Number of OTU features
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (otu_data, metadata) DataFrames
        """
        print(f"Generating synthetic obesity microbiome data (n={n_samples})...")
        
        # Set random seed
        np.random.seed(random_state)
        
        # Generate OTU data
        # We'll create two different distributions for healthy and obese samples
        n_healthy = n_samples // 2
        n_obese = n_samples - n_healthy
        
        # Create sample IDs
        sample_ids = [f"S{i+1}" for i in range(n_samples)]
        
        # Generate OTU data with different distributions for healthy and obese
        healthy_otus = np.random.gamma(shape=0.5, scale=2.0, size=(n_healthy, n_features))
        obese_otus = np.random.gamma(shape=0.3, scale=3.0, size=(n_obese, n_features))
        
        # Combine data
        otu_values = np.vstack([healthy_otus, obese_otus])
        
        # Create OTU DataFrame
        otu_data = pd.DataFrame(
            otu_values,
            index=sample_ids,
            columns=[f"OTU_{i+1}" for i in range(n_features)]
        )
        
        # Generate metadata
        # BMI distribution: healthy (18.5-24.9), overweight (25-29.9), obese (30+)
        healthy_bmi = np.random.uniform(18.5, 24.9, size=n_healthy)
        obese_bmi = np.random.uniform(30.0, 40.0, size=n_obese)
        bmi_values = np.concatenate([healthy_bmi, obese_bmi])
        
        # Create labels
        labels = np.concatenate([np.zeros(n_healthy), np.ones(n_obese)])
        
        # Create metadata DataFrame
        metadata = pd.DataFrame({
            'BMI': bmi_values,
            'Label': labels
        }, index=sample_ids)
        
        # Add some additional metadata fields
        # Age
        metadata['Age'] = np.random.normal(45, 15, size=n_samples).astype(int)
        metadata.loc[metadata['Age'] < 18, 'Age'] = 18  # Minimum age
        metadata.loc[metadata['Age'] > 90, 'Age'] = 90  # Maximum age
        
        # Gender
        metadata['Gender'] = np.random.choice(['Male', 'Female'], size=n_samples)
        
        # Diet type
        diet_types = ['Omnivore', 'Vegetarian', 'Vegan', 'Pescatarian', 'Paleo', 'Keto']
        # Make diet distribution different between healthy and obese
        healthy_diets = np.random.choice(diet_types, size=n_healthy, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
        obese_diets = np.random.choice(diet_types, size=n_obese, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05])
        metadata['Diet'] = np.concatenate([healthy_diets, obese_diets])
        
        # Save synthetic data
        synthetic_dir = os.path.join(self.processed_dir, 'synthetic')
        os.makedirs(synthetic_dir, exist_ok=True)
        
        otu_path = os.path.join(synthetic_dir, 'synthetic_obesity_otus.csv')
        meta_path = os.path.join(synthetic_dir, 'synthetic_obesity_metadata.csv')
        
        otu_data.to_csv(otu_path)
        metadata.to_csv(meta_path)
        
        print(f"Saved synthetic data to {synthetic_dir}")
        print(f"Generated {n_samples} samples ({n_healthy} healthy, {n_obese} obese) with {n_features} OTUs")
        
        return otu_data, metadata
    
    def prepare_data_for_classification(self, otu_data, metadata, test_size=0.2, random_state=42):
        """
        Prepare data for classification by splitting into training and testing sets.
        
        Parameters:
        -----------
        otu_data : pd.DataFrame
            OTU data
        metadata : pd.DataFrame
            Metadata with BMI and Label columns
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Dictionary with X_train, X_test, y_train, y_test, and metadata
        """
        # Ensure OTU data and metadata have the same samples
        common_samples = set(otu_data.index).intersection(set(metadata.index))
        otu_data = otu_data.loc[list(common_samples)]
        metadata = metadata.loc[list(common_samples)]
        
        # Extract features and labels
        X = otu_data
        y = metadata['Label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Get metadata for train and test sets
        meta_train = metadata.loc[X_train.index]
        meta_test = metadata
(Content truncated due to size limit. Use line ranges to read in chunks)