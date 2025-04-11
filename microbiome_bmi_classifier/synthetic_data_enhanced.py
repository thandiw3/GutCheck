"""
Enhanced synthetic data generation module for microbiome-based BMI classification.

This module provides advanced methods for generating realistic synthetic microbiome data
based on actual distributions and patterns observed in real microbiome datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import os

class MicrobiomeSyntheticDataGenerator:
    """
    Class for generating realistic synthetic microbiome data.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the synthetic data generator.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Store generated data
        self.otu_data = None
        self.metadata = None
        self.feature_params = None
    
    def fit(self, real_otu_data=None, real_metadata=None):
        """
        Fit the generator to real data to learn distributions.
        
        Parameters:
        -----------
        real_otu_data : pd.DataFrame, optional
            Real OTU data to learn from
        real_metadata : pd.DataFrame, optional
            Real metadata to learn from
            
        Returns:
        --------
        self
            The fitted generator
        """
        # If real data is provided, learn from it
        if real_otu_data is not None:
            print("Learning from real OTU data...")
            self._learn_otu_distributions(real_otu_data)
        
        if real_metadata is not None:
            print("Learning from real metadata...")
            self._learn_metadata_distributions(real_metadata)
        
        return self
    
    def _learn_otu_distributions(self, real_otu_data):
        """
        Learn OTU abundance distributions from real data.
        
        Parameters:
        -----------
        real_otu_data : pd.DataFrame
            Real OTU data to learn from
        """
        # Store feature names
        self.feature_names = real_otu_data.columns.tolist()
        
        # Learn distribution parameters for each OTU
        self.feature_params = {}
        
        for feature in self.feature_names:
            # Get feature values
            values = real_otu_data[feature].values
            
            # Calculate basic statistics
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            # Calculate sparsity (proportion of zeros)
            sparsity = np.mean(values == 0)
            
            # Fit distribution to non-zero values
            non_zero = values[values > 0]
            
            if len(non_zero) > 10:  # Only fit if enough non-zero values
                # Try to fit gamma distribution (common for microbiome data)
                try:
                    shape, loc, scale = stats.gamma.fit(non_zero)
                    distribution = 'gamma'
                    params = {'shape': shape, 'loc': loc, 'scale': scale}
                except:
                    # Fallback to lognormal distribution
                    try:
                        shape, loc, scale = stats.lognorm.fit(non_zero)
                        distribution = 'lognorm'
                        params = {'s': shape, 'loc': loc, 'scale': scale}
                    except:
                        # Fallback to normal distribution
                        distribution = 'normal'
                        params = {'loc': mean, 'scale': std}
            else:
                # Not enough data, use normal distribution
                distribution = 'normal'
                params = {'loc': mean, 'scale': std}
            
            # Store parameters
            self.feature_params[feature] = {
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val,
                'sparsity': sparsity,
                'distribution': distribution,
                'params': params
            }
        
        # Learn correlations between features
        self._learn_feature_correlations(real_otu_data)
    
    def _learn_feature_correlations(self, real_otu_data):
        """
        Learn correlations between features.
        
        Parameters:
        -----------
        real_otu_data : pd.DataFrame
            Real OTU data to learn from
        """
        # Calculate correlation matrix
        self.correlation_matrix = real_otu_data.corr(method='spearman').values
        
        # Learn multivariate distribution using PCA and GMM
        # This helps capture complex relationships between features
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(real_otu_data)
        
        # Apply PCA to reduce dimensionality
        n_components = min(50, scaled_data.shape[1])  # Use at most 50 components
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Fit Gaussian Mixture Model to PCA result
        n_components_gmm = min(5, pca_result.shape[0] // 10)  # Use at most 5 components or 1/10 of samples
        gmm = GaussianMixture(
            n_components=max(2, n_components_gmm),
            covariance_type='full',
            random_state=self.random_state
        )
        gmm.fit(pca_result)
        
        # Store models for later use
        self.scaler = scaler
        self.pca = pca
        self.gmm = gmm
    
    def _learn_metadata_distributions(self, real_metadata):
        """
        Learn metadata distributions from real data.
        
        Parameters:
        -----------
        real_metadata : pd.DataFrame
            Real metadata to learn from
        """
        # Store metadata columns
        self.metadata_columns = real_metadata.columns.tolist()
        
        # Learn distribution parameters for each metadata column
        self.metadata_params = {}
        
        for column in self.metadata_columns:
            # Get column values
            values = real_metadata[column].values
            
            # Check if column is numeric
            if np.issubdtype(values.dtype, np.number):
                # Numeric column
                mean = np.mean(values)
                std = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                # Try to fit distribution
                try:
                    # Try normal distribution
                    loc, scale = stats.norm.fit(values)
                    distribution = 'normal'
                    params = {'loc': loc, 'scale': scale}
                    
                    # Check if better fit with other distributions
                    if column.lower() == 'bmi' or column.lower() == 'age':
                        # These often follow gamma or lognormal
                        try:
                            shape, loc, scale = stats.gamma.fit(values)
                            distribution = 'gamma'
                            params = {'shape': shape, 'loc': loc, 'scale': scale}
                        except:
                            pass
                except:
                    # Fallback to empirical distribution
                    distribution = 'empirical'
                    params = {'values': values}
                
                self.metadata_params[column] = {
                    'type': 'numeric',
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val,
                    'distribution': distribution,
                    'params': params
                }
            else:
                # Categorical column
                value_counts = pd.Series(values).value_counts(normalize=True)
                categories = value_counts.index.tolist()
                probabilities = value_counts.values
                
                self.metadata_params[column] = {
                    'type': 'categorical',
                    'categories': categories,
                    'probabilities': probabilities
                }
    
    def generate_data(self, n_samples=1000, n_features=None, class_balance=0.5, 
                     add_noise=True, noise_level=0.1, output_dir=None):
        """
        Generate synthetic microbiome data.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        n_features : int, optional
            Number of features to generate (default: use all learned features)
        class_balance : float
            Proportion of samples in the positive class (obese)
        add_noise : bool
            Whether to add noise to the generated data
        noise_level : float
            Level of noise to add
        output_dir : str, optional
            Directory to save generated data
            
        Returns:
        --------
        tuple
            (otu_data, metadata) DataFrames
        """
        print(f"Generating {n_samples} synthetic samples...")
        
        # Determine number of samples in each class
        n_obese = int(n_samples * class_balance)
        n_healthy = n_samples - n_obese
        
        # Generate data based on learned distributions if available
        if hasattr(self, 'gmm') and hasattr(self, 'pca') and hasattr(self, 'scaler'):
            # Generate data using learned multivariate distribution
            otu_data = self._generate_from_multivariate(n_samples, n_features)
        elif hasattr(self, 'feature_params'):
            # Generate data using learned univariate distributions
            otu_data = self._generate_from_univariate(n_samples, n_features)
        else:
            # Generate data using default distributions
            otu_data = self._generate_default(n_samples, n_features)
        
        # Generate metadata
        metadata = self._generate_metadata(n_samples, n_obese, n_healthy)
        
        # Add noise if requested
        if add_noise:
            otu_data = self._add_noise(otu_data, noise_level)
        
        # Save data if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            otu_path = os.path.join(output_dir, 'synthetic_otu_data.csv')
            meta_path = os.path.join(output_dir, 'synthetic_metadata.csv')
            
            otu_data.to_csv(otu_path)
            metadata.to_csv(meta_path)
            
            print(f"Saved synthetic data to {output_dir}")
        
        # Store generated data
        self.otu_data = otu_data
        self.metadata = metadata
        
        return otu_data, metadata
    
    def _generate_from_multivariate(self, n_samples, n_features=None):
        """
        Generate OTU data using learned multivariate distribution.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        n_features : int, optional
            Number of features to generate
            
        Returns:
        --------
        pd.DataFrame
            Generated OTU data
        """
        # Generate samples from GMM
        pca_samples, _ = self.gmm.sample(n_samples)
        
        # Transform back to original space
        original_space_samples = self.pca.inverse_transform(pca_samples)
        
        # Inverse transform scaling
        raw_samples = self.scaler.inverse_transform(original_space_samples)
        
        # Create DataFrame
        if n_features is None or n_features >= len(self.feature_names):
            # Use all features
            feature_names = self.feature_names
        else:
            # Select subset of features
            feature_names = self.feature_names[:n_features]
        
        # Create sample IDs
        sample_ids = [f"S{i+1}" for i in range(n_samples)]
        
        # Create DataFrame
        otu_data = pd.DataFrame(
            raw_samples[:, :len(feature_names)],
            columns=feature_names,
            index=sample_ids
        )
        
        # Ensure non-negative values
        otu_data[otu_data < 0] = 0
        
        # Apply sparsity pattern
        for feature in feature_names:
            if feature in self.feature_params:
                sparsity = self.feature_params[feature]['sparsity']
                
                # Generate mask for zeros
                zero_mask = np.random.random(n_samples) < sparsity
                
                # Apply mask
                otu_data.loc[zero_mask, feature] = 0
        
        return otu_data
    
    def _generate_from_univariate(self, n_samples, n_features=None):
        """
        Generate OTU data using learned univariate distributions.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        n_features : int, optional
            Number of features to generate
            
        Returns:
        --------
        pd.DataFrame
            Generated OTU data
        """
        # Determine features to generate
        if n_features is None or n_features >= len(self.feature_params):
            # Use all features
            features_to_generate = list(self.feature_params.keys())
        else:
            # Select subset of features
            features_to_generate = list(self.feature_params.keys())[:n_features]
        
        # Create sample IDs
        sample_ids = [f"S{i+1}" for i in range(n_samples)]
        
        # Initialize DataFrame
        otu_data = pd.DataFrame(index=sample_ids)
        
        # Generate data for each feature
        for feature in features_to_generate:
            params = self.feature_params[feature]
            
            # Generate sparsity pattern (zeros)
            zero_mask = np.random.random(n_samples) < params['sparsity']
            
            # Generate non-zero values
            if params['distribution'] == 'gamma':
                values = stats.gamma.rvs(
                    params['params']['shape'],
                    loc=params['params']['loc'],
                    scale=params['params']['scale'],
                    size=n_samples
                )
            elif params['distribution'] == 'lognorm':
                values = stats.lognorm.rvs(
                    params['params']['s'],
                    loc=params['params']['loc'],
                    scale=params['params']['scale'],
                    size=n_samples
                )
            else:
                # Normal distribution
                values = stats.norm.rvs(
                    loc=params['params']['loc'],
                    scale=params['params']['scale'],
                    size=n_samples
                )
            
            # Apply sparsity pattern
            values[zero_mask] = 0
            
            # Ensure non-negative values
            values[values < 0] = 0
            
            # Add to DataFrame
            otu_data[feature] = values
        
        # Apply correlation structure
        if hasattr(self, 'correlation_matrix') and self.correlation_matrix is not None:
            # This is a simplified approach to induce correlations
            # For a more accurate approach, we would need to use copulas or other methods
            
            # Get subset of correlation matrix for selected features
            feature_indices = [self.feature_names.index(f) for f in features_to_generate]
            sub_corr_matrix = self.correlation_matrix[np.ix_(feature_indices, feature_i
(Content truncated due to size limit. Use line ranges to read in chunks)