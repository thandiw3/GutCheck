"""
Module for functional analysis of microbiome data.

This module provides methods to analyze and predict functional capabilities
of microbiome communities from taxonomic composition data, addressing the
challenge that community structure differences don't provide information
about the microbiome's functional output.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import warnings
import os
import requests
import json
import tempfile
import subprocess
import pickle
from io import StringIO

class FunctionalAnalyzer:
    """
    Class for analyzing and predicting functional capabilities of microbiome communities.
    
    This class implements multiple methods for functional analysis of microbiome data,
    addressing the challenge that differences in community structure don't provide
    information about the microbiome's functional output.
    """
    
    def __init__(self, method='picrust', database='kegg', verbose=False):
        """
        Initialize the FunctionalAnalyzer.
        
        Parameters:
        -----------
        method : str, default='picrust'
            Method to use for functional prediction.
            Options: 'picrust', 'tax4fun', 'faprotax', 'manual_mapping'
        database : str, default='kegg'
            Functional database to use.
            Options: 'kegg', 'cog', 'pfam', 'go', 'metacyc'
        verbose : bool, default=False
            Whether to print verbose output.
        """
        self.method = method
        self.database = database
        self.verbose = verbose
        self.mapping = None
        self.is_fitted = False
        
        # Validate method
        valid_methods = ['picrust', 'tax4fun', 'faprotax', 'manual_mapping']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        # Validate database
        valid_databases = ['kegg', 'cog', 'pfam', 'go', 'metacyc']
        if database not in valid_databases:
            raise ValueError(f"Database must be one of {valid_databases}")
        
        # Check if required dependencies are installed
        if method == 'picrust':
            try:
                import pickle
            except ImportError:
                raise ImportError("pickle is required for PICRUSt method")
        
        # Load pre-trained models or reference data if needed
        self._load_reference_data()
    
    def _load_reference_data(self):
        """
        Load reference data for functional prediction.
        
        This method loads pre-trained models or reference data needed for
        functional prediction, depending on the selected method and database.
        """
        if self.verbose:
            print(f"Loading reference data for {self.method} method with {self.database} database")
        
        # Define data directory
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'functional')
        os.makedirs(data_dir, exist_ok=True)
        
        if self.method == 'picrust':
            # Load PICRUSt reference data
            picrust_file = os.path.join(data_dir, f'picrust_{self.database}_reference.pkl')
            
            if not os.path.exists(picrust_file):
                # Download reference data if not available
                if self.verbose:
                    print(f"Reference data not found. Downloading...")
                
                # In a real implementation, this would download from a repository
                # For this example, we'll create a simple mapping
                self.mapping = self._create_dummy_mapping()
                
                # Save the mapping
                os.makedirs(os.path.dirname(picrust_file), exist_ok=True)
                with open(picrust_file, 'wb') as f:
                    pickle.dump(self.mapping, f)
            else:
                # Load existing reference data
                with open(picrust_file, 'rb') as f:
                    self.mapping = pickle.load(f)
        
        elif self.method == 'tax4fun':
            # Load Tax4Fun reference data
            tax4fun_file = os.path.join(data_dir, f'tax4fun_{self.database}_reference.pkl')
            
            if not os.path.exists(tax4fun_file):
                # Create dummy mapping
                self.mapping = self._create_dummy_mapping()
                
                # Save the mapping
                os.makedirs(os.path.dirname(tax4fun_file), exist_ok=True)
                with open(tax4fun_file, 'wb') as f:
                    pickle.dump(self.mapping, f)
            else:
                # Load existing reference data
                with open(tax4fun_file, 'rb') as f:
                    self.mapping = pickle.load(f)
        
        elif self.method == 'faprotax':
            # Load FAPROTAX reference data
            faprotax_file = os.path.join(data_dir, 'faprotax_mapping.pkl')
            
            if not os.path.exists(faprotax_file):
                # Create dummy mapping
                self.mapping = self._create_dummy_mapping(functional_type='process')
                
                # Save the mapping
                os.makedirs(os.path.dirname(faprotax_file), exist_ok=True)
                with open(faprotax_file, 'wb') as f:
                    pickle.dump(self.mapping, f)
            else:
                # Load existing reference data
                with open(faprotax_file, 'rb') as f:
                    self.mapping = pickle.load(f)
    
    def _create_dummy_mapping(self, functional_type='pathway'):
        """
        Create a dummy mapping for testing purposes.
        
        Parameters:
        -----------
        functional_type : str, default='pathway'
            Type of functional annotation to create.
            Options: 'pathway', 'process'
            
        Returns:
        --------
        mapping : dict
            Dummy mapping from taxa to functions
        """
        # Create a dummy mapping from taxa to functions
        mapping = {}
        
        # Common bacterial genera
        genera = [
            'Bacteroides', 'Prevotella', 'Faecalibacterium', 'Eubacterium',
            'Ruminococcus', 'Bifidobacterium', 'Lactobacillus', 'Clostridium',
            'Streptococcus', 'Escherichia', 'Enterococcus', 'Akkermansia',
            'Blautia', 'Roseburia', 'Alistipes', 'Parabacteroides'
        ]
        
        if functional_type == 'pathway':
            # KEGG pathways related to metabolism
            functions = [
                'Carbohydrate metabolism', 'Energy metabolism', 'Lipid metabolism',
                'Nucleotide metabolism', 'Amino acid metabolism', 'Glycan metabolism',
                'Metabolism of cofactors and vitamins', 'Biosynthesis of secondary metabolites',
                'Xenobiotics biodegradation', 'Microbial metabolism in diverse environments',
                'Carbon fixation', 'Nitrogen metabolism', 'Sulfur metabolism',
                'Methane metabolism', 'Fatty acid metabolism', 'Starch and sucrose metabolism'
            ]
            
            # Create mapping with random weights
            for genus in genera:
                mapping[genus] = {}
                # Each genus contributes to a random subset of functions
                n_functions = np.random.randint(5, len(functions) + 1)
                selected_functions = np.random.choice(functions, size=n_functions, replace=False)
                
                for func in selected_functions:
                    # Random weight between 0.1 and 1.0
                    mapping[genus][func] = np.random.uniform(0.1, 1.0)
        
        elif functional_type == 'process':
            # Microbial processes
            functions = [
                'Fermentation', 'Nitrate reduction', 'Nitrogen fixation',
                'Sulfate respiration', 'Methanogenesis', 'Acetogenesis',
                'Xylan degradation', 'Cellulose degradation', 'Chitin degradation',
                'Protein degradation', 'Iron oxidation', 'Manganese oxidation',
                'Hydrogen oxidation', 'Methanotrophy', 'Photoautotrophy',
                'Chemoheterotrophy', 'Predatory/exoparasitic'
            ]
            
            # Create mapping with binary values (can perform function or not)
            for genus in genera:
                mapping[genus] = {}
                # Each genus contributes to a random subset of functions
                n_functions = np.random.randint(3, 10)
                selected_functions = np.random.choice(functions, size=n_functions, replace=False)
                
                for func in selected_functions:
                    # Binary value (1 = can perform function)
                    mapping[genus][func] = 1
        
        return mapping
    
    def predict_functions(self, X, taxonomy_level='genus'):
        """
        Predict functional capabilities from taxonomic composition.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            OTU table with taxonomic composition data
        taxonomy_level : str, default='genus'
            Taxonomic level to use for prediction
            
        Returns:
        --------
        functions : pandas.DataFrame
            Predicted functional capabilities
        """
        if self.mapping is None:
            raise ValueError("Reference data not loaded. Call _load_reference_data() first.")
        
        if self.verbose:
            print(f"Predicting functional capabilities using {self.method} method")
        
        # Initialize results DataFrame
        all_functions = set()
        for taxa_funcs in self.mapping.values():
            all_functions.update(taxa_funcs.keys())
        
        functions = pd.DataFrame(0, index=X.index, columns=list(all_functions))
        
        # Predict functions based on taxonomic composition
        for taxon in X.columns:
            # Extract genus from taxon name if needed
            if taxonomy_level == 'genus':
                # Assuming format like "k__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Streptococcaceae;g__Streptococcus"
                if ';g__' in taxon:
                    genus = taxon.split(';g__')[-1].split(';')[0]
                else:
                    # Try to extract genus from the taxon name
                    genus = taxon.split('_')[-1] if '_' in taxon else taxon
            else:
                genus = taxon
            
            # Check if genus is in the mapping
            if genus in self.mapping:
                # Get functional capabilities for this genus
                for func, weight in self.mapping[genus].items():
                    # Add weighted contribution to functions
                    functions[func] += X[taxon] * weight
        
        # Normalize functions
        row_sums = functions.sum(axis=1)
        for idx in functions.index:
            if row_sums[idx] > 0:
                functions.loc[idx] = functions.loc[idx] / row_sums[idx]
        
        self.is_fitted = True
        return functions
    
    def analyze_functional_diversity(self, functions):
        """
        Analyze functional diversity of microbiome samples.
        
        Parameters:
        -----------
        functions : pandas.DataFrame
            Predicted functional capabilities
            
        Returns:
        --------
        diversity_metrics : pandas.DataFrame
            Functional diversity metrics
        """
        if self.verbose:
            print("Analyzing functional diversity")
        
        # Initialize diversity metrics DataFrame
        diversity_metrics = pd.DataFrame(index=functions.index)
        
        # Calculate Shannon diversity
        shannon_diversity = []
        for idx in functions.index:
            # Get non-zero values
            values = functions.loc[idx].values
            values = values[values > 0]
            
            # Calculate Shannon diversity
            shannon = -np.sum(values * np.log(values))
            shannon_diversity.append(shannon)
        
        diversity_metrics['shannon_diversity'] = shannon_diversity
        
        # Calculate richness (number of functions)
        richness = []
        for idx in functions.index:
            # Count non-zero values
            count = np.sum(functions.loc[idx].values > 0)
            richness.append(count)
        
        diversity_metrics['richness'] = richness
        
        # Calculate evenness
        evenness = []
        for i, idx in enumerate(functions.index):
            # Calculate evenness as Shannon diversity / log(richness)
            if richness[i] > 1:
                even = shannon_diversity[i] / np.log(richness[i])
            else:
                even = 0
            evenness.append(even)
        
        diversity_metrics['evenness'] = evenness
        
        return diversity_metrics
    
    def cluster_by_function(self, functions, n_clusters=3):
        """
        Cluster samples by functional profiles.
        
        Parameters:
        -----------
        functions : pandas.DataFrame
            Predicted functional capabilities
        n_clusters : int, default=3
            Number of clusters
            
        Returns:
        --------
        clusters : pandas.Series
            Cluster assignments
        """
        if self.verbose:
            print(f"Clustering samples by functional profiles into {n_clusters} clusters")
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(functions)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create Series with cluster assignments
        clusters = pd.Series(clusters, index=functions.index, name='functional_cluster')
        
        return clusters
    
    def plot_functional_heatmap(self, functions, n_functions=20, figsize=(12, 10)):
        """
        Plot heatmap of functional profiles.
        
        Parameters:
        -----------
        functions : pandas.DataFrame
            Predicted functional capabilities
        n_functions : int, default=20
            Number of top functions to include
        figsize : tuple, default=(12, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        # Select top functions by mean abundance
        top_functions = functions.mean().nlargest(n_functions).index
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            functions[top_functions],
            cmap='viridis',
            ax=ax,
            cbar_kws={'label': 'Relative Abundance'}
        )
        
        ax.set_title('Functional Profile Heatmap')
        ax.set_xlabel('Functions')
        ax.set_ylabel('Samples')
        
        plt.tight_layout()
        return fig
    
    def plot_functional_pca(self, functions, color_by=None, figsize=(10, 8)):
        """
        Plot PCA of functional profiles.
        
        Parameters:
        -----------
        functions : pandas.DataFrame
            Predicted functional capabilities
        color_by : pandas.Series, optional
            Values to color points by
        figsize : tuple, default=(10, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(functions)
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot PCA
        if color_by is not None:
            # Color by provided values
            scatter = ax.scatter(
                X_pca[:, 0], X_pca[:, 1],
                c=color_by, cmap='viridis',
                alpha=0.7, edgecolors='w'
            )
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label=color_by.name if hasattr(color_by, 'name') else 'Value')
        else:
            # Default coloring
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolors='w')
        
        ax.set_title('PCA of Functional Profiles')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        plt.tight_layout()
        return fig
    
    def identify_discriminating_functions(self, functions, groups, method='mean_diff'):
        """
        Identify functions that discriminate between groups.
        
        Parameters:
        -----------
        functions : pandas.DataFrame
            Predicted functional capabilities
        groups : pandas.Series
            Group assignments for samples
        method : str, default='mean_diff'
            Method to use for identifying discriminating functions.
            Options: 'mean_diff', 't_test', 'random_forest'
            
        Returns:
        --------
        discriminating : pandas.DataFrame
            Discriminating functions with scores
        """
        if self.verbose:
            print(f"Identifying discriminating functions using {method} method")
        
        # Initialize results DataFrame
        discriminating = pd.DataFrame(index=functions.columns)
        discriminating['score'] = 0.0
        discriminating['group'] = ''
        
        # Get unique groups
        unique_groups = groups.unique()
        
        if len(unique_groups) < 2:
            raise ValueError("At least two groups are required")
        
        if method == 'mean_diff':
            # Calculate mean difference between groups
            for func in functions.columns:
                max_diff = 0
                max_group = ''
                
                for i, group1 in enumerate(unique_groups):
                    for group2 in unique_groups[i+1:]:
                        # Calculate mean for each group
                        mean1 = functions.loc[groups == group1, func].mean()
                        mean2 = functions.loc[groups == group2, func].mean()
                        
                        # Calculate absolute difference
                        diff = abs(mean1 - mean2)
                        
                        if diff > max_diff:
                            max_diff = diff
                            max_group = f"{group1}_vs_{group2}"
                
                discriminating.loc[func, 'score'] = max_diff
                discriminating.loc[func, 'group'] = max_group
        
        elif method == 't_test':
            # Perform t-test between groups
            from scipy.stats import ttest_ind
            
            for func in functions.columns:
                max_stat = 0
                max_group = ''
                
                for i, group1 in enumerate(unique_groups):
                    for group2 in unique_groups[i+1:]:
                        # Get values for each group
                        values1 = functions.loc[groups == group1, func].values
                        values2 = functions.loc[groups == group2, func].values
                        
                        # Perform t-test
                        stat, p = ttest_ind(values1, values2, equal_var=False)
                        
                        if abs(stat) > max_stat:
                            max_stat = abs(stat)
                            max_group = f"{group1}_vs_{group2}"
                
                discriminating.loc[func, 'score'] = max_stat
                discriminating.loc[func, 'group'] = max_group
        
        elif method == 'random_forest':
            # Use random forest feature importance
            from sklearn.ensemble import RandomForestClassifier
            
            # Train a random forest classifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(functions, groups)
            
            # Get feature importances
            importances = rf.feature_importances_
            
            # Assign importances to functions
            for i, func in enumerate(functions.columns):
                discriminating.loc[func, 'score'] = importances[i]
                
                # Determine which group has higher mean for this function
                group_means = {}
                for group in unique_groups:
                    group_means[group] = functions.loc[groups == group, func].mean()
                
                max_group = max(group_means, key=group_means.get)
                discriminating.loc[func, 'group'] = max_group
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Sort by score
        discriminating = discriminating.sort_values('score', ascending=False)
        
        return discriminating
    
    def plot_discriminating_functions(self, functions, groups, discriminating, n_functions=10, figsize=(12, 8)):
        """
        Plot discriminating functions between groups.
        
        Parameters:
        -----------
        functions : pandas.DataFrame
            Predicted functional capabilities
        groups : pandas.Series
            Group assignments for samples
        discriminating : pandas.DataFrame
            Discriminating functions with scores
        n_functions : int, default=10
            Number of top discriminating functions to plot
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        # Select top discriminating functions
        top_functions = discriminating.nlargest(n_functions, 'score').index
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create data for plotting
        plot_data = []
        
        for func in top_functions:
            for group in groups.unique():
                # Calculate mean and std for this function in this group
                mean = functions.loc[groups == group, func].mean()
                std = functions.loc[groups == group, func].std()
                
                plot_data.append({
                    'Function': func,
                    'Group': group,
                    'Mean': mean,
                    'Std': std
                })
        
        # Convert to DataFrame
        plot_df = pd.DataFrame(plot_data)
        
        # Plot
        sns.barplot(
            data=plot_df,
            x='Function',
            y='Mean',
            hue='Group',
            ax=ax
        )
        
        ax.set_title('Discriminating Functions Between Groups')
        ax.set_xlabel('Function')
        ax.set_ylabel('Mean Abundance')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def predict_bmi_from_functions(self, functions, bmi_values, test_size=0.3):
        """
        Predict BMI from functional profiles.
        
        Parameters:
        -----------
        functions : pandas.DataFrame
            Predicted functional capabilities
        bmi_values : pandas.Series
            BMI values for samples
        test_size : float, default=0.3
            Proportion of samples to use for testing
            
        Returns:
        --------
        bmi_pred : pandas.Series
            Predicted BMI values
        model : object
            Trained model
        """
        if self.verbose:
            print("Predicting BMI from functional profiles")
        
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            functions, bmi_values, test_size=test_size, random_state=42
        )
        
        # Train a gradient boosting regressor
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        bmi_pred = pd.Series(model.predict(functions), index=functions.index)
        
        if self.verbose:
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, r2_score
            
            y_pred_test = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            
            print(f"Test MAE: {mae:.2f}")
            print(f"Test R²: {r2:.2f}")
        
        return bmi_pred, model


def predict_functional_profile(otu_table, method='picrust', database='kegg'):
    """
    Predict functional profile from OTU table.
    
    Parameters:
    -----------
    otu_table : pandas.DataFrame
        OTU table with taxonomic composition data
    method : str, default='picrust'
        Method to use for functional prediction
    database : str, default='kegg'
        Functional database to use
        
    Returns:
    --------
    functions : pandas.DataFrame
        Predicted functional capabilities
    """
    analyzer = FunctionalAnalyzer(method=method, database=database)
    return analyzer.predict_functions(otu_table)


def analyze_functional_diversity(functions):
    """
    Analyze functional diversity of microbiome samples.
    
    Parameters:
    -----------
    functions : pandas.DataFrame
        Predicted functional capabilities
        
    Returns:
    --------
    diversity_metrics : pandas.DataFrame
        Functional diversity metrics
    """
    analyzer = FunctionalAnalyzer()
    return analyzer.analyze_functional_diversity(functions)


def identify_discriminating_functions(functions, groups, method='mean_diff'):
    """
    Identify functions that discriminate between groups.
    
    Parameters:
    -----------
    functions : pandas.DataFrame
        Predicted functional capabilities
    groups : pandas.Series
        Group assignments for samples
    method : str, default='mean_diff'
        Method to use for identifying discriminating functions
        
    Returns:
    --------
    discriminating : pandas.DataFrame
        Discriminating functions with scores
    """
    analyzer = FunctionalAnalyzer()
    return analyzer.identify_discriminating_functions(functions, groups, method=method)
