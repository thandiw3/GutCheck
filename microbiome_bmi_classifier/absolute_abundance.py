"""
Module for absolute abundance estimation in microbiome data.

This module provides methods to estimate absolute abundance from relative abundance
data using various techniques including spike-in controls, flow cytometry data integration,
and computational estimation methods.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
import warnings

class AbsoluteAbundanceEstimator:
    """
    Class for estimating absolute abundance from relative abundance data.
    
    This class implements multiple methods for converting relative abundance
    data to absolute abundance estimates, addressing a key limitation in
    microbiome analysis for clinical applications.
    """
    
    def __init__(self, method='total_sum_scaling', verbose=False):
        """
        Initialize the AbsoluteAbundanceEstimator.
        
        Parameters:
        -----------
        method : str, default='total_sum_scaling'
            Method to use for absolute abundance estimation.
            Options: 'total_sum_scaling', 'spike_in', 'flow_cytometry', 
                    'qpcr_calibration', 'computational'
        verbose : bool, default=False
            Whether to print verbose output.
        """
        self.method = method
        self.verbose = verbose
        self.calibration_model = None
        self.scaling_factors = None
        self.is_fitted = False
        
        # Validate method
        valid_methods = ['total_sum_scaling', 'spike_in', 'flow_cytometry', 
                         'qpcr_calibration', 'computational']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
    
    def fit(self, X, reference_data=None, total_counts=None, spike_in_cols=None, 
            qpcr_data=None, flow_cytometry_data=None, sample_ids=None):
        """
        Fit the estimator to the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Relative abundance data (OTU table)
        reference_data : pandas.DataFrame, optional
            Reference data for calibration
        total_counts : numpy.ndarray or list, optional
            Total microbial counts per sample (e.g., from qPCR)
        spike_in_cols : list, optional
            Column names of spike-in controls
        qpcr_data : pandas.DataFrame, optional
            qPCR data for specific taxa
        flow_cytometry_data : pandas.DataFrame, optional
            Flow cytometry data for total microbial counts
        sample_ids : list, optional
            Sample IDs to match between datasets
            
        Returns:
        --------
        self : AbsoluteAbundanceEstimator
            Fitted estimator
        """
        if self.verbose:
            print(f"Fitting absolute abundance estimator using {self.method} method")
        
        if self.method == 'total_sum_scaling':
            # Simple scaling based on assumed total microbial load
            if total_counts is None:
                # Default to 10^9 cells per sample (typical for gut microbiome)
                total_counts = np.ones(X.shape[0]) * 1e9
                warnings.warn("No total counts provided. Using default value of 10^9 cells per sample.")
            
            self.scaling_factors = total_counts
            
        elif self.method == 'spike_in':
            # Use spike-in controls to estimate absolute abundance
            if spike_in_cols is None:
                raise ValueError("spike_in_cols must be provided for spike_in method")
            
            # Extract spike-in columns
            spike_in_data = X[spike_in_cols]
            
            # Calculate scaling factors based on known spike-in concentrations
            # Assuming the first spike-in column is the reference
            self.scaling_factors = 1.0 / spike_in_data[spike_in_cols[0]].values
            
        elif self.method == 'flow_cytometry':
            # Use flow cytometry data to calibrate absolute abundance
            if flow_cytometry_data is None:
                raise ValueError("flow_cytometry_data must be provided for flow_cytometry method")
            
            if sample_ids is None:
                # Assume the indices match
                if X.shape[0] != flow_cytometry_data.shape[0]:
                    raise ValueError("X and flow_cytometry_data must have the same number of samples")
                self.scaling_factors = flow_cytometry_data.values.flatten()
            else:
                # Match samples by ID
                matched_counts = []
                for idx in X.index:
                    if idx in flow_cytometry_data.index:
                        matched_counts.append(flow_cytometry_data.loc[idx].values[0])
                    else:
                        raise ValueError(f"Sample {idx} not found in flow_cytometry_data")
                self.scaling_factors = np.array(matched_counts)
            
        elif self.method == 'qpcr_calibration':
            # Use qPCR data to calibrate absolute abundance
            if qpcr_data is None:
                raise ValueError("qpcr_data must be provided for qpcr_calibration method")
            
            # Build a calibration model using qPCR data for specific taxa
            # and relative abundance data
            X_cal = X.copy()
            
            # Match samples between X and qPCR data
            if sample_ids is not None:
                X_cal = X_cal.loc[sample_ids]
                qpcr_data = qpcr_data.loc[sample_ids]
            
            # Train a model to predict total abundance from relative abundance
            self.calibration_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.calibration_model.fit(X_cal, qpcr_data.values.flatten())
            
            # Predict scaling factors for all samples
            self.scaling_factors = self.calibration_model.predict(X)
            
        elif self.method == 'computational':
            # Use computational methods to estimate absolute abundance
            # This method uses correlations between taxa to infer absolute abundance
            
            # Calculate correlation matrix
            corr_matrix = X.corr()
            
            # Identify taxa that are likely to have constant absolute abundance
            # based on low correlation with other taxa
            mean_corr = corr_matrix.abs().mean()
            potential_invariant_taxa = mean_corr[mean_corr < 0.3].index.tolist()
            
            if len(potential_invariant_taxa) == 0:
                # If no invariant taxa found, use all taxa but with lower weight for highly correlated ones
                weights = 1 - mean_corr
                self.scaling_factors = np.array([(X[col] * weights[col]).sum() for col in X.columns])
            else:
                # Use invariant taxa as reference
                invariant_taxa = potential_invariant_taxa[:min(3, len(potential_invariant_taxa))]
                self.scaling_factors = X[invariant_taxa].mean(axis=1).values
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Transform relative abundance to absolute abundance.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Relative abundance data (OTU table)
            
        Returns:
        --------
        X_abs : pandas.DataFrame
            Absolute abundance estimates
        """
        if not self.is_fitted:
            raise ValueError("Estimator must be fitted before transform")
        
        if self.verbose:
            print("Transforming relative abundance to absolute abundance")
        
        # Create a copy of the input data
        X_abs = X.copy()
        
        # Apply scaling factors to each sample
        for i, (idx, row) in enumerate(X.iterrows()):
            X_abs.loc[idx] = row * self.scaling_factors[i]
        
        return X_abs
    
    def fit_transform(self, X, **kwargs):
        """
        Fit the estimator and transform the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Relative abundance data (OTU table)
        **kwargs : dict
            Additional arguments to pass to fit()
            
        Returns:
        --------
        X_abs : pandas.DataFrame
            Absolute abundance estimates
        """
        return self.fit(X, **kwargs).transform(X)
    
    def plot_comparison(self, X_rel, X_abs, n_taxa=10, figsize=(12, 8)):
        """
        Plot comparison between relative and absolute abundance.
        
        Parameters:
        -----------
        X_rel : pandas.DataFrame
            Relative abundance data
        X_abs : pandas.DataFrame
            Absolute abundance data
        n_taxa : int, default=10
            Number of top taxa to plot
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        # Select top taxa by mean abundance
        top_taxa = X_rel.mean().nlargest(n_taxa).index
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot relative abundance
        X_rel_top = X_rel[top_taxa]
        X_rel_top.plot(kind='bar', stacked=True, ax=axes[0], colormap='viridis')
        axes[0].set_title('Relative Abundance')
        axes[0].set_ylabel('Relative Abundance')
        axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Plot absolute abundance
        X_abs_top = X_abs[top_taxa]
        X_abs_top.plot(kind='bar', stacked=True, ax=axes[1], colormap='viridis')
        axes[1].set_title('Estimated Absolute Abundance')
        axes[1].set_ylabel('Absolute Abundance (cells/sample)')
        axes[1].set_xlabel('Samples')
        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        return fig
    
    def evaluate_with_ground_truth(self, X_rel, ground_truth, **kwargs):
        """
        Evaluate absolute abundance estimation against ground truth.
        
        Parameters:
        -----------
        X_rel : pandas.DataFrame
            Relative abundance data
        ground_truth : pandas.DataFrame
            Ground truth absolute abundance data
        **kwargs : dict
            Additional arguments to pass to fit()
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        # Fit and transform
        X_abs = self.fit_transform(X_rel, **kwargs)
        
        # Calculate metrics
        metrics = {}
        
        # Calculate correlation between estimated and ground truth
        correlations = []
        for col in X_abs.columns:
            if col in ground_truth.columns:
                corr = stats.spearmanr(X_abs[col], ground_truth[col])[0]
                correlations.append(corr)
        
        metrics['mean_correlation'] = np.mean(correlations)
        metrics['median_correlation'] = np.median(correlations)
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(X_abs.values - ground_truth.values))
        metrics['mae'] = mae
        
        # Calculate root mean squared error
        rmse = np.sqrt(np.mean((X_abs.values - ground_truth.values) ** 2))
        metrics['rmse'] = rmse
        
        return metrics


def estimate_absolute_abundance(otu_table, method='total_sum_scaling', **kwargs):
    """
    Estimate absolute abundance from relative abundance data.
    
    Parameters:
    -----------
    otu_table : pandas.DataFrame
        OTU table with relative abundance data
    method : str, default='total_sum_scaling'
        Method to use for absolute abundance estimation
    **kwargs : dict
        Additional arguments to pass to AbsoluteAbundanceEstimator.fit()
        
    Returns:
    --------
    absolute_abundance : pandas.DataFrame
        Estimated absolute abundance
    """
    estimator = AbsoluteAbundanceEstimator(method=method)
    return estimator.fit_transform(otu_table, **kwargs)


def simulate_spike_in_data(otu_table, n_spike_ins=3, concentration_range=(1e3, 1e6)):
    """
    Simulate spike-in control data for absolute abundance estimation.
    
    Parameters:
    -----------
    otu_table : pandas.DataFrame
        OTU table with relative abundance data
    n_spike_ins : int, default=3
        Number of spike-in controls to simulate
    concentration_range : tuple, default=(1e3, 1e6)
        Range of spike-in concentrations
        
    Returns:
    --------
    otu_table_with_spike_ins : pandas.DataFrame
        OTU table with added spike-in controls
    spike_in_cols : list
        Names of spike-in control columns
    """
    # Create a copy of the input data
    otu_table_with_spike_ins = otu_table.copy()
    
    # Generate spike-in concentrations
    spike_in_cols = []
    for i in range(n_spike_ins):
        col_name = f"SPIKE_IN_{i+1}"
        spike_in_cols.append(col_name)
        
        # Generate random concentrations within the specified range
        concentrations = np.random.uniform(
            low=concentration_range[0],
            high=concentration_range[1],
            size=otu_table.shape[0]
        )
        
        # Add to the OTU table
        otu_table_with_spike_ins[col_name] = concentrations
    
    # Renormalize to maintain relative abundance
    row_sums = otu_table_with_spike_ins.sum(axis=1)
    for idx in otu_table_with_spike_ins.index:
        otu_table_with_spike_ins.loc[idx] = otu_table_with_spike_ins.loc[idx] / row_sums[idx]
    
    return otu_table_with_spike_ins, spike_in_cols


def simulate_qpcr_data(otu_table, n_taxa=5, noise_level=0.1):
    """
    Simulate qPCR data for specific taxa.
    
    Parameters:
    -----------
    otu_table : pandas.DataFrame
        OTU table with relative abundance data
    n_taxa : int, default=5
        Number of taxa to simulate qPCR data for
    noise_level : float, default=0.1
        Level of noise to add to the simulated data
        
    Returns:
    --------
    qpcr_data : pandas.DataFrame
        Simulated qPCR data
    selected_taxa : list
        Names of taxa with qPCR data
    """
    # Select taxa with highest mean abundance
    selected_taxa = otu_table.mean().nlargest(n_taxa).index.tolist()
    
    # Create qPCR data frame
    qpcr_data = pd.DataFrame(index=otu_table.index)
    
    # Simulate total microbial load (cells/sample)
    total_load = np.random.uniform(1e8, 1e10, size=otu_table.shape[0])
    
    # Calculate absolute abundance for selected taxa
    for taxon in selected_taxa:
        # Convert relative to absolute
        absolute = otu_table[taxon] * total_load
        
        # Add noise
        noise = np.random.normal(0, noise_level * absolute.mean(), size=len(absolute))
        absolute_with_noise = absolute + noise
        
        # Ensure non-negative values
        absolute_with_noise = np.maximum(0, absolute_with_noise)
        
        # Add to qPCR data
        qpcr_data[taxon] = absolute_with_noise
    
    return qpcr_data, selected_taxa


def simulate_flow_cytometry_data(otu_table, mean_load=1e9, std_dev=1e8):
    """
    Simulate flow cytometry data for total microbial load.
    
    Parameters:
    -----------
    otu_table : pandas.DataFrame
        OTU table with relative abundance data
    mean_load : float, default=1e9
        Mean total microbial load (cells/sample)
    std_dev : float, default=1e8
        Standard deviation of total microbial load
        
    Returns:
    --------
    flow_data : pandas.DataFrame
        Simulated flow cytometry data
    """
    # Create flow cytometry data frame
    flow_data = pd.DataFrame(index=otu_table.index, columns=['total_count'])
    
    # Simulate total microbial load (cells/sample)
    total_load = np.random.normal(mean_load, std_dev, size=otu_table.shape[0])
    
    # Ensure non-negative values
    total_load = np.maximum(0, total_load)
    
    # Add to flow cytometry data
    flow_data['total_count'] = total_load
    
    return flow_data
