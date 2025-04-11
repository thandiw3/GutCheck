"""
Enhanced preprocessing module for microbiome data analysis.

This module provides specialized preprocessing techniques for microbiome data,
including compositional data analysis, zero handling, and normalization methods
specific to microbiome studies.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import skbio.diversity.alpha as alpha_diversity
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


class MicrobiomePreprocessor:
    """
    Class for preprocessing microbiome data with specialized techniques.
    """
    
    def __init__(self, zero_handling='pseudo', normalization='clr', 
                 scaling='standard', min_prevalence=0.1, min_abundance=0.0001):
        """
        Initialize the microbiome preprocessor.
        
        Parameters:
        -----------
        zero_handling : str
            Method for handling zeros: 'pseudo' (add pseudocount), 'impute' (KNN imputation),
            or 'none' (no zero handling).
        normalization : str
            Normalization method: 'clr' (centered log-ratio), 'relative' (relative abundance),
            'tss' (total sum scaling), or 'none' (no normalization).
        scaling : str
            Scaling method: 'standard', 'robust', 'power', or 'none'.
        min_prevalence : float
            Minimum prevalence (proportion of samples) for a feature to be retained.
        min_abundance : float
            Minimum mean abundance for a feature to be retained.
        """
        self.zero_handling = zero_handling
        self.normalization = normalization
        self.scaling = scaling
        self.min_prevalence = min_prevalence
        self.min_abundance = min_abundance
        self.scaler = None
        self.feature_names_ = None
        self.removed_features_ = None
        self.is_fitted = False
    
    def fit_transform(self, data, otu_pattern="OTU_", sample_id_col="SampleID"):
        """
        Fit the preprocessor to the data and transform it.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing OTU counts and associated metadata.
        otu_pattern : str
            Pattern to identify OTU columns.
        sample_id_col : str
            Name of the sample ID column.
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed data with OTU features (and sample IDs if present).
        """
        # Identify OTU columns
        otu_columns = [col for col in data.columns if otu_pattern in col]
        self.feature_names_ = otu_columns.copy()
        if not otu_columns:
            raise ValueError(f"No OTU columns found with pattern '{otu_pattern}'")
        
        # Extract OTU data and optionally store sample IDs
        otu_data = data[otu_columns].copy()
        sample_ids = data[sample_id_col].copy() if sample_id_col in data.columns else None
        
        # Filter features by prevalence and abundance
        otu_data = self._filter_features(otu_data)
        
        # Handle zeros
        otu_data = self._handle_zeros(otu_data)
        
        # Normalize data
        otu_data = self._normalize(otu_data)
        
        # Scale data
        otu_data = self._scale(otu_data)
        
        # Restore sample IDs if available
        if sample_ids is not None:
            otu_data[sample_id_col] = sample_ids
        
        self.is_fitted = True
        return otu_data
    
    def transform(self, data, otu_pattern="OTU_", sample_id_col="SampleID"):
        """
        Transform new data using the fitted preprocessor.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing OTU counts and metadata.
        otu_pattern : str
            Pattern to identify OTU columns.
        sample_id_col : str
            Name of the sample ID column.
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed data.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet.")
        
        otu_columns = [col for col in data.columns if otu_pattern in col]
        if not otu_columns:
            raise ValueError(f"No OTU columns found with pattern '{otu_pattern}'")
        
        otu_data = data[otu_columns].copy()
        sample_ids = data[sample_id_col].copy() if sample_id_col in data.columns else None
        
        # Keep only features that were retained during fitting
        common_features = [col for col in otu_columns if col in self.feature_names_]
        otu_data = otu_data[common_features]
        
        # Apply zero handling, normalization, and scaling using fitted parameters
        otu_data = self._handle_zeros(otu_data)
        otu_data = self._normalize(otu_data)
        if self.scaler is not None and self.scaling != 'none':
            scaled_values = self.scaler.transform(otu_data)
            otu_data = pd.DataFrame(scaled_values, columns=otu_data.columns, index=otu_data.index)
        
        if sample_ids is not None:
            otu_data[sample_id_col] = sample_ids
        
        return otu_data
    
    def _filter_features(self, otu_data):
        """
        Filter features based on prevalence and abundance.
        
        Parameters:
        -----------
        otu_data : pd.DataFrame
            DataFrame containing OTU counts.
            
        Returns:
        --------
        pd.DataFrame
            Filtered OTU data.
        """
        prevalence = (otu_data > 0).mean()
        mean_abundance = otu_data.mean()
        keep_features = (prevalence >= self.min_prevalence) & (mean_abundance >= self.min_abundance)
        
        self.removed_features_ = otu_data.columns[~keep_features].tolist()
        self.feature_names_ = otu_data.columns[keep_features].tolist()
        filtered_data = otu_data.loc[:, keep_features]
        
        print(f"Removed {len(self.removed_features_)} features with low prevalence or abundance.")
        print(f"Retained {len(self.feature_names_)} features.")
        return filtered_data
    
    def _handle_zeros(self, otu_data):
        """
        Handle zero values in the OTU data.
        
        Parameters:
        -----------
        otu_data : pd.DataFrame
            DataFrame containing OTU counts.
            
        Returns:
        --------
        pd.DataFrame
            OTU data after zero handling.
        """
        if self.zero_handling == 'pseudo':
            # Add a pseudocount (half of the minimum non-zero value)
            min_nonzero = otu_data[otu_data > 0].min().min()
            pseudocount = min_nonzero / 2
            return otu_data + pseudocount
        elif self.zero_handling == 'impute':
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            imputed_values = imputer.fit_transform(otu_data)
            return pd.DataFrame(imputed_values, columns=otu_data.columns, index=otu_data.index)
        else:  # 'none'
            return otu_data
    
    def _normalize(self, otu_data):
        """
        Normalize OTU data.
        
        Parameters:
        -----------
        otu_data : pd.DataFrame
            DataFrame containing OTU counts.
            
        Returns:
        --------
        pd.DataFrame
            Normalized data.
        """
        if self.normalization == 'clr':
            # Centered log-ratio transformation (requires strictly positive values)
            if (otu_data <= 0).any().any():
                raise ValueError("CLR transformation requires positive values. Consider using zero_handling='pseudo'.")
            log_data = np.log(otu_data)
            geom_mean = log_data.mean(axis=1)
            clr_data = log_data.subtract(geom_mean, axis=0)
            return clr_data
        elif self.normalization == 'relative':
            return otu_data.div(otu_data.sum(axis=1), axis=0)
        elif self.normalization == 'tss':
            return otu_data.div(otu_data.sum(axis=1), axis=0) * 1_000_000
        else:  # 'none'
            return otu_data
    
    def _scale(self, otu_data):
        """
        Scale the data.
        
        Parameters:
        -----------
        otu_data : pd.DataFrame
            DataFrame containing OTU data.
            
        Returns:
        --------
        pd.DataFrame
            Scaled data.
        """
        if self.scaling == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling == 'power':
            self.scaler = PowerTransformer(method='yeo-johnson')
        else:  # 'none'
            return otu_data
        
        scaled_values = self.scaler.fit_transform(otu_data)
        return pd.DataFrame(scaled_values, columns=otu_data.columns, index=otu_data.index)


def calculate_alpha_diversity(otu_data, metrics=None):
    """
    Calculate alpha diversity metrics for each sample.
    
    Parameters:
    -----------
    otu_data : pd.DataFrame
        DataFrame of OTU counts (rows: samples, columns: features).
    metrics : list, optional
        List of alpha diversity metrics to compute.
        Default metrics: ['shannon', 'simpson', 'observed_otus', 'chao1', 'pielou_e'].
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with computed alpha diversity metrics.
    """
    if metrics is None:
        metrics = ['shannon', 'simpson', 'observed_otus', 'chao1', 'pielou_e']
    
    results = {}
    for metric in metrics:
        if metric == 'shannon':
            results['Shannon'] = otu_data.apply(alpha_diversity.shannon, axis=1)
        elif metric == 'simpson':
            results['Simpson'] = otu_data.apply(alpha_diversity.simpson, axis=1)
        elif metric == 'observed_otus':
            results['Observed_OTUs'] = (otu_data > 0).sum(axis=1)
        elif metric == 'chao1':
            results['Chao1'] = otu_data.apply(alpha_diversity.chao1, axis=1)
        elif metric == 'pielou_e':
            shannon = otu_data.apply(alpha_diversity.shannon, axis=1)
            observed = (otu_data > 0).sum(axis=1)
            results['Pielou_E'] = shannon / np.log(observed)
    
    return pd.DataFrame(results, index=otu_data.index)


def detect_outliers(data, method='zscore', threshold=3):
    """
    Detect outliers in the data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with features.
    method : str
        Outlier detection method: 'zscore' or 'iqr'.
    threshold : float
        Threshold for determining outliers.
        
    Returns:
    --------
    pd.Series
        Boolean series indicating whether each sample is an outlier.
    """
    if method == 'zscore':
        z_scores = data.apply(zscore)
        return (z_scores.abs() > threshold).any(axis=1)
    elif method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        return ((data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))).any(axis=1)
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def plot_sample_distribution(otu_data, title="Sample Distribution", save_path=None):
    """
    Plot the distribution of samples based on total OTU counts.
    
    Parameters:
    -----------
    otu_data : pd.DataFrame
        DataFrame containing OTU counts.
    title : str
        Overall title for the plot.
    save_path : str, optional
        File path to save the plot.
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with histogram and box plot of total counts.
    """
    total_counts = otu_data.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(total_counts, kde=True)
    plt.title("Histogram of Total Counts")
    plt.xlabel("Total Counts")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=total_counts)
    plt.title("Box Plot of Total Counts")
    plt.ylabel("Total Counts")
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_feature_distribution(otu_data, top_n=20, log_scale=True, save_path=None):
    """
    Plot the distribution of the top features based on mean abundance.
    
    Parameters:
    -----------
    otu_data : pd.DataFrame
        DataFrame containing OTU counts.
    top_n : int
        Number of top features to display.
    log_scale : bool
        Whether to display the y-axis in log scale.
    save_path : str, optional
        File path to save the plot.
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with a boxplot of top features.
    """
    mean_abundance = otu_data.mean().sort_values(ascending=False)
    top_features = mean_abundance.head(top_n).index
    
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(data=otu_data[top_features])
    plt.title(f"Distribution of Top {top_n} Features")
    plt.xlabel("Feature")
    plt.ylabel("Abundance" + (" (log scale)" if log_scale else ""))
    
    if log_scale:
        plt.yscale('log')
    
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def perform_pca(data, n_components=2, plot=True, save_path=None):
    """
    Perform Principal Component Analysis (PCA) on the data and optionally plot the results.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing features.
    n_components : int
        Number of principal components.
    plot : bool
        Whether to generate a scatter plot of the first two principal components.
    save_path : str, optional
        File path to save the plot.
        
    Returns:
    --------
    dict
        Dictionary containing the PCA model, a DataFrame of PCA results, the explained variance,
        and (if plotted) the matplotlib figure.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=pca_result,
                          columns=[f'PC{i+1}' for i in range(n_components)],
                          index=data.index)
    explained_variance = pca.explained_variance_ratio_ * 100
    
    results = {
        'pca': pca,
        'pca_df': pca_df,
        'explained_variance': explained_variance
    }
    
    if plot and n_components >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
        plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
        plt.title('PCA of Microbiome Data')
        plt.grid(alpha=0.3)
        
        results['plot'] = plt.gcf()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return results
