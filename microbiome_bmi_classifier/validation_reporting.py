"""
Module for validation and reporting in microbiome analysis.

This module provides methods for validating microbiome analysis results
and generating comprehensive reports, addressing the challenge of
standardization and predetermined validation for clinical use.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import os
import json
import datetime
import warnings
from jinja2 import Template
import base64
from io import BytesIO
import pickle

class ValidationReporter:
    """
    Class for validating microbiome analysis results and generating reports.
    
    This class implements methods for validating analysis results and generating
    comprehensive reports, addressing the challenge of standardization and
    predetermined validation for clinical use.
    """
    
    def __init__(self, validation_type='classification', verbose=False):
        """
        Initialize the ValidationReporter.
        
        Parameters:
        -----------
        validation_type : str, default='classification'
            Type of validation to perform.
            Options: 'classification', 'regression', 'clustering', 'diversity'
        verbose : bool, default=False
            Whether to print verbose output
        """
        self.validation_type = validation_type
        self.verbose = verbose
        self.results = {}
        self.figures = {}
        self.metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'validation_type': validation_type
        }
        
        # Validate validation type
        valid_types = ['classification', 'regression', 'clustering', 'diversity']
        if validation_type not in valid_types:
            raise ValueError(f"Validation type must be one of {valid_types}")
    
    def validate_classification(self, y_true, y_pred, y_prob=None, 
                              class_names=None, cv=None, X=None, model=None):
        """
        Validate classification results.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_prob : array-like, optional
            Predicted probabilities
        class_names : list, optional
            Names of classes
        cv : int or cross-validation generator, optional
            Cross-validation strategy
        X : array-like, optional
            Feature matrix (required if cv is provided)
        model : object, optional
            Trained model (required if cv is provided)
            
        Returns:
        --------
        results : dict
            Dictionary with validation results
        """
        if self.validation_type != 'classification':
            warnings.warn(f"Validation type is {self.validation_type}, but validate_classification was called")
        
        if self.verbose:
            print("Validating classification results")
        
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Determine class names if not provided
        if class_names is None:
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            class_names = [f"Class_{i}" for i in unique_classes]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                'precision': precision_score(y_true, y_pred, average=None)[i] if i < len(precision_score(y_true, y_pred, average=None)) else np.nan,
                'recall': recall_score(y_true, y_pred, average=None)[i] if i < len(recall_score(y_true, y_pred, average=None)) else np.nan,
                'f1': f1_score(y_true, y_pred, average=None)[i] if i < len(f1_score(y_true, y_pred, average=None)) else np.nan
            }
        
        # Calculate ROC curve and AUC if probabilities are provided
        roc_data = None
        if y_prob is not None:
            # For binary classification
            if len(class_names) == 2:
                fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                roc_auc = auc(fpr, tpr)
                
                roc_data = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist(),
                    'auc': roc_auc
                }
            # For multi-class, calculate one-vs-rest ROC curves
            else:
                roc_data = {}
                for i, class_name in enumerate(class_names):
                    # Convert to one-vs-rest
                    y_true_binary = (y_true == i).astype(int)
                    y_prob_binary = y_prob[:, i] if y_prob.ndim > 1 else (y_pred == i).astype(float)
                    
                    fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_binary)
                    roc_auc = auc(fpr, tpr)
                    
                    roc_data[class_name] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': thresholds.tolist(),
                        'auc': roc_auc
                    }
        
        # Perform cross-validation if requested
        cv_results = None
        if cv is not None and X is not None and model is not None:
            if isinstance(cv, int):
                cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Get cross-validated predictions
            y_cv_pred = cross_val_predict(model, X, y_true, cv=cv)
            
            # Calculate cross-validated metrics
            cv_results = {
                'accuracy': accuracy_score(y_true, y_cv_pred),
                'precision': precision_score(y_true, y_cv_pred, average='weighted'),
                'recall': recall_score(y_true, y_cv_pred, average='weighted'),
                'f1': f1_score(y_true, y_cv_pred, average='weighted')
            }
        
        # Store results
        self.results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'per_class_metrics': per_class_metrics,
            'roc_data': roc_data,
            'cross_validation': cv_results
        }
        
        # Create figures
        self._create_classification_figures(y_true, y_pred, y_prob, class_names, cm)
        
        return self.results
    
    def validate_regression(self, y_true, y_pred, cv=None, X=None, model=None):
        """
        Validate regression results.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        cv : int or cross-validation generator, optional
            Cross-validation strategy
        X : array-like, optional
            Feature matrix (required if cv is provided)
        model : object, optional
            Trained model (required if cv is provided)
            
        Returns:
        --------
        results : dict
            Dictionary with validation results
        """
        if self.validation_type != 'regression':
            warnings.warn(f"Validation type is {self.validation_type}, but validate_regression was called")
        
        if self.verbose:
            print("Validating regression results")
        
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Perform cross-validation if requested
        cv_results = None
        if cv is not None and X is not None and model is not None:
            from sklearn.model_selection import cross_val_predict, KFold
            
            if isinstance(cv, int):
                cv = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Get cross-validated predictions
            y_cv_pred = cross_val_predict(model, X, y_true, cv=cv)
            
            # Calculate cross-validated metrics
            cv_results = {
                'mae': mean_absolute_error(y_true, y_cv_pred),
                'mse': mean_squared_error(y_true, y_cv_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_cv_pred)),
                'r2': r2_score(y_true, y_cv_pred)
            }
        
        # Store results
        self.results = {
            'metrics': metrics,
            'residuals': residuals.tolist(),
            'cross_validation': cv_results
        }
        
        # Create figures
        self._create_regression_figures(y_true, y_pred, residuals)
        
        return self.results
    
    def validate_clustering(self, X, labels, true_labels=None, 
                          feature_names=None, sample_ids=None):
        """
        Validate clustering results.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        labels : array-like
            Cluster labels
        true_labels : array-like, optional
            True labels for external validation
        feature_names : list, optional
            Names of features
        sample_ids : list, optional
            Sample IDs
            
        Returns:
        --------
        results : dict
            Dictionary with validation results
        """
        if self.validation_type != 'clustering':
            warnings.warn(f"Validation type is {self.validation_type}, but validate_clustering was called")
        
        if self.verbose:
            print("Validating clustering results")
        
        # Convert inputs to numpy arrays
        X = np.array(X)
        labels = np.array(labels)
        
        # Calculate internal validation metrics
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        internal_metrics = {
            'silhouette': silhouette_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels)
        }
        
        # Calculate external validation metrics if true labels are provided
        external_metrics = None
        if true_labels is not None:
            from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
            
            external_metrics = {
                'adjusted_rand': adjusted_rand_score(true_labels, labels),
                'adjusted_mutual_info': adjusted_mutual_info_score(true_labels, labels),
                'normalized_mutual_info': normalized_mutual_info_score(true_labels, labels)
            }
        
        # Calculate cluster statistics
        unique_labels = np.unique(labels)
        cluster_stats = {}
        
        for label in unique_labels:
            mask = (labels == label)
            cluster_size = np.sum(mask)
            cluster_mean = np.mean(X[mask], axis=0)
            cluster_std = np.std(X[mask], axis=0)
            
            cluster_stats[f"Cluster_{label}"] = {
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(labels) * 100),
                'mean': cluster_mean.tolist(),
                'std': cluster_std.tolist()
            }
        
        # Store results
        self.results = {
            'internal_metrics': internal_metrics,
            'external_metrics': external_metrics,
            'cluster_stats': cluster_stats,
            'n_clusters': len(unique_labels),
            'n_samples': len(labels)
        }
        
        # Create figures
        self._create_clustering_figures(X, labels, true_labels, feature_names, sample_ids)
        
        return self.results
    
    def validate_diversity(self, otu_table, metadata=None, group_col=None):
        """
        Validate diversity analysis results.
        
        Parameters:
        -----------
        otu_table : pandas.DataFrame
            OTU table with samples as rows and taxa as columns
        metadata : pandas.DataFrame, optional
            Sample metadata
        group_col : str, optional
            Column in metadata for grouping samples
            
        Returns:
        --------
        results : dict
            Dictionary with validation results
        """
        if self.validation_type != 'diversity':
            warnings.warn(f"Validation type is {self.validation_type}, but validate_diversity was called")
        
        if self.verbose:
            print("Validating diversity analysis results")
        
        # Calculate alpha diversity metrics
        from skbio.diversity import alpha
        
        alpha_diversity = {}
        
        # Shannon diversity
        shannon = []
        for idx in otu_table.index:
            values = otu_table.loc[idx].values
            values = values[values > 0]  # Remove zeros
            shannon.append(alpha.shannon(values))
        
        alpha_diversity['shannon'] = shannon
        
        # Simpson diversity
        simpson = []
        for idx in otu_table.index:
            values = otu_table.loc[idx].values
            values = values[values > 0]  # Remove zeros
            simpson.append(alpha.simpson(values))
        
        alpha_diversity['simpson'] = simpson
        
        # Observed OTUs
        observed_otus = []
        for idx in otu_table.index:
            values = otu_table.loc[idx].values
            observed_otus.append(np.sum(values > 0))
        
        alpha_diversity['observed_otus'] = observed_otus
        
        # Calculate beta diversity metrics
        from skbio.diversity import beta
        from scipy.spatial.distance import squareform
        
        # Bray-Curtis distance
        bc_distances = []
        for i in range(len(otu_table)):
            for j in range(i+1, len(otu_table)):
                bc = beta.braycurtis(otu_table.iloc[i].values, otu_table.iloc[j].values)
                bc_distances.append(bc)
        
        # Convert to distance matrix
        bc_matrix = squareform(bc_distances)
        
        # Calculate group comparisons if metadata and group_col are provided
        group_comparisons = None
        if metadata is not None and group_col is not None:
            # Ensure all samples in OTU table are in metadata
            common_samples = set(otu_table.index).intersection(set(metadata.index))
            if len(common_samples) < len(otu_table):
                warnings.warn(f"Only {len(common_samples)} of {len(otu_table)} samples found in metadata")
            
            # Filter OTU table and metadata to common samples
            otu_filtered = otu_table.loc[common_samples]
            meta_filtered = metadata.loc[common_samples]
            
            # Get groups
            groups =
(Content truncated due to size limit. Use line ranges to read in chunks)