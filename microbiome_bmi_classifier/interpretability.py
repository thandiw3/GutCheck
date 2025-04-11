"""
Model interpretability module for microbiome-based BMI classification.

This module provides methods to explain model predictions and understand feature importance
using various interpretability techniques such as SHAP, LIME, feature importance analysis,
partial dependence plots, and permutation importance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, plot_partial_dependence
from sklearn.ensemble import RandomForestClassifier
import os
import joblib

# Try to import optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import eli5
    from eli5.sklearn import PermutationImportance
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False


class ModelInterpreter:
    """
    Class for interpreting machine learning models for microbiome data.
    """
    
    def __init__(self, model=None, feature_names=None, class_names=None, output_dir=None):
        """
        Initialize the model interpreter.
        
        Parameters:
        -----------
        model : estimator, optional
            Trained model to interpret.
        feature_names : list, optional
            List of feature names.
        class_names : list, optional
            List of class names.
        output_dir : str, optional
            Directory to save interpretation results.
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.output_dir = output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize explainers for SHAP and LIME
        self.shap_explainer = None
        self.lime_explainer = None
    
    def set_model(self, model, feature_names=None, class_names=None):
        """
        Set the model to interpret.
        
        Parameters:
        -----------
        model : estimator
            Trained model to interpret.
        feature_names : list, optional
            List of feature names.
        class_names : list, optional
            List of class names.
            
        Returns:
        --------
        self
            The model interpreter.
        """
        self.model = model
        if feature_names is not None:
            self.feature_names = feature_names
        if class_names is not None:
            self.class_names = class_names
        
        # Reset explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        return self
    
    def explain_feature_importance(self, X, top_n=20, figsize=(12, 10), save_path=None):
        """
        Explain feature importance using the model's built-in feature importance.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix (used to extract feature names if not provided).
        top_n : int
            Number of top features to show.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save the plot.
            
        Returns:
        --------
        tuple
            (matplotlib.figure.Figure, pd.DataFrame) with the plot and importance values.
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Obtain feature names if not already provided
        if self.feature_names is None:
            if hasattr(X, 'columns'):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Retrieve feature importances from the model
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
            if importances.ndim > 1:
                importances = np.mean(importances, axis=0)
        elif hasattr(self.model, 'estimators_') and hasattr(self.model.estimators_[0], 'feature_importances_'):
            importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
        else:
            raise ValueError("Model does not provide built-in feature importances. Consider using explain_permutation_importance().")
        
        # Build a DataFrame for feature importances
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        top_importance_df = importance_df.head(top_n)
        
        # Plot feature importances using Seaborn
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=top_importance_df, ax=ax)
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        # Save the figure if a path is provided or if an output directory is set
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance plot to {save_path}")
        elif self.output_dir:
            default_path = os.path.join(self.output_dir, 'feature_importance.png')
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature importance plot to {default_path}")
        
        return fig, importance_df
    
    def explain_permutation_importance(self, X, y, n_repeats=10, random_state=42, 
                                      top_n=20, figsize=(12, 10), save_path=None):
        """
        Explain feature importance using permutation importance.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix.
        y : array-like
            Target labels.
        n_repeats : int
            Number of times to permute each feature.
        random_state : int
            Random seed for reproducibility.
        top_n : int
            Number of top features to display.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save the plot.
            
        Returns:
        --------
        tuple
            (matplotlib.figure.Figure, pd.DataFrame) with the plot and permutation importance values.
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Obtain feature names if not already provided
        if self.feature_names is None:
            if hasattr(X, 'columns'):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Compute permutation importance
        perm_importance = permutation_importance(
            self.model, X, y, 
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        top_importance_df = importance_df.head(top_n)
        
        # Plot with error bars
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=top_importance_df,
                    xerr=top_importance_df['Std'], ax=ax)
        ax.set_title(f'Top {top_n} Features by Permutation Importance')
        ax.set_xlabel('Mean Decrease in Accuracy')
        ax.set_ylabel('Feature')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved permutation importance plot to {save_path}")
        elif self.output_dir:
            default_path = os.path.join(self.output_dir, 'permutation_importance.png')
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"Saved permutation importance plot to {default_path}")
        
        return fig, importance_df
    
    def explain_with_shap(self, X, y=None, sample_idx=None, n_samples=100, 
                         plot_type='summary', figsize=(12, 10), save_path=None):
        """
        Explain model predictions using SHAP values.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix.
        y : array-like, optional
            Target labels (if needed for background sampling).
        sample_idx : int, optional
            Index of sample to explain for single-instance plots.
        n_samples : int
            Number of samples to use as the background distribution.
        plot_type : str
            Type of SHAP plot to generate:
                - 'summary': Summary plot.
                - 'bar': Bar plot.
                - 'waterfall': Waterfall plot for a single prediction.
                - 'force': Force plot for a single prediction.
                - 'dependence': Dependence plot for a feature.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save the plot.
            
        Returns:
        --------
        tuple
            (matplotlib.figure.Figure, object) containing the plot and SHAP values.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for this method. Install with 'pip install shap'.")
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Ensure feature names are available
        if self.feature_names is None:
            if hasattr(X, 'columns'):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Create SHAP explainer if not already done
        if self.shap_explainer is None:
            if n_samples < X.shape[0]:
                background = shap.sample(X, n_samples)
            else:
                background = X
            if isinstance(self.model, RandomForestClassifier):
                self.shap_explainer = shap.TreeExplainer(self.model, background)
            else:
                try:
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba, background,
                        feature_names=self.feature_names
                    )
                except Exception:
                    self.shap_explainer = shap.Explainer(self.model, background)
        
        # Compute SHAP values
        shap_values = self.shap_explainer(X)
        plt.figure(figsize=figsize)
        
        if plot_type == 'summary':
            shap.summary_plot(
                shap_values, X, 
                feature_names=self.feature_names,
                plot_type='bar' if hasattr(shap_values, 'values') else None,
                show=False
            )
        elif plot_type == 'bar':
            shap.plots.bar(shap_values, show=False)
        elif plot_type == 'waterfall':
            sample_idx = 0 if sample_idx is None else sample_idx
            shap.plots.waterfall(shap_values[sample_idx], show=False)
        elif plot_type == 'force':
            sample_idx = 0 if sample_idx is None else sample_idx
            shap.plots.force(shap_values[sample_idx], show=False)
        elif plot_type == 'dependence':
            # For dependence, if sample_idx is not provided, choose the feature with the highest average impact.
            if sample_idx is None:
                if hasattr(shap_values, 'values'):
                    feature_importance = np.abs(shap_values.values).mean(axis=0)
                else:
                    feature_importance = np.abs(shap_values.abs.mean(0).values)
                feature_idx = np.argmax(feature_importance)
            else:
                feature_idx = sample_idx
            shap.plots.scatter(shap_values[:, feature_idx], show=False)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        
        # Get current figure and save if required
        fig = plt.gcf()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved SHAP plot to {save_path}")
        elif self.output_dir:
            default_path = os.path.join(self.output_dir, f'shap_{plot_type}.png')
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            print(f"Saved SHAP plot to {default_path}")
        
        return fig, shap_values
    
    def explain_with_lime(self, X, sample_idx=0, num_features=10, figsize=(12, 8), save_path=None):
        """
        Explain a single prediction using LIME.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix.
        sample_idx : int
            Index of the sample to explain.
        num_features : int
            Number of features to include in the explanation.
        figsize : tuple
            Figure size.
        save_path : str, optional
            Path to save the plot.
            
        Returns:
        --------
        tuple
            (matplotlib.figure.Figure, object) with the LIME explanation plot and explanation object.
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is required for this method. Install with 'pip install lime'.")
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        # Ensure feature names are set
        if self.feature_names is None:
            if hasattr(X, 'columns'):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Create LIME explainer if necessary
        if self.lime_explainer is None:
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                X.values if hasattr(X, 'values') else X,
                feature_names=self.feature_names,
                class_names=self.class_names,)
