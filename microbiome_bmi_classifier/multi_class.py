"""
Multi-class classification module for microbiome-based BMI classification.

This module extends the binary classification capabilities to support multi-class
classification for different BMI categories (underweight, normal, overweight, obese).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import os
import joblib


class MultiClassBMIClassifier:
    """
    Class for multi-class BMI classification.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the multi-class classifier.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility.
        """
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.class_names = None
    
    def preprocess_data(self, X, y=None):
        """
        Preprocess data for classification.
        
        Scales features using StandardScaler and, if provided, encodes labels.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix.
        y : array-like, optional
            Target labels.
            
        Returns:
        --------
        tuple
            (X_processed, y_processed) where X is scaled and y is encoded (if provided).
        """
        # Fit the scaler on X and transform
        X_processed = self.scaler.fit_transform(X)
        
        # Store feature names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Process labels if provided
        y_processed = None
        if y is not None:
            # If labels are not numeric, encode them
            if not pd.api.types.is_numeric_dtype(y):
                y_processed = self.label_encoder.fit_transform(y)
            else:
                y_processed = y
            # Store the corresponding class names
            self.class_names = self.label_encoder.classes_.tolist()
        
        return X_processed, y_processed
    
    def create_bmi_classes(self, bmi_values, class_type='four'):
        """
        Create BMI classes from continuous BMI values.
        
        Parameters:
        -----------
        bmi_values : array-like
            Continuous BMI values.
        class_type : str
            Type of classification:
              - 'four': Underweight, Normal, Overweight, Obese.
              - 'three': Normal, Overweight, Obese.
              - 'binary': Healthy (BMI < 30) vs Obese (BMI >= 30).
            
        Returns:
        --------
        array-like
            Encoded BMI class labels.
        """
        if class_type == 'four':
            # Four BMI classes
            classes = np.empty(len(bmi_values), dtype=object)
            classes[bmi_values < 18.5] = 'Underweight'
            classes[(bmi_values >= 18.5) & (bmi_values < 25)] = 'Normal'
            classes[(bmi_values >= 25) & (bmi_values < 30)] = 'Overweight'
            classes[bmi_values >= 30] = 'Obese'
            self.class_names = ['Underweight', 'Normal', 'Overweight', 'Obese']
        elif class_type == 'three':
            # Three BMI classes (excluding underweight)
            classes = np.empty(len(bmi_values), dtype=object)
            classes[bmi_values < 25] = 'Normal'
            classes[(bmi_values >= 25) & (bmi_values < 30)] = 'Overweight'
            classes[bmi_values >= 30] = 'Obese'
            self.class_names = ['Normal', 'Overweight', 'Obese']
        elif class_type == 'binary':
            # Binary classification
            classes = np.empty(len(bmi_values), dtype=object)
            classes[bmi_values < 30] = 'Healthy'
            classes[bmi_values >= 30] = 'Obese'
            self.class_names = ['Healthy', 'Obese']
        else:
            raise ValueError(f"Unknown class_type: {class_type}")
        
        encoded_classes = self.label_encoder.fit_transform(classes)
        return encoded_classes
    
    def _create_model(self, model_type, **kwargs):
        """
        Create a base model for multi-class classification.
        
        Parameters:
        -----------
        model_type : str
            Type of model to create.
            Options:
                - 'random_forest'
                - 'gradient_boosting'
                - 'svm'
                - 'logistic_regression'
                - 'neural_network'
        **kwargs : dict
            Additional keyword arguments for the model.
            
        Returns:
        --------
        estimator
            Instantiated model.
        """
        if model_type == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state, **kwargs)
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=self.random_state, **kwargs)
        elif model_type == 'svm':
            return SVC(random_state=self.random_state, probability=True, **kwargs)
        elif model_type == 'logistic_regression':
            return LogisticRegression(random_state=self.random_state, max_iter=1000, **kwargs)
        elif model_type == 'neural_network':
            return MLPClassifier(random_state=self.random_state, max_iter=1000, **kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def fit(self, X, y, model_type='random_forest', multiclass_strategy='ovr', **kwargs):
        """
        Fit the multi-class classifier.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix.
        y : array-like
            Target labels.
        model_type : str
            Type of model to use. Options include:
                - 'random_forest'
                - 'gradient_boosting'
                - 'svm'
                - 'logistic_regression'
                - 'neural_network'
        multiclass_strategy : str
            Multi-class classification strategy:
                - 'ovr': One-vs-Rest.
                - 'ovo': One-vs-One.
                - 'native': Use the model’s native multi-class support.
        **kwargs : dict
            Additional parameters to pass to the base model.
            
        Returns:
        --------
        self
            The fitted classifier.
        """
        # Preprocess the features and labels
        X_processed, y_processed = self.preprocess_data(X, y)
        
        # Create base model
        base_model = self._create_model(model_type, **kwargs)
        
        # Wrap the base model with the specified multi-class strategy
        if multiclass_strategy == 'ovr':
            from sklearn.multiclass import OneVsRestClassifier
            self.model = OneVsRestClassifier(base_model)
        elif multiclass_strategy == 'ovo':
            from sklearn.multiclass import OneVsOneClassifier
            self.model = OneVsOneClassifier(base_model)
        elif multiclass_strategy == 'native':
            self.model = base_model
        else:
            raise ValueError(f"Unknown multiclass_strategy: {multiclass_strategy}")
        
        # Fit the model on the processed data
        self.model.fit(X_processed, y_processed)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix.
            
        Returns:
        --------
        array-like
            Predicted class labels (decoded to original labels).
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        X_processed, _ = self.preprocess_data(X)
        y_pred_encoded = self.model.predict(X_processed)
        # Convert encoded predictions back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix.
            
        Returns:
        --------
        array-like
            Predicted class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("The model does not support probability predictions.")
        
        X_processed, _ = self.preprocess_data(X)
        return self.model.predict_proba(X_processed)
    
    def evaluate(self, X, y, metrics=None):
        """
        Evaluate the classifier on provided data.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix.
        y : array-like
            True labels.
        metrics : list, optional
            List of metrics to compute. Default metrics: accuracy, precision, recall, f1.
            
        Returns:
        --------
        dict
            Dictionary with evaluation results including confusion matrix and classification report.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Default metrics to compute if none provided
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        X_processed, y_processed = self.preprocess_data(X, y)
        y_pred_encoded = self.model.predict(X_processed)
        
        results = {}
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_processed, y_pred_encoded)
        if 'precision' in metrics:
            results['precision'] = precision_score(y_processed, y_pred_encoded, average='weighted')
        if 'recall' in metrics:
            results['recall'] = recall_score(y_processed, y_pred_encoded, average='weighted')
        if 'f1' in metrics:
            results['f1'] = f1_score(y_processed, y_pred_encoded, average='weighted')
        
        results['confusion_matrix'] = confusion_matrix(y_processed, y_pred_encoded)
        results['classification_report'] = classification_report(
            y_processed, y_pred_encoded, target_names=self.class_names, output_dict=True
        )
        
        # Calculate multi-class ROC AUC if possible
        if hasattr(self.model, 'predict_proba'):
            try:
                y_pred_proba = self.model.predict_proba(X_processed)
                results['roc_auc'] = roc_auc_score(
                    y_processed, y_pred_proba, multi_class='ovr', average='weighted'
                )
            except Exception:
                results['roc_auc'] = None
        
        return results
    
    def cross_validate(self, X, y, cv=5, metrics=None):
        """
        Perform cross-validation evaluation.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Feature matrix.
        y : array-like
            Target labels.
        cv : int
            Number of cross-validation folds.
        metrics : list, optional
            List of metrics to compute. Default: accuracy, precision, recall, f1.
            
        Returns:
        --------
        dict
            Dictionary with cross-validation results.
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        X_processed, y_processed = self.preprocess_data(X, y)
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        results = {}
        
        for metric in metrics:
            if metric == 'accuracy':
                scores = cross_val_score(self.model, X_processed, y_processed, cv=cv_strategy, scoring='accuracy')
            elif metric == 'precision':
                scores = cross_val_score(self.model, X_processed, y_processed, cv=cv_strategy, scoring='precision_weighted')
            elif metric == 'recall':
                scores = cross_val_score(self.model, X_processed, y_processed, cv=cv_strategy, scoring='recall_weighted')
            elif metric == 'f1':
                scores = cross_val_score(self.model, X_processed, y_processed, cv=cv_strategy, scoring='f1_weighted')
            else:
                continue
            
            results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred=None, figsize=(10, 8), normalize=False, save_path=None):
        """
        Plot the confusion matrix.
        
        Parameters:
        -----------
        y_true : array-like
            True labels.
        y_pred : array-like, optional
            Predicted labels. If None, a ValueError is raised.
        figsize : tuple
            Figure size.
        normalize : bool
            Whether to normalize the confusion matrix.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plotted figure.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        # Ensure y_true is numeric (encoded) if not, transform it
        if not np.issubdtype(np.array(y_true).dtype, np.number):
            y_true_encoded = self.label_encoder.transform(y_true)
        else:
            y_true_encoded = y_true
        
        if y_pred is None:
            raise ValueError("y_pred must be provided for plotting the confusion matrix.")
        if not np.issubdtype(np.array(y_pred).dtype, np.number):
            y_pred_encoded = self.label_encoder.transform(y_pred)
        else:
            y_pred_encoded = y_pred
        
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=self.class_names, yticklabels=self.class_names, ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix plot to {save_path}")
        
        return fig
    