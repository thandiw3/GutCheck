"""
Enhanced model options for GutCheck.

This module provides multiple classification algorithms beyond the original Random Forest,
including SVM, Gradient Boosting, Neural Networks, and ensemble methods.
It also implements hyperparameter tuning and model evaluation with cross-validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
import shap
import eli5
from eli5.sklearn import PermutationImportance

class ModelSelector:
    """
    Class for selecting, training, and evaluating different classification models.
    """
    
    def __init__(self, model_type='random_forest', class_weight='balanced', random_state=42):
        """
        Initialize the model selector with specified model type.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use. Options: 'random_forest', 'svm', 'gradient_boosting', 
            'neural_network', 'voting_ensemble'
        class_weight : str or dict
            Class weights to handle imbalanced data.
        random_state : int
            Random seed for reproducibility.
        """
        self.model_type = model_type
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = self._initialize_model()
        self.is_fitted = False
        self.feature_importances_ = None
        
    def _initialize_model(self):
        """Initialize the selected model with default parameters."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
        elif self.model_type == 'svm':
            return SVC(
                probability=True,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        elif self.model_type == 'neural_network':
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.model_type == 'voting_ensemble':
            rf = RandomForestClassifier(
                n_estimators=100,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
            svm = SVC(
                probability=True,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
            gb = GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
            return VotingClassifier(
                estimators=[('rf', rf), ('svm', svm), ('gb', gb)],
                voting='soft'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Parameters:
        -----------
        X : array-like
            Training features.
        y : array-like
            Target labels.
        
        Returns:
        --------
        self : object
            Returns self after fitting.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif self.model_type == 'voting_ensemble':
            # For voting ensemble, use the feature importances from the RF component if available
            if hasattr(self.model.named_estimators_['rf'], 'feature_importances_'):
                self.feature_importances_ = self.model.named_estimators_['rf'].feature_importances_
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : array-like
            Features to predict.
        
        Returns:
        --------
        array-like
            Predicted class labels.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        -----------
        X : array-like
            Features to predict.
        
        Returns:
        --------
        array-like
            Predicted class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict_proba(X)
    
    def tune_hyperparameters(self, X, y, param_grid=None, cv=5, scoring='accuracy'):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Parameters:
        -----------
        X : array-like
            Training features.
        y : array-like
            Target labels.
        param_grid : dict
            Dictionary with parameter names as keys and lists of parameter values.
        cv : int
            Number of cross-validation folds.
        scoring : str
            Scoring metric to use.
            
        Returns:
        --------
        self : object
            Returns self with the best model from hyperparameter tuning.
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best {scoring}: {grid_search.best_score_:.4f}")
        
        # Update model with best estimator and mark as fitted
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        # Update feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        
        return self
    
    def _get_default_param_grid(self):
        """Get default parameter grid for the selected model type."""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'svm':
            return {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear', 'poly']
            }
        elif self.model_type == 'gradient_boosting':
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'neural_network':
            return {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        elif self.model_type == 'voting_ensemble':
            # For ensemble, we'll just tune the weights of individual classifiers.
            return {
                'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]
            }
        else:
            return {}
    
    def evaluate(self, X, y, cv=5):
        """
        Evaluate the model using cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Features.
        y : array-like
            Target labels.
        cv : int
            Number of cross-validation folds.
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Calculate cross-validation scores for various metrics
        accuracy = cross_val_score(self.model, X, y, cv=cv_strategy, scoring='accuracy')
        precision = cross_val_score(self.model, X, y, cv=cv_strategy, scoring='precision')
        recall = cross_val_score(self.model, X, y, cv=cv_strategy, scoring='recall')
        f1 = cross_val_score(self.model, X, y, cv=cv_strategy, scoring='f1')
        roc_auc = cross_val_score(self.model, X, y, cv=cv_strategy, scoring='roc_auc')
        
        results = {
            'accuracy': {
                'mean': accuracy.mean(),
                'std': accuracy.std(),
                'values': accuracy
            },
            'precision': {
                'mean': precision.mean(),
                'std': precision.std(),
                'values': precision
            },
            'recall': {
                'mean': recall.mean(),
                'std': recall.std(),
                'values': recall
            },
            'f1': {
                'mean': f1.mean(),
                'std': f1.std(),
                'values': f1
            },
            'roc_auc': {
                'mean': roc_auc.mean(),
                'std': roc_auc.std(),
                'values': roc_auc
            }
        }
        
        return results
    
    def evaluate_test_set(self, X_test, y_test):
        """
        Evaluate the model on a test set.
        
        Parameters:
        -----------
        X_test : array-like
            Test features.
        y_test : array-like
            Test labels.
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return results
    
    def plot_roc_curve(self, X_test, y_test, save_path=None):
        """
        Plot ROC curve for the model.
        
        Parameters:
        -----------
        X_test : array-like
            Test features.
        y_test : array-like
            Test labels.
        save_path : str, optional
            Path to save the plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object of the ROC curve.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_precision_recall_curve(self, X_test, y_test, save_path=None):
        """
        Plot Precision-Recall curve for the model.
        
        Parameters:
        -----------
        X_test : array-like
            Test features.
        y_test : array-like
            Test labels.
        save_path : str, optional
            Path to save the plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object of the Precision-Recall curve.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """
        Plot confusion matrix for the model.
        
        Parameters:
        -----------
        X_test : array-like
            Test features.
        y_test : array-like
            Test labels.
        save_path : str, optional
            Path to save the plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object of the confusion matrix.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()
    
    def plot_feature_importance(self, feature_names, top_n=20, save_path=None):
        """
        Plot feature importance for the model.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names.
        top_n : int
            Number of top features to display.
        save_path : str, optional
            Path to save the plot.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object of the feature importance plot.
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        if self.feature_importances_ is None:
            raise ValueError("Model does not have feature importances.")
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.feature_importances_
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        top_importance_df = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=top_importance_df)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()