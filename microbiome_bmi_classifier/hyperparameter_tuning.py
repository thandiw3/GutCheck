"""
Hyperparameter tuning module for microbiome-based classification models.

This module provides comprehensive hyperparameter tuning capabilities for
various classification algorithms using grid search, random search, and
Bayesian optimization approaches.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    StratifiedKFold, learning_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
import os
import joblib
import time
from scipy.stats import randint, uniform, loguniform

# Try to import optional dependencies for Bayesian and Optuna searches
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterTuner:
    """
    Class for tuning hyperparameters of classification models.
    """
    
    def __init__(self, random_state=42, n_jobs=-1, cv=5, scoring='accuracy'):
        """
        Initialize the hyperparameter tuner.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility.
        n_jobs : int
            Number of parallel jobs.
        cv : int
            Number of cross-validation folds.
        scoring : str or callable
            Scoring metric for evaluation.
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cv = cv
        self.scoring = scoring
        self.results = {}
        self.best_params = {}
        self.best_estimators = {}
        
        # Define cross-validation strategy
        self.cv_strategy = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
    
    def grid_search(self, X, y, estimator, param_grid, model_name=None):
        """
        Perform grid search for hyperparameter tuning.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix.
        y : array-like
            Target labels.
        estimator : estimator
            Base estimator to tune.
        param_grid : dict
            Parameter grid to search.
        model_name : str, optional
            Name of the model.
            
        Returns:
        --------
        dict
            Grid search results.
        """
        if model_name is None:
            model_name = estimator.__class__.__name__
        
        print(f"Performing grid search for {model_name}...")
        start_time = time.time()
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=self.cv_strategy,
            n_jobs=self.n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        # Fit grid search
        grid_search.fit(X, y)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Store results
        self.results[model_name] = {
            'grid_search': grid_search,
            'cv_results': grid_search.cv_results_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'runtime': runtime
        }
        
        self.best_params[model_name] = grid_search.best_params_
        self.best_estimators[model_name] = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print(f"Runtime: {runtime:.2f} seconds")
        
        return self.results[model_name]
    
    def random_search(self, X, y, estimator, param_distributions, n_iter=100, model_name=None):
        """
        Perform random search for hyperparameter tuning.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix.
        y : array-like
            Target labels.
        estimator : estimator
            Base estimator to tune.
        param_distributions : dict
            Parameter distributions to sample from.
        n_iter : int
            Number of parameter settings to sample.
        model_name : str, optional
            Name of the model.
            
        Returns:
        --------
        dict
            Random search results.
        """
        if model_name is None:
            model_name = estimator.__class__.__name__
        
        print(f"Performing random search for {model_name} with {n_iter} iterations...")
        start_time = time.time()
        
        # Create random search
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=self.cv_strategy,
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=self.random_state,
            return_train_score=True
        )
        
        # Fit random search
        random_search.fit(X, y)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Store results
        self.results[model_name] = {
            'random_search': random_search,
            'cv_results': random_search.cv_results_,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'runtime': runtime
        }
        
        self.best_params[model_name] = random_search.best_params_
        self.best_estimators[model_name] = random_search.best_estimator_
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best cross-validation score: {random_search.best_score_:.4f}")
        print(f"Runtime: {runtime:.2f} seconds")
        
        return self.results[model_name]
    
    def bayesian_search(self, X, y, estimator, search_spaces, n_iter=50, model_name=None):
        """
        Perform Bayesian optimization for hyperparameter tuning using scikit-optimize.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix.
        y : array-like
            Target labels.
        estimator : estimator
            Base estimator to tune.
        search_spaces : dict
            Search spaces for parameters.
        n_iter : int
            Number of parameter settings to sample.
        model_name : str, optional
            Name of the model.
            
        Returns:
        --------
        dict
            Bayesian search results.
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization. Install with 'pip install scikit-optimize'.")
        
        if model_name is None:
            model_name = estimator.__class__.__name__
        
        print(f"Performing Bayesian optimization for {model_name} with {n_iter} iterations...")
        start_time = time.time()
        
        # Create Bayesian search
        bayes_search = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_spaces,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=self.cv_strategy,
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=self.random_state,
            return_train_score=True
        )
        
        # Fit Bayesian search
        bayes_search.fit(X, y)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Store results
        self.results[model_name] = {
            'bayes_search': bayes_search,
            'cv_results': bayes_search.cv_results_,
            'best_params': bayes_search.best_params_,
            'best_score': bayes_search.best_score_,
            'runtime': runtime
        }
        
        self.best_params[model_name] = bayes_search.best_params_
        self.best_estimators[model_name] = bayes_search.best_estimator_
        
        print(f"Best parameters: {bayes_search.best_params_}")
        print(f"Best cross-validation score: {bayes_search.best_score_:.4f}")
        print(f"Runtime: {runtime:.2f} seconds")
        
        return self.results[model_name]
    
    def optuna_search(self, X, y, estimator_fn, param_fn, n_trials=100, model_name=None):
        """
        Perform hyperparameter tuning using Optuna.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix.
        y : array-like
            Target labels.
        estimator_fn : callable
            Function that creates an estimator from trial parameters.
        param_fn : callable
            Function that defines parameter search space for a trial.
        n_trials : int
            Number of trials.
        model_name : str, optional
            Name of the model.
            
        Returns:
        --------
        dict
            Optuna search results.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for this search method. Install with 'pip install optuna'.")
        
        if model_name is None:
            model_name = "OptunaModel"
        
        print(f"Performing Optuna optimization for {model_name} with {n_trials} trials...")
        start_time = time.time()
        
        # Define objective function
        def objective(trial):
            # Get parameters for this trial
            params = param_fn(trial)
            
            # Create estimator with these parameters
            estimator = estimator_fn(params)
            
            # Evaluate using cross-validation
            scores = cross_val_score(
                estimator, X, y, 
                cv=self.cv_strategy, 
                scoring=self.scoring,
                n_jobs=self.n_jobs
            )
            return scores.mean()
        
        # Create study
        study = optuna.create_study(direction='maximize')
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        
        # Create best estimator and fit on full data
        best_estimator = estimator_fn(best_params)
        best_estimator.fit(X, y)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Store results
        self.results[model_name] = {
            'study': study,
            'best_params': best_params,
            'best_score': study.best_value,
            'runtime': runtime
        }
        
        self.best_params[model_name] = best_params
        self.best_estimators[model_name] = best_estimator
        
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation score: {study.best_value:.4f}")
        print(f"Runtime: {runtime:.2f} seconds")
        
        return self.results[model_name]
    
    def tune_random_forest(self, X, y, method='grid', param_grid=None, n_iter=100):
        """
        Tune hyperparameters for Random Forest classifier.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix.
        y : array-like
            Target labels.
        method : str
            Tuning method: 'grid', 'random', 'bayes', or 'optuna'.
        param_grid : dict, optional
            Parameter grid or distributions.
        n_iter : int
            Number of iterations for random, Bayesian, or Optuna search.
            
        Returns:
        --------
        dict
            Tuning results.
        """
        model_name = "RandomForest"
        
        # Define default parameter grid if not provided
        if param_grid is None:
            if method == 'grid':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            elif method == 'random':
                param_grid = {
                    'n_estimators': randint(50, 500),
                    'max_depth': randint(5, 50),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', None]
                }
            elif method == 'bayes':
                param_grid = {
                    'n_estimators': Integer(50, 500),
                    'max_depth': Integer(5, 50),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 10),
                    'max_features': Categorical(['sqrt', 'log2', None])
                }
        
        # Create base estimator
        base_estimator = RandomForestClassifier(random_state=self.random_state)
        
        # Perform tuning
        if method == 'grid':
            return self.grid_search(X, y, base_estimator, param_grid, model_name)
        elif method == 'random':
            return self.random_search(X, y, base_estimator, param_grid, n_iter, model_name)
        elif method == 'bayes':
            return self.bayesian_search(X, y, base_estimator, param_grid, n_iter, model_name)
        elif method == 'optuna':
            # Define parameter function for Optuna
            def param_fn(trial):
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 50),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
            
            # Define estimator function
            def estimator_fn(params):
                return RandomForestClassifier(random_state=self.random_state, **params)
            
            return self.optuna_search(X, y, estimator_fn, param_fn, n_iter, model_name)
        else:
            raise ValueError(f"Unknown tuning method: {method}")
    
    def tune_gradient_boosting(self, X, y, method='grid', param_grid=None, n_iter=100):
        """
        Tune hyperparameters for Gradient Boosting classifier.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix.
        y : array-like
            Target labels.
        method : str
            Tuning method: 'grid', 'random', 'bayes', or 'optuna'.
        param_grid : dict, optional
            Parameter grid or distributions.
        n_iter : int
            Number of iterations for random, Bayesian, or Optuna search.
            
        Returns:
        --------
        dict
            Tuning results.
        """
        model_name = "GradientBoosting"
        
        # Define default parameter grid or distributions if not provided
        if param_grid is None:
            if method == 'grid':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.7, 0.8, 0.9, 1.0]
                }
            elif method == 'random':
                param_grid = {
                    'n_estimators': randint(50, 500),
                    'learning_rate': uniform(0.01, 0.29),  # Range: 0.01 to 0.3
                    'max_depth': randint(3, 10),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'subsample': uniform(0.7, 0.3)  # Range: 0.7 to 1.0
                }
            elif method == 'bayes':
                param_grid = {
                    'n_estimators': Integer(50, 500),
                    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'max_depth': Integer(3, 10),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 10),
                    'subsample': Real(0.7, 1.0)
                }
        
        # Create base estimator
        base_estimator = GradientBoostingClassifier(random_state=self.random_state)
        
        # Perform tuning
        if method == 'grid':
            return self.grid_search(X, y, base_estimator, param_grid, model_name)
        elif method == 'random':
            return self.random_search(X, y, base_estimator, param_grid, n_iter, model_name)
        elif method == 'bayes':
            return self.bayesian_search(X, y, base_estimator, param_grid, n_iter, model_name)
        elif method == 'optuna':
            # Define parameter function for Optuna
            def param_fn(trial):
                return {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0)
                }
            
            # Define estimator function
            def estimator_fn(params):
                return GradientBoostingClassifier(random_state=self.random_state, **params)
            
            return self.optuna_search(X, y, estimator_fn, param_fn, n_iter, model_name)
        else:
            raise ValueError(f"Unknown tuning method: {method}")
