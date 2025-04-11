"""
Module for contamination control in microbiome analysis.

This module provides methods for identifying and controlling environmental
contamination in microbiome samples, addressing the challenge of maintaining
a comprehensive database of environmental normal flora and typical contaminants.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.stats import spearmanr
import os
import json
import pickle
import warnings
from collections import defaultdict

class ContaminationController:
    """
    Class for identifying and controlling contamination in microbiome samples.
    
    This class implements methods for identifying and controlling environmental
    contamination in microbiome samples, addressing the challenge of maintaining
    a comprehensive database of environmental normal flora and typical contaminants.
    """
    
    def __init__(self, contaminant_db=None, method='frequency', verbose=False):
        """
        Initialize the ContaminationController.
        
        Parameters:
        -----------
        contaminant_db : str or dict, optional
            Path to contaminant database file or dictionary of known contaminants
        method : str, default='frequency'
            Method to use for contamination identification.
            Options: 'frequency', 'correlation', 'prevalence', 'negative_control'
        verbose : bool, default=False
            Whether to print verbose output
        """
        self.method = method
        self.verbose = verbose
        self.contaminants = {}
        self.environment_profile = {}
        self.negative_controls = {}
        self.is_fitted = False
        
        # Validate method
        valid_methods = ['frequency', 'correlation', 'prevalence', 'negative_control']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        # Load contaminant database if provided
        if contaminant_db is not None:
            self.load_contaminant_database(contaminant_db)
    
    def load_contaminant_database(self, contaminant_db):
        """
        Load a database of known contaminants.
        
        Parameters:
        -----------
        contaminant_db : str or dict
            Path to contaminant database file or dictionary of known contaminants
            
        Returns:
        --------
        self : ContaminationController
            Updated controller object
        """
        if self.verbose:
            print("Loading contaminant database")
        
        if isinstance(contaminant_db, str):
            # Load from file
            if contaminant_db.endswith('.json'):
                with open(contaminant_db, 'r') as f:
                    self.contaminants = json.load(f)
            elif contaminant_db.endswith('.pkl'):
                with open(contaminant_db, 'rb') as f:
                    self.contaminants = pickle.load(f)
            elif contaminant_db.endswith('.csv'):
                # Assume CSV format with columns: taxon, environment, frequency
                df = pd.read_csv(contaminant_db)
                
                self.contaminants = {}
                for _, row in df.iterrows():
                    taxon = row['taxon']
                    environment = row.get('environment', 'unknown')
                    frequency = row.get('frequency', 1.0)
                    
                    if environment not in self.contaminants:
                        self.contaminants[environment] = {}
                    
                    self.contaminants[environment][taxon] = frequency
            else:
                raise ValueError(f"Unsupported file format: {contaminant_db}")
        else:
            # Use provided dictionary
            self.contaminants = contaminant_db
        
        return self
    
    def create_environment_profile(self, samples=None, environment_type='lab'):
        """
        Create a profile of the environmental microbiome.
        
        Parameters:
        -----------
        samples : pandas.DataFrame, optional
            OTU table with environmental samples
        environment_type : str, default='lab'
            Type of environment
            
        Returns:
        --------
        self : ContaminationController
            Updated controller object
        """
        if self.verbose:
            print(f"Creating environment profile for {environment_type}")
        
        if samples is not None:
            # Calculate mean abundance for each taxon
            self.environment_profile[environment_type] = samples.mean(axis=0).to_dict()
        elif environment_type in self.contaminants:
            # Use contaminant database
            self.environment_profile[environment_type] = self.contaminants[environment_type]
        else:
            # Create default profile with common lab contaminants
            self.environment_profile[environment_type] = self._create_default_profile(environment_type)
        
        return self
    
    def _create_default_profile(self, environment_type):
        """
        Create a default environmental profile with common contaminants.
        
        Parameters:
        -----------
        environment_type : str
            Type of environment
            
        Returns:
        --------
        profile : dict
            Default environmental profile
        """
        # Common contaminants by environment type
        if environment_type == 'lab':
            # Common lab contaminants
            return {
                'Propionibacterium': 0.8,
                'Staphylococcus': 0.7,
                'Streptococcus': 0.6,
                'Corynebacterium': 0.5,
                'Pseudomonas': 0.4,
                'Acinetobacter': 0.4,
                'Escherichia': 0.3,
                'Bacillus': 0.3,
                'Ralstonia': 0.3,
                'Bradyrhizobium': 0.2
            }
        elif environment_type == 'skin':
            # Common skin microbiome
            return {
                'Propionibacterium': 0.9,
                'Staphylococcus': 0.8,
                'Corynebacterium': 0.7,
                'Streptococcus': 0.5,
                'Micrococcus': 0.4,
                'Brevibacterium': 0.3,
                'Acinetobacter': 0.3,
                'Malassezia': 0.3,
                'Dermacoccus': 0.2,
                'Lactobacillus': 0.1
            }
        elif environment_type == 'water':
            # Common water contaminants
            return {
                'Pseudomonas': 0.8,
                'Acinetobacter': 0.7,
                'Sphingomonas': 0.6,
                'Flavobacterium': 0.5,
                'Legionella': 0.4,
                'Methylobacterium': 0.4,
                'Mycobacterium': 0.3,
                'Sphingobium': 0.3,
                'Novosphingobium': 0.2,
                'Bradyrhizobium': 0.2
            }
        elif environment_type == 'reagent':
            # Common reagent contaminants
            return {
                'Ralstonia': 0.8,
                'Bradyrhizobium': 0.7,
                'Burkholderia': 0.6,
                'Propionibacterium': 0.5,
                'Pseudomonas': 0.5,
                'Sphingomonas': 0.4,
                'Methylobacterium': 0.4,
                'Acinetobacter': 0.3,
                'Stenotrophomonas': 0.3,
                'Escherichia': 0.2
            }
        else:
            # Generic contaminants
            return {
                'Propionibacterium': 0.5,
                'Staphylococcus': 0.5,
                'Pseudomonas': 0.5,
                'Acinetobacter': 0.5,
                'Escherichia': 0.5,
                'Ralstonia': 0.5,
                'Bradyrhizobium': 0.5,
                'Sphingomonas': 0.5,
                'Burkholderia': 0.5,
                'Streptococcus': 0.5
            }
    
    def add_negative_controls(self, control_samples, control_type='extraction'):
        """
        Add negative control samples for contamination identification.
        
        Parameters:
        -----------
        control_samples : pandas.DataFrame
            OTU table with negative control samples
        control_type : str, default='extraction'
            Type of negative control
            
        Returns:
        --------
        self : ContaminationController
            Updated controller object
        """
        if self.verbose:
            print(f"Adding {control_samples.shape[0]} negative controls of type {control_type}")
        
        # Store negative controls
        self.negative_controls[control_type] = control_samples
        
        return self
    
    def identify_contaminants(self, samples, dna_concentrations=None, 
                             environment_type='lab', threshold=0.5):
        """
        Identify potential contaminants in samples.
        
        Parameters:
        -----------
        samples : pandas.DataFrame
            OTU table with samples
        dna_concentrations : pandas.Series, optional
            DNA concentrations for each sample
        environment_type : str, default='lab'
            Type of environment
        threshold : float, default=0.5
            Threshold for contamination identification
            
        Returns:
        --------
        contaminants : pandas.DataFrame
            DataFrame with potential contaminants and scores
        """
        if self.verbose:
            print(f"Identifying contaminants using {self.method} method")
        
        # Ensure environment profile exists
        if environment_type not in self.environment_profile:
            self.create_environment_profile(environment_type=environment_type)
        
        # Initialize results DataFrame
        contaminants = pd.DataFrame(index=samples.columns)
        contaminants['is_contaminant'] = False
        contaminants['score'] = 0.0
        contaminants['evidence'] = ''
        
        if self.method == 'frequency':
            # Identify contaminants based on frequency in environment profile
            for taxon in samples.columns:
                # Check if taxon is in environment profile
                if taxon in self.environment_profile[environment_type]:
                    # Get frequency in environment
                    freq = self.environment_profile[environment_type][taxon]
                    
                    # Mark as contaminant if frequency exceeds threshold
                    if freq >= threshold:
                        contaminants.loc[taxon, 'is_contaminant'] = True
                        contaminants.loc[taxon, 'score'] = freq
                        contaminants.loc[taxon, 'evidence'] = f"Frequency in {environment_type}: {freq:.2f}"
        
        elif self.method == 'correlation':
            # Identify contaminants based on correlation with DNA concentration
            if dna_concentrations is None:
                raise ValueError("DNA concentrations required for correlation method")
            
            # Ensure sample IDs match
            if not all(idx in dna_concentrations.index for idx in samples.index):
                raise ValueError("Sample IDs in DNA concentrations must match samples")
            
            # Calculate correlation for each taxon
            for taxon in samples.columns:
                # Get abundances for this taxon
                abundances = samples[taxon]
                
                # Calculate correlation with DNA concentration
                corr, p_value = spearmanr(abundances, dna_concentrations[abundances.index])
                
                # Strong negative correlation suggests contamination
                # (higher abundance in low DNA concentration samples)
                if corr < -threshold and p_value < 0.05:
                    contaminants.loc[taxon, 'is_contaminant'] = True
                    contaminants.loc[taxon, 'score'] = abs(corr)
                    contaminants.loc[taxon, 'evidence'] = f"Correlation with DNA concentration: {corr:.2f} (p={p_value:.3f})"
        
        elif self.method == 'prevalence':
            # Identify contaminants based on prevalence across samples
            # (contaminants tend to be present in many samples)
            
            # Calculate prevalence (fraction of samples with non-zero abundance)
            prevalence = (samples > 0).mean()
            
            # Mark taxa with high prevalence as potential contaminants
            for taxon in samples.columns:
                if prevalence[taxon] >= threshold:
                    contaminants.loc[taxon, 'is_contaminant'] = True
                    contaminants.loc[taxon, 'score'] = prevalence[taxon]
                    contaminants.loc[taxon, 'evidence'] = f"Prevalence across samples: {prevalence[taxon]:.2f}"
        
        elif self.method == 'negative_control':
            # Identify contaminants based on presence in negative controls
            if not self.negative_controls:
                raise ValueError("Negative controls required for negative_control method")
            
            # Combine all negative controls
            all_controls = pd.concat(self.negative_controls.values())
            
            # Calculate mean abundance in negative controls
            control_abundance = all_controls.mean()
            
            # Mark taxa present in negative controls as contaminants
            for taxon in samples.columns:
                if taxon in control_abundance and control_abundance[taxon] > 0:
                    # Calculate ratio of abundance in controls to samples
                    sample_abundance = samples[taxon].mean()
                    ratio = control_abundance[taxon] / sample_abundance if sample_abundance > 0 else float('inf')
                    
                    # Mark as contaminant if ratio exceeds threshold
                    if ratio >= threshold:
                        contaminants.loc[taxon, 'is_contaminant'] = True
                        contaminants.loc[taxon, 'score'] = ratio
                        contaminants.loc[taxon, 'evidence'] = f"Ratio of abundance in controls to samples: {ratio:.2f}"
        
        # Sort by score
        contaminants = contaminants.sort_values('score', ascending=False)
        
        self.is_fitted = True
        return contaminants
    
    def remove_contaminants(self, samples, contaminants=None, method='subtract', 
                           min_abundance=0.0):
        """
        Remove identified contaminants from samples.
        
        Parameters:
        -----------
        samples : pandas.DataFrame
            OTU table with samples
        contaminants : pandas.DataFrame, optional
            DataFrame with contaminants from identify_contaminants()
        method : str, default='subtract'
            Method to use for contaminant removal.
            Options: 'subtract', 'zero', 'filter'
        min_abundance : float, default=0.0
            Minimum abundance after contaminant removal
            
        Returns:
        --------
        cleaned_samples : pandas.DataFrame
            OTU table with contaminants removed
        """
        if not self.is_fitted and contaminants is None:
            raise ValueError("Must call identify_contaminants() first or provide contaminants DataFrame")
        
        if self.verbose:
            print(f"Removing contaminants using {method} method")
        
        # Use provided contaminants or from previous identification
        if contaminants is None:
            # This would be set by identify_contaminants()
            raise ValueError("Contaminants DataFrame must be provided")
        
        # Get list of contaminant taxa
        contaminant_taxa = contaminants[contaminants['is_contaminant']].index.tolist()
        
        if not contaminant_taxa:
            # No contaminants identified
            if self.verbose:
                print("No contaminants identified")
            return samples.copy()
        
        # Create copy of samples
        cleaned_samples = samples.copy()
        
        if method == 'subtract':
            # Subtract contaminant abundance from samples
            for taxon in contaminant_taxa:
                if taxon in self.negative_controls:
                    # Use mean abundance in negative controls
                    all_controls = pd.concat(self.negative_controls.values())
                    contaminant_abundance = all_controls[taxon].mean()
                else:
                    # Use mean abundance in samples
                    contaminant_abundance = samples[taxon].mean() * 0.5
                
                # Subtract contaminant abundance
                cleaned_samples[taxon] = cleaned_samples[taxon] - contaminant_abundance
                
                # Ensure non-negative values
                cleaned_samples[taxon] = cleaned_samples[taxon].clip(lower=min_abundance)
        
        elif method == 'zero':
            # Set contaminant abundance to zero
            for taxon in contaminant_taxa:
                cleaned_samples[taxon] = min_abundance
        
        elif method == 'filter':
            # Remove contaminant taxa from samples
            cleaned_samples = cleaned_samples.drop(columns=contaminant_taxa)
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Renormalize to maintain relative abundance
        if method != 'filter':
            row_sums = cleaned_samples.sum(axis=1)
            for idx in cleaned_samples.index:
                if row_sums[idx] > 0:
                    cleaned_samples.loc[idx] = cleaned_samples.loc[idx] / row_sums[idx]
        
        return cleaned_samples
    
    def plot_contaminants(self, samples, contaminants, n_contaminants=10, figsize=(12, 8)):
        """
        Plot identified contaminants.
        
        Parameters:
        -----------
        samples : pandas.DataFrame
            OTU table with samples
        contaminants : pandas.DataFrame
            DataFrame with contaminants from identify_contaminants()
        n_contaminants : int, default=10
            Number of top contaminants to plot
        figsize : tuple, default=(12, 8)
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        # Select top contaminants
        top_contaminants = contaminants[contaminants['is_contaminant']].nlargest(n_contaminants, 'score')
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot contaminant scores
        sns.barplot(
            x=top_contaminants.index,
            y='score',
            data=top_contaminants,
            ax=axes[0]
        )
        
        axes[0].set_title('Contaminant Scores')
        axes[0].set_xlabel('Taxon')
        axes[0].set_ylabel('Score')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        
        # Plot contaminant abundance
        contaminant_abundance = samples[top_contaminants.index].mean().sort_values(ascending=False)
        
        sns.barplot(
            x=contaminant_abundance.index,
            y=contaminant_abundance.values,
            ax=axes[1]
        )
        
        axes[1].set_title('Contaminant Abundance')
        axes[1].set_xlabel('Taxon')
        axes[1].set_ylabel('Mean Abundance')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_before_after(self, samples, cleaned_samples, contaminants, n_samples=5, figsize=(12, 10)):
        """
        Plot samples before and after contaminant removal.
        
        Parameters:
        -----------
        samples : pandas.DataFrame
            OTU table with samples before contaminant removal
        cleaned_samples : pandas.DataFrame
            OTU table with samples after contaminant removal
        contaminants : pandas.DataFrame
            DataFrame with contaminants from identify_contaminants()
        n_samples : int, default=5
            Number of samples to plot
        figsize : tuple, default=(12, 10)
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        # Select random samples
        if samples.shape[0] > n_samples:
            sample_ids = np.random.choice(samples.index, size=n_samples, replace=False)
        else:
            sample_ids = samples.index
        
        # Get contaminant taxa
        contaminant_taxa = contaminants[contaminants['is_contaminant']].index.tolist()
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 2, figsize=figsize)
        
        for i, sample_id in enumerate(sample_ids):
            # Plot before
            samples.loc[sample_id].plot(
                kind='bar',
                ax=axes[i, 0],
                color=['red' if taxon in contaminant_taxa else 'blue' for taxon in samples.columns]
            )
            
            axes[i, 0].set_title(f'Before: {sample_id}')
            axes[i, 0].set_xlabel('Taxon')
            axes[i, 0].set_ylabel('Abundance')
            axes[i, 0].set_xticklabels([])
            
            # Plot after
            if all(col in cleaned_samples.columns for col in samples.columns):
                # Same columns
                cleaned_samples.loc[sample_id].plot(
                    kind='bar',
                    ax=axes[i, 1],
                    color=['red' if taxon in contaminant_taxa else 'blue' for taxon in cleaned_samples.columns]
                )
            else:
                # Different columns (e.g., after filtering)
                cleaned_samples.loc[sample_id].plot(
                    kind='bar',
                    ax=axes[i, 1],
                    color='blue'
                )
            
            axes[i, 1].set_title(f'After: {sample_id}')
            axes[i, 1].set_xlabel('Taxon')
            axes[i, 1].set_ylabel('Abundance')
            axes[i, 1].set_xticklabels([])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Contaminant'),
            Patch(facecolor='blue', label='Non-contaminant')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        return fig


def identify_contaminants(samples, method='frequency', **kwargs):
    """
    Identify potential contaminants in samples.
    
    Parameters:
    -----------
    samples : pandas.DataFrame
        OTU table with samples
    method : str, default='frequency'
        Method to use for contamination identification
    **kwargs : dict
        Additional arguments to pass to ContaminationController.identify_contaminants()
        
    Returns:
    --------
    contaminants : pandas.DataFrame
        DataFrame with potential contaminants and scores
    """
    controller = ContaminationController(method=method)
    return controller.identify_contaminants(samples, **kwargs)


def remove_contaminants(samples, contaminants, method='subtract', **kwargs):
    """
    Remove identified contaminants from samples.
    
    Parameters:
    -----------
    samples : pandas.DataFrame
        OTU table with samples
    contaminants : pandas.DataFrame
        DataFrame with contaminants from identify_contaminants()
    method : str, default='subtract'
        Method to use for contaminant removal
    **kwargs : dict
        Additional arguments to pass to ContaminationController.remove_contaminants()
        
    Returns:
    --------
    cleaned_samples : pandas.DataFrame
        OTU table with contaminants removed
    """
    controller = ContaminationController()
    return controller.remove_contaminants(samples, contaminants, method=method, **kwargs)


def create_negative_controls(n_controls=3, n_taxa=100, contamination_level=0.1):
    """
    Create simulated negative control samples.
    
    Parameters:
    -----------
    n_controls : int, default=3
        Number of negative control samples to create
    n_taxa : int, default=100
        Number of taxa in the samples
    contamination_level : float, default=0.1
        Level of contamination in the samples
        
    Returns:
    --------
    controls : pandas.DataFrame
        OTU table with negative control samples
    """
    # Create sample IDs
    sample_ids = [f'NC_{i+1}' for i in range(n_controls)]
    
    # Create taxon IDs
    taxon_ids = [f'Taxon_{i+1}' for i in range(n_taxa)]
    
    # Create empty DataFrame
    controls = pd.DataFrame(0.0, index=sample_ids, columns=taxon_ids)
    
    # Add contamination
    for i in range(n_controls):
        # Select random subset of taxa as contaminants
        n_contaminants = np.random.randint(5, 20)
        contaminant_taxa = np.random.choice(taxon_ids, size=n_contaminants, replace=False)
        
        # Add contamination
        for taxon in contaminant_taxa:
            controls.loc[sample_ids[i], taxon] = np.random.uniform(0, contamination_level)
        
        # Normalize to sum to 1
        controls.loc[sample_ids[i]] = controls.loc[sample_ids[i]] / controls.loc[sample_ids[i]].sum()
    
    return controls
