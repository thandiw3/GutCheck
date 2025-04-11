"""
Visualization module for microbiome data analysis.

This module provides comprehensive visualization tools for exploratory data analysis,
feature visualization, and results interpretation for microbiome data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import shap
import eli5
from eli5.sklearn import PermutationImportance
import networkx as nx
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr

class MicrobiomeVisualizer:
    """
    Class for comprehensive visualization of microbiome data.
    """
    
    def __init__(self, output_dir=None, style='whitegrid', palette='viridis', context='notebook'):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save visualizations
        style : str
            Seaborn style
        palette : str
            Color palette
        context : str
            Plotting context
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set visualization style
        sns.set_style(style)
        sns.set_palette(palette)
        sns.set_context(context)
        
        # Store figures
        self.figures = {}
    
    def plot_abundance_distribution(self, otu_data, top_n=20, log_scale=True, figsize=(14, 10)):
        """
        Plot the distribution of OTU abundances.
        
        Parameters:
        -----------
        otu_data : pd.DataFrame
            DataFrame containing OTU counts
        top_n : int
            Number of top OTUs to show
        log_scale : bool
            Whether to use log scale
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Calculate mean abundance for each OTU
        mean_abundance = otu_data.mean().sort_values(ascending=False)
        
        # Select top OTUs
        top_otus = mean_abundance.head(top_n).index
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Bar plot of mean abundances
        sns.barplot(x=top_otus, y=mean_abundance.loc[top_otus], ax=axes[0])
        axes[0].set_title(f'Mean Abundance of Top {top_n} OTUs')
        axes[0].set_xlabel('OTU')
        axes[0].set_ylabel('Mean Abundance')
        if log_scale:
            axes[0].set_yscale('log')
            axes[0].set_ylabel('Mean Abundance (log scale)')
        axes[0].tick_params(axis='x', rotation=90)
        
        # Plot 2: Box plot of abundances
        sns.boxplot(data=otu_data[top_otus], ax=axes[1])
        axes[1].set_title(f'Distribution of Top {top_n} OTUs')
        axes[1].set_xlabel('OTU')
        axes[1].set_ylabel('Abundance')
        if log_scale:
            axes[1].set_yscale('log')
            axes[1].set_ylabel('Abundance (log scale)')
        axes[1].tick_params(axis='x', rotation=90)
        
        plt.tight_layout()
        
        # Save figure
        if self.output_dir:
            fig_path = os.path.join(self.output_dir, 'abundance_distribution.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {fig_path}")
        
        # Store figure
        self.figures['abundance_distribution'] = fig
        
        return fig
    
    def plot_alpha_diversity(self, otu_data, metadata=None, group_col=None, figsize=(14, 10)):
        """
        Plot alpha diversity metrics.
        
        Parameters:
        -----------
        otu_data : pd.DataFrame
            DataFrame containing OTU counts
        metadata : pd.DataFrame, optional
            DataFrame containing metadata
        group_col : str, optional
            Column in metadata to group by
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        from skbio.diversity import alpha
        
        # Calculate alpha diversity metrics
        diversity = pd.DataFrame(index=otu_data.index)
        diversity['Shannon'] = otu_data.apply(alpha.shannon, axis=1)
        diversity['Simpson'] = otu_data.apply(alpha.simpson, axis=1)
        diversity['Observed_OTUs'] = (otu_data > 0).sum(axis=1)
        
        # Create figure
        if group_col and metadata is not None:
            # Ensure indices match
            if not all(idx in metadata.index for idx in diversity.index):
                raise ValueError("Not all OTU data indices are present in metadata")
            
            # Add group information
            diversity['Group'] = metadata.loc[diversity.index, group_col]
            
            # Create figure with group comparison
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            
            # Plot each diversity metric by group
            sns.boxplot(x='Group', y='Shannon', data=diversity, ax=axes[0])
            sns.stripplot(x='Group', y='Shannon', data=diversity, color='black', size=3, ax=axes[0])
            axes[0].set_title('Shannon Diversity')
            
            sns.boxplot(x='Group', y='Simpson', data=diversity, ax=axes[1])
            sns.stripplot(x='Group', y='Simpson', data=diversity, color='black', size=3, ax=axes[1])
            axes[1].set_title('Simpson Diversity')
            
            sns.boxplot(x='Group', y='Observed_OTUs', data=diversity, ax=axes[2])
            sns.stripplot(x='Group', y='Observed_OTUs', data=diversity, color='black', size=3, ax=axes[2])
            axes[2].set_title('Observed OTUs')
            
            # Add statistical test
            from scipy.stats import mannwhitneyu
            
            for i, metric in enumerate(['Shannon', 'Simpson', 'Observed_OTUs']):
                groups = diversity['Group'].unique()
                if len(groups) == 2:  # Only for binary comparison
                    group1 = diversity[diversity['Group'] == groups[0]][metric]
                    group2 = diversity[diversity['Group'] == groups[1]][metric]
                    stat, p = mannwhitneyu(group1, group2)
                    axes[i].text(0.5, 0.9, f'p = {p:.4f}', transform=axes[i].transAxes, ha='center')
        
        else:
            # Create figure without grouping
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            
            # Plot each diversity metric
            sns.histplot(diversity['Shannon'], kde=True, ax=axes[0])
            axes[0].set_title('Shannon Diversity')
            
            sns.histplot(diversity['Simpson'], kde=True, ax=axes[1])
            axes[1].set_title('Simpson Diversity')
            
            sns.histplot(diversity['Observed_OTUs'], kde=True, ax=axes[2])
            axes[2].set_title('Observed OTUs')
        
        plt.tight_layout()
        
        # Save figure
        if self.output_dir:
            fig_path = os.path.join(self.output_dir, 'alpha_diversity.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {fig_path}")
        
        # Store figure
        self.figures['alpha_diversity'] = fig
        
        return fig
    
    def plot_beta_diversity(self, otu_data, metadata=None, group_col=None, method='pca', 
                           n_components=2, figsize=(12, 10), interactive=False):
        """
        Plot beta diversity using dimensionality reduction.
        
        Parameters:
        -----------
        otu_data : pd.DataFrame
            DataFrame containing OTU counts
        metadata : pd.DataFrame, optional
            DataFrame containing metadata
        group_col : str, optional
            Column in metadata to group by
        method : str
            Dimensionality reduction method: 'pca', 'tsne', or 'umap'
        n_components : int
            Number of components
        figsize : tuple
            Figure size
        interactive : bool
            Whether to create an interactive plot with plotly
            
        Returns:
        --------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
            The figure object
        """
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(otu_data)
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=n_components)
            embedding = reducer.fit_transform(scaled_data)
            explained_var = reducer.explained_variance_ratio_ * 100
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
            embedding = reducer.fit_transform(scaled_data)
            explained_var = None
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            embedding = reducer.fit_transform(scaled_data)
            explained_var = None
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create DataFrame with embedding
        embedding_df = pd.DataFrame(
            embedding, 
            columns=[f'{method.upper()}{i+1}' for i in range(n_components)],
            index=otu_data.index
        )
        
        # Add group information if available
        if group_col and metadata is not None:
            # Ensure indices match
            if not all(idx in metadata.index for idx in embedding_df.index):
                raise ValueError("Not all OTU data indices are present in metadata")
            
            # Add group information
            embedding_df['Group'] = metadata.loc[embedding_df.index, group_col]
        
        # Create plot
        if interactive:
            # Create interactive plot with plotly
            if 'Group' in embedding_df.columns:
                fig = px.scatter(
                    embedding_df, 
                    x=embedding_df.columns[0], 
                    y=embedding_df.columns[1],
                    color='Group',
                    title=f'Beta Diversity ({method.upper()})',
                    labels={
                        embedding_df.columns[0]: f'{method.upper()}1' + (f' ({explained_var[0]:.1f}%)' if explained_var is not None else ''),
                        embedding_df.columns[1]: f'{method.upper()}2' + (f' ({explained_var[1]:.1f}%)' if explained_var is not None else '')
                    }
                )
            else:
                fig = px.scatter(
                    embedding_df, 
                    x=embedding_df.columns[0], 
                    y=embedding_df.columns[1],
                    title=f'Beta Diversity ({method.upper()})',
                    labels={
                        embedding_df.columns[0]: f'{method.upper()}1' + (f' ({explained_var[0]:.1f}%)' if explained_var is not None else ''),
                        embedding_df.columns[1]: f'{method.upper()}2' + (f' ({explained_var[1]:.1f}%)' if explained_var is not None else '')
                    }
                )
            
            # Save figure
            if self.output_dir:
                fig_path = os.path.join(self.output_dir, f'beta_diversity_{method}_interactive.html')
                fig.write_html(fig_path)
                print(f"Saved interactive figure to {fig_path}")
        
        else:
            # Create static plot with matplotlib
            fig, ax = plt.subplots(figsize=figsize)
            
            if 'Group' in embedding_df.columns:
                # Plot with group colors
                for group, group_data in embedding_df.groupby('Group'):
                    ax.scatter(
                        group_data[embedding_df.columns[0]], 
                        group_data[embedding_df.columns[1]],
                        label=group,
                        alpha=0.7
                    )
                ax.legend(title=group_col)
            else:
                # Plot without grouping
                ax.scatter(
                    embedding_df[embedding_df.columns[0]], 
                    embedding_df[embedding_df.columns[1]],
                    alpha=0.7
                )
            
            # Add labels
            if explained_var is not None:
                ax.set_xlabel(f'{method.upper()}1 ({explained_var[0]:.1f}%)')
                ax.set_ylabel(f'{method.upper()}2 ({explained_var[1]:.1f}%)')
            else:
                ax.set_xlabel(f'{method.upper()}1')
                ax.set_ylabel(f'{method.upper()}2')
            
            ax.set_title(f'Beta Diversity ({method.upper()})')
            ax.grid(alpha=0.3)
            
            # Save figure
            if self.output_dir:
                fig_path = os.path.join(self.output_dir, f'beta_diversity_{method}.png')
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"Saved figure to {fig_path}")
        
        # Store figure
        self.figures[f'beta_diversity_{method}'] = fig
        
        return fig
    
    def plot_heatmap(self, otu_data, metadata=None, group_col=None, top_n=50, 
                    method='average', metric='correlation', figsize=(16, 12)):
        """
        Plot heatmap of OTU abundances with hierarchical clustering.
        
        Parameters:
        -----------
        otu_data : pd.DataFrame
            DataFrame containing OTU counts
        metadata : pd.DataFrame, optional
            DataFrame containing metadata
        group_col : str, optional
            Column in metadata to group by
        top_n : int
            Number of top OTUs to include
        method : str
            Linkage method for hierarchical clustering
        metric : str
            Distance metric for hierarchical clustering
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        # Select top OTUs by mean abundance
        mean_abundance = otu_data.mean().sort_values(ascending=False)
        top_otus = mean_abundance.head(top_n).index
        data_subset = otu_data[top_otus]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create row colors if group information is available
        row_colors = None
        if group_col and metadata is not None:
            # Ensure indices match
            if not all(idx in metadata.index for idx in data_subset.index):
                raise ValueError("Not all OTU data indices are present in metadata")
            
            # Get group information
            groups = metadata.loc[data_subset.index, group_col]
            
            # Create color mapping
            unique_groups = groups.unique()
            color_map = dict(zip(unique_groups, sns.color_palette("Set2", len(unique_groups))))
            row_colors = groups.map(color_map)
        
        # Create clustered heatmap
        g = sns.clustermap(
            data_subset,
            method=method,
            metric=metric,
            figsize=figsize,
            cmap='viridis',
            row_colors=row_colors,
            xticklabels=True,
            yticklabels=False,
            cbar_kws={'label': 'Abundance'}
        )
        
        # Add title
        plt.suptitle(f'Heatmap of Top {top_n} OTUs', y=1.02)
        
        # Add legend if group information is available
        if row_colors is not None:
            handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in color_map.values()]
            plt.legend(
 
(Content truncated due to size limit. Use line ranges to read in chunks)