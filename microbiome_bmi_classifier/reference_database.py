"""
Module for improved reference database management in microbiome analysis.

This module provides methods for creating, validating, and using high-quality
reference databases for microbiome analysis, addressing the challenge of
database quality issues in clinical microbiome applications.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import requests
import gzip
import shutil
import json
import pickle
import hashlib
import warnings
import datetime
import sqlite3
from collections import defaultdict

class ReferenceDatabase:
    """
    Class for managing reference databases for microbiome analysis.
    
    This class implements methods for creating, validating, and using high-quality
    reference databases, addressing the challenge of database quality issues in
    clinical microbiome applications.
    """
    
    def __init__(self, database_name='gutcheck_reference', database_type='16S', 
                 version='1.0', verbose=False):
        """
        Initialize the ReferenceDatabase.
        
        Parameters:
        -----------
        database_name : str, default='gutcheck_reference'
            Name of the reference database
        database_type : str, default='16S'
            Type of reference database.
            Options: '16S', 'ITS', 'WGS', 'functional'
        version : str, default='1.0'
            Version of the database
        verbose : bool, default=False
            Whether to print verbose output
        """
        self.database_name = database_name
        self.database_type = database_type
        self.version = version
        self.verbose = verbose
        self.sequences = {}
        self.taxonomy = {}
        self.metadata = {}
        self.quality_scores = {}
        self.database_path = None
        self.is_loaded = False
        
        # Validate database type
        valid_types = ['16S', 'ITS', 'WGS', 'functional']
        if database_type not in valid_types:
            raise ValueError(f"Database type must be one of {valid_types}")
        
        # Set up database directory
        self.database_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data',
            'reference_databases',
            f"{database_name}_v{version}"
        )
        
        # Create directory if it doesn't exist
        os.makedirs(self.database_dir, exist_ok=True)
    
    def import_from_sources(self, sources, merge_strategy='consensus'):
        """
        Import sequences and taxonomy from multiple reference sources.
        
        Parameters:
        -----------
        sources : list of dict
            List of source configurations, each with:
            - 'name': Source name
            - 'url' or 'path': URL or file path to the source
            - 'format': Format of the source (fasta, genbank, etc.)
            - 'weight': Weight for consensus merging (0-1)
        merge_strategy : str, default='consensus'
            Strategy for merging data from multiple sources.
            Options: 'consensus', 'weighted', 'priority'
            
        Returns:
        --------
        self : ReferenceDatabase
            Updated database object
        """
        if self.verbose:
            print(f"Importing data from {len(sources)} sources")
        
        # Initialize containers for each source
        source_sequences = {}
        source_taxonomy = {}
        source_metadata = {}
        
        # Import data from each source
        for i, source in enumerate(sources):
            source_name = source.get('name', f"Source_{i+1}")
            
            if self.verbose:
                print(f"Importing from {source_name}")
            
            # Determine source location
            if 'url' in source:
                # Download from URL
                source_data = self._download_source(source['url'], source.get('format', 'fasta'))
            elif 'path' in source:
                # Load from file
                source_data = self._load_source_file(source['path'], source.get('format', 'fasta'))
            else:
                raise ValueError(f"Source {source_name} must have 'url' or 'path'")
            
            # Store data for this source
            source_sequences[source_name] = source_data.get('sequences', {})
            source_taxonomy[source_name] = source_data.get('taxonomy', {})
            source_metadata[source_name] = source_data.get('metadata', {})
        
        # Merge data from all sources
        if merge_strategy == 'consensus':
            self._merge_by_consensus(source_sequences, source_taxonomy, source_metadata)
        elif merge_strategy == 'weighted':
            weights = {s.get('name', f"Source_{i+1}"): s.get('weight', 1.0) 
                      for i, s in enumerate(sources)}
            self._merge_by_weight(source_sequences, source_taxonomy, source_metadata, weights)
        elif merge_strategy == 'priority':
            # Use sources in the order they were provided
            priority_order = [s.get('name', f"Source_{i+1}") for i, s in enumerate(sources)]
            self._merge_by_priority(source_sequences, source_taxonomy, source_metadata, priority_order)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Calculate quality scores
        self._calculate_quality_scores()
        
        return self
    
    def _download_source(self, url, format='fasta'):
        """
        Download reference data from a URL.
        
        Parameters:
        -----------
        url : str
            URL to download from
        format : str, default='fasta'
            Format of the data
            
        Returns:
        --------
        source_data : dict
            Dictionary with sequences, taxonomy, and metadata
        """
        if self.verbose:
            print(f"Downloading from {url}")
        
        # Create temporary file
        temp_file = os.path.join(self.database_dir, 'temp_download')
        
        # Download file
        response = requests.get(url, stream=True)
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Check if file is gzipped
        if url.endswith('.gz'):
            # Decompress
            with gzip.open(temp_file, 'rb') as f_in:
                with open(temp_file + '_decompressed', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Replace temp file with decompressed version
            os.remove(temp_file)
            os.rename(temp_file + '_decompressed', temp_file)
        
        # Parse file based on format
        source_data = self._parse_file(temp_file, format)
        
        # Clean up
        os.remove(temp_file)
        
        return source_data
    
    def _load_source_file(self, file_path, format='fasta'):
        """
        Load reference data from a local file.
        
        Parameters:
        -----------
        file_path : str
            Path to the file
        format : str, default='fasta'
            Format of the data
            
        Returns:
        --------
        source_data : dict
            Dictionary with sequences, taxonomy, and metadata
        """
        if self.verbose:
            print(f"Loading from {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if file is gzipped
        if file_path.endswith('.gz'):
            # Create temporary file
            temp_file = os.path.join(self.database_dir, 'temp_file')
            
            # Decompress
            with gzip.open(file_path, 'rb') as f_in:
                with open(temp_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Parse decompressed file
            source_data = self._parse_file(temp_file, format)
            
            # Clean up
            os.remove(temp_file)
        else:
            # Parse file directly
            source_data = self._parse_file(file_path, format)
        
        return source_data
    
    def _parse_file(self, file_path, format):
        """
        Parse reference data file.
        
        Parameters:
        -----------
        file_path : str
            Path to the file
        format : str
            Format of the data
            
        Returns:
        --------
        source_data : dict
            Dictionary with sequences, taxonomy, and metadata
        """
        sequences = {}
        taxonomy = {}
        metadata = {}
        
        if format == 'fasta':
            # Parse FASTA file
            for record in SeqIO.parse(file_path, 'fasta'):
                seq_id = record.id
                sequences[seq_id] = str(record.seq)
                
                # Extract taxonomy from description if available
                if 'taxonomy=' in record.description:
                    tax_str = record.description.split('taxonomy=')[1].split()[0]
                    taxonomy[seq_id] = tax_str
                
                # Extract other metadata
                metadata[seq_id] = {'description': record.description}
        
        elif format == 'genbank':
            # Parse GenBank file
            for record in SeqIO.parse(file_path, 'genbank'):
                seq_id = record.id
                sequences[seq_id] = str(record.seq)
                
                # Extract taxonomy
                if 'organism' in record.annotations:
                    taxonomy[seq_id] = record.annotations['organism']
                
                # Extract other metadata
                metadata[seq_id] = record.annotations
        
        elif format == 'csv':
            # Parse CSV file
            df = pd.read_csv(file_path)
            
            # Extract sequences
            if 'sequence' in df.columns:
                for idx, row in df.iterrows():
                    seq_id = row.get('id', f"Seq_{idx}")
                    sequences[seq_id] = row['sequence']
            
            # Extract taxonomy
            if 'taxonomy' in df.columns:
                for idx, row in df.iterrows():
                    seq_id = row.get('id', f"Seq_{idx}")
                    taxonomy[seq_id] = row['taxonomy']
            
            # Extract other metadata
            for idx, row in df.iterrows():
                seq_id = row.get('id', f"Seq_{idx}")
                metadata[seq_id] = row.to_dict()
        
        elif format == 'json':
            # Parse JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract data based on JSON structure
            if 'sequences' in data:
                sequences = data['sequences']
            
            if 'taxonomy' in data:
                taxonomy = data['taxonomy']
            
            if 'metadata' in data:
                metadata = data['metadata']
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return {
            'sequences': sequences,
            'taxonomy': taxonomy,
            'metadata': metadata
        }
    
    def _merge_by_consensus(self, source_sequences, source_taxonomy, source_metadata):
        """
        Merge data from multiple sources using consensus approach.
        
        Parameters:
        -----------
        source_sequences : dict
            Dictionary of sequences from each source
        source_taxonomy : dict
            Dictionary of taxonomy from each source
        source_metadata : dict
            Dictionary of metadata from each source
        """
        if self.verbose:
            print("Merging data using consensus approach")
        
        # Collect all sequence IDs
        all_seq_ids = set()
        for sequences in source_sequences.values():
            all_seq_ids.update(sequences.keys())
        
        # Merge sequences
        for seq_id in all_seq_ids:
            # Collect all versions of this sequence
            seq_versions = []
            for source, sequences in source_sequences.items():
                if seq_id in sequences:
                    seq_versions.append(sequences[seq_id])
            
            # Use most common sequence
            if seq_versions:
                from collections import Counter
                seq_counter = Counter(seq_versions)
                self.sequences[seq_id] = seq_counter.most_common(1)[0][0]
        
        # Merge taxonomy
        for seq_id in all_seq_ids:
            # Collect all versions of taxonomy
            tax_versions = []
            for source, taxonomies in source_taxonomy.items():
                if seq_id in taxonomies:
                    tax_versions.append(taxonomies[seq_id])
            
            # Use most common taxonomy
            if tax_versions:
                from collections import Counter
                tax_counter = Counter(tax_versions)
                self.taxonomy[seq_id] = tax_counter.most_common(1)[0][0]
        
        # Merge metadata (combine all fields)
        for seq_id in all_seq_ids:
            self.metadata[seq_id] = {}
            
            for source, metadata in source_metadata.items():
                if seq_id in metadata:
                    # Update with metadata from this source
                    self.metadata[seq_id].update(metadata[seq_id])
    
    def _merge_by_weight(self, source_sequences, source_taxonomy, source_metadata, weights):
        """
        Merge data from multiple sources using weighted approach.
        
        Parameters:
        -----------
        source_sequences : dict
            Dictionary of sequences from each source
        source_taxonomy : dict
            Dictionary of taxonomy from each source
        source_metadata : dict
            Dictionary of metadata from each source
        weights : dict
            Dictionary of weights for each source
        """
        if self.verbose:
            print("Merging data using weighted approach")
        
        # Collect all sequence IDs
        all_seq_ids = set()
        for sequences in source_sequences.values():
            all_seq_ids.update(sequences.keys())
        
        # Merge sequences
        for seq_id in all_seq_ids:
            # Collect all versions of this sequence with weights
            seq_versions = []
            for source, sequences in source_sequences.items():
                if seq_id in sequences:
                    weight = weights.get(source, 1.0)
                    seq_versions.extend([sequences[seq_id]] * int(weight * 10))
            
            # Use most common sequence
            if seq_versions:
                from collections import Counter
                seq_counter = Counter(seq_versions)
                self.sequences[seq_id] = seq_counter.most_common(1)[0][0]
        
        # Merge taxonomy
        for seq_id in all_seq_ids:
            # Collect all versions of taxonomy with weights
            tax_versions = []
            for source, taxonomies in source_taxonomy.items():
                if seq_id in taxonomies:
                    weight = weights.get(source, 1.0)
                    tax_versions.extend([taxonomies[seq_id]] * int(weight * 10))
            
            # Use most common taxonomy
            if tax_versions:
                from collections import Counter
                tax_counter = Counter(tax_versions)
                self.taxonomy[seq_id] = tax_counter.most_common(1)[0][0]
        
        # Merge metadata (prioritize sources with higher weights)
        for seq_id in all_seq_ids:
            self.metadata[seq_id] = {}
            
      
(Content truncated due to size limit. Use line ranges to read in chunks)