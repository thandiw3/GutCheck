"""
Integration test script for GutCheck challenge mitigation features.

This script tests the new features implemented to address the challenges
in microbiome analysis for clinical applications.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import GutCheck modules
from microbiome_bmi_classifier.absolute_abundance import (
    AbsoluteAbundanceEstimator, 
    simulate_spike_in_data,
    simulate_qpcr_data,
    simulate_flow_cytometry_data
)
from microbiome_bmi_classifier.functional_analysis import (
    FunctionalAnalyzer,
    predict_functional_profile,
    analyze_functional_diversity
)
from microbiome_bmi_classifier.reference_database import (
    ReferenceDatabase,
    create_reference_database,
    create_clinical_reference_database
)
from microbiome_bmi_classifier.contamination_control import (
    ContaminationController,
    identify_contaminants,
    remove_contaminants
)
from microbiome_bmi_classifier.validation_reporting import (
    ValidationReporter,
    validate_classification,
    validate_regression,
    validate_clustering,
    validate_diversity,
    generate_report
)

# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_output')
os.makedirs(output_dir, exist_ok=True)

def test_absolute_abundance():
    """Test absolute abundance estimation features."""
    print("\n=== Testing Absolute Abundance Estimation ===")
    
    # Create synthetic OTU table
    n_samples = 20
    n_taxa = 50
    
    # Generate random OTU table with relative abundances
    np.random.seed(42)
    otu_table = pd.DataFrame(
        np.random.dirichlet(np.ones(n_taxa) * 0.5, size=n_samples),
        columns=[f"Taxon_{i}" for i in range(n_taxa)]
    )
    
    # Test different estimation methods
    methods = ['total_sum_scaling', 'spike_in', 'computational']
    
    for method in methods:
        print(f"Testing {method} method...")
        
        if method == 'spike_in':
            # Simulate spike-in data
            otu_with_spike, spike_cols = simulate_spike_in_data(otu_table)
            
            # Estimate absolute abundance
            estimator = AbsoluteAbundanceEstimator(method=method)
            abs_abundance = estimator.fit_transform(otu_with_spike, spike_in_cols=spike_cols)
        elif method == 'flow_cytometry':
            # Simulate flow cytometry data
            flow_data = simulate_flow_cytometry_data(otu_table)
            
            # Estimate absolute abundance
            estimator = AbsoluteAbundanceEstimator(method=method)
            abs_abundance = estimator.fit_transform(otu_table, flow_cytometry_data=flow_data)
        elif method == 'qpcr_calibration':
            # Simulate qPCR data
            qpcr_data, selected_taxa = simulate_qpcr_data(otu_table)
            
            # Estimate absolute abundance
            estimator = AbsoluteAbundanceEstimator(method=method)
            abs_abundance = estimator.fit_transform(otu_table, qpcr_data=qpcr_data)
        else:
            # Use default method
            estimator = AbsoluteAbundanceEstimator(method=method)
            abs_abundance = estimator.fit_transform(otu_table)
        
        # Plot comparison
        fig = estimator.plot_comparison(otu_table, abs_abundance, n_taxa=5)
        fig.savefig(os.path.join(output_dir, f"abs_abundance_{method}.png"))
        plt.close(fig)
        
        print(f"  Transformed shape: {abs_abundance.shape}")
        print(f"  Min value: {abs_abundance.values.min()}")
        print(f"  Max value: {abs_abundance.values.max()}")
        print(f"  Mean value: {abs_abundance.values.mean()}")
    
    print("Absolute abundance estimation tests completed.")
    return True

def test_functional_analysis():
    """Test functional analysis capabilities."""
    print("\n=== Testing Functional Analysis ===")
    
    # Create synthetic OTU table
    n_samples = 30
    n_taxa = 50
    
    # Generate random OTU table with relative abundances
    np.random.seed(42)
    otu_table = pd.DataFrame(
        np.random.dirichlet(np.ones(n_taxa) * 0.5, size=n_samples),
        columns=[f"Genus_{i}" for i in range(n_taxa)]
    )
    
    # Create sample groups for testing
    groups = pd.Series(np.random.choice(['Healthy', 'Obese'], size=n_samples), 
                      index=otu_table.index)
    
    # Test functional prediction
    print("Testing functional prediction...")
    analyzer = FunctionalAnalyzer(method='picrust', database='kegg')
    functions = analyzer.predict_functions(otu_table)
    
    print(f"  Predicted functions shape: {functions.shape}")
    print(f"  Number of functions: {functions.shape[1]}")
    
    # Test functional diversity analysis
    print("Testing functional diversity analysis...")
    diversity = analyzer.analyze_functional_diversity(functions)
    
    print(f"  Diversity metrics: {', '.join(diversity.columns)}")
    
    # Test discriminating functions
    print("Testing discriminating functions identification...")
    discriminating = analyzer.identify_discriminating_functions(functions, groups)
    
    print(f"  Top discriminating function: {discriminating.index[0]}")
    print(f"  Score: {discriminating['score'].iloc[0]}")
    
    # Test visualization
    fig = analyzer.plot_functional_heatmap(functions, n_functions=10)
    fig.savefig(os.path.join(output_dir, "functional_heatmap.png"))
    plt.close(fig)
    
    fig = analyzer.plot_functional_pca(functions, color_by=groups)
    fig.savefig(os.path.join(output_dir, "functional_pca.png"))
    plt.close(fig)
    
    fig = analyzer.plot_discriminating_functions(functions, groups, discriminating)
    fig.savefig(os.path.join(output_dir, "discriminating_functions.png"))
    plt.close(fig)
    
    # Test BMI prediction from functions
    print("Testing BMI prediction from functions...")
    bmi_values = pd.Series(np.random.uniform(18, 35, size=n_samples), index=otu_table.index)
    bmi_pred, model = analyzer.predict_bmi_from_functions(functions, bmi_values)
    
    print(f"  Predicted BMI range: {bmi_pred.min():.2f} - {bmi_pred.max():.2f}")
    
    print("Functional analysis tests completed.")
    return True

def test_reference_database():
    """Test reference database improvements."""
    print("\n=== Testing Reference Database ===")
    
    # Create a test reference database
    print("Creating test reference database...")
    db = ReferenceDatabase(database_name='test_db', database_type='16S')
    
    # Add some test sequences
    db.sequences = {
        'seq1': 'ACGTACGTACGTACGT',
        'seq2': 'TGCATGCATGCATGCA',
        'seq3': 'ACGTACGTNNNNACGT',
        'seq4': 'TGCATGCATGCATGCA',  # Duplicate of seq2
    }
    
    # Add taxonomy
    db.taxonomy = {
        'seq1': 'k__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Streptococcaceae;g__Streptococcus;s__pneumoniae',
        'seq2': 'k__Bacteria;p__Bacteroidetes;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Bacteroides;s__fragilis',
        'seq3': 'k__Bacteria;p__Proteobacteria',  # Incomplete taxonomy
    }
    
    # Add metadata
    db.metadata = {
        'seq1': {'source': 'NCBI', 'isolation_source': 'human gut'},
        'seq2': {'source': 'NCBI', 'isolation_source': 'human gut'},
        'seq3': {'source': 'NCBI', 'isolation_source': 'soil'},
    }
    
    # Calculate quality scores
    db._calculate_quality_scores()
    
    # Validate database
    print("Validating database...")
    validation = db.validate()
    
    print(f"  Sequence count: {validation['sequence_count']}")
    print(f"  Taxonomy count: {validation['taxonomy_count']}")
    print(f"  Metadata count: {validation['metadata_count']}")
    print(f"  Average quality: {validation['average_quality']:.2f}")
    print(f"  Issues found: {len(validation['issues'])}")
    
    # Filter by quality
    print("Filtering by quality...")
    db.filter_by_quality(min_quality=0.5)
    
    print(f"  Sequences after filtering: {len(db.sequences)}")
    
    # Save database
    print("Saving database...")
    db_path = db.save(format='json')
    
    print(f"  Saved to: {db_path}")
    
    # Load database
    print("Loading database...")
    db2 = ReferenceDatabase()
    db2.load(db_path)
    
    print(f"  Loaded sequences: {len(db2.sequences)}")
    
    # Test search
    print("Testing search functionality...")
    results = db.search_by_taxonomy('Firmicutes')
    
    print(f"  Found {len(results)} sequences with Firmicutes")
    
    # Test clinical annotations
    print("Testing clinical annotations...")
    clinical_data = {
        'seq1': {'disease': 'pneumonia', 'pathogenicity': 'high'},
        'seq2': {'disease': 'inflammatory bowel disease', 'pathogenicity': 'medium'}
    }
    
    db.add_clinical_annotations(clinical_data)
    
    print(f"  Clinical annotations for seq1: {db.metadata['seq1'].get('clinical', {})}")
    
    print("Reference database tests completed.")
    return True

def test_contamination_control():
    """Test contamination control features."""
    print("\n=== Testing Contamination Control ===")
    
    # Create synthetic OTU table
    n_samples = 30
    n_taxa = 50
    
    # Generate random OTU table with relative abundances
    np.random.seed(42)
    otu_table = pd.DataFrame(
        np.random.dirichlet(np.ones(n_taxa) * 0.5, size=n_samples),
        columns=[f"Taxon_{i}" for i in range(n_taxa)]
    )
    
    # Add known contaminants
    contaminants = ['Taxon_0', 'Taxon_1', 'Taxon_2']
    for contaminant in contaminants:
        # Increase abundance of contaminants
        otu_table[contaminant] = otu_table[contaminant] * 2
    
    # Normalize to maintain relative abundance
    row_sums = otu_table.sum(axis=1)
    for idx in otu_table.index:
        otu_table.loc[idx] = otu_table.loc[idx] / row_sums[idx]
    
    # Create negative controls
    neg_controls = pd.DataFrame(
        np.zeros((5, n_taxa)),
        columns=otu_table.columns
    )
    
    # Add contaminants to negative controls
    for contaminant in contaminants:
        neg_controls[contaminant] = np.random.uniform(0.2, 0.5, size=5)
    
    # Normalize negative controls
    row_sums = neg_controls.sum(axis=1)
    for idx in neg_controls.index:
        neg_controls.loc[idx] = neg_controls.loc[idx] / row_sums[idx]
    
    # Test different contamination identification methods
    methods = ['frequency', 'prevalence', 'negative_control']
    
    for method in methods:
        print(f"Testing {method} method...")
        
        if method == 'negative_control':
            # Create controller with negative controls
            controller = ContaminationController(method=method)
            controller.add_negative_controls(neg_controls)
            
            # Identify contaminants
            identified = controller.identify_contaminants(otu_table, threshold=0.1)
        else:
            # Create controller
            controller = ContaminationController(method=method)
            
            # Create environment profile
            controller.create_environment_profile()
            
            # Identify contaminants
            identified = controller.identify_contaminants(otu_table, threshold=0.5)
        
        # Count identified contaminants
        n_identified = sum(identified['is_contaminant'])
        print(f"  Identified {n_identified} contaminants")
        
        # Check if known contaminants were identified
        known_identified = sum(identified.loc[contaminants, 'is_contaminant'])
        print(f"  Identified {known_identified}/{len(contaminants)} known contaminants")
        
        # Test contaminant removal
        cleaned = controller.remove_contaminants(otu_table, identified, method='subtract')
        
        print(f"  Original mean of contaminants: {otu_table[contaminants].values.mean():.4f}")
        print(f"  Cleaned mean of contaminants: {cleaned[contaminants].values.mean():.4f}")
        
        # Plot contaminants
        fig = controller.plot_contaminants(otu_table, identified)
        fig.savefig(os.path.join(output_dir, f"contaminants_{method}.png"))
        plt.close(fig)
        
        # Plot before/after
        fig = controller.plot_before_after(otu_table, cleaned, identified, n_samples=3)
        fig.savefig(os.path.join(output_dir, f"contaminants_removal_{method}.png"))
        plt.close(fig)
    
    print("Contamination control tests completed.")
    return True

def test_validation_reporting():
    """Test validation and reporting methods."""
    print("\n=== Testing Validation and Reporting ===")
    
    # Test classification validation
    print("Testing classification validation...")
    
    # Create synthetic classification data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Validate classification
    reporter = ValidationReporter(validation_type='classification')
    results = reporter.validate_classification(
        y_test, y_pred, y_prob, class_names=['Class_0', 'Class_1'],
        cv=5, X=X_test, model=model
    )
    
    print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"  F1 Score: {results['metrics']['f1']:.4f}")
    
    # Generate reports
    for format in ['html', 'markdown', 'json']:
        report = reporter.generate_report(
            output_file=os.path.join(output_dir, f"classification_report.{format}"),
            report_format=format
        )
        print(f"  Generated {format} report")
    
    # Test regression validation
    print("Testing regression validation...")
    
    # Create synthetic regression data
    X = np.random.rand(100, 10)
    y = np.random.rand(100) * 10 + X[:, 0] * 5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Validate regression
    reporter = ValidationReporter(validation_type='regression')
    results = reporter.validate_regression(
        y_test, y_pred, cv=5, X=X_test, model=model
    )
    
    print(f"  R²: {results['metrics']['r2']:.4f}")
    print(f"  RMSE: {results['metrics']['rmse']:.4f}")
    
    # Generate report
    report = reporter.generate_report(
        output_file=os.path.join(output_dir, "regression_report.html")
    )
    print("  Generated HTML report")
    
    # Test clustering validation
    print("Testing clustering validation...")
    
    # Create synthetic clustering data
    from sklearn.cluster import KMeans
    
    # Generate data with clusters
    X, y_true = make_classification(n_samples=100, n_features=10, n_classes=3, 
                                  n_clusters_per_class=1, random_state=42)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Validate clustering
    reporter = ValidationReporter(validation_type='clustering')
    results = reporter.validate_clustering(
        X, labels, true_labels=y_true, 
        feature_names=[f"Feature_{i}" for i in range(X.shape[1])]
    )
    
    print(f"  Silhouette score: {results['internal_metrics']['silhouette']:.4f}")
    print(f"  Adjusted Rand index: {results['external_metrics']['adjusted_rand']:.4f}")
    
    # Generate report
    report = reporter.generate_report(
        output_file=os.path.join(output_dir, "clustering_report.html")
    )
    print("  Generated HTML report")
    
    # Test diversity validation
    print("Testing diversity validation...")
    
    # Create synthetic OTU table
    n_samples = 30
    n_taxa = 50
    
    # Generate random OTU table with relative abundances
    np.random.seed(42)
    otu_table = pd.DataFrame(
        np.random.dirichlet(np.ones(n_taxa) * 0.5, size=n_samples),
        columns=[f"Taxon_{i}" for i in range(n_taxa)]
    )
    
    # Create metadata
    metadata = pd.DataFrame({
        'group': np.random.choice(['Healthy', 'Obese'], size=n_samples)
    }, index=otu_table.index)
    
    # Validate diversity
    reporter = ValidationReporter(validation_type='diversity')
    results = reporter.validate_diversity(
        otu_table, metadata=metadata, group_col='group'
    )
    
    print(f"  Alpha diversity metrics: {', '.join(results['alpha_diversity'].keys())}")
    print(f"  Beta diversity metrics: {', '.join(results['beta_diversity'].keys())}")
    
    # Generate report
    report = reporter.generate_report(
        output_file=os.path.join(output_dir, "diversity_report.html")
    )
    print("  Generated HTML report")
    
    print("Validation and reporting tests completed.")
    return True

def run_all_tests():
    """Run all tests for challenge mitigation features."""
    print("Starting tests for GutCheck challenge mitigation features...")
    
    # Run tests
    tests = [
        test_absolute_abundance,
        test_functional_analysis,
        test_reference_database,
        test_contamination_control,
        test_validation_reporting
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Error in {test.__name__}: {str(e)}")
            results.append(False)
    
    # Print summary
    print("\n=== Test Summary ===")
    for i, test in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{test.__name__}: {status}")
    
    # Overall result
    if all(results):
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
    
    return all(results)

if __name__ == "__main__":
    run_all_tests()
