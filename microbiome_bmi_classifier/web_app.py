"""
Web-based frontend for GutCheck microbiome-based BMI classification.

This module provides a Flask-based web application with a React frontend
for pathologists, lab technicians, and researchers to use the GutCheck tool.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import uuid
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from datetime import datetime
import joblib

# Import GutCheck modules
from microbiome_bmi_classifier.data_processing import process_data, load_data
from microbiome_bmi_classifier.feature_extraction import extract_features
from microbiome_bmi_classifier.models import train_model, evaluate_model, compare_models
from microbiome_bmi_classifier.synthetic_data_enhanced import generate_synthetic_data
from microbiome_bmi_classifier.preprocessing import preprocess_data
from microbiome_bmi_classifier.evaluation import cross_validate, evaluate_metrics
from microbiome_bmi_classifier.visualization import plot_diversity, plot_abundance, plot_heatmap, plot_pca
from microbiome_bmi_classifier.feature_selection import select_features
from microbiome_bmi_classifier.hyperparameter_tuning import tune_hyperparameters
from microbiome_bmi_classifier.multi_class import train_multi_class_model
from microbiome_bmi_classifier.interpretability import interpret_model

# Create Flask app
app = Flask(__name__, 
            static_folder='../web/build/static',
            template_folder='../web/build')

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results')
ALLOWED_EXTENSIONS = {'csv', 'txt', 'tsv'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Set configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def allowed_file(filename):
    """
    Check if file has allowed extension.
    
    Parameters:
    -----------
    filename : str
        Filename to check
        
    Returns:
    --------
    bool
        True if file has allowed extension
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """
    Serve the main page.
    
    Returns:
    --------
    str
        Rendered HTML template
    """
    return render_template('index.html')


@app.route('/<path:path>')
def static_proxy(path):
    """
    Serve static files.
    
    Parameters:
    -----------
    path : str
        Path to static file
        
    Returns:
    --------
    Response
        Static file
    """
    return send_from_directory(app.template_folder, path)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload.
    
    Returns:
    --------
    Response
        JSON response with upload status
    """
    if 'otu_file' not in request.files:
        return jsonify({'error': 'No OTU file provided'}), 400
    
    otu_file = request.files['otu_file']
    
    if otu_file.filename == '':
        return jsonify({'error': 'No OTU file selected'}), 400
    
    if not allowed_file(otu_file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Create session ID
    session_id = str(uuid.uuid4())
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    # Save OTU file
    otu_filename = secure_filename(otu_file.filename)
    otu_path = os.path.join(session_folder, otu_filename)
    otu_file.save(otu_path)
    
    # Save metadata file if provided
    metadata_path = None
    if 'metadata_file' in request.files:
        metadata_file = request.files['metadata_file']
        if metadata_file.filename != '' and allowed_file(metadata_file.filename):
            metadata_filename = secure_filename(metadata_file.filename)
            metadata_path = os.path.join(session_folder, metadata_filename)
            metadata_file.save(metadata_path)
    
    return jsonify({
        'session_id': session_id,
        'otu_file': otu_filename,
        'metadata_file': os.path.basename(metadata_path) if metadata_path else None
    }), 200


@app.route('/api/train', methods=['POST'])
def train():
    """
    Train a model.
    
    Returns:
    --------
    Response
        JSON response with training results
    """
    # Get request data
    data = request.json
    
    # Validate required fields
    if 'session_id' not in data:
        return jsonify({'error': 'No session ID provided'}), 400
    
    # Get session folder
    session_id = data['session_id']
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    
    if not os.path.exists(session_folder):
        return jsonify({'error': 'Invalid session ID'}), 400
    
    # Get file paths
    otu_file = data.get('otu_file')
    metadata_file = data.get('metadata_file')
    
    if not otu_file:
        return jsonify({'error': 'No OTU file specified'}), 400
    
    otu_path = os.path.join(session_folder, otu_file)
    
    if not os.path.exists(otu_path):
        return jsonify({'error': 'OTU file not found'}), 400
    
    metadata_path = None
    if metadata_file:
        metadata_path = os.path.join(session_folder, metadata_file)
        if not os.path.exists(metadata_path):
            return jsonify({'error': 'Metadata file not found'}), 400
    
    # Get training parameters
    model_type = data.get('model_type', 'random_forest')
    preprocessing = data.get('preprocessing', 'standard')
    feature_selection = data.get('feature_selection', False)
    feature_selection_method = data.get('feature_selection_method', 'rfe')
    n_features = data.get('n_features', 20)
    cross_validation = data.get('cross_validation', False)
    cv_folds = data.get('cv_folds', 5)
    test_size = data.get('test_size', 0.2)
    random_state = data.get('random_state', 42)
    visualize = data.get('visualize', True)
    interpret = data.get('interpret', False)
    tune_hyperparameters = data.get('tune_hyperparameters', False)
    
    # Create results folder
    results_folder = os.path.join(app.config['RESULTS_FOLDER'], session_id)
    os.makedirs(results_folder, exist_ok=True)
    
    try:
        # Load data
        X, y = load_data(otu_path, metadata_path)
        
        # Preprocess data
        X_processed = preprocess_data(X, method=preprocessing)
        
        # Feature selection
        if feature_selection:
            X_processed, selected_features = select_features(
                X_processed, y, 
                method=feature_selection_method,
                n_features=n_features
            )
            
            # Save selected features
            selected_features_path = os.path.join(results_folder, 'selected_features.txt')
            with open(selected_features_path, 'w') as f:
                for feature in selected_features:
                    f.write(f"{feature}\n")
        
        # Hyperparameter tuning
        best_params = None
        if tune_hyperparameters:
            best_params, tuning_results = tune_hyperparameters(
                X_processed, y,
                model_type=model_type,
                method='grid',
                cv=cv_folds,
                random_state=random_state,
                output_dir=results_folder
            )
        
        # Train model
        model, X_train, X_test, y_train, y_test = train_model(
            X_processed, y,
            model_type=model_type,
            test_size=test_size,
            random_state=random_state,
            params=best_params
        )
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Cross-validation
        cv_results = None
        if cross_validation:
            cv_results = cross_validate(
                X_processed, y,
                model_type=model_type,
                cv=cv_folds,
                random_state=random_state,
                params=best_params
            )
        
        # Visualizations
        visualization_paths = {}
        if visualize:
            # Create visualizations directory
            vis_dir = os.path.join(results_folder, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # ROC curve
            try:
                from microbiome_bmi_classifier.evaluation import plot_roc_curve
                roc_path = os.path.join(vis_dir, 'roc_curve.png')
                plot_roc_curve(model, X_test, y_test, save_path=roc_path)
                visualization_paths['roc_curve'] = os.path.relpath(roc_path, app.config['RESULTS_FOLDER'])
            except:
                pass
            
            # Feature importance
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                feature_names = X.columns if hasattr(X, 'columns') else None
                importance_path = os.path.join(vis_dir, 'feature_importance.png')
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                else:
                    importances = np.abs(model.coef_[0])
                
                plt.figure(figsize=(12, 8))
                
                if feature_names is not None:
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    top_indices = indices[:20]  # Show top 20 features
                    
                    plt.barh(range(len(top_indices)), importances[top_indices])
                    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
                else:
                    # Sort features by importance
                    indices = np.argsort(importances)[::-1]
                    top_indices = indices[:20]  # Show top 20 features
                    
                    plt.barh(range(len(top_indices)), importances[top_indices])
                    plt.yticks(range(len(top_indices)), [f"Feature {i}" for i in top_indices])
                
                plt.xlabel('Feature Importance')
                plt.ylabel('Feature')
                plt.title('Top 20 Features by Importance')
                plt.tight_layout()
                plt.savefig(importance_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                visualization_paths['feature_importance'] = os.path.relpath(importance_path, app.config['RESULTS_FOLDER'])
            
            # PCA plot
            try:
                pca_path = os.path.join(vis_dir, 'pca_plot.png')
                if hasattr(X, 'values'):
                    plot_pca(X.values, y, save_path=pca_path)
                else:
                    plot_pca(X, y, save_path=pca_path)
                visualization_paths['pca'] = os.path.relpath(pca_path, app.config['RESULTS_FOLDER'])
            except:
                pass
        
        # Model interpretation
        interpretation_paths = {}
        if interpret:
            # Create interpretations directory
            interp_dir = os.path.join(results_folder, 'interpretations')
            os.makedirs(interp_dir, exist_ok=True)
            
            # Get feature names
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
            
            # Interpret model
            try:
                paths = interpret_model(
                    model, X_test, y_test,
                    feature_names=feature_names,
                    class_names=['Healthy', 'Obese'],
                    output_dir=interp_dir
                )
                
                # Convert paths to relative paths
                for key, path in paths.items():
                    if isinstance(path, str) and os.path.exists(path):
                        interpretation_paths[key] = os.path.relpath(path, app.config['RESULTS_FOLDER'])
            except Exception as e:
                interpretation_paths['error'] = str(e)
        
        # Save model
        model_path = os.path.join(results_folder, 'model.pkl')
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_type': model_type,
            'preprocessing': preprocessing,
            'feature_selection': feature_selection,
            'feature_selection_method': feature_selection_method if feature_selection else None,
            'n_features': n_features if feature_selection else X_processed.shape[1],
            'metrics': {k: float(v) for k, v in metrics.items() if k not in ['confusion_matrix', 'classification_report']},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'random_state': random_state
        }
        
        metadata_path = os.path.join(results_folder, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Prepare response
        response = {
            'success': True,
            'session_id': session_id,
            'model_path': os.path.relpath(model_path, app.config['RESULTS_FOLDER']),
            'metadata': metadata,
            'visualization_paths': visualization_paths,
            'interpretation_paths': interpretation_paths
        }
        
        # Add cross-validation results if available
        if cv_results:
            response['cv_results'] = {
                k: {'mean': float(v['mean']), 'std': float(v['std'])}
                for k, v in cv_results.items()
            }
        
        return jsonify(response), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make predictions with a trained model.
    
    Returns:
    --------
    Response
        JSON response with predictions
    """
    # Get request data
    data = request.json
    
    # Validate required fields
    if 'session_id' not in data:
        return jsonify({'error': 'No session ID provided'}), 400
    
    if 'model_path' not in data:
        return jsonify({'error': 'No model path provided'}), 400
    
    if 'otu_file' not in data:
        return jsonify({'error': 'No OTU file specified'}), 400
    
    # Get session folder
    session_id = data['session_id']
    session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    
    if not os.path.exists(session_folder):
        return jsonify({'error': 'Invalid session ID'}), 400
    
    # Get file paths
    otu_file = data['otu_file']
    otu_path = os.path.join(session_folder, otu_file)
    
    if not os.path.exists(otu_path):
        return jsonify({'error': 'OTU file not found'}), 400
    
    # Get model path
    model_path = os.path.join(app.config['RESULTS_FOLDER'], data['model_path'])
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 400
    
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Load data
        X = pd.read_csv(otu_path, index_col=0)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[:, 1].tolist()
        
        # Create results folder
        results_folder = os.path.join(app.config['RESULTS_FOLDER'
(Content truncated due to size limit. Use line ranges to read in chunks)