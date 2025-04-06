import argparse
import os
from .synthetic_data import create_synthetic_data
from .data_processing import load_data, preprocess_data
from .feature_extraction import extract_features
from .classification import train_model, evaluate_model, cross_validate_model

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Microbiome BMI Classifier")

    # Options for generating synthetic data
    parser.add_argument('--generate-synthetic', action='store_true', help="Generate synthetic data for testing purposes.")
    
    # Options for loading real data
    parser.add_argument('--otu-file', type=str, help="OTU data file (CSV).")
    parser.add_argument('--metadata-file', type=str, help="Metadata file (CSV).")

    # Model training and evaluation options
    parser.add_argument('--train', action='store_true', help="Train the model.")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the model.")
    parser.add_argument('--model-file', type=str, help="Path to the trained model for evaluation.")
    parser.add_argument('--cross-validate', action='store_true', help="Perform cross-validation on the model.")

    parser.add_argument('--output-dir', type=str, default='.', help="Directory to save the output files (default: current directory).")
    parser.add_argument('--output-file', type=str, default="features.csv", help="Output file for feature extraction.")
    
    args = parser.parse_args()

    # Process synthetic or real data
    if args.generate_synthetic:
        print("Generating synthetic data...")
        data = create_synthetic_data()
    elif args.otu_file and args.metadata_file:
        print(f"Loading data from {args.otu_file} and {args.metadata_file}...")
        data = load_data(args.otu_file)
        metadata = load_data(args.metadata_file)
        data = preprocess_data(data)
        data = pd.concat([data, metadata], axis=1)
    else:
        print("Error: Either synthetic data generation or OTU and metadata files must be provided.")
        return

    # Feature extraction
    print("Extracting features...")
    extract_features(data, args.output_file)

    # Train model if requested
    if args.train:
        print("Training the model...")
        train_model(args.output_file, 'trained_model.pkl')

    # Evaluate the model if requested
    if args.evaluate:
        print("Evaluating the model...")
        if not args.model_file:
            print("Error: Please provide a model file for evaluation.")
            return
        evaluate_model(args.output_file, args.model_file)

    # Cross-validation if requested
    if args.cross_validate:
        print("Performing cross-validation...")
        cross_validate_model(args.output_file)

if __name__ == '__main__':
    main()
