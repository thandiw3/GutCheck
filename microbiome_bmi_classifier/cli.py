import argparse
import os
from .data_processing import load_data, preprocess_data, split_data
from .feature_extraction import extract_features
from .synthetic_data import create_synthetic_data
from .classification import train_model, evaluate_model, cross_validate_model

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="GutCheck - Microbiome BMI Classifier")
    
    # Arguments for data processing and model training
    parser.add_argument('--otu-file', type=str, help="OTU data file (CSV). Optional if --generate-synthetic is used.")
    parser.add_argument('--metadata-file', type=str, help="Metadata file (CSV). Optional if --generate-synthetic is used.")
    parser.add_argument('--output-dir', type=str, default='.', help="Directory to save the output files (default: current directory).")
    parser.add_argument('--train', action='store_true', help="Train the model.")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the model.")
    parser.add_argument('--model-file', type=str, help="Path to the trained model for evaluation.")
    parser.add_argument('--generate-synthetic', action='store_true', help="Generate synthetic data for testing purposes.")
    
    # Parse the arguments
    args = parser.parse_args()

    # Load or generate data
    if args.generate_synthetic:
        # Generate synthetic data
        otu_data, metadata, combined_data = create_synthetic_data()
        print("Synthetic OTU data, metadata, and combined data generated.")
        data = combined_data
    elif args.otu_file and args.metadata_file:
        # Load data from files
        otu_data = load_data(args.otu_file)
        metadata = load_data(args.metadata_file)
        data = preprocess_data(otu_data, metadata)
    else:
        print("Error: Please provide either input files or use --generate-synthetic.")
        return

    # Extract features
    features = extract_features(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(features, metadata)

    # Training or evaluating model
    if args.train:
        model = train_model(X_train, y_train)
        print("Model trained successfully.")
        model_path = os.path.join(args.output_dir, 'trained_model.pkl')
        # Save the trained model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Trained model saved at {model_path}")
    
    if args.evaluate:
        if args.model_file:
            # Load the trained model
            with open(args.model_file, 'rb') as f:
                model = pickle.load(f)
            evaluate_model(model, X_test, y_test)
        else:
            print("Error: Please provide the path to a trained model with --model-file for evaluation.")
            return

    # Cross-validation (optional)
    if not args.train and not args.evaluate:
        cross_validate_model(X_train, y_train)

if __name__ == "__main__":
    main()
