import os
import sys
import argparse
from .data_processing import load_data, preprocess_data, split_data
from .feature_extraction import extract_features
from .classification import train_model, evaluate_model, cross_validate_model
from .synthetic_data import create_synthetic_data

def main():
    parser = argparse.ArgumentParser(description="GutCheck: Microbiome BMI Classifier")
    
    # Optional arguments for file paths
    parser.add_argument('--otu-file', type=str, help="OTU data file (CSV). If not provided, synthetic data will be generated.")
    parser.add_argument('--metadata-file', type=str, help="Metadata file (CSV). If not provided, synthetic data will be generated.")
    parser.add_argument('--output-dir', type=str, default='.', help="Directory to save output files.")
    parser.add_argument('--cross-validate', action='store_true', help="Perform cross-validation.")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the trained model on test data.")
    parser.add_argument('--model-file', type=str, help="Path to the trained model for evaluation.")
    
    args = parser.parse_args()

    # If OTU and metadata files are provided, use them. Otherwise, generate synthetic data.
    if args.otu_file and args.metadata_file:
        print("Loading real data...")
        # Load and process real data
        otu_data = load_data(args.otu_file)
        metadata = load_data(args.metadata_file)
    else:
        print("Generating synthetic data...")
        # Generate synthetic data for testing
        otu_data, metadata = create_synthetic_data()

    # Preprocess data
    print("Preprocessing data...")
    data = preprocess_data(otu_data, metadata)

    # Split the data into training and testing sets
    print("Splitting data...")
    train_data, test_data = split_data(data)

    # Extract features from the data
    print("Extracting features...")
    features = extract_features(data)

    # If cross-validation is enabled, perform it
    if args.cross_validate:
        print("Performing cross-validation...")
        model = train_model(features, train_data["BMI"])  # Assuming 'BMI' is the label column
        cv_scores = cross_validate_model(model, features, train_data["BMI"])
        print(f"Cross-validation scores: {cv_scores}")
        return

    # Train the model
    print("Training the model...")
    model = train_model(features, train_data["BMI"])

    # Save the trained model
    print(f"Saving model to {args.output_dir}...")
    model_file = os.path.join(args.output_dir, 'trained_model.pkl')
    joblib.dump(model, model_file)

    # If evaluate flag is set, evaluate the model on test data
    if args.evaluate and args.model_file:
        print(f"Evaluating model from {args.model_file}...")
        model = joblib.load(args.model_file)
        test_features = extract_features(test_data)
        accuracy = evaluate_model(model, test_features, test_data["BMI"])
        print(f"Model accuracy on test data: {accuracy}")

    print(f"Process finished successfully!")

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
