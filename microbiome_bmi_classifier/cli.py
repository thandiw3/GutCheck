import argparse
import os
from .synthetic_data import create_synthetic_data
from .feature_extraction import extract_features
from .data_processing import preprocess_data, split_data
from .classification import train_model, evaluate_model, cross_validate_model

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="GutCheck: Microbiome BMI Classifier")

    # Define command-line arguments
    parser.add_argument('--generate-synthetic', action='store_true', help='Generate synthetic data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--cross-validate', action='store_true', help='Cross-validate the model')
    parser.add_argument('--output-dir', type=str, default='./', help='Directory to save output')

    # Parse arguments
    args = parser.parse_args()

    # Step 1: Generate synthetic data
    if args.generate_synthetic:
        print("Generating synthetic data...")
        data = create_synthetic_data()

    # Step 2: Extract features from synthetic data
    print("Extracting features...")
    features = extract_features(data)

    # Step 3: Preprocess the features
    print("Preprocessing data...")
    X = preprocess_data(features)
    y = data[["Label"]]  # Only label for now

    # Step 4: Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(features)

    # Step 5: Train the model
    if args.train:
        print("Training the model...")
        model = train_model(X_train, y_train)

    # Step 6: Evaluate the model
    if args.evaluate:
        print("Evaluating the model...")
        evaluate_model(model, X_test, y_test)

    # Step 7: Cross-validation
    if args.cross_validate:
        print("Cross-validating the model...")
        cross_validate_model(model, X, y)

if __name__ == "__main__":
    main()

if __name__ == '__main__':
    main()
