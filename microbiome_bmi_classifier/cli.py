import argparse
import os
from microbiome_bmi_classifier.data_preprocess import load_data, preprocess_data, split_data
from microbiome_bmi_classifier.feature_extraction import extract_features
from microbiome_bmi_classifier.classification import train_model, evaluate_model
from microbiome_bmi_classifier.synthetic_data import create_synthetic_data
import pickle

def main():
    # Setup command line argument parser
    parser = argparse.ArgumentParser(description="GutCheck: Microbiome-based classification tool")
    
    # Arguments for data processing and model training
    parser.add_argument('--otu-file', type=str, help="OTU data file (CSV).")
    parser.add_argument('--metadata-file', type=str, help="Metadata file (CSV).")
    parser.add_argument('--output-dir', type=str, default='.', help="Directory to save the output files (default: current directory).")
    parser.add_argument('--train', action='store_true', help="Train the model.")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the model.")
    parser.add_argument('--model-file', type=str, help="Path to the trained model for evaluation.")
    parser.add_argument('--generate-synthetic', action='store_true', help="Generate synthetic data for testing purposes.")
    
    args = parser.parse_args()

    # Handle synthetic data generation if no input files are provided
    if args.generate_synthetic:
        print("Generating synthetic data...")
        data = create_synthetic_data()
    elif args.otu_file and args.metadata_file:
        print(f"Loading OTU data from {args.otu_file} and metadata from {args.metadata_file}...")
        data = load_data(args.otu_file, args.metadata_file)
    else:
        print("Error: Please provide either input files or use --generate-synthetic.")
        return

    # Preprocessing step
    print("Preprocessing the data...")
    data = preprocess_data(data)

    # Feature extraction
    print("Extracting features from the data...")
    features = extract_features(data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(features)

    # Training the model if --train flag is set
    if args.train:
        print("Training the model...")
        model = train_model(X_train, y_train)

        # Save trained model
        model_file = os.path.join(args.output_dir, 'trained_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model trained and saved to {model_file}")

    # Evaluate the model if --evaluate flag is set
    if args.evaluate:
        if not args.model_file:
            print("Error: Please provide the model file path with --model-file for evaluation.")
            return

        print(f"Evaluating the model from {args.model_file}...")
        with open(args.model_file, 'rb') as f:
            model = pickle.load(f)
        
        evaluation_results = evaluate_model(model, X_test, y_test)
        print(f"Evaluation Results: {evaluation_results}")
        # Optionally save evaluation results to a file
        with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(str(evaluation_results))
        print(f"Evaluation results saved to {os.path.join(args.output_dir, 'evaluation_results.txt')}")

if __name__ == "__main__":
    main()
