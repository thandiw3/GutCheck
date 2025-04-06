import argparse
import os
import pandas as pd
from microbiome_bmi_classifier.data_preprocess import load_data, preprocess_data, split_data
from microbiome_bmi_classifier.synthetic_data import create_synthetic_data
from microbiome_bmi_classifier.feature_extraction import extract_features
from microbiome_bmi_classifier.classification import train_model, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="GutCheck: Microbiome-based classification tool")
    
    # Add arguments for different functionalities
    parser.add_argument('--otu-file', type=str, help="OTU data file (CSV).")
    parser.add_argument('--metadata-file', type=str, help="Metadata file (CSV).")
    parser.add_argument('--generate-synthetic-data', action='store_true', help="Generate synthetic microbiome data for testing.")
    parser.add_argument('--preprocess', action='store_true', help="Preprocess the data (drop NAs).")
    parser.add_argument('--extract-features', action='store_true', help="Extract features from the data.")
    parser.add_argument('--train', action='store_true', help="Train the model.")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the trained model.")
    parser.add_argument('--output-dir', type=str, default='.', help="Directory to save the outputs (default: current directory).")
    parser.add_argument('--model-file', type=str, help="Path to the trained model file for evaluation.")
    
    args = parser.parse_args()

    # Generate synthetic data if specified
    if args.generate_synthetic_data:
        synthetic_data = create_synthetic_data()
        print("Synthetic data generated.")
        synthetic_data.to_csv(os.path.join(args.output_dir, 'synthetic_data.csv'))

    # Load data if OTU and metadata files are provided
    if args.otu_file and args.metadata_file:
        print(f"Loading data from {args.otu_file} and {args.metadata_file}...")
        data = load_data(args.otu_file, args.metadata_file)
    else:
        print("No OTU or metadata files provided. Generating synthetic data...")
        data = create_synthetic_data()
    
    # Preprocess the data if the --preprocess flag is set
    if args.preprocess:
        print("Preprocessing the data...")
        data = preprocess_data(data)
        data.to_csv(os.path.join(args.output_dir, 'preprocessed_data.csv'))
        print(f"Preprocessed data saved to {os.path.join(args.output_dir, 'preprocessed_data.csv')}")
    
    # Extract features if the --extract-features flag is set
    if args.extract_features:
        print("Extracting features from the data...")
        features = extract_features(data)
        features.to_csv(os.path.join(args.output_dir, 'extracted_features.csv'))
        print(f"Features extracted and saved to {os.path.join(args.output_dir, 'extracted_features.csv')}")

    # Train the model if the --train flag is set
    if args.train:
        print("Training the model...")
        X_train, X_test, y_train, y_test = split_data(data)
        model = train_model(X_train, y_train)
        model_file = os.path.join(args.output_dir, 'trained_model.pkl')
        pd.to_pickle(model, model_file)
        print(f"Model trained and saved to {model_file}")
    
    # Evaluate the model if the --evaluate flag is set
    if args.evaluate:
        if not args.model_file:
            print("Error: Please provide a model file using --model-file to evaluate.")
            return
        print(f"Evaluating the model from {args.model_file}...")
        model = pd.read_pickle(args.model_file)
        X_train, X_test, y_train, y_test = split_data(data)
        evaluation_results = evaluate_model(model, X_test, y_test)
        print(f"Evaluation Results: {evaluation_results}")
        # Optionally, save the evaluation results to a file
        with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
            f.write(str(evaluation_results))
        print(f"Evaluation results saved to {os.path.join(args.output_dir, 'evaluation_results.txt')}")

if __name__ == "__main__":
    main()
