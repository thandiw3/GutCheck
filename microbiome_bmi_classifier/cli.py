import argparse
import os
from feature_extraction import extract_features
from data_processing import load_data, preprocess_data, split_data
from synthetic_data import create_synthetic_data
from classification import train_model, evaluate_model, cross_validate_model

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Microbiome-based BMI classification pipeline")

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

    # Check if synthetic data needs to be generated
    if args.generate_synthetic:
        print("Generating synthetic data...")
        data = create_synthetic_data()  # Generates synthetic data (OTU and metadata combined)
        output_file = os.path.join(args.output_dir, 'synthetic_data.csv')
        data.to_csv(output_file)  # Save the synthetic data to CSV

        print(f"Synthetic data saved to {output_file}")

        # Now extract features from the saved file
        extract_features(output_file, os.path.join(args.output_dir, 'features.csv'))

    # Otherwise, use the provided OTU and metadata files
    elif args.otu_file and args.metadata_file:
        print("Loading and processing data...")
        data = load_data(args.otu_file, args.metadata_file)
        data = preprocess_data(data)
        data = split_data(data)

        output_file = os.path.join(args.output_dir, 'features.csv')
        extract_features(data, output_file)

    else:
        print("Error: Please provide either input files or use --generate-synthetic.")
        return

    # Optionally train or evaluate the model
    if args.train:
        print("Training the model...")
        train_model(data, args.output_dir)

    if args.evaluate and args.model_file:
        print("Evaluating the model...")
        evaluate_model(data, args.model_file)

    if not args.train and not args.evaluate:
        print("No training or evaluation specified. Exiting.")

if __name__ == "__main__":
    main()
