import os
import sys
import argparse
from microbiome_bmi_classifier.data_preprocess import load_data, preprocess_data, split_data
from microbiome_bmi_classifier.feature_extraction import extract_features
from microbiome_bmi_classifier.model import train_model, evaluate_model, save_model
from microbiome_bmi_classifier.synthetic_data import generate_synthetic_data

def main():
    parser = argparse.ArgumentParser(description="GutCheck: Microbiome BMI Classifier")
    
    # Optional arguments for file paths
    parser.add_argument('--otu-file', type=str, help="OTU data file (CSV). If not provided, synthetic data will be generated.")
    parser.add_argument('--metadata-file', type=str, help="Metadata file (CSV). If not provided, synthetic data will be generated.")
    parser.add_argument('--output-dir', type=str, default='.', help="Directory to save output files.")
    
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
        otu_data, metadata = generate_synthetic_data()

    # Preprocess data
    print("Preprocessing data...")
    data = preprocess_data(otu_data, metadata)

    # Split the data into training and testing sets
    print("Splitting data...")
    train_data, test_data = split_data(data)

    # Extract features from the data
    print("Extracting features...")
    features = extract_features(data)

    # Train the model (default)
    print("Training the model...")
    model = train_model(features)

    # Save the trained model
    print(f"Saving model to {args.output_dir}...")
    model_file = os.path.join(args.output_dir, 'trained_model.pkl')
    save_model(model, model_file)

    print(f"Model training completed and saved to {model_file}")
    print("Process finished successfully!")

if __name__ == '__main__':
    main()
