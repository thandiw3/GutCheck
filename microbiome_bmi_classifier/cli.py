import sys
import os
import argparse
from .synthetic_data import generate_synthetic_data
from .feature_extraction import extract_features
from .classification import train_model, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="GutCheck: Microbiome Classification Tool")
    parser.add_argument('--generate-synthetic', action='store_true', help="Generate synthetic microbiome data")
    parser.add_argument('--train', action='store_true', help="Train the classification model")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the trained model")
    parser.add_argument('--output-dir', type=str, default='.', help="Directory to save the output files")
    
    args = parser.parse_args()

    if args.generate_synthetic:
        # Generate synthetic data with dynamic sample size
        print("Generating synthetic data...")
        data = generate_synthetic_data(min_samples=10, max_samples=100, num_features=50)  # Adjust as needed
        
        # Save data to the output directory
        output_file = os.path.join(args.output_dir, "synthetic_data.csv")
        data.to_csv(output_file, index=False)
        print(f"Synthetic data saved to {output_file}")

    if args.train:
        if not args.generate_synthetic:
            print("You must generate synthetic data first!")
            sys.exit(1)

        # Extract features from the synthetic data
        print("Extracting features...")
        features = extract_features(data)

        # Train the model
        print("Training the model...")
        X_train = features.drop(columns=["Label", "SampleID"])
        y_train = features["Label"]
        model = train_model(X_train, y_train)
        
        # Save model to output directory
        model_path = os.path.join(args.output_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
    
    if args.evaluate:
        if not args.train:
            print("You must train a model first!")
            sys.exit(1)

        # Load the trained model
        model_path = os.path.join(args.output_dir, "model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Evaluate the model
        print("Evaluating the model...")
        evaluate_model(model, X_train, y_train)  # Add proper evaluation logic here

if __name__ == "__main__":
    main()
