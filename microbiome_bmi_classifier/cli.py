import pandas as pd
import numpy as np
from feature_extraction import extract_features
from classification import train_model, evaluate_model
from synthetic_data import create_synthetic_data

def main():
    # Check if synthetic data should be generated
    generate_synthetic = True  # You can replace this with argument parsing

    if generate_synthetic:
        print("Generating synthetic data...")
        otu_data, metadata = create_synthetic_data()  # Returns OTU data and metadata
        
        # Combine OTU data and metadata into a single DataFrame
        data = pd.concat([otu_data, metadata], axis=1)

        # Save the combined synthetic data to a CSV file
        output_file = 'synthetic_data.csv'  # Define your output file name
        data.to_csv(output_file)
        print(f"Synthetic data saved to {output_file}")

    # After generating and saving synthetic data, proceed with other tasks
    extract_features('synthetic_data.csv', 'extracted_features.csv')  # Extract features from synthetic data
    train_model('extracted_features.csv')  # Train your model (implement the logic)
    evaluate_model('extracted_features.csv')  # Evaluate your model (implement the logic)

if __name__ == "__main__":
    main()
