import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_features(input_file, output_file):
    # Load the raw microbiome data (abundance counts)
    data = pd.read_csv(input_file, index_col=0)

    # Check the columns and basic info about the data
    print(f"Data columns: {data.columns}")

    # Calculate diversity indices (e.g., Shannon, Simpson, Richness)
    # Shannon Index (H) = -sum(p * log(p)) for each OTU in a sample
    # Simpson Index (D) = 1 - sum(p^2) for each OTU in a sample
    shannon_index = data.apply(lambda x: -np.sum(x * np.log(x + 1e-6)), axis=1)
    simpson_index = data.apply(lambda x: 1 - np.sum(x**2), axis=1)
    richness = data.apply(lambda x: np.count_nonzero(x > 0), axis=1)

    # Create a new DataFrame to store features
    features = pd.DataFrame({
        'Shannon_Index': shannon_index,
        'Simpson_Index': simpson_index,
        'Richness': richness
    }, index=data.index)

    # Standardize the features (important for models like SVM or logistic regression)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)

    # Save the extracted features to a CSV file
    features_scaled_df.to_csv(output_file)
    print(f"Feature extraction complete. Features saved to {output_file}")


