import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_features(data):
    # Assuming data has 'SampleID' as index and OTUs as columns
    print(f"Data columns: {data.columns}")

    # Ensure the OTU columns are numeric
    otu_columns = [col for col in data.columns if col.startswith('OTU')]
    
    # Convert all OTU columns to numeric, forcing errors to NaN (so we can handle them)
    data[otu_columns] = data[otu_columns].apply(pd.to_numeric, errors='coerce')
    
    # Handle NaN values by filling them with 0 (or use another strategy like interpolation)
    data[otu_columns] = data[otu_columns].fillna(0)

    # Calculate diversity indices (e.g., Shannon, Simpson, Richness)
    shannon_index = data[otu_columns].apply(lambda x: -np.sum(x * np.log(x + 1e-6)), axis=1)
    simpson_index = data[otu_columns].apply(lambda x: 1 - np.sum(x**2), axis=1)
    richness = data[otu_columns].apply(lambda x: np.count_nonzero(x > 0), axis=1)

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

    # Append the OTU abundances to the feature set
    features_final = pd.concat([features_scaled_df, data], axis=1)

    return features_final
