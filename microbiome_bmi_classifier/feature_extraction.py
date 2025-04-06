import pandas as pd

def extract_features(data):
    """
    Extract statistical features (mean, std, min, max) for each OTU column
    and include BMI and Label as additional features.
    """
    # Ensure that the data only contains numeric values for OTU columns
    otu_columns = [col for col in data.columns if col.startswith('OTU')]
    otu_data = data[otu_columns]

    # Convert the OTU data to numeric, coercing any errors (like strings) into NaN
    otu_data = otu_data.apply(pd.to_numeric, errors='coerce')

    # Calculate summary statistics for each OTU column
    features = pd.DataFrame({
        'mean': otu_data.mean(axis=0),
        'std': otu_data.std(axis=0),
        'min': otu_data.min(axis=0),
        'max': otu_data.max(axis=0)
    })

    # Add BMI and Label as additional columns to the features DataFrame
    features['BMI'] = data['BMI'].mean()  # Mean of BMI across samples
    features['Label'] = data['Label'].mean()  # Mean of Label across samples

    # Add BMI category based on the average BMI
    features['BMI_category'] = 'Healthy' if features['BMI'].iloc[0] < 30 else 'Obese'

    # Save the extracted features to a CSV file
    features.to_csv('extracted_features_synthetic_data.csv')
    return features
