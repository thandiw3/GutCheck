import pandas as pd

def extract_features(data):
    otu_columns = [col for col in data.columns if col.startswith('OTU_')]
    otu_data = data[otu_columns]

    features = pd.DataFrame({
        'mean': otu_data.mean(axis=0),
        'std': otu_data.std(axis=0),
        'min': otu_data.min(axis=0),
        'max': otu_data.max(axis=0)
    })

    features['BMI'] = data['BMI'].mean()
    features['Label'] = data['Label'].mean()
    features['BMI_category'] = 'Healthy' if features['BMI'].iloc[0] < 30 else 'Obese'

    features.to_csv('extracted_features_synthetic_data.csv')
    return features
