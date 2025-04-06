import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(otu_file, metadata_file):
    print("Loading OTU and metadata files...")

    # Load OTU table with no header, and first column as SampleID
    otu_df = pd.read_csv(otu_file, header=None)
    otu_df.rename(columns={0: 'SampleID'}, inplace=True)
    otu_df.set_index('SampleID', inplace=True)

    # Assign default OTU column names
    otu_df.columns = [f'OTU_{i+1}' for i in range(otu_df.shape[1])]

    # Load metadata and use SampleID as index
    metadata_df = pd.read_csv(metadata_file)
    metadata_df = metadata_df[['SampleID', 'BMI']]
    metadata_df.set_index('SampleID', inplace=True)

    # Merge on SampleID
    merged_data = otu_df.join(metadata_df, how='inner')

    # Create binary label
    merged_data['Label'] = merged_data['BMI'].apply(lambda x: 1 if x >= 30 else 0)

    print(f"Merged data shape: {merged_data.shape}")
    return merged_data

def preprocess_data(df):
    df = df.dropna()
    return df

def split_data(df):
    df["Label"] = df["BMI"].apply(lambda x: 1 if x >= 30 else 0)
    X = df.drop(columns=["BMI", "sampleid", "Label"])
    y = df["Label"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

