import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(otu_file=None, metadata_file=None):
    if otu_file is None or metadata_file is None:
        print("No data provided. Creating synthetic data...")
        return create_synthetic_data()

    otu_df = pd.read_csv(otu_file, index_col=0)
    metadata_df = pd.read_csv(metadata_file)

    merged_df = otu_df.merge(metadata_df, left_index=True, right_on='sampleid')
    return merged_df

def create_synthetic_data():
    np.random.seed(42)
    synthetic_otus = pd.DataFrame(
        np.random.rand(100, 20), 
        columns=[f"OTU_{i}" for i in range(20)]
    )
    synthetic_otus.index = [f"Sample_{i}" for i in range(100)]
    metadata = pd.DataFrame({
        "sampleid": synthetic_otus.index,
        "BMI": np.random.normal(loc=25, scale=5, size=100)
    })
    merged = synthetic_otus.merge(metadata, left_index=True, right_on='sampleid')
    return merged

def preprocess_data(df):
    df = df.dropna()
    return df

def split_data(df):
    df["Label"] = df["BMI"].apply(lambda x: 1 if x >= 30 else 0)
    X = df.drop(columns=["BMI", "sampleid", "Label"])
    y = df["Label"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

