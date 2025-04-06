import pandas as pd
from sklearn.model_selection import train_test_split
from microbiome_bmi_classifier.synthetic_data import create_synthetic_data

def load_data(otu_file=None, metadata_file=None):
    if otu_file is None or metadata_file is None:
        print("No data provided. Creating synthetic data...")
        return create_synthetic_data()

    print("Loading real data...")

    # Load OTU table with no header row and set SampleID as index
    otu_df = pd.read_csv(otu_file, header=None)
    sample_ids = otu_df.iloc[:, 0]
    otu_data = otu_df.iloc[:, 1:]
    otu_data.columns = [f'OTU_{i+1}' for i in range(otu_data.shape[1])]
    otu_data.index = sample_ids
    otu_data.index.name = "SampleID"

    # Load metadata
    metadata = pd.read_csv(metadata_file)
    metadata.set_index("SampleID", inplace=True)

    # Merge data on SampleID
    merged_data = otu_data.join(metadata, how="inner")

    # Add Label: 1 if BMI >= 30 (Obese), else 0 (Healthy)
    merged_data["Label"] = merged_data["BMI"].apply(lambda x: 1 if x >= 30 else 0)

    return merged_data

def preprocess_data(data):
    print("Preprocessing data...")
    return data.dropna()

def split_data(data):
    X = data.drop(columns=["Label"])
    y = data["Label"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


