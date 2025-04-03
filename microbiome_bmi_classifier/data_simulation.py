import numpy as np
import pandas as pd

def generate_otu_table(n_samples=1000, n_otus=50, seed=42):
    """
    Generate a synthetic OTU table with random values for each sample and OTU.
    """
    np.random.seed(seed)
    otu_data = np.random.randint(0, 10000, size=(n_samples, n_otus))
    sample_ids = [f"Sample_{i}" for i in range(n_samples)]
    otu_columns = [f"OTU_{j}" for j in range(n_otus)]
    otu_df = pd.DataFrame(otu_data, index=sample_ids, columns=otu_columns)
    return otu_df

def generate_metadata(n_samples=100, seed=42):
    """
    Generate synthetic metadata, including BMI and disease label (Obese/Healthy).
    """
    np.random.seed(seed)
    bmi_values = np.random.normal(loc=27, scale=5, size=n_samples)  # Generate random BMI values
    labels = (bmi_values >= 30).astype(int)  # Obese if BMI >= 30, else Healthy
    metadata_df = pd.DataFrame({
        "Sample_ID": [f"Sample_{i}" for i in range(n_samples)],
        "BMI": bmi_values,  # Add BMI values to the metadata
        "Label": labels      # Add disease label based on BMI
    }).set_index("Sample_ID")
    return metadata_df

def generate_synthetic_data():
    """
    Generate synthetic OTU data and metadata, then combine them into one DataFrame.
    """
    otu_df = generate_otu_table()  # Generate OTU data
    metadata_df = generate_metadata()  # Generate metadata (BMI, Label)
    
    # Merge OTU data and metadata on the Sample_ID
    full_data = otu_df.join(metadata_df, how="inner")
    
    # Save the full dataset to CSV
    full_data.to_csv('synthetic_data.csv')
    return full_data

# Generate the synthetic data and save it to CSV
generate_synthetic_data()

