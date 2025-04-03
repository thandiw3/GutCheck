import numpy as np
import pandas as pd

def generate_otu_table(n_samples=100, n_otus=50, seed=42):
    """Generate a synthetic OTU table with random counts."""
    np.random.seed(seed)
    
    # Simulate OTU counts (random values between 0 and 10,000)
    otu_data = np.random.randint(0, 10000, size=(n_samples, n_otus))
    
    # Sample IDs
    sample_ids = [f"Sample_{i}" for i in range(n_samples)]
    
    # OTU columns (OTU_0, OTU_1, ..., OTU_n)
    otu_columns = [f"OTU_{j}" for j in range(n_otus)]
    
    # Create DataFrame
    otu_df = pd.DataFrame(otu_data, index=sample_ids, columns=otu_columns)
    
    return otu_df

def generate_metadata(n_samples=100, seed=42):
    """Generate synthetic metadata with BMI values and classifications (Healthy/Obese)."""
    np.random.seed(seed)
    
    # Generate BMI values (Normal distribution: mean=27, std=5)
    bmi_values = np.random.normal(loc=27, scale=5, size=n_samples)
    
    # Classify into Healthy (0) or Obese (1) based on BMI threshold (BMI >= 30 is Obese)
    labels = (bmi_values >= 30).astype(int)
    
    # Create DataFrame with Sample_ID, BMI, and Label
    metadata_df = pd.DataFrame({
        "Sample_ID": [f"Sample_{i}" for i in range(n_samples)],
        "BMI": bmi_values,
        "Label": labels
    }).set_index("Sample_ID")
    
    return metadata_df

def generate_synthetic_data(n_samples=100, n_otus=50, seed=42):
    """Generate a full synthetic dataset with OTU counts and BMI metadata."""
    otu_df = generate_otu_table(n_samples=n_samples, n_otus=n_otus, seed=seed)
    metadata_df = generate_metadata(n_samples=n_samples, seed=seed)
    
    # Merge OTU table and metadata on Sample_ID
    full_data = otu_df.join(metadata_df, how="inner")
    
    return otu_df, metadata_df, full_data

def save_synthetic_data(n_samples=100, n_otus=50, seed=42):
    """Generate synthetic data and save as CSV files."""
    otu_df, metadata_df, full_data = generate_synthetic_data(n_samples=n_samples, n_otus=n_otus, seed=seed)
    
    # Save the OTU table and metadata to CSV
    otu_df.to_csv("otu_table.csv")
    metadata_df.to_csv("metadata.csv")
    full_data.to_csv("synthetic_data.csv")
    
    print("Synthetic data saved as 'otu_table.csv', 'metadata.csv', and 'synthetic_data.csv'")

# Run the function to save the data
if __name__ == "__main__":
    save_synthetic_data(n_samples=100, n_otus=50, seed=42)

