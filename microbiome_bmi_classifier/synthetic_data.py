import pandas as pd
import numpy as np

def create_synthetic_data(num_samples=100, num_otus=50):
    np.random.seed(42)
    sample_ids = [f"Sample_{i+1}" for i in range(num_samples)]
    otu_data = np.random.rand(num_samples, num_otus)

    otu_df = pd.DataFrame(otu_data, columns=[f"OTU_{i+1}" for i in range(num_otus)])
    otu_df["SampleID"] = sample_ids
    otu_df.set_index("SampleID", inplace=True)

    bmi_values = np.random.normal(loc=25, scale=5, size=num_samples)
    labels = (bmi_values >= 30).astype(int)

    metadata = pd.DataFrame({
        "BMI": bmi_values,
        "Label": labels
    }, index=otu_df.index)

    # Combine OTU data and metadata into one dataframe
    combined_data = pd.concat([otu_df, metadata], axis=1)

    # Return the separate otu_data, metadata, and combined data
    return otu_df, metadata, combined_data
