import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(otu_file: str, metadata_file: str):
    """Load OTU table and metadata, merging them on sample ID."""
    otu_df = pd.read_csv(otu_file, index_col=0)
    metadata_df = pd.read_csv(metadata_file, index_col=0)
    
    # Merge OTU table and metadata on Sample_ID
    merged_df = otu_df.join(metadata_df, how='inner')
    
    return merged_df

def preprocess_data(data: pd.DataFrame, normalize=False):
    """Preprocess the data by handling missing values and normalizing the OTU counts."""
    # Handle missing data (e.g., fill missing values with zeros)
    data = data.fillna(0)
    
    if normalize:
        # Normalize OTU counts using Min-Max scaling (for example)
        otu_columns = [col for col in data.columns if col.startswith("OTU")]
        data[otu_columns] = data[otu_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    
    return data

def split_data(data: pd.DataFrame, test_size=0.2, random_state=42):
    """Split the data into train and test sets."""
    # Separate features (OTU data) and target (BMI label)
    X = data.drop(columns=['BMI', 'Label'])
    y = data['Label']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

# Define paths to the 3 files
otu_file = 'otu_table.csv'  # Path to the OTU table file
metadata_file = 'metadata.csv'  # Path to the metadata file

# Step 1: Load the data
merged_data = load_data(otu_file, metadata_file)

# Step 2: Preprocess the data (normalize OTU counts)
processed_data = preprocess_data(merged_data, normalize=True)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(processed_data)

# Display the first few rows of the processed data
print(processed_data.head())

# Print the shapes of the split datasets
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
