import pandas as pd
import numpy as np

def load_data(input_file):
    """
    Load data from the given file.
    """
    data = pd.read_csv(input_file, index_col=0)
    return data

def preprocess_data(data):
    """
    Preprocess the data (e.g., remove any unnecessary columns, handle missing values, etc.).
    """
    # Example preprocessing: drop columns or rows with missing data
    data = data.dropna()
    return data

