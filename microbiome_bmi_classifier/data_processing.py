from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Drop any unwanted columns (e.g., BMI and Label are features, not OTUs)
    data = data.drop(columns=["BMI", "Label"])
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data

def split_data(features, labels):
    """
    Splits the dataset into training and testing sets.
    
    Parameters:
    features (pd.DataFrame): The feature matrix.
    labels (pd.Series): The target labels.
    
    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

