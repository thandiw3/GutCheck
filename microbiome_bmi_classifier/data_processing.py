from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Drop any unwanted columns (e.g., BMI and Label are features, not OTUs)
    data = data.drop(columns=["BMI", "Label"])
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data

def split_data(data):
    # Split the data into train and test sets
    X = data.drop(columns=["BMI", "Label"])  # Features (OTUs)
    y = data[["BMI", "Label"]]  # Target (BMI and Label)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

