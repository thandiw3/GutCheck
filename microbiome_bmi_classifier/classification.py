import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def classify_samples(data, bmi_threshold=30):
    """
    Classify samples as Healthy (0) or Obese (1) based on BMI using a RandomForestClassifier.
    Args:
    - data (DataFrame): The data containing OTU features, BMI, and Label.
    - bmi_threshold (float): The BMI threshold to classify Obese vs Healthy. Default is 30.
    
    Returns:
    - model (RandomForestClassifier): Trained classification model.
    - X_test (DataFrame): Test features.
    - y_test (Series): Test labels.
    """
    
    # Create the target variable 'Label' based on BMI threshold
    data['Label'] = (data['BMI'] >= bmi_threshold).astype(int)  # Label 1 for Obese, 0 for Healthy
    
    # Select features (exclude 'BMI' and 'Label' columns)
    features = [col for col in data.columns if col.startswith('OTU') or col in ['mean', 'std', 'min', 'max']]
    X = data[features]  # Features for classification
    y = data['Label']   # Target variable (Label)
    
    # Split data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy and print classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return model, X_test, y_test

# Example usage
if __name__ == "__main__":
    # Load the extracted features
    data = pd.read_csv('extracted_features_synthetic_data.csv', index_col=0)
    
    # Call the classify_samples function
    model, X_test, y_test = classify_samples(data)

