import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def model():
    # Load the feature-extracted data
    data = pd.read_csv('extracted_features_synthetic_data.csv')

    # Ensure Label is in the correct format: binary (0, 1)
    print("Data columns: ", data.columns)
    print("Unique values in Label column: ", data['Label'].unique())
    
    # Check the data type of 'Label'
    print("Data type of 'Label' column: ", data['Label'].dtype)

    # Features and target
    X = data.drop(columns=['Label', 'BMI', 'BMI_category'])
    y = data['Label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
