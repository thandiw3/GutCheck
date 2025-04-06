import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(input_file, output_file, test_size=0.2):
    """
    Train a model using the input features, with an option to split data into train and test sets.
    """
    # Load the feature data
    data = pd.read_csv(input_file, index_col=0)

    # Split into features and labels
    X = data.drop(columns=["BMI", "Label"])
    y = data["Label"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize and train a random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the trained model
    joblib.dump(model, output_file)
    print(f"Model training complete and saved as {output_file}")

def evaluate_model(input_file, model_file):
    """
    Evaluate the trained model using the test data.
    """
    # Load the test data
    data = pd.read_csv(input_file, index_col=0)

    # Split into features and labels
    X = data.drop(columns=["BMI", "Label"])
    y = data["Label"]

    # Load the trained model
    model = joblib.load(model_file)

    # Make predictions and evaluate the model
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model evaluation accuracy: {accuracy * 100:.2f}%")

def cross_validate_model(input_file, cv=5):
    """
    Perform cross-validation using the dataset and a RandomForestClassifier.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier

    # Load the data
    data = pd.read_csv(input_file, index_col=0)

    # Split into features and labels
    X = data.drop(columns=["BMI", "Label"])
    y = data["Label"]

    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean() * 100:.2f}%")
