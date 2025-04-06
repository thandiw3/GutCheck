from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

def train_model(X_train, y_train):
    """
    Train a Random Forest classifier on the training data.
    
    Parameters:
    X_train (pd.DataFrame): The training feature data.
    y_train (pd.Series): The training labels.
    
    Returns:
    model: The trained classifier.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.
    
    Parameters:
    model: The trained classifier.
    X_test (pd.DataFrame): The test feature data.
    y_test (pd.Series): The test labels.
    
    Returns:
    None
    """
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

def cross_validate_model(model, X, y):
    """
    Perform cross-validation on the model.
    
    Parameters:
    model: The classifier to be cross-validated.
    X (pd.DataFrame): The feature data.
    y (pd.Series): The labels.
    
    Returns:
    None
    """
    scores = cross_val_score(model, X, y, cv=5)
    print("Cross-validation scores:", scores)
    print("Average cross-validation score:", scores.mean())
