from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

def train_model(X_train, y_train, model_type='logistic'):
    """
    Train the model using the specified model type.
    :param X_train: Training feature matrix.
    :param y_train: Training labels.
    :param model_type: Type of model to train ('logistic' or 'rf' for Random Forest).
    :return: Trained model.
    """
    print(f"Training {model_type} model...")
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data and return evaluation metrics.
    :param model: The trained model to evaluate.
    :param X_test: Test feature matrix.
    :param y_test: True labels for the test data.
    :return: Evaluation results (accuracy and classification report).
    """
    print("Evaluating the model...")

    # Predict on the test data
    y_pred = model.predict(X_test)
    
    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=["Healthy", "Obese"])
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm
    }

def cross_validate_model(X, y, model_type='logistic', cv_folds=5):
    """
    Perform cross-validation to evaluate the model's performance.
    :param X: Feature matrix.
    :param y: Labels.
    :param model_type: Model type ('logistic' or 'rf').
    :param cv_folds: Number of folds for cross-validation.
    :return: Cross-validation results (mean accuracy).
    """
    print(f"Performing {cv_folds}-fold cross-validation...")

    model = train_model(X, y, model_type)
    
    # Use StratifiedKFold to ensure the data is split in a way that preserves the label distribution
    skf = StratifiedKFold(n_splits=cv_folds)
    accuracies = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train the model on the training split
        model.fit(X_train, y_train)
        
        # Predict on the test split
        y_pred = model.predict(X_test)
        
        # Calculate the accuracy for this fold
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    print(f"Mean accuracy from cross-validation: {mean_accuracy:.4f}")
    return mean_accuracy

