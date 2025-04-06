import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def classify_features(input_file, output_model_file):
    # Load the extracted features and labels
    data = pd.read_csv(input_file, index_col=0)

    # Assuming 'BMI_Category' is the label column and other columns are features
    X = data.drop(columns=['BMI_Category'])
    y = data['BMI_Category']

    # Encode labels (if they're categorical)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict the labels on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Save the trained model
    import joblib
    joblib.dump(clf, output_model_file)
    print(f"Model saved to {output_model_file}")

if __name__ == "__main__":
    # Specify the input file (feature-extracted data) and output model file
    input_file = 'extracted_features.csv'
    output_model_file = 'trained_model.joblib'
    
    classify_features(input_file, output_model_file)
