import argparse
from microbiome_bmi_classifier.data_processing import load_data, preprocess_data
from microbiome_bmi_classifier.feature_extraction import extract_features
from microbiome_bmi_classifier.classification import model

def main():
    parser = argparse.ArgumentParser(description="GutCheck: Microbiome BMI Classifier")
    parser.add_argument("--otu_file", type=str, help="Path to OTU table CSV file")
    parser.add_argument("--metadata_file", type=str, help="Path to metadata CSV file")

    args = parser.parse_args()

    # Load & preprocess
    data = load_data(args.otu_file, args.metadata_file)
    data = preprocess_data(data)

    # Feature extraction
    features = extract_features(data)

    # Classification
    model()

if __name__ == "__main__":
    main()
