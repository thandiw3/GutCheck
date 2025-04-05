import argparse
from microbiome_bmi_classifier.data_processing import load_data
from microbiome_bmi_classifier.model import classify_bmi
from microbiome_bmi_classifier.utils import generate_synthetic_data

def main():
    print("🧪 Welcome to GutCheck!")

    parser = argparse.ArgumentParser(description="Classify obesity from microbiome data.")
    parser.add_argument('--otu', type=str, help="Path to OTU table CSV file")
    parser.add_argument('--meta', type=str, help="Path to metadata CSV file")
    args = parser.parse_args()

    if args.otu and args.meta:
        print("📂 Loading user-provided data...")
        X, y = load_data(args.otu, args.meta)
    else:
        print("⚠️ No data provided. Generating synthetic data instead...")
        X, y = generate_synthetic_data()

    acc, _ = classify_bmi(X, y)
    print(f"✅ BMI classification complete! Accuracy: {acc:.2f}")

if __name__ == "__main__":
    main()

