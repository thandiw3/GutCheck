import pandas as pd
import numpy as np

def generate_synthetic_data(min_samples=10, max_samples=1000, num_features=100):
    """
    Generate synthetic microbiome data for classification.
    The number of samples generated will be random between min_samples and max_samples.
    Each time this function is called, new random data is generated.
    """
    # Randomly choose a number of samples between min_samples and max_samples
    num_samples = np.random.randint(min_samples, max_samples + 1)
    
    print(f"Generating {num_samples} synthetic samples with {num_features} OTU features...")

    # Generate random OTU data: num_samples x num_features matrix
    otus = np.random.rand(num_samples, num_features)
    
    # Generate random BMI values between 18 and 35 (for example)
    bmi_values = np.random.uniform(18, 35, size=num_samples)
    
    # Generate random Labels (0 or 1 for healthy or obese)
    labels = np.random.choice([0, 1], size=num_samples)
    
    # Create the DataFrame
    data = pd.DataFrame(otus, columns=[f"OTU_{i+1}" for i in range(num_features)])
    data['BMI'] = bmi_values
    data['Label'] = labels
    data['SampleID'] = [f"S{i+1}" for i in range(num_samples)]  # Sample IDs
    
    # Return the DataFrame
    return data





