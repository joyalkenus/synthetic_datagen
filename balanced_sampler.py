import pandas as pd
import numpy as np

def create_balanced_dataset(input_file, output_file, samples_per_class=2500):
    """
    Create a balanced dataset by sampling equal numbers of rows for each preference.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        samples_per_class (int): Number of samples to take for each preference value
    """
    # Read the dataset
    df = pd.read_csv(input_file)
    
    # Split dataset by preference
    pref_0 = df[df['Preference'] == 0]
    pref_1 = df[df['Preference'] == 1]
    
    # Sample equal numbers from each preference
    # If there aren't enough samples, take all available samples
    sampled_0 = pref_0.sample(n=min(samples_per_class, len(pref_0)), random_state=42)
    sampled_1 = pref_1.sample(n=min(samples_per_class, len(pref_1)), random_state=42)
    
    # Combine the sampled datasets
    balanced_df = pd.concat([sampled_0, sampled_1])
    
    # Shuffle the combined dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create a new index column starting from 0
    balanced_df['Index'] = range(len(balanced_df))
    
    # Reorder columns to ensure Index is first
    cols = ['Index'] + [col for col in balanced_df.columns if col != 'Index']
    balanced_df = balanced_df[cols]
    
    # Save the balanced dataset
    balanced_df.to_csv(output_file, index=False)
    
    # Print statistics
    print(f"Original dataset size: {len(df)}")
    print(f"Original preference distribution:")
    print(df['Preference'].value_counts())
    print(f"\nBalanced dataset size: {len(balanced_df)}")
    print(f"Balanced preference distribution:")
    print(balanced_df['Preference'].value_counts())

if __name__ == "__main__":
    create_balanced_dataset('real_train_data_7000.csv', 'balanced_dataset.csv') 