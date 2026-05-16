import pandas as pd
import numpy as np
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.LoadData import cs_train, cs_test

def clean_dataset(df, name="Dataset"):
    print(f"\n{'='*20} Cleaning {name} {'='*20}")
    
    # -------------------------
    # Basic Information
    # -------------------------
    print(f"Original Shape: {df.shape}")
    
    # -------------------------
    # Remove Duplicates
    # -------------------------
    duplicates = df.duplicated().sum()
    print(f"Number of duplicated rows: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Shape after dropping duplicates: {df.shape}")

    # -------------------------
    # Missing Values Handling
    # -------------------------
    missing_before = df.isnull().sum()
    if missing_before.sum() > 0:
        print("\nMissing values before cleaning:")
        print(missing_before[missing_before > 0])
    else:
        print("\nNo missing values found.")
    
    # Drop columns with >70% missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    columns_to_drop = missing_percentage[missing_percentage > 70].index
    if len(columns_to_drop) > 0:
        print(f"\nColumns dropped due to >70% missing values: {list(columns_to_drop)}")
        df = df.drop(columns=columns_to_drop)
    
    # Fill numerical missing values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled missing values in '{col}' with median: {median_val}")

    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=['object', 'string', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Filled missing values in '{col}' with mode: {mode_val}")
            
    print(f"\nMissing values after cleaning: {df.isnull().sum().sum()}")
    
    return df

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Clean Training Data
    cleaned_train = clean_dataset(cs_train.copy(), name="Training Data")
    train_save_path = "data/cleaned_cs_training.csv"
    cleaned_train.to_csv(train_save_path, index=False)
    print(f"\nCleaned training dataset saved to {train_save_path}")

    # Clean Test Data
    cleaned_test = clean_dataset(cs_test.copy(), name="Test Data")
    test_save_path = "data/cleaned_cs_test.csv"
    cleaned_test.to_csv(test_save_path, index=False)
    print(f"Cleaned test dataset saved to {test_save_path}")
