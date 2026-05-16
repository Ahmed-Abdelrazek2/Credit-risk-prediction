# src/LoadData.py
# This script loads CSV and Excel data and makes them available for import

import pandas as pd

# ── Paths ─────────────────────────────────────────────
train_path = r"C:\Users\ARWA\credit_risk_predictor\data\cs-training.csv"
test_path  = r"C:\Users\ARWA\credit_risk_predictor\data\cs-test.csv"
sample_path = r"C:\Users\ARWA\credit_risk_predictor\data\sampleEntry.csv"
dict_path = r"C:\Users\ARWA\credit_risk_predictor\data\Data Dictionary.xls"


# ── Load CSV ─────────────────────────────────────────
def load_csv(path):
    """
    Load a CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(path)
    print(f"Loaded CSV: {path} | Shape: {df.shape}")
    print(df.head())
    return df


# ── Load Excel ───────────────────────────────────────
def load_excel(path, sheet_name=0, engine='xlrd'):
    """
    Load an Excel file into a pandas DataFrame.
    """
    df = pd.read_excel(path, sheet_name=sheet_name, engine=engine)
    print(f"Loaded Excel: {path} | Sheet: {sheet_name} | Shape: {df.shape}")
    print(df.head())
    return df


# ── Load data as global variables ─────────────────────
cs_train = load_csv(train_path)
cs_test  = load_csv(test_path)
sample_entry = load_csv(sample_path)
data_dict = load_excel(dict_path, sheet_name=0)

if __name__ == "__main__":
    print("Running LoadData.py directly...")
    print("cs_train shape:", cs_train.shape)
    print("cs_test shape:", cs_test.shape)
    print("sample_entry shape:", sample_entry.shape)
    print("data_dict shape:", data_dict.shape)
    print("All data loaded successfully.")