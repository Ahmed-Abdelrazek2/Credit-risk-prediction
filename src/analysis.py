# analysis.py
# Data Analysis & Visualization for Credit Risk / Student Data
# Uses DataFrames already loaded (from LoadData.py)

import sys
import os

import matplotlib

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.LoadData import cs_train, cs_test, sample_entry, data_dict
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# ── Data Overview ─────────────────────────────────────────
def data_overview(df):
    """Print first rows, info, description, and missing values"""
    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Data Info ---")
    print(df.info())

    print("\n--- Statistical Summary ---")
    print(df.describe(include='all'))

    print("\n--- Missing Values ---")
    print(df.isnull().sum())


# ── Target Distribution ───────────────────────────────────
def plot_target_distribution(df, target):
    """Plot distribution of the target variable"""
    plt.figure(figsize=(6,4))
    sns.countplot(x=target, data=df)
    plt.title(f"Distribution of {target}")
    plt.show()


# ── Correlation Heatmap ───────────────────────────────────
def plot_correlation_heatmap(df, numeric_cols):
    """Plot correlation heatmap for numeric features"""
    plt.figure(figsize=(12,8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap of Numeric Features")
    plt.show()


# ── Full EDA ─────────────────────────────────────────────
def run_eda(df, target, numeric_cols):
    """Run full analysis and visualization"""
    data_overview(df)
    if target in df.columns:
        plot_target_distribution(df, target)
    plot_correlation_heatmap(df, numeric_cols)


# ── Main Function ─────────────────────────────────────────
def main():
    """
    Main function to run analysis.
    Assumes cs_train is already loaded from LoadData.py
    """
    print("Starting analysis in analysis.py...")

    # Define target & numeric columns
    target = "SeriousDlqin2yrs"
    numeric_cols = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents"
    ]

    print("Running full EDA on training data...")
    # Run EDA
    run_eda(cs_train, target, numeric_cols)
    print("EDA completed successfully.")


# ── Run as script ─────────────────────────────────────────
if __name__ == "__main__":
    main()