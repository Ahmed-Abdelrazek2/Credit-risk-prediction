# =========================
# 📦 Import Libraries
# =========================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# =========================
# 📂 Load Data
# =========================
print("📂 Loading data...")
df = pd.read_csv("cs-training.csv")
print(f"Shape: {df.shape}")

# =========================
# 🧹 Cleaning
# =========================
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Remove invalid ages
df = df[df['age'] > 0]

# Cap outliers في الـ late payments (قيم > 90 غالباً errors)
late_cols = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate'
]
for col in late_cols:
    df[col] = df[col].clip(upper=90)

# Fix extreme DebtRatio (outliers واضحة)
df['DebtRatio'] = df['DebtRatio'].clip(upper=df['DebtRatio'].quantile(0.99))

# Fix extreme RevolvingUtilization
df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=1.5)

# Fix MonthlyIncome outliers
df['MonthlyIncome'] = df['MonthlyIncome'].clip(upper=df['MonthlyIncome'].quantile(0.99))

print("✅ Cleaning done.")

# =========================
# 🔧 Feature Engineering
# =========================

# 1. Total late payments (weighted by severity)
df['TotalLatePayments'] = (
    df['NumberOfTime30-59DaysPastDueNotWorse'] * 1 +
    df['NumberOfTime60-89DaysPastDueNotWorse'] * 2 +
    df['NumberOfTimes90DaysLate'] * 3
)

# 2. Has any late payment (binary flag)
df['HasLatePayment'] = (df['TotalLatePayments'] > 0).astype(int)

# 3. Debt to income
df['DebtToIncome'] = df['DebtRatio'] / (df['MonthlyIncome'].replace(0, np.nan) + 1)

# 4. Credit stress
df['CreditStress'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['DebtRatio']

# 5. Income per person (family-adjusted)
df['IncomePerPerson'] = df['MonthlyIncome'] / (df['NumberOfDependents'].fillna(0) + 1)

# 6. Age group (risk by life stage)
df['AgeGroup'] = pd.cut(
    df['age'],
    bins=[0, 30, 40, 50, 60, 100],
    labels=[0, 1, 2, 3, 4]
).astype(float)

# 7. High utilization flag (>= 0.8 = risky)
df['HighUtilization'] = (df['RevolvingUtilizationOfUnsecuredLines'] >= 0.8).astype(int)

# 8. Loan density (loans per age decade)
df['LoanDensity'] = df['NumberOfOpenCreditLinesAndLoans'] / (df['age'] / 10)

# 9. Real estate burden
df['RealEstateBurden'] = df['NumberRealEstateLoansOrLines'] * df['DebtRatio']

print("✅ Feature engineering done.")
print(f"Total features: {df.shape[1] - 1}")

# =========================
# 🎯 Split Features / Target
# =========================
X = df.drop(columns=['SeriousDlqin2yrs'])
y = df['SeriousDlqin2yrs']

print(f"\n📊 Class distribution:")
print(y.value_counts())
print(f"Imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.1f}:1")

# =========================
# ✂️ Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# ⚙️ Models (3 approaches)
# =========================

models = {
    "LogReg Balanced": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight='balanced',   # ← حل Class Imbalance
            C=0.1,
            random_state=42
        ))
    ]),

    "RandomForest Balanced": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',   # ← حل Class Imbalance
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "GradientBoosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        ))
    ]),
}

# =========================
# 🚀 Train & Evaluate All
# =========================
results = {}

print("\n" + "="*60)
for name, pipeline in models.items():
    print(f"\n🔄 Training: {name}")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    results[name] = {"auc": auc, "pipeline": pipeline}

    print(f"ROC-AUC Score: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

print("="*60)

# =========================
# 🏆 Best Model Summary
# =========================
best_name = max(results, key=lambda x: results[x]['auc'])
print(f"\n🏆 Best Model: {best_name}")
print(f"   ROC-AUC: {results[best_name]['auc']:.4f}")

# =========================
# 📊 Feature Importance (RandomForest)
# =========================
rf_pipeline = results["RandomForest Balanced"]["pipeline"]
rf_model = rf_pipeline.named_steps["model"]
feature_names = X.columns.tolist()

importances = pd.Series(rf_model.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=True).tail(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Feature Importance
axes[0].barh(importances.index, importances.values, color='steelblue')
axes[0].set_title('Top 15 Feature Importances (RandomForest)', fontsize=13)
axes[0].set_xlabel('Importance')

# Confusion Matrix for best model
best_pipeline = results[best_name]["pipeline"]
y_pred_best = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Default', 'Default'])
disp.plot(ax=axes[1], colorbar=False)
axes[1].set_title(f'Confusion Matrix - {best_name}', fontsize=13)

plt.tight_layout()
plt.savefig("model_results.png", dpi=150, bbox_inches='tight')
print("\n📊 Chart saved: model_results.png")

# =========================
# 💾 ROC-AUC Comparison
# =========================
print("\n📈 Final ROC-AUC Comparison:")
print("-" * 35)
for name, res in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
    print(f"  {name:<25} → {res['auc']:.4f}")
print("-" * 35)
print("\n✅ Done!")# =========================
# 📦 Import Libraries
# =========================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# =========================
# 📂 Load Data
# =========================
print("📂 Loading data...")
df = pd.read_csv("cs-training.csv")
print(f"Shape: {df.shape}")

# =========================
# 🧹 Cleaning
# =========================
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Remove invalid ages
df = df[df['age'] > 0]

# Cap outliers في الـ late payments (قيم > 90 غالباً errors)
late_cols = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate'
]
for col in late_cols:
    df[col] = df[col].clip(upper=90)

# Fix extreme DebtRatio (outliers واضحة)
df['DebtRatio'] = df['DebtRatio'].clip(upper=df['DebtRatio'].quantile(0.99))

# Fix extreme RevolvingUtilization
df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=1.5)

# Fix MonthlyIncome outliers
df['MonthlyIncome'] = df['MonthlyIncome'].clip(upper=df['MonthlyIncome'].quantile(0.99))

print("✅ Cleaning done.")

# =========================
# 🔧 Feature Engineering
# =========================

# 1. Total late payments (weighted by severity)
df['TotalLatePayments'] = (
    df['NumberOfTime30-59DaysPastDueNotWorse'] * 1 +
    df['NumberOfTime60-89DaysPastDueNotWorse'] * 2 +
    df['NumberOfTimes90DaysLate'] * 3
)

# 2. Has any late payment (binary flag)
df['HasLatePayment'] = (df['TotalLatePayments'] > 0).astype(int)

# 3. Debt to income
df['DebtToIncome'] = df['DebtRatio'] / (df['MonthlyIncome'].replace(0, np.nan) + 1)

# 4. Credit stress
df['CreditStress'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['DebtRatio']

# 5. Income per person (family-adjusted)
df['IncomePerPerson'] = df['MonthlyIncome'] / (df['NumberOfDependents'].fillna(0) + 1)

# 6. Age group (risk by life stage)
df['AgeGroup'] = pd.cut(
    df['age'],
    bins=[0, 30, 40, 50, 60, 100],
    labels=[0, 1, 2, 3, 4]
).astype(float)

# 7. High utilization flag (>= 0.8 = risky)
df['HighUtilization'] = (df['RevolvingUtilizationOfUnsecuredLines'] >= 0.8).astype(int)

# 8. Loan density (loans per age decade)
df['LoanDensity'] = df['NumberOfOpenCreditLinesAndLoans'] / (df['age'] / 10)

# 9. Real estate burden
df['RealEstateBurden'] = df['NumberRealEstateLoansOrLines'] * df['DebtRatio']

print("✅ Feature engineering done.")
print(f"Total features: {df.shape[1] - 1}")

# =========================
# 🎯 Split Features / Target
# =========================
X = df.drop(columns=['SeriousDlqin2yrs'])
y = df['SeriousDlqin2yrs']

print(f"\n📊 Class distribution:")
print(y.value_counts())
print(f"Imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.1f}:1")

# =========================
# ✂️ Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# ⚙️ Models (3 approaches)
# =========================

models = {
    "LogReg Balanced": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight='balanced',   # ← حل Class Imbalance
            C=0.1,
            random_state=42
        ))
    ]),

    "RandomForest Balanced": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',   # ← حل Class Imbalance
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "GradientBoosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        ))
    ]),
}

# =========================
# 🚀 Train & Evaluate All
# =========================
results = {}

print("\n" + "="*60)
for name, pipeline in models.items():
    print(f"\n🔄 Training: {name}")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    results[name] = {"auc": auc, "pipeline": pipeline}

    print(f"ROC-AUC Score: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

print("="*60)

# =========================
# 🏆 Best Model Summary
# =========================
best_name = max(results, key=lambda x: results[x]['auc'])
print(f"\n🏆 Best Model: {best_name}")
print(f"   ROC-AUC: {results[best_name]['auc']:.4f}")

# =========================
# 📊 Feature Importance (RandomForest)
# =========================
rf_pipeline = results["RandomForest Balanced"]["pipeline"]
rf_model = rf_pipeline.named_steps["model"]
feature_names = X.columns.tolist()

importances = pd.Series(rf_model.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=True).tail(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Feature Importance
axes[0].barh(importances.index, importances.values, color='steelblue')
axes[0].set_title('Top 15 Feature Importances (RandomForest)', fontsize=13)
axes[0].set_xlabel('Importance')

# Confusion Matrix for best model
best_pipeline = results[best_name]["pipeline"]
y_pred_best = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Default', 'Default'])
disp.plot(ax=axes[1], colorbar=False)
axes[1].set_title(f'Confusion Matrix - {best_name}', fontsize=13)

plt.tight_layout()
plt.savefig("model_results.png", dpi=150, bbox_inches='tight')
print("\n📊 Chart saved: model_results.png")

# =========================
# 💾 ROC-AUC Comparison
# =========================
print("\n📈 Final ROC-AUC Comparison:")
print("-" * 35)
for name, res in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
    print(f"  {name:<25} → {res['auc']:.4f}")
print("-" * 35)
print("\n✅ Done!")