# Credit Risk Prediction Using Machine Learning

## Project Description

This project focuses on developing an end-to-end machine learning solution for predicting customer loan default, a critical task in the financial industry. Leveraging the 'Give Me Some Credit' dataset, the system identifies individuals likely to default on their loans, enabling proactive risk management and informed decision-making for financial institutions. The solution encompasses a comprehensive workflow, from data loading and exploratory analysis to advanced modeling, evaluation, and explainability, ensuring both predictive accuracy and interpretability.

## Business Problem

Loan default poses a significant financial risk to banks and lending institutions. Accurately predicting which customers are likely to default allows these institutions to mitigate potential losses, optimize lending strategies, and allocate resources more effectively. The challenge lies in building a robust and interpretable model that can handle imbalanced datasets, identify key risk factors, and provide actionable insights for business stakeholders. This project addresses this challenge by providing a data-driven approach to credit risk assessment.

## Dataset

The project utilizes the **Give Me Some Credit** dataset, a publicly available dataset commonly used for credit risk modeling. The target variable, `SeriousDlqin2yrs`, indicates whether a customer defaulted on their loan within two years. 

*   **Class 0:** Customer did not default.
*   **Class 1:** Customer defaulted within two years.

## Workflow

The project follows a structured machine learning workflow, ensuring a systematic approach to model development and deployment:

1.  **Data Loading:** Initial ingestion of the raw dataset.
2.  **Exploratory Data Analysis (EDA):** In-depth analysis to understand data characteristics, distributions, and relationships.
3.  **Data Cleaning & Preprocessing:** Handling missing values, outliers, and transforming features for model readiness.
4.  **Train/Test Split:** Dividing the dataset into training (80%) and testing (20%) sets using stratified sampling to maintain class distribution.
5.  **Baseline Model (Random Forest):** Development of an initial model to establish performance benchmarks.
6.  **Challenger Model (XGBoost):** Implementation of a more advanced model to potentially surpass baseline performance.
7.  **Stratified 5-Fold Cross Validation:** Robust model validation technique to ensure generalization and reduce overfitting.
8.  **Threshold Optimization:** Fine-tuning the classification threshold to balance precision and recall based on business objectives.
9.  **Business Cost Optimization:** Integrating business costs into the decision-making process to maximize financial benefits.
10. **Model Evaluation:** Comprehensive assessment of model performance using various metrics.
11. **SHAP Explainability:** Utilizing SHAP (SHapley Additive exPlanations) to interpret model predictions and understand feature importance.
12. **Final Business Conclusions:** Deriving actionable insights and recommendations for business stakeholders.

## Technologies

The project is implemented using Python and leverages a suite of powerful libraries for data science and machine learning:

*   **pandas:** For data manipulation and analysis.
*   **numpy:** For numerical operations.
*   **scikit-learn:** For machine learning algorithms, model selection, and preprocessing.
*   **xgboost:** For the Challenger Model, a highly efficient and flexible gradient boosting library.
*   **shap:** For model explainability and interpretation.
*   **matplotlib & seaborn:** For data visualization.
*   **scipy:** For scientific computing.
*   **tqdm:** For progress bars.
*   **imblearn:** For handling imbalanced datasets.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/credit-risk-prediction.git
    cd credit-risk-prediction
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the environment is set up, you can run the project notebooks or scripts to reproduce the analysis and model training. 

*   **Data Loading and Preprocessing:** Execute the data loading and preprocessing scripts to prepare the dataset.
*   **Model Training:** Run the training scripts for both the Random Forest (baseline) and XGBoost (challenger) models.
*   **Evaluation and Explainability:** Explore the evaluation metrics and SHAP plots to understand model performance and feature contributions.

## Results

The project achieved the following key results:

*   **Optimal Decision Threshold:** 0.44
*   **Expected Cost:** $39,181,200
*   **ROC-AUC:** 0.8445
*   **PR-AUC:** 0.3623
*   **Gini Coefficient:** 0.6889
*   **KS Statistic:** 0.5350
*   **Brier Score:** 0.1226

### Classification Report

**Class 0 (No Default)**
*   Precision: 0.98
*   Recall: 0.80
*   F1-Score: 0.88

**Class 1 (Default)**
*   Precision: 0.21
*   Recall: 0.73
*   F1-Score: 0.32

### Confusion Matrix

```
[[89216 22289]
 [ 2135  5872]]
```

## Evaluation Metrics

The model's performance was rigorously evaluated using a suite of metrics relevant to imbalanced classification and business impact:

*   **ROC-AUC (Receiver Operating Characteristic - Area Under the Curve):** Measures the model's ability to distinguish between positive and negative classes across various threshold settings. A higher ROC-AUC indicates better discrimination.
*   **PR-AUC (Precision-Recall - Area Under the Curve):** Particularly useful for imbalanced datasets, PR-AUC focuses on the trade-off between precision and recall. A higher PR-AUC signifies better performance in identifying positive cases without generating too many false positives.
*   **Gini Coefficient:** Derived from the ROC curve, it's a measure of inequality and model performance, ranging from 0 (random model) to 1 (perfect model).
*   **KS Statistic (Kolmogorov-Smirnov Statistic):** Measures the maximum difference between the cumulative true positive rate and cumulative false positive rate, indicating the model's ability to separate positive and negative classes.
*   **Brier Score:** A proper scoring rule that measures the accuracy of probabilistic predictions. Lower Brier scores indicate better calibration and accuracy.
*   **Precision:** The proportion of true positive predictions among all positive predictions. Important when the cost of false positives is high.
*   **Recall (Sensitivity):** The proportion of true positive predictions among all actual positive cases. Important when the cost of false negatives is high.
*   **F1-Score:** The harmonic mean of precision and recall, providing a balanced measure of a model's accuracy.
*   **Confusion Matrix:** A table that summarizes the performance of a classification algorithm, showing true positives, true negatives, false positives, and false negatives.

## Project Structure

```
credit_risk_prediction/
├── .gitignore
├── README.md
├── REPORT.md
├── DOCUMENTATION.md
├── CONTRIBUTING.md
├── LICENSE
├── requirements.txt
├── data/
│   └── raw/
│       └── GiveMeSomeCredit.csv  # Example raw data file
│   └── processed/
│       └── clean_cs_training.csv # Cleaned dataset
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Evaluation_Explainability.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── models.py
│   └── utils.py
├── reports/
│   └── figures/
│       ├── roc_curve.png
│       ├── pr_curve.png
│       ├── calibration_curve.png
│       ├── shap_summary_plot.png
│       └── feature_importance.png
└── tests/
    ├── test_data_loader.py
    ├── test_preprocessor.py
    └── test_models.py
```

## Future Work

To further enhance the Credit Risk Prediction project, the following areas are recommended for future exploration and development:

*   **Hyperparameter Tuning:** Implement advanced hyperparameter optimization techniques (e.g., GridSearchCV, RandomizedSearchCV, Bayesian Optimization) for both Random Forest and XGBoost models to find the optimal set of parameters that maximize performance.
*   **Ensemble Learning:** Explore ensemble methods beyond simple stacking or blending, such as weighted averaging or more complex meta-learners, to combine the strengths of multiple models and improve predictive accuracy and robustness.
*   **Probability Calibration:** Investigate and apply probability calibration techniques (e.g., Platt Scaling, Isotonic Regression) to ensure that the predicted probabilities are well-calibrated and reflect true likelihoods, which is crucial for risk assessment and decision-making.
*   **Drift Monitoring:** Implement a system for monitoring model and data drift in a production environment. This involves tracking changes in input data distributions and model performance over time, triggering alerts for potential degradation.
*   **Model Retraining:** Establish an automated pipeline for periodic model retraining using fresh data to ensure the model remains relevant and accurate as underlying patterns evolve.
*   **API Deployment:** Develop a RESTful API (e.g., using Flask or FastAPI) to serve the trained model, allowing other applications to easily integrate credit risk predictions. This would involve containerization (e.g., Docker) for consistent deployment.
*   **Streamlit Dashboard:** Create an interactive web dashboard using Streamlit to visualize model performance, feature importance, and allow business users to explore predictions and scenarios in a user-friendly manner.
*   **MLOps Integration:** Integrate the project into a full MLOps pipeline, encompassing automated testing, continuous integration/continuous deployment (CI/CD) for models, version control for data and models, and robust monitoring in production.
*   **Alternative Algorithms:** Explore other advanced machine learning algorithms suitable for imbalanced classification, such as LightGBM, CatBoost, or deep learning approaches, to potentially achieve higher performance.
*   **Feature Engineering:** Conduct more extensive feature engineering, including creating interaction terms, polynomial features, or using domain-specific knowledge to derive new predictive features from the existing dataset.
*   **Advanced Outlier Detection and Treatment:** Investigate more sophisticated outlier detection methods (e.g., Isolation Forest, One-Class SVM) and treatment strategies to improve model robustness.
*   **Fairness and Bias Analysis:** Conduct a thorough analysis of model fairness and potential biases across different demographic groups to ensure equitable outcomes and comply with ethical AI principles.
*   **Cost-Sensitive Learning:** Implement cost-sensitive learning algorithms or adjust existing algorithms to directly optimize for business costs, rather than relying solely on threshold optimization.
*   **Real-time Prediction:** Explore architectures and technologies for real-time credit risk prediction, which would involve streaming data processing and low-latency model serving.
*   **Explainable AI (XAI) Beyond SHAP:** Investigate other XAI techniques like LIME (Local Interpretable Model-agnostic Explanations) or partial dependence plots to provide a more comprehensive understanding of model behavior.

## Team Members

*   Ahmed Abdelrazek
*   Arwa Mahmoud Hassan
*   Malak Adel Abdelrahman
*   Nour Hazem Nasr
*   Sama Mohamed Elsayed
*   Mohamed Reda Mohamed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
