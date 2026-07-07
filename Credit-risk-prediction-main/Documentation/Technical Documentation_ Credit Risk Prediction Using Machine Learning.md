# Technical Documentation: Credit Risk Prediction Using Machine Learning

## 1. Project Architecture

The Credit Risk Prediction project is designed with a modular and scalable architecture, adhering to best practices for machine learning development. The core principle is to separate concerns, ensuring that each component is responsible for a specific task, thereby enhancing maintainability, testability, and reusability. The architecture can be broadly categorized into data ingestion, data processing, model development, model evaluation, and deployment preparation.

At its foundation, the system relies on a structured data flow, beginning with raw data input and progressing through various transformation stages. The use of distinct modules for each stage (e.g., data loading, preprocessing, modeling) allows for independent development and easier integration. This modularity also facilitates the adoption of MLOps principles in future iterations, enabling continuous integration, continuous delivery, and continuous monitoring of the machine learning pipeline.

The architecture is designed to be flexible, allowing for easy interchangeability of components. For instance, different preprocessing techniques or modeling algorithms can be experimented with by modifying specific modules without affecting the entire pipeline. This flexibility is crucial for iterative development and for adapting to evolving business requirements or data characteristics. The system is primarily built using Python, leveraging its rich ecosystem of data science and machine learning libraries, which are integrated to form a cohesive and efficient workflow.

## 2. Folder Structure

The project adheres to a well-defined folder structure to organize code, data, and documentation, promoting clarity and ease of navigation for developers and stakeholders. This structure is designed to be intuitive and scalable, accommodating future expansions of the project.

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
│       └── GiveMeSomeCredit.csv  # Original, immutable dataset
│   └── processed/
│       └── clean_cs_training.csv # Cleaned and preprocessed dataset
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb    # Data Cleaning and Preprocessing steps
│   ├── 03_Modeling.ipynb         # Model training and selection
│   └── 04_Evaluation_Explainability.ipynb # Model evaluation and SHAP analysis
├── src/
│   ├── __init__.py               # Makes 'src' a Python package
│   ├── data_loader.py            # Functions for loading raw data
│   ├── preprocessor.py           # Functions for data cleaning and transformation
│   ├── models.py                 # Classes/functions for model definition and training
│   └── utils.py                  # Utility functions (e.g., metrics, visualization helpers)
├── reports/
│   └── figures/
│       ├── roc_curve.png         # ROC Curve visualization
│       ├── pr_curve.png          # Precision-Recall Curve visualization
│       ├── calibration_curve.png # Calibration Curve visualization
│       ├── shap_summary_plot.png # SHAP Summary Plot
│       └── feature_importance.png # Feature Importance Plot
└── tests/
    ├── __init__.py               # Makes 'tests' a Python package
    ├── test_data_loader.py       # Unit tests for data_loader.py
    ├── test_preprocessor.py      # Unit tests for preprocessor.py
    └── test_models.py            # Unit tests for models.py
```

**Explanation of Directories:**

*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `README.md`: Provides a high-level overview of the project, its purpose, installation, and usage.
*   `REPORT.md`: Contains the academic report detailing the project's methodology, results, and conclusions.
*   `DOCUMENTATION.md`: This technical documentation, detailing architecture, code, and pipelines.
*   `CONTRIBUTING.md`: Guidelines for contributing to the project.
*   `LICENSE`: The project's license (MIT License).
*   `requirements.txt`: Lists all Python dependencies required for the project.
*   `data/`: Stores all data files.
    *   `raw/`: Contains the original, immutable raw dataset. This ensures data provenance and reproducibility.
    *   `processed/`: Stores cleaned and preprocessed datasets, ready for model training.
*   `notebooks/`: Houses Jupyter notebooks used for exploratory analysis, experimentation, and step-by-step workflow demonstration. These are typically for development and research, not production.
*   `src/`: Contains the core Python source code, organized into modular scripts.
    *   `data_loader.py`: Encapsulates functions responsible for loading data from various sources.
    *   `preprocessor.py`: Contains functions and classes for data cleaning, feature engineering, and transformation.
    *   `models.py`: Defines the machine learning models, including their architecture, training logic, and prediction methods.
    *   `utils.py`: A collection of helper functions, such as custom evaluation metrics, visualization utilities, or common data manipulation routines.
*   `reports/`: Stores generated reports, figures, and other output artifacts.
    *   `figures/`: Contains plots and visualizations generated during EDA, model evaluation, and explainability analysis.
*   `tests/`: Contains unit tests for the Python modules in the `src/` directory, ensuring code quality and correctness.

## 3. Source Code Modules

The `src/` directory is the heart of the project's codebase, containing well-structured Python modules that implement the core functionalities. Each module is designed to be cohesive and loosely coupled, promoting reusability and maintainability.

### `data_loader.py`

This module is responsible for all data ingestion operations. Its primary function is to load the raw dataset into a suitable data structure (e.g., pandas DataFrame) for further processing. Key functionalities include:

*   **`load_raw_data(file_path: str) -> pd.DataFrame`**: A function that takes the path to the raw data file (e.g., `GiveMeSomeCredit.csv`) and returns a pandas DataFrame. It handles file reading, ensuring proper encoding and initial data type inference.
*   **Error Handling**: Includes mechanisms to handle file not found errors, corrupted files, or issues during data parsing, providing informative error messages.
*   **Configuration**: May incorporate configuration parameters (e.g., column names, data types) to make the loading process flexible and adaptable to minor changes in the raw data schema.

### `preprocessor.py`

This module encapsulates all data cleaning, transformation, and feature engineering steps. It is designed to convert raw data into a clean, normalized, and feature-rich format suitable for machine learning models. Key functionalities include:

*   **`remove_duplicates(df: pd.DataFrame) -> pd.DataFrame`**: Identifies and removes duplicate rows from the DataFrame.
*   **`handle_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame`**: Implements strategies for outlier detection and treatment (e.g., winsorization, capping) for specified numerical columns.
*   **`feature_engineering(df: pd.DataFrame) -> pd.DataFrame`**: Creates new features from existing ones to enhance model predictive power. Examples might include interaction terms, polynomial features, or ratio features relevant to credit risk.
*   **`scale_features(df: pd.DataFrame, scaler_type: str = 'MinMaxScaler') -> pd.DataFrame`**: Applies feature scaling (e.g., MinMaxScaler, StandardScaler) to numerical features, which is often crucial for distance-based algorithms and gradient-based optimization.
*   **`preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame`**: Orchestrates the sequence of preprocessing steps, applying them in a defined order to the input DataFrame. This function serves as the main entry point for data preparation.
*   **Data Validation**: Includes internal checks to ensure data quality and consistency after each transformation step.

### `models.py`

This module contains the implementation of the machine learning models used in the project, including both the baseline (Random Forest) and challenger (XGBoost) models. It provides functions or classes for model instantiation, training, prediction, and saving/loading trained models.

*   **`train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> RandomForestClassifier`**: Trains a Random Forest classifier with specified hyperparameters on the training data.
*   **`train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> xgb.XGBClassifier`**: Trains an XGBoost classifier with specified hyperparameters on the training data.
*   **`predict(model, X: pd.DataFrame) -> np.ndarray`**: Generates predictions (class labels) using a trained model.
*   **`predict_proba(model, X: pd.DataFrame) -> np.ndarray`**: Generates probability predictions using a trained model.
*   **`save_model(model, path: str)`**: Serializes and saves a trained model to disk.
*   **`load_model(path: str)`**: Deserializes and loads a trained model from disk.

### `utils.py`

This module serves as a repository for general utility functions that support various aspects of the project but do not fit neatly into other specific modules. This includes custom metrics, visualization helpers, and other common functionalities.

*   **`calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict`**: Computes a comprehensive set of evaluation metrics (e.g., ROC-AUC, PR-AUC, Gini, KS, Brier, Precision, Recall, F1-score, Confusion Matrix) given true labels, predicted labels, and predicted probabilities.
*   **`plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: str)`**: Generates and saves the Receiver Operating Characteristic (ROC) curve.
*   **`plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: str)`**: Generates and saves the Precision-Recall (PR) curve.
*   **`plot_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: str)`**: Generates and saves the calibration curve.
*   **`plot_shap_summary(model, X: pd.DataFrame, save_path: str)`**: Generates and saves a SHAP summary plot to visualize global feature importance.
*   **`plot_feature_importance(model, feature_names: list, save_path: str)`**: Generates and saves a feature importance plot (e.g., based on Gini importance for tree-based models).
*   **`cost_optimization(y_true: np.ndarray, y_proba: np.ndarray, cost_matrix: dict) -> tuple`**: Implements the business cost optimization logic to find the optimal decision threshold and calculate the minimum expected cost.

## 4. Data Flow

The data flow within the Credit Risk Prediction project is a sequential process designed to transform raw input into actionable insights. It ensures data integrity, reproducibility, and efficient processing through distinct stages.

1.  **Raw Data Ingestion:** The process begins with the `data_loader.py` module, which reads the `GiveMeSomeCredit.csv` file from the `data/raw/` directory. This raw data is the initial input, containing all original features and the target variable.

2.  **Data Preprocessing:** The loaded raw data is then passed to the `preprocessor.py` module. Here, a series of transformations are applied:
    *   Duplicate records are identified and removed.
    *   Outliers in critical financial features are treated (e.g., winsorization or capping).
    *   Potentially, new features are engineered to capture more complex relationships.
    *   Numerical features are scaled to a consistent range (e.g., 0-1 or mean 0, variance 1).
    The output of this stage is a clean, processed dataset, `clean_cs_training.csv`, which is saved in the `data/processed/` directory.

3.  **Train/Test Split:** The `clean_cs_training.csv` dataset is loaded and split into training and testing sets. Crucially, this split is performed using stratified sampling (80% train, 20% test) to ensure that the proportion of default cases is maintained in both subsets, which is vital for robust model training and evaluation, especially with imbalanced data.

4.  **Model Training:** The training data (`X_train`, `y_train`) is fed into the `models.py` module. Both the Random Forest (baseline) and XGBoost (challenger) models are instantiated and trained using this data. During training, the models learn the underlying patterns and relationships between the features and the target variable.

5.  **Prediction:** Once trained, the models are used to generate predictions on the unseen test data (`X_test`). This involves both class label predictions (`y_pred`) and probability predictions (`y_proba`).

6.  **Model Evaluation:** The predictions (`y_pred`, `y_proba`) and the true labels (`y_test`) are passed to the `utils.py` module for comprehensive evaluation. This module calculates various performance metrics (ROC-AUC, PR-AUC, Gini, KS, Brier, Precision, Recall, F1-score, Confusion Matrix) and generates relevant visualizations (ROC curve, PR curve, Calibration curve). These metrics and plots are saved in the `reports/figures/` directory.

7.  **Threshold and Business Cost Optimization:** The `utils.py` module also performs threshold optimization, identifying the optimal decision threshold that minimizes the predefined business cost. This step integrates the financial implications directly into the model's operational decision-making.

8.  **SHAP Explainability:** For interpretability, the trained model and test data are used to compute SHAP values via the `utils.py` module. This generates insights into feature contributions for individual predictions and overall model behavior, with visualizations saved in `reports/figures/`.

This structured data flow ensures traceability, reproducibility, and a clear understanding of how data is transformed and utilized throughout the machine learning pipeline.

## 5. Model Pipeline

The model pipeline orchestrates the sequence of operations from raw data to trained models, ensuring a streamlined and reproducible process. It integrates the functionalities provided by the `data_loader.py`, `preprocessor.py`, and `models.py` modules.

1.  **Data Loading:** The pipeline begins by invoking `data_loader.py` to load the raw `GiveMeSomeCredit.csv` dataset. This step retrieves the initial data for processing.

2.  **Data Preprocessing:** The loaded raw data is then passed to the `preprocessor.py` module, specifically utilizing the `preprocess_pipeline` function. This function executes a series of predefined steps:
    *   Duplicate removal.
    *   Outlier treatment.
    *   Feature engineering (if any).
    *   Feature scaling.
    The output is the `clean_cs_training.csv` dataset, which is then used for model training.

3.  **Train/Test Split:** The preprocessed data is split into training and testing sets using `sklearn.model_selection.train_test_split` with `stratify` parameter set to the target variable to maintain class distribution. This ensures that both the training and testing sets are representative of the overall dataset's class balance.

4.  **Model Instantiation and Training:**
    *   **Baseline Model (Random Forest):** An instance of `RandomForestClassifier` from `sklearn.ensemble` is created with predefined hyperparameters. The model is then trained using the `X_train` and `y_train` datasets.
    *   **Challenger Model (XGBoost):** An instance of `xgb.XGBClassifier` is created with its specific hyperparameters. This model is also trained using the `X_train` and `y_train` datasets.

5.  **Model Persistence:** After training, both the Random Forest and XGBoost models are serialized and saved to disk using the `save_model` function from `models.py`. This allows for easy loading and deployment of the trained models without needing to retrain them every time.

This pipeline ensures that the models are trained on consistently prepared data, facilitating fair comparison and reliable performance.

## 6. Evaluation Pipeline

The evaluation pipeline is designed to rigorously assess the performance of the trained models, providing a comprehensive understanding of their strengths and weaknesses. It leverages the `utils.py` module for metric calculation and visualization.

1.  **Model Loading:** The trained models (Random Forest and XGBoost) are loaded from disk using the `load_model` function from `models.py`. This ensures that the evaluation is performed on the final, trained models.

2.  **Prediction on Test Set:** The loaded models are used to generate predictions on the `X_test` dataset. Both class labels (`y_pred`) and probability scores (`y_proba`) are obtained.

3.  **Metric Calculation:** The `calculate_metrics` function from `utils.py` is invoked, taking `y_true`, `y_pred`, and `y_proba` as inputs. This function computes a wide array of performance metrics, including:
    *   ROC-AUC
    *   PR-AUC
    *   Gini Coefficient
    *   KS Statistic
    *   Brier Score
    *   Precision, Recall, F1-score for each class
    *   Confusion Matrix
    These metrics provide a quantitative assessment of the model's performance from various perspectives.

4.  **Visualization Generation:** The `utils.py` module is used to generate several key visualizations that aid in understanding model behavior:
    *   `plot_roc_curve`: Visualizes the trade-off between true positive rate and false positive rate.
    *   `plot_pr_curve`: Illustrates the trade-off between precision and recall, particularly useful for imbalanced datasets.
    *   `plot_calibration_curve`: Assesses how well the predicted probabilities align with actual probabilities.
    *   `plot_shap_summary`: Provides a global view of feature importance and impact.
    *   `plot_feature_importance`: Shows the relative importance of features as determined by the model.
    All generated plots are saved as image files (e.g., PNG) in the `reports/figures/` directory.

5.  **Threshold and Business Cost Optimization:** The `cost_optimization` function from `utils.py` is applied to the model's probability predictions and true labels, along with a predefined cost matrix. This step identifies the optimal decision threshold that minimizes the total expected business cost, providing a financially informed evaluation of the model.

6.  **SHAP Explainability:** SHAP values are computed for the model's predictions on the test set. This involves using the `shap` library to generate explanations for individual predictions and aggregate insights into global feature importance. The results are often visualized using `plot_shap_summary` and other SHAP-specific plots.

This comprehensive evaluation pipeline ensures that the models are thoroughly assessed, not only on statistical performance but also on their business impact and interpretability.

## 7. Deployment Preparation

Preparing the credit risk prediction model for deployment involves packaging the trained model and its dependencies, ensuring it can be easily integrated into a production environment. The goal is to create a robust, scalable, and maintainable solution.

1.  **Model Serialization:** The trained machine learning model (e.g., the optimized XGBoost model) is serialized using `joblib` or `pickle` (or `dill` for more complex objects) and saved to a persistent storage location. This allows the model to be loaded quickly in a production environment without needing to retrain it.

2.  **Dependency Management:** The `requirements.txt` file precisely lists all Python libraries and their versions used in the project. This file is crucial for recreating the exact development environment in production, preventing dependency conflicts and ensuring consistent model behavior.

3.  **API Development (Recommended):** For real-time predictions, it is highly recommended to wrap the model inference logic within a RESTful API. Frameworks like Flask or FastAPI are excellent choices for this purpose. The API would expose an endpoint that accepts customer features as input and returns credit risk predictions (e.g., probability of default, class label).

4.  **Containerization (Recommended):** Packaging the application (including the API, model, and dependencies) into a Docker container is a best practice for deployment. Docker ensures that the application runs consistently across different environments (development, staging, production) by encapsulating everything it needs to run. This eliminates 
the 
problem of "it works on my machine" and simplifies deployment.

5.  **Orchestration and Automation:** For complex pipelines, tools like Apache Airflow, MLflow, or Kubeflow can be used to orchestrate the entire machine learning workflow, from data ingestion to model deployment. This ensures automation, monitoring, and reproducibility of the pipeline.

## 8. Maintenance Guide

Maintaining the Credit Risk Prediction system in a production environment is crucial for ensuring its continued accuracy, reliability, and relevance. This guide outlines key maintenance activities.

### 8.1. Data Monitoring

*   **Data Quality Checks:** Regularly monitor incoming data for quality issues such as missing values, incorrect data types, and out-of-range values. Implement automated alerts for anomalies.
*   **Data Drift Detection:** Track changes in the distribution of input features over time. Significant data drift can indicate that the model is being fed data different from what it was trained on, potentially leading to performance degradation.
*   **Concept Drift Detection:** Monitor changes in the relationship between input features and the target variable. Concept drift means the underlying patterns the model learned are no longer valid, necessitating model retraining.

### 8.2. Model Monitoring

*   **Performance Monitoring:** Continuously track key model performance metrics (e.g., ROC-AUC, PR-AUC, precision, recall, F1-score) on live data. Compare these metrics against established baselines to detect performance degradation.
*   **Prediction Drift:** Monitor the distribution of model predictions over time. Sudden shifts can indicate issues with the model or changes in the input data.
*   **Explainability Monitoring:** Periodically analyze SHAP values or other explainability outputs to ensure that feature importances remain consistent and logical. Unexpected changes might signal model issues or data anomalies.

### 8.3. Retraining and Redeployment

*   **Scheduled Retraining:** Establish a schedule for periodic model retraining (e.g., monthly, quarterly) using fresh data. The frequency should be determined by the rate of data and concept drift observed.
*   **Triggered Retraining:** Implement automated triggers for retraining when significant data or concept drift is detected, or when model performance falls below acceptable thresholds.
*   **Version Control for Models:** Maintain strict version control for trained models, ensuring that each deployed model can be traced back to its training data, code, and hyperparameters.
*   **A/B Testing:** When deploying new model versions, consider A/B testing to compare the performance of the new model against the existing one in a live environment before full rollout.

### 8.4. Infrastructure and Security

*   **Resource Monitoring:** Monitor computational resources (CPU, memory, GPU) and storage utilization to ensure the system has adequate capacity and to detect potential bottlenecks.
*   **Security Audits:** Regularly conduct security audits of the deployment environment and API endpoints to identify and mitigate vulnerabilities.
*   **Access Control:** Implement robust access control mechanisms to ensure that only authorized personnel can access and modify the model and its infrastructure.

### 8.5. Documentation Updates

*   **Keep Documentation Current:** Ensure that all technical documentation, including this guide, is regularly updated to reflect any changes in the model, data pipeline, or deployment strategy.
*   **Incident Response Plan:** Develop and maintain an incident response plan for addressing model failures, performance degradation, or security breaches.

By adhering to this maintenance guide, the Credit Risk Prediction system can remain a valuable and reliable asset for financial institutions, continuously adapting to new data and business challenges.
