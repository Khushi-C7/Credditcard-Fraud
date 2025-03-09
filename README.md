# Credit Card Fraud Detection Analysis

This project analyzes a credit card transaction dataset to detect fraudulent activities using machine learning techniques.

## Dataset

The dataset used is `creditcard.csv`, which contains credit card transactions made in September 2013 by European cardholders. It includes features like `Time`, `V1` to `V28` (anonymized features), `Amount`, and the target variable `Class` (1 for fraud, 0 for normal).

## Libraries Used

-   `pandas`: For data manipulation and analysis.
-   `numpy`: For numerical operations.
-   `matplotlib`: For data visualization.
-   `seaborn`: For enhanced data visualization.
-   `sklearn`: For machine learning algorithms and metrics.
-   `scipy`: For scientific computing.
-   `pycaret`: For automated machine learning workflows.
-   `google.colab`: For executing the code in a Colab environment.

## Project Structure

1.  **Data Loading and Exploration:**
    -   Loaded the `creditcard.csv` dataset.
    -   Checked the shape of the dataset and the presence of null values.
    -   Visualized the distribution of normal and fraudulent transactions.
    -   Analyzed the `Amount` feature for both normal and fraudulent transactions.
    -   Generated a correlation heatmap to understand feature relationships.
2.  **Data Preprocessing:**
    -   Separated the features (`X`) and target variable (`y`).
    -   Split the data into training and testing sets.
    -   Imputed missing values in the training and testing sets using `SimpleImputer`.
3.  **Anomaly Detection with Isolation Forest:**
    -   Implemented the Isolation Forest algorithm for anomaly detection.
    -   Trained the model on the training data.
    -   Made predictions on the test data.
    -   Evaluated the model's performance using accuracy, classification report, and confusion matrix.
    -   Counted the number of errors made by the model.
4.  **Anomaly Detection with One-Class SVM:**
    -   Implemented the One-Class SVM algorithm for anomaly detection.
    -   Trained the model on the imputed training data.
    -   Made predictions on the test data.
    -   Evaluated the model's performance using accuracy, classification report, and confusion matrix.
    -   Counted the number of errors made by the model.
5.  **Automated Machine Learning with PyCaret:**
    -   Used PyCaret for automated machine learning workflows.
    -   Set up the environment with the loaded data and target variable.
    -   Compared different models using `compare_models()`.
    -   Created and tuned a Random Forest model.
    -   Made predictions on the test data using the tuned model.

## Key Findings

-   The dataset is highly imbalanced, with significantly more normal transactions than fraudulent ones.
-   Fraudulent transactions tend to have a different distribution of `Amount` compared to normal transactions.
-   Both Isolation Forest and One-Class SVM were used to detect anomalies.
-   PyCaret simplifies the process of comparing and tuning machine learning models.
-   Random Forest performed well after tuning.

## Usage

1.  Ensure you have the required libraries installed.
2.  Place the `creditcard.csv` file in the same directory as the script or update the file path accordingly.
3.  Run the script in a Python environment or Google Colab.
4.  The results and visualizations will be displayed.

## Future Improvements

-   Explore other anomaly detection algorithms.
-   Implement more advanced feature engineering techniques.
-   Address the class imbalance issue using techniques like oversampling or undersampling.
-   Fine-tune the models further to improve performance.
-   Deploy the model for real-time fraud detection.
-   Test and compare more models using Pycaret.
-   Consider using cross validation techniques for better model evaluation.# Credditcard-Fraud
