# Loan Approval Prediction Using Machine Learning

## Overview

This project uses supervised machine learning to predict whether a loan application is likely to be approved based on applicant, financial, and loan-related information. It is implemented in Python in a Jupyter Notebook.

## Dataset

The project uses `loan_approval_data.csv`.

- **Records:** 1,000
- **Columns:** 20
- **Prediction target:** `Loan_Approved` (`Yes` / `No`)
- **Target counts in the original CSV:** 298 Yes, 652 No, 50 missing

The dataset includes fields such as applicant income, co-applicant income, employment status, age, credit score, existing loans, debt-to-income ratio, savings, collateral value, loan amount, loan term, loan purpose, property area, education level, gender, and employer category.

## Workflow

The notebook covers the following steps:

1. Load the dataset and inspect its structure and summary statistics.
2. Handle missing values using mean imputation for numerical columns and most-frequent-value imputation for categorical columns.
3. Perform exploratory data analysis with class distribution charts, histograms, box plots, and a correlation heatmap.
4. Remove the `Applicant_ID` identifier column.
5. Encode categorical features using label encoding and one-hot encoding.
6. Separate features (`X`) and target (`y`), then split the data into training and testing sets.
7. Scale features with `StandardScaler`.
8. Train and evaluate Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes classifiers.
9. Apply feature engineering by adding squared `DTI_Ratio` and `Credit_Score` features, then train and evaluate the models again.

## Machine Learning Models

- Logistic Regression
- K-Nearest Neighbors (KNN, with `n_neighbors=5`)
- Gaussian Naive Bayes

The notebook evaluates models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The notebook's saved results show that, after the feature-engineering step, Logistic Regression has the highest accuracy among the three evaluated models (87.5%). Model choice should not be based on accuracy alone; precision, recall, the confusion matrix, and the cost of incorrect loan decisions should also be considered.

## Project Structure

```text
loan-approval-prediction-project/
├── loan_approval_prediction.ipynb
├── loan_approval_data.csv
└── README.md
```

## Requirements

Install the Python libraries used by the notebook:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## How to Run

1. Download or clone this repository.
2. Keep `loan_approval_prediction.ipynb` and `loan_approval_data.csv` in the same directory.
3. Install the required libraries.
4. Start Jupyter Notebook or JupyterLab:

   ```bash
   jupyter lab
   ```

5. Open `loan_approval_prediction.ipynb`.
6. Run the notebook cells from top to bottom.

## Important Modeling Note

The current notebook applies the most-frequent-value imputer to all categorical columns, which includes `Loan_Approved`. The original dataset contains 50 rows with a missing target value. For a more reliable evaluation, remove rows where `Loan_Approved` is missing before training, and fit preprocessing steps only on the training data (ideally using a scikit-learn `Pipeline` and `ColumnTransformer`). This helps avoid target-label imputation and data leakage.

## Disclaimer

This is an educational machine learning project. Its predictions are not financial advice and should not be used as the sole basis for real lending decisions. A production loan-decision system would require stronger validation, fairness and bias testing, explainability, privacy safeguards, and compliance with applicable regulations.
