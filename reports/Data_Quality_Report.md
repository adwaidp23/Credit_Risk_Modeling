# Data Quality Report

## 1. Dataset Overview
- **File:** `Loan_Data.csv`
- **Total Records:** 10,000 borrower rows
- **Number of Features:** 8 (1 identifier, 6 predictors, 1 target variable)

## 2. Completeness
- **Missing Values:** There are 0 missing values across all columns. The dataset is 100% complete and does not require imputation.

## 3. Variable Analysis
- **`default` (Target Variable):** Binary indicator with an 18.51% default rate. This mild class imbalance should be considered during model training (e.g., using `class_weight='balanced'` or SMOTE).
- **`fico_score`:** Ranges appropriately from 408 to 850, with a mean of ~638. Lower FICO scores exhibit a strong correlation with higher default rates.
- **`income`:** Values range from $1,000 to $148,412.
- **`credit_lines_outstanding`:** Values range from 0 to 5.
- **`loan_amt_outstanding` & `total_debt_outstanding`:** Continuous variables; ratio analysis (DTI, LTI) reveals critical stress points where defaults concentrate.

## 4. EDA Findings
- **Feature Correlation:** The variables most highly correlated with default are `fico_score` (negative correlation) and `dti_ratio` (positive correlation).
- **Non-Linear Relationships:** We observed that at extremely low income levels, default risk is disproportionately high.
- **Outliers:** No extreme, erroneous outliers detected. All variables fall within realistic banking bounds.

## 5. Conclusion
The dataset is of excellent quality, clean, and ready for further modelling. Feature Engineering effectively derived more predictive continuous indicators (`dti_ratio`, `lti_ratio`, `credit_utilisation`) without exposing data leakage or colinearity issues.
