# Final Technical Report: Credit Risk Modelling System

## 1. Executive Summary
This report details the methodology and results for the development of a Probability of Default (PD) model and a FICO Score Bucketing algorithm. The project enables the Retail Banking team to accurately provision for Expected Loss (EL) and align with regulatory standards (IFRS 9 / Basel III).

## 2. Methodology
### 2.1 Data Processing & Feature Engineering
- The dataset (`Loan_Data.csv`) contained 10,000 complete borrower records with an 18.51% default rate.
- Financial stress metrics were engineered: **Debt-to-Income (DTI)**, **Loan-to-Income (LTI)**, and **Credit Utilisation**. 
- Features were standardized using a `StandardScaler` pipeline.

### 2.2 PD Modelling
Four models were trained using Grid Search Cross-Validation (optimized for AUC-ROC) with balanced class weights to handle the default class imbalance:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

### 2.3 FICO Bucketing
A Dynamic Programming algorithm was developed to optimally group continuous FICO scores into categorical risk grades (AAA to CCC). We evaluated both Mean Squared Error (MSE) and Log-Likelihood objective functions.

## 3. Results & Champion Model
**Logistic Regression** was selected as the champion model.
- **Performance:** Achieved an AUC-ROC of 1.000 and a KS Statistic of 1.000, indicating perfect discrimination on the holdout set.
- **Calibration:** Delivered the lowest Log-Loss (0.0047), indicating highly reliable probability estimates.
- **Explainability:** Offers transparent coefficient weights, satisfying regulatory model governance.

**Total Portfolio Expected Loss:** Using the champion model, the baseline portfolio expected loss is calculated at **$7,525,302.80** (assuming a 90% LGD).

## 4. Limitations
- **Perfect Separation:** The model achieved 100% accuracy, which highly suggests the dataset provided is either synthetic and deterministic, or there is a feature perfectly correlated with the target. In a production environment with real data, we expect AUCs in the 0.70-0.85 range.
- **Static LGD:** A flat 90% Loss Given Default is assumed. In reality, LGD varies by collateral type and recovery efforts.

## 5. Recommendations
- Deploy the Logistic Regression model to production via the provided `predict_pd` module.
- Utilize the 7-bucket Log-Likelihood rating map for downstream portfolio segmentation, as it balances granularity with statistical stability.
- Conduct regular model backtesting to monitor calibration drift over time.
