# Champion Model Selection Rationale

## 1. Model Evaluation Metrics

Based on the model training and cross-validation across four candidate algorithms, the following performance metrics were observed on the hold-out test set:

| Model | AUC-ROC | KS Statistic | Log-Loss |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **1.000000** | **1.000000** | **0.004754** |
| Decision Tree | 0.995381 | 0.950025 | 0.091337 |
| Random Forest | 0.999870 | 0.989935 | 0.015846 |
| XGBoost | 0.999982 | 0.996070 | 0.010506 |

## 2. Selection Rationale

The **Logistic Regression** model has been selected as the Champion Model for the Probability of Default (PD) task for the following reasons:

1. **Perfect Discrimination:** The model achieved an AUC of 1.000 and a KS Statistic of 1.000, indicating perfect separation between defaulters and non-defaulters in this dataset.
2. **Best Calibration:** It achieved the lowest log-loss (0.004754) among all models, meaning its predicted probabilities are exceptionally well-calibrated and reliable.
3. **Regulatory Alignment & Interpretability:** In banking contexts (IFRS 9 / Basel III), interpretability is crucial. Logistic regression provides clear coefficients indicating exactly how each feature (e.g., FICO score, DTI ratio) impacts the probability of default, making it superior to black-box models like Random Forest and XGBoost for this use case.
4. **Simplicity:** It is the least computationally expensive model to score in production.

## 3. Portfolio Expected Loss

Using the Champion Model, the Total Portfolio Expected Loss (EL) across all current borrowers was calculated.
- **Formula:** $EL = PD \times EAD \times LGD$
- **LGD Assumption:** 90% (10% recovery rate)
- **Total Portfolio EL:** **$7,525,302.80**
