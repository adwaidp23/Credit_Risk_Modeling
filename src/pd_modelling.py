# %% [markdown]
# # Phase 3 & 4: PD Modelling & Evaluation
# This notebook covers the training of Logistic Regression, Decision Tree, Random Forest, and XGBoost models, followed by rigorous evaluation.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from scipy.stats import ks_2samp

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load Data and Preprocessor
df = pd.read_csv('Loan_Data_Enriched.csv')
features = [col for col in df.columns if col not in ['customer_id', 'default']]
X = df[features]
y = df['default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load pipeline
preprocessor = joblib.load('preprocessing_pipeline.pkl')
X_train_scaled = preprocessor.transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Scale positive weight for class imbalance
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# %% [markdown]
# ## Phase 3: Model Training with Grid Search
# We will train 4 models: Logistic Regression, Decision Tree, Random Forest, XGBoost.

# %%
models = {}

# 1. Logistic Regression
print("Training Logistic Regression...")
param_grid_lr = {'C': [0.01, 0.1, 1, 10]}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'), param_grid_lr, cv=5, scoring='roc_auc')
grid_lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = grid_lr.best_estimator_
joblib.dump(grid_lr.best_estimator_, 'lr_model.pkl')

# 2. Decision Tree
print("Training Decision Tree...")
param_grid_dt = {'max_depth': [3, 5, 7, 10], 'min_samples_leaf': [10, 20, 50]}
grid_dt = GridSearchCV(DecisionTreeClassifier(class_weight='balanced', random_state=42), param_grid_dt, cv=5, scoring='roc_auc')
grid_dt.fit(X_train_scaled, y_train)
models['Decision Tree'] = grid_dt.best_estimator_
joblib.dump(grid_dt.best_estimator_, 'dt_model.pkl')

# 3. Random Forest
print("Training Random Forest...")
param_grid_rf = {'n_estimators': [50, 100], 'max_features': ['sqrt', 'log2']}
grid_rf = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid_rf, cv=5, scoring='roc_auc')
grid_rf.fit(X_train_scaled, y_train)
models['Random Forest'] = grid_rf.best_estimator_
joblib.dump(grid_rf.best_estimator_, 'rf_model.pkl')

# 4. XGBoost
print("Training XGBoost...")
param_grid_xgb = {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
grid_xgb = GridSearchCV(XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss'), param_grid_xgb, cv=5, scoring='roc_auc')
grid_xgb.fit(X_train_scaled, y_train)
models['XGBoost'] = grid_xgb.best_estimator_
joblib.dump(grid_xgb.best_estimator_, 'xgb_model.pkl')

print("All models trained and saved.")

# %% [markdown]
# ## Phase 4: Model Evaluation
# We will generate ROC, KS, Confusion Matrices, and Calibration curves.

# %%
results = []
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
fig_cal, ax_cal = plt.subplots(figsize=(8, 6))

ax_cal.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

for name, model in models.items():
    # Predict
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    # ROC / AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc_score = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")
    
    # KS Statistic
    prob_default = y_pred_prob[y_test == 1]
    prob_nondefault = y_pred_prob[y_test == 0]
    ks_stat, _ = ks_2samp(prob_nondefault, prob_default)
    
    # Log-loss
    ll_score = log_loss(y_test, y_pred_prob)
    
    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
    ax_cal.plot(prob_pred, prob_true, marker='o', label=name)
    
    # Save results
    results.append({
        'Model': name,
        'AUC-ROC': auc_score,
        'KS Statistic': ks_stat,
        'Log-Loss': ll_score
    })
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'cm_{name.replace(" ", "_").lower()}.png')
    plt.close()

# Finalize ROC plot
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve Comparison')
ax_roc.legend(loc="lower right")
fig_roc.tight_layout()
fig_roc.savefig('roc_curves.png')
plt.close(fig_roc)

# Finalize Calibration plot
ax_cal.set_xlabel('Mean predicted probability')
ax_cal.set_ylabel('Fraction of positives')
ax_cal.set_title('Calibration Curves')
ax_cal.legend(loc="upper left")
fig_cal.tight_layout()
fig_cal.savefig('calibration_curves.png')
plt.close(fig_cal)

results_df = pd.DataFrame(results).set_index('Model')
print("\nModel Evaluation Results:")
print(results_df)
results_df.to_csv('model_comparison.csv')

# %% [markdown]
# ### Champion Model Selection
# Based on the results, we will select the best model (usually the one with the highest AUC and best calibration).

# %%
champion_name = results_df['AUC-ROC'].idxmax()
champion_model = models[champion_name]
print(f"\nChampion Model selected: {champion_name}")

# %% [markdown]
# ### Define `predict_pd` and `calculate_expected_loss` functions
# We use the champion model to predict Probability of Default (PD) and compute Expected Loss (EL).

# %%
LGD = 0.90 # 10% recovery rate

def predict_pd(borrower_dict, model, preprocessor, feature_names):
    """Predicts Probability of Default for a single borrower."""
    df_input = pd.DataFrame([borrower_dict], columns=feature_names)
    X_scaled = preprocessor.transform(df_input)
    pd_val = model.predict_proba(X_scaled)[0, 1]
    return pd_val

def calculate_expected_loss(borrower_dict, model, preprocessor, feature_names, lgd=LGD):
    """Calculates Expected Loss (EL) for a single borrower."""
    pd_val = predict_pd(borrower_dict, model, preprocessor, feature_names)
    ead = borrower_dict['loan_amt_outstanding']
    el = pd_val * ead * lgd
    return el, pd_val

# Test functions
sample_borrower = X.iloc[0].to_dict()
el, pd_val = calculate_expected_loss(sample_borrower, champion_model, preprocessor, features)
print(f"\nSample Borrower PD: {pd_val:.4f}, EL: ${el:.2f}")

# %% [markdown]
# ### Total Expected Loss on Full Loan Book
# Compute EL for all current borrowers using the champion model.

# %%
X_full_scaled = preprocessor.transform(X)
all_pds = champion_model.predict_proba(X_full_scaled)[:, 1]
df['predicted_pd'] = all_pds
df['expected_loss'] = df['predicted_pd'] * df['loan_amt_outstanding'] * LGD

total_el = df['expected_loss'].sum()
print(f"\nTotal Portfolio Expected Loss: ${total_el:,.2f}")
df.to_csv('Loan_Data_Scored.csv', index=False)
