# %% [markdown]
# # Credit Risk Modeling - Phase 1 & 2: Setup, EDA & Feature Engineering
# This notebook covers Exploratory Data Analysis and Feature Engineering.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# %% [markdown]
# ## Phase 1: EDA
# Load the dataset and verify shape, dtypes, null counts, and value ranges.

# %%
df = pd.read_csv('Loan_Data.csv')
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)
print("\nSummary Statistics:\n", df.describe())

# %% [markdown]
# ### Plot distribution of each feature
# Saving histograms and box plots for the continuous features.

# %%
numerical_features = ['loan_amt_outstanding', 'total_debt_outstanding', 'income', 'fico_score', 'years_employed', 'credit_lines_outstanding']

# Create distributions
for col in numerical_features:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[col], kde=True, ax=ax[0])
    ax[0].set_title(f'Distribution of {col}')
    
    sns.boxplot(x='default', y=col, data=df, ax=ax[1])
    ax[1].set_title(f'{col} by Default Status')
    
    plt.tight_layout()
    plt.savefig(f'{col}_distribution.png')
    plt.close()

# %% [markdown]
# ### Analyse default rate by FICO band, income decile, credit lines, employment years

# %%
# FICO Band
df['fico_band'] = pd.cut(df['fico_score'], bins=[0, 580, 670, 740, 800, 850], labels=['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional'])
fico_default = df.groupby('fico_band', observed=False)['default'].mean()

plt.figure(figsize=(8, 5))
sns.barplot(x=fico_default.index, y=fico_default.values)
plt.title('Default Rate by FICO Band')
plt.ylabel('Default Rate')
plt.savefig('default_rate_by_fico_band.png')
plt.close()

# Income Decile
df['income_decile'] = pd.qcut(df['income'], 10, labels=False)
income_default = df.groupby('income_decile', observed=False)['default'].mean()

plt.figure(figsize=(8, 5))
sns.barplot(x=income_default.index, y=income_default.values)
plt.title('Default Rate by Income Decile')
plt.ylabel('Default Rate')
plt.savefig('default_rate_by_income_decile.png')
plt.close()

# Credit Lines
lines_default = df.groupby('credit_lines_outstanding', observed=False)['default'].mean()
plt.figure(figsize=(8, 5))
sns.barplot(x=lines_default.index, y=lines_default.values)
plt.title('Default Rate by Credit Lines Outstanding')
plt.ylabel('Default Rate')
plt.savefig('default_rate_by_credit_lines.png')
plt.close()

# Years Employed
emp_default = df.groupby('years_employed', observed=False)['default'].mean()
plt.figure(figsize=(8, 5))
sns.barplot(x=emp_default.index, y=emp_default.values)
plt.title('Default Rate by Years Employed')
plt.ylabel('Default Rate')
plt.savefig('default_rate_by_years_employed.png')
plt.close()

# %% [markdown]
# ### Compute pairwise correlations; generate heatmap

# %%
plt.figure(figsize=(10, 8))
corr = df.drop(columns=['fico_band', 'income_decile', 'customer_id']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# %% [markdown]
# ## Phase 2: Feature Engineering
# Raw features are augmented with derived ratios that capture the relative financial stress of each borrower.

# %%
# Drop temporary EDA columns
df = df.drop(columns=['fico_band', 'income_decile'])

# 2.1 Compute Debt-to-Income (DTI) ratio
df['dti_ratio'] = df['total_debt_outstanding'] / df['income']

# 2.2 Compute Loan-to-Income (LTI) ratio
df['lti_ratio'] = df['loan_amt_outstanding'] / df['income']

# 2.3 Compute credit utilisation
# Maximum credit lines is 5 based on data
df['credit_utilisation'] = df['credit_lines_outstanding'] / 5

# Display enriched dataset info
print("\nEnriched Dataset Head:\n", df.head())
df.to_csv('Loan_Data_Enriched.csv', index=False)

# %% [markdown]
# ### Assess feature importance using a quick Random Forest fit

# %%
features = [col for col in df.columns if col not in ['customer_id', 'default']]
X = df[features]
y = df['default']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importance.values, y=importance.index)
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.savefig('feature_importance.png')
plt.close()

# %% [markdown]
# ### Preprocessing Pipeline & Train/Test Split

# %%
# Train/test split: 80% train, 20% test (stratified by default)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# Scale continuous features using StandardScaler for logistic regression
continuous_features = ['loan_amt_outstanding', 'total_debt_outstanding', 'income', 
                       'fico_score', 'years_employed', 'credit_lines_outstanding',
                       'dti_ratio', 'lti_ratio', 'credit_utilisation']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_features)
    ], remainder='passthrough')

# Fit pipeline on training data
preprocessor.fit(X_train)

# Save the preprocessor pipeline
joblib.dump(preprocessor, 'preprocessing_pipeline.pkl')

print("\nPreprocessing pipeline saved to 'preprocessing_pipeline.pkl'.")
print("Phase 1 & 2 Completed Successfully.")
