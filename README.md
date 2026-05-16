# 📊 Credit Risk Modelling System

A production-grade **Probability of Default (PD)** model and **dynamic FICO score bucketing** engine, aligned with **IFRS 9 / Basel III** provisioning frameworks.

Built for the Retail Banking Quantitative Risk Team to automate expected loss estimation and provide explainable, audit-ready risk assessments.

---

## 🏗️ Project Structure

```
Credit_Risk_Modeling/
├── app.py                    # Streamlit interactive dashboard
├── src/
│   ├── eda_and_features.py   # Exploratory data analysis & feature engineering
│   ├── fico_bucketing.py     # Dynamic programming FICO bucketing (LL & MSE)
│   ├── pd_modelling.py       # Model training, evaluation & selection
│   └── api.py                # FastAPI inference endpoint
├── notebooks/
│   ├── eda_and_features.ipynb
│   ├── fico_bucketing.ipynb
│   └── pd_modelling.ipynb
├── data/
│   ├── raw/                  # Source data (Loan_Data.csv)
│   └── processed/            # Engineered feature datasets (git-ignored)
├── models/                   # Trained model artifacts (git-ignored)
├── outputs/                  # FICO rating maps & model metrics (git-ignored)
├── reports/
│   ├── Final_Report.md
│   ├── Champion_Rationale.md
│   ├── Data_Quality_Report.md
│   └── figures/              # Generated plots (git-ignored)
├── tests/
│   └── test_system.py        # End-to-end system tests
├── Dockerfile.api            # Docker image for FastAPI service
├── Dockerfile.streamlit      # Docker image for Streamlit dashboard
├── docker-compose.yml        # Multi-service orchestration
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
└── LICENSE
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/Credit_Risk_Modeling.git
cd Credit_Risk_Modeling
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Modelling Pipeline

```bash
# Step 1: EDA & Feature Engineering
python src/eda_and_features.py

# Step 2: FICO Score Bucketing
python src/fico_bucketing.py

# Step 3: Train & Evaluate PD Models
python src/pd_modelling.py
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

### 4. Start the REST API

```bash
uvicorn src.api:app --reload --port 8000
```

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Inference API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📈 Key Results

| Metric | Value |
|---|---|
| Champion Model | Logistic Regression |
| AUC-ROC | 1.000 |
| FICO Buckets | N ∈ {5, 7, 10} |
| Optimisation Methods | Log-Likelihood & MSE Dynamic Programming |
| Portfolio Expected Loss | **$7.52M** (baseline) |
| LGD Assumption | 45% |

---

## 🔬 Methodology

### Probability of Default (PD) Model
Four classifiers were trained and benchmarked: **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**. Logistic Regression was selected as the champion model for its superior AUC-ROC, calibration, and regulatory interpretability.

### FICO Score Bucketing
A dynamic programming algorithm partitions the FICO score range (300–850) into N optimal risk bands by minimising either **Log-Likelihood loss** or **Mean Squared Error** within each bucket. This generates monotone, non-overlapping rating grades aligned with Basel II IRB requirements.

### Expected Loss
```
EL = PD × LGD × EAD
```
Where LGD is assumed at 45% and EAD equals the outstanding loan amount.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data & Modelling | Python, Pandas, NumPy, Scikit-learn, XGBoost |
| Dashboard | Streamlit |
| REST API | FastAPI, Uvicorn |
| Containerisation | Docker, Docker Compose |
| Testing | Pytest |

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

## 🤝 Contributing

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before submitting pull requests.
