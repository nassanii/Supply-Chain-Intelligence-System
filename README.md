# Supply Chain Intelligence System: Late Delivery Risk Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade Machine Learning system built to forecast delivery delay risks (`Late_delivery_risk`) in global supply chain operations using real-world transactional data from DataCo Supply Chain.

---

## Executive Summary & Business Value

In global logistics and supply chain management, late deliveries result in customer dissatisfaction, contract penalties, and increased operational churn. This system processes raw transactional order data, engineers predictive features, transforms variables to avoid data leakage, and provides a modular production inference engine (`src/predict.py`) to evaluate new incoming supply chain orders.

---

## Project Architecture & Modular Design

The codebase follows professional software engineering standards, strictly separating exploratory analysis (notebooks) from modular, reusable production Python code (`src/`):

```text
Supply Chain Intelligence System/
│
├── data/
│   ├── raw/                       # Original raw dataset
│   └── processed/                 # Cleaned, featured & compressed datasets (.npz)
│
├── models/
│   ├── preprocessor.joblib        # Fitted Scikit-Learn ColumnTransformer pipeline
│   └── best_model.joblib          # Saved XGBoost classifier artifact
│
├── notebooks/                     # Exploratory Data Analysis & Prototyping
│   ├── data_cleaning.ipynb        # Data sanitization & missing value handling
│   ├── eda.ipynb                  # Exploratory Data Analysis & visual insights
│   ├── feature_engineering.ipynb  # Creation of temporal & transactional features
│   ├── preprocessing.ipynb        # Pipeline construction & data split verification
│   └── model_training.ipynb       # Model comparison, evaluation & hyperparameter tuning
│
├── src/                           # Production Source Code
│   ├── preprocessing.py           # Modular data loader, transformer & splitting engine
│   └── predict.py                 # Core inference pipeline class
│
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## Pipeline Stages & Methodology

### 1. Data Cleaning & Feature Engineering
* Handled missing values dynamically.
* Removed data leakage variables (`Days for shipping (real)`, `Delivery Status`, `Order Status`).
* Engineered temporal features: `order month`, `order_dayOfWeek`, `is_weekend`, `order_total_value`, and `Is_International`.

### 2. Data Preprocessing (Leakage-Free)
* Applied `train_test_split` with `stratify=y` **before** fitting scalers/encoders to strictly prevent Data Leakage.
* **Numerical Features**: Standardized via `StandardScaler`.
* **Low-Cardinality Categorical Features**: Encoded via `OneHotEncoder(handle_unknown='ignore')`.
* **High-Cardinality Categorical Features**: Encoded via `OrdinalEncoder` to avoid the curse of dimensionality.

### 3. Model Benchmarking & Results
Three models were benchmarked on the stratified test dataset:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Best Model)** | **0.825** | **0.810** | **0.842** | **0.826** | **0.891** |
| Random Forest Classifier | 0.812 | 0.798 | 0.835 | 0.816 | 0.878 |
| Logistic Regression | 0.694 | 0.680 | 0.710 | 0.695 | 0.745 |

*Winning Model*: **XGBoost Classifier** achieved the highest overall F1-Score and ROC-AUC (0.891).

---

## Quick Start Instructions

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/Supply-Chain-Intelligence-System.git
cd "Supply Chain Intelligence System"

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Preprocessing Pipeline
To process raw data and fit preprocessor artifacts:
```bash
python src/preprocessing.py
```

### 3. Run Production Inference Engine
To run predictions on new incoming order payloads:
```bash
python src/predict.py
```

---

## Author
Developed as a production-grade Machine Learning portfolio system.
