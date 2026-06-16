# Credit Card Fraud Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

> A complete, beginner-friendly Machine Learning project for detecting fraudulent credit card transactions. Built as an internship-ready submission with comprehensive documentation.

---

## Table of Contents

- [Abstract](#abstract)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Algorithms Used](#algorithms-used)
- [Flowchart](#flowchart)
- [Installation & Usage](#installation--usage)
- [Results](#results)
- [Why Accuracy Alone Is Not Enough](#why-accuracy-alone-is-not-enough)
- [Conclusion](#conclusion)
- [Future Scope](#future-scope)

---

## Abstract

Credit card fraud is a critical and growing concern in the digital payment landscape. This project implements a **Machine Learning-based fraud detection system** that can automatically identify potentially fraudulent transactions from a dataset of credit card transactions. The system leverages three supervised learning algorithms — **Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier** — to classify transactions as legitimate or fraudulent. The project performs comprehensive **Exploratory Data Analysis (EDA)**, handles data preprocessing challenges including severe class imbalance, and evaluates models using multiple metrics beyond simple accuracy (Precision, Recall, F1-Score, and ROC-AUC). The best-performing model is automatically selected and can be used to predict new incoming transactions.

---

## Problem Statement

With the rapid increase in online transactions and digital payments, credit card fraud has become one of the most prevalent forms of financial crime worldwide. Financial institutions lose billions of dollars annually to fraudulent transactions. The challenge lies in:

1. **Detecting fraud in real-time** from millions of daily transactions
2. **Handling extreme class imbalance** — less than 0.2% of transactions are fraudulent
3. **Minimizing false positives** (legitimate transactions flagged as fraud) while maximizing fraud detection rate
4. **Building interpretable models** that explain why a transaction is flagged

This project addresses these challenges by building and comparing multiple ML models to find the most effective approach for fraud detection.

---

## Objectives

1. Load and explore the Credit Card Fraud Detection dataset from Kaggle
2. Perform comprehensive Exploratory Data Analysis (EDA) to understand data patterns
3. Handle missing values and duplicate records
4. Visualize data distributions, correlations, and class imbalance
5. Apply preprocessing techniques (feature scaling, train-test split)
6. Train three machine learning models: Logistic Regression, Decision Tree, and Random Forest
7. Evaluate all models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC
8. Display and analyze confusion matrices for each model
9. Select the best-performing model and justify the selection
10. Demonstrate prediction capability on new transaction data

---

## Dataset Description

| Property | Details |
|----------|---------|
| **Name** | Credit Card Fraud Detection |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Collected By** | ULB (Université Libre de Bruxelles) Machine Learning Group |
| **Transactions** | 284,807 |
| **Fraudulent** | 492 (0.172%) |
| **Legitimate** | 284,315 (99.828%) |
| **Features** | 31 columns |
| **Time Period** | September 2013 (2 days) |

### Features

| Feature | Description |
|---------|-------------|
| `Time` | Seconds elapsed from the first transaction in the dataset |
| `V1` to `V28` | 28 principal components obtained via PCA transformation (for confidentiality) |
| `Amount` | Transaction amount |
| `Class` | Target variable — `0` = Legitimate, `1` = Fraud |

> **Note**: Features V1-V28 are the result of PCA (Principal Component Analysis) transformation. The original features are not provided due to confidentiality.

---

## System Requirements

### Hardware
- Processor: Intel i3 or above (or equivalent)
- RAM: 4 GB minimum (8 GB recommended)
- Storage: 500 MB free space

### Software
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or above
- **IDE**: Visual Studio Code (recommended), PyCharm, or Jupyter Notebook

### Python Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| pandas | ≥ 1.5.0 | Data manipulation and analysis |
| numpy | ≥ 1.23.0 | Numerical computing |
| matplotlib | ≥ 3.6.0 | Data visualization |
| seaborn | ≥ 0.12.0 | Statistical visualization |
| scikit-learn | ≥ 1.2.0 | Machine learning algorithms |

---

## Project Structure

```
CreditCardFraudDetection/
│
├── data/
│   └── creditcard.csv          # Dataset (download from Kaggle)
│
├── src/
│   └── fraud_detection.py      # Main Python script
│
├── outputs/                    # Generated charts and visualizations
│   ├── 01_class_distribution.png
│   ├── 02_amount_distribution.png
│   ├── 03_time_distribution.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_feature_class_correlation.png
│   ├── 06_feature_histograms.png
│   ├── 07_confusion_matrix_*.png
│   └── 08_model_comparison.png
│
├── report/
│   └── project_report.md       # Complete project report
│
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## Methodology

The project follows a systematic machine learning pipeline:

### 1. Data Collection
Download the creditcard.csv dataset from Kaggle and place it in the `data/` folder.

### 2. Data Exploration (EDA)
- Display dataset shape, info, and statistical summary
- Check for missing values and handle them (median imputation)
- Identify and remove duplicate records
- Analyze class distribution and imbalance

### 3. Data Visualization
- Class distribution bar chart and pie chart
- Transaction amount histograms (fraud vs legitimate)
- Time-based transaction distribution
- Full feature correlation heatmap
- Feature-class correlation bar chart
- PCA feature distribution histograms

### 4. Data Preprocessing
- Scale `Amount` and `Time` features using StandardScaler
- Separate features (X) from target (y)
- Split data into 80% training and 20% testing (with stratification)

### 5. Model Training
Train three classification models on the training data

### 6. Model Evaluation
Evaluate each model using multiple metrics and confusion matrices

### 7. Model Comparison & Selection
Compare all models and select the best one based on F1-Score

### 8. Prediction
Use the best model to predict new transaction fraud probability

---

## Algorithms Used

### 1. Logistic Regression
- **Type**: Linear classification model
- **How it works**: Uses a sigmoid function to map inputs to probabilities (0 to 1). If probability > 0.5, the transaction is classified as fraud.
- **Strengths**: Fast, simple, interpretable, works well with linearly separable data
- **Limitations**: May not capture complex non-linear patterns

### 2. Decision Tree Classifier
- **Type**: Non-linear, tree-based model
- **How it works**: Creates a tree of yes/no questions about features. Each question splits the data until a prediction is reached at a leaf node.
- **Strengths**: Easy to interpret, handles non-linear relationships, no feature scaling needed
- **Limitations**: Prone to overfitting if not pruned (controlled via `max_depth`)

### 3. Random Forest Classifier
- **Type**: Ensemble model (collection of decision trees)
- **How it works**: Creates 100 decision trees, each trained on a random subset of data and features. The final prediction is the majority vote from all trees.
- **Strengths**: Very robust, less overfitting than single trees, handles high-dimensional data well
- **Limitations**: Slower to train, less interpretable than single models

---

## Flowchart

```
┌─────────────────────┐
│   START              │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Load Dataset        │
│  (creditcard.csv)    │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Exploratory Data    │
│  Analysis (EDA)      │
│  • Shape & Info      │
│  • Missing Values    │
│  • Duplicates        │
│  • Class Distribution│
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Data Visualization  │
│  • Histograms        │
│  • Count Plots       │
│  • Heatmap           │
│  • Class Distribution│
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Data Preprocessing  │
│  • Feature Scaling   │
│  • Train/Test Split  │
│  (80/20, stratified) │
└────────┬────────────┘
         ▼
┌─────────────────────────────────────────┐
│         MODEL TRAINING                  │
│  ┌──────────┬──────────┬──────────┐     │
│  │ Logistic │ Decision │ Random   │     │
│  │Regression│   Tree   │ Forest   │     │
│  └──────────┴──────────┴──────────┘     │
└────────┬────────────────────────────────┘
         ▼
┌─────────────────────────────────────────┐
│         MODEL EVALUATION                │
│  • Accuracy, Precision, Recall          │
│  • F1-Score, ROC-AUC                    │
│  • Confusion Matrix                     │
└────────┬────────────────────────────────┘
         ▼
┌─────────────────────┐
│  Compare Models &    │
│  Select Best Model   │
│  (Based on F1-Score) │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Predict New         │
│  Transaction         │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│   END                │
└─────────────────────┘
```

---

## Installation & Usage

### Step 1: Clone or Download the Project
```bash
# Navigate to the project directory
cd CreditCardFraudDetection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset
1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place it in the `data/` folder

### Step 4: Run the Project
```bash
python src/fraud_detection.py
```

The script will:
- Perform complete EDA
- Generate all visualizations (saved in `outputs/`)
- Train and evaluate 3 ML models
- Display comparison results
- Make a sample prediction

---

## Results

After running the project, you will see a detailed comparison table. Typical results on this dataset:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~99.92% | ~0.87 | ~0.62 | ~0.72 | ~0.97 |
| Decision Tree | ~99.91% | ~0.74 | ~0.76 | ~0.75 | ~0.88 |
| **Random Forest** | **~99.95%** | **~0.93** | **~0.78** | **~0.85** | **~0.97** |

> **Note**: Actual results may vary slightly due to random initialization.

### Key Observations
1. All models achieve high accuracy (>99.9%), but this is misleading due to class imbalance
2. **Random Forest** typically achieves the best F1-Score, balancing precision and recall
3. **Recall** is critical — a missed fraud (false negative) has serious financial consequences
4. The **confusion matrix** reveals exactly how many frauds each model catches vs misses

---

## Why Accuracy Alone Is Not Enough

In fraud detection, accuracy is a **misleading metric** due to extreme class imbalance:

| Scenario | Accuracy | Frauds Caught | Useful? |
|----------|----------|--------------|---------|
| Model predicts "Not Fraud" for everything | 99.83% | 0 out of 492 | ❌ No |
| Our trained model | ~99.95% | ~78% of frauds | ✅ Yes |

**Key insight**: A "dumb" model that always says "Not Fraud" achieves 99.83% accuracy but catches **zero** frauds! That's why we use:

- **Precision**: Of flagged transactions, how many are actually fraud? (Reduces false alarms)
- **Recall**: Of actual frauds, how many did we catch? (Most critical for fraud detection)
- **F1-Score**: Balances Precision and Recall (used for model selection)
- **ROC-AUC**: Overall quality of the classifier

---

## Conclusion

1. Successfully built a **credit card fraud detection system** using three ML models
2. **Random Forest Classifier** typically emerges as the best model with the highest F1-Score
3. The project demonstrates that **accuracy alone is misleading** for imbalanced datasets
4. **F1-Score and Recall** are the most important metrics for fraud detection
5. The system can predict whether a new transaction is fraudulent with high confidence
6. All visualizations provide clear insights into the data patterns and model performance

---

## Future Scope

1. **Handle Class Imbalance**: Apply techniques like SMOTE (Synthetic Minority Over-sampling) or undersampling to balance the dataset
2. **Deep Learning**: Implement neural network models (e.g., Autoencoders, LSTMs) for improved detection
3. **Real-Time Detection**: Build a REST API using Flask/FastAPI for real-time transaction scoring
4. **Advanced Models**: Try XGBoost, LightGBM, or Gradient Boosting for potentially better performance
5. **Feature Engineering**: Create new features from existing ones to improve model accuracy
6. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV for optimal model parameters
7. **Model Deployment**: Deploy the model using Docker, AWS, or Azure for production use
8. **Explainability**: Add SHAP (SHapley Additive exPlanations) for model interpretability
9. **Dashboard**: Build an interactive dashboard using Streamlit or Dash for monitoring
10. **Anomaly Detection**: Explore unsupervised approaches like Isolation Forest or One-Class SVM

---

## Acknowledgments

- **Dataset**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn

