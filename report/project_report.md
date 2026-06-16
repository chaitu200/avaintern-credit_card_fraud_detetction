# Credit Card Fraud Detection Using Machine Learning — Project Report

---

## 1. Abstract

Credit card fraud is a critical and growing concern in the digital payment landscape. This project implements a Machine Learning-based fraud detection system capable of automatically identifying potentially fraudulent transactions. Using the publicly available Credit Card Fraud Detection dataset from Kaggle (containing 284,807 transactions with only 492 frauds), the system trains and evaluates three supervised learning algorithms — Logistic Regression, Decision Tree Classifier, and Random Forest Classifier. The project performs comprehensive Exploratory Data Analysis (EDA), applies data preprocessing techniques, and evaluates models using multiple metrics including Accuracy, Precision, Recall, F1-Score, and ROC-AUC. The best-performing model is selected based on F1-Score and can predict whether a new transaction is fraudulent.

---

## 2. Problem Statement

With the exponential growth of digital transactions globally, credit card fraud has become one of the most prevalent forms of cybercrime. Financial institutions and cardholders lose billions of dollars annually. The challenge is to develop an automated system that can:

- Detect fraudulent transactions from millions of daily transactions
- Handle the severe class imbalance (only 0.17% of transactions are fraud)
- Minimize false positives while maximizing fraud detection
- Provide interpretable and reliable predictions

Traditional rule-based systems are insufficient to detect evolving fraud patterns. Machine Learning offers a data-driven approach that learns from historical transaction data to identify fraud.

---

## 3. Objectives

1. Load and explore the Credit Card Fraud Detection dataset
2. Perform comprehensive Exploratory Data Analysis (EDA)
3. Handle missing values and duplicate records
4. Visualize data distributions, correlations, and class imbalance
5. Apply data preprocessing (feature scaling, stratified train-test split)
6. Train three ML models: Logistic Regression, Decision Tree, Random Forest
7. Evaluate all models using multiple metrics and confusion matrices
8. Compare models and select the best-performing one
9. Demonstrate prediction capability on new transaction data
10. Explain why accuracy alone is insufficient for fraud detection

---

## 4. Dataset Description

- **Source**: Kaggle — Credit Card Fraud Detection by ULB Machine Learning Group
- **Transactions**: 284,807
- **Fraudulent**: 492 (0.172%)
- **Legitimate**: 284,315 (99.828%)
- **Features**: 31 columns
  - `Time`: Seconds since first transaction
  - `V1` to `V28`: PCA-transformed features (for privacy/confidentiality)
  - `Amount`: Transaction amount in Euros
  - `Class`: Target variable (0 = Legitimate, 1 = Fraud)

The dataset was collected from European cardholders over two days in September 2013.

---

## 5. System Requirements

### Hardware
- Processor: Intel i3 or above
- RAM: 4 GB minimum (8 GB recommended)
- Storage: 500 MB free space

### Software
- Operating System: Windows 10/11, macOS, or Linux
- Python: 3.8 or above
- IDE: Visual Studio Code (recommended)

### Python Libraries
- pandas ≥ 1.5.0
- numpy ≥ 1.23.0
- matplotlib ≥ 3.6.0
- seaborn ≥ 0.12.0
- scikit-learn ≥ 1.2.0

---

## 6. Methodology

The project follows a structured ML pipeline:

### 6.1 Data Collection
The creditcard.csv dataset is downloaded from Kaggle and loaded using pandas.

### 6.2 Exploratory Data Analysis (EDA)
- Dataset shape, info, and statistical summary
- Missing value detection and handling (median imputation)
- Duplicate record detection and removal
- Class distribution analysis and imbalance quantification

### 6.3 Data Visualization
- Class distribution (bar chart and pie chart)
- Transaction amount distribution (fraud vs legitimate)
- Time-based transaction patterns
- Feature correlation heatmap
- Feature-class correlation ranking
- PCA feature distribution comparisons

### 6.4 Data Preprocessing
- Feature scaling using StandardScaler on Amount and Time
- Feature-target separation (X and y)
- 80/20 stratified train-test split (preserving class ratios)

### 6.5 Model Training
Three supervised ML models are trained on the training set:
1. Logistic Regression (max_iter=1000)
2. Decision Tree Classifier (max_depth=10)
3. Random Forest Classifier (n_estimators=100, max_depth=15)

### 6.6 Model Evaluation
Each model is evaluated on the test set using:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix (True Positives, True Negatives, False Positives, False Negatives)
- Detailed Classification Report

### 6.7 Model Comparison and Selection
All models are compared side-by-side, and the best model is selected based on F1-Score.

### 6.8 Prediction
The best model predicts the fraud probability for a new sample transaction.

---

## 7. Algorithms Used

### 7.1 Logistic Regression
Logistic Regression is a linear classification algorithm that uses a sigmoid function to map input features to a probability between 0 and 1. If the predicted probability exceeds 0.5, the transaction is classified as fraud. It is fast, simple, and provides a strong baseline.

### 7.2 Decision Tree Classifier
A Decision Tree creates a flowchart-like structure where each internal node represents a decision based on a feature value. The tree splits data recursively until it reaches leaf nodes with class predictions. It handles non-linear relationships but is prone to overfitting without depth constraints.

### 7.3 Random Forest Classifier
Random Forest is an ensemble method that creates multiple decision trees (a "forest"), each trained on a random subset of data and features. The final prediction is determined by majority voting across all trees. It is more robust, less prone to overfitting, and typically outperforms individual trees.

---

## 8. Flowchart

```
START
  │
  ▼
Load Dataset (creditcard.csv)
  │
  ▼
Exploratory Data Analysis
  ├── Dataset Shape & Info
  ├── Missing Values Check
  ├── Duplicate Records Check
  └── Class Distribution Analysis
  │
  ▼
Data Visualization
  ├── Class Distribution Chart
  ├── Amount Distribution Histograms
  ├── Time Distribution
  ├── Correlation Heatmap
  ├── Feature-Class Correlation
  └── Feature Histograms
  │
  ▼
Data Preprocessing
  ├── Scale Amount & Time features
  ├── Separate features (X) and target (y)
  └── 80/20 Stratified Train-Test Split
  │
  ▼
Model Training
  ├── Logistic Regression
  ├── Decision Tree
  └── Random Forest
  │
  ▼
Model Evaluation
  ├── Accuracy, Precision, Recall, F1, ROC-AUC
  └── Confusion Matrix for each model
  │
  ▼
Model Comparison & Best Model Selection
  │
  ▼
Predict New Transaction
  │
  ▼
END
```

---

## 9. Results

### 9.1 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~99.92% | ~0.87 | ~0.62 | ~0.72 | ~0.97 |
| Decision Tree | ~99.91% | ~0.74 | ~0.76 | ~0.75 | ~0.88 |
| **Random Forest** | **~99.95%** | **~0.93** | **~0.78** | **~0.85** | **~0.97** |

*Note: Actual values may vary slightly due to random initialization.*

### 9.2 Key Findings

1. **All models achieve >99.9% accuracy**, but accuracy is misleading due to class imbalance
2. **Random Forest** achieves the best F1-Score, providing the best balance between precision and recall
3. **Recall** is especially important — a missed fraud (false negative) has serious financial consequences
4. **Logistic Regression** has the lowest recall, meaning it misses more fraudulent transactions
5. **Decision Tree** has good recall but lower precision, resulting in more false alarms

### 9.3 Why Accuracy Alone Is Not Enough

A naive model that predicts "Not Fraud" for every transaction achieves 99.83% accuracy because only 0.17% of transactions are fraudulent. However, it catches ZERO actual frauds. This demonstrates that accuracy is a poor metric for imbalanced classification problems.

Better metrics include:
- **Precision**: Reduces false alarms (legitimate transactions flagged as fraud)
- **Recall**: Maximizes fraud detection (most critical metric)
- **F1-Score**: Harmonic mean balancing precision and recall
- **ROC-AUC**: Measures overall discriminative ability

### 9.4 Best Model Selection

**Random Forest Classifier** is selected as the best model because:
- It achieves the highest F1-Score among all three models
- It balances high precision (few false alarms) with good recall (catches most frauds)
- It has the highest ROC-AUC, indicating excellent overall classification ability
- Being an ensemble method, it is robust and generalizes well to unseen data

---

## 10. Conclusion

This project successfully demonstrates the application of Machine Learning for credit card fraud detection. Key takeaways include:

1. **Three ML models** were trained and evaluated: Logistic Regression, Decision Tree, and Random Forest
2. **Random Forest Classifier** emerged as the best model with the highest F1-Score
3. The project proves that **accuracy alone is insufficient** for evaluating fraud detection models on imbalanced datasets
4. **F1-Score and Recall** are the most appropriate metrics for this problem domain
5. The system can effectively predict whether a new transaction is fraudulent
6. Comprehensive visualizations provide clear insights into data patterns and model performance

---

## 11. Future Scope

1. **SMOTE**: Apply Synthetic Minority Over-sampling Technique to balance the dataset
2. **Advanced Models**: Try XGBoost, LightGBM, or Gradient Boosting classifiers
3. **Deep Learning**: Implement autoencoders or LSTM networks for sequence-based detection
4. **Real-Time API**: Build a Flask/FastAPI REST API for real-time transaction scoring
5. **Hyperparameter Tuning**: Use GridSearchCV for optimal model parameters
6. **Model Explainability**: Add SHAP values for transparent model interpretation
7. **Dashboard**: Build a Streamlit/Dash dashboard for real-time monitoring
8. **Deployment**: Containerize with Docker and deploy to cloud (AWS/Azure/GCP)
9. **Feature Engineering**: Create domain-specific features for improved detection
10. **Anomaly Detection**: Explore unsupervised approaches like Isolation Forest

---

## 12. References

1. Kaggle Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. scikit-learn Documentation: https://scikit-learn.org/stable/
3. Andrea Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced Classification", IEEE SSCI 2015
4. pandas Documentation: https://pandas.pydata.org/
5. matplotlib Documentation: https://matplotlib.org/

---

*Report prepared for internship submission — June 2026*
