"""
=============================================================================
    CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
=============================================================================

Project Title : Credit Card Fraud Detection Using Machine Learning
Author        : [Your Name]
Date          : June 2026
Dataset       : creditcard.csv (Kaggle - Credit Card Fraud Detection)
                https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
=============================================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# Fix Unicode encoding for Windows console (cp1252 cannot render special chars)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------------------------

def setup_directories():
    """Create the outputs directory and return project root + output path."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory ready: {output_dir}")
    return project_root, output_dir


def configure_plot_style():
    """Apply a consistent, professional visual style for all charts."""
    sns.set_style("darkgrid")
    sns.set_palette("viridis")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 100,
        "savefig.dpi": 150,
    })
    print("[INFO] Plot style configured.")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_dataset(project_root):
    """Load creditcard.csv; exit with a helpful message if missing."""
    data_path = os.path.join(project_root, "data", "creditcard.csv")

    if not os.path.exists(data_path):
        print("=" * 70)
        print("  ERROR: Dataset file not found!")
        print("=" * 70)
        print(f"\n  Expected: {data_path}")
        print("\n  Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"  Place it in:   {os.path.join(project_root, 'data')}")
        print("=" * 70)
        sys.exit(1)

    print("\n[INFO] Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"[OK]   Loaded {df.shape[0]:,} transactions, {df.shape[1]} features.")
    return df


# ---------------------------------------------------------------------------
# Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------

def display_basic_info(df):
    """Print shape, dtypes, head/tail, and descriptive statistics."""
    print("\n" + "=" * 70)
    print("  EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    print(f"\nDataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\nDataset Information:")
    print("-" * 50)
    df.info()

    print("\nFirst 5 Rows:")
    print("-" * 50)
    print(df.head().to_string())

    print("\nLast 5 Rows:")
    print("-" * 50)
    print(df.tail().to_string())

    print("\nStatistical Summary:")
    print("-" * 50)
    print(df.describe().to_string())

    print(f"\nColumn Names ({len(df.columns)} total):")
    print(list(df.columns))


def check_missing_values(df):
    """Detect and fill missing values with column medians."""
    print("\n" + "-" * 50)
    print("Checking for Missing Values:")
    print("-" * 50)

    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing == 0:
        print("[OK] No missing values found.")
    else:
        print(f"[WARN] Total missing values: {total_missing}")
        print(missing[missing > 0])
        print("[INFO] Filling missing values with column median...")
        df = df.fillna(df.median(numeric_only=True))
        print("[OK] Missing values handled.")

    return df


def check_duplicates(df):
    """Detect and remove exact duplicate rows."""
    print("\n" + "-" * 50)
    print("Checking for Duplicate Records:")
    print("-" * 50)

    num_duplicates = df.duplicated().sum()

    if num_duplicates == 0:
        print("[OK] No duplicate records found.")
    else:
        print(f"[WARN] Found {num_duplicates:,} duplicate records.")
        df = df.drop_duplicates()
        print(f"[OK] Duplicates removed. New shape: {df.shape}")

    return df


def analyze_class_distribution(df):
    """Print class counts, percentages, and imbalance ratio."""
    print("\n" + "-" * 50)
    print("Class Distribution Analysis:")
    print("-" * 50)

    class_counts = df["Class"].value_counts()
    class_pct = df["Class"].value_counts(normalize=True) * 100

    print(f"\n  Legitimate (0): {class_counts[0]:>10,}  ({class_pct[0]:.3f}%)")
    print(f"  Fraudulent (1): {class_counts[1]:>10,}  ({class_pct[1]:.3f}%)")
    print(f"  {'=' * 45}")
    print(f"  Total:          {len(df):>10,}")

    ratio = class_counts[0] / class_counts[1]
    print(f"\n  Imbalance Ratio: 1 fraud per {ratio:.0f} legitimate transactions")
    print(f"\n  KEY INSIGHT: A naive 'always legitimate' model would score")
    print(f"  {class_pct[0]:.2f}% accuracy but catch ZERO frauds.")
    print("  This is why we rely on Precision, Recall, F1, and ROC-AUC.")


# ---------------------------------------------------------------------------
# Data Visualization
# ---------------------------------------------------------------------------

COLORS = {"legit": "#2ecc71", "fraud": "#e74c3c"}


def _save_figure(output_dir, filename):
    """Save the current figure and close it."""
    path = os.path.join(output_dir, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def plot_class_distribution(df, output_dir):
    """Bar chart + pie chart of fraud vs legitimate counts."""
    print("\n[INFO] Generating Class Distribution Chart...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = [COLORS["legit"], COLORS["fraud"]]
    class_counts = df["Class"].value_counts()

    axes[0].bar(
        ["Legitimate (0)", "Fraud (1)"],
        class_counts.values,
        color=colors,
        edgecolor="black",
    )
    axes[0].set_title("Transaction Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Number of Transactions")
    axes[0].set_xlabel("Transaction Class")
    for i, count in enumerate(class_counts.values):
        axes[0].text(i, count + 500, f"{count:,}", ha="center", fontweight="bold", fontsize=11)

    axes[1].pie(
        class_counts.values,
        labels=["Legitimate", "Fraud"],
        autopct="%1.3f%%",
        colors=colors,
        startangle=90,
        explode=(0, 0.1),
        shadow=True,
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Class Distribution (Percentage)", fontweight="bold")

    plt.tight_layout()
    _save_figure(output_dir, "01_class_distribution.png")


def plot_transaction_amount_distribution(df, output_dir):
    """Histograms of transaction amounts split by class."""
    print("[INFO] Generating Transaction Amount Distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df[df["Class"] == 0]["Amount"], bins=50, color=COLORS["legit"],
                 edgecolor="black", alpha=0.7)
    axes[0].set_title("Legitimate Transaction Amounts", fontweight="bold")
    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, 2500)

    axes[1].hist(df[df["Class"] == 1]["Amount"], bins=50, color=COLORS["fraud"],
                 edgecolor="black", alpha=0.7)
    axes[1].set_title("Fraudulent Transaction Amounts", fontweight="bold")
    axes[1].set_xlabel("Amount ($)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    _save_figure(output_dir, "02_amount_distribution.png")


def plot_time_distribution(df, output_dir):
    """Overlaid histograms of transaction time (hours) by class."""
    print("[INFO] Generating Time Distribution Plot...")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.hist(df[df["Class"] == 0]["Time"] / 3600, bins=48,
            color=COLORS["legit"], alpha=0.6, label="Legitimate", edgecolor="black")
    ax.hist(df[df["Class"] == 1]["Time"] / 3600, bins=48,
            color=COLORS["fraud"], alpha=0.8, label="Fraud", edgecolor="black")

    ax.set_title("Transaction Distribution Over Time", fontweight="bold")
    ax.set_xlabel("Time (Hours from First Transaction)")
    ax.set_ylabel("Number of Transactions")
    ax.legend()

    plt.tight_layout()
    _save_figure(output_dir, "03_time_distribution.png")


def plot_correlation_heatmap(df, output_dir):
    """Full feature correlation heatmap."""
    print("[INFO] Generating Correlation Heatmap...")

    corr_matrix = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(20, 16))

    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=16)

    plt.tight_layout()
    _save_figure(output_dir, "04_correlation_heatmap.png")


def plot_feature_correlation_with_class(df, output_dir):
    """Horizontal bar chart ranking features by correlation to Class."""
    print("[INFO] Generating Feature-Class Correlation Chart...")

    correlations = df.corr(numeric_only=True)["Class"].drop("Class")
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=True).index)

    fig, ax = plt.subplots(figsize=(10, 10))
    bar_colors = [COLORS["fraud"] if x > 0 else "#3498db" for x in correlations.values]
    ax.barh(correlations.index, correlations.values, color=bar_colors, edgecolor="black", alpha=0.8)
    ax.set_title("Feature Correlation with Fraud (Class)", fontweight="bold")
    ax.set_xlabel("Correlation Coefficient")
    ax.axvline(x=0, color="black", linewidth=0.8)

    plt.tight_layout()
    _save_figure(output_dir, "05_feature_class_correlation.png")


def plot_feature_histograms(df, output_dir):
    """Density histograms for the 12 most important PCA features."""
    print("[INFO] Generating Feature Histograms...")

    features = ["V1", "V2", "V3", "V4", "V10", "V11",
                "V12", "V14", "V16", "V17", "V18", "V19"]

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))

    for idx, (ax, feat) in enumerate(zip(axes.flatten(), features)):
        ax.hist(df[df["Class"] == 0][feat], bins=50, color=COLORS["legit"],
                alpha=0.6, label="Legit", density=True)
        ax.hist(df[df["Class"] == 1][feat], bins=50, color=COLORS["fraud"],
                alpha=0.7, label="Fraud", density=True)
        ax.set_title(f"{feat} Distribution", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    plt.suptitle("Feature Distributions: Legitimate vs Fraud",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_figure(output_dir, "06_feature_histograms.png")


# ---------------------------------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------------------------------

def preprocess_data(df):
    """Scale Amount/Time, split into 80/20 train/test with stratification."""
    print("\n" + "=" * 70)
    print("  DATA PREPROCESSING")
    print("=" * 70)

    scaler = StandardScaler()

    df["Scaled_Amount"] = scaler.fit_transform(df[["Amount"]])
    print("[INFO] 'Amount' scaled with StandardScaler.")

    df["Scaled_Time"] = scaler.fit_transform(df[["Time"]])
    print("[INFO] 'Time' scaled with StandardScaler.")

    df = df.drop(columns=["Time", "Amount"])
    print("[INFO] Original 'Time' and 'Amount' columns dropped.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    print(f"\n[INFO] Features (X): {X.shape}  |  Target (y): {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"[INFO] Test:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    print(f"[INFO] Train frauds: {(y_train == 1).sum():,} / {len(y_train):,}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train):
    """Train and return a Logistic Regression classifier."""
    print("\n  Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("  [OK] Logistic Regression trained.")
    return model


def train_decision_tree(X_train, y_train):
    """Train and return a Decision Tree classifier (max_depth=10)."""
    print("  Training Decision Tree Classifier...")
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    print("  [OK] Decision Tree trained.")
    return model


def train_random_forest(X_train, y_train):
    """Train and return a Random Forest classifier (100 trees)."""
    print("  Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("  [OK] Random Forest trained.")
    return model


# ---------------------------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name, output_dir):
    """Compute metrics, print results, save confusion matrix, return dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    print(f"\n  {'=' * 45}")
    print(f"  {model_name} -- Evaluation Results")
    print(f"  {'=' * 45}")
    for key in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        val = metrics[key]
        print(f"  | {key:<10s}: {val:.4f}  ({val * 100:.2f}%)")
    print(f"  {'=' * 45}")

    print(f"\n  Classification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"]).plot(
        ax=ax, cmap="Blues", values_format="d"
    )
    ax.set_title(f"Confusion Matrix -- {model_name}", fontweight="bold", fontsize=14)

    safe_name = model_name.replace(" ", "_").lower()
    _save_figure(output_dir, f"07_confusion_matrix_{safe_name}.png")

    return metrics


# ---------------------------------------------------------------------------
# Model Comparison & Selection
# ---------------------------------------------------------------------------

def compare_models(results_list, output_dir):
    """Print comparison table, plot grouped bar chart, return best index."""
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(results_list)
    print("\n" + results_df.set_index("Model").to_string(float_format=lambda x: f"{x:.4f}"))

    best_idx = results_df["F1-Score"].idxmax()
    best_name = results_df.loc[best_idx, "Model"]
    best_f1 = results_df.loc[best_idx, "F1-Score"]

    print(f"\n  BEST MODEL: {best_name}")
    print(f"  F1-Score:   {best_f1:.4f} ({best_f1 * 100:.2f}%)")
    print(f"\n  WHY F1-SCORE?")
    print(f"  Accuracy alone is misleading on imbalanced data (~99.8% legit).")
    print(f"  F1-Score balances Precision (fewer false alarms) and")
    print(f"  Recall (catching actual frauds).")

    # Grouped bar chart
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metric_cols))
    width = 0.25
    bar_colors = ["#3498db", "#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, name in enumerate(results_df["Model"]):
        vals = results_df[results_df["Model"] == name][metric_cols].values.flatten()
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=bar_colors[i], edgecolor="black", alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Evaluation Metrics", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Model Comparison Across All Metrics", fontweight="bold", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_cols)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_figure(output_dir, "08_model_comparison.png")

    return best_idx, results_df


# ---------------------------------------------------------------------------
# Prediction on a New Transaction
# ---------------------------------------------------------------------------

def predict_new_transaction(model, model_name, feature_columns):
    """Generate a random sample transaction and predict fraud probability."""
    print("\n" + "=" * 70)
    print("  PREDICT A NEW TRANSACTION")
    print("=" * 70)
    print(f"\n  Model: {model_name}")
    print(f"  Features required: {len(feature_columns)}")
    print("\n  Generating a sample transaction for demonstration...")

    sample_data = {col: np.random.normal(0, 1) for col in feature_columns}
    sample_df = pd.DataFrame([sample_data])

    prediction = model.predict(sample_df)[0]
    probability = model.predict_proba(sample_df)[0]

    label = "LEGITIMATE TRANSACTION" if prediction == 0 else "FRAUDULENT TRANSACTION"
    print(f"\n  {'=' * 50}")
    print(f"  PREDICTION: {label}")
    print(f"  {'=' * 50}")
    print(f"  | P(Legitimate): {probability[0]:.4f}  ({probability[0] * 100:.2f}%)")
    print(f"  | P(Fraud):      {probability[1]:.4f}  ({probability[1] * 100:.2f}%)")
    print(f"  {'=' * 50}")

    print("\n  Sample Features (first 5):")
    for i, (col, val) in enumerate(sample_data.items()):
        if i >= 5:
            print(f"      ... and {len(sample_data) - 5} more features")
            break
        print(f"      {col}: {val:.4f}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    """Orchestrate the full fraud detection pipeline end-to-end."""
    print("=" * 70)
    print("  CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING")
    print("=" * 70)

    # Setup
    project_root, output_dir = setup_directories()
    configure_plot_style()

    # Load data
    df = load_dataset(project_root)

    # EDA
    display_basic_info(df)
    df = check_missing_values(df)
    df = check_duplicates(df)
    analyze_class_distribution(df)

    # Visualization
    print("\n" + "=" * 70)
    print("  DATA VISUALIZATION")
    print("=" * 70)
    plot_class_distribution(df, output_dir)
    plot_transaction_amount_distribution(df, output_dir)
    plot_time_distribution(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_feature_correlation_with_class(df, output_dir)
    plot_feature_histograms(df, output_dir)
    print("\n[OK] All visualizations saved.")

    # Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Training
    print("\n" + "=" * 70)
    print("  MODEL TRAINING")
    print("=" * 70)
    model_lr = train_logistic_regression(X_train, y_train)
    model_dt = train_decision_tree(X_train, y_train)
    model_rf = train_random_forest(X_train, y_train)
    print("\n[OK] All models trained.")

    # Evaluation
    print("\n" + "=" * 70)
    print("  MODEL EVALUATION")
    print("=" * 70)
    results = [
        evaluate_model(model_lr, X_test, y_test, "Logistic Regression", output_dir),
        evaluate_model(model_dt, X_test, y_test, "Decision Tree", output_dir),
        evaluate_model(model_rf, X_test, y_test, "Random Forest", output_dir),
    ]

    # Comparison
    best_idx, results_df = compare_models(results, output_dir)

    # Prediction
    models = [model_lr, model_dt, model_rf]
    predict_new_transaction(
        models[best_idx],
        results_df.loc[best_idx, "Model"],
        X_train.columns.tolist(),
    )

    # Summary
    print("\n" + "=" * 70)
    print("  PROJECT EXECUTION COMPLETE")
    print("=" * 70)
    print(f"\n  Charts saved to: {output_dir}")
    print(f"  Best Model:      {results_df.loc[best_idx, 'Model']}")
    print(f"  Models trained:  3")
    print(f"  Visualizations:  8")
    print("\n  Thank you for running the Credit Card Fraud Detection Project!")
    print("=" * 70)


if __name__ == "__main__":
    main()
