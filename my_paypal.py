import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


def load_data(path: str = "creditcard.csv") -> pd.DataFrame:
    """
    Load the credit card transaction dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file containing the dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """
    df = pd.read_csv(path)
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Perform basic exploratory analysis on the dataset and save simple plots.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to explore.
    """
    print("=== Dataset shape ===")
    print(df.shape)

    print("\n=== Head (first 5 rows) ===")
    print(df.head())

    print("\n=== Missing values per column ===")
    print(df.isnull().sum())

    print("\n=== Class distribution (0 = non-fraud, 1 = fraud) ===")
    print(df["Class"].value_counts())

    fraud_ratio = df["Class"].mean() * 100
    print(f"\nFraud ratio: {fraud_ratio:.4f}%")

    # Simple bar plot for class distribution
    df["Class"].value_counts().plot(
        kind="bar",
        title="Class distribution (0 = non-fraud, 1 = fraud)"
    )
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.close()


def prepare_data(df: pd.DataFrame):
    """
    Prepare the data for modeling:
    - Split into train and test sets.
    - Scale 'Time' and 'Amount' features.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to prepare.

    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Test labels.
    """
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Work on copies to avoid SettingWithCopyWarning
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()

    # Scale only 'Time' and 'Amount'; PCA components are already scaled
    for col in ["Time", "Amount"]:
        X_train.loc[:, col] = scaler.fit_transform(X_train[[col]])
        X_test.loc[:, col] = scaler.transform(X_test[[col]])

    return X_train, X_test, y_train, y_test


def build_model() -> LogisticRegression:
    """
    Build a logistic regression model with class weighting
    to handle class imbalance.

    Returns
    -------
    LogisticRegression
        Configured logistic regression model.
    """
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1,
    )
    return model


def train_model(model: LogisticRegression,
                X_train: pd.DataFrame,
                y_train: pd.Series) -> LogisticRegression:
    """
    Train the model on the training data.

    Parameters
    ----------
    model : LogisticRegression
        The model to train.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.

    Returns
    -------
    LogisticRegression
        Trained model.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict:
    """
    Evaluate the model using AUPRC, classification report,
    confusion matrix, and save the Precision-Recall curve.

    Parameters
    ----------
    model : LogisticRegression
        Trained model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.

    Returns
    -------
    dict
        Dictionary containing evaluation results.
    """
    # Predicted probabilities for the positive class (fraud)
    y_scores = model.predict_proba(X_test)[:, 1]

    # Area Under the Precision-Recall Curve (AUPRC)
    auprc = average_precision_score(y_test, y_scores)
    print("\n=== AUPRC (Area Under Precision-Recall Curve) ===")
    print(f"AUPRC: {auprc:.6f}")

    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png")
    plt.close()

    # Threshold selection (starting with 0.5; can be tuned)
    threshold = 0.5
    y_pred = (y_scores >= threshold).astype(int)

    print(f"\n=== Classification report at threshold = {threshold} ===")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return {
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "thresholds": thresholds,
        "y_scores": y_scores,
        "y_pred": y_pred,
        "confusion_matrix": cm,
    }


def main() -> None:
    """
    End-to-end pipeline:
    - Load data
    - Explore data
    - Prepare data
    - Build model
    - Train model
    - Evaluate model
    """
    print("Starting FriendPay fraud detection project...\n")

    # 1. Load data
    df = load_data("creditcard.csv")

    # 2. Explore data
    explore_data(df)

    # 3. Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # 4. Build model
    model = build_model()

    # 5. Train model
    model = train_model(model, X_train, y_train)

    # 6. Evaluate model
    results = evaluate_model(model, X_test, y_test)

    print("\nPipeline completed successfully.")
    print("Saved plots:")
    print("- class_distribution.png")
    print("- precision_recall_curve.png")
    print("\nYou can use these plots in your presentation slides.")


if __name__ == "__main__":
    main()
