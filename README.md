# Welcome to My Paypal
***

## Task
FriendPay, a competitor to PayPal, is experiencing financial losses due to undetected fraudulent credit card transactions.
Your task is to build a fraud detection model capable of identifying fraudulent transactions with high recall while maintaining business‑friendly precision.

The challenge lies in the dataset’s extreme imbalance:

284,807 total transactions

Only 492 fraud cases (≈0.17%)

Traditional accuracy metrics are misleading, so the project focuses on AUPRC (Area Under the Precision‑Recall Curve).

## Description
This project implements a complete machine learning pipeline:

Data Exploration
Inspected dataset structure, missing values, and class imbalance

Visualized fraud vs non‑fraud distribution

Data Preparation
Stratified train/test split

Standardized Time and Amount

PCA features left unchanged

Modeling
Logistic Regression with class_weight="balanced"

Handles severe class imbalance without oversampling

Evaluation
AUPRC

Precision‑Recall Curve

Classification Report

Confusion Matrix

Results
AUPRC: 0.7189

Recall (fraud): 0.92

Precision (fraud): 0.06

Accuracy: 0.98 (not meaningful due to imbalance)

The model successfully detects 92% of fraudulent transactions, significantly reducing financial losses for FriendPay.

## Installation
Ensure the dataset file creditcard.csv is placed in the project directory.

pip install scikit-learn pandas numpy matplotlib

## Usage
Run the main pipeline:

Code
python my_paypal.py
The script will:

Load and explore the dataset

Prepare training and test data

Train the fraud detection model

Evaluate performance

Save plots for presentation:

class_distribution.png

precision_recall_curve.png
```
./my_project argument1 argument2
```

### The Core Team


<span><i>Made at <a href='https://qwasar.io'>Qwasar SV -- Software Engineering School</a></i></span>
<span><img alt='Qwasar SV -- Software Engineering School's Logo' src='https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png' width='20px' /></span>
