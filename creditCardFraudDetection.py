# Credit Card Fraud Detection using Machine Learning

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("creditcard.csv")

print("Dataset Shape:", data.shape)
print(data.head())

# -----------------------------
# 2. Feature & Target Split
# -----------------------------
X = data.drop("Class", axis=1)
y = data["Class"]

# -----------------------------
# 3. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])
X["Time"] = scaler.fit_transform(X[["Time"]])

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. Handle Class Imbalance (SMOTE)
# -----------------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_resampled.value_counts())

# -----------------------------
# 6. Model Training
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# -----------------------------
# 7. Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 8. Evaluation
# -----------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
