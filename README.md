# 💳 Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using machine learning classification techniques. Due to the highly imbalanced nature of fraud data, special preprocessing and evaluation methods are applied to ensure reliable detection.

---

## 📌 Project Objective

The main goal of this project is to:
- Identify fraudulent credit card transactions
- Handle class imbalance in transaction data
- Train a machine learning classification model
- Evaluate performance using precision, recall, and F1-score

---

## 📊 Dataset Description

The project uses the **Kaggle Credit Card Fraud Detection dataset (`creditcard.csv`)**.

- Each row represents a transaction
- Most features are anonymized (`V1` to `V28`)
- `Time` and `Amount` are numerical features
- Target column:
  - `0` → Genuine transaction  
  - `1` → Fraudulent transaction

⚠️ The dataset is **highly imbalanced**, with very few fraudulent transactions.

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

## 🧠 Machine Learning Approach

### 1. Data Preprocessing
- Checked for missing values
- Feature scaling using `StandardScaler`
- Normalized `Time` and `Amount` columns

### 2. Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)**
- Balanced fraudulent and genuine transaction samples

### 3. Model Used
- **Logistic Regression** (Classification Algorithm)

### 4. Train-Test Split
- 80% Training Data
- 20% Testing Data

---

## 📈 Model Evaluation Metrics

Due to the critical nature of fraud detection, the following metrics are used:

- **Precision** – Correctly identified frauds among predicted frauds
- **Recall** – Correctly detected frauds from actual frauds
- **F1-Score** – Balance between precision and recall
- **Confusion Matrix**

Accuracy is not preferred because of class imbalance.

---

## ▶️ How to Run the Project

### 1. Install Required Libraries
```bash
pip install pandas numpy scikit-learn imbalanced-learn
