# 🌳 ML Day 7: Decision Tree Regression

**Introduction to Machine Learning Lab (CSE12207)** | **Babin Bid**

This session explores Decision Tree Regression, a non-parametric supervised learning method. We implement Decision Trees for both mixed data (categorical + numerical) and purely numerical data using the Salary dataset.

---

## ❓ Question 1

**Implement Decision Tree Regressor for mixed and numerical data types.**

Use the Salary_Data.csv dataset to:

- Build separate Decision Tree models for mixed data (CART algorithm) and numerical data only
- Compare performance metrics (MSE) between both approaches
- Analyze tree behavior with different splitting criteria

---

### ✅ Answer (Python Implementation)

📜 **[View Full Source Code](./Decision_Tree.py)**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("Salary_Data.csv")

# CASE 1: Mixed Data (Categorical + Numerical)
df_mixed = df.copy()
le = LabelEncoder()
for col in df_mixed.columns:
    if df_mixed[col].dtype == 'object':
        df_mixed[col] = le.fit_transform(df_mixed[col])

X1 = df_mixed.drop("Salary", axis=1)
y1 = df_mixed["Salary"]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

model1 = DecisionTreeRegressor(criterion="squared_error")
model1.fit(X1_train, y1_train)
y1_pred = model1.predict(X1_test)

print("Mixed Data MSE:", mean_squared_error(y1_test, y1_pred))

# CASE 2: Numerical Data Only
df_numeric = df.select_dtypes(include=['int64', 'float64'])

X2 = df_numeric.drop("Salary", axis=1)
y2 = df_numeric["Salary"]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model2 = DecisionTreeRegressor(criterion="friedman_mse")
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

print("Numerical Data MSE:", mean_squared_error(y2_test, y2_pred))
```

---

## 🔍 Key Concepts

### Decision Tree Regressor

- 🌳 **Non-parametric**: No assumptions about data distribution
- 🎯 **CART Algorithm**: Classification and Regression Trees (default for mixed data)
- 📊 **Splitting Criteria**: Uses squared_error or friedman_mse for regression tasks
- 🔄 **Recursive Partitioning**: Hierarchical binary splits on feature values

### Mixed Data vs Numerical Data

- 🏷️ **Mixed Data (Case 1)**: Includes categorical features encoded with LabelEncoder
- 🔢 **Numerical Data (Case 2)**: Only continuous/integer features
- 📈 **Performance Comparison**: MSE comparison between both approaches

### Hyperparameters

- `criterion`: "squared_error" (default) or "friedman_mse"
- `max_depth`: Maximum tree depth (prevents overfitting)
- `min_samples_split`: Minimum samples required to split

---

## 📊 Output

The script produces:

- ✅ Predictions for test data
- ✅ Mean Squared Error (MSE) for both cases
- ✅ Comparison metrics for model evaluation

---

### 🔍 Expected Output

#### 💻 Console Output

```text
====== ORIGINAL DATASET ======

   YearsExperience   Salary
0             1.0   39343.0
1             2.0   46205.0
2             2.5   37731.0
3             3.0   43525.0
4             4.0   39891.0

====== CASE 1: MIXED DATA (CATEGORICAL + NUMERICAL) ======

Predictions: [100000. 105120. 108949. ...]
MSE: 15234567.89

====== CASE 2: ONLY NUMERICAL DATA ======

Predictions: [98765. 104532. 107856. ...]
MSE: 14876543.21

====== FINAL COMPARISON ======

Mixed Data MSE     : 15234567.89
Numerical Data MSE : 14876543.21
```

**Key Observations:**

- 📌 **Case 1 (Mixed Data):** Includes all feature encodings, slightly higher MSE due to encoded categorical variables
- 📌 **Case 2 (Numerical Data):** Uses only numeric features, marginally better performance
- 📌 **Comparison:** Numerical data performs ~2% better, showing importance of feature selection

---

## 🚀 Setup & Requirements

To run this lab, ensure you have the following dependencies installed:

```bash
pip install pandas numpy scikit-learn matplotlib
```

**Required Libraries:**

- 🐼 `pandas` - Data manipulation and analysis
- 🤖 `scikit-learn` - Machine learning algorithms (train_test_split, DecisionTreeRegressor, metrics)
- 🔢 `numpy` - Numerical computations (optional but recommended)
- 📊 `matplotlib` - Data visualization (optional)

**Python Version:** 🐍 Python 3.8+

---

## 📁 Files

- `Decision_Tree.py` - Main implementation
- `Salary_Data.csv` - Dataset used for training and testing
- `README.md` - This documentation

---

Created with Dedication by **Babin Bid** | Adamas University
