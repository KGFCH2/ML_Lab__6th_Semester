# 🌲 ML Day 8: Ensemble Learning (AdaBoost & Random Forest)

**Introduction to Machine Learning Lab (CSE12207)** | **Babin Bid**

This session focuses on Ensemble Learning techniques, specifically **AdaBoost** (Boosting) and **Random Forest** (Bagging). We use the `titanic_toy.csv` dataset to implement these classifiers and evaluate their performance in predicting survival.

---

## ❓ Question 1

**Implement AdaBoost and Random Forest classifiers using the Titanic dataset.**

Use the `titanic_toy.csv` dataset to:

- Preprocess data by handling missing values and encoding categorical features.
- Build an AdaBoost model to improve weak learners sequentially.
- Build a Random Forest model using multiple decision trees to improve overall robustness.
- Output the accuracy score for both models.

---

### ✅ Answer (Python Implementation)

#### 🔹 1. AdaBoost Implementation

📜 **[View Full Source Code](./AdaBoost.py)**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("titanic_toy.csv")

# Handle missing values
df = df.dropna()

# Convert categorical columns to numeric
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Split features & target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AdaBoost model
model = AdaBoostClassifier(n_estimators=50, learning_rate=1)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred))
```

#### 🔹 2. Random Forest Implementation

📜 **[View Full Source Code](./RandomForest.py)**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("titanic_toy.csv")

# Handle missing values
df = df.dropna()

# Convert categorical columns to numeric
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Split features & target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 🔍 Key Concepts

### Ensemble Learning

- 🚀 **Boosting (AdaBoost)**: Focuses on training weak learners sequentially, where each new learner corrects errors made by previous ones.
- 🌲 **Bagging (Random Forest)**: Combines several decision trees on different subsets of the data to reduce variance and avoid overfitting.

### Evaluation Metrics

- 🎯 **Accuracy**: The ratio of correctly predicted instances to the total instances.
- 📊 **Robustness**: Random Forest is generally less prone to overfitting compared to individual decision trees.

---

## 📊 Sample Output

When executed, the scripts produce the following typical results:

```text
AdaBoost Accuracy: 0.7185185185185186
Random Forest Accuracy: 0.6814814814814815
```

---

Created with Dedication by **Babin Bid** | Adamas University
