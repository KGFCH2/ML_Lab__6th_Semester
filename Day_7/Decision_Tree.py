# Decision Tree: Mixed Data + Numerical Data (Single Code, Separate Outputs)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("Salary_Data.csv")

print("\n====== ORIGINAL DATASET ======\n")
print(df.head())


# ============================================================
# 🔹 CASE 1: Numerical + Categorical Data (CART)
# ============================================================

df_mixed = df.copy()

# Encode categorical columns (though none in this dataset)
le = LabelEncoder()
for col in df_mixed.columns:
    if df_mixed[col].dtype == 'object':
        df_mixed[col] = le.fit_transform(df_mixed[col])

# Features and Target
X1 = df_mixed.drop("Salary", axis=1)   # change target if needed
y1 = df_mixed["Salary"]

# Split
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

# Model
model1 = DecisionTreeRegressor(criterion="squared_error")
model1.fit(X1_train, y1_train)

# Prediction
y1_pred = model1.predict(X1_test)

# Output
print("\n====== CASE 1: MIXED DATA (CATEGORICAL + NUMERICAL) ======\n")
print("Predictions:", y1_pred)
print("MSE:", mean_squared_error(y1_test, y1_pred))


# ============================================================
# 🔹 CASE 2: Only Numerical Data
# ============================================================

df_numeric = df.select_dtypes(include=['int64', 'float64'])

# Features and Target
X2 = df_numeric.drop("Salary", axis=1)   # change target if needed
y2 = df_numeric["Salary"]

# Split
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

# Model
model2 = DecisionTreeRegressor(criterion="friedman_mse")
model2.fit(X2_train, y2_train)

# Prediction
y2_pred = model2.predict(X2_test)

# Output
print("\n====== CASE 2: ONLY NUMERICAL DATA ======\n")
print("Predictions:", y2_pred)
print("MSE:", mean_squared_error(y2_test, y2_pred))


# ============================================================
# 🔥 FINAL COMPARISON
# ============================================================

print("\n====== FINAL COMPARISON ======\n")
print("Mixed Data MSE     :", mean_squared_error(y1_test, y1_pred))
print("Numerical Data MSE :", mean_squared_error(y2_test, y2_pred))