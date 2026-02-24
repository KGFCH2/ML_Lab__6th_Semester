# Multiple Linear Regression using Ridge Regularization 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("titanic_toy.csv")

print("Shape:", df.shape)
print(df.head())

# -------------------------------
# 2. Split Features and Target
# -------------------------------
X = df.iloc[:, :-1].values   # all columns except last
y = df.iloc[:, -1].values    # last column

# -------------------------------
# 3. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Handle Missing Values
# -------------------------------
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# -------------------------------
# 5. Standardize Features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6. Ridge Model
# -------------------------------
ridge = Ridge(alpha=1.0)  # you can tune alpha
ridge.fit(X_train_scaled, y_train)

# -------------------------------
# 7. Predictions
# -------------------------------
y_pred = ridge.predict(X_test_scaled)

# -------------------------------
# 8. Model Parameters
# -------------------------------
print("Intercept:", ridge.intercept_)
print("Coefficients:", ridge.coef_)

# -------------------------------
# 9. Evaluation
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# -------------------------------
# 10. Plot Actual vs Predicted
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ridge Regression: Actual vs Predicted")
plt.grid(True)
plt.savefig('ridge_actual_vs_predicted.png')
plt.show()
