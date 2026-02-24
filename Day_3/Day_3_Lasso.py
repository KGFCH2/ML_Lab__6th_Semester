# Lasso multiple linear regression 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("titanic_toy.csv")

print(df.head())
print(df.shape)

# --- Impute missing values for numeric columns ---
imputer = SimpleImputer(strategy='mean')
# imputer works on numpy arrays; apply to all columns (numeric expected)
df[df.columns] = imputer.fit_transform(df)

# Features (all columns except last)
feature_names = list(df.columns[:-1])
X = df.iloc[:, :-1].values


# Target (last column)
y = df.iloc[:, -1].values

# Train-test split (optional but recommended)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features (VERY IMPORTANT for LASSO)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LASSO model
lasso = Lasso(alpha=0.1)  # tune alpha
lasso.fit(X_train_scaled, y_train)

# Predictions
y_pred = lasso.predict(X_test_scaled)

# Coefficients
print("Intercept:", lasso.intercept_)
print("Coefficients:", lasso.coef_)

# Which features were selected
selected_features = np.where(lasso.coef_ != 0)[0]
print("Selected feature indices:", selected_features)
selected_feature_names = [feature_names[i] for i in selected_features]
print("Selected features:", selected_feature_names)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

# Plot: Predicted vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
minv = min(y_test.min(), y_pred.min())
maxv = max(y_test.max(), y_pred.max())
plt.plot([minv, maxv], [minv, maxv], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Lasso: Predicted vs Actual')
plt.grid(True)
plt.tight_layout()
plt.savefig('lasso_predicted_vs_actual.png')
plt.show()
