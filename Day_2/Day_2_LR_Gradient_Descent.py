# Linear Regression using Gradient Descent

import pandas as pd
import numpy as np

# -----------------------------------
# 1. Load data
# -----------------------------------
data = pd.read_csv("D:\\Vs Code\\ML_LAB\\Day_2\\test_data1.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

m = len(y)

# -----------------------------------
# 2. Feature normalization
# -----------------------------------
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# -----------------------------------
# 3. Add bias term
# -----------------------------------
X = np.c_[np.ones((m, 1)), X]

# -----------------------------------
# 4. Initialize coefficients
# -----------------------------------
w = np.zeros((X.shape[1], 1))
alpha = 0.01

# ✅ N given by user
N = int(input("Enter number of iterations (N): "))

# -----------------------------------
# 5. Gradient Descent
# -----------------------------------
for i in range(N):

    # (a) y_cap = Xw
    y_cap = X @ w

    # (b) error = y - y_cap
    error = y - y_cap

    # (c) gradient
    gradients = (-2 / m) * (X.T @ error)

    # (d) update weights
    w = w - alpha * gradients

    # Print first few iterations (for lab clarity)
    if i < 5:
        print(f"\nIteration {i+1}")
        print("Gradient:")
        for j in range(len(gradients)):
            print(f"∂J/∂w{j} = {gradients[j][0]}")

        print("Updated Weights:")
        for j in range(len(w)):
            print(f"w{j} = {w[j][0]}")

# -----------------------------------
# Final coefficients
# -----------------------------------
print("\nFinal Coefficients after Gradient Descent:")
for i, coef in enumerate(w):
    print(f"w{i} = {coef[0]}")
