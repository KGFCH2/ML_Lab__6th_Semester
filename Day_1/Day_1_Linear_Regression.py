# Linear Regression

import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
data = pd.read_csv("test_data1.csv")

X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values

# -----------------------------
# Step 2: Number of observations
# -----------------------------
n = len(X)
print("Number of observations (n):", n)

# -----------------------------
# Step 3: Mean of X and Y
# -----------------------------
mean_x = sum(X) / n
mean_y = sum(Y) / n

print("Mean of X:", mean_x)
print("Mean of Y:", mean_y)

# -----------------------------
# Step 4: Cross-deviation
# SSxy = ΣXY - n * meanX * meanY
# SSxx = ΣX² - n * meanX²
# -----------------------------
SSxy = sum(X * Y) - n * mean_x * mean_y
SSxx = sum(X * X) - n * mean_x * mean_x

print("SSxy:", SSxy)
print("SSxx:", SSxx)

# -----------------------------
# Step 5: Regression Coefficients
# β1 = SSxy / SSxx
# β0 = meanY - β1 * meanX
# -----------------------------
β1 = SSxy / SSxx
β0 = mean_y - β1 * mean_x

print("Slope (β1):", β1)
print("Intercept (β0):", β0)

# -----------------------------
# Step 6: Plotting the data
# -----------------------------
plt.scatter(X, Y, color="blue", label="Data Points")

# Predictor values
ypred = β0 + β1 * X

# Regression line
plt.plot(X, ypred, color="red", label="Regression Line")

# Labels and title
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using Python")
plt.legend()
plt.savefig('linear_regression.png')
plt.show()
