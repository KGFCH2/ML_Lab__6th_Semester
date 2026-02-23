# 📉 ML Day 2: Linear Regression using Gradient Descent
**Introduction to Machine Learning Lab (CSE12207)** | **Babin Bid**

In this session, we optimized the Linear Regression model using the **Gradient Descent** algorithm, involving iterative updates of weights to minimize the cost function.

---

### ❓ Question 1
**Implement Linear Regression using Gradient Descent optimization.**

The weights are updated iteratively using the formula:
$w = w - \alpha \times \text{gradient}$

---

### ✅ Answer (Python Implementation)
📜 **[View Full Source Code](./Day_2_LR_Gradient_Descent.py)**

```python
import pandas as pd
import numpy as np

# 1. Load and Prepare Data
data = pd.read_csv("test_data1.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)
m = len(y)

# 2. Feature Normalization & Bias Term
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
X = np.c_[np.ones((m, 1)), X]

# 3. Initialize Parameters
w = np.zeros((X.shape[1], 1))
alpha = 0.01
N = int(input("Enter number of iterations (N): "))

# 4. Gradient Descent Loop
for i in range(N):
    y_cap = X @ w
    error = y - y_cap
    gradients = (-2 / m) * (X.T @ error)
    w = w - alpha * gradients

# 5. Output Final Coefficients
print("\nFinal Coefficients:")
for i, coef in enumerate(w):
    print(f"w{i} = {coef[0]}")
```

---

### 🔍 Expected Output (Text)

**Console Output Example (N=1000):**
```text
Iteration 1
Gradient:
∂J/∂w0 = -25.746
∂J/∂w1 = -18.293

...

Final Coefficients after Gradient Descent:
w0 = 12.873
w1 = 4.312
```

---
<p align="center">Created with ❤️ by <b>Babin Bid</b> | Adamas University</p>
