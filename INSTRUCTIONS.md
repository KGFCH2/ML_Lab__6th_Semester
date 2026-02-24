# 📖 ML Lab Instructions & Directory Map
**Course:** Introduction to Machine Learning | **Author:** Babin Bid

This document explains the organization of the Machine Learning lab and how to utilize the provided scripts and datasets.

---

## 📂 Directory Structure

### 📉 [Day_1/](./Day_1/)
*   **Purpose:** Simple Linear Regression implementation from scratch.
*   **Contents:**
    *   `Day_1_Linear_Regression.py`: Core Python implementation.
    *   `test_data1.csv`: Dataset used for training.
    *   `README.md`: Documentation, formulas, and visual results.

### 📉 [Day_2/](./Day_2/)
*   **Purpose:** Gradient Descent algorithm for regression optimization.
*   **Contents:**
    *   `Day_2_LR_Gradient_Descent.py`: Weight optimization script.
    *   `test_data1.csv`: Dataset for optimization testing.
    *   `README.md`: Iteration walkthrough and console output.

### 📉 [Day_3/](./Day_3/)
*   **Purpose:** L1 and L2 Regularization techniques (Lasso & Ridge).
*   **Contents:**
    *   `Day_3_Lasso.py`: Lasso regression for feature selection.
    *   `Day_3_Ridge.py`: Ridge regression for multicollinearity.
    *   `titanic.csv`: Real-world dataset used for training.
    *   `README.md`: Comparison of L1 vs L2 results.

### 📉 [Day_4/](./Day_4/)
*   **Purpose:** Manual implementation of the K-Nearest Neighbors classifier.
*   **Contents:**
    *   `KNN.py`: Manual implementation without libraries.
    *   `Iris.csv`: Standard classification dataset.
    *   `README.md`: Accuracy analysis and K-value plots.

---

## 🛠️ Usage Instructions

1.  **Dependency Check:** Install required libraries using:
    ```bash
    pip install pandas numpy matplotlib scikit-learn
    ```
2.  **Running Scripts:** Navigate to the specific day's folder and execute the `.py` file:
    ```bash
    python <filename>.py
    ```
3.  **Visualizations:** Most scripts generate plots using `matplotlib`. Ensure you have a windowing system or IDE support for displaying plots.

---
<p align="center">Created with ❤️ by <b>Babin Bid</b> | Adamas University</p>
