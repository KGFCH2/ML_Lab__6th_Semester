# 📖 ML Lab Instructions & Directory Map

**Course:** Introduction to Machine Learning | **Author:** Babin Bid

This document explains the organization of the Machine Learning lab and how to utilize the provided scripts and datasets.

---

## 📂 Directory Structure

### 📉 [Day_1/](./Day_1/)

- 📌 **Purpose:** Simple Linear Regression implementation from scratch.
- 📋 **Contents:**
  - `Day_1_Linear_Regression.py`: Core Python implementation.
  - `test_data1.csv`: Dataset used for training.
  - `README.md`: Documentation, formulas, and visual results.

### 📉 [Day_2/](./Day_2/)

- 📌 **Purpose:** Gradient Descent algorithm for regression optimization.
- 📋 **Contents:**
  - `Day_2_LR_Gradient_Descent.py`: Weight optimization script.
  - `test_data1.csv`: Dataset for optimization testing.
  - `README.md`: Iteration walkthrough and console output.

### 📉 [Day_3/](./Day_3/)

- 📌 **Purpose:** L1 and L2 Regularization techniques (Lasso & Ridge).
- 📋 **Contents:**
  - `Day_3_Lasso.py`: Lasso regression for feature selection.
  - `Day_3_Ridge.py`: Ridge regression for multicollinearity.
  - `titanic.csv`: Real-world dataset used for training.
  - `README.md`: Comparison of L1 vs L2 results.

### 📉 [Day_4/](./Day_4/)

- 📌 **Purpose:** Manual implementation of the K-Nearest Neighbors classifier.
- 📋 **Contents:**
  - `KNN.py`: Manual implementation without libraries.
  - `Iris.csv`: Standard classification dataset.
  - `README.md`: Accuracy analysis and K-value plots.

### 📉 [Day_5/](./Day_5/)

- 📌 **Purpose:** Logistic Regression for binary classification.
- 📋 **Contents:**
  - `Logistic_Regression.py`: Manual implementation of logistic regression.
  - `Salary_Data.csv`: Dataset for salary prediction.
  - `README.md`: Sigmoid function explanation and accuracy metrics.

### 📊 [Day_6/](./Day_6/)

- 📌 **Purpose:** Unsupervised clustering algorithms (K-Means & DBSCAN).
- 📋 **Contents:**
  - `K-Means_Clustering.py`: K-Means implementation with elbow method.
  - `DBScan_Clustering.py`: Density-based clustering algorithm.
  - `titanic_toy.csv`: Titanic dataset for clustering analysis.
  - `README.md`: Cluster visualization and algorithm comparison.

### 🌳 [Day_7/](./Day_7/)

- 📌 **Purpose:** Decision Tree Regression for mixed and numerical data.
- 📋 **Contents:**
  - `Decision_Tree.py`: Decision Tree Regressor implementation.
  - `Salary_Data.csv`: Dataset for salary prediction.
  - `README.md`: CART algorithm explanation and MSE comparison.

### 🌲 [Day_8/](./Day_8/)

- 📌 **Purpose:** Ensemble Learning methods (AdaBoost & Random Forest).
- 📋 **Contents:**
  - `AdaBoost.py`: AdaBoost Classifier implementation.
  - `RandomForest.py`: Random Forest Classifier implementation.
  - `titanic_toy.csv`: Dataset for survival prediction.
  - `README.md`: Ensemble methods introduction and evaluation.

---

## 🛠️ Usage Instructions

1. **Dependency Check:** Install required libraries using:

    ```bash
    pip install pandas numpy matplotlib scikit-learn
    ```

2. **Running Scripts:** Navigate to the specific day's folder and execute the `.py` file:

    ```bash
    python <filename>.py
    ```

3. **Visualizations:** Most scripts generate plots using `matplotlib`. Ensure you have a windowing system or IDE support for displaying plots.

---

## ✅ Execution Checklist

- ✔️ Ensure Python 3.8+ is installed on your system
- ✔️ Install all required dependencies via pip
- ✔️ Navigate to the respective Day folder
- ✔️ Run the Python script: `python script_name.py`
- ✔️ View output in console and generated visualizations

---

## 📚 Learning Outcomes

After completing all labs, you will understand:

- 🎓 Supervised vs Unsupervised Learning
- 🎓 Regression and Classification algorithms
- 🎓 Hyperparameter tuning and model evaluation
- 🎓 Data preprocessing and feature engineering
- 🎓 Mathematical foundations of ML algorithms

---

Created with Dedication by **Babin Bid** | Adamas University
