# K-Nearest Neighbors (KNN) Algorithm Implementation 

# ------------------------------------------------------------
# KNN Classifier: 
# Step 1: Split the data into 80% training and 20% testing 
# Step 2: Select the values of K (take odd values) 
# Step 3: Calculate distance using Euclidean distance 
# Step 4: Sort the distances in ascending order 
# Step 5: Pick first K samples from the sorted list 
# Step 6: Count class labels among K neighbors 
# Step 7: Assign the class that most frequently occurs.
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Step 1: Load and Clean Dataset
# --------------------------------------------------
data = pd.read_csv("KNNAlgorithmDataset.csv")

# Drop unnecessary columns
data = data.drop(columns=["id", "Unnamed: 32"])

# Convert diagnosis column to numeric
# M = 1 (Malignant), B = 0 (Benign)
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# Separate features and target
X = data.drop("diagnosis", axis=1).values
y = data["diagnosis"].values

# --------------------------------------------------
# Step 2: Split Data (80% Train, 20% Test)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Step 3: Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Step 4: Euclidean Distance Function
# --------------------------------------------------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# --------------------------------------------------
# Step 5–7: KNN Prediction Function
# --------------------------------------------------
def predict(X_train, y_train, X_test, K):
    predictions = []

    for test_point in X_test:
        distances = []

        # Calculate distance
        for i in range(len(X_train)):
            distance = euclidean_distance(test_point, X_train[i])
            distances.append((distance, y_train[i]))

        # Sort distances
        distances.sort(key=lambda x: x[0])

        # Pick first K neighbors
        k_neighbors = distances[:K]

        # Count class labels
        labels = [neighbor[1] for neighbor in k_neighbors]
        most_common = Counter(labels).most_common(1)

        # Assign majority class
        predictions.append(most_common[0][0])

    return np.array(predictions)

# --------------------------------------------------
# Step 8: Choose K and Evaluate
# --------------------------------------------------
K = 3
y_pred = predict(X_train, y_train, X_test, K)

accuracy = np.sum(y_pred == y_test) / len(y_test)

print("K Value:", K)
print("Accuracy:", accuracy)

# --------------------------------------------------
# Optional: Plot K vs Accuracy (to find best K)
# --------------------------------------------------
k_values = [1,3,5,7,9,11,13]
accuracies = []

for k in k_values:
    y_pred = predict(X_train, y_train, X_test, k)
    acc = np.sum(y_pred == y_test) / len(y_test)
    accuracies.append(acc)

plt.plot(k_values, accuracies)
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")
plt.savefig('k_vs_accuracy.png')
plt.show()
