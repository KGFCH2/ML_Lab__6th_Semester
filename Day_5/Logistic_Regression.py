# Logistic Regression Implementation

# ---------------------------------------------------------------------------------
# Package Required: pandas, train-test-split, logistic regression, metric, numpy
# Step 1: Read the Data
# Step 2: Split the data into training and testing 
# Step 3: Scale the data 
# Step 4: Train the Logistic Regression model
# Step 5: Model prediction
# Step 6: Performance evaluation
# ---------------------------------------------------------------------------------


# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score


# Step 2: Load Dataset
data = pd.read_csv("Salary_Data.csv")

print("Dataset Preview:")
print(data.head())


# Step 3: Convert Salary into Classification (Binary)
# 1 = High Salary, 0 = Low Salary

data["HighSalary"] = (data["Salary"] >= 65000).astype(int)

print("\nDataset After Creating Target Column:")
print(data.head())


# Step 4: Define Features (X) and Target (y)

X = data[["YearsExperience"]].values
y = data["HighSalary"].values


# Step 5: Split Dataset into Training and Testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 6: Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 7: Train Logistic Regression Model

model = LogisticRegression()

model.fit(X_train, y_train)


# Step 8: Model Prediction

y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)


# Step 9: Performance Evaluation

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Step 10: ROC Curve Implementation

# Get probability predictions
y_prob = model.predict_proba(X_test)[:,1]

# Compute ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Compute AUC score
auc_score = roc_auc_score(y_test, y_prob)

print("\nAUC Score:", auc_score)


# Step 11: Plot ROC Curve

plt.figure()

plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.2f)" % auc_score)

plt.plot([0,1], [0,1], linestyle='--')   # Random classifier line

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve for Logistic Regression")

plt.legend()

plt.savefig('roc_curve.png')
plt.show()