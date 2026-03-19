# ================================
# K-MEANS CLUSTERING
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv('titanic_toy.csv')

# Preprocessing
df = df.select_dtypes(include=[np.number])  # keep only numeric
df = df.dropna()

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ================================
# K-Means for K = 2, 3, 4
# ================================
k_values = [2, 3, 4]

print("=== K-MEANS RESULTS ===")

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, labels)
    print(f"K = {k}, Silhouette Score = {score:.4f}")

# ================================
# Elbow Method
# ================================
inertia = []
K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.savefig("kmeans_elbow.png")
plt.show()