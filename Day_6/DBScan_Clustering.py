import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv('titanic_toy.csv')

# Preprocessing
df = df.select_dtypes(include=[np.number]).dropna()

# Scaling
X = StandardScaler().fit_transform(df)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise = list(labels).count(-1)

print("DBSCAN Results:")
print("Clusters:", n_clusters)
print("Noise Points:", noise)

if n_clusters > 1:
    print("Silhouette Score:", round(silhouette_score(X, labels), 4))
else:
    print("Silhouette Score: Not applicable")

# PCA for graph
X_pca = PCA(n_components=2).fit_transform(X)

# Plot
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
plt.title("DBSCAN Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("dbscan_clusters.png")
plt.show()