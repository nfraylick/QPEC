import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X_q = np.load("q_features.npy")
y = np.load("labels.npy")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_q)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis')
plt.title("PCA of Quantum Features")
plt.savefig("pca_quantum.png")
plt.show()