# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import mnist

# # Load MNIST dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X = X_train.reshape((X_train.shape[0], -1))

# # Center the data
# X_mean = np.mean(X, axis=0)
# X_centered = X - X_mean

# # Compute covariance matrix
# cov_matrix = np.cov(X_centered.T)

# # Compute eigenvalues and eigenvectors
# eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# # Sort eigenvalues and eigenvectors in descending order
# idx = eig_vals.argsort()[::-1]
# eig_vals = eig_vals[idx]
# eig_vecs = eig_vecs[:, idx]

# # Select top k eigenvectors
# k = 154
# top_k_eig_vecs = eig_vecs[:, :k]

# # Encode data
# X_pca = np.dot(X_centered, top_k_eig_vecs)

# # Reconstruct images
# X_reconstructed = np.dot(X_pca, top_k_eig_vecs.T) + X_mean

# # Compute explained variance ratio
# explained_variance_ratio = eig_vals / np.sum(eig_vals)

# # Plot explained variance ratio
# plt.plot(np.cumsum(explained_variance_ratio))
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance Ratio')
# plt.show()
import numpy as np

# Load MNIST dataset
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X = X_train.reshape((X_train.shape[0], -1))

# Center the data
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# Compute covariance matrix
cov_matrix = np.cov(X_centered.T)

# Compute eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
idx = eig_vals.argsort()[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

# Select top k eigenvectors
k = 154
top_k_eig_vecs = eig_vecs[:, :k]

# Encode data
X_pca = np.dot(X_centered, top_k_eig_vecs)

# Reconstruct images
X_reconstructed = np.dot(X_pca, top_k_eig_vecs.T) + X_mean
