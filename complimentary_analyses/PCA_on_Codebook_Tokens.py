# PCA on Codebook Tokens
# Perform PCA to cluster the tokens and analyse which tokens are similar

from sklearn.decomposition import PCA
import torch

# Assuming all_structure_tokens is a dictionary with PDB filenames as keys and token tensors as values
all_tokens = []
for tokens in all_structure_tokens.values():
    all_tokens.append(tokens.view(-1).cpu().numpy())  # Flatten each tensor and convert to numpy array

all_tokens = np.vstack(all_tokens)  # Stack all token arrays vertically

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_tokens)

# Plot PCA results
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of ESM3 Structural Tokens')
plt.show()