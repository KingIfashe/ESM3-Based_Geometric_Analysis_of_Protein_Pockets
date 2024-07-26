## calculate code/token nearest neighbour and pairwise similarities 

# this is for nearest neighbour 

#!pip install esm
#!pip install huggingface_hub

from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient
import torch
import torch.nn.functional as F
from esm.pretrained import ESM3_structure_encoder_v0
from esm.utils.structure.protein_chain import ProteinChain
from Bio.PDB import PDBParser, Superimposer
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model
encoder = ESM3_structure_encoder_v0("cuda" if torch.cuda.is_available() else "cpu")

# Get the codebook
codebook = encoder.codebook

# Extract embeddings for all possible tokens
all_tokens = torch.arange(4096).to(encoder.codebook.embeddings.device)
token_embeddings = codebook.embeddings  # Direct access to embeddings
token_embeddings_np = token_embeddings.detach().cpu().numpy()

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(token_embeddings_np)

# Set diagonal to -1 to ignore self-similarity
np.fill_diagonal(similarity_matrix, -1)

# Find the highest similarity (nearest neighbor) for each token
nearest_neighbor_similarities = np.max(similarity_matrix, axis=1)

# Create a histogram of similarities
plt.figure(figsize=(12, 6))
plt.hist(nearest_neighbor_similarities, bins=100, edgecolor='black')
plt.title('Distribution of Cosine Similarities to Nearest Neighbor')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.show()

# Print summary statistics
print(f"Mean similarity to nearest neighbor: {np.mean(nearest_neighbor_similarities):.4f}")
print(f"Median similarity to nearest neighbor: {np.median(nearest_neighbor_similarities):.4f}")
print(f"Min similarity to nearest neighbor: {np.min(nearest_neighbor_similarities):.4f}")
print(f"Max similarity to nearest neighbor: {np.max(nearest_neighbor_similarities):.4f}")

# Create a table of the 10 most diverse and 10 least diverse tokens
most_diverse_indices = np.argsort(nearest_neighbor_similarities)[:10]
least_diverse_indices = np.argsort(nearest_neighbor_similarities)[-10:]

print("\nMost Diverse Tokens:")
print("Token ID | Similarity to Nearest Neighbor")
for idx in most_diverse_indices:
    print(f"{idx:8d} | {nearest_neighbor_similarities[idx]:.4f}")

print("\nLeast Diverse Tokens:")
print("Token ID | Similarity to Nearest Neighbor")
for idx in least_diverse_indices:
    print(f"{idx:8d} | {nearest_neighbor_similarities[idx]:.4f}")

# Scatter plot of token index vs similarity to nearest neighbor
plt.figure(figsize=(12, 6))
plt.scatter(range(len(nearest_neighbor_similarities)), nearest_neighbor_similarities, alpha=0.5)
plt.title('Token Index vs Similarity to Nearest Neighbor')
plt.xlabel('Token Index')
plt.ylabel('Cosine Similarity to Nearest Neighbor')
plt.show()

####################################################################################################################################################
####################################################################################################################################################

# this is for pairwise similarity 

import torch
from esm.pretrained import ESM3_structure_encoder_v0
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model
encoder = ESM3_structure_encoder_v0("cuda" if torch.cuda.is_available() else "cpu")

# Get the codebook
codebook = encoder.codebook

# Extract embeddings for all possible tokens
all_tokens = torch.arange(4096).to(encoder.codebook.embeddings.device)
token_embeddings = codebook.embeddings  # Direct access to embeddings
token_embeddings_np = token_embeddings.detach().cpu().numpy()

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(token_embeddings_np)

# Flatten the similarity matrix and remove self-similarities
similarities = similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)]

# Calculate frequencies of similarities
unique, counts = np.unique(similarities.round(decimals=2), return_counts=True)

# Plot cosine similarity against frequency
plt.figure(figsize=(12, 6))
plt.bar(unique, counts, width=0.01)
plt.title('Cosine Similarity Distribution')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.show()

# Print summary statistics
print(f"Mean similarity: {np.mean(similarities):.4f}")
print(f"Median similarity: {np.median(similarities):.4f}")
print(f"Min similarity: {np.min(similarities):.4f}")
print(f"Max similarity: {np.max(similarities):.4f}")

# Create a table of similarity ranges and their frequencies
ranges = np.arange(0, 1.1, 0.1)
hist, _ = np.histogram(similarities, bins=ranges)

print("\nSimilarity Range | Frequency")
print("-----------------|----------")
for i in range(len(ranges)-1):
    print(f"{ranges[i]:.1f} - {ranges[i+1]:.1f}     | {hist[i]}")

# Scatter plot of all pairwise similarities
plt.figure(figsize=(12, 6))
plt.scatter(range(len(similarities)), np.sort(similarities), alpha=0.1)
plt.title('All Pairwise Cosine Similarities (Sorted)')
plt.xlabel('Pair Index')
plt.ylabel('Cosine Similarity')
plt.show()