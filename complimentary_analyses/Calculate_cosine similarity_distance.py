## Calculate cosine similarity, distance matrix, PCA, T-SNE

#!pip install esm
#!pip install huggingface_hub


from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient
from esm.pretrained import ESM3_structure_encoder_v0
from esm.utils.structure.protein_chain import ProteinChain
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser, Superimposer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# Login to Hugging Face Hub
login(token='', add_to_git_credential=True)

model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
encoder = ESM3_structure_encoder_v0("cuda")

# The encoder is already an instance of StructureTokenEncoder
# So directly access its codebook
codebook = encoder.codebook

# Extract embeddings for all possible tokens
all_tokens = torch.arange(4096).to(encoder.codebook.embeddings.device)
token_embeddings = codebook.embeddings  # Direct access to embeddings
token_embeddings_np = token_embeddings.detach().cpu().numpy()

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(token_embeddings_np)
distance_matrix = 1 - similarity_matrix

# Cluster the tokens
n_clusters = 10  # You can adjust this number
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(token_embeddings_np)

# Visualise distance matrix
plt.figure(figsize=(10, 10))
plt.imshow(distance_matrix, cmap='viridis')
plt.colorbar()
plt.title('Distance Matrix of Token Embeddings')
plt.show()

# Visualise cluster distribution
plt.figure(figsize=(10, 5))
plt.hist(cluster_labels, bins=n_clusters)
plt.title('Distribution of Tokens across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Tokens')
plt.show()

# Function to analyse a specific token
def analyse_token(token_id):
    print(f"Analysis for Token {token_id}:")
    print(f"  Cluster: {cluster_labels[token_id]}")
    similar_tokens = np.argsort(similarity_matrix[token_id])[-6:-1][::-1]
    print(f"  Most similar tokens: {similar_tokens}")
    print(f"  Average distance to other tokens: {np.mean(distance_matrix[token_id])}")
    print()

# Analyse the first 5 tokens as an example
for token in range(5):
    analyse_token(token)

# Analyse cluster centroids
cluster_centroids = kmeans.cluster_centers_
for i, centroid in enumerate(cluster_centroids):
    closest_token = np.argmin(np.sum((token_embeddings_np - centroid)**2, axis=1))
    print(f"Cluster {i} centroid closest to token {closest_token}")

# Visualise token embeddings in 2D using PCA

pca = PCA(n_components=2)
token_embeddings_2d = pca.fit_transform(token_embeddings_np)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(token_embeddings_2d[:, 0], token_embeddings_2d[:, 1], c=cluster_labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('2D Visualisation of Token Embeddings')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')


# Reduce dimensions to 2D for visualisation using t-SNE
tsne = TSNE(n_components=2, random_state=42)
token_embeddings_2d_tsne = tsne.fit_transform(token_embeddings_np)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(token_embeddings_2d_tsne[:, 0], token_embeddings_2d_tsne[:, 1], c=cluster_labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('2D Visualisation of Token Embeddings using t-SNE')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.show()

# Function to analyse a specific cluster
def analyse_cluster(cluster_id):
    print(f"Analysis for Cluster {cluster_id}:")
    token_ids = np.where(cluster_labels == cluster_id)[0]
    print(f"  Number of tokens in cluster: {len(token_ids)}")
    for token_id in token_ids[:10]:  # Print first 5 tokens as an example
        print(f"  Token {token_id}: {token_embeddings_np[token_id]}")
    print()

# Analyse the first 5 clusters as an example
for cluster in range(10):
    analyse_cluster(cluster)

# Find and print the closest token to each cluster centroid
for i, centroid in enumerate(cluster_centroids):
    closest_token = np.argmin(np.sum((token_embeddings_np - centroid) ** 2, axis=1))
    print(f"Cluster {i} centroid closest to token {closest_token}")