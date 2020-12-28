#%%
import pandas as pd
from keyword_vectorization import WordVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
NUM_CLUSTER = 50
wv = WordVectorizer()
X = wv.generate()
SSE = []
cluster_row_indices = [[] for _ in range(NUM_CLUSTER)]
kmeans = KMeans(n_clusters = NUM_CLUSTER, init = 'k-means++', random_state=42)
kmeans.fit(X)
# print(cluster_row_indices)
# print(kmeans.labels_)
for row_idx, i in enumerate(kmeans.labels_):
    cluster_row_indices[i].append(row_idx)
cluster_row_indices = np.array(cluster_row_indices)
centers = []
for cluster_indices in cluster_row_indices:
    X_in_cluster = X[cluster_indices]
    print(X_in_cluster.shape)
    centers.append(np.mean(X_in_cluster, axis=0))
centers = np.array(centers)

# print(centers)
# print(centers.shape)

words = []
vectors = []
with open('datasets/glove6b/glove.6B.50d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        words.append(word)
        vector = np.asarray(values[1:], "float32")
        vectors.append(vector)
vectors = np.array(vectors)
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(vectors)
nearest_word_idx = neigh.kneighbors(centers, return_distance=False)
words = np.array(words, dtype=object)
nearest_word_idx = nearest_word_idx.flatten()
print(nearest_word_idx)
print(words[nearest_word_idx])


# cluster 1 words:
# print(cluster_0_embeddings)
# nearest_word_idx = neigh.kneighbors(cluster_0_embeddings, return_distance=False)
# words = np.array(words, dtype=object)
# nearest_word_idx = nearest_word_idx.flatten()
# print(nearest_word_idx)
# print(words[nearest_word_idx])