#%%
import pandas as pd
from keyword_vectorization import WordVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random

class ClusterDictionary():
    def __init__(self, cluster_size, seed):
        wv = WordVectorizer()
        words, X = wv.generate()
        kmeans = KMeans(n_clusters = cluster_size, init = 'k-means++', random_state=seed)
        kmeans.fit(X)
        self.cluster_size = cluster_size
        self.labels = kmeans.labels_
        self.words = words
        self.X = X
        cluster_row_indices = [[] for _ in range(cluster_size)]
        for row_idx, i in enumerate(kmeans.labels_):
            cluster_row_indices[i].append(row_idx)
        cluster_row_indices = np.array(cluster_row_indices)
        centers = []
        for cluster_indices in cluster_row_indices:
            X_in_cluster = X[cluster_indices]
            centers.append(np.mean(X_in_cluster, axis=0))
        centers = np.array(centers)
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
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(vectors)
        nearest_word_idx = neigh.kneighbors(centers, return_distance=False)
        words = np.array(words, dtype=object)
        nearest_word_idx = nearest_word_idx.flatten()
        self.nearest_words = words[nearest_word_idx]

    def get_central_words(self):
        return self.nearest_words

    def generate_dict(self):
        words = self.words
        cluster_dict = [{'center':self.nearest_words[i], 'words': set()} for i in range(self.cluster_size)]
        for row_idx, i in enumerate(self.labels):
            cluster_dict[i]['words'].add(words[row_idx])
        return cluster_dict

ClusterDictionary(50,2778463823)