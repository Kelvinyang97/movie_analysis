import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from keyword_vectorization import WordVectorizer
import heapq

class VoteTopKCenters():
    def get_embeddings(self, corpus_dataset, dict_dataset='datasets/glove6b/glove.6B.50d.txt'):
        dataset = pd.read_csv(corpus_dataset)
        keywords = dataset.iloc[:, 0].values
        keywords_new = []
        for k in keywords:
            kl = k.split()
            if len(kl)<3:
                keywords_new.append(kl) 
        embeddings_dict = {}
        with open(dict_dataset, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

        keyword_embeddings = []
        keywords_in = []
        for l in keywords_new:
            embeddings = []
            words = []
            for k in l:
                if k.lower() in embeddings_dict:
                    words.append(k.lower())
                    embeddings.append(embeddings_dict[k.lower()])
            if words:
                embeddings = np.array(embeddings)
                mean_embedding = np.mean(embeddings, axis=0)
                keyword_embeddings.append(mean_embedding)
                keywords_in.append(words)
            
        keyword_embeddings = np.array(keyword_embeddings)
        keywords_in = np.array(keywords_in)
        self.centers = keywords_in
        self.center_embeddings = keyword_embeddings
        return keywords_in, keyword_embeddings
    def get_top_k_voted(self, k, num_votes_each):
        self.votes = [0]*len(self.centers)
        wv = WordVectorizer()
        _, X = wv.generate()
        neigh = NearestNeighbors(n_neighbors=num_votes_each)
        neigh.fit(self.center_embeddings)
        nearest_word_idx = neigh.kneighbors(X, return_distance=False)
        nearest_word_idx = nearest_word_idx.flatten()
        for i in nearest_word_idx:
            self.votes[i] += 1
        votes_idx = [(-x, i) for i, x in enumerate(self.votes)]
        heapq.heapify(votes_idx)
        top_k_idx = []
        while len(top_k_idx) < k:
            top_k_idx.append(heapq.heappop(votes_idx)[1])
        self.top_k_words = self.centers[top_k_idx]
        self.top_k_embeddings = self.center_embeddings[top_k_idx]
        return self.top_k_words, self.top_k_embeddings
    
    def cluster_words(self):
        wv = WordVectorizer()
        words, X = wv.generate()
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(self.top_k_embeddings)
        nearest_word_idx = neigh.kneighbors(X, return_distance=False)
        nearest_word_idx = nearest_word_idx.flatten()
        words_by_cluster = [[] for _ in range(len(self.top_k_words))]
        for i, cluster_idx in enumerate(nearest_word_idx):
            words_by_cluster[cluster_idx].append(words[i])
        self.words_by_cluster = words_by_cluster
        return self.words_by_cluster


# v = VoteTopKCenters()
# keywords, keyword_embeddings = v.get_embeddings('datasets/movie-keywords.csv')
# top_k_words, top_k_embeddings = v.get_top_k_voted(50, 5)
# words_by_cluster = v.cluster_words()