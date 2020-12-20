import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


# Importing the dataset
dataset = pd.read_csv('datasets/Movies-3200-tru.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# genres = X[:,1]
# # print(genres)
# # print(genres.shape)
# json_vec = np.vectorize(json.loads)
# genres = json_vec(genres)

# # print(genres)
# genresParsed = []
# for idx, row in enumerate(genres):
#     genresParsed.append(list())
#     for cell in row:
#         genresParsed[idx].append(cell['name'].lower())



keywords = X[:,2]
json_vec = np.vectorize(json.loads)
keywords = json_vec(keywords)

# print(keywords)
keywordsParsed = []
for idx, row in enumerate(keywords):
    keywordsParsed.append(list())
    for cell in row:
        keywordsParsed[idx].append(cell['name'])
# print(keywordsParsed)

embeddings_dict = {}
with open('datasets/glove6b/glove.6B.50d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
embeddings_dict['science fiction'] = embeddings_dict['science-fiction']
distinctKeywordEmbeddings = []

# not_in_dict = set()
# not_in_words = set()
corpus = set()
for row in keywordsParsed:
    for cell in row:
        corpus = corpus.union(set(cell.split()))
        # if cell not in embeddings_dict:
        #     not_in_dict.add(cell)
        #     words = cell.split()
        #     not_in_words = not_in_words.union(set(words))
        # else:
        #     distinctKeywordEmbeddings.append(embeddings_dict[cell])
keyword_embeddings = []
for c in corpus:
    if c in embeddings_dict:
        keyword_embeddings.append(embeddings_dict[c])
keyword_embeddings = np.array(keyword_embeddings)
print(keyword_embeddings)
print(keyword_embeddings.shape)


X = keyword_embeddings
# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 20, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, label = 'Cluster 5')
plt.scatter(X[y_hc == 5, 0], X[y_hc == 5, 1], s = 100, label = 'Cluster 6')
plt.scatter(X[y_hc == 6, 0], X[y_hc == 6, 1], s = 100, label = 'Cluster 7')
plt.scatter(X[y_hc == 7, 0], X[y_hc == 7, 1], s = 100, label = 'Cluster 8')
plt.scatter(X[y_hc == 8, 0], X[y_hc == 8, 1], s = 100, label = 'Cluster 9')
plt.scatter(X[y_hc == 9, 0], X[y_hc == 9, 1], s = 100, label = 'Cluster 10')
plt.scatter(X[y_hc == 10, 0], X[y_hc == 10, 1], s = 100, label = 'Cluster 11')
plt.scatter(X[y_hc == 11, 0], X[y_hc == 11, 1], s = 100, label = 'Cluster 12')
plt.scatter(X[y_hc == 12, 0], X[y_hc == 12, 1], s = 100, label = 'Cluster 13')
plt.scatter(X[y_hc == 13, 0], X[y_hc == 13, 1], s = 100, label = 'Cluster 14')
plt.scatter(X[y_hc == 14, 0], X[y_hc == 14, 1], s = 100, label = 'Cluster 15')
plt.scatter(X[y_hc == 15, 0], X[y_hc == 15, 1], s = 100, label = 'Cluster 16')
plt.scatter(X[y_hc == 16, 0], X[y_hc == 16, 1], s = 100, label = 'Cluster 17')
plt.scatter(X[y_hc == 17, 0], X[y_hc == 17, 1], s = 100, label = 'Cluster 18')
plt.scatter(X[y_hc == 18, 0], X[y_hc == 18, 1], s = 100, label = 'Cluster 19')
plt.scatter(X[y_hc == 19, 0], X[y_hc == 19, 1], s = 100, label = 'Cluster 20')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()



# array = np.array(distinctKeywordEmbeddings)
# array = np.unique(array, axis=0)
# print(array)
# print(array.shape)
# print(not_in_dict)
# print('length of not in dict')
# print(len(not_in_dict))
# print('size of distinc words')
# print(len(not_in_words))
