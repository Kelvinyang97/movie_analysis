#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from keyword_vectorization import WordVectorizer
from sklearn.decomposition import PCA

wv = WordVectorizer()
X = wv.generate()

# Applying PCA
dimensions = [2*x for x in range(1,25)]
for dim in dimensions:
    pca = PCA(n_components = dim)
    X_reducecd = pca.fit_transform(X)
    SSE = []
    cluster_range = [5*x for x in range(1,40)]
    for cluster in cluster_range:
        kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init = 'k-means++')
        kmeans.fit(X)
        SSE.append(kmeans.inertia_)

    # converting the results into a dataframe and plotting them
    frame = pd.DataFrame({'Cluster':cluster_range, 'SSE':SSE})
    plt.figure(figsize = (24,8))
    plt.plot(frame['Cluster'], frame['SSE'], marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title(f"elbow graph for pca dim {dim}")
# %%
