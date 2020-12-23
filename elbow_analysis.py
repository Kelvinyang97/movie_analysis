#%%
# ELBOW analysis
import pandas as pd
from keyword_vectorization import WordVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wv = WordVectorizer()
X = wv.generate()
SSE = []
for multiple in range(1, 100):
    cluster = 10*multiple
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init = 'k-means++')
    kmeans.fit(X)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':[10*x for x in range(1, 100)], 'SSE':SSE})
plt.figure(figsize = (24,8))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
# %%
