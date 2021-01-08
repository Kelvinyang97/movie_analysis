import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from voting import VoteTopKCenters
from keyword_vectorization import WordVectorizer
class KeywordEncoder():
    def generateEncodedColumn(self, path='datasets/Movies-3200-tru.csv', freq=False):
        v = VoteTopKCenters()
        _, _ = v.get_embeddings('datasets/movie-keywords.csv')
        top_k_words, _ = v.get_top_k_voted(50, 5)
        _ = v.cluster_words()
        word_cluster_dict = v.get_word_cluster_dict()
        wv = WordVectorizer()
        allWordsByRow = wv.keywordsAll()
        columnEncodings = np.full((len(allWordsByRow),len(top_k_words)),0)
        for x, row in enumerate(allWordsByRow):
            for _, word in enumerate(row):
                if word in word_cluster_dict: 
                    if freq:
                        columnEncodings[x,word_cluster_dict[word]] += 1
                    else:
                        columnEncodings[x,word_cluster_dict[word]] = 1
        df = pd.DataFrame(columnEncodings)
        df.to_csv('datasets/encoding.csv', index=False)
        return columnEncodings
ke = KeywordEncoder()
ke.generateEncodedColumn()