import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

class WordVectorizer():
    def generate(self, corpus_dataset='datasets/Movies-3200-tru.csv', dict_dataset='datasets/glove6b/glove.6B.50d.txt'):
        # Importing the dataset
        dataset = pd.read_csv(corpus_dataset)
        X = dataset.iloc[:, :-1].values

        keywords = X[:,2]
        json_vec = np.vectorize(json.loads)
        keywords = json_vec(keywords)

        # print(keywords)
        keywordsParsed = []
        for idx, row in enumerate(keywords):
            keywordsParsed.append(list())
            for cell in row:
                keywordsParsed[idx].append(cell['name'])

        embeddings_dict = {}
        with open(dict_dataset, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        embeddings_dict['science fiction'] = embeddings_dict['science-fiction']

        corpus = set()
        for row in keywordsParsed:
            for cell in row:
                corpus = corpus.union(set(cell.split()))
        keyword_embeddings = []
        keywords = []
        for c in corpus:
            if c in embeddings_dict:
                keywords.append(c)
                keyword_embeddings.append(embeddings_dict[c])
        keywords = np.array(keywords)
        keyword_embeddings = np.array(keyword_embeddings)
        return keyword_embeddings
