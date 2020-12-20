import numpy as np
embeddings_dict = {}
with open('datasets/glove6b/glove.6B.50d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
        print(word)
        print(vector)