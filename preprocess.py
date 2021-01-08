# Data Preprocessing Tools

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keyword_vectorization import WordVectorizer

# Importing the dataset
dataset = pd.read_csv('datasets/Movies-3200-tru.csv')
X = dataset.iloc[:, [-2,0]].values
y = dataset.iloc[:, -1].values
keyword_dataset = pd.read_csv('datasets/encoding.csv')
keyword_encodings = keyword_dataset.values
wv = WordVectorizer()
genre_encodings = wv.genresAllBinarize()

Xwithgenres = np.concatenate([X,genre_encodings, keyword_encodings], axis=1)
X = Xwithgenres[:,[range(1,Xwithgenres.shape[1])]]
title = Xwithgenres[:,0]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 0] = sc.fit_transform(X_train[:, 0])
X_test[:, 0] = sc.transform(X_test[:, 0])