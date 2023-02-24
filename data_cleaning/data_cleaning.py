#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:36:07 2018
@author: abhijeet
"""
import csv
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.5')
import pandas as pd
# Importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
import string
import re
import numpy as np
from collections import Counter
from tqdm import tqdm

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


# Cleaning the text sentences so that punctuation marks, stop words & digits are removed
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    # remove the mark &amp;
    amp_free = " ".join(i for i in stop_free.split() if "&amp;" not in i)
    # remove the url
    url_free = " ".join(i for i in amp_free.split() if "https://" not in i)
    punc_free = ''.join(ch for ch in url_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+", "", normalized)
    y = processed.split()
    return y



train_clean_sentences = []
train_sentences = []
df = pd.read_csv("../data/articles.csv", sep=',', dtype=str)
df = df.dropna(axis=0, subset=['selftext'])
for index, line in enumerate(df['selftext']):
    train_sentences.append(line)
    line = line.strip()
    cleaned = clean(line)
    cleaned = ' '.join(cleaned)
    train_clean_sentences.append(cleaned)
    if index == 20001:
        break

df1 = pd.read_csv("../data_extraction/data/articles_puberty.csv", sep=',', dtype=str)
df1 = df1.dropna(axis=0, subset=['selftext'])
for line in df1['selftext']:
    train_sentences.append(line)
    line = line.strip()
    cleaned = clean(line)
    cleaned = ' '.join(cleaned)
    train_clean_sentences.append(cleaned)
print(len(train_clean_sentences))

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_clean_sentences)

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


lsa = make_pipeline(TruncatedSVD(n_components=100))
X_lsa = lsa.fit_transform(X)

print("finish lsa")


print(f"n_samples: {X_lsa.shape[0]}, n_features: {X_lsa.shape[1]}")


from sklearn.cluster import KMeans

clusters = 2
kmeans = KMeans(
    n_clusters=clusters,
    max_iter=100,
    n_init=1,
).fit(X_lsa)

cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
print(f"Number of elements asigned to each cluster: {cluster_sizes}")

original_space_centroids = lsa[0].inverse_transform(kmeans.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(clusters):
    print(f"Cluster {i}: ", end="")
    for ind in order_centroids[i, :50]:
        print(f"{terms[ind]} ", end="")
    print()
import os
dic = {}
color = ['blue', 'yellow', 'green', 'red', 'black']
plt.figure()
path = "../data/svd100/" + str(clusters)
os.makedirs(path, exist_ok=True)
files = []
for i in range(clusters):
    files.append(open(os.path.join(path, str(i) + ".txt"), "w", encoding="utf-8"))
for i in tqdm(range(2000)):
    y = kmeans.predict([X_lsa[i]])
    plt.scatter(X_lsa[i][0],X_lsa[i][1], c=color[y[0]])
    files[y[0]].write(train_sentences[i] + "\n")
    # files[y[0]].write(train_sentences[i])
    # if y[0] in dic:
    #     dic[y[0]].append(train_sentences[i])
    # else:
    #     dic[y[0]] = [train_sentences[i]]

plt.savefig(os.path.join(path, 'test.png'))

