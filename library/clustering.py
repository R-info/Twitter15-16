import time
import json
import umap
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from library import DS_PATH

def get_topic_num(lda_vec, log=False):
    best = max(lda_vec)

    if log:
        print(f"min : {min(lda_vec)}")
        print(f"max : {best}")
        print(f"avg : {sum(lda_vec)/len(lda_vec)}")

    return np.where(lda_vec == best)[0][0]

def get_topics_label(lda_vec):
    topic_preds = []
    for vec in lda_vec:
        topic_preds.append(get_topic_num(vec))
    return topic_preds

def data_clustering(cluster_model, data):
    start = time.time()

    cluster_model.fit(data)
    if hasattr(cluster_model, 'labels_'):
        labels = cluster_model.labels_.astype(np.int)
    else:
        labels = cluster_model.predict(data)

    end = time.time()
    print(f"-- processing time: {round((end - start)/60, 2)} minutes")

    return labels

def k_means_clustering(data, n_topics: int = 16, text_data=None, reducer=None):
    print("K-Means Clustering...")

    kmeans = cluster.MiniBatchKMeans(n_clusters=n_topics)
    labels = data_clustering(kmeans, data)

    return labels


def gaussian_mixture_clustering(data, n_topics: int = 16, text_data=None, reducer=None):
    print("Gaussian Mixture Clustering...")

    gmm = mixture.GaussianMixture(n_components=n_topics, covariance_type='full')
    labels = data_clustering(gmm, data)

    return labels
