import numpy as np

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def compute_silhouette_score(X, labels):
    labels = labels.to_numpy()
    X = X.to_numpy()
    unique_clusters = np.unique(labels)
    silhouette_scores = []

    for i, point in enumerate(X):
        cluster = labels[i]

        #Find all points in same cluster or diff cluster
        same_cluster = X[labels == cluster]
        other_clusters = [X[labels == c] for c in unique_clusters if c != cluster]

        if len(same_cluster) > 1:
            a_i = np.mean([euclidean_distance(point, p) for p in same_cluster if not np.array_equal(p, point)])
        else:
            a_i = 0
        
        b_i = np.min([np.mean([euclidean_distance(point, p) for p in other_cluster]) 
                      for other_cluster in other_clusters]) if other_clusters else 0
        if max(a_i, b_i) != 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0
        silhouette_scores.append(s_i)
    return np.mean(silhouette_scores)

import pandas as pd
import sys

def fetchDataset(filename):
    data = pd.read_csv(filename)
    if data is None:
        print("Error reading file")
        sys.exit(1)
    included_columns = data.columns.where(data.columns != '0').dropna()
    return data[included_columns]
