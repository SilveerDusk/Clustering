import numpy as np
import pandas as pd
import sys
from sklearn.metrics import silhouette_score, calinski_harabasz_score, rand_score


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def normalize(value, min_val, max_val):
    """Apply Min-Max normalization to scale values between 0 and 1."""
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

def compute_silhouette_score(X, labels):
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



def fetchDataset(filename):
    data = pd.read_csv(filename)
    if data is None:
        print("Error reading file")
        sys.exit(1)
    included_columns = data.columns.where(data.columns != '0').dropna()
    return data[included_columns]




def compute_cluster_statistics(df, cluster_col='cluster', ground_truth=None, printIt=True):
    feature_cols = [col for col in df.columns if col != cluster_col]
    data_points = df[feature_cols].values
    cluster_labels = df[cluster_col].values
    unique_clusters = np.unique(cluster_labels[~pd.isna(cluster_labels)]) 
    
    cluster_radii = []
    intercluster_distances = []
    cluster_centroids = {}
    
    for cluster in unique_clusters:
        cluster_points = df[df[cluster_col] == cluster][feature_cols].values
        num_points = len(cluster_points)
        
        if num_points == 0:
            continue
        
        centroid = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        avg_dist = np.mean(distances)
        sse = np.sum(distances ** 2)
        cluster_radius = max_dist
        
        cluster_radii.append(cluster_radius)
        cluster_centroids[cluster] = centroid
        
        if printIt:
            print(f"\nCluster {cluster}:")
            print(f"  Number of Points: {num_points}")
            print(f"  Centroid: {centroid}")
            print(f"  Min Distance to Centroid: {min_dist:.4f}")
            print(f"  Max Distance to Centroid: {max_dist:.4f}")
            print(f"  Average Distance to Centroid: {avg_dist:.4f}")
            print(f"  Sum of Squared Errors (SSE): {sse:.4f}")
            print(f"  Cluster Radius: {cluster_radius:.4f}")
    
    for i, cluster_i in enumerate(unique_clusters):
        for j, cluster_j in enumerate(unique_clusters):
            if i >= j:
                continue
            
            centroid_i = cluster_centroids[cluster_i]
            centroid_j = cluster_centroids[cluster_j]
            dist = np.linalg.norm(centroid_i - centroid_j)
            intercluster_distances.append(dist)
    
    avg_intercluster_distance = np.mean(intercluster_distances) if intercluster_distances else np.nan
    radius_to_intercluster_ratio = np.mean(np.array(cluster_radii)) / np.mean(np.array(intercluster_distances))
    
    if printIt:
        print("\nOverall Clustering Statistics:")
        print(f"  Average Inter-cluster Distance: {avg_intercluster_distance:.4f}")
        print(f"  Average Ratio of Cluster Radii to Intercluster Distances: {radius_to_intercluster_ratio:.4f}")
    
    silhouette_avg = None
    if len(unique_clusters) > 1:
        silhouette_avg = silhouette_score(data_points, cluster_labels)
        if printIt:
            print(f"  Silhouette Score of Clustering: {silhouette_avg:.4f}")
    
    calinski_harabasz_index = calinski_harabasz_score(data_points, cluster_labels)
    if printIt:
        print(f"  Calinski-Harabasz Index: {calinski_harabasz_index:.4f}")
    
    rand_index = None
    if ground_truth is not None:
        ground_truth_labels = np.array(ground_truth)
        rand_index = rand_score(ground_truth_labels, cluster_labels)
        if printIt:
            print(f"  Rand Index: {rand_index:.4f}")
    
    return silhouette_avg, calinski_harabasz_index, rand_index
