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



def compute_cluster_statistics(clusters, ground_truth=None):
    print("\nCluster Statistics:")
    
    # Variables to compute overall clustering metrics
    cluster_radii = []
    intercluster_distances = []
    all_points = []
    
    for i, cluster in enumerate(clusters):
        num_points = len(cluster)
        centroid = np.mean(cluster, axis=0)
        distances = np.linalg.norm(cluster - centroid, axis=1)

        min_dist = np.min(distances)
        max_dist = np.max(distances)
        avg_dist = np.mean(distances)
        sse = np.sum(distances ** 2)
        cluster_radius = max_dist  # Cluster radius is the max distance from centroid
        
        cluster_radii.append(cluster_radius)
        all_points.extend(cluster)
        
        print(f"\nCluster {i + 1}:")
        print(f"  Number of Points: {num_points}")
        print(f"  Centroid: {centroid}")
        print(f"  Min Distance to Centroid: {min_dist:.4f}")
        print(f"  Max Distance to Centroid: {max_dist:.4f}")
        print(f"  Average Distance to Centroid: {avg_dist:.4f}")
        print(f"  Sum of Squared Errors (SSE): {sse:.4f}")
        print(f"  Cluster Radius: {cluster_radius:.4f}")

    # Calculate inter-cluster distances
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            # Compute the distance between centroids of cluster i and cluster j
            centroid_i = np.mean(clusters[i], axis=0)
            centroid_j = np.mean(clusters[j], axis=0)
            dist = np.linalg.norm(centroid_i - centroid_j)
            intercluster_distances.append(dist)

    # Compute overall clustering metrics
    avg_intercluster_distance = np.mean(intercluster_distances)
    radius_to_intercluster_ratio = np.mean(np.array(cluster_radii) / np.array(intercluster_distances))

    print("\nOverall Clustering Statistics:")
    print(f"  Average Inter-cluster Distance: {avg_intercluster_distance:.4f}")
    print(f"  Average Ratio of Cluster Radii to Intercluster Distances: {radius_to_intercluster_ratio:.4f}")

    # Silhouette Score for all clusters
    silhouette_avg = silhouette_score(np.array(all_points), np.array([i for i, cluster in enumerate(clusters) for _ in cluster]))
    print(f"  Silhouette Score of Clustering: {silhouette_avg:.4f}")

    # Calinski-Harabasz Index
    calinski_harabasz_index = calinski_harabasz_score(np.array(all_points), np.array([i for i, cluster in enumerate(clusters) for _ in cluster]))
    print(f"  Calinski-Harabasz Index: {calinski_harabasz_index:.4f}")
    
    # Rand Index if ground truth labels are available
    if ground_truth is not None:
        ground_truth_labels = np.array(ground_truth)
        predicted_labels = np.array([i for i, cluster in enumerate(clusters) for _ in cluster])
        rand_index = rand_score(ground_truth_labels, predicted_labels)
        print(f"  Rand Index: {rand_index:.4f}")