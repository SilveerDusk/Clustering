import utils
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import copy
from pprint import pprint
import plotly.express as px


def hierarchicalClustering(df, threshold, linkage, datafile):
    clusters = initialize_clusters(df)
    merge_history = []
    cluster_indices = list(range(len(clusters)))
    threshold_clusters = None 
    num_df = pd.read_csv(datafile)
    if datafile == "datasets/iris.csv":
        ground_truth = num_df.iloc[:, -1]
    else:
        ground_truth = None
    num_df = num_df.select_dtypes(include=["number"])


    while len(clusters) > 1:
        dist_matrix = compute_distances(clusters, linkage_type=linkage)
        i, j = find_closest_clusters(distance_matrix=dist_matrix)

        # get merge distance
        merge_distance = dist_matrix[i, j]
        new_cluster_size = len(clusters[i]) + len(clusters[j])

        # Currently doesn't work right now with threshold
        if threshold is not None and merge_distance > threshold and threshold_clusters is None:
            threshold_clusters = copy.deepcopy(clusters)

        # Add to merge history for plotting
        merge_history.append([cluster_indices[i], cluster_indices[j], merge_distance, new_cluster_size])

        # Merge clusters
        merge_clusters(clusters=clusters, i=i, j=j)

        # Assign new index for the merged cluster
        new_cluster_id = len(df) + len(merge_history) - 1
        cluster_indices[i] = new_cluster_id 
        del cluster_indices[j]

    merge_history = np.array(merge_history)
    plot_dendrogram(merge_history, df.shape[0])
    final_clusters = threshold_clusters if threshold_clusters is not None else clusters
    df["cluster"] = None
    print("Finalized Clusters\n")
    print(final_clusters)
    for cluster_number, cluster in enumerate(final_clusters):
        for point in cluster:
            index = np.where((df.iloc[:, :-1] == point).all(axis=1))[0]
            if len(index) > 0:
                df.loc[index, 'cluster'] = cluster_number  # Assign to all occurrences

    silhouette, CHscore, Rand = utils.compute_cluster_statistics(df, ground_truth=ground_truth)
    df["clusterStr"] = df["cluster"].astype(str)
    fig = px.scatter_matrix(
        df,
        dimensions=num_df.columns, 
        color = "clusterStr",
        title = f"Agglomerative Clustering using data: {datafile}, threshold: {threshold}, Silhouette Score: {silhouette}, CHscore: {CHscore}",
        labels={"clusterStr": "Cluster"},
        opacity=0.8,
        hover_data=df.columns
    )
    fig.show()

    return final_clusters


def initialize_clusters(df):
    clusters = [np.array([row]) for row in df.values]
    return clusters


def compute_distances(clusters, linkage_type):
    num_clusters = len(clusters)
    distance_matrix = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(i+1, num_clusters):
            if linkage_type == "centroid":
                centroid_i = np.mean(list(clusters[i]), axis=0)
                centroid_j = np.mean(list(clusters[j]), axis=0)
                distance_matrix[i, j] = np.linalg.norm(centroid_i - centroid_j)
            elif linkage_type == "single":
                min_dist = np.inf
                for p_i in clusters[i]:
                    for p_j in clusters[j]:
                        dist = np.linalg.norm(np.array(p_i) - np.array(p_j))
                        if dist < min_dist:
                            min_dist = dist
                distance_matrix[i,j] = min_dist
            elif linkage_type == "complete":
                max_dist = -np.inf
                for p_i in clusters[i]:
                    for p_j in clusters[j]:
                        dist = np.linalg.norm(np.array(p_i) - np.array(p_j))
                        if dist > max_dist:
                            max_dist = dist
                distance_matrix[i,j] = max_dist
            else:
                raise ValueError("Unsupported linkage type")

            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix

def find_closest_clusters(distance_matrix):
    min_dist = np.inf
    closest_clusters = (None, None)

    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            if distance_matrix[i, j] < min_dist:
                min_dist = distance_matrix[i, j]
                closest_clusters = (i, j)
    return closest_clusters

def merge_clusters(clusters, i, j):
    new_cluster = np.vstack((clusters[i], clusters[j]))  
    clusters[i] = new_cluster
    del clusters[j] 


def plot_dendrogram(merge_history, num_points):
    linkage_matrix = np.array(merge_history)
    
    if linkage_matrix.shape[0] != num_points - 1 or linkage_matrix.shape[1] != 4:
        raise ValueError(f"Invalid linkage matrix shape: {linkage_matrix.shape}. Expected ({num_points - 1}, 4).")

    plt.figure(figsize=(10, 6))

    dendrogram(linkage_matrix, labels=np.arange(num_points))
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.savefig("dendrogramScratch.png")


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 hlclustering <Filename> [<threshold>]")
        sys.exit(1)

    datafile = sys.argv[1]
    if len(sys.argv) == 3:
        threshold = float(sys.argv[2])
    else:
        threshold = None
    df = pd.read_csv(datafile)
    df = df.select_dtypes(include=["number"])
    hierarchicalClustering(df, threshold, linkage="centroid", datafile=datafile)

if __name__ == "__main__":
    main()