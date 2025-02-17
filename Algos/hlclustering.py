import sys
import numpy as np
import pandas as pd

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 hlclustering <Filename> [<threshold>]")
        sys.exit(1)

    datafile = sys.argv[1]
    if len(sys.argv) == 3:
        threshold = sys.argv[2]

    df = pd.read_csv(datafile)
    
    clusters = initialize_clusters(df)
    while len(clusters) > 1:
        dist_matrix = compute_distances(clusters, linkage_type="centroid")
        i, j = find_closest_clusters(distance_matrix=dist_matrix)
        merge_clusters(clusters=clusters, i=i, j=j)
    return clusters[0]


def initialize_clusters(df):
    clusters = [tuple(row) for row in df.values] 
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
    new_cluster = tuple(set(clusters[i]) | set(clusters[j]))
    clusters[i] = new_cluster
    del clusters[j]
    clusters.insert(min(i, j), new_cluster)




if __name__ == "__main__":
    main()