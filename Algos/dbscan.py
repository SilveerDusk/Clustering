import sys
from utils import fetchDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from utils import compute_silhouette_score

def plot_circles(X, eps):
    for point in X:
        circle = plt.Circle(point, eps, fill=False, linestyle='--')
        plt.gca().add_artist(circle)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def get_neighbors(X, point_idx, eps):
    distances = [euclidean_distance(X.iloc[point_idx].values, X.iloc[other_idx].values) for other_idx in range(len(X))]
    return [i for i, dist in enumerate(distances) if dist <= eps]

def find_core_points(X, eps, min_pts):
    core_points = []
    for i in range(len(X)):
        if len(get_neighbors(X, i, eps)) >= min_pts:
            core_points.append(i)
    return core_points

def expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_pts):
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor = neighbors[i]
        if labels[neighbor] == -1:  # Noise becomes border point
            labels[neighbor] = cluster_id
        elif labels[neighbor] == 0:  # Unvisited
            labels[neighbor] = cluster_id
            new_neighbors = get_neighbors(X, neighbor, eps)
            if len(new_neighbors) >= min_pts:
                neighbors.extend(new_neighbors)
        i += 1
    return labels

def dbscan(X, eps, min_pts):
    labels = [0] * len(X)  # 0: unvisited, -1: noise
    cluster_id = 0
    core_points = find_core_points(X, eps, min_pts)
    
    for point_idx in range(len(X)):
        if labels[point_idx] != 0:
            continue
        if point_idx in core_points:
            cluster_id += 1
            neighbors = get_neighbors(X, point_idx, eps)
            labels = expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_pts)
        else:
            labels[point_idx] = -1  # Noise

    silhouette = compute_silhouette_score(X.to_numpy(), np.array(labels))
    X["cluster"] = labels

    return silhouette, X

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 dbscan.py <Filename> <eps> <min_pts>")
        sys.exit(1)

    datafile = sys.argv[1]
    eps = float(sys.argv[2])
    min_pts = int(sys.argv[3])

    X = fetchDataset(datafile)
    # Ensure the data is numeric
    X = X.astype(float)

    # Plot the data
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], alpha=0.7)
    plt.title("Sample Data for DBSCAN")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()

    # Run DBSCAN and plot the results
    labels = dbscan(X, eps, min_pts)
    unique_labels = set(labels)

    feature_pairs = list(combinations(range(X.shape[1]), 2))
    n_pairs = len(feature_pairs)
    grid_size = int(np.ceil(np.sqrt(n_pairs)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    if grid_size == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for ax, (i, j) in zip(axes, feature_pairs):
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]  # Black for noise
            class_member_mask = (np.array(labels) == k)
            xy = X[class_member_mask]
            ax.scatter(xy.iloc[:, i], xy.iloc[:, j], s=50, color=tuple(col), label=f'Cluster {k}')
        ax.set_xlabel(f'Feature {i}')
        ax.set_ylabel(f'Feature {j}')
        ax.set_title(f'Clusters for Feature Pair ({i}, {j})')

    # Hide any unused subplots
    for ax in axes[len(feature_pairs):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()