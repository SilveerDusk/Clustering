import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from utils import fetchDataset


def euclidean_distance(x1, x2):
  return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
  return np.sum(np.abs(x1 - x2))

def cosine_similarity(x1, x2):
  return np.dot(x1, x2) / (np.sqrt(np.dot(x1, x1)) * np.sqrt(np.dot(x2, x2)))

def initialize_centroids(X, k):
  n_samples, n_features = X.shape
  centroids = np.zeros((k, n_features))
  for i in range(k):
    centroid =  X.iloc[np.random.choice(range(n_samples))]
    centroids[i] = centroid
  return centroids

def closest_centroid(sample, centroids, metric):
  closest_i = None
  closest_distance = float('inf')
  for i, centroid in enumerate(centroids):
    distance = metric(sample, centroid)
    if distance < closest_distance:
      closest_i = i
      closest_distance = distance
  return closest_i

def form_clusters(X, centroids, k, metric):
  n_samples = X.shape[0]
  clusters = [[] for _ in range(k)]
  for sample_i, sample in X.iterrows():
    centroid_i = closest_centroid(sample, centroids, metric)
    clusters[centroid_i].append(sample_i)
  return clusters

def calculate_new_centroids(X, clusters):
  n_features = X.shape[1]
  centroids = np.zeros((len(clusters), n_features))
  for i, cluster in enumerate(clusters):
    centroid = np.mean(X.iloc[cluster], axis=0)
    centroids[i] = centroid
  return centroids

def calculate_loss(X, clusters, centroids, metric):
  loss = 0.0
  for i, cluster in enumerate(clusters):
      for sample_i in cluster:
          loss += metric(X.iloc[sample_i].values, centroids[i]) ** 2
  return loss

def plot_clusters(X, clusters, centroids):
  n_samples, n_features = X.shape
  colors = ['r', 'g', 'b', 'y', 'c', 'm']
  feature_pairs = list(combinations(range(n_features), 2))
  n_pairs = len(feature_pairs)
  
  # Determine the grid size for the subplots
  grid_size = int(np.ceil(np.sqrt(n_pairs)))
  
  fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
  axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration
  
  for ax, (i, j) in zip(axes, feature_pairs):
      for cluster_i, cluster in enumerate(clusters):
          for sample_i in cluster:
              ax.scatter(X.iloc[sample_i, i], X.iloc[sample_i, j], color=colors[cluster_i])
      for centroid_i, centroid in enumerate(centroids):
          ax.scatter(centroid[i], centroid[j], s=130, c=colors[centroid_i], marker='x')
      ax.set_xlabel(f'Feature {i}')
      ax.set_ylabel(f'Feature {j}')
      ax.set_title(f'Clusters for Feature Pair ({i}, {j})')
  
  # Hide any unused subplots
  for ax in axes[n_pairs:]:
      ax.axis('off')
  
  plt.tight_layout()
  plt.show()

def main():
  if len(sys.argv) != 3:
    print("Usage: python3 kmeans <Filename> <k>")
    sys.exit(1)

  datafile = sys.argv[1]
  k = sys.argv[2]

  X = fetchDataset(datafile)
  X = X.astype(float)
  k = int(k)

  centroids = initialize_centroids(X, k)
  clusters = form_clusters(X, centroids, k, euclidean_distance)
  new_centroids = calculate_new_centroids(X, clusters)
  loss = calculate_loss(X, clusters, new_centroids, euclidean_distance)
  prev_loss = float('inf')

  print("Initial Centroids:\n", centroids)
  print("Clusters:\n", clusters)
  print("New Centroids:\n", new_centroids)
  print("Loss:", loss)

  while prev_loss != loss or centroids.any() != new_centroids.any():
    centroids = new_centroids
    prev_loss = loss
    clusters = form_clusters(X, centroids, k, euclidean_distance)
    new_centroids = calculate_new_centroids(X, clusters)
    loss = calculate_loss(X, clusters, new_centroids, euclidean_distance)

    print("Loss:", loss)

  final_clusters = form_clusters(X, new_centroids, k, euclidean_distance)
  print("Final Clusters:\n", final_clusters)
  plot_clusters(X, final_clusters, new_centroids)


if __name__ == "__main__":
  main()