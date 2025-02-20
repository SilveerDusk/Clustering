import sys
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import plotly.express as px
import utils
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

def hlclustering_and_evaluate(df, n_clusters=None, threshold=None):

    num_df = df.select_dtypes(include=["number"])

    if threshold is not None:
        model = AgglomerativeClustering(metric="euclidean", linkage="single", distance_threshold=threshold, n_clusters=None)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="single", distance_threshold=None)

    df["cluster"] = model.fit_predict(num_df)
    
    if len(set(df["cluster"])) < 2:
        return -1, None  # Invalid clustering

    silhouette = utils.compute_silhouette_score(num_df, df["cluster"])

    return silhouette, df

def generate_dendrogram(df, filename="dendrogram.png"):
    """Generates and saves the dendrogram of the hierarchical clustering."""
    num_df = df.select_dtypes(include=["number"])
    
    # Compute the linkage matrix using scipy's linkage function
    linkage_matrix = sch.linkage(num_df, method="single", metric="euclidean")
    
    plt.figure(figsize=(12, 6))
    sch.dendrogram(linkage_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    
    # Save the dendrogram as a file
    plt.savefig(filename)
    plt.close()
    print(f"Dendrogram saved as {filename}")

def grid_search(datafile, thresholds, n_clusters):
    df = pd.read_csv(datafile)
    best_score = float("-inf")
    best_params = None
    best_df = None

    for threshold in thresholds:
        for num_clusters in n_clusters:
            print(num_clusters)
            silhouette, clustered_df = hlclustering_and_evaluate(df.copy(), num_clusters, threshold)
            if clustered_df is not None:
                num_clusters_found = clustered_df["cluster"].nunique()

                if silhouette > best_score and silhouette != 1.0 and num_clusters == num_clusters_found:
                    best_score = silhouette
                    best_params = (threshold, num_clusters)
                    best_df = clustered_df

            print(f"Threshold: {threshold}, n_clusters: {num_clusters}, Silhouette: {silhouette:.4f}")

    print(f"\nBest Params: Threshold={best_params[0]} and n_clusters: {best_params[1]} with Silhouette={best_score:.4f}")
    
    if best_df is not None:
        generate_dendrogram(best_df)
        best_df["clusterStr"] = best_df["cluster"].astype(str)
        fig = px.scatter(
            best_df,
            x=best_df.select_dtypes(include=["number"]).columns[0],
            y=best_df.select_dtypes(include=["number"]).columns[1],
            color="clusterStr",
            title=f"Best HL Clustering for data: {datafile}, Threshold={best_params[0]}, n_clusters: {best_params[1]}, Score={best_score:.4f}",
            labels={"clusterStr": "Cluster"},
            opacity=0.8,
            hover_data=best_df.columns
        )
        fig.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 hlclusteringGridSearch.py <Filename>")
        sys.exit(1)

    datafile = sys.argv[1]

    thresholds = np.arange(4, 15, 0.5).tolist()
    n_clusters = list(range(2, 10))
    print(n_clusters)

    grid_search(datafile,thresholds=thresholds, n_clusters= n_clusters)