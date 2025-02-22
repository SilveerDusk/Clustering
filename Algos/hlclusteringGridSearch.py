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

    num_df["cluster"] = model.fit_predict(num_df)
    
    if len(set(num_df["cluster"])) < 2:
        return None  

    return num_df

def generate_dendrogram(df, filename="dendrogram.png"):
    """Generates and saves the dendrogram of the hierarchical clustering."""
    num_df = df.select_dtypes(include=["number"])
    
    linkage_matrix = sch.linkage(num_df, method="single", metric="euclidean")
    
    plt.figure(figsize=(12, 6))
    sch.dendrogram(linkage_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    
    plt.savefig(filename)
    plt.close()
    print(f"Dendrogram saved as {filename}")

def grid_search(datafile, thresholds, n_clusters):
    df = pd.read_csv(datafile)
    best_score = float("-inf")
    best_params = None
    best_df = None
    best_CHScore = None
    best_silhouette = None

    for threshold in thresholds:
        for num_clusters in n_clusters:
            
            clustered_df = hlclustering_and_evaluate(df.copy(), num_clusters, threshold)
            if clustered_df is not None:

                silhouette, CHScore, Rand = utils.compute_cluster_statistics(clustered_df, printIt=False)

                num_clusters_found = clustered_df["cluster"].nunique()
                
                if silhouette > best_score and silhouette != 1.0 and num_clusters == num_clusters_found:
                    best_score = silhouette
                    best_silhouette = silhouette
                    best_CHScore = CHScore
                    best_params = (threshold, num_clusters)
                    best_df = clustered_df

            print(f"Threshold: {threshold}, n_clusters: {num_clusters}, Silhouette: {silhouette:.4f}")

    print(f"\nBest Params: Threshold={best_params[0]} and n_clusters: {best_params[1]} with Silhouette={best_silhouette:.4f} and CHScore={best_CHScore:.4f}")
    
    if best_df is not None:
        generate_dendrogram(best_df)
        utils.compute_cluster_statistics(best_df, printIt=True)

        best_df["clusterStr"] = best_df["cluster"].astype(str)
        best_num_df = best_df.select_dtypes(include=["number"])

        fig = px.scatter_matrix(
            best_df,
            dimensions=best_num_df.columns[:(len(best_num_df.columns)-1)], 
            color = "clusterStr",
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
    n_clusters = list(range(2, 5))

    grid_search(datafile,thresholds=thresholds, n_clusters= n_clusters)