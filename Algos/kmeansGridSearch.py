import sys
from sklearn.cluster import DBSCAN, KMeans
import pandas as pd
import plotly.express as px
import utils
import numpy as np
from kmeans import kmeans



def kmeans_and_evaluate(df, k):
    num_df = df.select_dtypes(include=["number"])


    model = KMeans(n_clusters=k).fit(num_df)
    df["cluster"] = model.predict(num_df)

    # Ignore clusters with only one sample to avoid silhouette errors
    if len(set(df["cluster"])) < 2:
        return -1, None  # Invalid clustering

    silhouette = utils.compute_silhouette_score(num_df, df["cluster"])
    
    return silhouette, df



def grid_search(datafile, kvalues):
    df = utils.fetchDataset(datafile)
    best_score = float("-inf")
    best_params = None
    best_df = None

    for k in kvalues:
    
        silhouette, clustered_df = kmeans(df.copy(), k)

        if silhouette > best_score:
            best_score = silhouette
            best_params = k
            best_df = clustered_df

        print(f"K: {k}, Silhouette: {silhouette:.4f}")

    print(f"\nBest Params: K={best_params} with Silhouette={best_score:.4f}")
    
    if best_df is not None:
        best_df["clusterStr"] = best_df["cluster"].astype(str)
        best_num_df = best_df.select_dtypes(include=["number"])
        print(best_num_df.columns)
        fig = px.scatter_matrix(
            best_df,
            dimensions=best_num_df.columns[:(len(best_num_df.columns)-1)], 
            color = "clusterStr",
            title = f"Kmeans using K: {best_params}, Silhouette Score: {best_score}",
            labels={"clusterStr": "Cluster"},
            opacity=0.8,
            hover_data=best_df.columns
        )
        fig.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 kmeansGridSearch.py <Filename>")
        sys.exit(1)

    datafile = sys.argv[1]

    kvalues = list(range(2, 10))

    grid_search(datafile, kvalues)