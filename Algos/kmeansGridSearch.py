import sys
from sklearn.cluster import DBSCAN, KMeans
import pandas as pd
import plotly.express as px
from sklearn.metrics import calinski_harabasz_score
import utils
import numpy as np
from kmeans import kmeans


MIN_SIL = 0.0
MAX_SIL = 1.0

MIN_CH = 500
MAX_CH = 2500

def kmeans_and_evaluate(df, k):
    num_df = df.select_dtypes(include=["number"])


    model = KMeans(n_clusters=k).fit(num_df)
    df["cluster"] = model.predict(num_df)

    if len(set(df["cluster"])) < 2:
        return None  

    return df



def grid_search(datafile, kvalues, method):
    df = utils.fetchDataset(datafile)
    best_score = float("-inf")
    best_params = None
    best_silhouette = None
    best_CH = None
    best_df = None

    for k in kvalues:
        if method == "Scratch":
            clustered_df = kmeans(df.copy(), k)
        elif method == "SKL":
            clustered_df = kmeans_and_evaluate(df.copy(), k)
        if clustered_df is not None:

            silhouette, CHScore, Rand = utils.compute_cluster_statistics(clustered_df, printIt=False)

            if silhouette > best_score:
                best_score = silhouette
                best_silhouette = silhouette
                best_CH = CHScore
                best_params = k
                best_df = clustered_df

            print(f"K: {k}, Silhouette: {silhouette:.4f}, CH Score: {CHScore:.4f}")

    print(f"\nBest Params: K={best_params} with Silhouette={best_silhouette:.4f}, CHScore={best_CH:.4f} ")
    
    if best_df is not None:
        utils.compute_cluster_statistics(best_df)
        best_df["clusterStr"] = best_df["cluster"].astype(str)

        best_num_df = best_df.select_dtypes(include=["number"])
        print(best_num_df.columns)
        fig = px.scatter_matrix(
            best_df,
            dimensions=best_num_df.columns[:(len(best_num_df.columns)-1)], 
            color = "clusterStr",
            title = f"Kmeans using K: {best_params}, Avg Score: {best_score}, CH_Score: {best_CH}, Silhouette Score: {best_silhouette}",
            labels={"clusterStr": "Cluster"},
            opacity=0.8,
            hover_data=best_df.columns
        )
        fig.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 kmeansGridSearch.py <Filename>")
        sys.exit(1)

    datafile = sys.argv[1]
    method = sys.argv[2]



    kvalues = list(range(2, 10))

    grid_search(datafile, kvalues, method)