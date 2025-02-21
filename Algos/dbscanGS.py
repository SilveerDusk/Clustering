import sys
from sklearn.cluster import DBSCAN
import pandas as pd
import plotly.express as px
import utils
import numpy as np
from dbscan import dbscan
from utils import fetchDataset



def dbscan_and_evaluate(df, epsilon, numPoints):
    num_df = df.select_dtypes(include=["number"])
    model = DBSCAN(eps=epsilon, min_samples=numPoints)
    df["cluster"] = model.fit_predict(num_df)

    # Ignore clusters with only one sample to avoid silhouette errors
    if len(set(df["cluster"])) < 2:
        return -1, None  # Invalid clustering

    silhouette = utils.compute_silhouette_score(num_df, df["cluster"])
    
    return silhouette, df



def grid_search(datafile, epsilons, min_samples):
    df = fetchDataset(datafile)
    best_score = float("-inf")
    best_params = None
    best_df = None

    for epsilon in epsilons:
        for numPoints in min_samples:
            silhouette, clustered_df = dbscan(df.copy(), epsilon, numPoints)

            if silhouette > best_score:
                best_score = silhouette
                best_params = (epsilon, numPoints)
                best_df = clustered_df

            print(f"Epsilon: {epsilon}, MinPts: {numPoints}, Silhouette: {silhouette:.4f}")

    print(f"\nBest Params: Epsilon={best_params[0]}, MinPts={best_params[1]} with Silhouette={best_score:.4f}")
    
    if best_df is not None:
        best_df["clusterStr"] = best_df["cluster"].astype(str)
        best_num_df = best_df.select_dtypes(include=["number"])
        print(best_num_df.columns)
        fig = px.scatter_matrix(
            best_df,
            dimensions=best_num_df.columns[:(len(best_num_df.columns)-1)], 
            color = "clusterStr",
            title = f"DBSCAN using Epsilon: {epsilon}, Min Samples: {numPoints}, Silhouette Score: {best_score}",
            labels={"clusterStr": "Cluster"},
            opacity=0.8,
            hover_data=best_df.columns
        )
        fig.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 dbscanGridSearch.py <Filename>")
        sys.exit(1)

    datafile = sys.argv[1]

    epsilons = np.arange(0.5, 10, 0.5).tolist()
    min_samples = list(range(2, 10))

    grid_search(datafile, epsilons, min_samples)