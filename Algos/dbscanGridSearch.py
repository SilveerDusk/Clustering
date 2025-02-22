import sys
from sklearn.cluster import DBSCAN
import pandas as pd
import plotly.express as px
from sklearn.metrics import silhouette_score
import utils
import numpy as np
from dbscan import dbscan




def dbscan_and_evaluate(df, epsilon, numPoints):
    num_df = df.select_dtypes(include=["number"])
    model = DBSCAN(eps=epsilon, min_samples=numPoints)
    df["cluster"] = model.fit_predict(num_df)

    # Ignore clusters with only one sample to avoid silhouette errors
    if len(set(df["cluster"])) < 2:
        return None  # Invalid clustering

    return df





def grid_search(datafile, epsilons, min_samples, method):
    df = pd.read_csv(datafile)
    best_score = float("-inf")
    best_params = None
    best_df = None
    best_silhouette = None
    best_CHScore = None

    for epsilon in epsilons:
        for numPoints in min_samples:
            if method == "Scratch":
                clustered_df = dbscan(df.copy(), epsilon, numPoints)
            if method == "SKL":
                clustered_df = dbscan_and_evaluate(df.copy(), epsilon, numPoints)
            if clustered_df is not None:
                silhouette, CHScore, Rand = utils.compute_cluster_statistics(clustered_df, printIt=False)
                
                if silhouette > best_score:
                    best_score = silhouette
                    best_silhouette = silhouette
                    best_CHScore = CHScore
                    best_params = (epsilon, numPoints)
                    best_df = clustered_df

                print(f"Epsilon: {epsilon}, MinPts: {numPoints}, Silhouette: {silhouette:.4f}")

    print(f"\nBest Params: Epsilon={best_params[0]}, MinPts={best_params[1]} with Silhouette={best_silhouette:.4f} and CHScore={best_CHScore:.4f}")
    
    if best_df is not None:
        utils.compute_cluster_statistics(best_df, printIt=True)

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
    if len(sys.argv) < 2:
        print("Usage: python3 dbscanGridSearch.py <Filename> <Method>")
        sys.exit(1)

    datafile = sys.argv[1]
    method = sys.argv[2]

    epsilons = np.arange(0.5, 10, 0.5).tolist()
    min_samples = list(range(2, 10))

    grid_search(datafile, epsilons, min_samples, method)