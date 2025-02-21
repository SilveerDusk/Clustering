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


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

# Min and Max values for normalization (Update these based on data range)
MIN_SIL = 0.0  # Silhouette scores range from -1 to 1, but we usually consider [0,1]
MAX_SIL = 1.0

MIN_CH = 500  # Minimum CHScore observed (adjust if needed)
MAX_CH = 2500  # Maximum CHScore observed (adjust if needed)


def normalize(value, min_val, max_val):
    """Apply Min-Max normalization to scale values between 0 and 1."""
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0


def kmeans_and_evaluate(df, k):
    num_df = df.select_dtypes(include=["number"])


    model = KMeans(n_clusters=k).fit(num_df)
    df["cluster"] = model.predict(num_df)

    if len(set(df["cluster"])) < 2:
        return -1, None  

    silhouette = utils.compute_silhouette_score(num_df, df["cluster"])
    CHScore = calinski_harabasz_score(num_df, df["cluster"])



    silhouette_norm = normalize(silhouette, MIN_SIL, MAX_SIL)
    CHScore_norm = normalize(CHScore, MIN_CH, MAX_CH)

    avgScore = (silhouette_norm + CHScore_norm) / 2
    return avgScore, silhouette, CHScore, df



def grid_search(datafile, kvalues):
    df = utils.fetchDataset(datafile)
    best_score = float("-inf")
    best_params = None
    best_silhouette = None
    best_CH = None
    best_df = None

    for k in kvalues:
    
        avgScore, silhouette, CHScore, clustered_df = kmeans_and_evaluate(df.copy(), k)

        if avgScore > best_score:
            best_score = avgScore
            best_silhouette = silhouette
            best_CH = CHScore
            best_params = k
            best_df = clustered_df

        print(f"K: {k}, avgScore: {avgScore:.4f}, Silhouette: {silhouette:.4f}, CH Score: {CHScore:.4f}")

    print(f"\nBest Params: K={best_params} with avg Score={best_score:.4f}, Silhouette={best_silhouette:.4f}, CHScore={best_CH:.4f} ")
    
    if best_df is not None:
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
    if len(sys.argv) != 2:
        print("Usage: python3 kmeansGridSearch.py <Filename>")
        sys.exit(1)

    datafile = sys.argv[1]

    kvalues = list(range(2, 10))

    grid_search(datafile, kvalues)