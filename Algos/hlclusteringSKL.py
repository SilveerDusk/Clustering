import sys
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import utils

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 hlclustering <Filename> [<threshold>]")
        sys.exit(1)
    threshold = None
    datafile = sys.argv[1]
    if len(sys.argv) == 3:
        threshold = sys.argv[2]

    df = pd.read_csv(datafile)
    num_df = df.select_dtypes(include=["number"])

    if threshold is not None:
        model = AgglomerativeClustering(metric="euclidean", linkage="single", distance_threshold=threshold, n_clusters=None)
    else:
        model = AgglomerativeClustering(n_clusters=4, metric="euclidean", linkage="single", distance_threshold=None)

    df["cluster"] = model.fit_predict(num_df)

    df["clusterStr"] = df["cluster"].astype(str)
    silhouette = utils.compute_silhouette_score(num_df, df["cluster"])
    print(silhouette)

    fig = px.scatter(
       df,
       #Currently just shows uses the first two numeric as X and Y
       x = num_df.columns[0],
       y = num_df.columns[1],
       color = "clusterStr",
       title = f"Agglomerative Clustering using data: {datafile}, threshold {threshold}, Silhouette Score: {silhouette}",
       labels={"clusterStr": "Cluster"},
       opacity=0.8,
       hover_data=df.columns
    )
    fig.show()


if __name__ == "__main__":
    main()