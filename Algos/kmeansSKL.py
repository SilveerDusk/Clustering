import sys

from sklearn.cluster import KMeans
import utils
import pandas as pd
import plotly.express as px

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 kmeansSKL.py <Filename> <k>")
        sys.exit(1)

    datafile = sys.argv[1]
    k = int(sys.argv[2])

    df = pd.read_csv(datafile)
    num_df = df.select_dtypes(include=["number"])

    model = KMeans(n_clusters=k).fit(df)
    df["cluster"] = model.predict(df)

    df["clusterStr"] = df["cluster"].astype(str)
    silhouette = utils.compute_silhouette_score(num_df, df["cluster"])
    print(silhouette)

    fig = px.scatter(
       df,
       #Currently just shows uses the first two numeric as X and Y
       x = num_df.columns[0],
       y = num_df.columns[1],
       color = "clusterStr",
       title = f"Agglomerative Clustering using data: {datafile}, k: {k}, Silhouette Score: {silhouette}",
       labels={"clusterStr": "Cluster"},
       opacity=0.8,
       hover_data=df.columns
    )
    fig.show()


if __name__ == "__main__":
    main()