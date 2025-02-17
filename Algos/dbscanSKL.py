import sys
from sklearn.cluster import DBSCAN
import pandas as pd
import plotly.express as px
import utils



def main():
    if len(sys.argv) != 4:
        print("Usage: python3 dbscanSKL.py <Filename> <epsilon> <NumPoints>")
        sys.exit(1)

    datafile = sys.argv[1]
    epsilon = float(sys.argv[2])
    numPoints = int(sys.argv[3])

    df = pd.read_csv(datafile)

    num_df = df.select_dtypes(include=["number"])
    model = DBSCAN(eps=epsilon, min_samples=numPoints)
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
       title = f"DBSCAN using Epsilon: {epsilon}, Min Samples: {numPoints}, Silhouette Score: {silhouette}",
       labels={"clusterStr": "Cluster"},
       opacity=0.8,
       hover_data=df.columns
    )
    fig.show()





if __name__ == "__main__":
    main()
