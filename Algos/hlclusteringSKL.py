# from sklearn.cluster import AgglomerativeClustering
# import pandas as pd
# from scipy.cluster.hierarchy import dendrogram
# from ISLP.cluster import compute_linkage
# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import cut_tree


# def hlclusterSKL(datafile, metric):
#     df = pd.read_csv(datafile)
#     print(df)
#     model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=metric)
#     model.fit(df)


#     linkage_single = compute_linkage(model)
#     fig, ax = plt.subplots(1, 1, figsize=(20, 20))
#     dendrogram(linkage_single, ax=ax, color_threshold=2.7,
#             above_threshold_color='black')
#     plt.show()

#     clusters_single = cut_tree(model, height = 8).T[0]
#     clusters_single

#     clusters = pd.Series(clusters_single).map({
#         0: "orange",
#         1: "red",
#     })

#     df.plot.scatter(x="1", y="1.1",
#                         color=clusters)

# model = hlclusterSKL("datasets/many_clusters.csv", "single")

from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, cut_tree, linkage
import matplotlib.pyplot as plt

def hlclusterSKL(datafile, metric):
    df = pd.read_csv(datafile)

    df = df.select_dtypes(include=["number"])
    print(df.head())

    linkage_matrix = linkage(df, method=metric)

    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(linkage_matrix, ax=ax, color_threshold=2.7, above_threshold_color='black')
    plt.show()

    clusters_single = cut_tree(linkage_matrix, height=15).flatten()

    #colors
    cluster_colors = pd.Series(clusters_single).map({
        0: "orange",
        1: "red",
        2: "blue",
        3: "purple",
        4: "pink",
        5: "green",
        6: "yellow", 
        7: "black",
        8: "brown",
        9: "light_blue"
        
    })

    if {"1", "1.1"}.issubset(df.columns):
        df.plot.scatter(x="1", y="1.1", c=cluster_colors)
        plt.show()
    else:
        print("Columns for plotting not found in the dataset.")

    return clusters_single

clusters = hlclusterSKL("datasets/many_clusters.csv", "centroid")
