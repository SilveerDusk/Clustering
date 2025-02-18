import pandas as pd
import sys

def fetchDataset(filename):
    data = pd.read_csv(filename)
    if data is None:
        print("Error reading file")
        sys.exit(1)
    included_columns = data.columns.where(data.columns != '0').dropna()
    return data[included_columns]
