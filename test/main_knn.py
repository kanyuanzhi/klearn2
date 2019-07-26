from knn import KDTree, KNN
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('iris.data', header=None)
    # print(df)
    x = df.iloc[:, [0, 1, 2, 3]].values
    y = df.iloc[:, 4].values
    # y = np.where(y == "Iris-setosa", 1, -1)

    kdtree = KDTree(x, y)

    kdtree.draw()

    # knn = KNN(x, y)
    # knn.draw(knn.kdtree.root)
