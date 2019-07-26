from knn import KDTree
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('iris.data', header=None)
    # print(df)
    x = df.iloc[:, [0, 1, 2, 3]].values
    y = df.iloc[:, 4].values
    # y = np.where(y == "Iris-setosa", 1, -1)
    print(x)
    print(y)

    a = x[0:2, 0]
    print(a)
    print(np.median(a))
    aaa = np.where(a == 4.9)
    print(np.where(a == 4.9))

    kdtree = KDTree(x)

    kdtree.draw()

    print(kdtree.root.data)
