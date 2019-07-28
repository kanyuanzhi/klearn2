from knn import KDTree, KNN
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('iris.data', header=None)
    # print(df)
    x = df.iloc[:, [0, 1, 2, 3]].values
    y = df.iloc[:, 4].values
    # y = np.where(y == "Iris-setosa", 1, -1)
    x = [[6.27,5.50],[1.24,-2.86],[17.05,-12.79],[-6.88,-5.4],[-2.96,-0.5],
         [7.75,-22.68],[10.80,-5.03],[-4.6,-10.55],[-4.96,12.61],[1.75,12.26],
         [15.31,-13.16],[7.83,15.70],[14.63,-0.35]]
    x = np.array(x)
    y = [1,2,1,1,1,2,1,2,1,1,2,2,1]
    y = np.array(y)
    print(x)
    test = [5.1, 3.5, 1.4, 0.1]



    knn = KNN(x, y)
    # result = knn.classify(test)
    # print(result)


    knn.treeDisplay((10,10))

