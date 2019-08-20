from naive_bayes import MultinomialNB, GaussianNB
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('iris.data', header=None)
    # print(df)
    x = df.iloc[:, [0, 1, 2, 3]].values
    y = df.iloc[:, 4].values
    # y = np.where(y == "Iris-setosa", 1, -1)
    # print(y)
    target = np.array([5.1, 3.5, 1.4, 0.2])
    target = np.array([5.9, 3.0, 4.2, 1.5])

    gnb = GaussianNB(x, y)
    gnb.fit()
    print(gnb.predict(target))

    # X = np.array([
    #     [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    #     ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']
    # ])
    # X = X.T
    # y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    #
    # target = np.array([2, 'S'])
    #
    # mnb = MultinomialNB(X, y)
    # mnb.setSmoothFactor(0)
    # mnb.fit()
    # print(mnb.predict(target))
