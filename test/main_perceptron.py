from perceptron import Perceptron
import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('iris.data', header=None)
    # print(df)
    x = df.iloc[0:100, [0, 2]].values
    y = df.iloc[0:100, 4].values
    y = np.where(y == "Iris-setosa", 1, -1)
    # print(x)
    # print(y)

    # perceptron = Perceptron(x,y,mode="verbose")
    perceptron = Perceptron(x, y)
    w, b = perceptron.fit()
    print(w, b)

    perceptron2 = Perceptron(x, y)
    w, b = perceptron2.fitDual()
    print(w, b)
    perceptron2.draw()
