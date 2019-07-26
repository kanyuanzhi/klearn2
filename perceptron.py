from klearn import Klearn

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class Perceptron(Klearn):
    def __init__(self, x, y, eta=0.1, epoch=1000, *args, **kwargs):
        super().__init__(x, y, *args, **kwargs)
        self.eta = eta
        self.epoch = epoch
        self.w = np.zeros(self.x.shape[1]).reshape(-1, 1)
        self.a = np.zeros(self.x.shape[0]).reshape(-1, 1)
        self.b = 0
        self.w_list = [self.w]
        self.a_list = [self.a]
        self.b_list = [self.b]

    def train(self):
        count = 0
        while count <= self.epoch:
            error = False
            count += 1
            if self.verbose:
                print(count)
                print(self.lossFunction())
            for xi, yi in zip(self.x, self.y):
                delta = (np.dot(xi.reshape(1, -1), self.w)+self.b)*yi
                if delta <= 0.0:
                    error = True
                    self.w += self.eta*yi*xi.reshape(-1, 1)
                    self.b += self.eta*yi
                    self.w_list.append(self.w)
                    self.b_list.append(self.b)
                    break
            if not error:
                # 没有误分类的点
                break
        return self.w, self.b

    def trainDual(self):
        gram_matrix = np.dot(self.x, self.x.T)
        count = 0
        while count <= self.epoch:
            error = False
            count += 1
            if self.verbose:
                print(count)
                print(self.lossFunction())
            for i in range(self.x.shape[0]):
                delta = self.y[i]*(np.sum(self.a * self.y * gram_matrix[:,i])+self.b)

            # for xi, yi in zip(self.x,self.y):
            #     delta = yi*(np.sum(self.a * self.y * gram_matrix[:,self.x.argwhere(q==xi)])+self.b)
                if delta <= 0.0:
                    error = True
                    self.a[i] += self.eta
                    self.b = self.eta * self.y[i]
                    w = 0
                    for j in range(self.x.shape[0]):
                        # print(j)
                        w += self.a[j] * self.x.T[:,j]
                    self.w = w.reshape(-1,1)
                    self.a_list.append(self.a)
                    self.w_list.append(self.w)
                    self.b_list.append(self.b)
                    break
            if not error:
                break
        return self.w, self.b

    def lossFunction(self):
        return np.sum((np.dot(self.x, self.w)+self.b)*self.y*(-1))

    def draw(self):
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(self.y))])
        x_min = np.min(self.x[:, 0])-1
        x_max = np.max(self.x[:, 0])+1
        index = np.arange(x_min, x_max, 0.1)
        for idx, cl in enumerate(np.unique(self.y)):
            plt.scatter(x=self.x[self.y == cl, 0], y=self.x[self.y == cl, 1], alpha=0.8, c=colors[idx],
                        marker=markers[idx], label=cl)

        # plt.scatter(x=self.x[:, 0], y=self.x[:, 1])
        if self.verbose:
            for i in range(len(self.w_list)):
                plt.plot(index, (-self.b_list[i]-self.w_list[i][0][0]
                                 * index)/self.w_list[i][1][0], label="epoch:"+str(i+1))
        else:
            plt.plot(
                index, (-self.b-self.w[0][0] * index)/self.w[1][0])
        plt.legend()
        plt.show()
