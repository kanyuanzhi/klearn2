from klearn import Klearn
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


class NaiveBayes(Klearn):
    def __init__(self, x, y, *args, **kwargs):
        super().__init__(x, y, *args, **kwargs)

    def fit(self):
        pass

    def predict(self, target):
        pass


class MultinomialNB(NaiveBayes):
    def __init__(self, x, y, *args, **kwargs):
        super().__init__(x, y, *args, **kwargs)
        self.dimensions = x.shape[1]  # 数据维度
        self.P_y = {}  # P(Y=y)
        self.P_condition = {}  # P(X=x|Y=y)
        self.smooth_factor = 0.0

    def fit(self):
        count_y = {}
        count_condition = {}
        for y in self.y:  # 统计个数
            if y in count_y:
                count_y[y] += 1
            else:
                count_y[y] = 1
        for y in count_y:  # 计算概率
            self.P_y[y] = (count_y[y] + self.smooth_factor) / float(len(self.y) + len(count_y) * self.smooth_factor)
            self.P_condition[y] = [{} for i in range(self.dimensions)]
            count_condition[y] = [{} for i in range(self.dimensions)]

        for index, y in enumerate(self.y):
            for d in range(self.dimensions):
                if self.x[index][d] in count_condition[y][d]:
                    count_condition[y][d][self.x[index][d]] += 1
                else:
                    count_condition[y][d][self.x[index][d]] = 1
                self.P_condition[y][d][self.x[index][d]] = \
                    (count_condition[y][d][self.x[index][d]] + self.smooth_factor) / float(
                        count_y[y] + self.dimensions * self.smooth_factor)

    def predict(self, target):
        candidate_y = {}
        for y in self.P_y:
            p = self.P_y[y]
            for d in range(self.dimensions):
                p *= self.P_condition[y][d][target[d]]
            candidate_y[y] = p
        candidate_y = sorted(candidate_y.items(), key=lambda item: item[1], reverse=True)
        print(candidate_y)
        return candidate_y[0][0]

    def setSmoothFactor(self, a):
        self.smooth_factor = a


class GaussianNB(NaiveBayes):
    def __init__(self, x, y, *args, **kwargs):
        super().__init__(x, y, *args, **kwargs)
        self.dimensions = x.shape[1]  # 数据维度
        self.P_y = {}  # P(Y=y)
        self.P_condition = {}  # P(X=x|Y=y)
        self.mean_value = {}  # 每个类别的样本均值
        self.variance = {}  # 每个类别的样本方差

    def fit(self):
        x_y = {}  # 数据按标签分类
        for index, y in enumerate(self.y):  # 分类
            if y in x_y:
                x_y[y] = np.vstack((x_y[y], np.array([self.x[index]])))
            else:
                x_y[y] = np.array([self.x[index]])

        for y in x_y:
            self.P_y[y] = x_y[y].shape[0] / float(len(self.y))
            self.mean_value[y] = np.mean(x_y[y], axis=0)
            self.variance[y] = np.std(x_y[y], axis=0)

    def predict(self, target):
        candidate_y = {}
        for y in self.P_y:
            p = self.P_y[y]
            for d in range(self.dimensions):
                p *= GaussianNB.computeGaussianProb(self.mean_value[y][d], self.variance[y][d], target[d])
            candidate_y[y] = p
        candidate_y = sorted(candidate_y.items(), key=lambda item: item[1], reverse=True)
        print(candidate_y)
        return candidate_y[0][0]

    @staticmethod
    def computeGaussianProb(m, v, x):
        return 1.0 / (v * np.sqrt(2 * np.pi)) * np.exp(-(x - m) ** 2 / (2 * v ** 2))
