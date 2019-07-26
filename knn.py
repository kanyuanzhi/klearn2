from klearn import Klearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import networkx as nx

np.seterr(divide='ignore', invalid='ignore')


class Knn(Klearn):
    def __init__(self, x, y, *args, **kwargs):
        super().__init__(x, y, *args, **kwargs)
        if 'k' in kwargs:
            self.k = kwargs['k']
        else:
            self.k = 5


class KDTree(object):
    def __init__(self, x):
        self.x = x
        rows = x.shape[0]
        self.dimensions = x.shape[1]
        self.root = self.create(x, 0)
        # median = np.median(self.x[:,0])

    def getMedian(self, data):
        uniques = np.unique(data)
        size = len(uniques)
        if size % 2 == 0:  # 判断列表长度为偶数
            median = uniques[size // 2]
        if size % 2 == 1:  # 判断列表长度为奇数
            median = uniques[(size - 1) // 2]
        return median

    def create(self, x, d):
        if len(x) == 1:
            return KDNode(x[0])
        else:
            x = x[x[:,d].argsort()]
            target_x = x[:, d] # d维所在列

            median = self.getMedian(target_x)
            index = np.where(target_x == median)[0][0]
            current_node = KDNode(x[index])
            left_x = x[0:index]
            right_x = x[index+1:len(x)]
            if len(left_x) > 0:
                left_node = self.create(np.array(left_x), (d + 1) % self.dimensions)
                current_node.left_node = left_node
                left_node.parent = current_node

            if len(right_x) > 0:
                right_node = self.create(np.array(right_x), (d + 1) % self.dimensions)
                current_node.right_node = right_node
                right_node.parent = current_node

            return current_node

    def create_graph(self, G, node, pos={}, x=0, y=0, layer=1):
        pos[node.data.tostring()] = (x, y)
        print(node.data.tostring())
        if node.left_node:
            G.add_edge(node.data.tostring(), node.left_node.data.tostring())
            l_x, l_y = x - 1 / 2 ** layer, y - 1
            l_layer = layer + 1
            self.create_graph(G, node.left_node, x=l_x, y=l_y, pos=pos, layer=l_layer)
        if node.right_node:
            G.add_edge(node.data.tostring(), node.right_node.data.tostring())
            r_x, r_y = x + 1 / 2 ** layer, y - 1
            r_layer = layer + 1
            self.create_graph(G, node.right_node, x=r_x, y=r_y, pos=pos, layer=r_layer)
        return (G, pos)

    def draw(self):  # 以某个节点为根画图
        graph = nx.DiGraph()
        graph, pos = self.create_graph(graph, self.root)
        fig, ax = plt.subplots(figsize=(20, 20))  # 比例可以根据树的深度适当调节
        nx.draw_networkx(graph, pos, ax=ax, node_size=1000)
        plt.show()


class KDNode(object):
    def __init__(self, data):
        self.data = data
        self.index = 0
        self.parent = None
        self.left_node = None
        self.right_node = None


