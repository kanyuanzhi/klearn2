from klearn import Klearn
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

np.seterr(divide='ignore', invalid='ignore')


class KNN(Klearn):
    def __init__(self, x, y, *args, **kwargs):
        super().__init__(x, y, *args, **kwargs)
        if 'k' in kwargs:
            self.k = kwargs['k']
        else:
            self.k = 5

        self.kdtree = KDTree(x, y)

    def draw(self):
        draw(self.kdtree.root)


class KDTree(object):
    # 构造kd树
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dimensions = x.shape[1]  # 数据维度
        self.origin_index_list = np.array([i for i in range(x.shape[0])]).reshape(-1, 1)  # 原始索引
        self.root = self.create(x, y, self.origin_index_list, 0)

    def getMedian(self, data):
        uniques = np.unique(data)  # 去重
        size = len(uniques)
        if size % 2 == 0:  # 判断列表长度为偶数
            median = uniques[size // 2]
        else:  # 判断列表长度为奇数
            median = uniques[(size - 1) // 2]
        return median

    def create(self, x, y, origin_index_list, d):
        if len(x) == 1:
            return KDNode(x[0], y[0], origin_index_list[0], d)
        else:
            order_list = x[:, d].argsort()
            x = x[order_list]
            y = y[order_list]
            origin_index_list = origin_index_list[order_list]

            target_x = x[:, d]  # d维所在列
            median = self.getMedian(target_x)
            index = np.where(target_x == median)[0][0]

            current_node = KDNode(x[index], y[index], origin_index_list[index], d)

            left_x = x[0:index]
            right_x = x[index + 1:len(x)]
            left_y = y[0:index]
            right_y = y[index + 1:len(y)]
            left_ori = origin_index_list[0:index]
            right_ori = origin_index_list[index + 1:len(y)]
            if len(left_x) > 0:
                left_node = self.create(np.array(left_x), left_y, left_ori, (d + 1) % self.dimensions)
                current_node.left_node = left_node
                left_node.parent = current_node

            if len(right_x) > 0:
                right_node = self.create(np.array(right_x), right_y, right_ori, (d + 1) % self.dimensions)
                current_node.right_node = right_node
                right_node.parent = current_node

            return current_node

    def draw(self):
        draw(self.root)


def createGraph(G, node, pos={}, x=0, y=0, layer=1):
    pos[node.origin_index] = (x, y)
    if node.left_node:
        G.add_edge(node.origin_index, node.left_node.origin_index)
        l_x, l_y = x - 1 / 2 ** layer, y - 1
        l_layer = layer + 1
        createGraph(G, node.left_node, x=l_x, y=l_y, pos=pos, layer=l_layer)
    if node.right_node:
        G.add_edge(node.origin_index, node.right_node.origin_index)
        r_x, r_y = x + 1 / 2 ** layer, y - 1
        r_layer = layer + 1
        createGraph(G, node.right_node, x=r_x, y=r_y, pos=pos, layer=r_layer)
    return G, pos


def draw(root, size=(25, 20)):  # 以某个节点为根画图
    graph = nx.DiGraph()
    graph, pos = createGraph(graph, root)
    fig, ax = plt.subplots(figsize=size)  # 比例可以根据树的深度适当调节
    nx.draw_networkx(graph, pos, ax=ax, node_size=400)
    plt.show()


class KDNode(object):
    # kd树的节点
    def __init__(self, data, tag, origin_index, d):
        self.data = data  # 数据
        self.tag = tag  # 标签
        self.origin_index = origin_index[0]  # 在原始数据列中的索引
        self.dimension = d  # 维度
        self.parent = None
        self.left_node = None
        self.right_node = None
