from klearn import Klearn
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


class KNN(Klearn):
    def __init__(self, x, y, *args, **kwargs):
        super().__init__(x, y, *args, **kwargs)
        self.dimensions = x.shape[1]  # 数据维度
        self.kdtree = KDTree(x, y)
        self.k = 0
        self.target = None  # 待分类点
        self.candidate_nodes = None  # 与待分类点最近的k个点的集合

    def classify(self, target, k=3, **kwargs):
        self.k = k
        self.target = target
        self.candidate_nodes = KNN.CandidateNodes(self.k, target)
        self.search(self.kdtree.root)
        candidate_tags = {}
        for node in self.candidate_nodes.nodes:
            if node.tag in candidate_tags:
                candidate_tags[str(node.tag)] += 1.0
            else:
                candidate_tags[str(node.tag)] = 1.0
        candidate_tags_sorted = sorted(candidate_tags.items(), key=lambda d: d[1], reverse=True)
        if "mode" in kwargs:
            if kwargs["mode"] == "verbose":
                for item in candidate_tags_sorted:
                    print(item[0], str(round(item[1] / self.k * 100, 2)) + "%")
        return candidate_tags_sorted[0][0]
        # return [node.origin_index for node in self.candidate_nodes.nodes]

    def search(self, current):
        current = self.find_leaf(current)
        self.candidate_nodes.append(current)

        while current.parent:
            # 有父节点
            if current.parent.not_visited:
                # 没被访问过
                # current = current.parent
                self.candidate_nodes.append(current.parent)
                if current.parent.children_count == 1:
                    # current是current.parent的唯一子节点
                    current = current.parent
                else:
                    perpendicular_distance = abs(
                        self.target[current.parent.dimension] - current.parent.data[current.parent.dimension])  # 垂直距离
                    if perpendicular_distance < max(self.candidate_nodes.distances):
                        # 若垂直距离小于候选节点中的最大距离，则需进入父节点的另一个子节点寻找
                        if current == current.parent.left:
                            current = current.parent.right
                            self.search(current)
                        else:
                            current = current.parent.left
                            self.search(current)
                    else:
                        current = current.parent
            else:
                # 被访问过
                current = current.parent

    @staticmethod
    def distance(target, current):
        s = 0.0
        for i in range(len(target)):
            s += (target[i] - current.data[i]) ** 2
        return s ** 0.5

    def find_leaf(self, current):
        # print(current.origin_index)
        while current.left or current.right:
            if not current.left:
                current = current.right
            elif not current.right:
                current = current.left
            else:
                if self.target[current.dimension] < current.data[current.dimension]:
                    current = current.left
                else:
                    current = current.right
        return current

    def treeDisplay(self, size=(25, 20)):
        draw(self.kdtree.root, size)

    class CandidateNodes(object):
        def __init__(self, k, target):
            self.k = k
            self.target = target
            self.nodes = []
            self.distances = []
            self.size = 0

        def append(self, node):
            if self.size < self.k:
                self.nodes.append(node)
                distance = KNN.distance(self.target, node)
                self.distances.append(distance)
                self.size += 1
            else:
                distance = KNN.distance(self.target, node)
                max_distance = max(self.distances)
                if distance < max_distance:
                    index = self.distances.index(max_distance)
                    self.nodes.pop(index)
                    self.distances.pop(index)
                    self.nodes.append(node)
                    self.distances.append(distance)
            node.not_visited = False


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
            median = uniques[size // 2 - 1]  # 取中间左侧一位作中位数
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

            current = KDNode(x[index], y[index], origin_index_list[index], d)

            left_x = x[0:index]
            right_x = x[index + 1:len(x)]
            left_y = y[0:index]
            right_y = y[index + 1:len(y)]
            left_oil = origin_index_list[0:index]
            right_oil = origin_index_list[index + 1:len(y)]
            if len(left_x) > 0:
                left = self.create(np.array(left_x), left_y, left_oil, (d + 1) % self.dimensions)
                current.left = left
                current.children_count += 1
                left.parent = current

            if len(right_x) > 0:
                right = self.create(np.array(right_x), right_y, right_oil, (d + 1) % self.dimensions)
                current.right = right
                current.children_count += 1
                right.parent = current

            return current

    def draw(self, size=(25, 20)):
        draw(self.root, size)


def createGraph(G, node, pos={}, x=0, y=0, layer=1):
    pos[node.origin_index] = (x, y)
    if node.left:
        G.add_edge(node.origin_index, node.left.origin_index)
        l_x, l_y = x - 1 / 2 ** layer, y - 1
        l_layer = layer + 1
        createGraph(G, node.left, x=l_x, y=l_y, pos=pos, layer=l_layer)
    if node.right:
        G.add_edge(node.origin_index, node.right.origin_index)
        r_x, r_y = x + 1 / 2 ** layer, y - 1
        r_layer = layer + 1
        createGraph(G, node.right, x=r_x, y=r_y, pos=pos, layer=r_layer)
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
        self.children_count = 0
        self.left = None
        self.right = None
        self.not_visited = True
