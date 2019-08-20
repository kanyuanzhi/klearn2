class Klearn(object):
    def __init__(self, x, y, *args, **kwargs):
        self.x = x
        self.y = y
        self.verbose = False
        if "mode" in kwargs:
            if kwargs["mode"] == "verbose":
                self.verbose = True

    def printX(self):
        print(self.x)

    def fit(self, *args):
        # 训练
        pass

    def predict(self, *args):
        # 预测
        pass

    def draw(self, *args):
        # 绘图
        pass
