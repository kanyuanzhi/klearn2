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
    
    def train(self, *args):
        pass

    def draw(self, *args):
        pass
