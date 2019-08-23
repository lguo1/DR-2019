import numpy as np
class Chicken:
    def __init__(self):
        self.u = np.array([[[0,2],[7,6]],[[0,2],[7,6]]])
        self.shape = self.u.shape[1:]

class Prison:
    def __init__(self):
        self.u = np.array([[[-1,0],[-3,-2]],[[-1,0],[-3,-2]]])
        self.shape = self.u.shape[1:]

class Sexes:
    def __init__(self):
        self.u = np.array([[[4,0],[0,1]],[[1,0],[0,4]]])
        self.shape = self.u.shape[1:]

class RPS:
    def __init__(self):
        self.u = np.array([[[0,1,-1],[-1,0,1],[1,-1,0]],[[0,1,-1],[-1,0,1],[1,-1,0]]])
        self.shape = self.u.shape[1:]

class RPS2:
    def __init__(self):
        self.u = np.array([[[0,5,-5],[-5,0,5],[5,-5,0]],[[0,1,-1],[-1,0,1],[1,-1,0]]])
        self.shape = self.u.shape[1:]

class RPS3:
    def __init__(self):
        self.u = np.array([[[0,100,50],[50,0,100],[100,50,0]],[[0,1,-1],[-1,0,1],[1,-1,0]]])
        self.shape = self.u.shape[1:]
