import numpy as np
class Chicken:
    # Game of Chicken
    def __init__(self):
        self.u = np.array([[[0,2],[7,6]],[[0,2],[7,6]]])
        self.shape = self.u.shape[1:]

class Prison:
    # Prisoner Dilemma
    def __init__(self):
        self.u = np.array([[[-1,0],[-3,-2]],[[-1,0],[-3,-2]]])
        self.shape = self.u.shape[1:]

class Sexes:
    # Battle of Sexes
    def __init__(self):
        self.u = np.array([[[4,0],[0,1]],[[1,0],[0,4]]])
        self.shape = self.u.shape[1:]

class RPS:
    # Standard RPS
    def __init__(self):
        self.u = np.array([[[0,1,-1],[-1,0,1],[1,-1,0]],[[0,1,-1],[-1,0,1],[1,-1,0]]])
        self.shape = self.u.shape[1:]

class RPS2:
    # general sum RPS, with sum = 4
    def __init__(self):
        self.u = np.array([[[2,5,-5],[-5,2,5],[5,-5,2]],[[2,1,-1],[-1,2,1],[1,-1,2]]])
        self.shape = self.u.shape[1:]

class RPS3:
    # non-general sum RPS but with the same mixed Nash equilibrium
    def __init__(self):
        self.u = np.array([[[0,100,50],[50,0,100],[100,50,0]],[[0,1,-1],[-1,0,1],[1,-1,0]]])
        self.shape = self.u.shape[1:]
